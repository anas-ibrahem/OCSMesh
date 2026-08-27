import sys
import time
import random
import traceback
from typing import List, Callable, Any, Tuple

# Attempt to import mpi4py safely for HPC environments
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


class TaskItem:
    """
    Container holding task data and metadata.
    Passed dynamically from Manager to Workers.
    """
    def __init__(self, task_id: int, payload: Any):
        self.task_id = task_id
        self.payload = payload


class TaskSpec:
    """
    User-facing task container for multi-task workloads.
    Specifies an action name (or callable function) and input payload.
    """
    def __init__(self, action: Any, payload: Any = None):
        self.action = action
        self.payload = payload


class HPCDynamicTaskRunner:
    """
    A rank-check-free wrapper for HPC clusters.
    
    Features:
    1. Dynamic Manager-Worker queue for unbalanced workloads (max core utilization).
    2. Works on all HPC MPI implementations (Slurm, PBS, OpenMPI, MPICH, Intel MPI).
    3. Supports heterogeneous multi-task workloads defined in user code.
    4. Zero rank checks required in user code.
    5. Safe exception handling prevents hanging/zombie cluster instances.
    6. Fallback for single-process non-MPI execution.
    """
    
    # Message Tags for Manager-Worker Protocol
    TAG_REQUEST_TASK = 10  # Worker -> Manager: "I am ready for a task"
    TAG_TASK_PAYLOAD = 11  # Manager -> Worker: "Here is your task"
    TAG_RESULT       = 12  # Worker -> Manager: "Here is my completed result"
    TAG_TERMINATE    = 99  # Manager -> Worker: "No more tasks, shut down loop"

    def __init__(self, catch_worker_exceptions: bool = True):
        """
        Parameters:
        - catch_worker_exceptions: If True, captures worker errors and returns
          failure reports without crashing or hanging the HPC job.
        """
        self.catch_worker_exceptions = catch_worker_exceptions
        
        if MPI_AVAILABLE:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

    def run(self, task_payloads: List[Any], worker_fn: Any = None) -> List[Tuple[int, bool, Any]]:
        """
        Public entry point called identically by all processes.
        
        Parameters:
        - task_payloads: List of inputs or TaskSpec items to process.
        - worker_fn: Can be a single Callable, a Dict[str, Callable] method registry,
                     or None if TaskSpec objects carry callable functions directly.
        
        Returns:
        - List of (task_id, success_flag, result_or_error) on Rank 0.
        - None on Worker ranks.
        """
        if self.size == 1:
            return self._run_sequential(task_payloads, worker_fn)
        
        # In multi-process MPI, encapsulate manager vs worker roles internally
        if self.rank == 0:
            return self._run_manager(task_payloads)
        else:
            self._run_worker(worker_fn)
            return None

    def _execute_task(self, item_payload: Any, worker_fn: Any) -> Any:
        """
        Dynamic task execution resolver supporting multi-task patterns.
        """
        # Case 1: worker_fn is a function registry dictionary mapping action names -> functions
        if isinstance(worker_fn, dict):
            action = getattr(item_payload, 'action', None)
            data = getattr(item_payload, 'payload', item_payload)
            if action not in worker_fn:
                raise KeyError(f"Action '{action}' is not registered in worker_fn registry.")
            return worker_fn[action](data)

        # Case 2: item_payload is a TaskSpec where action is directly callable
        if hasattr(item_payload, 'action') and callable(item_payload.action):
            data = getattr(item_payload, 'payload', None)
            return item_payload.action(data) if data is not None else item_payload.action()

        # Case 3: worker_fn is a single callable function
        if callable(worker_fn):
            return worker_fn(item_payload)

        raise ValueError("Unable to determine how to execute task. Check TaskSpec or worker_fn definition.")

    def _run_manager(self, raw_payloads: List[Any]) -> List[Tuple[int, bool, Any]]:
        """
        Manager process (Rank 0): Dynamically assigns tasks to idle workers.
        """
        # Package tasks into TaskItems with tracking IDs
        task_queue = [TaskItem(task_id=idx, payload=item) for idx, item in enumerate(raw_payloads)]
        total_tasks = len(task_queue)
        
        results_map = {}
        active_workers = self.size - 1
        
        while active_workers > 0:
            status = MPI.Status()
            # Wait for a request or result from ANY worker process
            msg = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker_rank = status.Get_source()
            tag = status.Get_tag()
            
            # Record incoming completed result if available
            if tag == self.TAG_RESULT:
                task_id, success, result_data = msg
                results_map[task_id] = (task_id, success, result_data)

            # Send next available task OR termination signal
            if task_queue:
                next_task = task_queue.pop(0)
                self.comm.send(next_task, dest=worker_rank, tag=self.TAG_TASK_PAYLOAD)
            else:
                # No tasks left: Send poison pill to tell worker to exit loop
                self.comm.send(None, dest=worker_rank, tag=self.TAG_TERMINATE)
                active_workers -= 1

        # Sort results back into original input order
        ordered_results = [results_map[idx] for idx in range(total_tasks)]
        return ordered_results

    def _run_worker(self, worker_fn: Any):
        """
        Worker process (Ranks 1..N-1): Requests tasks, executes worker_fn, returns results.
        """
        # Signal manager that this worker is ready for work
        self.comm.send(None, dest=0, tag=self.TAG_REQUEST_TASK)
        
        while True:
            status = MPI.Status()
            # Wait for incoming task assignment from Manager
            task_item = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            
            # Check for termination signal ("poison pill")
            if tag == self.TAG_TERMINATE or task_item is None:
                break
            
            # Execute computation safely using multi-task execution resolver
            try:
                result_value = self._execute_task(task_item.payload, worker_fn)
                success = True
            except Exception as err:
                if not self.catch_worker_exceptions:
                    # Instantly abort entire MPI cluster to prevent zombie instances
                    sys.stderr.write(f"\n[CRITICAL ERROR] Rank {self.rank} crashed on task {task_item.task_id}: {err}\n")
                    sys.stderr.flush()
                    self.comm.Abort(1)
                
                err_msg = f"Rank {self.rank} Error ({type(err).__name__}): {str(err)}"
                result_value = err_msg
                success = False

            # Send completed result back and request next task
            self.comm.send((task_item.task_id, success, result_value), dest=0, tag=self.TAG_RESULT)

    def _run_sequential(self, raw_payloads: List[Any], worker_fn: Any) -> List[Tuple[int, bool, Any]]:
        """Fallback for running without mpiexec."""
        print("[Info] Running in Single-Process (Non-MPI) Mode...")
        results = []
        for idx, payload in enumerate(raw_payloads):
            try:
                res = self._execute_task(payload, worker_fn)
                results.append((idx, True, res))
            except Exception as err:
                results.append((idx, False, f"Error ({type(err).__name__}): {str(err)}"))
        return results


# ==========================================
# 1. USER COMPUTATION FUNCTIONS (Multi-Task)
# ==========================================
def task_data_simulation(payload: dict) -> str:
    """Task Type 1: Monte Carlo / Data Simulation."""
    time.sleep(payload.get("duration", 0.1))
    return f"Simulated {payload['samples']} samples in {payload['duration']:.2f}s"

def task_matrix_factorization(payload: dict) -> str:
    """Task Type 2: Heavy Linear Algebra Matrix Operation."""
    time.sleep(payload.get("duration", 0.2))
    return f"Factorized {payload['matrix_size']}x{payload['matrix_size']} matrix in {payload['duration']:.2f}s"

def task_graph_analytics(payload: dict) -> str:
    """Task Type 3: Graph Traversal / Analysis (with simulated crash)."""
    if payload.get("trigger_error", False):
        raise RuntimeError("Graph cycle error: numerical instability detected")
    time.sleep(payload.get("duration", 0.1))
    return f"Analyzed graph with {payload['nodes']} nodes in {payload['duration']:.2f}s"


# ==========================================
# 2. USER PIPELINE (Zero Rank Checks)
# ==========================================
if __name__ == "__main__":
    # Define a registry mapping action names to user functions
    task_registry = {
        'simulate': task_data_simulation,
        'factorize': task_matrix_factorization,
        'analyze_graph': task_graph_analytics
    }

    # Define a heterogeneous list of tasks in user code
    workload = [
        TaskSpec('simulate', {'samples': 10000, 'duration': 0.1}),
        TaskSpec('factorize', {'matrix_size': 2048, 'duration': 0.5}),
        TaskSpec('analyze_graph', {'nodes': 500, 'duration': 0.2}),
        TaskSpec('simulate', {'samples': 50000, 'duration': 0.8}),
        TaskSpec('analyze_graph', {'nodes': 100, 'trigger_error': True}),  # Faulty task
        TaskSpec('factorize', {'matrix_size': 1024, 'duration': 0.3}),
        TaskSpec('simulate', {'samples': 25000, 'duration': 0.15}),
    ]

    # Instantiate runner on ALL ranks (Zero 'if rank == 0' checks needed!)
    runner = HPCDynamicTaskRunner(catch_worker_exceptions=True)

    start_clock = time.time()
    
    # Run the heterogeneous multi-task workload dynamically
    results = runner.run(task_payloads=workload, worker_fn=task_registry)

    # Output execution summary on Rank 0
    if results is not None:
        total_time = time.time() - start_clock
        print(f"\n==================================================")
        print(f"HPC Multi-Task Workload Completed in {total_time:.2f} seconds")
        print(f"==================================================")
        
        for task_id, success, output in results:
            status_str = "SUCCESS" if success else "FAILED "
            print(f"  [Task {task_id:02d}] [{status_str}] -> {output}")