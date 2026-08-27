import sys
import time
import math
import concurrent.futures
from typing import List, Dict, Any, Callable, Optional

try:
    from hpc_dynamic_task_runner import HPCDynamicTaskRunner, TaskSpec
    HPC_RUNNER_AVAILABLE = True
except ImportError:
    HPC_RUNNER_AVAILABLE = False
    class TaskSpec:
        def __init__(self, action: Any, payload: Any = None):
            self.action = action
            self.payload = payload


class JobProcessor:
    """
    A unified processing facade where all worker processing functions
    are purely internal instance methods of JobProcessor.
    
    No external functions are required or called.
    """
    def __init__(self, mode: str = "serial"):
        """
        Parameters:
        - mode: Execution backend ('serial', 'multiprocessing', or 'mpi').
        """
        self.mode = mode.lower()
        self._task_queue: List[TaskSpec] = []
        
        # Internal worker registry mapping action keys directly to internal instance methods
        self.worker_registry: Dict[str, Callable] = {
            'factorize': self._worker_factorize,
            'simulate': self._worker_simulate
        }

    def _worker_factorize(self, payload: dict) -> dict:
        """Internal Method 1: Factorize a number."""
        n = payload.get("number", 1000007)
        temp = n
        factors = []
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors.append(d)
                temp //= d
            d += 1
        if temp > 1:
            factors.append(temp)
        time.sleep(payload.get("delay", 0.1))
        return {"number": n, "factors": factors}

    def _worker_simulate(self, payload: dict) -> dict:
        """Internal Method 2: Data simulation."""
        samples = payload.get("samples", 1000)
        time.sleep(payload.get("delay", 0.1))
        return {"samples": samples, "value": math.sin(samples)}

    def do_factorize(self, number: int, delay: float = 0.1):
        """User method A: Queues a prime factorization task internally."""
        payload = {'number': number, 'delay': delay}
        self._task_queue.append(TaskSpec('factorize', payload))

    def do_simulate(self, samples: int, delay: float = 0.1):
        """User method B: Queues a data simulation task internally."""
        payload = {'samples': samples, 'delay': delay}
        self._task_queue.append(TaskSpec('simulate', payload))

    def get_output(self) -> Optional[List[Any]]:
        """
        Main user entry point to compute the accumulated workload.
        Dispatches all queued tasks using configured backend mode.
        Returns result list on primary process, and None on MPI workers.
        """
        if not self._task_queue:
            print("[JobProcessor] Warning: No tasks were queued before get_output() was called.")
            return []

        tasks = list(self._task_queue)
        self._task_queue.clear()  # Reset queue for subsequent runs
        return self._process(tasks)

    def _execute_single_task(self, task: TaskSpec) -> Any:
        """Helper to resolve and run a single task in non-MPI modes."""
        action = task.action
        payload = task.payload
        if action in self.worker_registry:
            return self.worker_registry[action](payload)
        elif callable(action):
            return action(payload) if payload is not None else action()
        else:
            raise ValueError(f"Unable to resolve task action '{action}'")

    def _process(self, tasks: List[TaskSpec]) -> Optional[List[Any]]:
        """Internal dispatch router based on execution mode."""
        if self.mode == "serial":
            return self._process_serial(tasks)
        elif self.mode == "multiprocessing":
            return self._process_multiprocessing(tasks)
        elif self.mode == "mpi":
            return self._process_mpi(tasks)
        else:
            raise ValueError(f"Unsupported execution mode: '{self.mode}'")

    def _process_serial(self, tasks: List[TaskSpec]) -> List[Any]:
        """Serial execution mode (single thread/process)."""
        print("[JobProcessor] Executing in SERIAL mode...")
        results = []
        for idx, task in enumerate(tasks):
            try:
                res = self._execute_single_task(task)
                results.append((idx, True, res))
            except Exception as err:
                results.append((idx, False, f"Error: {str(err)}"))
        return results

    def _process_multiprocessing(self, tasks: List[TaskSpec]) -> List[Any]:
        """Local shared-memory multiprocessing mode using ProcessPoolExecutor."""
        print("[JobProcessor] Executing in MULTIPROCESSING mode...")
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(self._execute_single_task, task): idx for idx, task in enumerate(tasks)}
            completed_map = {}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    res = future.result()
                    completed_map[idx] = (idx, True, res)
                except Exception as err:
                    completed_map[idx] = (idx, False, f"Error: {str(err)}")
            results = [completed_map[i] for i in range(len(tasks))]
        return results

    def _process_mpi(self, tasks: List[TaskSpec]) -> Optional[List[Any]]:
        """
        Distributed MPI execution mode.
        Passes internal bound methods in self.worker_registry to HPCDynamicTaskRunner.
        """
        if not HPC_RUNNER_AVAILABLE:
            raise RuntimeError("HPC Dynamic Task Runner is not available or mpi4py is not installed.")

        runner = HPCDynamicTaskRunner(catch_worker_exceptions=True)
        return runner.run(task_payloads=tasks, worker_fn=self.worker_registry)


if __name__ == "__main__":
    # 1. Instantiate orchestrator specifying backend ('serial', 'multiprocessing', or 'mpi')
    execution_mode = "mpi"
    pipeline = JobProcessor(mode=execution_mode)

    # 2. Queue workload by calling method actions directly on the pipeline object
    pipeline.do_factorize(number=999999999989, delay=0.2)
    pipeline.do_simulate(samples=5000, delay=0.1)
    pipeline.do_factorize(number=123456789012, delay=0.15)
    pipeline.do_simulate(samples=20000, delay=0.05)

    # 3. Trigger execution and retrieve final output report
    start_time = time.time()
    results = pipeline.get_output()

    # 4. Print clean execution report on Rank 0 / primary process
    if results is not None:
        elapsed = time.time() - start_time
        print(f"\n========================================================")
        print(f"   Pipeline completed in {elapsed:.2f}s using mode: '{execution_mode}'")
        print(f"========================================================")
        for task_id, success, output in results:
            status = "SUCCESS" if success else "FAILED "
            print(f"  [Task {task_id:02d}] [{status}] -> {output}")
        print("========================================================\n")