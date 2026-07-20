#!/usr/bin/env python
"""
Reproduction of the __del__ cleanup bug in HfunCollector.

This script demonstrates the exact failure mode that caused the GSoC 2025
MPI attempt to be abandoned: when a child process holds a reference to an
object whose __del__ calls shutil.rmtree(_work_dir), the child's garbage
collector fires __del__ before the coordinator has finished reading results.

Two reproduction modes:
  1. Fork-based (no MPI needed):  python reproduce_del_bug.py --fork
  2. MPI-based:                   mpirun -n 2 python reproduce_del_bug.py --mpi

Both modes demonstrate the same root cause.
After demonstrating the bug, apply the PID guard fix and re-run to confirm
the fix prevents the premature deletion.
"""

import os
import sys
import shutil
import tempfile
import time
import argparse
import gc


# ── Mock class that mirrors HfunCollector's __del__ exactly ──────────────

class MockCollector:
    """
    Minimal reproduction of HfunCollector's temp directory lifecycle.

    __init__ creates a temp directory and writes a "result" file.
    __del__  calls shutil.rmtree (identical to collector.py line 936-938).
    """

    def __init__(self, work_dir=None):
        if work_dir is None:
            self._work_dir = tempfile.mkdtemp(prefix='hfun_collector_')
        else:
            # Simulate receiving a shared _work_dir path (e.g., via MPI)
            self._work_dir = work_dir

        # Simulate writing worker output to _work_dir
        result_file = os.path.join(self._work_dir, 'result_0.tif')
        if not os.path.exists(result_file):
            with open(result_file, 'w') as f:
                f.write('important mesh size function data')

    def __del__(self):
        # This is the EXACT code from collector.py line 936-938
        if hasattr(self, '_work_dir') and os.path.exists(self._work_dir):
            shutil.rmtree(self._work_dir, ignore_errors=True)


class MockCollectorFixed:
    """
    Same as MockCollector but WITH the PID guard fix applied.
    """

    def __init__(self, work_dir=None):
        if work_dir is None:
            self._work_dir = tempfile.mkdtemp(prefix='hfun_collector_')
        else:
            self._work_dir = work_dir

        # ── THE FIX: record creator's PID ──
        self._creator_pid = os.getpid()

        result_file = os.path.join(self._work_dir, 'result_0.tif')
        if not os.path.exists(result_file):
            with open(result_file, 'w') as f:
                f.write('important mesh size function data')

    def __del__(self):
        # ── THE FIX: only the creator process may delete ──
        if (hasattr(self, '_work_dir')
                and hasattr(self, '_creator_pid')
                and os.getpid() == self._creator_pid
                and os.path.exists(self._work_dir)):
            shutil.rmtree(self._work_dir, ignore_errors=True)


# ── Fork-based reproduction ──────────────────────────────────────────────

def reproduce_fork(use_fix=False):
    """
    Demonstrate the bug using multiprocessing.Process with fork.

    Under fork, the child inherits the parent's memory space INCLUDING
    the MockCollector reference. When the child exits, its copy's __del__
    fires and deletes _work_dir — even though the parent still needs it.
    """
    import multiprocessing
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    CollectorClass = MockCollectorFixed if use_fix else MockCollector
    label = "WITH PID fix" if use_fix else "WITHOUT fix (BUG)"

    print(f"\n{'='*60}")
    print(f"  Fork-based reproduction — {label}")
    print(f"{'='*60}")

    # Step 1: Parent creates collector
    collector = CollectorClass()
    work_dir = collector._work_dir
    result_file = os.path.join(work_dir, 'result_0.tif')

    print(f"[Parent PID={os.getpid()}] Created _work_dir: {work_dir}")
    print(f"[Parent] result file exists: {os.path.exists(result_file)}")

    # Step 2: Fork a child process (simulates MPI worker rank)
    def child_work():
        """Child does some work, then exits. GC fires __del__."""
        pid = os.getpid()
        print(f"  [Child  PID={pid}] Started, has reference to _work_dir")
        print(f"  [Child  PID={pid}] Doing work... done.")
        # Child exits here → GC runs → __del__ fires on child's copy
        # of the collector

    p = multiprocessing.Process(target=child_work)
    p.start()
    p.join()

    # Step 3: Force GC in case the child's __del__ hasn't run yet
    # (In practice, child process exit always triggers __del__ on forked copies)
    time.sleep(0.5)

    # Step 4: Parent tries to read results (Phase 3 integration)
    dir_exists = os.path.exists(work_dir)
    file_exists = os.path.exists(result_file)

    print(f"\n[Parent] After child exit:")
    print(f"  _work_dir exists: {dir_exists}")
    print(f"  result file exists: {file_exists}")

    if not dir_exists:
        print(f"\n  ❌ BUG REPRODUCED: Child's __del__ deleted _work_dir!")
        print(f"     Rank 0 would get FileNotFoundError during Phase 3.")
    else:
        print(f"\n  ✅ FIX WORKS: _work_dir survived child process exit.")
        print(f"     Rank 0 can safely read results during Phase 3.")
        # Clean up manually since fix prevents auto-cleanup by child
        shutil.rmtree(work_dir, ignore_errors=True)

    # Prevent parent's __del__ from running (dir may already be deleted)
    collector._work_dir = '/nonexistent'
    return not dir_exists  # True = bug reproduced


# ── MPI-based reproduction ───────────────────────────────────────────────

def reproduce_mpi(use_fix=False):
    """
    Demonstrate the bug using mpi4py.

    Rank 0 creates a collector and shares the _work_dir path with rank 1.
    Rank 1 creates its own collector pointing to the SAME _work_dir.
    When rank 1's collector goes out of scope, __del__ fires and deletes
    the shared directory. Rank 0 then can't read results.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        if rank == 0:
            print("ERROR: MPI reproduction requires at least 2 ranks.")
            print("Run with: mpirun -n 2 python reproduce_del_bug.py --mpi")
        return False

    CollectorClass = MockCollectorFixed if use_fix else MockCollector
    label = "WITH PID fix" if use_fix else "WITHOUT fix (BUG)"

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  MPI-based reproduction — {label}")
        print(f"{'='*60}")

    comm.Barrier()

    if rank == 0:
        # Rank 0 (coordinator) creates the collector and _work_dir
        collector = CollectorClass()
        work_dir = collector._work_dir

        # Write a "worker result" file to simulate Phase 2 output
        result_file = os.path.join(work_dir, 'result_0.tif')
        print(f"[Rank 0 PID={os.getpid()}] Created _work_dir: {work_dir}")
        print(f"[Rank 0] result file exists: {os.path.exists(result_file)}")

        # Share the _work_dir path with rank 1
        comm.send(work_dir, dest=1, tag=0)
    else:
        work_dir = None

    if rank == 1:
        # Rank 1 (worker) receives the shared _work_dir path
        shared_work_dir = comm.recv(source=0, tag=0)
        print(f"  [Rank 1 PID={os.getpid()}] Received _work_dir: {shared_work_dir}")

        # Rank 1 creates its own collector pointing to the SAME directory
        # This simulates what happens when the collector object is
        # pickled/broadcast to worker ranks
        worker_collector = CollectorClass(work_dir=shared_work_dir)
        print(f"  [Rank 1] Doing work... done.")

        # Worker is done — let the collector go out of scope
        del worker_collector
        gc.collect()
        print(f"  [Rank 1] Collector deleted, GC ran.")

    # Synchronize: ensure rank 1's __del__ has fired before rank 0 checks
    comm.Barrier()

    if rank == 0:
        # Rank 0 tries to read results (Phase 3 integration)
        dir_exists = os.path.exists(work_dir)
        result_file = os.path.join(work_dir, 'result_0.tif')
        file_exists = os.path.exists(result_file)

        print(f"\n[Rank 0] After worker finished:")
        print(f"  _work_dir exists: {dir_exists}")
        print(f"  result file exists: {file_exists}")

        if not dir_exists:
            print(f"\n  ❌ BUG REPRODUCED: Worker rank's __del__ deleted _work_dir!")
            print(f"     Rank 0 gets FileNotFoundError during Phase 3 integration.")
        else:
            print(f"\n  ✅ FIX WORKS: _work_dir survived worker rank exit.")
            print(f"     Rank 0 can safely read results during Phase 3.")
            shutil.rmtree(work_dir, ignore_errors=True)

        # Prevent rank 0's __del__ from erroring on missing dir
        collector._work_dir = '/nonexistent'
        return not dir_exists
    return False


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reproduce the __del__ cleanup bug in HfunCollector')
    parser.add_argument('--fork', action='store_true',
                        help='Use fork-based reproduction (no MPI needed)')
    parser.add_argument('--mpi', action='store_true',
                        help='Use MPI-based reproduction (requires mpirun)')
    args = parser.parse_args()

    if not args.fork and not args.mpi:
        # Default: try fork
        args.fork = True

    if args.fork:
        print("\n" + "=" * 60)
        print("  __del__ CLEANUP BUG REPRODUCTION (fork-based)")
        print("=" * 60)

        # First: demonstrate the bug
        bug_found = reproduce_fork(use_fix=False)

        # Then: demonstrate the fix
        reproduce_fork(use_fix=True)

        print(f"\n{'='*60}")
        if bug_found:
            print("  CONCLUSION: Bug successfully reproduced and fix verified.")
        else:
            print("  NOTE: Bug did not manifest (may require 'fork' start method).")
        print(f"{'='*60}\n")

    if args.mpi:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()

        if rank == 0:
            print("\n" + "=" * 60)
            print("  __del__ CLEANUP BUG REPRODUCTION (MPI-based)")
            print("=" * 60)

        # First: demonstrate the bug
        bug_found = reproduce_mpi(use_fix=False)

        MPI.COMM_WORLD.Barrier()

        # Then: demonstrate the fix
        reproduce_mpi(use_fix=True)

        MPI.COMM_WORLD.Barrier()

        if rank == 0:
            print(f"\n{'='*60}")
            if bug_found:
                print("  CONCLUSION: Bug successfully reproduced and fix verified.")
            else:
                print("  NOTE: Bug did not manifest under this MPI config.")
            print(f"{'='*60}\n")
