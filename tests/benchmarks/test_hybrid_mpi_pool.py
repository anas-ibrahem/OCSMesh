"""Proof-of-concept: Hybrid MPI + Pool with auto core detection.

Simulates our tile-processing pattern with CPU-bound work.
Compares three modes:
  1. serial       — 1 process, no Pool, all tiles sequentially
  2. mpi_no_pool  — N ranks, nprocs=1, 1 tile per rank
  3. mpi_hybrid   — N ranks, each with Pool(available_cores), 1 tile per rank

Usage (local, no MPI):
  python test_hybrid_mpi_pool.py --mode serial --n-tiles 8

Usage (MPI, pure — 1 core per rank, rank 0 = coordinator):
  srun --ntasks=9 --cpus-per-task=1 python test_hybrid_mpi_pool.py --mode mpi_no_pool --n-tiles 8

Usage (MPI, hybrid — 4 cores per rank, rank 0 = coordinator):
  srun --ntasks=9 --cpus-per-task=4 python test_hybrid_mpi_pool.py --mode mpi_hybrid --n-tiles 8

Usage (MPI, hybrid, auto-detect cores):
  srun --ntasks=9 --cpus-per-task=4 python test_hybrid_mpi_pool.py --mode mpi_hybrid --n-tiles 8 --auto-cores

Note: --ntasks = n_tiles + 1 (rank 0 is the coordinator, does not process tiles).

The --auto-cores flag uses os.sched_getaffinity(0) instead of
a manual --cores-per-rank value.
"""

import argparse
import os
import time
from multiprocessing import Pool


# ---------------------------------------------------------------------------
# CPU-bound work: simulates add_feature's heavy math (KDTree + distance)
# ---------------------------------------------------------------------------

def _heavy_work(chunk_id, size=2_000_000):
    """Burn CPU for a measurable amount of time.

    This simulates the per-window KDTree query inside add_feature.
    Each call takes ~0.3-0.5s on a modern core.
    """
    total = 0.0
    for i in range(size):
        total += (i * 0.999999) ** 0.5
    return chunk_id, total


def process_tile(tile_id, n_chunks, nprocs):
    """Process one tile: split into n_chunks of CPU work.

    With nprocs=1: sequential (no Pool).
    With nprocs>1: Pool(nprocs) to parallelize chunks.

    This mirrors how add_feature works:
    - nprocs=1 → pool=None → sequential starmap
    - nprocs>1 → Pool(nprocs) → parallel starmap
    """
    args = [(tile_id * 100 + c, 500_000) for c in range(n_chunks)]

    if nprocs <= 1:
        # Sequential — same as pool=None in add_feature
        results = [_heavy_work(*a) for a in args]
    else:
        # Parallel — same as Pool(nprocs).starmap in add_feature
        # This ONLY works if we are NOT a daemon process.
        # MPI ranks are full processes → this is safe.
        # Pool workers are daemons → this would crash.
        with Pool(processes=nprocs) as p:
            results = p.starmap(_heavy_work, args)

    return tile_id, len(results)


def available_cores():
    """Auto-detect cores available to this process via affinity mask.

    When SLURM sets --cpus-per-task=4, this returns 4.
    When no binding is set, returns all cores on the machine.
    Falls back to os.cpu_count() on macOS (no sched_getaffinity).
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Mode 1: Serial (baseline)
# ---------------------------------------------------------------------------

def run_serial(n_tiles, n_chunks):
    """Process all tiles sequentially, no parallelism at all."""
    print(f"[serial] Processing {n_tiles} tiles with {n_chunks} chunks each")
    print(f"[serial] Available cores: {available_cores()} (unused)")

    start = time.time()
    for tile_id in range(n_tiles):
        process_tile(tile_id, n_chunks, nprocs=1)
    elapsed = time.time() - start

    print(f"[serial] Done in {elapsed:.2f}s")
    return elapsed


# ---------------------------------------------------------------------------
# Mode 2: MPI, no internal Pool (nprocs=1 per rank)
# ---------------------------------------------------------------------------

def run_mpi_no_pool(n_tiles, n_chunks):
    """Distribute tiles across MPI ranks. Each rank: nprocs=1.

    Rank 0 is the coordinator — it distributes work but does not
    process tiles. Ranks 1..N are the workers.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_workers = size - 1  # rank 0 = coordinator

    if rank == 0:
        print(f"[mpi_no_pool] {n_workers} workers (rank 0 = coordinator), "
              f"{n_tiles} tiles, {n_chunks} chunks/tile")
        print(f"[mpi_no_pool] Available cores per rank: {available_cores()} "
              f"(each worker uses 1)")
        if n_tiles > n_workers:
            print(f"[mpi_no_pool] WARNING: {n_tiles} tiles > {n_workers} "
                  f"workers — some workers will handle multiple tiles")

    comm.Barrier()
    start = time.time()

    if rank == 0:
        # Coordinator: distribute tiles round-robin to workers
        for tile_id in range(n_tiles):
            dest = (tile_id % n_workers) + 1  # ranks 1..N
            comm.send(tile_id, dest=dest, tag=1)
        # Send stop signal to all workers
        for w in range(1, size):
            comm.send(-1, dest=w, tag=1)
    else:
        # Worker: receive and process tiles until stop signal
        while True:
            tile_id = comm.recv(source=0, tag=1)
            if tile_id == -1:
                break
            process_tile(tile_id, n_chunks, nprocs=1)

    comm.Barrier()
    elapsed = time.time() - start

    if rank == 0:
        print(f"[mpi_no_pool] Done in {elapsed:.2f}s")
    return elapsed


# ---------------------------------------------------------------------------
# Mode 3: MPI + internal Pool (hybrid)
# ---------------------------------------------------------------------------

def run_mpi_hybrid(n_tiles, n_chunks, cores_per_rank=None):
    """Distribute tiles across MPI ranks. Each rank: Pool(cores_per_rank).

    Rank 0 is the coordinator — it distributes work but does not
    process tiles. Ranks 1..N are the workers.

    If cores_per_rank is None, auto-detect from affinity mask.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_workers = size - 1  # rank 0 = coordinator

    if cores_per_rank is None:
        cores_per_rank = available_cores()

    if rank == 0:
        print(f"[mpi_hybrid] {n_workers} workers (rank 0 = coordinator), "
              f"{n_tiles} tiles, {n_chunks} chunks/tile")
        print(f"[mpi_hybrid] Each worker uses Pool({cores_per_rank})")
        print(f"[mpi_hybrid] Total cores utilized: "
              f"{n_workers} × {cores_per_rank} = {n_workers * cores_per_rank}")
        if n_tiles > n_workers:
            print(f"[mpi_hybrid] WARNING: {n_tiles} tiles > {n_workers} "
                  f"workers — some workers will handle multiple tiles")

    comm.Barrier()
    start = time.time()

    if rank == 0:
        # Coordinator: distribute tiles round-robin to workers
        for tile_id in range(n_tiles):
            dest = (tile_id % n_workers) + 1  # ranks 1..N
            comm.send(tile_id, dest=dest, tag=1)
        # Send stop signal to all workers
        for w in range(1, size):
            comm.send(-1, dest=w, tag=1)
    else:
        # Worker: receive and process tiles until stop signal
        # KEY: nprocs > 1, so Pool is created inside.
        # This works because MPI ranks are NOT daemon processes.
        while True:
            tile_id = comm.recv(source=0, tag=1)
            if tile_id == -1:
                break
            process_tile(tile_id, n_chunks, nprocs=cores_per_rank)

    comm.Barrier()
    elapsed = time.time() - start

    if rank == 0:
        print(f"[mpi_hybrid] Done in {elapsed:.2f}s")
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Proof-of-concept: hybrid MPI + Pool")
    parser.add_argument(
        '--mode', choices=['serial', 'mpi_no_pool', 'mpi_hybrid'],
        default='serial',
        help='Execution mode')
    parser.add_argument(
        '--n-tiles', type=int, default=8,
        help='Number of tiles to process (simulates DEMs)')
    parser.add_argument(
        '--n-chunks', type=int, default=8,
        help='Chunks of work per tile (simulates contour lines)')
    parser.add_argument(
        '--cores-per-rank', type=int, default=None,
        help='Cores for internal Pool per rank (hybrid mode only). '
             'Default: auto-detect from affinity.')
    parser.add_argument(
        '--auto-cores', action='store_true',
        help='Force auto-detection via os.sched_getaffinity')

    args = parser.parse_args()

    if args.auto_cores:
        args.cores_per_rank = None  # force auto-detect

    if args.mode == 'serial':
        run_serial(args.n_tiles, args.n_chunks)
    elif args.mode == 'mpi_no_pool':
        run_mpi_no_pool(args.n_tiles, args.n_chunks)
    elif args.mode == 'mpi_hybrid':
        run_mpi_hybrid(
            args.n_tiles, args.n_chunks,
            cores_per_rank=args.cores_per_rank)


if __name__ == '__main__':
    main()
