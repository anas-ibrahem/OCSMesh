# Hybrid MPI + Pool Benchmark Guide

## What Changed

### `spawn` → `forkserver` (in `ocsmesh/mpi.py`)

Previously, under MPI, `multiprocessing` was set to `spawn` — which re-launches
a **full Python interpreter** for every Pool worker. This is safe but extremely slow.

Now it uses `forkserver` — which pre-forks a clean server process **once**, then
forks workers from that server. Same MPI safety, much faster Pool startup.

### Why Hybrid MPI + Pool?

| Strategy | How it works | Problem |
|---|---|---|
| `serial_mp` | 1 tile at a time, Pool(N) for add_feature | Tiles are sequential |
| `mpi_no_pool` | 1 tile per rank, nprocs=1 | Cores inside rank are unused |
| **`mpi_hybrid`** | **1 tile per rank, Pool(cores_per_rank)** | **None — 100% utilization** |

MPI ranks are **NOT** daemon processes, so they CAN create child Pools.
This is what makes hybrid possible.

## Formula

```
ntasks       = n_tiles + 1        (rank 0 = coordinator)
cpus_per_task = total_cores / ntasks
```

Example: 15 DEMs on an 80-core Hercules node:
```
ntasks       = 16
cpus_per_task = 5   (80 / 16 = 5)
```

Cores are auto-detected via `os.sched_getaffinity(0)` — no code config needed.

## Configs

| Config | Refinements | What it tests |
|---|---|---|
| **A** | Contours only | Pure contour parallelization |
| **B** | Contours + channels | Channel calls add_patch → add_feature |
| **C** | B + flow limiter + const_val | 3-phase file-path workers |
| **D** | C + patch + topo-bound constraint | Full pipeline incl. serial constraints |

## Running on Hercules

### Prerequisites

```bash
module load python  # or your conda env
# Ensure ocsmesh is installed: pip install -e .
```

### Step 1: Serial Baseline (no MPI needed)

Run each config to get the serial_mp timing:

```bash
python tests/benchmarks/benchmark_hybrid_A.py --mode serial_mp --tiles 15 --nprocs 80 --size 120
python tests/benchmarks/benchmark_hybrid_B.py --mode serial_mp --tiles 15 --nprocs 80 --size 120
python tests/benchmarks/benchmark_hybrid_C.py --mode serial_mp --tiles 15 --nprocs 80 --size 120
python tests/benchmarks/benchmark_hybrid_D.py --mode serial_mp --tiles 15 --nprocs 80 --size 120
```

### Step 2: Pure MPI (no internal Pool)

16 ranks = 15 tiles + 1 coordinator, 1 core each:

```bash
srun --ntasks=16 --cpus-per-task=1 \
    python tests/benchmarks/benchmark_hybrid_A.py --mode mpi_no_pool --tiles 15
srun --ntasks=16 --cpus-per-task=1 \
    python tests/benchmarks/benchmark_hybrid_B.py --mode mpi_no_pool --tiles 15
srun --ntasks=16 --cpus-per-task=1 \
    python tests/benchmarks/benchmark_hybrid_C.py --mode mpi_no_pool --tiles 15
srun --ntasks=16 --cpus-per-task=1 \
    python tests/benchmarks/benchmark_hybrid_D.py --mode mpi_no_pool --tiles 15
```

### Step 3: Hybrid MPI + Pool (the new approach)

16 ranks, 5 cores each = 80 cores total:

```bash
srun --ntasks=16 --cpus-per-task=5 \
    python tests/benchmarks/benchmark_hybrid_A.py --mode mpi_hybrid --tiles 15
srun --ntasks=16 --cpus-per-task=5 \
    python tests/benchmarks/benchmark_hybrid_B.py --mode mpi_hybrid --tiles 15
srun --ntasks=16 --cpus-per-task=5 \
    python tests/benchmarks/benchmark_hybrid_C.py --mode mpi_hybrid --tiles 15
srun --ntasks=16 --cpus-per-task=5 \
    python tests/benchmarks/benchmark_hybrid_D.py --mode mpi_hybrid --tiles 15
```

### SLURM Batch Script (all at once)

```bash
#!/bin/bash
#SBATCH --job-name=ocsmesh-hybrid-bench
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --output=hybrid_bench_%j.log

TILES=15
SIZE=120

echo "========================================="
echo "Hercules node: $(hostname)"
echo "Total cores: $(nproc)"
echo "SLURM ntasks: $SLURM_NTASKS"
echo "SLURM cpus-per-task: $SLURM_CPUS_PER_TASK"
echo "========================================="

# Serial baseline (runs on 1 rank only, uses all cores via Pool)
echo ""
echo ">>> SERIAL BASELINE (nprocs=80)"
for cfg in A B C D; do
    python tests/benchmarks/benchmark_hybrid_${cfg}.py \
        --mode serial_mp --tiles $TILES --nprocs 80 --size $SIZE \
        --json results_serial_${cfg}.json
done

# Hybrid MPI + Pool
echo ""
echo ">>> HYBRID MPI + POOL"
for cfg in A B C D; do
    srun python tests/benchmarks/benchmark_hybrid_${cfg}.py \
        --mode mpi_hybrid --tiles $TILES --size $SIZE \
        --json results_hybrid_${cfg}.json
done

echo ""
echo "Done. Check results_*.json for detailed timings."
```

### Save and submit:

```bash
sbatch tests/benchmarks/run_hybrid_bench.sh
```

## Expected Results

| Config | serial_mp (80 cores) | mpi_hybrid (16×5) | Expected speedup |
|---|---|---|---|
| A (contours) | baseline | fastest | 5-15× |
| B (+ channels) | baseline | fastest | 5-15× |
| C (+ flow/const) | baseline | fastest | 5-15× |
| D (+ constraints) | baseline | limited | 3-10× (constraints are serial) |

## Troubleshooting

### "Fork support not available" warning
The `forkserver` start method should prevent this. If you still see it:
```bash
export OMPI_MCA_mpi_warn_on_fork=0
```

### Pool workers stuck / slow
Check that `--cpus-per-task` is set correctly in SLURM:
```bash
python -c "import os; print(len(os.sched_getaffinity(0)))"
# Should print the same number as --cpus-per-task
```

### Hercules CPU count
Hercules nodes have SMT (hyperthreading). Each "CPU" in SLURM = 1 hardware thread.
For CPU-bound work, use physical cores only:
```bash
#SBATCH --hint=nomultithread
# OR
#SBATCH --threads-per-core=1
```

### Values differ between modes
Any difference is a **bug**, not rounding. `add_feature` merges with `np.minimum`
which is order-independent. File an issue with the JSON output.


__

# Hybrid MPI + Pool, Config B (the one Soroosh tested):
srun --ntasks=16 --cpus-per-task=5 \
    python tests/benchmarks/benchmark_hybrid_B.py --mode mpi_hybrid --tiles 15
--