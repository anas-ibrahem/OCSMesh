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

### Collective entry (bug fix)

`MPIExecutor.run()` is a **collective** call — every rank must reach it.
The scripts therefore call `build_and_run()` on *all* ranks; guarding it
with `if rank == 0` deadlocks. Only rank 0 keeps/reports the results.

## Auto-sizing: one command, no manual core math

You do **not** pass `--tiles` or `--cpus-per-task` unless you want to
override something. `tests/benchmarks/hybrid_util.py` works it out:

| Quantity | How it is decided |
|---|---|
| tiles | `--tiles 0` (default) → `n_ranks - 1`, one tile per worker rank |
| cores/rank | first match of: `--cores-per-rank` → `SLURM_CPUS_PER_TASK` → affinity mask if the launcher already pinned the rank → `node_cores / worker_ranks_on_this_node` |

That last rule is the important one: under a plain `mpiexec -n 16` every
rank sees the *whole* node in `os.sched_getaffinity(0)`, so using it
directly would make all 15 workers spawn a full-size Pool and
oversubscribe the node ~15x. The number of ranks sharing the node is
found with `comm.Split_type(MPI.COMM_TYPE_SHARED)`; rank 0 is a
coordinator and is excluded from the divisor.

So on an 80-core node:

```
mpiexec -n 16 ...   →  15 tiles, 80 / 15 = 5 cores per worker rank
```

And under SLURM, `--cpus-per-task=5` is honoured as-is.

## Output

Each mode prints per-stage timing plus a resource line:

```
--- mpi_hybrid (5 cores/rank) ---
  contours: 3.90s  channels: 3.45s  total: 7.36s
  wall 7.44s | cpu 10.7s | busy 1.4/15 cores | utilization 10% (incl. MPI idle-spin)
```

* `wall` — barrier-to-barrier wall clock across all ranks.
* `cpu` — CPU seconds summed over **every rank** plus their reaped
  children (i.e. the Pool workers), via `os.times()` + `allreduce`.
* `busy` — `cpu / wall`, the average number of cores actually running.
* `utilization` — `busy / cores allocated to the run`.

> Caveat: idle Open MPI ranks **busy-wait** inside `recv`/`Barrier`, and
> that spin is charged as CPU time, inflating utilization. Damp it with
> `export OMPI_MCA_mpi_yield_when_idle=1`.

`--json out.json` writes the same numbers plus the core plan.

## Configs

| Config | Refinements | What it tests |
|---|---|---|
| **A** | Contours only | Pure contour parallelization |
| **B** | Contours + channels | Channel calls add_patch → add_feature |
| **C** | B + flow limiter + const_val | 3-phase file-path workers |
| **D** | C + patch + topo-bound constraint | Full pipeline incl. serial constraints |

## Quick local run

`-n 5` = 1 coordinator + 4 workers → 4 tiles, cores split automatically:

```bash
mpiexec -n 5 python tests/benchmarks/benchmark_hybrid_A.py
mpiexec -n 5 python tests/benchmarks/benchmark_hybrid_B.py
mpiexec -n 5 python tests/benchmarks/benchmark_hybrid_C.py
mpiexec -n 5 python tests/benchmarks/benchmark_hybrid_D.py
```

Each runs all three modes and verifies the outputs are bitwise identical.
Add `--oversubscribe` if Open MPI complains about slot count.

## Running on Hercules

### Prerequisites

```bash
module load python  # or your conda env
# Ensure ocsmesh is installed: pip install -e .
```

### Step 1: Serial Baseline (no MPI)

`--nprocs 0` (default) uses every core visible to the process:

```bash
for cfg in A B C D; do
    python tests/benchmarks/benchmark_hybrid_${cfg}.py \
        --mode serial_mp --tiles 15 --size 120 \
        --json results_serial_${cfg}.json
done
```

### Step 2: Pure MPI (no internal Pool)

16 ranks = 15 tiles + 1 coordinator, 1 core each:

```bash
for cfg in A B C D; do
    srun --ntasks=16 --cpus-per-task=1 \
        python tests/benchmarks/benchmark_hybrid_${cfg}.py --mode mpi_no_pool
done
```

### Step 3: Hybrid MPI + Pool (the new approach)

16 ranks, 5 cores each = 80 cores total:

```bash
for cfg in A B C D; do
    srun --ntasks=16 --cpus-per-task=5 \
        python tests/benchmarks/benchmark_hybrid_${cfg}.py --mode mpi_hybrid
done
```

`--tiles` is omitted on purpose: 16 ranks → 15 tiles.

### SLURM Batch Script (all at once)

```bash
#!/bin/bash
#SBATCH --job-name=ocsmesh-hybrid-bench
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --output=hybrid_bench_%j.log

SIZE=120
export OMPI_MCA_mpi_yield_when_idle=1   # keep utilization numbers honest

echo "========================================="
echo "Hercules node: $(hostname)"
echo "Total cores: $(nproc)"
echo "SLURM ntasks: $SLURM_NTASKS"
echo "SLURM cpus-per-task: $SLURM_CPUS_PER_TASK"
echo "========================================="

# Serial baseline: 1 process, all cores via Pool.
# Run it outside srun so idle ranks do not steal CPU from the baseline.
for cfg in A B C D; do
    python tests/benchmarks/benchmark_hybrid_${cfg}.py \
        --mode serial_mp --tiles 15 --size $SIZE \
        --json results_serial_${cfg}.json
done

# Hybrid MPI + Pool
for cfg in A B C D; do
    srun python tests/benchmarks/benchmark_hybrid_${cfg}.py \
        --mode mpi_hybrid --size $SIZE \
        --json results_hybrid_${cfg}.json
done

echo "Done. Check results_*.json for detailed timings."
```

### Save and submit:

```bash
sbatch tests/benchmarks/run_hybrid_bench.sh
```

## CLI reference

All four scripts share the same flags:

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `all` | `serial_mp`, `mpi_no_pool`, `mpi_hybrid`, `all` |
| `--tiles` | `0` | `0` = one tile per worker rank |
| `--size` | `120` | DEM tile is `size x size` pixels |
| `--nprocs` | `0` | serial_mp Pool size; `0` = all visible cores |
| `--cores-per-rank` | `0` | override the auto core budget for `mpi_hybrid` |
| `--json` | — | write timings + utilization + core plan to a file |

## Expected Results

| Config | serial_mp (80 cores) | mpi_hybrid (16×5) | Expected speedup |
|---|---|---|---|
| A (contours) | baseline | fastest | 5-15× |
| B (+ channels) | baseline | fastest | 5-15× |
| C (+ flow/const) | baseline | fastest | 5-15× |
| D (+ constraints) | baseline | limited | 3-10× (constraints are serial) |

On a laptop with tiny tiles (`--size <= 120`) expect **no** speedup: MPI
and Pool startup dominates. Those runs are for correctness, not timing.

## Troubleshooting

### "Fork support not available" warning
The `forkserver` start method should prevent this. If you still see it:
```bash
export OMPI_MCA_mpi_warn_on_fork=0
```

### "There are not enough slots available"
Plain `mpiexec` limits ranks to physical cores. For local testing:
```bash
mpiexec -n 5 --oversubscribe python tests/benchmarks/benchmark_hybrid_B.py
```

### Utilization looks impossibly high
Idle ranks spin in `recv`. Set `OMPI_MCA_mpi_yield_when_idle=1`, and do
not run the `serial_mp` baseline under `srun`/`mpiexec` — the idle ranks
compete with it for CPU and skew the baseline.

### cores/rank is 1 when you expected more
Check the `[source]` tag in the header line the script prints. If it says
`affinity (launcher-bound)`, the launcher pinned each rank to one core:
```bash
srun --cpus-per-task=5 ...        # SLURM
mpiexec --bind-to none -n 16 ...  # Open MPI
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