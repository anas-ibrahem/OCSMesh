#!/bin/bash

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocsmesh-dev

# Generate timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="log_e2e_CD_${TIMESTAMP}.txt"

# Tee all output (stdout & stderr) to both console and the timestamped log file
exec > >(tee -a "${LOG_FILE}") 2>&1

# 1. Unpin MPI ranks so Python multiprocessing Pool can use all available cores
export I_MPI_PIN=0

# 2. Specify shared memory fabric required by Intel MPI for wait mode on single-node
export I_MPI_FABRICS=shm

# 3. Tell Intel MPI to yield/sleep when idle instead of busy-spinning
export I_MPI_WAIT_MODE=1

echo "========================================="
echo "Logging to: ${LOG_FILE}"
echo "Running on node: $(hostname)"
echo "Total cores: $(nproc)"
echo "Intel MPI fabrics: $I_MPI_FABRICS"
echo "Intel MPI wait mode: $I_MPI_WAIT_MODE"
echo "Intel MPI pin mode: $I_MPI_PIN"
echo "========================================="

# ---------------------------------------------------------
# VARIANT 1: 15 Tiles (Size 1500x1500)
# 16 ranks = 1 coordinator + 15 workers (1 core/rank)
# On a 16-core node, this gives 1 core per worker rank.
# mpi_hybrid degenerates into mpi_no_pool here (no room
# for thread pools), but it tests full e2e correctness
# with many tiles and verifies no deadlocks.
# ---------------------------------------------------------
SIZE_15=1500
TILES_15=15
RANKS_15=16

for cfg in C D; do
    echo "================================================================"
    echo " Config $cfg — 15 tiles (1500x1500), 16 ranks (1 core/rank)"
    echo "================================================================"
    PYTHONWARNINGS=ignore mpiexec -n $RANKS_15 python tests/benchmarks/benchmark_e2e_${cfg}.py \
        --mode all --tiles $TILES_15 --size $SIZE_15 \
        --json results_e2e_${cfg}_15tiles_${TIMESTAMP}.json
done

# ---------------------------------------------------------
# VARIANT 2: 4 Heavy Tiles (Size 3000x3000)
# 5 ranks = 1 coordinator + 4 workers (4 cores/rank)
# This is the true hybrid config: each worker gets 4 cores
# for its thread pool, showing the real speedup from
# intra-tile parallelism.
# ---------------------------------------------------------
SIZE_HEAVY=3000
TILES_HEAVY=4
RANKS_HEAVY=5

for cfg in C D; do
    echo "================================================================"
    echo " Config $cfg — 4 heavy tiles (3000x3000), 5 ranks (4 cores/rank)"
    echo "================================================================"
    PYTHONWARNINGS=ignore mpiexec -n $RANKS_HEAVY python tests/benchmarks/benchmark_e2e_${cfg}.py \
        --mode all --tiles $TILES_HEAVY --size $SIZE_HEAVY \
        --json results_e2e_${cfg}_heavy_${TIMESTAMP}.json
done

echo "Done. All outputs saved to ${LOG_FILE} and timestamped JSON files."
