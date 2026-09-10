#!/bin/bash

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocsmesh-dev

# Generate timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="log_4tiles_heavy_${TIMESTAMP}.txt"

# Tee all output (stdout & stderr) to both console and the timestamped log file
exec > >(tee -a "${LOG_FILE}") 2>&1

# 1. Unpin MPI ranks so Python multiprocessing Pool can use all available cores
export I_MPI_PIN=0

# 2. Specify shared memory fabric required by Intel MPI for wait mode on single-node
export I_MPI_FABRICS=shm

# 3. Tell Intel MPI to yield/sleep when idle instead of busy-spinning
export I_MPI_WAIT_MODE=1

SIZE=3000
TILES=4

echo "========================================="
echo "Logging to: ${LOG_FILE}"
echo "Running on node: $(hostname)"
echo "Total cores: $(nproc)"
echo "Intel MPI fabrics: $I_MPI_FABRICS"
echo "Intel MPI wait mode: $I_MPI_WAIT_MODE"
echo "Intel MPI pin mode: $I_MPI_PIN"
echo "Tiles: $TILES, Size: ${SIZE}x${SIZE} (Heavy Resolution & Features)"
echo "========================================="

for cfg in A B C D; do
    echo "================================================================"
    echo " Running Config $cfg (4 Heavy Tiles 3000x3000, All Features)"
    echo "================================================================"

    # 1. Serial Baseline: 1 Python process + 16 Pool workers
    echo -e "\n---> 1. serial_mp (1 Python process + 16 Pool Workers)..."
    PYTHONWARNINGS=ignore python tests/benchmarks/benchmark_hybrid_${cfg}.py \
        --mode serial_mp --tiles $TILES --size $SIZE --nprocs 16 \
        --json results_4tiles_heavy_${cfg}_serial_${TIMESTAMP}.json

    # 2. Pure MPI: 4 MPI worker ranks (mpiexec -n 5 = 1 coord + 4 workers, 1 core/rank)
    echo -e "\n---> 2. mpi_no_pool (4 Pure MPI Ranks)..."
    PYTHONWARNINGS=ignore mpiexec -n 5 python tests/benchmarks/benchmark_hybrid_${cfg}.py \
        --mode mpi_no_pool --tiles $TILES --size $SIZE \
        --json results_4tiles_heavy_${cfg}_pure_mpi4_${TIMESTAMP}.json

    # 3. Hybrid MPI: 4 MPI worker ranks x 4 Pool workers/rank (mpiexec -n 5 = 1 coord + 4 workers x 4 cores)
    echo -e "\n---> 3. mpi_hybrid (4 MPI Worker Ranks x 4 Pool Workers/rank)..."
    PYTHONWARNINGS=ignore mpiexec -n 5 python tests/benchmarks/benchmark_hybrid_${cfg}.py \
        --mode mpi_hybrid --tiles $TILES --size $SIZE \
        --json results_4tiles_heavy_${cfg}_hybrid4x4_${TIMESTAMP}.json
done

echo "Done. All outputs saved to ${LOG_FILE} and timestamped JSON files."
