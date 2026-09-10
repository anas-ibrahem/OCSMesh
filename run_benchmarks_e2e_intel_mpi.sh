#!/bin/bash

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocsmesh-dev

# Generate timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="log_all_${TIMESTAMP}.txt"

# Tee all output (stdout & stderr) to both console and the timestamped log file
exec > >(tee -a "${LOG_FILE}") 2>&1

# 1. Unpin MPI ranks so Python multiprocessing Pool can use all available cores
export I_MPI_PIN=0

# 2. Specify shared memory fabric required by Intel MPI for wait mode on single-node
export I_MPI_FABRICS=shm

# 3. Tell Intel MPI to yield/sleep when idle instead of busy-spinning
export I_MPI_WAIT_MODE=1

SIZE=1500
TILES=16

echo "========================================="
echo "Logging to: ${LOG_FILE}"
echo "Running on node: $(hostname)"
echo "Total cores: $(nproc)"
echo "Intel MPI fabrics: $I_MPI_FABRICS"
echo "Intel MPI wait mode: $I_MPI_WAIT_MODE"
echo "Intel MPI pin mode: $I_MPI_PIN"
echo "Tiles: $TILES, Size: ${SIZE}x${SIZE}"
echo "========================================="

for cfg in A B C D; do
    echo "========================================="
    echo "Running Config $cfg (16 tiles, size 1500)"
    echo "========================================="

    # Runs serial_mp, mpi_no_pool, and mpi_hybrid in sequence with direct comparison output
    PYTHONWARNINGS=ignore mpiexec -n 5 python tests/benchmarks/benchmark_e2e_${cfg}.py \
        --mode all --tiles $TILES --size $SIZE \
        --json results_e2e_${cfg}_${TIMESTAMP}.json
done

echo "Done. All outputs saved to ${LOG_FILE} and timestamped JSON files."
