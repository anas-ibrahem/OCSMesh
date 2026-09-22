#!/bin/bash
# =============================================================================
#  HfunCollector.__init__() scalability bug — reproduction & fix validation
#
#  Proves the N_ranks × N_tiles concurrent temp-file write amplification:
#
#    Phase 1 — Bug reproduction:  run WITHOUT the is_manager() guard
#              (revert the fix temporarily or use a patched build).
#              Expected: ranks × tiles tmp .tif files written.
#
#    Phase 2 — Fix validation:    run WITH the is_manager() guard in place.
#              Expected: tiles tmp .tif files written (rank 0 only).
#
#  Both phases run the same script. The script itself detects the current
#  state of the code and reports BUG REPRODUCED or FIX CONFIRMED.
#
#  Quick run (16 ranks = 15 tiles + 1 coordinator, small tiles):
#    bash tests/benchmarks_mpi/run_mock_init_scaling.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${SCRIPT_DIR}/mock_init_scaling.py"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${SCRIPT_DIR}/log_init_scaling_${TIMESTAMP}.txt"

exec > >(tee -a "${LOG_FILE}") 2>&1

# Intel MPI environment (same as e2e benchmarks)
export I_MPI_PIN=0
export I_MPI_FABRICS=shm
export I_MPI_WAIT_MODE=1

echo "========================================="
echo " Init-scaling benchmark — ${TIMESTAMP}"
echo " Logging to: ${LOG_FILE}"
echo " Node: $(hostname)  Cores: $(nproc)"
echo "========================================="


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1: Small run — quick reproduction check
#  16 ranks, 15 tiles, small tile size (fast, ~10s)
#  Shows the bug/fix clearly with minimal resources.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " PHASE 1: Reproduction check (15 tiles, 120×120, 16 ranks)"
echo " Expected BUG: 16 × 15 = 240 tmp files"
echo " Expected FIX:      15 tmp files (rank 0 only)"
echo "================================================================"

PYTHONWARNINGS=ignore mpiexec -n 16 python "${BENCH}" \
    --tiles 15 --size 120 \
    --json "${SCRIPT_DIR}/results_init_scaling_small_${TIMESTAMP}.json"

sync; sleep 5


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2: Larger run — realistic tile size
#  16 ranks, 15 tiles, 500×500 (closer to real workload)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " PHASE 2: Realistic run (15 tiles, 500×500, 16 ranks)"
echo " Expected BUG: 16 × 15 = 240 tmp files"
echo " Expected FIX:      15 tmp files (rank 0 only)"
echo "================================================================"

PYTHONWARNINGS=ignore mpiexec -n 16 python "${BENCH}" \
    --tiles 15 --size 500 \
    --json "${SCRIPT_DIR}/results_init_scaling_large_${TIMESTAMP}.json"

sync; sleep 5


echo ""
echo "Done. Log:  ${LOG_FILE}"
echo "JSON: ${SCRIPT_DIR}/results_init_scaling_*_${TIMESTAMP}.json"
