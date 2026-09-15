#!/bin/bash
# =============================================================================
#  Config D: full pipeline
#  (contours + channels + flow limiter + const val + patch + constraint)
#
#  ISOLATION STRATEGY:
#    Phase 1 — Quick correctness check: all modes in one process (small tiles)
#              to verify raster-exact matching between serial and MPI.
#    Phase 2 — Fair timing: each mode in a SEPARATE mpiexec invocation with
#              sync+sleep between runs.  This prevents cumulative memory
#              fragmentation, OS page-cache warming, and Python allocator
#              artifacts from contaminating later runs.
#
#  NOTE: patches and constraints are NOT MPI-parallelized yet — they run
#        serially on rank 0 in all modes.  The timing gains come from
#        contours, channels, flow limiters, and const_val.
# =============================================================================

set -euo pipefail

# Generate timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="log_e2e_D_${TIMESTAMP}.txt"

# Tee all output (stdout & stderr) to both console and the timestamped log file
exec > >(tee -a "${LOG_FILE}") 2>&1

# Intel MPI environment
export I_MPI_PIN=0            # Unpin ranks so Pool workers can use all cores
export I_MPI_FABRICS=shm      # Shared memory fabric for single-node
export I_MPI_WAIT_MODE=1      # Yield/sleep when idle instead of busy-spinning

BENCH="tests/benchmarks/benchmark_e2e_D.py"

echo "========================================="
echo " Config D Benchmark — ${TIMESTAMP}"
echo " Logging to: ${LOG_FILE}"
echo " Node: $(hostname)  Cores: $(nproc)"
echo " I_MPI_PIN=$I_MPI_PIN  I_MPI_FABRICS=$I_MPI_FABRICS  I_MPI_WAIT_MODE=$I_MPI_WAIT_MODE"
echo "========================================="


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1: CORRECTNESS VALIDATION
#  All modes run together in one process with small tiles.
#  Fast (~30s).  Confirms raster-exact matching (OK / FAIL).
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " PHASE 1: Correctness validation (4 tiles, 300x300, 5 ranks)"
echo "================================================================"
PYTHONWARNINGS=ignore mpiexec -n 5 python ${BENCH} \
    --mode all --tiles 4 --size 300 \
    --json results_D_correctness_${TIMESTAMP}.json

sync; sleep 5


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2: FAIR TIMING BENCHMARKS
#  Each mode runs in a SEPARATE mpiexec process for clean memory state.
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------
# VARIANT 1: 15 Tiles (1500x1500), 16 ranks (1 core/rank)
# mpi_hybrid degenerates into mpi_no_pool here (no room
# for per-rank Pools), but tests correctness at scale.
# ---------------------------------------------------------
SIZE_15=1500
TILES_15=15
RANKS_15=16

echo ""
echo "================================================================"
echo " PHASE 2 — VARIANT 1: 15 tiles (${SIZE_15}x${SIZE_15})"
echo "================================================================"

echo ""
echo "--- [V1] serial_mp (Pool nprocs=$(nproc)) ---"
PYTHONWARNINGS=ignore python ${BENCH} \
    --mode serial_mp --tiles $TILES_15 --size $SIZE_15 \
    --json results_D_15t_serial_${TIMESTAMP}.json
sync; sleep 5

echo ""
echo "--- [V1] mpi_no_pool (${RANKS_15} ranks, 1 core/rank) ---"
PYTHONWARNINGS=ignore mpiexec -n $RANKS_15 python ${BENCH} \
    --mode mpi_no_pool --tiles $TILES_15 --size $SIZE_15 \
    --json results_D_15t_npool_${TIMESTAMP}.json
sync; sleep 5

echo ""
echo "--- [V1] mpi_hybrid (${RANKS_15} ranks, 1 core/rank — degenerates) ---"
PYTHONWARNINGS=ignore mpiexec -n $RANKS_15 python ${BENCH} \
    --mode mpi_hybrid --tiles $TILES_15 --size $SIZE_15 \
    --json results_D_15t_hybrid_${TIMESTAMP}.json
sync; sleep 5


# ---------------------------------------------------------
# VARIANT 2: 4 Heavy Tiles (3000x3000), 5 ranks (4 cores/rank)
# True hybrid config: each worker gets 4 cores for its Pool.
# ---------------------------------------------------------
SIZE_HEAVY=3000
TILES_HEAVY=4
RANKS_HEAVY=5

echo ""
echo "================================================================"
echo " PHASE 2 — VARIANT 2: 4 heavy tiles (${SIZE_HEAVY}x${SIZE_HEAVY})"
echo "================================================================"

echo ""
echo "--- [V2] serial_mp (Pool nprocs=$(nproc)) ---"
PYTHONWARNINGS=ignore python ${BENCH} \
    --mode serial_mp --tiles $TILES_HEAVY --size $SIZE_HEAVY \
    --json results_D_4t_serial_${TIMESTAMP}.json
sync; sleep 5

echo ""
echo "--- [V2] mpi_no_pool (${RANKS_HEAVY} ranks, 1 core/rank) ---"
PYTHONWARNINGS=ignore mpiexec -n $RANKS_HEAVY python ${BENCH} \
    --mode mpi_no_pool --tiles $TILES_HEAVY --size $SIZE_HEAVY \
    --json results_D_4t_npool_${TIMESTAMP}.json
sync; sleep 5

echo ""
echo "--- [V2] mpi_hybrid (${RANKS_HEAVY} ranks, ~4 cores/rank) ---"
PYTHONWARNINGS=ignore mpiexec -n $RANKS_HEAVY python ${BENCH} \
    --mode mpi_hybrid --tiles $TILES_HEAVY --size $SIZE_HEAVY \
    --json results_D_4t_hybrid_${TIMESTAMP}.json
sync; sleep 5


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY: Read JSON files and print timing comparison table
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " TIMING SUMMARY"
echo "================================================================"

python3 - "${TIMESTAMP}" <<'PYEOF'
import json, sys, os

ts = sys.argv[1]
variants = [
    ("15t", "15 tiles (1500x1500), 16 ranks"),
    ("4t",  "4 heavy tiles (3000x3000), 5 ranks"),
]
modes = [
    ("serial", "serial_mp"),
    ("npool",  "mpi_no_pool"),
    ("hybrid", "mpi_hybrid"),
]

def read_wall(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    for v in d.get("results", {}).values():
        return v.get("wall_s")
    return None

for tag, desc in variants:
    print(f"\n  === {desc} ===")
    walls = {}
    for file_tag, mode_name in modes:
        path = f"results_D_{tag}_{file_tag}_{ts}.json"
        w = read_wall(path)
        walls[mode_name] = w
        if w is not None:
            print(f"    {mode_name:15s}  {w:8.1f}s", end="")
            if walls.get("serial_mp") and mode_name != "serial_mp":
                print(f"   speedup {walls['serial_mp'] / w:.2f}x", end="")
            print()
        else:
            print(f"    {mode_name:15s}  (no data)")

    # Check mpi_hybrid vs mpi_no_pool parity at 1 core/rank
    np_w = walls.get("mpi_no_pool")
    hy_w = walls.get("mpi_hybrid")
    if np_w and hy_w and tag == "15t":
        ratio = hy_w / np_w
        if 0.85 <= ratio <= 1.15:
            print(f"    ✓ mpi_hybrid / mpi_no_pool = {ratio:.2f}x (within 15% — good)")
        else:
            print(f"    ⚠ mpi_hybrid / mpi_no_pool = {ratio:.2f}x (>15% gap — investigate)")

print()
PYEOF

echo "Done. Log: ${LOG_FILE}"
echo "JSON files: results_D_*_${TIMESTAMP}.json"
