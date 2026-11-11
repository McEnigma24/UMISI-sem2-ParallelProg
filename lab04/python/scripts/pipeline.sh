#!/bin/bash
# pipeline.sh [local|grid] WIDTH HEIGHT [WORKERS]
set -euo pipefail

source "$(dirname "$0")/../local/_common_" 2>/dev/null || true
if [[ "$(basename "$PWD")" == "scripts" ]]; then
    cd ..
fi

MODE="${1:-local}"
WIDTH="${2:-1280}"
HEIGHT="${3:-960}"
WORKERS="${4:-8}"

if [[ "$MODE" == "grid" ]]; then
    NP="${SLURM_NTASKS:?SLURM_NTASKS required}"
    WORKERS=$((NP - 1))
    launch_mpi() {
        mpirun -np "$NP" uv run python -m mpi4py.futures _src/julia.py "$@"
    }
else
    NP=$((WORKERS + 1))
    launch_mpi() {
        uv run mpiexec --oversubscribe -n "$NP" python -m mpi4py.futures _src/julia.py "$@"
    }
fi

RESULTS_ROOT="results/${MODE}"
RUN_DIR="${RESULTS_ROOT}/$(date +%Y-%m-%d_%H-%M-%S)_${WIDTH}x${HEIGHT}_${WORKERS}w"
mkdir -p "$RUN_DIR/baseline" "$RUN_DIR/sleep" "$RESULTS_ROOT"
exec > >(tee "$RUN_DIR/log.txt") 2>&1

echo "=== Pipeline Julia ==="
echo "mode=$MODE results=$RESULTS_ROOT workers=$WORKERS NP=$NP ${WIDTH}x${HEIGHT}"
echo "RUN_DIR=$RUN_DIR"

echo "--- baseline ---"
launch_mpi \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --out-dir "$RUN_DIR/baseline" \
    --variant baseline \
    --platform "$MODE"

export JULIA_SLEEP_LINE=1
echo "--- sleep ---"
launch_mpi \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --out-dir "$RUN_DIR/sleep" \
    --variant sleep \
    --sleep-line 1 \
    --sleep-from "$RUN_DIR/baseline" \
    --sleep-factor 1.0 \
    --platform "$MODE"

echo "--- analyze ---"
uv run python _src/analyze.py "$RUN_DIR/baseline"
uv run python _src/analyze.py "$RUN_DIR/sleep"

echo "RUN_DIR=$RUN_DIR"
echo "=== Done ==="
