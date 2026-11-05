#!/bin/bash
# Wspólny pipeline: baseline → sleep (z sleep z baseline CSV) → analyze
set -euo pipefail

source "$(dirname "$0")/../local/_common_" 2>/dev/null || true
if [[ "$(basename "$PWD")" == "scripts" ]]; then
    cd ..
fi

MODE="${1:-local}"
WORKERS="${2:-8}"

export JULIA_WIDTH="${JULIA_WIDTH:-1280}"
export JULIA_HEIGHT="${JULIA_HEIGHT:-960}"

mkdir -p results results/plots output

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

echo "=== Pipeline Julia (mode=$MODE, workers=$WORKERS, NP=$NP, ${JULIA_WIDTH}x${JULIA_HEIGHT}) ==="

echo "--- baseline ---"
launch_mpi --tag baseline

SLEEP_TIME=$(uv run python -c "
from pathlib import Path
import sys
sys.path.insert(0, '_src')
from julia import max_worker_time_from_csv
print(f'{max_worker_time_from_csv(Path(\"results/lines_baseline.csv\"), 1.0):.6f}')
")

export JULIA_SLEEP_LINE=1
export JULIA_SLEEP_TIME="$SLEEP_TIME"
echo "--- sleep (linia 1, sleep=${JULIA_SLEEP_TIME}s) ---"
launch_mpi --tag sleep --sleep-line 1 --sleep-from results/lines_baseline.csv --sleep-factor 1.0

echo "--- analyze ---"
uv run python _src/analyze.py

echo "=== Done ==="
