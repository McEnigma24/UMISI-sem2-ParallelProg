#!/bin/bash
# pipeline.sh PROBLEM BACKEND DT "N_LIST" REPEATS SEED
set -euo pipefail

if [[ "$(basename "$PWD")" == "scripts" ]]; then
    cd ..
fi

PROBLEM="${1:-density}"
BACKEND="${2:-cupy_kernel}"
DT="${3:-1e-3}"
N_LIST="${4:-1e5}"
REPEATS="${5:-1}"
SEED="${6:-42}"

DT_TAG="${DT//./p}"
RESULTS_ROOT="results/local"
RUN_DIR="${RESULTS_ROOT}/$(date +%Y-%m-%d_%H-%M-%S)_${PROBLEM}_${BACKEND}_dt${DT_TAG}"
mkdir -p "$RESULTS_ROOT" "$RUN_DIR"
exec > >(tee "$RUN_DIR/log.txt") 2>&1

echo "=== Pipeline Langevin GPU ==="
echo "problem=$PROBLEM backend=$BACKEND dt=$DT n_list=$N_LIST repeats=$REPEATS seed=$SEED"
echo "RUN_DIR=$RUN_DIR"

export PYTHONPATH="${PWD}/_src:${PYTHONPATH:-}"
if [[ -d /usr/lib/wsl/lib ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -d /usr/lib/cuda ]]; then
    export CUDA_PATH="${CUDA_PATH:-/usr/lib/cuda}"
fi

uv run python _src/main.py \
    --problem "$PROBLEM" \
    --backend "$BACKEND" \
    --dt "$DT" \
    --n-list "$N_LIST" \
    --repeats "$REPEATS" \
    --seed "$SEED" \
    --out-dir "$RUN_DIR"

uv run python _src/analyze.py "$RUN_DIR"

echo "RUN_DIR=$RUN_DIR"
echo "=== Done ==="