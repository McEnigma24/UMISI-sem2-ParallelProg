#!/bin/bash
# pipeline.sh local|grid N DEPTH WORKERS [REPEATS] [SEED]
set -euo pipefail

source "$(dirname "$0")/../local/_common_" 2>/dev/null || true
if [[ "$(basename "$PWD")" == "scripts" ]]; then
    cd ..
fi

MODE="${1:-local}"
N="${2:-12}"
DEPTH="${3:-2}"
WORKERS="${4:-4}"
REPEATS="${5:-1}"
SEED="${6:-42}"

if [[ "$MODE" == "grid" ]]; then
    NP="${SLURM_NTASKS:?SLURM_NTASKS required}"
    WORKERS=$((NP - 1))
    launch_mpi() {
        mpirun -np "$NP" uv run python -m mpi4py.futures _src/tsp.py "$@"
    }
else
    NP=$((WORKERS + 1))
    launch_mpi() {
        uv run mpiexec --oversubscribe -n "$NP" python -m mpi4py.futures _src/tsp.py "$@"
    }
fi

RESULTS_ROOT="results/${MODE}"
RUN_DIR="${RESULTS_ROOT}/$(date +%Y-%m-%d_%H-%M-%S)_N${N}_d${DEPTH}_${WORKERS}w"
mkdir -p "$RUN_DIR" "$RESULTS_ROOT"
exec > >(tee "$RUN_DIR/log.txt") 2>&1

echo "=== Pipeline TSP B&B ==="
echo "mode=$MODE N=$N depth=$DEPTH workers=$WORKERS NP=$NP repeats=$REPEATS seed=$SEED"
echo "RUN_DIR=$RUN_DIR"

for rep in $(seq 1 "$REPEATS"); do
    OUT="${RUN_DIR}/rep${rep}"
    echo "--- rep ${rep}/${REPEATS} ---"
    launch_mpi \
        --n "$N" \
        --depth "$DEPTH" \
        --seed "$SEED" \
        --out-dir "$OUT" \
        --platform "$MODE"
    echo "--- analyze rep ${rep} ---"
    uv run python _src/analyze.py "$OUT"
done

echo "RUN_DIR=$RUN_DIR"
echo "=== Done ==="
