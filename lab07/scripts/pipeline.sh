#!/bin/bash
# pipeline.sh local|grid L STEPS BETA P SYNC [REPEATS] [SEED]
set -euo pipefail

source "$(dirname "$0")/../local/_common_" 2>/dev/null || true
if [[ "$(basename "$PWD")" == "scripts" ]]; then
    cd ..
fi

MODE="${1:-local}"
L="${2:-256}"
STEPS="${3:-1000}"
BETA="${4:-1}"
P="${5:-4}"
SYNC="${6:-border}"
REPEATS="${7:-1}"
SEED="${8:-42}"

if [[ "$MODE" == "grid" ]]; then
    NP="${SLURM_NTASKS:?SLURM_NTASKS required}"
    P="$NP"
    launch_mpi() {
        mpirun -np "$NP" uv run python _src/ising.py "$@"
    }
else
    NP="$P"
    launch_mpi() {
        uv run mpiexec --oversubscribe -n "$NP" python _src/ising.py "$@"
    }
fi

MEASURE_EVERY="${MEASURE_EVERY:-$(( STEPS > 100 ? STEPS / 100 : 1 ))}"

RESULTS_ROOT="results/${MODE}"
RUN_DIR="${RESULTS_ROOT}/$(date +%Y-%m-%d_%H-%M-%S)_L${L}_${SYNC}_${P}p"
mkdir -p "$RUN_DIR" "$RESULTS_ROOT"
exec > >(tee "$RUN_DIR/log.txt") 2>&1

echo "=== Pipeline Ising Metropolis ==="
echo "mode=$MODE L=$L steps=$STEPS beta=$BETA P=$P sync=$SYNC repeats=$REPEATS seed=$SEED"
echo "RUN_DIR=$RUN_DIR"

for rep in $(seq 1 "$REPEATS"); do
    OUT="${RUN_DIR}/rep${rep}"
    echo "--- rep ${rep}/${REPEATS} ---"
    SNAP=""
    if [[ "$STEPS" -ge 1000 ]]; then
        SNAP="--snapshot-every $(( STEPS / 10 ))"
    fi
    launch_mpi \
        --L "$L" \
        --steps "$STEPS" \
        --beta "$BETA" \
        --sync "$SYNC" \
        --seed "$(( SEED + rep - 1 ))" \
        --measure-every "$MEASURE_EVERY" \
        $SNAP \
        --out-dir "$OUT" \
        --platform "$MODE"
    echo "--- analyze rep ${rep} ---"
    uv run python _src/analyze.py "$OUT"
done

echo "RUN_DIR=$RUN_DIR"
echo "=== Done ==="