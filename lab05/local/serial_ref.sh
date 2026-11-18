#!/bin/bash
# T(1) dla tego samego N co grid — do compare (S, E, f). Użyj: SWEEP_N=14 bash local/serial_ref.sh
source "_common_" 2>/dev/null

N="${SWEEP_N:-14}"
SEED="${SWEEP_SEED:-42}"
for d in 1 2 3 4 5; do
    echo "######## T(1) N=${N} d=${d} ########"
    bash scripts/pipeline.sh local "$N" "$d" 1 1 "$SEED"
done
echo "Potem: uv run python _src/compare.py"
