#!/bin/bash
# T(1) dla sweepu — compare.py (S, E, f)
source "_common_" 2>/dev/null

L="${SWEEP_L:-960}"
STEPS="${SWEEP_STEPS:-200}"
SEED="${SWEEP_SEED:-42}"
for sync in border smart; do
    echo "######## T(1) L=${L} sync=${sync} ########"
    bash scripts/pipeline.sh local "$L" "$STEPS" 1 1 "$sync" 1 "$SEED"
done
echo "Potem: uv run python _src/compare.py"