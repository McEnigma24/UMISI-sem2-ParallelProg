#!/bin/bash
source "_common_" 2>/dev/null

# Kalibracja: STEPS tak, aby T(1) ~ 15 s (dostosuj po pierwszym smoke)
STEPS="${SWEEP_STEPS:-200}"
BETA=1
SEED="${SWEEP_SEED:-42}"
REPEATS="${SWEEP_REPEATS:-3}"

L_LIST=(960 1920)
P_LIST=(1 2 3 4 6 8 12)
SYNC_LIST=(border smart)

mkdir -p results/local results/compare

for L in "${L_LIST[@]}"; do
    for sync in "${SYNC_LIST[@]}"; do
        for p in "${P_LIST[@]}"; do
            if (( L % p != 0 )); then
                continue
            fi
            echo "######## sweep L=${L} sync=${sync} P=${p} ########"
            bash scripts/pipeline.sh local "$L" "$STEPS" "$BETA" "$p" "$sync" "$REPEATS" "$SEED"
        done
    done
done

uv run python _src/compare.py