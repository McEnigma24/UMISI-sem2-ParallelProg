#!/bin/bash
source "_common_" 2>/dev/null

# Ustal po kalibracji (smoke): rozmiar instancji TSP
N="${SWEEP_N:-12}"
SEED="${SWEEP_SEED:-42}"
REPEATS="${SWEEP_REPEATS:-1}"

WORKERS=(1 2 4 6 8 11)
D_LIST=(1 2 3 4 5)

mkdir -p results/local results/grid results/compare

for d in "${D_LIST[@]}"; do
    for p in "${WORKERS[@]}"; do
        echo "######## sweep N=${N} d=${d} workers=${p} ########"
        bash scripts/pipeline.sh local "$N" "$d" "$p" "$REPEATS" "$SEED"
    done
done

uv run python _src/compare.py
