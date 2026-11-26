#!/bin/bash
source "_common_" 2>/dev/null

L=256
STEPS=10000
P=4
SEED=42

mkdir -p results/local results/compare

for BETA in 1 4; do
    echo "######## wizualizacja beta=${BETA} ########"
    bash scripts/pipeline.sh local "$L" "$STEPS" "$BETA" "$P" border 1 "$SEED"
    run_dir=$(ls -td results/local/*_L${L}_border_${P}p | head -1)
    dest="results/local/viz_beta${BETA}/rep1"
    mkdir -p "$(dirname "$dest")"
    rm -rf "$dest"
    cp -a "${run_dir}/rep1" "$dest"
done

uv run python _src/compare.py