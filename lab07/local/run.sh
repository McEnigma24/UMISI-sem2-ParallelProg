#!/bin/bash
source "_common_" 2>/dev/null

clear
mkdir -p results/local results/grid results/compare output

P="${1:-4}"
L="${2:-256}"
STEPS="${3:-1000}"
BETA="${4:-1}"
SYNC="${5:-border}"
REPEATS="${6:-1}"
SEED="${7:-42}"

bash scripts/pipeline.sh local "$L" "$STEPS" "$BETA" "$P" "$SYNC" "$REPEATS" "$SEED" 2>&1 | tee output/out
uv run python _src/compare.py