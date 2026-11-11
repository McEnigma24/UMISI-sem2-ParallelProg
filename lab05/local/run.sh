#!/bin/bash
source "_common_" 2>/dev/null

clear
rm -f output/*
mkdir -p results/local results/grid results/compare

WORKERS="${1:-4}"
N="${2:-12}"
DEPTH="${3:-2}"
REPEATS="${4:-1}"
SEED="${5:-42}"

bash scripts/pipeline.sh local "$N" "$DEPTH" "$WORKERS" "$REPEATS" "$SEED" 2>&1 | tee output/out
uv run python _src/compare.py
