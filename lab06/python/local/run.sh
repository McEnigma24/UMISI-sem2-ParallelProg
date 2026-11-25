#!/bin/bash
source "_common_" 2>/dev/null

clear
rm -f output/*
mkdir -p results/local results/compare output

PROBLEM="${1:-density}"
BACKEND="${2:-cupy_kernel}"
N_LIST="${3:-1e3 1e4 1e5 3e5 1e6}"

bash scripts/pipeline.sh "$PROBLEM" "$BACKEND" 1e-3 "$N_LIST" 1 42 2>&1 | tee output/out
uv run python _src/compare.py