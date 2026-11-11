#!/bin/bash
source "_common_" 2>/dev/null

clear
rm -f output/*
mkdir -p results/local results/grid results/compare

WORKERS="${1:-8}"
WIDTH="${2:-1280}"
HEIGHT="${3:-960}"

bash scripts/pipeline.sh local "$WIDTH" "$HEIGHT" "$WORKERS" 2>&1 | tee output/out
uv run python _src/compare.py
