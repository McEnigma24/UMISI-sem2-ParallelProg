#!/bin/bash
source "_common_" 2>/dev/null

clear
rm -f output/*
mkdir -p results results/plots

WORKERS="${1:-8}"
export JULIA_WIDTH=1280
export JULIA_HEIGHT=960

bash scripts/pipeline.sh local "$WORKERS" | tee output/out
