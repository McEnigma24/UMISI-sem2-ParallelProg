#!/bin/bash
source "_common_" 2>/dev/null

clear
rm -f output/*

proc_number="${1:-4}"
n="${2:-100}"
uv run mpiexec -n "$proc_number" python _src/main_shared.py "$n" | tee output/out
