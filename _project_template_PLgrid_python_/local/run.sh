#!/bin/bash
source "_common_" 2>/dev/null

rm -f output/*

proc_number="${1:-4}"
uv run mpiexec -n "$proc_number" python _src/main.py
