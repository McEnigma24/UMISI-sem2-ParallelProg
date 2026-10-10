#!/bin/bash

proc_number="${1:-4}"

uv run mpiexec -n "$proc_number" python _src/main.py
