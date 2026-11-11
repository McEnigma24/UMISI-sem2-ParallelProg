#!/bin/bash
source "_common_" 2>/dev/null
source "grid/common_submit.sh"

N=14
WORKERS=(4 8 16 24 32 48)
D_LIST="1 2 3 4 5"
export TSP_SEED=42
export TSP_REPEATS=1

LOG="grid/submitted_jobs.txt"
mkdir -p grid output

{
    echo "# $(date -Iseconds) grid TSP sweep"
} >>"$LOG"

for p in "${WORKERS[@]}"; do
    echo "######## submit N=${N} workers=${p} ########"
    job_id=$(submit_tsp_job "$N" "$p" "$D_LIST")
    echo "${job_id} N=${N} workers=${p}" | tee -a "$LOG"
done

echo "Zgłoszono joby. Lista: $LOG"
echo "Po pobraniu wyników: uv run python _src/compare.py"
