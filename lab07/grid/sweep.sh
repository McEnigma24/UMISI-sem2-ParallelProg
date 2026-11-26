#!/bin/bash
source "_common_" 2>/dev/null
source "grid/common_submit.sh"

L=3840
STEPS=100
WORKERS=(1 2 4 8 12 16 24 32 48)
export ISING_SEED=42
export ISING_REPEATS=1

LOG="grid/submitted_jobs.txt"
mkdir -p grid output

{
    echo "# $(date -Iseconds) grid Ising sweep"
} >>"$LOG"

for p in "${WORKERS[@]}"; do
    if (( L % p != 0 )); then
        continue
    fi
    echo "######## submit L=${L} P=${p} ########"
    job_id=$(submit_ising_job "$L" "$p" "$STEPS")
    echo "${job_id} L=${L} P=${p}" | tee -a "$LOG"
done

echo "Zgłoszono joby. Lista: $LOG"