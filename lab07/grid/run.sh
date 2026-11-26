#!/bin/bash
source "_common_" 2>/dev/null
source "grid/common_submit.sh"

clear
rm -f output/*

L=3840
WORKERS=16
STEPS=100
export ISING_SEED=42
export ISING_REPEATS=1

job_id=$(submit_ising_job "$L" "$WORKERS" "$STEPS")
echo "$job_id" | tee grid/.latest_job_started
echo "Job ${job_id}: L=${L} P=${WORKERS} steps=${STEPS}"