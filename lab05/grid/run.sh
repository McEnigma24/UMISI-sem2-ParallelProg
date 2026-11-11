#!/bin/bash
source "_common_" 2>/dev/null
source "grid/common_submit.sh"

clear
rm -f output/*

# Jeden job — edytuj i uruchom: bash grid/run.sh
N=14
WORKERS=16
D_LIST="1 2 3 4 5"
export TSP_SEED=42
export TSP_REPEATS=1

job_id=$(submit_tsp_job "$N" "$WORKERS" "$D_LIST")
echo "$job_id" | tee grid/.latest_job_started
echo "Job ${job_id}: N=${N} workers=${WORKERS} d in (${D_LIST})"
