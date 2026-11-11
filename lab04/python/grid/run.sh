#!/bin/bash
source "_common_" 2>/dev/null
source "grid/common_submit.sh"

clear
rm -f output/*

# Jeden job — edytuj i uruchom: bash grid/run.sh
WORKERS=16
SIZES_ITEM=1280x960

WIDTH="${SIZES_ITEM%x*}"
HEIGHT="${SIZES_ITEM#*x}"

job_id=$(submit_julia_job "$WIDTH" "$HEIGHT" "$WORKERS")
echo "$job_id" | tee grid/.latest_job_started
echo "Job ${job_id}: ${WIDTH}x${HEIGHT} workers=${WORKERS}"
