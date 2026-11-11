#!/bin/bash
source "_common_" 2>/dev/null
source "grid/common_submit.sh"

# Na Aresie zwykle większe joby niż lokalnie — edytuj listy wg potrzeb.
WORKERS=(8 16 24 32 48)

SIZES=(
    1280x960
    1600x1200
    1920x1440
    2560x1920
    3200x2400
    3840x2880
    5120x3840
    6400x4800
)

LOG="grid/submitted_jobs.txt"
mkdir -p grid output

{
    echo "# $(date -Iseconds) grid sweep"
} >>"$LOG"

for n in "${WORKERS[@]}"; do
    for sz in "${SIZES[@]}"; do
        WIDTH="${sz%x*}"
        HEIGHT="${sz#*x}"
        echo "######## submit ${WIDTH}x${HEIGHT} workers=${n} ########"
        job_id=$(submit_julia_job "$WIDTH" "$HEIGHT" "$n")
        echo "${job_id} ${WIDTH}x${HEIGHT} workers=${n}" | tee -a "$LOG"
    done
done

echo "Zgłoszono joby. Lista: $LOG"
