#!/bin/bash
source "_common_" 2>/dev/null

# Liczba workerów (procesów MPI = worker + master)
WORKERS=(2 4 6 8 12)

# Rozmiary obrazu W x H
SIZES=(
    320x240
    640x480
    960x720
    1280x960
    1600x1200
    1920x1440
    2560x1920
    3840x1920   # szerszy (ta sama H)
    2560x3840   # dłuższy (ta sama W)
    2048x1536   # mniej Mpx, 4:3
)

for n in "${WORKERS[@]}"; do
    for sz in "${SIZES[@]}"; do
        WIDTH="${sz%x*}"
        HEIGHT="${sz#*x}"
        echo "######## sweep ${WIDTH}x${HEIGHT} workers=${n} ########"
        bash scripts/pipeline.sh local "$WIDTH" "$HEIGHT" "$n"
    done
done

uv run python _src/compare.py
