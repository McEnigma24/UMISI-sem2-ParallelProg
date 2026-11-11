#!/bin/bash
# Funkcje wspólne dla grid/run.sh i grid/sweep.sh (source, nie uruchamiaj sam).

submit_julia_job() {
    local width="$1"
    local height="$2"
    local workers="$3"
    export JULIA_WIDTH="$width"
    export JULIA_HEIGHT="$height"
    local np=$((workers + 1))

    sbatch --parsable \
        --ntasks="$np" \
        --job-name="julia_${width}x${height}_${workers}w" \
        --export=ALL,JULIA_WIDTH,JULIA_HEIGHT \
        grid/job
}
