#!/bin/bash
# Funkcje wspólne dla grid/run.sh i grid/sweep.sh (source, nie uruchamiaj sam).

submit_tsp_job() {
    local n="$1"
    local workers="$2"
    local d_list="${3:-1 2 3 4 5}"
    export TSP_N="$n"
    export TSP_WORKERS="$workers"
    export TSP_D_LIST="$d_list"
    export TSP_SEED="${TSP_SEED:-42}"
    export TSP_REPEATS="${TSP_REPEATS:-1}"
    local np=$((workers + 1))

    sbatch --parsable \
        --ntasks="$np" \
        --job-name="tsp_N${n}_${workers}w" \
        --export=ALL,TSP_N,TSP_WORKERS,TSP_D_LIST,TSP_SEED,TSP_REPEATS \
        grid/job
}
