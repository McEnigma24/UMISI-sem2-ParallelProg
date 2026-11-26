#!/bin/bash

submit_ising_job() {
    local L="$1"
    local workers="$2"
    local steps="${3:-100}"
    export ISING_L="$L"
    export ISING_STEPS="$steps"
    export ISING_WORKERS="$workers"
    export ISING_SEED="${ISING_SEED:-42}"
    export ISING_REPEATS="${ISING_REPEATS:-1}"
    export ISING_BETA="${ISING_BETA:-1}"
    export ISING_SYNC_LIST="${ISING_SYNC_LIST:-border smart}"
    local np="$workers"

    sbatch --parsable \
        --ntasks="$np" \
        --job-name="ising_L${L}_${workers}p" \
        --export=ALL,ISING_L,ISING_STEPS,ISING_WORKERS,ISING_SEED,ISING_REPEATS,ISING_BETA,ISING_SYNC_LIST \
        grid/job
}