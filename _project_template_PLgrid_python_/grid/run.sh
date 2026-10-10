#!/bin/bash
source "$(dirname "$0")/_common_"

rm -f output/*

# sbatch job
sbatch --parsable grid/job | tee grid/.latest_job_started
