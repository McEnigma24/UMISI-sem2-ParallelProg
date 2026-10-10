#!/bin/bash
source "_common_" 2>/dev/null

rm -f output/*

# sbatch job
sbatch --parsable grid/job | tee grid/.latest_job_started
