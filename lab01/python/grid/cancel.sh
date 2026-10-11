#!/bin/bash
source "_common_" 2>/dev/null

scancel $(cat grid/.latest_job_started)
