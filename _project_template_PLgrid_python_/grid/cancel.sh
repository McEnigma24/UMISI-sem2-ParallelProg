#!/bin/bash
source "$(dirname "$0")/_common_"

scancel $(cat grid/.latest_job_started)
