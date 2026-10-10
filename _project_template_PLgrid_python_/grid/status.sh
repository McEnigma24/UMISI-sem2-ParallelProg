#!/bin/bash
source "$(dirname "$0")/_common_"

watch squeue -u $USER
