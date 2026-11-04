#!/bin/bash
source "_common_" 2>/dev/null

watch squeue -u $USER
