#!/bin/bash
# On Ares, apptainer is in PATH only on compute nodes (/usr/bin/apptainer).

apptainer_cmd() {
    if command -v apptainer &>/dev/null; then
        apptainer "$@"
    elif [[ -x /usr/bin/apptainer ]]; then
        /usr/bin/apptainer "$@"
    else
        echo "apptainer not found. On Ares it is available only on compute nodes." >&2
        echo "Run ./grid/build_sif.sh (re-submits via srun) or submit a batch job." >&2
        return 127
    fi
}
