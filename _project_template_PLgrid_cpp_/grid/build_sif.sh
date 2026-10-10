#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Login nodes on Ares do not have apptainer; build on a compute node.
if [[ "${1:-}" != "--on-compute" ]] && ! command -v apptainer &>/dev/null && [[ ! -x /usr/bin/apptainer ]]; then
    echo "No apptainer on this host (expected on Ares login). Starting build via srun on a compute node..."
    exec srun \
        --partition="${PLGRID_PARTITION:-cpu-lowprio}" \
        --account="${PLGRID_ACCOUNT:-plgar2026-cpu}" \
        --time="${PLGRID_BUILD_TIME:-02:00:00}" \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=16G \
        bash "$0" --on-compute "$@"
fi

DEV_ONLY=false
for arg in "$@"; do
    [[ "$arg" == --dev-only ]] && DEV_ONLY=true
done

source "$SCRIPT_DIR/_apptainer.sh"

export APPTAINER_CACHEDIR="${SCRATCH:-/tmp}/${USER}/apptainer-cache"
export APPTAINER_TMPDIR="${SCRATCH:-/tmp}/${USER}/apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$REPO_ROOT/sif"

UBUNTU_TAG="${UBUNTU_TAG:-24.04}"
GEN_DIR="$REPO_ROOT/grid/apptainer/generated"
mkdir -p "$GEN_DIR"

RUNTIME_SIF="$REPO_ROOT/sif/runtime-base.sif"
RUNTIME_DEF="$GEN_DIR/runtime-base.def"
DEV_DEF="$GEN_DIR/dev-env.def"

if [[ "$DEV_ONLY" == false ]]; then
    sed "s/@UBUNTU_TAG@/${UBUNTU_TAG}/g" "$SCRIPT_DIR/apptainer/runtime-base.def.in" > "$RUNTIME_DEF"

    echo "Building $RUNTIME_SIF (ubuntu:${UBUNTU_TAG})"
    apptainer_cmd build --fakeroot -F "$RUNTIME_SIF" "$RUNTIME_DEF"
elif [[ ! -f "$RUNTIME_SIF" ]]; then
    echo "Missing $RUNTIME_SIF — run full ./grid/build_sif.sh first"
    exit 1
fi

sed "s|@RUNTIME_SIF@|${RUNTIME_SIF}|g" "$SCRIPT_DIR/apptainer/dev-env.def.in" > "$DEV_DEF"

DEV_SIF="$REPO_ROOT/sif/dev-env.sif"
echo "Building $DEV_SIF from $RUNTIME_SIF"
apptainer_cmd build --fakeroot -F "$DEV_SIF" "$DEV_DEF"

echo "Done. Images: $DEV_SIF, $RUNTIME_SIF"
