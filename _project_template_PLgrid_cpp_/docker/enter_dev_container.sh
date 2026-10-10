#!/bin/bash
source config



# DOCKER_IMG_PREFIX
DOCKER_TARGET="dev-env"
DOCKER_FULL_IMG_NAME="${DOCKER_IMG_PREFIX}${DOCKER_TARGET}"



# BUILD #
clear; # all things before - no need for allowing to run after errors, there is nothing to clear
docker build --target "$DOCKER_TARGET" -t "$DOCKER_FULL_IMG_NAME" .
docker image prune -f

# RUN #
clear; # clearing docker build logs
set +euo pipefail # not needed but, clears container if it run failes (no chance)
docker run --rm -it \
  "${DOCKER_HOST_USER[@]}" \
  -v "$(pwd):/workspace" \
  -w /workspace \
  "$DOCKER_FULL_IMG_NAME"

docker container prune -f
clear
