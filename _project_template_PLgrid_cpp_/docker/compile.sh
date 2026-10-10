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
clear_dir "$DIR_LOG"; set +euo pipefail # allowing script to run after errors
docker run --rm -it \
  "${DOCKER_HOST_USER[@]}" \
  -v "$(pwd):/workspace" \
  -w /workspace \
  "$DOCKER_FULL_IMG_NAME" \
  bash "./docker/_container_compile.sh" "$@"

compilation_status=$?
docker container prune -f


clear # clearing normal cmake build logs ->
cat $LOG_container_compile && echo -e "\n"
if [ $compilation_status -eq 0 ]; then
  echo "✅ SUCCESS"
  exit 0
else
  echo "❌ FAILED"
  exit 1
fi
