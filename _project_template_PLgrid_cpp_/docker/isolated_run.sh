#!/bin/bash
source config

# Wariant przenośny: binarka, input/ i run_time_config/ są wkopiowane do obrazu,
# więc kontener odpala się gdziekolwiek, bez repo pod ręką. Zapisy do output/
# zostają w środku kontenera i giną razem z nim.
# Wariant "na repo" (mount jak w dev-env) -> docker/run.sh

./docker/compile.sh "$@" || exit 1

# DOCKER_IMG_PREFIX
DOCKER_TARGET="runner"
DOCKER_FULL_IMG_NAME="${DOCKER_IMG_PREFIX}${DOCKER_TARGET}"



# BUILD #
clear; # all things before (like compile.sh) - no need for allowing to run after errors, there is nothing to clear
docker build --target "$DOCKER_TARGET" -t "$DOCKER_FULL_IMG_NAME" .
docker image prune -f


# RUN #
clear; # clearing docker build logs
clear_dir "$DIR_OUTPUT";
set +euo pipefail # allowing script to run after errors

container_id="$(docker run -d \
  "$DOCKER_FULL_IMG_NAME" \
  bash -lc 'exec /app/build/*.exe')"

stdbuf -oL docker logs -f "$container_id" 2>&1 | tee "$LOG_run" &
logs_pid=$!

run_status="$(docker wait "$container_id")"
wait "$logs_pid" &>/dev/null

docker rm -f "$container_id" &>/dev/null
docker container prune -f &>/dev/null


echo -en "\n\n" | tee -a "$LOG_run"
if [ "$run_status" -eq 0 ]; then
  echo "✅ SUCCESS" | tee -a "$LOG_run"
  exit 0
else
  echo "❌ FAILED" | tee -a "$LOG_run"
  exit 1
fi
