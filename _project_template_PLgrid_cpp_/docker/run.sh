#!/bin/bash
source config

# Wariant "na repo": kontener dostaje dokładnie ten sam mount co dev-env i uruchamia
# binarkę, która już leży w repo. input/, output/ i log/ to te prawdziwe katalogi
# z dysku, więc widać efekty od razu, bez wchodzenia do obrazu.
# Wariant izolowany (wszystko wkopiowane do obrazu) -> docker/run_isolated.sh

./docker/compile.sh "$@" || exit 1

# Obraz to gołe runtime-base, czyli to samo środowisko co runner, tylko bez COPY.
# Dzięki temu nie trzeba przebudowywać obrazu po każdej zmianie kodu,
# a binarka i tak startuje wyłącznie na bibliotekach runtime (bez toolchainu).
# DOCKER_IMG_PREFIX
DOCKER_TARGET="runtime-base"
DOCKER_FULL_IMG_NAME="${DOCKER_IMG_PREFIX}${DOCKER_TARGET}"



# BUILD #
clear; # all things before (like compile.sh) - no need for allowing to run after errors, there is nothing to clear
docker build --target "$DOCKER_TARGET" -t "$DOCKER_FULL_IMG_NAME" .
docker image prune -f


# RUN #
clear; # clearing docker build logs
clear_dir "$DIR_OUTPUT";
set +euo pipefail # allowing script to run after errors

if ! ls "$DIR_TARGET"/*.exe > /dev/null 2>&1; then
  echo "❌ FAILED - brak binarki w $DIR_TARGET/ (flaga -l buduje bibliotekę, nie ma czego uruchomić)"
  exit 1
fi

container_id="$(docker run -d \
  "${DOCKER_HOST_USER[@]}" \
  -v "$(pwd):/workspace" \
  -w /workspace \
  --env LD_LIBRARY_PATH="/workspace/$DIR_BUILD" \
  "$DOCKER_FULL_IMG_NAME" \
  bash -lc "exec ./$DIR_TARGET/*.exe")"

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
