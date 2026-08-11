#!/usr/bin/env bash
# Local half of the sensor-fusion wave: the 25% cells, which are the longest
# and would otherwise be solarflare's tail. Waits for the two remaining study
# cells (diagin100_r50_aux, diagdepth_*) to release the GPU before starting,
# so it never competes with the last open falsifier.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venvs/momentstem/bin/python}
SLOTS=${SLOTS:-3}
NEED_MB=${NEED_MB:-9000}
LOG=logs/sf_wave_local.log
: > "$LOG"

echo "$(date -Is) waiting for GPU (need ${NEED_MB} MiB free)" >> "$LOG"
while :; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${free:-0}" -ge "$NEED_MB" ] && break
  sleep 120
done
echo "$(date -Is) GPU free (${free} MiB), starting" >> "$LOG"

mkdir -p logs/sf_local
TASKS=()
for band in rgb nir all; do
  for arm in none aux; do
    for seed in 0 1 2; do
      TASKS+=("configs/sensorfusion/sf_eurosatms_${band}_${arm}_25pct.yaml $seed")
    done
  done
done
echo "$(date -Is) local half: ${#TASKS[@]} runs, SLOTS=$SLOTS" >> "$LOG"

run_one() {
  local cfg=$1 seed=$2 name
  name=$(basename "$cfg" .yaml)
  $PY train.py --config "$cfg" --seed "$seed" \
      >> "logs/sf_local/${name}_s${seed}.log" 2>&1
  echo "$(date -Is) done $name seed$seed rc=$?" >> "$LOG"
}
for t in "${TASKS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 10; done
  run_one $t &
done
wait
echo "$(date -Is) SF_LOCAL_COMPLETE" >> "$LOG"
