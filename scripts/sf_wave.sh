#!/usr/bin/env bash
# Multispectral EuroSAT sensor-fusion wave, run on solarflare.
# 18 cells = {rgb,nir,all} x {none,aux} x {5,10,25}%, 3 seeds each = 54 runs.
# SLOTS concurrent runs: these are dataloader-bound at 64px, so several fit
# alongside each other on one GPU with 16 cores idle.
set -u
cd ${WORKSTATION_ROOT}/momentstem/repo
PY=${WORKSTATION_ROOT}/momentstem/venv/bin/python
SLOTS=${SLOTS:-3}
LOG=${WORKSTATION_ROOT}/momentstem/sf_wave.log
: > "$LOG"

TASKS=()
for band in rgb nir all; do
  for arm in none aux; do
    for pct in 5 10; do   # 25% runs on the local half (sf_wave_local.sh)
      for seed in 0 1 2; do
        TASKS+=("configs/sensorfusion/sf_eurosatms_${band}_${arm}_${pct}pct.yaml $seed")
      done
    done
  done
done
echo "$(date -Is) sf_wave start: ${#TASKS[@]} runs, SLOTS=$SLOTS" >> "$LOG"

run_one() {
  local cfg=$1 seed=$2
  local name; name=$(basename "$cfg" .yaml)
  $PY train.py --config "$cfg" --seed "$seed" --data-root data_sf \
      >> "${WORKSTATION_ROOT}/momentstem/logs_sf/${name}_s${seed}.log" 2>&1
  echo "$(date -Is) done $name seed$seed rc=$?" >> "$LOG"
}

mkdir -p ${WORKSTATION_ROOT}/momentstem/logs_sf
for t in "${TASKS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 5; done
  run_one $t &
done
wait
echo "$(date -Is) SF_WAVE_COMPLETE" >> "$LOG"
