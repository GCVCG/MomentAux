#!/usr/bin/env bash
# Multispectral EuroSAT sensor-fusion wave, run on solarflare.
# 18 cells = {rgb,nir,all} x {none,aux} x {5,10,25}%, 3 seeds each = 54 runs.
# SLOTS concurrent runs: these are dataloader-bound at 64px, so several fit
# alongside each other on one GPU with 16 cores idle.
set -u
cd /media/HDD_4TB/amughrabi/momentstem/repo
PY=/media/HDD_4TB/amughrabi/momentstem/venv/bin/python
SLOTS=${SLOTS:-3}
LOG=/media/HDD_4TB/amughrabi/momentstem/sf_wave.log
: > "$LOG"

TASKS=()
for band in rgb nir all; do
  for arm in none aux; do
    for pct in 5 10 25; do
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
      >> "/media/HDD_4TB/amughrabi/momentstem/logs_sf/${name}_s${seed}.log" 2>&1
  echo "$(date -Is) done $name seed$seed rc=$?" >> "$LOG"
}

mkdir -p /media/HDD_4TB/amughrabi/momentstem/logs_sf
for t in "${TASKS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 5; done
  run_one $t &
done
wait
echo "$(date -Is) SF_WAVE_COMPLETE" >> "$LOG"
