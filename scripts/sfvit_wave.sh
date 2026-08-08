#!/usr/bin/env bash
# ViT-tiny on the multispectral sensor grid: 12 cells x 3 seeds = 36 runs.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venvs/momentstem/bin/python}
DR=${DATA_ROOT:-data}
SLOTS=${SLOTS:-4}
PCTS=${PCTS:-"5 10"}
LOG=logs/sfvit_wave.log
: > "$LOG"; mkdir -p logs/sfvit
for band in rgb nir all; do for arm in none aux; do for pct in $PCTS; do for seed in 0 1 2; do
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 5; done
  ( n=diagsfvit_eurosatms_${band}_${arm}_${pct}pct
    $PY train.py --config configs/sensorfusion/$n.yaml --seed $seed --data-root "$DR" \
      >> logs/sfvit/${n}_s${seed}.log 2>&1
    echo "$(date -Is) done $n seed$seed rc=$?" >> "$LOG" ) &
done; done; done; done
wait
echo "$(date -Is) SFVIT_COMPLETE" >> "$LOG"
