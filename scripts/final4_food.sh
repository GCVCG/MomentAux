#!/bin/bash
set -u
cd "$(dirname "$0")/.."
PY=$HOME/venvs/momentstem/bin/python
# wait for one of the running cells to exit before claiming GPU memory
while [ "$(pgrep -cf '[t]rain.py --config')" -ge 2 ]; do sleep 120; done
$PY train.py --config configs/diagnostics/diagdepth_r50_food_aux_25pct.yaml --seed 0 \
  --data-root ./data --out-root runs > logs/final4/diagdepth_r50_food_aux_25pct_seed0.log 2>&1 \
  && echo "OK food seed0" >> logs/final4/wave.log || echo "FAIL food seed0" >> logs/final4/wave.log
