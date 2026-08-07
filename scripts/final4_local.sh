#!/bin/bash
# The study's last 4 cells, run locally because BSC is priority-starved
# (2026-08-07). train.py's completed-run guard + run lock make this safe.
set -u
cd "$(dirname "$0")/.."
PY=$HOME/venvs/momentstem/bin/python
L=logs/final4
mkdir -p $L
run () { $PY train.py --config configs/diagnostics/$1.yaml --seed $2 \
           --data-root ./data --out-root runs > $L/$1_seed$2.log 2>&1 \
         && echo "OK $1 seed$2" >> $L/wave.log || echo "FAIL $1 seed$2" >> $L/wave.log; }
run diagdepth_r50_food_aux_25pct 0 &
run diagdepth_r50_path_aux_25pct 1 &
run diagin100_r50_aux 1 &
wait
echo "FINAL4_LOCAL_COMPLETE $(date -Is)" >> $L/wave.log
