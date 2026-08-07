#!/usr/bin/env bash
# Experiment C: SimSiam at 4x the cost-normalised pre-training budget.
# 2 cells x 3 seeds = 6 pretrains (800 ep) + 6 supervised runs.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venvs/momentstem/bin/python}
LOG=logs/sslbudget_wave.log
: > "$LOG"
echo "$(date -Is) sslbudget start" >> "$LOG"
for pct in 5 10; do
  for seed in 0 1 2; do
    out="runs/simsiam_pre800_c100_${pct}pct/seed${seed}"
    if [ ! -f "$out/pretrain.pt" ]; then
      mkdir -p "$out"
      $PY scripts/simsiam_pretrain.py \
          --config "configs/sslbudget/sslbudget_simsiam800_c100_${pct}pct.yaml" \
          --epochs 800 --seed "$seed" --out "$out/pretrain.pt" \
          >> "logs/simsiam_pre800_c100_${pct}pct_s${seed}.log" 2>&1
      echo "$(date -Is) pretrain ${pct}pct seed$seed rc=$?" >> "$LOG"
    fi
    $PY train.py --config "configs/sslbudget/sslbudget_simsiam800_c100_${pct}pct.yaml" \
        --seed "$seed" >> "logs/sslbudget_${pct}pct_s${seed}.log" 2>&1
    echo "$(date -Is) supervised ${pct}pct seed$seed rc=$?" >> "$LOG"
  done
done
echo "$(date -Is) SSLBUDGET_COMPLETE" >> "$LOG"
