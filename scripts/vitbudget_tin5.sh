#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venvs/momentstem/bin/python}
DR=${DATA_ROOT:-data}
CFG=configs/sslbudget/diagdeitsslbudget_simclr800_vit_tin_5pct.yaml
one() {
  local seed=$1
  local out="runs/simclr_pre800_vit_tin_5pct/seed${seed}"
  if [ ! -f "$out/pretrain.pt" ]; then
    mkdir -p "$out"
    $PY scripts/simclr_pretrain.py --data-root "$DR" --config "$CFG" \
        --epochs 800 --seed "$seed" --out "$out/pretrain.pt" \
        >> "logs/simclr_pre800_vit_tin_5pct_s${seed}.log" 2>&1
  fi
  $PY train.py --config "$CFG" --seed "$seed" --data-root "$DR" \
      >> "logs/vitbudget_tin_5pct_s${seed}.log" 2>&1
}
for s in 0 1 2; do one "$s" & done
wait
