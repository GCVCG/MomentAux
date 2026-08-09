#!/usr/bin/env bash
# One ViT budget cell: three seeds concurrently, 800-epoch SimCLR pretrain
# then the DeiT-augmented supervised stage from that init.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venvs/momentstem/bin/python}
DR=${DATA_ROOT:-data}
PCT=${PCT:?set PCT}
CFG=configs/sslbudget/diagdeitsslbudget_simclr800_vit_${PCT}pct.yaml
one() {
  local seed=$1
  local out="runs/simclr_pre800_vit_${PCT}pct/seed${seed}"
  if [ ! -f "$out/pretrain.pt" ]; then
    mkdir -p "$out"
    $PY scripts/simclr_pretrain.py --data-root "$DR" --config "$CFG" \
        --epochs 800 --seed "$seed" --out "$out/pretrain.pt" \
        >> "logs/simclr_pre800_vit_${PCT}pct_s${seed}.log" 2>&1
  fi
  $PY train.py --config "$CFG" --seed "$seed" --data-root "$DR" \
      >> "logs/vitbudget_${PCT}pct_s${seed}.log" 2>&1
}
for s in 0 1 2; do one "$s" & done
wait
