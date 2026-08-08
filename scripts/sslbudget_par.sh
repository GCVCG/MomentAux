#!/usr/bin/env bash
# Run the three seeds of one SSL-budget cell CONCURRENTLY.
# The sequential wave left the GPU at ~20% because one 800-epoch SimSiam
# pretrain on 2-5k images is dataloader-bound, not compute-bound.
#
# SAFETY: this replaces the sequential wave rather than running alongside it.
# Two trainers on the same (config, seed) is exactly the failure that caused
# the wrong-epoch checkpoint damage recorded 2026-08-07, so the caller MUST
# stop the wave first. train.py's own flock guard is the backstop.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venvs/momentstem/bin/python}
DR=${DATA_ROOT:-data}
PCT=${PCT:-5}
LOG=logs/sslbudget_par_${PCT}.log
: > "$LOG"
one() {
  local seed=$1
  local out="runs/simsiam_pre800_c100_${PCT}pct/seed${seed}"
  if [ ! -f "$out/pretrain.pt" ]; then
    mkdir -p "$out"
    $PY scripts/simsiam_pretrain.py --data-root "$DR" \
        --config "configs/sslbudget/diagsslbudget_simsiam800_c100_${PCT}pct.yaml" \
        --epochs 800 --seed "$seed" --out "$out/pretrain.pt" \
        >> "logs/simsiam_pre800_c100_${PCT}pct_s${seed}.log" 2>&1
    echo "$(date -Is) pretrain seed$seed rc=$?" >> "$LOG"
  fi
  $PY train.py --config "configs/sslbudget/diagsslbudget_simsiam800_c100_${PCT}pct.yaml" \
      --seed "$seed" --data-root "$DR" \
      >> "logs/sslbudget_${PCT}pct_s${seed}.log" 2>&1
  echo "$(date -Is) supervised seed$seed rc=$?" >> "$LOG"
}
for s in 0 1 2; do one "$s" & done
wait
echo "$(date -Is) SSLBUDGET_PAR_${PCT}_COMPLETE" >> "$LOG"
