#!/usr/bin/env bash
# E3 (transfer tax under weaker/delayed prior) + B4 (fixed step budget).
# Predictions and falsifiers were recorded in CLAUDE.md BEFORE this launched.
# Study venv only: anaconda base carries timm 0.6.7 and cannot build the ViTs.
set -u
cd /home/amughrabi/projects/MomentsCNNEncoder
PY=~/venvs/momentstem/bin/python
LOG=logs/refwave_local.log
mkdir -p logs
echo "=== refwave start $(date -Is) ===" >> "$LOG"
ok=0; fail=0
for cfg in configs/diagnostics/diagtaxlam_*.yaml configs/diagnostics/diagstep_*.yaml; do
  for seed in 0 1 2; do
    name=$(basename "$cfg" .yaml)
    if [ -f "runs/$name/seed$seed/final.json" ]; then
      echo "skip  $name seed$seed (already final)" >> "$LOG"; continue
    fi
    echo "--- $name seed$seed $(date -Is)" >> "$LOG"
    if $PY train.py --config "$cfg" --seed "$seed" >> "$LOG" 2>&1; then
      ok=$((ok+1))
    else
      fail=$((fail+1)); echo "FAIL $name seed$seed" >> "$LOG"
    fi
  done
done
echo "REFWAVE_LOCAL_COMPLETE ok=$ok fail=$fail $(date -Is)" >> "$LOG"
