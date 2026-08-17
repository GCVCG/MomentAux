#!/bin/bash
#SBATCH --job-name=diagvit_wave
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# diagvit envelope wave (2026-07-19, turing H100): ViT-tiny C100 pairs at
# 1/5/25% (10% already measured locally: +13.26). 3 seeds/cell, cheap ->
# expensive, then full-label linear probes for G. Diag-only (AdamW), never
# a headline row. Predictions recorded in CLAUDE.md BEFORE launch.

set -euo pipefail
PY=${CLUSTER_SCRATCH}/envs/momentstem/bin/python
cd ${CLUSTER_SCRATCH}/projects/MomentsCNNEncoder
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

for PCT in 1 5 25; do
  for KIND in none aux; do
    for SEED in 0 1 2; do
      echo "=== diagvit_${KIND}_${PCT}pct seed ${SEED} ==="
      $PY train.py --config configs/diagnostics/diagvit_${KIND}_${PCT}pct.yaml --seed $SEED
    done
  done
  echo "DIAGVIT_${PCT}PCT_PAIR_DONE"
done

for PCT in 1 5 25; do
  for KIND in none aux; do
    $PY analysis/linear_probe.py --run runs/diagvit_${KIND}_${PCT}pct \
        --config configs/diagnostics/diagvit_${KIND}_${PCT}pct.yaml
    echo "DIAGVIT_PROBE_DONE diagvit_${KIND}_${PCT}pct"
  done
done

echo "DIAGVIT_WAVE_COMPLETE"
