#!/bin/bash
#SBATCH --job-name=fillvit
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# Envelope COMPLETION wave (2026-07-22, user: "better to have the full
# experiments to have a clear picture"). Fills the missing fractions so the
# live configurations share the study's full grid (1,2,3,5,7,10,15,25,100%)
# instead of the sparse 4-5 points they were launched with.
set -euo pipefail
PY=${CLUSTER_SCRATCH}/envs/momentstem/bin/python
REPO=${CLUSTER_SCRATCH}/projects/MomentsCNNEncoder
cd "$REPO"
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true
for PCT in 2 3 7 15; do
  for KIND in none aux; do
    for SEED in 0 1 2; do
      for FAM in diagvit diagdeit; do
        echo "=== ${FAM}_${KIND}_${PCT}pct seed ${SEED} ==="
        $PY train.py --config configs/diagnostics/${FAM}_${KIND}_${PCT}pct.yaml --seed $SEED
      done
    done
  done
  echo "FILLVIT_PCT_DONE ${PCT}"
done
for PCT in 2 3 7 15 100; do
  for SEED in 0 1 2; do
    echo "=== simclr-vit pretrain ${PCT}pct seed ${SEED} ==="
    $PY scripts/simclr_pretrain.py --config configs/diagnostics/diagvit_none_${PCT}pct.yaml --seed $SEED --out runs/simclr_pre_vit_${PCT}pct/seed${SEED}/pretrain.pt
    $PY train.py --config configs/diagnostics/diagsslvit_simclr_${PCT}pct.yaml --seed $SEED
  done
  echo "FILLVIT_SSL_DONE ${PCT}"
done
echo "FILLVIT_WAVE_COMPLETE"
