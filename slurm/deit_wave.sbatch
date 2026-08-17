#!/bin/bash
#SBATCH --job-name=deit_wave
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# DeiT heavy-augmentation wave (2026-07-21): the real rival for the ViT
# claim. diagdeit_* == diagvit_* + `augment: deit` (DeiT's published
# augmentation hyper-parameters verbatim); ONE variable changed. Full C100
# envelope 1/5/10/25/100%, both members, 3 seeds + probes.
# Predictions + both falsifiers recorded in CLAUDE.md BEFORE launch.
# C100 = in-memory, no staging. AdamW + augment => diag-only.

set -euo pipefail
PY=${CLUSTER_SCRATCH}/envs/momentstem/bin/python
cd ${CLUSTER_SCRATCH}/projects/MomentsCNNEncoder
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

for PCT in 1 5 10 25 100; do
  for KIND in none aux; do
    for SEED in 0 1 2; do
      echo "=== diagdeit_${KIND}_${PCT}pct seed ${SEED} ==="
      $PY train.py --config configs/diagnostics/diagdeit_${KIND}_${PCT}pct.yaml --seed $SEED
    done
    echo "DEIT_CELL_DONE diagdeit_${KIND}_${PCT}pct"
  done
done

for PCT in 1 5 10 25 100; do
  for KIND in none aux; do
    $PY analysis/linear_probe.py --run runs/diagdeit_${KIND}_${PCT}pct \
        --config configs/diagnostics/diagdeit_${KIND}_${PCT}pct.yaml
    echo "DEIT_PROBE_DONE diagdeit_${KIND}_${PCT}pct"
  done
done
echo "DEIT_WAVE_COMPLETE"
