#!/bin/bash
#SBATCH --job-name=filltin
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
STAGE=/dev/shm/momentstem_${SLURM_JOB_ID}
mkdir -p "$STAGE"; trap 'rm -rf "$STAGE"' EXIT
echo "staging tin -> $STAGE"; time unzip -q "$REPO/data/tiny-imagenet-200.zip" -d "$STAGE"
for PCT in 3 7 15; do
  for KIND in none aux; do
    for SEED in 0 1 2; do
      echo "=== tin_${KIND}_${PCT}pct seed ${SEED} ==="
      $PY train.py --config configs/diagnostics/tin_${KIND}_${PCT}pct.yaml --seed $SEED --data-root "$STAGE"
    done
  done
  echo "FILLTIN_PCT_DONE ${PCT}"
done
echo "FILLTIN_WAVE_COMPLETE"
