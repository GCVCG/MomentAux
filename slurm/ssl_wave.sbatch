#!/bin/bash
#SBATCH --job-name=ssl_wave
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# SimCLR external-positioning wave (2026-07-20, queued; --gres=gpu:1 takes
# whichever GPU frees first). Per seed: 200-epoch SimCLR pretrain on the
# committed C100 5% subset IMAGES (no labels, no outside data), then the
# frozen supervised recipe from that init (diagssl_simclr_5pct, diag-only).
# diagssl_none400_5pct = the 2x-compute plain-supervised control (the step-
# count confound lesson). Comparators on record: baseline 25.23 (1x),
# champion aux 30.53 (~1.02x). Predictions in CLAUDE.md BEFORE launch.

set -euo pipefail
PY=${CLUSTER_SCRATCH}/envs/momentstem/bin/python
cd ${CLUSTER_SCRATCH}/projects/MomentsCNNEncoder
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

for SEED in 0 1 2; do
  echo "=== simclr pretrain seed ${SEED} ==="
  $PY scripts/simclr_pretrain.py --config configs/ablations_full/abl5_none.yaml \
      --seed $SEED --out runs/simclr_pre_5pct/seed${SEED}/pretrain.pt
  echo "=== diagssl_simclr_5pct seed ${SEED} ==="
  $PY train.py --config configs/diagnostics/diagssl_simclr_5pct.yaml --seed $SEED
done
echo "SSL_CELL_DONE diagssl_simclr_5pct"

for SEED in 0 1 2; do
  echo "=== diagssl_none400_5pct seed ${SEED} ==="
  $PY train.py --config configs/diagnostics/diagssl_none400_5pct.yaml --seed $SEED
done
echo "SSL_CELL_DONE diagssl_none400_5pct"

echo "SSL_WAVE_COMPLETE"
