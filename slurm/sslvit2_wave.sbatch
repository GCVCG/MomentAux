#!/bin/bash
#SBATCH --job-name=sslvit2_wave
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# sslvit2 (2026-07-21): completes the ViT-SSL envelope (1%, 25%) and tests
# the ViT COMBO (SimCLR init + moment aux). sslvit showed SSL merely TIES
# the prior at ViT@10% and LOSES at 5% -- the opposite of conv. Does SSL
# overtake above 10%, and do the two priors stack on attention?
# Predictions + falsifier recorded in CLAUDE.md BEFORE launch.

set -euo pipefail
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
cd /mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

for PCT in 1 25; do
  for SEED in 0 1 2; do
    echo "=== simclr-vit pretrain ${PCT}pct seed ${SEED} ==="
    $PY scripts/simclr_pretrain.py --config configs/diagnostics/diagvit_none_${PCT}pct.yaml \
        --seed $SEED --out runs/simclr_pre_vit_${PCT}pct/seed${SEED}/pretrain.pt
    echo "=== diagsslvit_simclr_${PCT}pct seed ${SEED} ==="
    $PY train.py --config configs/diagnostics/diagsslvit_simclr_${PCT}pct.yaml --seed $SEED
  done
  echo "SSLVIT2_CELL_DONE diagsslvit_simclr_${PCT}pct"
  $PY analysis/linear_probe.py --run runs/diagsslvit_simclr_${PCT}pct \
      --config configs/diagnostics/diagsslvit_simclr_${PCT}pct.yaml
  echo "SSLVIT2_PROBE_DONE diagsslvit_simclr_${PCT}pct"
done

# combo reuses the 10% ViT pretrains already on disk from the sslvit wave
for SEED in 0 1 2; do
  echo "=== diagsslvitaux_10pct seed ${SEED} ==="
  $PY train.py --config configs/diagnostics/diagsslvitaux_10pct.yaml --seed $SEED
done
echo "SSLVIT2_CELL_DONE diagsslvitaux_10pct"
$PY analysis/linear_probe.py --run runs/diagsslvitaux_10pct \
    --config configs/diagnostics/diagsslvitaux_10pct.yaml
echo "SSLVIT2_PROBE_DONE diagsslvitaux_10pct"
echo "SSLVIT2_WAVE_COMPLETE"
