#!/bin/bash
#SBATCH --job-name=sslvit_wave
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# SimCLR-on-ViT wave (2026-07-21): the decisive positioning experiment.
# SSL beats the aux prior at every conv fraction; ViT-tiny is the one regime
# where the prior is dramatic (+13.26@10%, G=14.85). Per seed: 200ep SimCLR
# pretrain on the cell's own subset images (AdamW, matching the diagvit
# recipe) -> frozen diagvit recipe from that init. Predictions + both
# falsifiers recorded in CLAUDE.md BEFORE launch. C100 = in-memory, no
# staging. AdamW => diag-only, never headline.

set -euo pipefail
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
cd /mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

for PCT in 10 5; do
  for SEED in 0 1 2; do
    echo "=== simclr-vit pretrain ${PCT}pct seed ${SEED} ==="
    $PY scripts/simclr_pretrain.py --config configs/diagnostics/diagvit_none_${PCT}pct.yaml \
        --seed $SEED --out runs/simclr_pre_vit_${PCT}pct/seed${SEED}/pretrain.pt
    echo "=== diagsslvit_simclr_${PCT}pct seed ${SEED} ==="
    $PY train.py --config configs/diagnostics/diagsslvit_simclr_${PCT}pct.yaml --seed $SEED
  done
  echo "SSLVIT_CELL_DONE diagsslvit_simclr_${PCT}pct"
  $PY analysis/linear_probe.py --run runs/diagsslvit_simclr_${PCT}pct \
      --config configs/diagnostics/diagsslvit_simclr_${PCT}pct.yaml
  echo "SSLVIT_PROBE_DONE diagsslvit_simclr_${PCT}pct"
done
echo "SSLVIT_WAVE_COMPLETE"
