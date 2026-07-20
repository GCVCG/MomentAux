#!/bin/bash
#SBATCH --job-name=ssl2_wave
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# ssl2 wave (2026-07-20): follow-ups the fired SSL falsifier demands.
# (a) diagssl_simclraux_5pct -- SimCLR init + champion aux (2.02x): is the
#     moment prior REDUNDANT with SimCLR features?
# (b) diagssl_simclr50_5pct -- 50ep pretrain + frozen recipe (1.25x): where
#     along the compute axis does SSL's win turn on?
# (c) probes for ALL diagssl cells vs abl5 ckpts: is SSL's +9.2 a G effect
#     or a readout effect under the law?
# Predictions in CLAUDE.md BEFORE launch.

set -euo pipefail
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
cd /mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

for SEED in 0 1 2; do
  echo "=== simclr pretrain50 seed ${SEED} ==="
  $PY scripts/simclr_pretrain.py --config configs/ablations_full/abl5_none.yaml \
      --seed $SEED --epochs 50 \
      --out runs/simclr_pre50_5pct/seed${SEED}/pretrain.pt
  echo "=== diagssl_simclr50_5pct seed ${SEED} ==="
  $PY train.py --config configs/diagnostics/diagssl_simclr50_5pct.yaml --seed $SEED
  echo "=== diagssl_simclraux_5pct seed ${SEED} ==="
  $PY train.py --config configs/diagnostics/diagssl_simclraux_5pct.yaml --seed $SEED
done
echo "SSL2_CELLS_DONE"

for CELL in diagssl_simclr_5pct diagssl_none400_5pct \
            diagssl_simclr50_5pct diagssl_simclraux_5pct; do
  $PY analysis/linear_probe.py --run runs/${CELL} \
      --config configs/diagnostics/${CELL}.yaml
  echo "SSL2_PROBE_DONE ${CELL}"
done

echo "SSL2_WAVE_COMPLETE"
