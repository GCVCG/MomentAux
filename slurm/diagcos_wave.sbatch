#!/bin/bash
#SBATCH --job-name=diagcos_wave
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# diagcos wave (2026-07-19, turing H200 -- user authorized both GPUs):
# cosine-classifier head at the left flank. Parent cells verbatim + head:
# cosine, both members of each pair get the head (Delta is cosine-vs-cosine).
# Tests head-EXPRESSIVITY vs label-INFORMATION as the source of the readout
# penalty. 3 seeds/cell, then full-label probes (feature-side control: G
# should be unchanged by the head). Predictions in CLAUDE.md BEFORE launch.

set -euo pipefail
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
cd /mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

for CELL in diagcos_c100_none_1pct diagcos_c100_aux_1pct \
            diagcos_tin_none_1pct diagcos_tin_aux_1pct; do
  for SEED in 0 1 2; do
    echo "=== ${CELL} seed ${SEED} ==="
    $PY train.py --config configs/diagnostics/${CELL}.yaml --seed $SEED
  done
  echo "DIAGCOS_CELL_DONE ${CELL}"
done

for CELL in diagcos_c100_none_1pct diagcos_c100_aux_1pct \
            diagcos_tin_none_1pct diagcos_tin_aux_1pct; do
  $PY analysis/linear_probe.py --run runs/${CELL} \
      --config configs/diagnostics/${CELL}.yaml
  echo "DIAGCOS_PROBE_DONE ${CELL}"
done

echo "DIAGCOS_WAVE_COMPLETE"
