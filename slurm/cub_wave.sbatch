#!/bin/bash
#SBATCH --job-name=cub_wave
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# CUB-200-2011 transplant wave (2026-07-20, queued behind diagvit/diagcos;
# --gres=gpu:1 takes whichever GPU frees first). First genuinely FINE-
# GRAINED dataset: champion config verbatim at 64px (pipeline = tin's after
# squash-resize). 100% is already 30/cls; 25% is ~8/cls deep-left-flank.
# Predictions in CLAUDE.md BEFORE launch. Probes measure G; NOTE the
# probe-ceiling rule bites at 100% (probe labels == cell labels, so the
# readout split is not interpretable there; G's aux-vs-base gap still is).

set -euo pipefail
PY=${CLUSTER_SCRATCH}/envs/momentstem/bin/python
cd ${CLUSTER_SCRATCH}/projects/MomentsCNNEncoder
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

for CELL in cub_none_25pct cub_aux_25pct cub_none_100pct cub_aux_100pct; do
  for SEED in 0 1 2; do
    echo "=== ${CELL} seed ${SEED} ==="
    $PY train.py --config configs/${CELL}.yaml --seed $SEED
  done
  echo "CUB_CELL_DONE ${CELL}"
done

for CELL in cub_none_25pct cub_aux_25pct cub_none_100pct cub_aux_100pct; do
  $PY analysis/linear_probe.py --run runs/${CELL} --config configs/${CELL}.yaml
  echo "CUB_PROBE_DONE ${CELL}"
done

echo "CUB_WAVE_COMPLETE"
