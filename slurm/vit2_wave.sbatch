#!/bin/bash
#SBATCH --job-name=vit2_wave
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# ViT wave 2 (2026-07-21): two open ViT questions.
# (1) diagvit@100% C100 -- does G(vit) finally FALL at full data (50k imgs)?
#     The curve is 5.9/13.2/14.9/14.0 flat through 25%; 100% is the right
#     flank test. GPU-bound (78k steps) -> uses the GPU well.
# (2) diagvit on tin @10% (64px, patch 8 -> 8x8 grid) -- does the huge ViT
#     rescue TRANSFER to a different input scale + 200-way? tin STAGED to
#     /dev/shm (RAM) so the 64px cells are not BeeGFS-IO-bound.
# AdamW => diag-only, never headline. 3 seeds + probes.

set -euo pipefail
PY=${CLUSTER_SCRATCH}/envs/momentstem/bin/python
REPO=${CLUSTER_SCRATCH}/projects/MomentsCNNEncoder
cd "$REPO"
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# C100@100% first (no staging needed)
for KIND in none aux; do
  for SEED in 0 1 2; do
    echo "=== diagvit_${KIND}_100pct seed ${SEED} ==="
    $PY train.py --config configs/diagnostics/diagvit_${KIND}_100pct.yaml --seed $SEED
  done
  echo "VIT2_CELL_DONE diagvit_${KIND}_100pct"
done
for KIND in none aux; do
  $PY analysis/linear_probe.py --run runs/diagvit_${KIND}_100pct \
      --config configs/diagnostics/diagvit_${KIND}_100pct.yaml
  echo "VIT2_PROBE_DONE diagvit_${KIND}_100pct"
done

# tin ViT: stage tin -> RAM
STAGE=/dev/shm/momentstem_${SLURM_JOB_ID}
mkdir -p "$STAGE"; trap 'rm -rf "$STAGE"' EXIT
echo "staging tin -> $STAGE"; time unzip -q "$REPO/data/tiny-imagenet-200.zip" -d "$STAGE"
DR="$STAGE"
for KIND in none aux; do
  for SEED in 0 1 2; do
    echo "=== diagvit_${KIND}_tin_10pct seed ${SEED} ==="
    $PY train.py --config configs/diagnostics/diagvit_${KIND}_tin_10pct.yaml --seed $SEED --data-root "$DR"
  done
  echo "VIT2_CELL_DONE diagvit_${KIND}_tin_10pct"
done
for KIND in none aux; do
  $PY analysis/linear_probe.py --run runs/diagvit_${KIND}_tin_10pct \
      --config configs/diagnostics/diagvit_${KIND}_tin_10pct.yaml --data-root "$DR"
  echo "VIT2_PROBE_DONE diagvit_${KIND}_tin_10pct"
done
echo "VIT2_WAVE_COMPLETE"
