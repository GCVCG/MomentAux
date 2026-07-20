#!/bin/bash
#SBATCH --job-name=tinsem_wave
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# tinsem wave (2026-07-20): the semantic-vs-arbitrary coarse-partition
# adjudication (Q6.9j caveat). tinsem = tinsuper with ONLY the block sort
# key changed (WordNet hypernym-path order vs lexicographic wnid), byte-
# identical pixels. 10 seeds; tinsuper deepened 3->10 for a power-matched
# reference. Probes in BOTH spaces (own 20-way + common 200-way stick).
# Predictions in CLAUDE.md BEFORE launch.

set -euo pipefail
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
cd /mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

for CELL in tinsem_none_1pct tinsem_aux_1pct; do
  for SEED in 0 1 2 3 4 5 6 7 8 9; do
    echo "=== ${CELL} seed ${SEED} ==="
    $PY train.py --config configs/diagnostics/${CELL}.yaml --seed $SEED
  done
  echo "TINSEM_CELL_DONE ${CELL}"
done

for CELL in tinsuper_none_1pct tinsuper_aux_1pct; do
  for SEED in 3 4 5 6 7 8 9; do
    echo "=== ${CELL} seed ${SEED} (deepening) ==="
    $PY train.py --config configs/diagnostics/${CELL}.yaml --seed $SEED
  done
  echo "TINSEM_CELL_DONE ${CELL}-deepened"
done

for CELL in tinsem_none_1pct tinsem_aux_1pct tinsuper_none_1pct tinsuper_aux_1pct; do
  $PY analysis/linear_probe.py --run runs/${CELL} \
      --config configs/diagnostics/${CELL}.yaml
  echo "TINSEM_PROBE_DONE ${CELL} own-space"
  $PY analysis/linear_probe.py --run runs/${CELL} \
      --config configs/diagnostics/${CELL}.yaml --probe-dataset tin
  echo "TINSEM_PROBE_DONE ${CELL} tin-200way"
done

echo "TINSEM_WAVE_COMPLETE"
