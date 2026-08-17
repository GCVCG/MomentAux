#!/bin/bash
#SBATCH --job-name=deepen_wave
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# Headline-deepening wave (2026-07-21): the C100 champion envelope cells at
# 1/5/10% are still 3-seed; C10 showed 3-seed deltas regress under power
# (+7.14 -> +6.66). Deepen the 1/5/10% pairs 3 -> 10 seeds so no headline
# number rests on a 3-seed sigma. C100 = single in-memory pickle -> ~0.6s/
# epoch, GPU-bound, no staging needed. Seeds 3..9 (0..2 already on disk).

set -euo pipefail
PY=${CLUSTER_SCRATCH}/envs/momentstem/bin/python
cd ${CLUSTER_SCRATCH}/projects/MomentsCNNEncoder
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

declare -A CFG=(
  [abl1_none]=configs/ablations_full/abl1_none.yaml
  [auxmag_1pct_sched0]=configs/diagnostics/auxmag_1pct_sched0.yaml
  [abl5_none]=configs/ablations_full/abl5_none.yaml
  [auxmag_5pct_sched0]=configs/diagnostics/auxmag_5pct_sched0.yaml
  [abl10_none]=configs/ablations_full/abl10_none.yaml
  [auxmag_10pct_sched0]=configs/diagnostics/auxmag_10pct_sched0.yaml
)
for CELL in abl1_none auxmag_1pct_sched0 abl5_none auxmag_5pct_sched0 abl10_none auxmag_10pct_sched0; do
  for SEED in 3 4 5 6 7 8 9; do
    echo "=== ${CELL} seed ${SEED} ==="
    $PY train.py --config "${CFG[$CELL]}" --seed $SEED
  done
  echo "DEEPEN_CELL_DONE ${CELL}"
done

# Re-probe the aux cells at full 10-seed power (baselines get probed for G).
for CELL in abl1_none auxmag_1pct_sched0 abl5_none auxmag_5pct_sched0 abl10_none auxmag_10pct_sched0; do
  $PY analysis/linear_probe.py --run runs/${CELL} --config "${CFG[$CELL]}"
  echo "DEEPEN_PROBE_DONE ${CELL}"
done

echo "DEEPEN_WAVE_COMPLETE"
