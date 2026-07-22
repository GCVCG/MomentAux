#!/bin/bash
#SBATCH --job-name=fillconv
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
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
REPO=/mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
cd "$REPO"
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true
declare -A BASE=( [3]=configs/diagnostics/abl3_none.yaml [7]=configs/diagnostics/abl7_none.yaml [15]=configs/diagnostics/abl15_none.yaml [25]=configs/diagnostics/abl25_none.yaml [100]=configs/ablations_full/ablf_none.yaml )
for PCT in 3 7 15 25 100; do
  for SEED in 0 1 2; do
    echo "=== simclr pretrain ${PCT}pct seed ${SEED} ==="
    $PY scripts/simclr_pretrain.py --config "${BASE[$PCT]}" --seed $SEED --out runs/simclr_pre_${PCT}pct/seed${SEED}/pretrain.pt
    $PY train.py --config configs/diagnostics/diagssl_simclr_${PCT}pct.yaml --seed $SEED
  done
  echo "FILLCONV_SSL_DONE ${PCT}"
done
for PCT in 25 100; do
  for KIND in none aux; do
    for SEED in 0 1 2; do
      echo "=== super${PCT}_${KIND} seed ${SEED} ==="
      $PY train.py --config configs/diagnostics/super${PCT}_${KIND}.yaml --seed $SEED
    done
  done
  echo "FILLCONV_SUPER_DONE ${PCT}"
done
echo "FILLCONV_WAVE_COMPLETE"
