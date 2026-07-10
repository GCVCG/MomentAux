#!/bin/bash
#SBATCH --job-name=momentstem
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# One job = one config x several seeds run sequentially (CIFAR-scale runs are
# small; packing seeds keeps us inside the 2-running-job QOS).
#
#   sbatch slurm/train.sbatch configs/<cell>.yaml 0 1 2
#   sbatch --gres=gpu:h200:1 slurm/train.sbatch configs/<cell>.yaml 0 1 2

set -euo pipefail
CONFIG="$1"; shift
echo "host=$(hostname) config=${CONFIG} seeds=$*"
nvidia-smi --query-gpu=name --format=csv,noheader || true

for SEED in "$@"; do
    echo "=== ${CONFIG} seed ${SEED} ==="
    python train.py --config "${CONFIG}" --seed "${SEED}"
done
