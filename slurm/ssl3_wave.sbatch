#!/bin/bash
#SBATCH --job-name=ssl3_wave
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# SSL data-dimension wave (2026-07-21): SimCLR-init vs champion-aux vs
# baseline across the C100 envelope (1/2/10%; 5% done, +3.9 SSL win).
# Per fraction: 200ep SimCLR pretrain on that fraction's committed subset
# images -> frozen recipe from that init. 3 seeds + probes. C100 in-memory.
# PREDICTION (CLAUDE.md): SimCLR gain SHRINKS as data falls while aux holds.

set -euo pipefail
PY=${CLUSTER_SCRATCH}/envs/momentstem/bin/python
cd ${CLUSTER_SCRATCH}/projects/MomentsCNNEncoder
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

declare -A BASECFG=(
  [1]=configs/ablations_full/abl1_none.yaml
  [2]=configs/diagnostics/abl2_none.yaml
  [10]=configs/ablations_full/abl10_none.yaml
)
for PCT in 1 2 10; do
  for SEED in 0 1 2; do
    echo "=== simclr pretrain ${PCT}pct seed ${SEED} ==="
    $PY scripts/simclr_pretrain.py --config "${BASECFG[$PCT]}" \
        --seed $SEED --out runs/simclr_pre_${PCT}pct/seed${SEED}/pretrain.pt
    echo "=== diagssl_simclr_${PCT}pct seed ${SEED} ==="
    $PY train.py --config configs/diagnostics/diagssl_simclr_${PCT}pct.yaml --seed $SEED
  done
  echo "SSL3_CELL_DONE diagssl_simclr_${PCT}pct"
  $PY analysis/linear_probe.py --run runs/diagssl_simclr_${PCT}pct \
      --config configs/diagnostics/diagssl_simclr_${PCT}pct.yaml
  echo "SSL3_PROBE_DONE diagssl_simclr_${PCT}pct"
done
echo "SSL3_WAVE_COMPLETE"
