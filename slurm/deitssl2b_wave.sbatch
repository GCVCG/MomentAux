#!/bin/bash
#SBATCH --job-name=deitssl2b
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# RESUBMIT of the deitssl2 stronger-view rebuttal (2026-07-22). deitssl2's
# original-cell G probes SUCCEEDED, but every DeiT-view pretrain FAILED with
# "flock: cannot open lock file .../pretrain.pt.lock: No such file or
# directory" -- flock cannot create the lockfile because the pretrain's parent
# dir did not exist yet (the deitssl wave reused fillvit's pre-made dirs; these
# paths are brand new). Fix: mkdir -p the parent before flock. Probes for the
# original cells are done, so this runs ONLY the sv pretrain+finetune+probe.
#
# PREDICTION (unchanged): stronger views recover SOME SSL value but do NOT
# overturn the flip -- deit-sslsv stays BELOW deit-aux at 10% and 25% (bands
# 30..40 @10%, 46..55 @25%). FALSIFIER: deit-sslsv >= deit-aux at BOTH.
set -uo pipefail
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
REPO=/mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
cd "$REPO"
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

for PCT in 10 25 5; do
  for SEED in 0 1 2; do
    PRE=runs/simclr_pre_vit_deit_${PCT}pct/seed${SEED}/pretrain.pt
    mkdir -p "$(dirname "$PRE")"          # <-- the fix
    echo "=== simclr-vit DEIT-VIEW pretrain ${PCT}pct seed ${SEED} ==="
    flock -w 21600 "${PRE}.lock" -c \
      "[ -f $PRE ] || $PY scripts/simclr_pretrain.py \
         --config configs/diagnostics/diagvit_none_${PCT}pct.yaml \
         --seed $SEED --augment deit --out $PRE" \
      || { echo "PRETRAIN FAIL ${PCT}/${SEED}"; continue; }
    echo "=== diagdeitsslsv_simclr_${PCT}pct seed ${SEED} ==="
    $PY train.py --config configs/diagnostics/diagdeitsslsv_simclr_${PCT}pct.yaml --seed $SEED \
      && echo "OK sv ${PCT} ${SEED}" || echo "FAIL sv ${PCT} ${SEED}"
  done
  echo "DEITSSL2B_PCT_DONE ${PCT}"
  $PY analysis/linear_probe.py --run runs/diagdeitsslsv_simclr_${PCT}pct \
      --config configs/diagnostics/diagdeitsslsv_simclr_${PCT}pct.yaml \
      && echo "PROBE OK sv ${PCT}" || echo "PROBE FAIL sv ${PCT}"
done
echo "DEITSSL2B_WAVE_COMPLETE"
