#!/bin/bash
#SBATCH --job-name=deitssl2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# STRONGER-VIEW REBUTTAL (2026-07-22) -- fired by the deitssl decision rule.
# deitssl landed with deit-ssl BELOW deit-aux at every fraction (aux-ssl
# +2.5..+7.3), and my bands MISSED LOW at 5/10% -- augmentation substituted
# for SimCLR MORE completely than predicted. That makes "was SimCLR given its
# best shot?" the sharpest objection: the deitssl pretrain used SimCLR's OWN
# 2020 augmentations while the supervised stage used DeiT augs. This wave gives
# the pretrain DeiT-strength views too (--augment deit: RandAugment +
# RandomErasing; Mixup/CutMix/label-smoothing are inapplicable to NT-Xent).
#
# NEW 2x-compute pretraining (unlike deitssl, which reused fillvit's pretrains).
# Scope: decisive fractions 5/10/25 only, 3 seeds = 9 pretrains + 9 finetunes.
#
# PREDICTION RECORDED IN ADVANCE: stronger views RECOVER SOME of SSL's value
# but do NOT overturn the flip -- deit-sslsv stays BELOW deit-aux at 10% and
# 25% (bands 30..40 @10%, 46..55 @25%). SimCLR's own ablation says crop+color
# is the critical pair with diminishing returns beyond it, so at 2500-12500
# images harder views help little and can hurt.
#   FALSIFIER (SSL was under-augmented all along): deit-sslsv >= deit-aux at
#     BOTH 10% and 25% => the flip was an augmentation-asymmetry artifact, and
#     the "prior wins under a modern recipe" claim must be withdrawn.
#
# Also RE-RUNS the deitssl G probes that failed on a CLI mismatch (linear_probe
# takes --run/--config, one cell at a time -- the wave passed --runs/--out).
set -uo pipefail
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
REPO=/mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
cd "$REPO"
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# --- corrected G probes for the ORIGINAL deitssl cells (aux-vs-base gap) ---
for PCT in 5 10 25; do
  for CELL in diagdeitssl_simclr diagdeit_none; do
    $PY analysis/linear_probe.py --run runs/${CELL}_${PCT}pct \
        --config configs/diagnostics/${CELL}_${PCT}pct.yaml \
        > runs/probe_${CELL}_${PCT}pct.json 2>/dev/null \
        && echo "PROBE OK ${CELL}_${PCT}" || echo "PROBE FAIL ${CELL}_${PCT}"
  done
done
echo "DEITSSL2_PROBES_DONE"

# --- stronger-view pretrain + finetune, decisive fractions ---
for PCT in 10 25 5; do
  for SEED in 0 1 2; do
    PRE=runs/simclr_pre_vit_deit_${PCT}pct/seed${SEED}/pretrain.pt
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
  echo "DEITSSL2_PCT_DONE ${PCT}"
  # G probe for the stronger-view cell
  $PY analysis/linear_probe.py --run runs/diagdeitsslsv_simclr_${PCT}pct \
      --config configs/diagnostics/diagdeitsslsv_simclr_${PCT}pct.yaml \
      > runs/probe_diagdeitsslsv_${PCT}pct.json 2>/dev/null \
      && echo "PROBE OK sv ${PCT}" || echo "PROBE FAIL sv ${PCT}"
done
echo "DEITSSL2_WAVE_COMPLETE"
