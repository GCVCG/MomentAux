#!/bin/bash
#SBATCH --job-name=vitenv
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# ViT/DeiT ENVELOPE ON TIN (2026-07-23, user: "don't worry about cost -- if
# the question is important, go"). The attention story rests on ONE tin
# fraction (10%): plain Delta +9.15, deit-aux Delta +16.9. This wave measures
# the full second-population attention envelope: does "permanent deficit"
# (C100: +9.88 at FULL data) and the modern-recipe flip hold across scale?
#
# PREDICTIONS RECORDED IN ADVANCE (C100-envelope-shaped, G unmeasured on tin
# beyond 10%):
#   diagvit pairs @1/2/5/15/25/100:
#     Delta_vit(tin) rises from suppressed left flank (+1..+4 @1-2%,
#     readout-negative at tiny baselines) to a peak +10..+15 @15-25%, and
#     stays LARGE at 100% (+6..+10) -- the permanent-deficit signature.
#     FALSIFIER: Delta_vit(tin@100%) <= +2 => permanent deficit is
#     C100-specific and the ViT headline must be scoped.
#   diagdeit pairs: amplification ~1.8-2.4x of the plain Delta at every
#     fraction (measured @10%: 1.85x). FALSIFIER: amplification < 1.3x
#     anywhere on the mid-band.
#   diagdeitssl @5/25 (+10% from genssl2): deit-ssl BELOW deit-aux at every
#     fraction (the flip transplants across the envelope).
# Ordering: cheap fractions first; the 100% pair LAST (longest, killable).
set -uo pipefail
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
REPO=/mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
cd "$REPO"
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

STAGE=/dev/shm/vitenv_${SLURM_JOB_ID}
mkdir -p "$STAGE"; trap 'rm -rf "$STAGE"' EXIT
ln -sfn "$REPO"/data/* "$STAGE"/ 2>/dev/null
rm -f "$STAGE/tiny-imagenet-200"
unzip -q "$REPO/data/tiny-imagenet-200.zip" -d "$STAGE"
DR=$STAGE

run3 () {                       # $1 = config basename
    for SEED in 0 1 2; do
        [ -f "runs/$1/seed${SEED}/final.json" ] && { echo "SKIP $1 ${SEED}"; continue; }
        echo "=== $1 seed ${SEED} ==="
        $PY train.py --config configs/diagnostics/$1.yaml --seed $SEED --data-root "$DR" \
          && echo "OK $1 ${SEED}" || echo "FAIL $1 ${SEED}"
    done
}

for PCT in 1 2 5 15 25; do
    for CFG in diagvit_none_tin_${PCT}pct diagvit_aux_tin_${PCT}pct \
               diagdeit_none_tin_${PCT}pct diagdeit_aux_tin_${PCT}pct; do
        run3 "$CFG"
    done
    echo "VITENV_PCT_DONE ${PCT}"
done

# deit-ssl at 5/25 (pretrains per fraction, augmentation-agnostic ckpts)
for PCT in 5 25; do
    for SEED in 0 1 2; do
        PRE=runs/simclr_pre_vit_tin_${PCT}pct/seed${SEED}/pretrain.pt
        mkdir -p "$(dirname "$PRE")"
        flock -w 43200 "${PRE}.lock" -c \
          "[ -f $PRE ] || $PY scripts/simclr_pretrain.py --config configs/diagnostics/diagvit_none_tin_${PCT}pct.yaml \
             --seed $SEED --out $PRE --data-root $DR" \
          || { echo "PRETRAIN FAIL vit_tin ${PCT}/${SEED}"; continue; }
        echo "=== diagdeitssl_tin_simclr_${PCT}pct seed ${SEED} ==="
        $PY train.py --config configs/diagnostics/diagdeitssl_tin_simclr_${PCT}pct.yaml --seed $SEED --data-root "$DR" \
          && echo "OK deitssl_tin ${PCT} ${SEED}" || echo "FAIL deitssl_tin ${PCT} ${SEED}"
    done
    echo "VITENV_SSL_DONE ${PCT}"
done

# the long pole last: tin@100% ViT + DeiT pairs
for CFG in diagvit_none_tin_100pct diagvit_aux_tin_100pct \
           diagdeit_none_tin_100pct diagdeit_aux_tin_100pct; do
    run3 "$CFG"
done
echo "VITENV_PCT_DONE 100"

# G probes at the decisive fractions
for CFG in diagvit_none_tin_25pct diagvit_aux_tin_25pct \
           diagvit_none_tin_100pct diagvit_aux_tin_100pct \
           diagdeit_none_tin_25pct diagdeit_aux_tin_25pct; do
    $PY analysis/linear_probe.py --run runs/${CFG} \
        --config configs/diagnostics/${CFG}.yaml --data-root "$DR" \
        && echo "PROBE OK ${CFG}" || echo "PROBE FAIL ${CFG}"
done
echo "VITENV_WAVE_COMPLETE"
