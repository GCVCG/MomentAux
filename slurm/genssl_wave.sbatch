#!/bin/bash
#SBATCH --job-name=genssl
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# GENERALIZATION CAMPAIGN (2026-07-23, user: "results are lacking for
# generalizability -- not all datasets are used for the configurations").
# The coverage audit showed: moment-aux [plain] spans 10 datasets x 5
# backbones (the core claim IS general), but EVERY SSL and DeiT comparison is
# C100-only. This wave transplants all three positioning claims to tin
# (2x images, 64px, 200-way -- different input scale AND label space).
#
# Track A -- conv SSL on tin @1/5/10% (vs aux +1.49/+2.13/+1.65):
#   PREDICTIONS (C100 precedent: SSL(2x) beats aux, margin grows with data):
#     diagssl_tin_simclr_1pct:  6.3..7.8   (base 5.30, aux 6.80 -- SSL ~ aux)
#     diagssl_tin_simclr_5pct:  23.5..26   (base 21.08, aux 23.20)
#     diagssl_tin_simclr_10pct: 35.5..38.5 (base 33.60, aux 35.24)
#   FALSIFIER (narrows the SSL claim): SSL <= aux at BOTH 5% and 10% =>
#     conv-SSL dominance is C100-specific.
# Track B -- ViT SSL on tin@10% (vs aux-ViT-tin +9.15, G=12.39):
#   PREDICTION: 13..18 (base 8.60, aux 17.75) -- at 10k imgs C100 says SSL
#     roughly catches up to the prior; betting it lands AT or BELOW aux.
#   FALSIFIER: > 19.75 (aux + 2) => SSL beats the prior on attention off-C100.
# Track C -- DeiT column on tin/ViT @10%:
#   PREDICTIONS: deit-base(tin) 10..16 (aug rescues the underfit ViT);
#     Delta_deit(aux-base) +10..+20 (stacking transplants; plain was +9.15);
#     deit-ssl BELOW deit-aux (the substitution flip transplants).
#   FALSIFIER: deit-ssl >= deit-aux on tin => the flip is C100-specific and
#     the modern-recipe claim must be scoped to CIFAR.
set -uo pipefail
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
REPO=/mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
cd "$REPO"
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

STAGE=/dev/shm/genssl_${SLURM_JOB_ID}
mkdir -p "$STAGE"; trap 'rm -rf "$STAGE"' EXIT
ln -sfn "$REPO"/data/* "$STAGE"/ 2>/dev/null
rm -f "$STAGE/tiny-imagenet-200"
echo "staging tin -> $STAGE"
unzip -q "$REPO/data/tiny-imagenet-200.zip" -d "$STAGE"
DR=$STAGE

pretrain_then () {            # $1 pretrain-dir  $2 parent-cfg  $3 train-cfg
    local PDIR=$1 PCFG=$2 TCFG=$3
    for SEED in 0 1 2; do
        PRE=runs/${PDIR}/seed${SEED}/pretrain.pt
        mkdir -p "$(dirname "$PRE")"
        echo "=== pretrain ${PDIR} seed ${SEED} ==="
        flock -w 43200 "${PRE}.lock" -c \
          "[ -f $PRE ] || $PY scripts/simclr_pretrain.py --config configs/diagnostics/${PCFG}.yaml \
             --seed $SEED --out $PRE --data-root $DR" \
          || { echo "PRETRAIN FAIL ${PDIR}/${SEED}"; continue; }
        echo "=== ${TCFG} seed ${SEED} ==="
        $PY train.py --config configs/diagnostics/${TCFG}.yaml --seed $SEED --data-root "$DR" \
          && echo "OK ${TCFG} ${SEED}" || echo "FAIL ${TCFG} ${SEED}"
    done
}

# ---- Track A: conv SSL on tin (cheap -> expensive) ----
for PCT in 1 5 10; do
    pretrain_then "simclr_pre_tin_${PCT}pct" "tin_none_${PCT}pct" "diagssl_tin_simclr_${PCT}pct"
    echo "GENSSL_A_DONE ${PCT}"
done

# ---- Track B: ViT SSL on tin@10% ----
pretrain_then "simclr_pre_vit_tin_10pct" "diagvit_none_tin_10pct" "diagsslvit_tin_simclr_10pct"
echo "GENSSL_B_DONE"

# ---- Track C: DeiT column on tin/ViT @10% ----
for CFG in diagdeit_none_tin_10pct diagdeit_aux_tin_10pct; do
    for SEED in 0 1 2; do
        echo "=== ${CFG} seed ${SEED} ==="
        $PY train.py --config configs/diagnostics/${CFG}.yaml --seed $SEED --data-root "$DR" \
          && echo "OK ${CFG} ${SEED}" || echo "FAIL ${CFG} ${SEED}"
    done
done
# deit-ssl reuses Track B's pretrains (augmentation-agnostic)
for SEED in 0 1 2; do
    echo "=== diagdeitssl_tin_simclr_10pct seed ${SEED} ==="
    $PY train.py --config configs/diagnostics/diagdeitssl_tin_simclr_10pct.yaml --seed $SEED --data-root "$DR" \
      && echo "OK deitssl_tin ${SEED}" || echo "FAIL deitssl_tin ${SEED}"
done
echo "GENSSL_C_DONE"

# ---- G probes on the decisive cells ----
for CELL in diagssl_tin_simclr_5pct diagssl_tin_simclr_10pct \
            diagsslvit_tin_simclr_10pct diagdeit_none_tin_10pct \
            diagdeit_aux_tin_10pct diagdeitssl_tin_simclr_10pct; do
    $PY analysis/linear_probe.py --run runs/${CELL} \
        --config configs/diagnostics/${CELL}.yaml --data-root "$DR" \
        && echo "PROBE OK ${CELL}" || echo "PROBE FAIL ${CELL}"
done
echo "GENSSL_WAVE_COMPLETE"
