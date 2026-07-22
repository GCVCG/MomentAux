#!/bin/bash
#SBATCH --job-name=deitssl
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# THE MISSING COLUMN (2026-07-22, user: "are we going to merge the DeiT-aug
# with SSL - aux?"). Every SSL-vs-prior head-to-head so far ran under plain
# crop+flip; the 2x3 {plain,deit} x {none,aux,simclr} has no deit-simclr cell.
#
# THE QUESTION: aux and augmentation STACK (different currencies -- structure
# vs nuisance-invariance). SimCLR's objective IS invariance to an augmentation
# family, so heavy augmentation may SUBSTITUTE for it. If so the head-to-head
# FLIPS under a modern recipe and the ViT positioning claim strengthens.
#
# PREDICTIONS RECORDED IN ADVANCE (bands sit BELOW the aux cell -- betting on
# partial substitution: SSL amplified by augmentation, but less than aux is):
#     @5%  24..30   (deit-base 12.55, deit-aux 28.89)
#     @10% 38..43   (deit-base 20.28, deit-aux 41.29)
#     @25% 50..57   (deit-base 32.27, deit-aux 57.32)
#   FALSIFIER A: deit-ssl >= deit-aux at BOTH 10% and 25% => equal
#     amplification, no flip; the positioning claim holds only under plain aug.
#   FALSIFIER B: deit-ssl <= deit-base + 5 => augmentation almost entirely
#     substitutes for SimCLR; the prior is the only intervention that survives.
#   PROBES: G(deit-ssl) vs G(deit-aux)=22.20. If SSL's currency is invariance,
#     its G should rise LESS under augmentation than aux's did (13.46 -> ?).
#   COMBO diagdeitsslaux_10pct: 'aux XOR SSL' was established WITHOUT
#     augmentation. Beating BOTH singles here would falsify it under a
#     modern recipe.
#
# The SimCLR pretrain checkpoint is AUGMENTATION-AGNOSTIC (only the supervised
# stage changes), so this reuses fillvit's pretrains -- ~27 supervised runs,
# no new pretraining. Fractions are ordered by pretrain readiness; the
# [ -f ] guard builds any pretrain fillvit has not reached yet, so the wave
# cannot deadlock behind it.
set -uo pipefail
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
REPO=/mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
cd "$REPO"
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

run_cell () {                       # $1 = config basename, $2 = pct
    local CFG=$1 PCT=$2
    for SEED in 0 1 2; do
        PRE=runs/simclr_pre_vit_${PCT}pct/seed${SEED}/pretrain.pt
        if [ ! -f "$PRE" ]; then
            echo "=== simclr-vit pretrain ${PCT}pct seed ${SEED} (not yet built) ==="
            flock -w 21600 "${PRE}.lock" -c \
              "[ -f $PRE ] || $PY scripts/simclr_pretrain.py \
                 --config configs/diagnostics/diagvit_none_${PCT}pct.yaml \
                 --seed $SEED --out $PRE" || { echo "PRETRAIN FAIL $PCT/$SEED"; continue; }
        fi
        echo "=== ${CFG} seed ${SEED} ==="
        $PY train.py --config configs/diagnostics/${CFG}.yaml --seed $SEED \
            && echo "OK ${CFG} ${SEED}" || echo "FAIL ${CFG} ${SEED}"
    done
}

# decisive fractions first (10/25 carry both falsifiers), then the envelope
for PCT in 10 25 5 1 2 3 7 15 100; do
    run_cell "diagdeitssl_simclr_${PCT}pct" "$PCT"
    echo "DEITSSL_PCT_DONE ${PCT}"
done

run_cell "diagdeitsslaux_10pct" 10
echo "DEITSSL_COMBO_DONE"

# G probes on the decisive cells, against the deit baseline ckpts
for PCT in 10 25 5; do
    $PY analysis/linear_probe.py \
        --runs runs/diagdeitssl_simclr_${PCT}pct runs/diagdeit_none_${PCT}pct \
        --out runs/probe_deitssl_${PCT}pct.json || echo "PROBE FAIL ${PCT}"
done
echo "DEITSSL_WAVE_COMPLETE"
