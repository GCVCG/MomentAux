#!/bin/bash
#SBATCH --job-name=domain
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# DOMAIN-GENERALIZATION WAVE (2026-07-23, user-approved: EuroSAT + DTD +
# PathMNIST biomedical + Food-101). All five existing populations are small
# natural-photo sets; these four change the IMAGE STATISTICS axis: satellite
# optics, pure texture, histopathology stains, fine-grained food. Champion
# pair VERBATIM (headline-eligible names), 3 seeds, envelopes cheap->expensive.
#
# PREDICTIONS RECORDED IN ADVANCE (banded from the 5-population precedent;
# G unmeasured on all four -> qualitative bands, probes measure G after):
#   DTD (texture, 47-way, tiny): the prior's inductive bias is EXACTLY
#     oriented energy -> largest expected gains: Delta@5% +2..+6; if the
#     prior fails HERE the "generic early vision" story is in trouble.
#   EuroSAT (10-way, satellite): baselines will be HIGH (easy 10-way);
#     right-flank logic -> small gains: Delta@1% +1..+4, ~0 by 25%.
#     FALSIFIER for domain-generality: Delta@1% <= 0.
#   PathMNIST (9-way, 90k, stains): baseline high at any fraction >= 5%
#     (huge data) -> Delta@1% +0.5..+3, ~0 from 10% up. A NEGATIVE delta
#     at 1% would be the first domain where the prior HURTS at low data.
#   Food-101 (101-way, fine-grained): CUB precedent (fine-grained weak
#     supervision) -> Delta@5% +1.5..+4.5, Delta@1% suppressed (left flank).
# Ordering: full envelopes cheapest-first across datasets; 100% cells LAST
# (pathmnist/food101 @100% are the long poles and can be killed w/o collateral).
set -uo pipefail
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
REPO=/mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
cd "$REPO"
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

run_pair () {                  # $1 dataset  $2 pct
    for KIND in none aux; do
        for SEED in 0 1 2; do
            CFG=${1}_${KIND}_${2}pct
            [ -f "runs/${CFG}/seed${SEED}/final.json" ] && { echo "SKIP ${CFG} ${SEED}"; continue; }
            echo "=== ${CFG} seed ${SEED} ==="
            $PY train.py --config configs/diagnostics/${CFG}.yaml --seed $SEED \
              && echo "OK ${CFG} ${SEED}" || echo "FAIL ${CFG} ${SEED}"
        done
    done
    echo "DOMAIN_PAIR_DONE ${1}@${2}"
}

# cheap fractions first, interleaved across datasets; 100% last
for PCT in 1 5 10 25; do
    for DS in dtd eurosat pathmnist food101; do
        [ -f "configs/diagnostics/${DS}_none_${PCT}pct.yaml" ] || continue
        run_pair "$DS" "$PCT"
    done
done
for DS in dtd eurosat pathmnist food101; do
    run_pair "$DS" 100
done

# G probes on the decisive fractions
for DS in dtd eurosat pathmnist food101; do
    for PCT in 5 25; do
        for KIND in none aux; do
            CFG=${DS}_${KIND}_${PCT}pct
            [ -f "configs/diagnostics/${CFG}.yaml" ] || continue
            $PY analysis/linear_probe.py --run runs/${CFG} \
                --config configs/diagnostics/${CFG}.yaml \
                && echo "PROBE OK ${CFG}" || echo "PROBE FAIL ${CFG}"
        done
    done
done
echo "DOMAIN_WAVE_COMPLETE"
