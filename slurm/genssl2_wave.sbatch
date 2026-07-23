#!/bin/bash
#SBATCH --job-name=genssl2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# REPAIR of genssl Tracks A+B (2026-07-23): simclr_pretrain.py carried a
# leftover cifar100-only guard, so every tin pretrain in genssl died at claim
# time (24 guard hits); Track C ran fine (no pretrain needed). The guard is
# now a verified whitelist (cifar100, tin) -- both paths smoke-tested (conv 15
# steps, vit 78 steps @64px). Predictions and falsifiers are UNCHANGED from
# the genssl header; the local 3090 fills 2/3/7/15/25% concurrently.
# Also re-runs diagdeitssl_tin_simclr_10pct (genssl's Track C reaches it
# without Track B's pretrain and will fail it) and the A/B probes.
set -uo pipefail
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
REPO=/mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
cd "$REPO"
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

STAGE=/dev/shm/genssl2_${SLURM_JOB_ID}
mkdir -p "$STAGE"; trap 'rm -rf "$STAGE"' EXIT
ln -sfn "$REPO"/data/* "$STAGE"/ 2>/dev/null
rm -f "$STAGE/tiny-imagenet-200"
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

for PCT in 1 5 10; do
    pretrain_then "simclr_pre_tin_${PCT}pct" "tin_none_${PCT}pct" "diagssl_tin_simclr_${PCT}pct"
    echo "GENSSL2_A_DONE ${PCT}"
done
pretrain_then "simclr_pre_vit_tin_10pct" "diagvit_none_tin_10pct" "diagsslvit_tin_simclr_10pct"
echo "GENSSL2_B_DONE"
for SEED in 0 1 2; do
    echo "=== diagdeitssl_tin_simclr_10pct seed ${SEED} ==="
    $PY train.py --config configs/diagnostics/diagdeitssl_tin_simclr_10pct.yaml --seed $SEED --data-root "$DR" \
      && echo "OK deitssl_tin ${SEED}" || echo "FAIL deitssl_tin ${SEED}"
done
echo "GENSSL2_C_DONE"

for CELL in diagssl_tin_simclr_5pct diagssl_tin_simclr_10pct \
            diagsslvit_tin_simclr_10pct diagdeitssl_tin_simclr_10pct; do
    $PY analysis/linear_probe.py --run runs/${CELL} \
        --config configs/diagnostics/${CELL}.yaml --data-root "$DR" \
        && echo "PROBE OK ${CELL}" || echo "PROBE FAIL ${CELL}"
done
echo "GENSSL2_WAVE_COMPLETE"
