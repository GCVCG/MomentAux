#!/bin/bash
#SBATCH --job-name=ms_probe
#SBATCH --account=${CLUSTER_ACCOUNT:?set CLUSTER_ACCOUNT}
#SBATCH --qos=acc_resc
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=08:00:00
#SBATCH --output=${CLUSTER_SCRATCH}/momentstem/logs/%x_%j.out
#
# ImageNet-scale G-probe lane (2026-08-06): the P1/S1 pass -- does Delta ~= G
# survive 1.28M images and 224px/86M params? Separate lane (own worklist/
# counter/lock) so the live ms_big training jobs are never disturbed; the
# 2026-08-03 lesson forbids touching a worklist under live workers.
#
# Protocol (pre-registered 2026-08-05): ImageNet64 cells are probed at FIXED
# SHOTS budgets (--shots-only; full-train LBFGS at 1.28M rows is impractical)
# -- those G values compare ONLY within the same budget. ImageNet-100 (126k
# imgs) takes the STANDARD full-train probe, directly comparable in protocol
# to every other G in the ledger, plus the same shots budgets as cross-check.
# diagin64_swin is EXCLUDED: its baseline is collapsed-at-chance, so a probe
# would measure a training failure, not a feature deficit (swin scope rule).

set -uo pipefail
source ${CLUSTER_SCRATCH}/momentstem/env.sh
cd "$MS_ROOT/repo"

WORKLIST=$MS_ROOT/worklist.probe
CTR=$MS_ROOT/queue.counter.probe
LOCK=$MS_ROOT/queue.lock.probe
export OUT=$MS_ROOT/runs
[ -s "$CTR" ] || echo 0 > "$CTR"
N=$(wc -l < "$WORKLIST")

# probes are dataloader-bound; extraction reads every image once, so give the
# loaders real workers (160-core node, 8 concurrent probes x 9 threads is fine)
export MS_PROBE_WORKERS=${MS_PROBE_WORKERS:-8}

STAGE=/dev/shm/ms_$SLURM_JOB_ID
if mkdir -p "$STAGE" 2>/dev/null; then
    ln -sfn "$MS_DATA"/* "$STAGE"/ 2>/dev/null
    export DR=$STAGE
    if [ -d "$MS_DATA/imagenet64" ]; then
        avail=$(df -k --output=avail /dev/shm 2>/dev/null | tail -1)
        need=$(du -sk "$MS_DATA/imagenet64" 2>/dev/null | cut -f1)
        if [ -n "$avail" ] && [ -n "$need" ] && [ "$avail" -gt $((need + 52428800)) ]; then
            rm -f "$STAGE/imagenet64"
            cp -r "$MS_DATA/imagenet64" "$STAGE/" && echo "staged imagenet64 -> shm"
        fi
    fi
    trap 'rm -rf "$STAGE"' EXIT
else
    export DR=$MS_DATA
fi
echo "host=$(hostname) DR=$DR tasks=$N nw=$MS_PROBE_WORKERS"

worker () {
    export CUDA_VISIBLE_DEVICES=$1
    IDLE=0
    while :; do
        i=$(flock "$LOCK" bash -c "
                v=\$(cat '$CTR' 2>/dev/null)
                case \"\$v\" in ''|*[!0-9]*) exit 1 ;; esac
                echo \$((v+1)) > '$CTR'
                echo \$v") \
            || { echo "[gpu$1] counter unreadable, backing off"; sleep 10; continue; }
        i=$((i+1))
        if [ "$i" -gt "$N" ]; then
            NEWN=$(wc -l < "$WORKLIST" 2>/dev/null || echo 0)
            if [ "$NEWN" -gt "$N" ]; then N=$NEWN; continue; fi
            IDLE=$((IDLE + 1))
            [ "$IDLE" -ge 5 ] && { echo "[gpu$1] drained"; break; }
            sleep 60; continue
        fi
        IDLE=0
        cmd=$(sed -n "${i}p" "$WORKLIST")
        [ -z "$cmd" ] && continue
        echo "[gpu$1] TASK $i/$N :: ${cmd:0:110}"
        eval "$cmd" > "$MS_ROOT/logs/probe_task_${i}.log" 2>&1 \
            && echo "[gpu$1] OK   $i" || echo "[gpu$1] FAIL $i (see logs/probe_task_${i}.log)"
    done
}

# 2 probes per GPU: extraction is the long pole and is loader-bound, but a
# vit_base extraction plus LBFGS can hold ~10GB; 2 x that fits any mix.
SLOTS=${MS_SLOTS:-2}
# ROUND-ROBIN over GPUs, not GPU-major. The loops used to nest the other way
# (all of gpu0's slots, then gpu1's ...), so slots claim in spawn order and a
# queue with fewer tasks than slots puts EVERY task on GPU 0. Measured on the
# ImageNet lane: 2 tasks left, both on gpu0 at 100% while gpus 1-3 sat at 0 MiB,
# running 3-5x slower than the same cells had on a free GPU. Interleaving means
# the first 4 claims land on 4 different GPUs.
for k in $(seq 1 "$SLOTS"); do
    for g in 0 1 2 3; do worker "$g" & done
done
wait
echo "PROBE_LANE_COMPLETE"
