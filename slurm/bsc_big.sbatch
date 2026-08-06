#!/bin/bash
#SBATCH --job-name=ms_big
#SBATCH --account=ub234
#SBATCH --qos=acc_resc
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scratch/ub234/momentstem/logs/%x_%j.out
#
# MareNostrum5 grid worker (2026-07-22). One job = one node = 4 H100s, each
# running an INDEPENDENT training stream that pulls the next task atomically
# from the shared worklist. Submit N of these for 4N concurrent GPUs.
#
# Why a shared queue rather than pre-partitioned job scripts: the tasks span
# ~1000x in cost (a 1% ViT cell vs a 100% SimCLR cell), so any static split
# leaves GPUs idle. Pulling on demand keeps all 4 busy until the queue drains.

set -uo pipefail
source /gpfs/scratch/ub234/momentstem/env.sh
cd "$MS_ROOT/repo"

WORKLIST=$MS_ROOT/worklist.big
CTR=$MS_ROOT/queue.counter.big
LOCK=$MS_ROOT/queue.lock.big
export OUT=$MS_ROOT/runs
N=$(wc -l < "$WORKLIST")

# --- stage the datasets into RAM: tin is 110k small files and GPFS is even
# worse at that than BeeGFS. Everything else is symlinked so one --data-root
# serves every dataset. Falls back to plain GPFS if /dev/shm is too small.
STAGE=/dev/shm/ms_$SLURM_JOB_ID
if mkdir -p "$STAGE" 2>/dev/null; then
    ln -sfn "$MS_DATA"/* "$STAGE"/ 2>/dev/null
    rm -f "$STAGE/tiny-imagenet-200"
    if unzip -q "$MS_DATA/tiny-imagenet-200.zip" -d "$STAGE" 2>/dev/null; then
        export DR=$STAGE
    else
        export DR=$MS_DATA
    fi
    # imagenet64 is a 16GB uint8 memmap hit RANDOMLY 1.28M times per epoch;
    # left as a GPFS symlink that read IS the bottleneck (same lesson as tin's
    # 110k small files: ~17s -> 2.2s per epoch once staged). Copy it into RAM
    # when the node has room -- ACC nodes have ~500GB, so 16GB is cheap.
    if [ -d "$MS_DATA/imagenet64" ]; then
        avail=$(df -k --output=avail /dev/shm 2>/dev/null | tail -1)
        need=$(du -sk "$MS_DATA/imagenet64" 2>/dev/null | cut -f1)
        if [ -n "$avail" ] && [ -n "$need" ] && [ "$avail" -gt $((need + 52428800)) ]; then
            rm -f "$STAGE/imagenet64"
            if cp -r "$MS_DATA/imagenet64" "$STAGE/" 2>/dev/null; then
                echo "staged imagenet64 -> /dev/shm ($((need/1024/1024))GB)"
                export DR=$STAGE
            fi
        else
            echo "imagenet64 NOT staged (shm avail=${avail}k need=${need}k); using GPFS"
        fi
    fi
    trap 'rm -rf "$STAGE"' EXIT
else
    export DR=$MS_DATA
fi
echo "host=$(hostname) DR=$DR tasks=$N"
nvidia-smi --query-gpu=index,name --format=csv,noheader

DEADLINE=$(( $(date +%s) + 22*3600 ))   # stop CLAIMING new work at 4.5h

IDLE_MAX=${MS_IDLE_MAX:-15}      # ~15 min of polling before a drained worker exits

worker () {                              # $1 = GPU index
    export CUDA_VISIBLE_DEVICES=$1
    IDLE=0
    while :; do
        [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "[gpu$1] deadline"; break; }
        # atomically claim the next line number
        i=$(flock "$LOCK" bash -c "read -r v < '$CTR' 2>/dev/null || v=0;
                                   echo \$((v+1)) > '$CTR'; echo \$v")
        i=$((i+1))
        if [ "$i" -gt "$N" ]; then
            # Queue empty. Do NOT exit immediately: on 2026-08-05 two of four
            # GPUs sat at 0% for hours because their workers drained while
            # stragglers ran on the others, and an exited worker never comes
            # back even when new tasks are APPENDED (which happens often).
            # Poll for a bounded window instead, then give up so the keeper's
            # "no workers live -> reconcile" path can still fire.
            NEWN=$(wc -l < "$WORKLIST" 2>/dev/null || echo 0)
            if [ "$NEWN" -gt "$N" ]; then
                echo "[gpu$1] worklist grew $N -> $NEWN, resuming"
                N=$NEWN
                continue
            fi
            IDLE=$((IDLE + 1))
            if [ "$IDLE" -ge "$IDLE_MAX" ]; then
                echo "[gpu$1] queue drained (idled ${IDLE}x)"; break
            fi
            sleep 60
            continue
        fi
        IDLE=0
        cmd=$(sed -n "${i}p" "$WORKLIST")
        [ -z "$cmd" ] && continue
        echo "[gpu$1] TASK $i/$N :: ${cmd:0:110}"
        # LANE-PREFIXED (2026-08-06): both lanes previously wrote
        # logs/task_${i}.log, so grid task 61 CLOBBERED big task 61. When all
        # 12 Stage-2 ImageNet-100 ViT runs failed, their tracebacks had already
        # been overwritten by unrelated diagdepth cells and the failure had to
        # be reproduced locally to diagnose. Keep the lanes' logs disjoint.
        eval "$cmd" > "$MS_ROOT/logs/big_task_${i}.log" 2>&1 \
            && echo "[gpu$1] OK   $i" || echo "[gpu$1] FAIL $i (see logs/big_task_${i}.log)"
    done
}

# CONCURRENCY PER GPU (2026-08-05, measured not assumed): a live node showed
# 0-6% GPU utilisation and 2-3GB of 65GB in use -- these 32-64px cells are
# DATALOADER-bound, so one training per H100 wastes ~95% of it. Run SLOTS
# trainings per GPU instead.
# The binding constraint is CPU, not GPU: the node has 160 cores (measured
# 2026-08-05 with nproc --all; an earlier note said 80), and the
# ledger records a 5x SLOWDOWN when workers oversubscribe cores (load 113 on
# 16 cores => 44 min/seed vs 5.4). Budget ~= SLOTS*4*(num_workers+1) threads
# and keep it under 80.
# BIG lane default is LOWER than the grid lane on purpose: its tasks are
# ~18h ImageNet64 cells, and there are only ~24 of them -- extra slots would
# just sit idle, while any CPU contention risks pushing an 18h task past the
# 24h walltime and orphaning it. Parallelism pays for MANY SHORT tasks.
SLOTS=${MS_SLOTS:-3}
echo "concurrency: $SLOTS training(s) per GPU = $((SLOTS*4)) per node"
for g in 0 1 2 3; do
    for k in $(seq 1 "$SLOTS"); do worker "$g" & done
done
wait
echo "BSC_WORKER_COMPLETE"
