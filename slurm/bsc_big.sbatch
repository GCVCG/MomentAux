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
    trap 'rm -rf "$STAGE"' EXIT
else
    export DR=$MS_DATA
fi
echo "host=$(hostname) DR=$DR tasks=$N"
nvidia-smi --query-gpu=index,name --format=csv,noheader

DEADLINE=$(( $(date +%s) + 22*3600 ))   # stop CLAIMING new work at 4.5h

worker () {                              # $1 = GPU index
    export CUDA_VISIBLE_DEVICES=$1
    while :; do
        [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "[gpu$1] deadline"; break; }
        # atomically claim the next line number
        i=$(flock "$LOCK" bash -c "read -r v < '$CTR' 2>/dev/null || v=0;
                                   echo \$((v+1)) > '$CTR'; echo \$v")
        i=$((i+1))
        [ "$i" -gt "$N" ] && { echo "[gpu$1] queue drained"; break; }
        cmd=$(sed -n "${i}p" "$WORKLIST")
        [ -z "$cmd" ] && continue
        echo "[gpu$1] TASK $i/$N :: ${cmd:0:110}"
        eval "$cmd" > "$MS_ROOT/logs/task_${i}.log" 2>&1 \
            && echo "[gpu$1] OK   $i" || echo "[gpu$1] FAIL $i (see logs/task_${i}.log)"
    done
}

for g in 0 1 2 3; do worker "$g" & done
wait
echo "BSC_WORKER_COMPLETE"
