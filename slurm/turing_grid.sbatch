#!/bin/bash
#SBATCH --job-name=ms_grid_t
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# turing grid worker (2026-07-22) -- the hedge against BSC queue uncertainty.
#
# It consumes the SAME worklist as the BSC workers but from the BACK
# (expensive -> cheap) while BSC consumes it from the FRONT (cheap ->
# expensive). The two clusters therefore never touch the same task until they
# meet in the middle, WITHOUT needing a shared lock across sites. turing's 2
# GPUs are best spent on the few very long cells anyway: the 50%/100% columns
# are only 20% of the runs but 63% of the compute.
#
# Stop both sides when (BSC counter + turing counter) reaches the task count.

set -uo pipefail
REPO=/mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
cd "$REPO"

WORKLIST=$REPO/worklist.txt
CTR=$REPO/queue.counter.turing        # counts DOWN from the end
LOCK=$REPO/queue.lock.turing
export OUT=$REPO/runs
N=$(wc -l < "$WORKLIST")

STAGE=/dev/shm/msgrid_${SLURM_JOB_ID}
if mkdir -p "$STAGE" 2>/dev/null; then
    ln -sfn "$REPO"/data/* "$STAGE"/ 2>/dev/null
    rm -f "$STAGE/tiny-imagenet-200"
    if unzip -q "$REPO/data/tiny-imagenet-200.zip" -d "$STAGE" 2>/dev/null; then
        export DR=$STAGE
    else
        export DR=$REPO/data
    fi
    trap 'rm -rf "$STAGE"' EXIT
else
    export DR=$REPO/data
fi
echo "host=$(hostname) DR=$DR tasks=$N (consuming BACKWARD)"

DEADLINE=$(( $(date +%s) + 46*3600 ))

while :; do
    [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "deadline"; break; }
    k=$(flock "$LOCK" bash -c "read -r v < '$CTR' 2>/dev/null || v=0;
                               echo \$((v+1)) > '$CTR'; echo \$v")
    i=$(( N - k ))
    [ "$i" -lt 1 ] && { echo "queue drained from the back"; break; }
    cmd=$(sed -n "${i}p" "$WORKLIST")
    [ -z "$cmd" ] && continue
    echo "TASK $i/$N :: ${cmd:0:110}"
    eval "${cmd//python /$PY }" > "$REPO/slurm/logs/gtask_${i}.log" 2>&1 \
        && echo "OK   $i" || echo "FAIL $i (see slurm/logs/gtask_${i}.log)"
done
echo "TURING_GRID_WORKER_COMPLETE"
