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
# It consumes worklist.turing -- a DISJOINT slice (scripts/split_worklist.py):
# the LIVE cells at 50%/100%, most expensive first. turing's 2 GPUs have no
# queue wait and a 48h walltime, which suits long runs; and per the 2026-07-22
# decision they are NOT spent on configurations the ledger already closed
# (those 1,467 runs go to BSC's 32 GPUs instead).

set -uo pipefail
REPO=/mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
cd "$REPO"

WORKLIST=$REPO/worklist.turing
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
    i=$(flock "$LOCK" bash -c "read -r v < '$CTR' 2>/dev/null || v=0;
                               echo \$((v+1)) > '$CTR'; echo \$v")
    i=$((i+1))
    [ "$i" -gt "$N" ] && { echo "turing queue drained"; break; }
    cmd=$(sed -n "${i}p" "$WORKLIST")
    [ -z "$cmd" ] && continue
    echo "TASK $i/$N :: ${cmd:0:110}"
    eval "${cmd//python /$PY }" > "$REPO/slurm/logs/gtask_${i}.log" 2>&1 \
        && echo "OK   $i" || echo "FAIL $i (see slurm/logs/gtask_${i}.log)"
done
echo "TURING_GRID_WORKER_COMPLETE"
