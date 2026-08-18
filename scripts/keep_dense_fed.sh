#!/bin/bash
# Dense-lane keeper: regenerate "what is still missing" and keep workers on it.
#
# WHY THIS EXISTS. The dense grid was hand-fed three times in one day, and each
# hand-feed hit a different edge of the same design flaw: a lane whose worklist
# is a FIXED LIST drains, its claim counter overruns, its workers exit, and the
# ~5% of tasks that die to the random SIGABRT are never retried. The recorded
# lesson from the GPU audit was "long-lived lanes fed by a reconcile do not have
# this problem" -- this is that reconcile.
#
# THE DESIGN PROPERTY THAT MAKES IT SAFE, and it is worth stating because it is
# what lets this refill under LIVE workers where the classification keeper has
# to wait for CUR==0: the worklist is always regenerated as "every (config,seed)
# that has no final.json". A worker claims an index under the flock but reads
# the line OUTSIDE it, so a refill can hand it a different line than the one it
# claimed. With a fixed list that would silently skip a task. With a
# missing-only list every line is valid work, so a mis-indexed claim costs
# nothing -- the task it "skipped" is simply still missing and reappears in the
# next pass. Duplicates are caught by train_dense.py's final.json guard and its
# non-blocking flock run lock.
set -uo pipefail

MS=${CLUSTER_SCRATCH}/momentstem
LANE=${LANE:-dense7}
WL=$MS/worklist.$LANE
CTR=$MS/queue.counter.$LANE
LOCK=$MS/queue.lock.$LANE
TARGET=${TARGET:-4}          # workers to keep running+pending
CAP=${CAP:-10}               # QOS GrpTRES node=10, shared with other projects

cd "$MS/repo" || exit 1
stamp() { date -u +%FT%TZ; }

# --- regenerate the missing list ------------------------------------------
# Written to a temp file with its exit status CHECKED. A generator that fails
# silently emits zero lines, which reads as "nothing missing" and would declare
# the lane COMPLETE and stop it forever -- the exact failure the 2026-08-01
# reconcile hardening was about.
if ! python3 "$MS/dense_missing.py" > "$WL.new" 2>"$MS/logs/dense_keeper.err"; then
    echo "$(stamp) STATE ERROR: generator FAILED, touching nothing"; exit 1
fi
M=$(wc -l < "$WL.new")

if [ "$M" -eq 0 ]; then
    rm -f "$WL.new"; touch "$MS/DENSE_COMPLETE"
    echo "$(stamp) STATE dense grid COMPLETE (0 missing)"; exit 0
fi

# --- swap worklist AND counter atomically w.r.t. a claim -------------------
# Doing these as two unsynchronised steps has a window in either order: a short
# list with a high counter makes workers exit, a low counter with a long list
# makes them burn the list. Both were live risks when this was done by hand.
CUR_CTR=$(cat "$CTR" 2>/dev/null || echo 0)
if [ "$CUR_CTR" -ge "$(wc -l < "$WL" 2>/dev/null || echo 0)" ]; then
    flock "$LOCK" bash -c "cp '$WL.new' '$WL'; echo 0 > '$CTR'"
    echo "$(stamp) STATE refilled: $M missing, counter reset (was $CUR_CTR)"
else
    rm -f "$WL.new"
    echo "$(stamp) STATE draining: counter $CUR_CTR/$(wc -l < "$WL"), $M missing"
fi

# --- top the lane up to TARGET workers, within the node cap ----------------
LIVE=$(squeue -u "$USER" -h -n "ms_$LANE" -t R,PD -o "%i" | wc -l)
USED=$(squeue -u "$USER" -h -t R,PD -o "%D" | awk '{s+=$1} END {print s+0}')
NEED=$(( TARGET - LIVE )); ROOM=$(( CAP - USED ))
[ "$NEED" -gt "$ROOM" ] && NEED=$ROOM
# Never queue more workers than there is work for; a worker per remaining task
# is already generous since each runs SLOTS*4 of them.
[ "$NEED" -gt "$M" ] && NEED=$M
echo "$(stamp) STATE live=$LIVE used=$USED/$CAP missing=$M submitting=$((NEED > 0 ? NEED : 0))"
i=0; while [ "$i" -lt "$NEED" ]; do sbatch "slurm/bsc_$LANE.sbatch" >/dev/null && i=$((i+1)) || break; done
