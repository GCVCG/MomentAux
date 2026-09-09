#!/usr/bin/env bash
# The 200-epoch ImageNet re-run on the LOCAL 3090 (2026-09-09, user: "free up
# the local machine and run the needed experiments"; BSC starved the chain for
# four days). Same 276 tasks as the cancelled BSC lanes, ordered CHEAPEST
# CELL FIRST (scripts/worklist_in_e200_local.txt) so partial results are
# scorable early; arms and seeds of a cell stay adjacent.
# SLOTS streams claim lines from one counter under flock (single node, so the
# lock is real). train.py's completed-run guard and run lock make restarts and
# a raised SLOTS safe. Ends with IN_E200_LOCAL_COMPLETE.
# ponytail: no per-slot memory check; the in64 100% cells sit last -- restart
# with SLOTS=1 when the queue reaches them if two do not fit in 24 GB.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/venvs/momentstem/bin/python}
WL=scripts/worklist_in_e200_local.txt
CTR=logs/in_e200/counter; LOCK=logs/in_e200/counter.lock
SLOTS=${SLOTS:-2}
mkdir -p logs/in_e200; [ -s $CTR ] || echo 0 > $CTR
echo $$ > logs/in_e200_wave.pid
N=$(wc -l < $WL)
worker() {
  while :; do
    i=$(flock $LOCK bash -c "read -r v < $CTR; echo \$((v+1)) > $CTR; echo \$v")
    [[ "$i" =~ ^[0-9]+$ ]] || { echo "bad counter '$i'"; return 1; }
    [ "$i" -ge "$N" ] && return 0
    read -r cell seed < <(sed -n "$((i+1))p" $WL)
    if [ -f runs/$cell/seed$seed/final.json ]; then echo "SKIP $cell s$seed"; continue; fi
    echo "START $cell s$seed slot$1 $(date -Is)"
    $PY train.py --config configs/diagnostics/$cell.yaml --seed $seed > logs/in_e200/${cell}_s$seed.log 2>&1
    echo "END   $cell s$seed rc=$? $(date -Is)"
  done
}
for s in $(seq $SLOTS); do worker $s & sleep 20; done
wait
echo "IN_E200_LOCAL_COMPLETE $(date -Is)"
