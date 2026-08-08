#!/usr/bin/env bash
# Generic GPU work queue.
#
# Reads one shell command per line from a task file and runs them in order,
# SLOTS at a time, but only once the GPU has NEED_MB free. That is what makes
# it safe to arm while something else is still running: it waits rather than
# competing, so nothing in flight is evicted.
#
# Lines starting with # are ignored. A task that fails is recorded with its
# exit code and does NOT stop the queue -- but the code is logged, because a
# runner that reports success over a crash has already cost this project two
# wasted cycles today.
set -u
cd "$(dirname "$0")/.."
TASKS=${TASKS:?set TASKS=path/to/tasklist}
SLOTS=${SLOTS:-2}
NEED_MB=${NEED_MB:-6000}
LOG=${LOG:-logs/queue.log}
mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) queue start: $(grep -cvE '^\s*(#|$)' "$TASKS") tasks, SLOTS=$SLOTS, need ${NEED_MB}MiB" >> "$LOG"

wait_for_gpu() {
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "${free:-0}" -ge "$NEED_MB" ] && return
    sleep 60
  done
}
wait_for_gpu
echo "$(date -Is) gpu free, running" >> "$LOG"

n=0
while IFS= read -r cmd <&3; do   # read on FD 3, not stdin
  case "$cmd" in ''|\#*) continue;; esac
  n=$((n+1))
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 10; done
  # stdin MUST be /dev/null: a task that reads stdin otherwise swallows the
  # rest of the task file and the queue silently stops after one item.
  ( eval "$cmd" >/dev/null 2>&1 </dev/null
    echo "$(date -Is) [$n] rc=$? :: ${cmd:0:110}" >> "$LOG" ) &
done 3< "$TASKS"
wait
ok=$(grep -c 'rc=0 ::' "$LOG"); bad=$(grep -c 'rc=[1-9]' "$LOG")
echo "$(date -Is) QUEUE_COMPLETE  ok=$ok failed=$bad" >> "$LOG"
