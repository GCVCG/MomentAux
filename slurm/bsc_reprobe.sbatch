#!/bin/bash
#SBATCH -J ms_reprobe
#SBATCH -o /gpfs/scratch/${CLUSTER_ACCOUNT}/momentstem/logs/ms_reprobe_%j.out
#SBATCH -t 20:00:00
#SBATCH -A ${CLUSTER_ACCOUNT}
#SBATCH -q acc_resc
#SBATCH -p acc
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -c 80
set -u
MS=/gpfs/scratch/${CLUSTER_ACCOUNT}/momentstem
cd $MS
source env.sh
# Build the worklist from the identity sweep's report: only checkpoints that
# VERIFIED are probed. Runs inside the job so a re-queue always rebuilds
# against the latest report rather than a stale file.
python $MS/repo/scripts/build_reprobe_worklist.py || exit 1
WL=$MS/worklist.reprobe
CTR=$MS/queue.counter.reprobe
LOCK=$MS/queue.lock.reprobe
[ -s "$CTR" ] || echo 0 > "$CTR"
N=$(wc -l < "$WL")
echo "reprobe lane: $N tasks"
cd $MS/repo
export MS_PROBE_WORKERS=3
SLOTS=3
run_slot() {
  local gpu=$1
  while true; do
    i=$(flock "$LOCK" bash -c "read -r v < '$CTR' || exit 1
        case \$v in ''|*[!0-9]*) exit 1;; esac
        echo \$((v+1)) > '$CTR'; echo \$v")
    [ -z "${i:-}" ] && { sleep 2; continue; }
    [ "$i" -ge "$N" ] && break
    cmd=$(sed -n "$((i+1))p" "$WL")
    [ -z "$cmd" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu $cmd >> $MS/logs/reprobe_g${gpu}.log 2>&1 \
      && echo "OK $i" || echo "FAIL $i :: $cmd"
  done
}
for g in 0 1 2 3; do
  for s in $(seq 1 $SLOTS); do run_slot $g & done
done
wait
echo "REPROBE_COMPLETE"
