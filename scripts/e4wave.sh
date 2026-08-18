#!/usr/bin/env bash
# E4: is the +26.01 ViT-B headline partly under-training?
# diagin100e200_vitb_{none,aux} -- ImageNet-100 @224px, ViT-B/16, DeiT aug,
# 200 epochs against the measured 100. 2 arms x 3 seeds.
#
# Moved onto the workstations 2026-08-15 as a RACE against BSC job 44631727,
# which is queued behind an unknown wait (the previous submission waited two
# days, then died in 15 min on stale BSC code). Whichever finishes first wins;
# the other should be cancelled. Both are kept because the GPUs are otherwise
# idle -- every other experiment in the campaign has landed.
#
# MEASURED on a 3090 before launching, not estimated: batch 128 @224px peaks
# at 10.0 GiB of 24 and runs at 305 ms/step, so 989 steps/epoch x 200 epochs
# = 16.7 h/run of compute. Data is pre-packed (train.bin + offsets, mmap'd),
# so the run is compute-bound, not dataloader-bound -- which is why ONE stream
# per GPU is correct here. A second stream would fit in memory (2x10 GiB) but
# buy nothing, unlike the small-image waves.
#
# SEEDS is the partition across machines. As everywhere else in this campaign
# the per-run flock is node-local and these machines share no filesystem, so
# the partition is the only thing keeping them disjoint.
#
# CAVEAT recorded here rather than discovered later: with a 3/3 split one
# seed's two arms land on different machines. That is acceptable ONLY because
# the two workstations were verified to produce byte-identical results (same
# venv pins; an identical-output smoke was run on both), and because Delta is
# a within-seed difference. If that verification ever lapses, split by seed
# PAIR instead so both arms of a seed stay on one machine.
set -u
cd "$(dirname "$0")/.."
echo $$ > logs/e4wave.pid

PY=${PY:-~/venvs/momentstem/bin/python}
CFG=configs/diagnostics
TASKS=${TASKS:?set TASKS="arm:seed arm:seed ..."}

echo "e4 wave: $(echo $TASKS | wc -w) task(s), 1 stream (compute-bound)"
for t in $TASKS; do
  arm=${t%%:*}; seed=${t##*:}
  cell=diagin100e200_vitb_${arm}
  if [ -f "runs/$cell/seed$seed/final.json" ]; then
    echo "--- SKIP $cell seed$seed (already final)"; continue
  fi
  echo "--- $cell seed$seed $(date -Is)"
  $PY train.py --config "$CFG/$cell.yaml" --seed "$seed" 2>&1
done

echo "E4WAVE_COMPLETE $(date -Is)"
