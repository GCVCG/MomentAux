#!/bin/bash
#SBATCH --job-name=ms_insmoke
#SBATCH --account=ub234
#SBATCH --qos=acc_resc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:40:00
#SBATCH --output=/gpfs/scratch/ub234/momentstem/logs/%x_%j.out
#
# Pre-flight for the ImageNet envelope campaign (2026-08-10). Two jobs:
#
#   1. PROVE EVERY PATH TRAINS on the machine that will run it. Six 1-epoch
#      runs cover both datasets, all six backbones and both arms, at the
#      cheapest fraction. Every past wave that crash-looped (the cifar10/stl10
#      pretrain whitelist, genssl's cifar100-only guard, the axteach teacher
#      checkpoints) failed deterministically on the FIRST task and would have
#      been caught by exactly this.
#
#   2. MEASURE PEAK GPU MEMORY per cell, because the slot count is sized from
#      it. SLOTS=3 exists because 8 slots x 19.5GB OOM'd a 63GB card and burned
#      359 claims without running them. Guessing again would repeat that.
#
# Writes to runs_smoke/, NEVER runs/: a smoke run that wrote a real final.json
# would make train.py's completed-run guard SKIP the real cell later.

set -uo pipefail
source /gpfs/scratch/ub234/momentstem/env.sh
cd "$MS_ROOT/repo"

OUT=$MS_ROOT/runs_smoke
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# The ImageNet packs are single large memory-mapped files, not the many small
# files that make tin worth staging, so they are read straight from $MS_DATA.
DR=$MS_DATA

CELLS="
diagin64_r18_aux_1pct
diagin64_vit_aux_1pct
diagin64_mnet_aux_1pct
diagin100_vits_aux_1pct
diagin100_r50_aux_1pct
diagin100_vitb_aux_1pct
"

ok=0; fail=0
for c in $CELLS; do
    echo "=============== $c"
    # Sample GPU memory while the cell runs; report the peak it reaches.
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "/tmp/mem_$c.txt" 2>/dev/null &
    mon=$!
    if python train.py --config "configs/diagnostics/$c.yaml" --seed 0 \
            --epochs 1 --data-root "$DR" --out-root "$OUT" ; then
        ok=$((ok+1)); echo "SMOKE_OK $c"
    else
        fail=$((fail+1)); echo "SMOKE_FAIL $c"
    fi
    kill $mon 2>/dev/null
    peak=$(sort -n "/tmp/mem_$c.txt" 2>/dev/null | tail -1)
    echo "PEAK_MIB $c ${peak:-unknown}"
done

echo "SMOKE SUMMARY ok=$ok fail=$fail"
[ "$fail" -eq 0 ] && echo "INSMOKE_COMPLETE" || echo "INSMOKE_FAILED"
