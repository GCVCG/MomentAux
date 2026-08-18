#!/bin/bash
#SBATCH --job-name=ms_detdecomp
#SBATCH --account=${CLUSTER_ACCOUNT:?set CLUSTER_ACCOUNT}
#SBATCH --qos=acc_resc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=06:00:00
#SBATCH --output=${CLUSTER_SCRATCH}/momentstem/logs/%x_%j.out
#
# The detection decomposition (2026-08-13). AP50 collapses at 1-2% data and the
# pre-declared floor rule makes those cells uninterpretable; fg_acc and fg_iou,
# both conditioned on ground-truth foreground, do not collapse. See
# analysis/det_decompose.py for why this is a decomposition rather than a
# metric swap, and CLAUDE.md for the T5/T6 bands recorded before it ran.
#
# EVALUATION ONLY -- reads checkpoints, trains nothing, writes no run record.
#
# ONE GPU AND ONE PROCESS, deliberately. Every cell must be measured under an
# identical protocol for its delta to mean anything, and the cheapest way to
# guarantee that is to measure them all in one process on one device. The pass
# is ~2 h; sharding it across 4 GPUs would save 90 minutes and introduce a
# device axis into a comparison whose whole content is a difference.
#
# SUBMIT WITH A DEPENDENCY on the training jobs, so it cannot read a
# half-written last.pt:
#   sbatch --dependency=afterany:<ms_det jobid>:<ms_detfix jobid> \
#          slurm/bsc_detdecomp.sbatch

set -uo pipefail
source ${CLUSTER_SCRATCH}/momentstem/env.sh
cd "$MS_ROOT/repo"

OUT=$MS_ROOT/results/det_decompose.json
mkdir -p "$(dirname "$OUT")"

# last.pt, not best.pt: final.json's recorded AP50/fg_acc are FINAL-epoch
# values while best.pt is whichever epoch scored best, and the two differ
# materially (0.31 against 0.44 on vocdet_none_1pct/seed0). AP50/loc_acc/fg_acc
# are recomputed in the same pass, so every number comes from one checkpoint --
# verified locally to reproduce final.json exactly (AP50 0.31, fg_acc 15.48).
python analysis/det_decompose.py \
    --runs "$MS_ROOT/runs_det" \
    --configs configs/det \
    --data-root "$MS_DATA" \
    --ckpt last.pt \
    --ap25 \
    --out "$OUT"
rc=$?

if [ $rc -eq 0 ]; then
    echo "DET_DECOMP_COMPLETE  $(python -c "import json,sys;print(len(json.load(open('$OUT'))),'runs')")"
else
    echo "DET_DECOMP_FAILED rc=$rc"
fi
exit $rc
