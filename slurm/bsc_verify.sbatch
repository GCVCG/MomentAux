#!/bin/bash
#SBATCH -J ms_verify
#SBATCH -o /gpfs/scratch/${CLUSTER_ACCOUNT}/momentstem/logs/ms_verify_%j.out
#SBATCH -t 12:00:00
#SBATCH -A ${CLUSTER_ACCOUNT}
#SBATCH -q acc_resc
#SBATCH -p acc
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -c 80
#
# Checkpoint IDENTITY audit over the whole cluster tree.  Verifying by LOADING
# is not enough -- a checkpoint can load perfectly and be the wrong network --
# so every probed cell's best.pt and last.pt is EVALUATED against its own
# recorded accuracy.
#
# Sharded across 2 nodes: submit twice with SHARD_LO=0 and SHARD_LO=8.
set -u
MS=/gpfs/scratch/${CLUSTER_ACCOUNT}/momentstem
cd $MS
source env.sh
mkdir -p $MS/verify

NSH=${NSH:-16}
LO=${SHARD_LO:-0}
HI=$(( LO + 8 ))

# Tiny-ImageNet lives as a ZIP and every other dataset as a directory, so the
# audit needs the SAME staging the training lanes use or every tin cell throws.
# (It used to throw, and before the classify-don't-raise fix that killed the
# whole shard on its first tin cell -- 114 of ~6,800 checkpoints examined,
# reported as a completed pass.)
STAGE=/dev/shm/msv_$SLURM_JOB_ID
if mkdir -p "$STAGE" 2>/dev/null; then
    ln -sfn "$MS/data"/* "$STAGE"/ 2>/dev/null
    rm -f "$STAGE/tiny-imagenet-200"
    if unzip -q "$MS/data/tiny-imagenet-200.zip" -d "$STAGE" 2>/dev/null; then
        DR=$STAGE
    else
        DR=$MS/data
    fi
    trap 'rm -rf "$STAGE"' EXIT
else
    DR=$MS/data
fi
echo "host=$(hostname) DR=$DR shards ${LO}..$((HI-1)) of $NSH"

export MS_VERIFY_WORKERS=4

run_pass () {                                  # $1 = best.pt | last.pt
    local ck=$1 tag=${1%%.*}
    for i in $(seq $LO $((HI-1))); do
        g=$(( (i - LO) % 4 ))
        CUDA_VISIBLE_DEVICES=$g python repo/analysis/verify_checkpoints.py \
            --all-probed --all-seeds --runs $MS/runs --data-root $DR \
            --ckpt $ck --tol 0.5 --shard $i/$NSH \
            --out $MS/verify/${tag}_shard${i}.json \
            > $MS/logs/verify_${tag}_${i}.log 2>&1 &
    done
    wait
    # A shard exits 1 when it FINDS damage, so the exit code cannot tell a
    # finding from a crash.  The report file can: assert every shard wrote one.
    local miss=0
    for i in $(seq $LO $((HI-1))); do
        [ -s "$MS/verify/${tag}_shard${i}.json" ] || { echo "MISSING ${tag}_shard${i}.json"; miss=1; }
    done
    [ $miss -eq 0 ] && echo "${tag^^} PASS OK" || echo "${tag^^} PASS INCOMPLETE"
    return $miss
}

rc=0
run_pass best.pt || rc=1
run_pass last.pt || rc=1
[ $rc -eq 0 ] && echo "VERIFY_COMPLETE" || echo "VERIFY_INCOMPLETE"
exit $rc
