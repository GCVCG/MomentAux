#!/bin/bash
#SBATCH --job-name=ms_dense3smoke
#SBATCH --account=ub234
#SBATCH --qos=acc_resc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:40:00
#SBATCH --output=/gpfs/scratch/ub234/momentstem/logs/%x_%j.out
#
# Pre-flight for the three NEW dense populations (ade20k, cityscapes,
# foodseg103), on the machine that will run them. Same discipline as the VOC
# pre-flight, and the checks are the same ones for the same reasons -- but
# this wave has more ways to go wrong, because all three datasets were staged
# on a different machine and shipped here.
#
# WHAT EACH CHECK IS FOR:
#   1. the staged trees on BSC pass the SAME verifier used locally. This is
#      the check that catches a truncated or partial rsync, and it is not
#      hypothetical: a nested-parent rsync silently dropped all 11,355 SBD
#      masks on 2026-08-10 while still printing DATA_SHIP_DONE.
#   2. the committed subsets reproduce from BSC's OWN copy of the data, so
#      both machines agree on which images each cell trains on. Without this
#      a cross-machine merge is not legitimate.
#   3. the joint-transform CONTENT tests pass under BSC's library versions,
#      including the new device-independence test for the confusion matrix.
#   4. BOTH ARMS of one cell per population train and write a final.json, and
#      the aux arm must log lam 1.000 -> 0.000. That single line is what
#      proves MomentAuxModel actually BUILT here rather than the cell quietly
#      training as a baseline -- the one failure that would produce a complete
#      but ONE-ARMED envelope.
#   5. peak memory per population, to size the slot count from measurement
#      rather than from a guess.
#
# Writes to runs_dense3_smoke/, never runs_dense/: a smoke final.json would
# make the completed-run guard skip the real cell.

set -uo pipefail
source /gpfs/scratch/ub234/momentstem/env.sh
cd "$MS_ROOT/repo"

OUT=$MS_ROOT/runs_dense3_smoke
mkdir -p "$OUT"
DR=$MS_DATA
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

fail=0
DATASETS="ade20k cityscapes foodseg103"

echo "=============== 1. the staged trees verify on BSC's own copy"
for ds in $DATASETS; do
    python scripts/prepare_seg_data.py --dataset "$ds" --verify --data-root "$DR" \
        || { echo "VERIFY_FAIL $ds"; fail=$((fail+1)); }
done

echo "=============== 2. committed subsets reproduce from BSC's data"
for ds in $DATASETS; do
    python scripts/make_dense_subsets.py --check --dataset "$ds" --data-root "$DR" \
        || { echo "SUBSET_FAIL $ds"; fail=$((fail+1)); }
done

echo "=============== 3. transform + confusion-matrix content tests"
python -m pytest tests/test_dense_transforms.py -q || fail=$((fail+1))

echo "=============== 4. both arms train, and the aux arm's lambda is live"
for ds in $DATASETS; do
    for arm in none aux; do
        c=${ds}_${arm}_1pct
        log=$OUT/${c}.log
        if python train_dense.py --config "configs/dense/$c.yaml" --seed 0 \
                --epochs 2 --eval-every 1 --data-root "$DR" --out-root "$OUT" \
                2>&1 | tee "$log"; then
            echo "SMOKE_OK $c"
        else
            echo "SMOKE_FAIL $c"; fail=$((fail+1))
        fi
    done
    # The aux arm must show the schedule running to exactly zero. Checked
    # rather than eyeballed, because a silently-baseline aux arm still writes
    # a perfectly good final.json.
    if grep -q "lam 1.000" "$OUT/${ds}_aux_1pct.log" \
       && grep -q "lam 0.000" "$OUT/${ds}_aux_1pct.log"; then
        echo "LAMBDA_OK $ds"
    else
        echo "LAMBDA_FAIL $ds -- aux arm did not run the schedule"; fail=$((fail+1))
    fi
done

echo "=============== 5. peak memory at the largest fraction, for slot sizing"
for ds in $DATASETS; do
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > /tmp/mem_$ds.txt 2>/dev/null &
    mon=$!
    python train_dense.py --config "configs/dense/${ds}_aux_25pct.yaml" --seed 0 \
        --epochs 1 --eval-every 1 --data-root "$DR" --out-root "$OUT" \
        && echo "SMOKE_OK ${ds}_aux_25pct" || { echo "SMOKE_FAIL ${ds}_aux_25pct"; fail=$((fail+1)); }
    kill $mon 2>/dev/null
    echo "PEAK_MIB $ds $(sort -n /tmp/mem_$ds.txt | tail -1)"
done

echo "=============== 6. dense probe path on a new population"
python analysis/dense_probe.py --cell ade20k_aux_1pct --seeds 0 \
    --runs "$OUT" --data-root "$DR" --epochs 1 \
    && echo "SMOKE_OK dense_probe" || { echo "SMOKE_FAIL dense_probe"; fail=$((fail+1)); }

echo "DENSE3 SMOKE SUMMARY failures=$fail"
[ "$fail" -eq 0 ] && echo "DENSE3_SMOKE_COMPLETE" || echo "DENSE3_SMOKE_FAILED"
