#!/bin/bash
#SBATCH --job-name=ms_denssmoke
#SBATCH --account=ub234
#SBATCH --qos=acc_resc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/scratch/ub234/momentstem/logs/%x_%j.out
#
# Pre-flight for the VOC segmentation envelope, on the machine that will run
# it. Same discipline as the ImageNet pre-flight, for the same reason: every
# wave in this study that crash-looped did so on its FIRST task, on something
# a one-epoch run would have caught (a dataset whitelist, a missing teacher
# checkpoint, a probe path).
#
# There is more to check here than usual, because this is a NEW task family:
#   1. both arms train and write a final.json
#   2. the committed subsets reproduce from BSC's own copy of the data --
#      if the two machines' VOC/SBD copies differ, every cell trains on
#      different images than it claims to
#   3. the joint-transform content tests pass under BSC's library versions
#   4. the dense probe runs (it is a different code path from training)
#
# Writes to runs_dense_smoke/, never runs_dense/: a smoke final.json would
# make the completed-run guard skip the real cell.

set -uo pipefail
source /gpfs/scratch/ub234/momentstem/env.sh
cd "$MS_ROOT/repo"

OUT=$MS_ROOT/runs_dense_smoke
mkdir -p "$OUT"
DR=$MS_DATA
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

fail=0

echo "=============== 1. subsets reproduce from BSC's own data"
python scripts/make_dense_subsets.py --check --data-root "$DR" || fail=$((fail+1))

echo "=============== 2. joint-transform content tests"
python -m pytest tests/test_dense_transforms.py -q || fail=$((fail+1))

echo "=============== 3. both arms train"
for c in voc_none_1pct voc_aux_1pct; do
    if python train_dense.py --config "configs/dense/$c.yaml" --seed 0 \
            --epochs 2 --eval-every 1 --data-root "$DR" --out-root "$OUT"; then
        echo "SMOKE_OK $c"
    else
        echo "SMOKE_FAIL $c"; fail=$((fail+1))
    fi
done

echo "=============== 4. peak memory, for slot sizing"
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > /tmp/dense_mem.txt 2>/dev/null &
mon=$!
python train_dense.py --config configs/dense/voc_aux_25pct.yaml --seed 0 \
    --epochs 1 --eval-every 1 --data-root "$DR" --out-root "$OUT" \
    && echo "SMOKE_OK voc_aux_25pct" || { echo "SMOKE_FAIL voc_aux_25pct"; fail=$((fail+1)); }
kill $mon 2>/dev/null
echo "PEAK_MIB $(sort -n /tmp/dense_mem.txt | tail -1)"

echo "=============== 5. dense probe path"
python analysis/dense_probe.py --cell voc_aux_1pct --seeds 0 \
    --runs "$OUT" --data-root "$DR" --epochs 1 \
    && echo "SMOKE_OK dense_probe" || { echo "SMOKE_FAIL dense_probe"; fail=$((fail+1)); }

echo "DENSE SMOKE SUMMARY failures=$fail"
[ "$fail" -eq 0 ] && echo "DENSE_SMOKE_COMPLETE" || echo "DENSE_SMOKE_FAILED"
