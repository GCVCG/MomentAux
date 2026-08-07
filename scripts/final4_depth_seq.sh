#!/bin/bash
# The two 64px depth cells, SEQUENTIALLY alongside the running ImageNet-100
# cell (measured: in100 R50 ~8GB + one depth R50 ~7.4GB fits 23.5GB; three
# did not). expandable_segments reduces fragmentation when sharing a card.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/venvs/momentstem/bin/python
for spec in "diagdepth_r50_path_aux_25pct 1" "diagdepth_r50_food_aux_25pct 0"; do
  set -- $spec
  $PY train.py --config configs/diagnostics/$1.yaml --seed $2 \
      --data-root ./data --out-root runs > logs/final4/$1_seed$2.log 2>&1 \
    && echo "OK $1 seed$2" >> logs/final4/wave.log \
    || echo "FAIL $1 seed$2" >> logs/final4/wave.log
done
echo "DEPTH_SEQ_COMPLETE $(date -Is)" >> logs/final4/wave.log
