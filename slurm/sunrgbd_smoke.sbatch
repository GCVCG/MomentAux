#!/bin/bash
#SBATCH --job-name=ms_sunrgbdsmoke
#SBATCH --account=${CLUSTER_ACCOUNT:?set CLUSTER_ACCOUNT}
#SBATCH --qos=acc_resc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:25:00
#SBATCH --output=${CLUSTER_SCRATCH}/momentstem/logs/%x_%j.out
set -uo pipefail
source ${CLUSTER_SCRATCH}/momentstem/env.sh
cd "$MS_ROOT/repo"
# SMOKE ONLY. --out-root is a SCRATCH tree, never $MS_ROOT/runs: a 1-epoch
# final.json in the real run dir is treated as a completed run by train.py's
# skip guard and would silently poison the cell. Exits non-zero on any
# failure so the lane, submitted with --dependency=afterok, cannot start
# unless this passes (the sarred-lane exit-gate lesson).
SMOKE=$MS_ROOT/runs_sunrgbdsmoke
mkdir -p "$SMOKE"
echo "host=$(hostname)"
# data guard: the pack must be present and loadable before anything trains
python - <<PY
import numpy as np
z = np.load("$MS_DATA/sunrgbd_64_meta.npz")
x = np.load("$MS_DATA/sunrgbd_64_images.npy", mmap_mode="r")
assert x.shape == (9504, 4, 64, 64) and x.dtype == np.float32, x.shape
assert len(z["train_idx"]) == 4845 and len(z["test_idx"]) == 4659
print("PACK_OK", x.shape)
PY
[ $? -ne 0 ] && { echo "SMOKE_FAIL pack"; exit 1; }
FAIL=0
for spec in "configs/sensorfusion/sf_sunrgbd_all_aux_3pct.yaml" \
            "configs/sensorfusion/sf_sunrgbd_depth_none_3pct.yaml" \
            "configs/sensorfusion/diagsf_sunrgbd_rgb_vit_aux_10pct.yaml"; do
  cfg=$(basename "$spec" .yaml)
  echo "=== SMOKE $cfg ==="
  python train.py --config "$spec" --seed 0 \
      --data-root "$MS_DATA" --out-root "$SMOKE" --epochs 1 \
    && echo "SMOKE_OK $cfg" || { echo "SMOKE_FAIL $cfg"; FAIL=1; }
  python - <<PY
import json, os
p = os.path.join("$SMOKE", "$cfg", "seed0", "final.json")
print("final.json:", p, "exists", os.path.exists(p))
if os.path.exists(p):
    d = json.load(open(p)); c = d.get("config", {})
    print("  dataset:", c.get("dataset"), "pct:", c.get("subset_pct"),
          "aux:", bool(c.get("moment_aux")), "acc:", d.get("final_test_acc"))
PY
done
echo "SUNRGBD_SMOKE_COMPLETE failures=$FAIL"
exit $FAIL
