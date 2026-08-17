#!/bin/bash
#SBATCH --job-name=ms_c1smoke
#SBATCH --account=ub234
#SBATCH --qos=acc_resc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:25:00
#SBATCH --output=/gpfs/scratch/ub234/momentstem/logs/%x_%j.out
set -uo pipefail
source /gpfs/scratch/ub234/momentstem/env.sh
cd "$MS_ROOT/repo"
# SMOKE ONLY. --out-root is a SCRATCH tree, never $MS_ROOT/runs: a 1-epoch
# final.json in the real run dir is treated as a completed run by train.py's
# skip guard and would silently poison the cell.
SMOKE=$MS_ROOT/runs_c1smoke
mkdir -p "$SMOKE"
echo "host=$(hostname)"
FAIL=0
# one pathmnist cell (the dataset whose staging was in doubt) and one food cell
for cfg in grid_mnet_path_aux_20pct_l03 grid_mag3_food_50pct_l03; do
  echo "=== SMOKE $cfg ==="
  python train.py --config configs/diagnostics/$cfg.yaml --seed 0 \
      --data-root "$MS_DATA" --out-root "$SMOKE" --epochs 1 \
    && echo "SMOKE_OK $cfg" || { echo "SMOKE_FAIL $cfg"; FAIL=1; }
  python - <<PY
import json,os
p=os.path.join("$SMOKE","$cfg","seed0","final.json")
print("final.json:",p,"exists",os.path.exists(p))
if os.path.exists(p):
    d=json.load(open(p)); c=d.get("config",{})
    print("  lam0:",c.get("moment_aux",{}).get("weight"),"epochs:",c.get("epochs"),
          "dataset:",c.get("dataset"),"pct:",c.get("subset_pct"),
          "acc:",d.get("final_test_acc"))
PY
done
echo "C1_SMOKE_COMPLETE failures=$FAIL"
