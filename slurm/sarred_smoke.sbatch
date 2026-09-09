#!/bin/bash
#SBATCH --job-name=ms_sarredsmoke
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
# skip guard and would silently poison the cell.
SMOKE=$MS_ROOT/runs_sarredsmoke
mkdir -p "$SMOKE"
echo "host=$(hostname)"
FAIL=0
# SAR channel-reduction smoke: pca1 (new buffers) + perchannel (batch fold), 1 epoch each
for cfg in diagsarred_pca1_so2sat_sar_aux_5pct diagsarred_perchannel_so2sat_sar_aux_5pct; do
  echo "=== SMOKE $cfg ==="
  python train.py --config configs/sensorfusion/$cfg.yaml --seed 0 \
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
echo "SARRED_SMOKE_COMPLETE failures=$FAIL"
exit $FAIL   # non-zero on any failure so the dependent lane (afterok) cannot start
