#!/bin/bash
#SBATCH --job-name=ms_r1smoke
#SBATCH --account=${CLUSTER_ACCOUNT:?set CLUSTER_ACCOUNT}
#SBATCH --qos=acc_resc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
#SBATCH --output=${CLUSTER_SCRATCH}/momentstem/logs/%x_%j.out
set -uo pipefail
source ${CLUSTER_SCRATCH}/momentstem/env.sh
cd "$MS_ROOT/repo"
# SMOKE ONLY. --out-root is a SCRATCH tree, never $MS_ROOT/runs: a 1-epoch
# final.json in the real run dir is treated as a completed run by train.py's
# skip guard and would silently poison the cell.
SMOKE=$MS_ROOT/runs_r1smoke
mkdir -p "$SMOKE"
echo "host=$(hostname)"
FAIL=0
# The AUX arm: the more complex path (moment_aux, tap blocks.8, head_norm) and
# the one that could fail on a vit_small tap. The none arm differs only by the
# absence of that block.
for cfg in diagin100e200_vits_aux; do
  echo "=== SMOKE $cfg ==="
  python train.py --config configs/diagnostics/$cfg.yaml --seed 0 \
      --data-root "$MS_DATA" --out-root "$SMOKE" --epochs 1 \
    && echo "SMOKE_OK $cfg" || { echo "SMOKE_FAIL $cfg"; FAIL=1; }
  python - <<PY
import json, os, yaml
p = os.path.join("$SMOKE", "$cfg", "seed0", "final.json")
print("final.json:", p, "exists", os.path.exists(p))
if os.path.exists(p):
    d = json.load(open(p)); c = d.get("config", {})
    print("  RECORDED cfg -> backbone:", c.get("backbone"),
          "dataset:", c.get("dataset"), "pct:", c.get("subset_pct"),
          "augment:", c.get("augment"), "optimizer:", c.get("optimizer"))
    print("  epochs(recorded):", c.get("epochs"),
          "epochs_overridden:", c.get("epochs_overridden"))
    print("  moment_aux:", c.get("moment_aux"))
    print("  params_trainable:", d.get("accounting", {}).get("params_trainable"),
          "image_size:", d.get("accounting", {}).get("flops_image_size"))
    print("  acc:", d.get("final_test_acc"))
# The recorded epochs is 1 BY CONSTRUCTION (--epochs 1 is a smoke override and
# train.py stamps epochs_overridden). What must be 200 is the SHIPPED FILE, so
# assert that separately -- that is the cell the wave will actually run.
f = yaml.safe_load(open("configs/diagnostics/$cfg.yaml"))
print("  SHIPPED FILE -> name:", f["name"], "backbone:", f["backbone"],
      "epochs:", f["epochs"])
assert f["epochs"] == 200 and f["backbone"] == "vit_small", "WRONG CELL"
print("  FILE ASSERT OK: epochs 200, vit_small")
PY
done
echo "R1_SMOKE_COMPLETE failures=$FAIL"
