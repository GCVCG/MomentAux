#!/bin/bash
#SBATCH --job-name=ms_vitlsmoke
#SBATCH --account=${CLUSTER_ACCOUNT:?set CLUSTER_ACCOUNT}
#SBATCH --qos=acc_resc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:40:00
#SBATCH --output=${CLUSTER_SCRATCH}/momentstem/logs/%x_%j.out
#
# Block F pre-flight (2026-08-23). SMOKE ONLY -- everything goes to a scratch
# --out-root, never $MS_ROOT/runs (a smoke final.json in the real run dir
# would make train.py's completed-run guard SKIP the real cell).
#  (1) 1-epoch diagin100_vitl_aux at the REAL batch size 128: proves ViT-L/16
#      + aux tap blocks.16 builds and trains on an H100 and MEASURES peak GPU
#      memory (nvidia-smi polling) and s/epoch. If it OOMs we STOP and report;
#      the batch size is part of the recipe and is not changed silently.
#  (2) the RESUME path on the real model/GPFS: a copy of the same config with
#      resume_every 1 is run for --epochs 3, KILLED after resume.pt exists and
#      epoch 2's metrics row is written, then started again: it must resume at
#      epoch 2, finish epoch 3, write final.json with resumed_from_epoch 2, and
#      clean up resume.pt. This is the mechanism the whole lane depends on.
set -uo pipefail
source ${CLUSTER_SCRATCH}/momentstem/env.sh
cd "$MS_ROOT/repo"
SMOKE=$MS_ROOT/runs_smoke/F
mkdir -p "$SMOKE"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
DR=$MS_DATA
echo "host=$(hostname)"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
FAIL=0

echo "=============== (1) 1-epoch diagin100_vitl_aux, batch 128"
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > "$SMOKE/mem_vitl.txt" 2>/dev/null &
mon=$!
if python train.py --config configs/diagnostics/diagin100_vitl_aux.yaml --seed 0 \
        --epochs 1 --data-root "$DR" --out-root "$SMOKE"; then
    echo "SMOKE_OK diagin100_vitl_aux"
else
    echo "SMOKE_FAIL diagin100_vitl_aux"; FAIL=1
fi
kill $mon 2>/dev/null
echo "PEAK_MIB_nvidia_smi $(sort -n "$SMOKE/mem_vitl.txt" | tail -1)"
echo "metrics.csv:"; cat "$SMOKE/diagin100_vitl_aux/seed0/metrics.csv"
python - <<PY
import json,os
p="$SMOKE/diagin100_vitl_aux/seed0/final.json"
if os.path.exists(p):
    d=json.load(open(p)); c=d["config"]
    print("final.json -> backbone", c["backbone"], "tap", c["moment_aux"]["tap"], "epochs(recorded)", c["epochs"],
          "overridden", c.get("epochs_overridden"), "resume_every", c.get("resume_every"),
          "params", d["accounting"]["params_trainable"], "acc", d["final_test_acc"], "wall", round(d["wall_seconds"]))
PY

if [ "$FAIL" -eq 0 ]; then
echo "=============== (2) resume round trip on the real model (epochs 3, resume_every 1, kill after epoch 2)"
CFG=$SMOKE/diagsmokeF_vitl_resume.yaml
sed -e 's/^name: diagin100_vitl_aux/name: diagsmokeF_vitl_resume/' -e 's/^resume_every: 10/resume_every: 1/' \
    configs/diagnostics/diagin100_vitl_aux.yaml > "$CFG"
grep -n "^name:\|^resume_every:" "$CFG"
D=$SMOKE/diagsmokeF_vitl_resume/seed0
rm -rf "$D"
python -u train.py --config "$CFG" --seed 0 --epochs 3 --data-root "$DR" --out-root "$SMOKE" > "$SMOKE/resume_run1.log" 2>&1 &
P=$!
t=0
while kill -0 $P 2>/dev/null; do
    # wait for the EPOCH-2 save specifically (the log line is printed after the
    # atomic os.replace), so run 2 must resume from exactly epoch 2
    if grep -q "resume state saved at epoch 2" "$SMOKE/resume_run1.log" 2>/dev/null; then
        echo "epoch-2 resume state saved at t=${t}s -> killing run 1"
        kill -9 $P; sleep 3; break
    fi
    sleep 10; t=$((t+10))
done
wait $P 2>/dev/null
tail -4 "$SMOKE/resume_run1.log"
ls -la "$D"
echo "--- run 2 (must resume at epoch 2)"
python train.py --config "$CFG" --seed 0 --epochs 3 --data-root "$DR" --out-root "$SMOKE" 2>&1 | grep -v Warning | tail -8
echo "metrics.csv after resume:"; cat "$D/metrics.csv"
python - <<PY
import json,os
p="$D/final.json"
ok = os.path.exists(p)
if ok:
    d=json.load(open(p))
    print("final.json: epochs", d["config"]["epochs"], "resumed_from_epoch", d.get("resumed_from_epoch"),
          "history", d.get("resume_history"), "best", d["best_test_acc"], "final", d["final_test_acc"])
    ok = d.get("resumed_from_epoch") == 2 and d["config"]["epochs"] == 3
print("resume.pt cleaned up:", not os.path.exists("$D/resume.pt"))
print("RESUME_SMOKE_" + ("OK" if ok and not os.path.exists("$D/resume.pt") else "FAIL"))
PY
fi
echo "VITL_SMOKE_COMPLETE failures=$FAIL"
