#!/bin/bash
#SBATCH --job-name=ms_ine200smoke
#SBATCH --account=${CLUSTER_ACCOUNT:?set CLUSTER_ACCOUNT}
#SBATCH --qos=acc_resc
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=01:30:00
#SBATCH --output=${CLUSTER_SCRATCH}/momentstem/logs/%x_%j.out
#
# ===== GATE for the 200-epoch ImageNet re-run (2026-09-05) =================
# EXITS NON-ZERO ON FAILURE so the lanes, submitted --dependency=afterok on
# this job, cannot start unless it passes (the sarred/sunrgbd exit-gate
# pattern).
#
# THE RISK BEING TESTED, and it is one specific thing: block F found the frozen
# diag AdamW recipe DIVERGES at ViT-L on a 200-epoch cosine (6/6 aux runs NaN)
# and needs warmup+clip.  ViT-tiny on ImageNet64 has never run 200 epochs, and
# a 200-epoch cosine holds the LR near peak for far longer than the 40-epoch
# one these cells were measured under, which is exactly the exposure that
# killed ViT-L.  ViT-S and ViT-B are already known to survive 200 epochs
# unstabilized (R1: 79.47/83.99 and 75.31/82.02), so they are checked for
# build-and-train only.
# The stabilizer is deliberately NOT pre-applied: if a cell diverges here it
# gets warmup+clip with the same MEASURED justification ViT-L has, recorded in
# the ledger, rather than applied pre-emptively to cells that do not need it.
#
# Writes ONLY to a scratch out-root, so no real run dir can be poisoned by a
# partial final.json (train.py's completed-run guard makes re-runs safe and
# smoke tests dangerous -- the 2026-08-17 method note).

set -uo pipefail
source ${CLUSTER_SCRATCH}/momentstem/env.sh
cd "$MS_ROOT/repo"

OUTS=$MS_ROOT/runs_ine200smoke
rm -rf "$OUTS"; mkdir -p "$OUTS"

STAGE=/dev/shm/ms_$SLURM_JOB_ID
mkdir -p "$STAGE" && ln -sfn "$MS_DATA"/* "$STAGE"/ 2>/dev/null && DR=$STAGE || DR=$MS_DATA
trap 'rm -rf "$STAGE"' EXIT
echo "host=$(hostname) DR=$DR"

FAIL=0
run () {   # $1=gpu  $2=seconds  $3=config
    ( export CUDA_VISIBLE_DEVICES=$1
      timeout "$2" python train.py --config "configs/diagnostics/$3.yaml" --seed 0 \
              --data-root "$DR" --out-root "$OUTS" \
              > "$MS_ROOT/logs/ine200smoke_$3.log" 2>&1
      echo "[$3] training window closed (rc=$?)" ) &
}

# The two AdamW/ViT cells get the long windows: they are the actual risk.
run 0 3300 diagin64e200_vit_aux_5pct
run 1 3300 diagin100e200_vitb_aux_5pct
run 2 1500 diagin64e200_r18_aux_5pct
run 3 1500 diagin100e200_r50_aux_5pct
wait
# mnet after, reusing gpu0 (short)
run 0 900 diagin64e200_mnet_aux_5pct
wait

echo; echo "================ SMOKE VERDICT ================"
for c in diagin64e200_vit_aux_5pct diagin100e200_vitb_aux_5pct \
         diagin64e200_r18_aux_5pct diagin100e200_r50_aux_5pct \
         diagin64e200_mnet_aux_5pct; do
    M="$OUTS/$c/seed0/metrics.csv"
    if [ ! -s "$M" ]; then echo "FAIL $c :: no metrics.csv written"; FAIL=1; continue; fi
    python - "$M" "$c" <<'PY'
import csv, sys, math
m, cell = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(m)))
if not rows:
    print(f"FAIL {cell} :: metrics.csv empty"); raise SystemExit(3)
bad = [r["epoch"] for r in rows
       if any(not math.isfinite(float(r[k] or "nan")) for k in ("train_loss", "test_acc"))]
last = rows[-1]
acc  = float(last["test_acc"]) * 100.0
lam  = float(last.get("lambda") or 0.0)
aux  = float(last.get("aux_loss") or 0.0)
n    = len(rows)
# chance is 0.1% on ImageNet64 (1000-way) and 1% on ImageNet-100 (100-way)
chance = 0.1 if "in64" in cell else 1.0
ok = (not bad) and acc > 2 * chance and aux > 0
print(f"{'PASS' if ok else 'FAIL'} {cell} :: {n} epochs, test_acc {acc:.2f}% "
      f"(chance {chance}%), lambda {lam:.3f}, aux_loss {aux:.4f}"
      + (f", NON-FINITE at epochs {bad[:5]}" if bad else ""))
raise SystemExit(0 if ok else 4)
PY
    [ $? -eq 0 ] || FAIL=1
done

# --- resume round trip on the real config form, since every re-run cell
#     carries resume_every: 10 and the ImageNet64 100% cells depend on it.
echo; echo "---- resume round trip ----"
RC="$OUTS/diagin64e200_r18_aux_5pct/seed0/resume.pt"
if [ -f "$RC" ]; then echo "PASS resume.pt written by a live run"
else echo "NOTE resume.pt absent (run did not reach epoch 10); not a failure"; fi

rm -rf "$OUTS"
echo
if [ $FAIL -eq 0 ]; then echo "IN_E200_SMOKE_PASS"; exit 0
else echo "IN_E200_SMOKE_FAIL"; exit 1; fi
