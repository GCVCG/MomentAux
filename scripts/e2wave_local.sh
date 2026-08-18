#!/usr/bin/env bash
# E2: the missing fusion arms on the two non-selection populations
# (EuroSAT, Food-101) x {aug, prior+aug, prior+SSL} x {5,10,25%}, 3 seeds.
#
# Moved here from BSC on 2026-08-14: the queued jobs were estimated to start
# 2026-08-16, two days out. The prior+SSL arms need the SimCLR pretrains, which
# were pulled from BSC into runs/simclr_pre_{eurosat,food101}_*pct and verified
# by loading (18/18) and by hash (18 distinct).
#
# SLOTS=2 alongside the running refwave stream: measured 16 cores at load 3.7
# and 4.7/24 GB GPU, and every config pins num_workers=2, so 3 streams use
# ~9 cores. num_workers is part of the reproducibility contract and is NOT
# touched here -- concurrency changes wall-clock only, never the data or
# augmentation stream.
#
# train.py carries the completed-run guard (final.json => skip) and an
# exclusive per-run flock, so re-running this script is idempotent and it
# cannot race the other local wave.
set -u
cd "$(dirname "$0")/.."
echo $$ > logs/e2wave_local.pid

# PY is overridable because the second workstation keeps its venv beside the
# repo rather than under $HOME. Both are timm 1.0.27 / torch 2.7.0+cu126,
# checked rather than assumed.
PY=${PY:-~/venvs/momentstem/bin/python}
SLOTS=${SLOTS:-2}
CFG=configs/diagnostics

# cheapest population first; all three arms of a fraction stay adjacent so a
# partial drain never leaves an incomplete comparison.
# CELLS partitions the wave across machines. The per-run flock is NODE-LOCAL
# and these two machines share no filesystem, so nothing prevents both from
# running the same (config, seed): the partition is the ONLY thing keeping them
# disjoint. Keep it explicit and keep a cell's three seeds on one machine.
#
# The split is by MEASURED COST, not by dataset. Both machines cost
# ~5.2 ms/image/epoch (measured independently on each), so a cell's cost is
# images x 200 epochs. Splitting by dataset instead put 4,743 run-min of
# Food-101 on one GPU against 1,347 of EuroSAT on the other -- 40 h vs 8 h.
if [ -n "${CELLS:-}" ]; then
  :                                    # caller supplied an explicit list
else
  CELLS=""
  for ds in ${DATASETS:-esat food}; do
    for pct in 5 10 25; do
      for arm in aug prioraug priorssl; do
        CELLS="$CELLS diagfuse_${ds}_${arm}_${pct}pct"
      done
    done
  done
fi
echo "wave cells: $(echo $CELLS | wc -w)  slots: ${SLOTS:-2}"

run_one() {
  local cell=$1 seed=$2
  if [ -f "runs/$cell/seed$seed/final.json" ]; then
    echo "--- SKIP $cell seed$seed (already final)"; return 0
  fi
  echo "--- $cell seed$seed $(date -Is)"
  $PY train.py --config "$CFG/$cell.yaml" --seed "$seed" 2>&1
}

n=0
for cell in $CELLS; do
  for seed in 0 1 2; do
    run_one "$cell" "$seed" &
    n=$((n+1))
    if [ $((n % SLOTS)) -eq 0 ]; then wait; fi
  done
done
wait

echo "E2WAVE_LOCAL_COMPLETE $(date -Is)"
