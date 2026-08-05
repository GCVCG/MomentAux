#!/bin/bash
# Local G-probe wave (2026-08-05): the 69 gap probes that close the last
# unprobed paired cells -- ResNet-18 (20), ConvNeXt (22), ViT-tiny (22),
# R50/R34 (5).
#
# WHY LOCAL RATHER THAN BSC, and this is the substantive reason, not a
# scheduling preference: 40 of these 69 cells have NO checkpoint on BSC. They
# are original local-study cells (abl*, c10_*, diagcnxadamw_*, ...) that were
# never trained there, so the queued BSC probe tasks would have died on
# FileNotFoundError -- the axteach crash-loop signature, sixth incident of this
# campaign. 56 of the 69 have their checkpoints HERE already; the other 13 are
# a 2.7GB pull. Local can do 69/69, BSC at most 29/69.
# Meanwhile the 8 ms_grid jobs sit at QOSGrpNodeLimit behind the ImageNet big
# lane, so BSC is not merely slower for these -- it is blocked.
#
# CONVNEXT PROBE PATH was broken until today: timm ConvNeXt has NO top-level
# `global_pool` attribute at all, so linear_probe.py's bare attribute access
# raised AttributeError before any branch ran. Fixed by deferring to timm's
# forward_head(pre_logits=True), gated on the attribute's ABSENCE so every
# other family keeps the exact branch its recorded G was measured under.
# Verified: feat dim == classifier.in_features on all 5 families; suite 102/102.
#
# Probes do NOT train the backbone -- frozen features, one forward pass, then
# an LBFGS linear head -- so this is ~1 epoch-equivalent per cell, not 200.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=$HOME/venvs/momentstem/bin/python
LOG=logs/probelocal_wave.log
TASKS=${1:?usage: probelocal_wave.sh <tasklist>}

# num_workers is SAFE to tune for probes (shuffle=False, every feature
# extracted -- worker count cannot change the result, only CPU pressure).
# 16 cores here: 4 concurrent probes x (3 workers + main) = 16 threads.
export MS_PROBE_WORKERS=${MS_PROBE_WORKERS:-3}
CONC=${CONC:-4}
export OUT=runs DR=./data

N=$(wc -l < "$TASKS")
echo "=== probelocal wave start $(date -Is) tasks=$N conc=$CONC nw=$MS_PROBE_WORKERS" >> "$LOG"

ok=0; fail=0
run_one () {
    local i=$1 cmd=$2
    local name; name=$(echo "$cmd" | grep -o '\-\-run "\$OUT/[^"]*"' | sed 's|.*\$OUT/||; s|"$||')
    if eval "$cmd" >> "logs/probelocal_task_${i}.log" 2>&1; then
        echo "OK   $i/$N $name" >> "$LOG"
    else
        echo "FAIL $i/$N $name (logs/probelocal_task_${i}.log)" >> "$LOG"
    fi
}

i=0
while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    i=$((i+1))
    while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 5; done
    run_one "$i" "$cmd" &
done < "$TASKS"
wait

echo "=== probelocal counts: OK=$(grep -c '^OK ' "$LOG") FAIL=$(grep -c '^FAIL ' "$LOG")" >> "$LOG"
echo "PROBELOCAL_COMPLETE $(date -Is)" >> "$LOG"
