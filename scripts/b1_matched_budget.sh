#!/usr/bin/env bash
# B1: does the readout term's negative branch survive a MATCHED label budget?
#
# The referee's top blocking issue. The linear evaluation normally fits on the
# full train split while the cell trained on 1-25% of it, so below the crossing
# a negative residual may follow from the protocol rather than from learning.
# This re-probes BOTH arms of every selected cell at that cell's OWN per-class
# budget, so probe and cell see the same number of labels.
#
# --shots-only is deliberate: it writes linear_probe_SHOTS.json and leaves
# linear_probe.json untouched, so no recorded G in the ledger can be
# overwritten. A smoke run once clobbered abl1_none/linear_probe.json that way.
#
# Cell list and per-cell budgets come from /tmp/b1_cells.json, built from the
# exporter (champion config only: energy-magnitude, tap layer3, lambda 1.0->0).
set -u
cd "$(dirname "$0")/.."
echo $$ > logs/b1.pid

PY=${PY:-~/venvs/momentstem/bin/python}
CELLS=${CELLS:-/tmp/b1_cells.json}

$PY - "$CELLS" <<'PYEOF' > /tmp/b1_tasks.txt
import json,sys,glob,os
for c in json.load(open(sys.argv[1])):
    for role in ("cell","base_cell"):
        name=c[role]
        cfg=None
        for d in ("configs/grid","configs/diagnostics","configs","configs/ablations_full","configs/ablations"):
            p=os.path.join(d,name+".yaml")
            if os.path.exists(p): cfg=p; break
        if cfg is None:
            hits=glob.glob(f"configs/**/{name}.yaml",recursive=True)
            cfg=hits[0] if hits else ""
        print(f"{name}\t{cfg}\t{c['shots']}")
PYEOF

total=$(wc -l < /tmp/b1_tasks.txt); i=0; miss=0
echo "b1: $total probe tasks (both arms of each cell, at the cell's own budget)"
while IFS=$'\t' read -r cell cfg shots; do
  i=$((i+1))
  if [ -z "$cfg" ]; then echo "--- [$i/$total] NO CONFIG $cell"; miss=$((miss+1)); continue; fi
  if [ -f "runs/$cell/linear_probe_SHOTS.json" ]; then
    echo "--- [$i/$total] SKIP $cell (already)"; continue; fi
  echo "--- [$i/$total] $cell  shots=$shots"
  $PY analysis/linear_probe.py --run "runs/$cell" --config "$cfg" \
      --shots "$shots" --shots-only 2>&1
done < /tmp/b1_tasks.txt

echo "B1_COMPLETE missing_config=$miss $(date -Is)"
