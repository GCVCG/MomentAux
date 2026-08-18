#!/usr/bin/env bash
# G probes for E2: is the fusion ordering feature-side or a readout artifact?
#
# The e2e result (2026-08-15) is that prior+SSL never beats the better single
# on either population, and prior+aug beats it in only ONE of six cells. The
# currency account explains that as "adding the prior at lambda0=1.0 costs
# whenever the other source is already strong" -- but that is an inference
# from e2e magnitudes, which is exactly what the probes exist to check. Until
# these run, nothing about E2 may be stated as feature-side.
#
# Probes every arm AND its comparators under one identical protocol, so the
# feature-side version of each e2e comparison can be formed:
#     probe(prior+aug) - probe(best single)   vs   its e2e counterpart
#     probe(prior+SSL) - probe(best single)   vs   its e2e counterpart
#
# All 30 cells are probed on the SAME machine to keep the protocol identical;
# a 3-seed probe measured 3:15 on food101, so the whole set is ~100 min and
# there is no reason to split it and introduce a machine term.
#
# The probe fits its head on the FULL train split while these cells trained on
# 5-25% of it, so the probe-ceiling rule is satisfied at every fraction here.
set -u
cd "$(dirname "$0")/.."
echo $$ > logs/e2probe.pid

PY=${PY:-~/venvs/momentstem/bin/python}

# comparator cells: (cell-stem, config-stem) -- configs live in configs/grid
COMPARATORS="
grid_esat_r18_e99fb4
grid_esat_r18_axmagnitudeL3_l10to00_hn_dd367d
diaggrid_simclr_esat
grid_food_r18_9ee7da
grid_food_r18_axmagnitudeL3_l10to00_hn_8bd74b
diaggrid_simclr_food
"

probe() {                     # $1 = cell, $2 = config path
  local cell=$1 cfg=$2
  if [ -f "runs/$cell/linear_probe.json" ]; then
    echo "--- SKIP $cell (already probed)"; return 0
  fi
  if [ ! -f "$cfg" ]; then echo "--- MISSING CONFIG $cfg"; return 1; fi
  if ! ls runs/$cell/seed*/best.pt >/dev/null 2>&1; then
    echo "--- NO CHECKPOINT $cell"; return 1; fi
  echo "--- $cell $(date -Is)"
  $PY analysis/linear_probe.py --run "runs/$cell" --config "$cfg" 2>&1
}

# E2 arms
for ds in esat food; do
  for arm in aug prioraug priorssl; do
    for p in 5 10 25; do
      c=diagfuse_${ds}_${arm}_${p}pct
      probe "$c" "configs/diagnostics/$c.yaml"
    done
  done
done

# comparators (baseline / prior / SSL)
for stem in $COMPARATORS; do
  for p in 5 10 25; do
    c=${stem}_${p}pct
    cfg=configs/grid/$c.yaml
    [ -f "$cfg" ] || cfg=configs/diagnostics/$c.yaml
    probe "$c" "$cfg"
  done
done

echo "E2PROBE_COMPLETE $(date -Is)"
