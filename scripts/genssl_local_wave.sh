#!/bin/bash
# genssl LOCAL envelope extension (2026-07-23, user: "you can run them on the
# local machine as the GPU is idle. Why not all portions?").
#
# Track A on turing runs the decisive fractions (1/5/10%); this fills the tin
# SSL envelope at 2/3/7/15/25% on the local 3090 so the SSL-vs-aux margin
# SHAPE is measured, not extrapolated. tin@100% is EXCLUDED deliberately: the
# SimCLR pretrain alone is ~2x a tin@100% train (~week-scale on a 3090); it
# goes to turing after genssl if the envelope shape justifies it.
#
# PREDICTIONS RECORDED IN ADVANCE (shape claims; comparator aux cells at
# 3/7/15% are landing in filltin as this launches):
#   - SSL-init > baseline at every fraction from 2% up (SimCLR is not
#     data-starved at >=2000 imgs on tin, unlike ViT@1%).
#   - The SSL-aux margin is UNIMODAL in fraction (C100 precedent:
#     +0.85/+2.38/+2.32/+3.90/+4.99/+5.01/+4.09 peaked at 7-10%), peaking
#     mid-band on tin too, near-zero by 25% (aux itself is +0.10 there).
#   - FALSIFIER (same as Track A): SSL <= aux across 5-15% => conv-SSL
#     dominance is C100-specific.
#
# Conventions: setsid nohup, logs/genssl_local_wave.log, PID file, COMPLETE
# marker. num_workers=2 pinned in configs (tin convention).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=python

STAGE=/dev/shm/genssl_local
mkdir -p "$STAGE"; trap 'rm -rf "$STAGE"' EXIT
if [ ! -d "$STAGE/tiny-imagenet-200" ]; then
    echo "staging tin -> $STAGE"
    cp -r data/tiny-imagenet-200 "$STAGE/" 2>/dev/null || true
fi
ln -sfn "$(pwd)"/data/* "$STAGE"/ 2>/dev/null || true
DR=$STAGE

for PCT in 2 3 7 15 25; do
    for SEED in 0 1 2; do
        PRE=runs/simclr_pre_tin_${PCT}pct/seed${SEED}/pretrain.pt
        mkdir -p "$(dirname "$PRE")"
        echo "=== pretrain tin ${PCT}pct seed ${SEED} ==="
        flock -w 43200 "${PRE}.lock" -c \
          "[ -f $PRE ] || $PY scripts/simclr_pretrain.py --config configs/diagnostics/tin_none_${PCT}pct.yaml \
             --seed $SEED --out $PRE --data-root $DR" \
          || { echo "PRETRAIN FAIL ${PCT}/${SEED}"; continue; }
        echo "=== diagssl_tin_simclr_${PCT}pct seed ${SEED} ==="
        $PY train.py --config configs/diagnostics/diagssl_tin_simclr_${PCT}pct.yaml --seed $SEED --data-root "$DR" \
          && echo "OK ssl_tin ${PCT} ${SEED}" || echo "FAIL ssl_tin ${PCT} ${SEED}"
    done
    echo "GENSSL_LOCAL_PCT_DONE ${PCT}"
done

for PCT in 5 15 25; do
    $PY analysis/linear_probe.py --run runs/diagssl_tin_simclr_${PCT}pct \
        --config configs/diagnostics/diagssl_tin_simclr_${PCT}pct.yaml --data-root "$DR" \
        && echo "PROBE OK ${PCT}" || echo "PROBE FAIL ${PCT}"
done
echo "GENSSL_LOCAL_WAVE_COMPLETE"
