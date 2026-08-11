#!/usr/bin/env bash
# Run the tin@5% ViT budget cell once the probe queue has drained, so the
# local GPU does not sit idle between jobs.
set -u
cd "$(dirname "$0")/.."
while pgrep -f 'linear_probe\.py' > /dev/null; do sleep 60; done
bash scripts/vitbudget_tin5.sh
echo "$(date -Is) TIN5_COMPLETE" >> logs/vitbudget_tin5.log
