#!/bin/bash
#SBATCH --job-name=widthoctave_wave
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#
# Width-vs-octave DECISIVE wave (2026-07-21, v2 after the v1 config-sync
# failure). The octave excess is real (auxmag3 +0.49, 4.4sigma at 10v10) but
# the width-matched control auxmag6o hit +0.46 at 3 seeds WITHOUT new
# frequency content -- so the driver may be target WIDTH, not the octave.
# Deepen auxmag6o_tin_5pct and tin_none_5pct 3->10 for the decisive 10v10
# vs auxmag3 (already 10 seeds, local).
#
# NOTE seeds 0-2 of every tin@5% cell live on the LOCAL 3090; turing has
# none. This job trains seeds 3..9 only; the 10-seed merge and ALL probes
# happen locally so probing is consistent on one machine (cross-machine
# seed mixing is precedented -- tinsuper's deepening was 7 turing seeds --
# and the config pins num_workers, so each seed's augmentation stream is
# machine-independent).
#
# FAST-PATH: tin's 110k small JPEGs are BeeGFS-IO-bound (~17s/epoch).
# Stage the 248MB zip -> /dev/shm (RAM tmpfs, ~8s) so tin runs at CIFAR
# speed. Byte-identical pixels, same augmentation -> protocol-neutral.

set -euo pipefail
PY=${CLUSTER_SCRATCH}/envs/momentstem/bin/python
REPO=${CLUSTER_SCRATCH}/projects/MomentsCNNEncoder
cd "$REPO"
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

STAGE=/dev/shm/momentstem_${SLURM_JOB_ID}
mkdir -p "$STAGE"; trap 'rm -rf "$STAGE"' EXIT
echo "staging tin -> $STAGE"; time unzip -q "$REPO/data/tiny-imagenet-200.zip" -d "$STAGE"
DR="$STAGE"
echo "staged: $(ls "$STAGE/tiny-imagenet-200" | tr '\n' ' ')"

for CELL in auxmag6o_tin_5pct tin_none_5pct; do
  for SEED in 3 4 5 6 7 8 9; do
    echo "=== ${CELL} seed ${SEED} ==="
    $PY train.py --config configs/diagnostics/${CELL}.yaml --seed $SEED --data-root "$DR"
  done
  echo "WIDTHOCTAVE_CELL_DONE ${CELL}"
done
echo "WIDTHOCTAVE_WAVE_COMPLETE"
