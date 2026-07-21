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
# Width-vs-octave DECISIVE wave (2026-07-21): the octave excess is real
# (auxmag3 +0.49, 4.4sigma at 10v10) but the width-matched control auxmag6o
# hit +0.46 at 3 seeds WITHOUT new frequency content -- so the driver may be
# target WIDTH, not the octave. Deepen auxmag6o_tin_5pct 3->10 for the
# decisive 10v10 vs auxmag3 (already 10 seeds); deepen tin_none_5pct 3->10
# for a 10-seed baseline. + probes.
# FAST-PATH: tin's 110k small JPEGs make it BeeGFS-IO-bound (~17s/epoch).
# Stage the 248MB zip -> /dev/shm (RAM tmpfs, ~8s one-time) so tin runs at
# CIFAR speed. Byte-identical pixels, same augmentation -> protocol-neutral.

set -euo pipefail
PY=/mnt/beegfs/amughrabi/envs/momentstem/bin/python
REPO=/mnt/beegfs/amughrabi/projects/MomentsCNNEncoder
cd "$REPO"
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# --- stage tin into RAM ---
STAGE=/dev/shm/momentstem_${SLURM_JOB_ID}
mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
echo "staging tin -> $STAGE"; time unzip -q "$REPO/data/tiny-imagenet-200.zip" -d "$STAGE"
DR="$STAGE"   # tin_root(DR) = DR/tiny-imagenet-200; subsets read from repo always
echo "staged: $(ls "$STAGE/tiny-imagenet-200" | tr '\n' ' ')"

declare -A CFG=(
  [auxmag6o_tin_5pct]=configs/diagnostics/auxmag6o_tin_5pct.yaml
  [tin_none_5pct]=configs/diagnostics/tin_none_5pct.yaml
)
for CELL in auxmag6o_tin_5pct tin_none_5pct; do
  for SEED in 3 4 5 6 7 8 9; do
    echo "=== ${CELL} seed ${SEED} ==="
    $PY train.py --config "${CFG[$CELL]}" --seed $SEED --data-root "$DR"
  done
  echo "WIDTHOCTAVE_CELL_DONE ${CELL}"
done

for CELL in auxmag6o_tin_5pct tin_none_5pct auxmag3_tin_5pct tin_aux_5pct; do
  CF=$(ls configs/*/${CELL}.yaml configs/${CELL}.yaml 2>/dev/null | head -1)
  $PY analysis/linear_probe.py --run runs/${CELL} --config "$CF" --data-root "$DR"
  echo "WIDTHOCTAVE_PROBE_DONE ${CELL}"
done
echo "WIDTHOCTAVE_WAVE_COMPLETE"
