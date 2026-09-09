#!/bin/bash
#SBATCH --job-name=ms_recsmoke
#SBATCH --account=${CLUSTER_ACCOUNT:?set CLUSTER_ACCOUNT}
#SBATCH --qos=acc_resc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:45:00
#SBATCH --output=${CLUSTER_SCRATCH}/momentstem/logs/%x_%j.out
#
# NODE_FAIL RECOVERY SMOKE (2026-08-24). A cluster-wide NODE_FAIL at 10:36
# killed every running job, and the two dependent lanes that fired afterwards
# (ms_vitl2/ms_vitl3) died in 15s without even creating their stdout files.
# So before four probe lanes are launched, ONE cheap job proves (a) the node,
# GPFS and env are usable again and (b) each probe CODE PATH constructs the
# model, loads a real checkpoint and extracts features -- including the two
# paths that have never run on BSC: the SUN RGB-D 1-/4-channel probes and the
# pca1 channel-reduction probe (which registers extra buffers).
# Every check writes to a THROWAWAY filename (--out-suffix _SMOKETEST) or a
# scratch runs tree, so no recorded probe output can be touched, and the
# outputs are deleted at the end. --shots-only keeps each check to feature
# extraction (seconds-to-minutes) instead of the full LBFGS fit.
set -uo pipefail
source ${CLUSTER_SCRATCH}/momentstem/env.sh
cd "$MS_ROOT/repo"
echo "host=$(hostname) date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
FAIL=0

run () {  # $1 = label, rest = command
    lbl=$1; shift
    echo "=== SMOKE $lbl ==="
    if "$@"; then echo "SMOKE_OK $lbl"; else echo "SMOKE_FAIL $lbl"; FAIL=1; fi
}

# 1. block-A trajectory probe path: an epoch checkpoint + --out-suffix.
run trajprobe_mnet_path python analysis/linear_probe.py \
    --run "$MS_ROOT/runs/diagtraj_grid_mnet_path_none_20pct" \
    --config configs/diagnostics/diagtraj_grid_mnet_path_none_20pct.yaml \
    --data-root "$MS_DATA" --ckpt ckpt_ep020.pt \
    --shots 25 --shots-only --out-suffix _SMOKETEST

# 2. SUN RGB-D fused source: 4-channel conv1, never probed before.
run sunrgbd_all_r18 python analysis/linear_probe.py \
    --run "$MS_ROOT/runs/sf_sunrgbd_all_aux_5pct" \
    --config configs/sensorfusion/sf_sunrgbd_all_aux_5pct.yaml \
    --data-root "$MS_DATA" --ckpt best.pt \
    --shots 25 --shots-only --out-suffix _SMOKETEST

# 3. SUN RGB-D depth on ViT-tiny: 1-channel input AND the ViT extraction path.
run sunrgbd_depth_vit python analysis/linear_probe.py \
    --run "$MS_ROOT/runs/diagsf_sunrgbd_depth_vit_aux_10pct" \
    --config configs/sensorfusion/diagsf_sunrgbd_depth_vit_aux_10pct.yaml \
    --data-root "$MS_DATA" --ckpt best.pt \
    --shots 25 --shots-only --out-suffix _SMOKETEST

# 4. block-G pca1 reduction: the checkpoint carries pc1_w / pc1_fitted buffers,
#    so this is the one that would fail on a state_dict mismatch.
run sarred_pca1 python analysis/linear_probe.py \
    --run "$MS_ROOT/runs/diagsarred_pca1_so2sat_sar_aux_5pct" \
    --config configs/sensorfusion/diagsarred_pca1_so2sat_sar_aux_5pct.yaml \
    --data-root "$MS_DATA" --ckpt best.pt \
    --shots 25 --shots-only --out-suffix _SMOKETEST

# 5. block-D dense probe path, into a SCRATCH runs tree so the real
#    dense_probe.json cannot be written with a 1-epoch number.
SD=$MS_ROOT/runs_densesmoke
rm -rf "$SD"; mkdir -p "$SD/diagstep_ade_none_5pct/seed0"
ln -sfn "$MS_ROOT/runs_dense/diagstep_ade_none_5pct/seed0/best.pt" \
        "$SD/diagstep_ade_none_5pct/seed0/best.pt"
run dense_ade python analysis/dense_probe.py --cell diagstep_ade_none_5pct \
    --seeds 0 --runs "$SD" --data-root "$MS_DATA" --epochs 1

# --- clean up every throwaway artefact ------------------------------------
rm -f "$MS_ROOT"/runs/diagtraj_grid_mnet_path_none_20pct/linear_probe_shots_SMOKETEST.json \
      "$MS_ROOT"/runs/sf_sunrgbd_all_aux_5pct/linear_probe_shots_SMOKETEST.json \
      "$MS_ROOT"/runs/diagsf_sunrgbd_depth_vit_aux_10pct/linear_probe_shots_SMOKETEST.json \
      "$MS_ROOT"/runs/diagsarred_pca1_so2sat_sar_aux_5pct/linear_probe_shots_SMOKETEST.json
rm -rf "$SD"
echo "leftover SMOKETEST files: $(ls $MS_ROOT/runs/*/*SMOKETEST*.json 2>/dev/null | wc -l)"
echo "RECOVERY_SMOKE_COMPLETE failures=$FAIL"
exit $FAIL   # non-zero => the afterok-dependent lanes cannot start
