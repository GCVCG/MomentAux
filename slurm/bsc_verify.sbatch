#!/bin/bash
#SBATCH -J ms_verify
#SBATCH -o /gpfs/scratch/${CLUSTER_ACCOUNT}/momentstem/logs/ms_verify_%j.out
#SBATCH -t 12:00:00
#SBATCH -A ${CLUSTER_ACCOUNT}
#SBATCH -q acc_resc
#SBATCH -p acc
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -c 80
set -u
MS=/gpfs/scratch/${CLUSTER_ACCOUNT}/momentstem
cd $MS
source env.sh
mkdir -p $MS/verify
export MS_VERIFY_WORKERS=4
NSH=8
for i in $(seq 0 $((NSH-1))); do
  g=$(( i % 4 ))
  CUDA_VISIBLE_DEVICES=$g python repo/analysis/verify_checkpoints.py \
      --all-probed --all-seeds --runs $MS/runs --data-root $MS/data \
      --ckpt best.pt --tol 0.5 --shard $i/$NSH \
      --out $MS/verify/best_shard${i}.json \
      > $MS/logs/verify_best_${i}.log 2>&1 &
done
wait
echo "BEST PASS DONE"
for i in $(seq 0 $((NSH-1))); do
  g=$(( i % 4 ))
  CUDA_VISIBLE_DEVICES=$g python repo/analysis/verify_checkpoints.py \
      --all-probed --all-seeds --runs $MS/runs --data-root $MS/data \
      --ckpt last.pt --tol 0.5 --shard $i/$NSH \
      --out $MS/verify/last_shard${i}.json \
      > $MS/logs/verify_last_${i}.log 2>&1 &
done
wait
echo "VERIFY_COMPLETE"
