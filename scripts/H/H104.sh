#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=22:40:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
for ep in {1..50}; do
  echo "checkpoint: ep$ep" >> H104.log 2>&1
  python3 -m polyspace.train.eval_downstream \
    --dataset hmdb51 \
    --root ./features/features_hmdb51_test.index.json \
    --split test \
    --student vjepa2 \
    --teachers videomae timesformer vivit \
    --converters ./checkpoints/H102/converters_ep10.pt \
    --fusion ./checkpoints/H103/fusion_ep$ep.pt \
    --features_fp16 \
    --use_cached_features \
    --frames 16 >> H104.log 2>&1
done