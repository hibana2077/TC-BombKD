#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=12:40:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
for ep in {1..30}; do
  echo "checkpoint: ep$ep" >> B099.log 2>&1
  python3 -m polyspace.train.eval_downstream \
    --features ./features/breakfast/features_breakfast_test.index.json \
    --split test \
    --teachers videomaessv2 timesformerssv2 vivit \
    --converters ./checkpoints/B097/converters_ep10.pt \
    --fusion ./checkpoints/B098/fusion_ep$ep.pt \
    --features_fp16 \
    --use_cached_features >> B099.log 2>&1
done
