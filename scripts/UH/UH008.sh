#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=32GB
#PBS -l walltime=48:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -m polyspace.train.train_fusion \
    --dataset uav \
    --root features_uav_train.index.json \
    --split train \
    --student vjepa2 \
    --teachers videomae timesformer vivit \
    --converters ./checkpoints/UH007/converters_ep10.pt \
    --classes 155 \
    --frames 16 \
    --batch 4 \
    --epochs 50 \
    --lr 3e-4 \
    --use_cached_features \
    --features_fp16 \
    --save_dir ./checkpoints/UH008 >> UH008.log 2>&1