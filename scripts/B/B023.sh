#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=24GB
#PBS -l walltime=05:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -m polyspace.train.train_fusion \
    --dataset breakfast \
    --root ./datasets/breakfast \
    --split train \
    --student vjepa2 \
    --teachers timesformer \
    --converters ./checkpoints/B022/converters_ep10.pt \
    --classes 51 \
    --frames 16 \
    --batch 8 \
    --epochs 50 \
    --lr 3e-4 \
    --save_dir ./checkpoints/B023 >> B023.log 2>&1
