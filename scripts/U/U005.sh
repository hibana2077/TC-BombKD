#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=10:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -m polyspace.train.train_fusion \
    --dataset ucf101 \
    --root ./datasets/UCF101 \
    --split train \
    --student vjepa2 \
    --teachers timesformer \
    --converters ./checkpoints/U004/converters_ep9.pt \
    --classes 101 \
    --frames 16 \
    --batch 8 \
    --epochs 50 \
    --lr 3e-4 \
    --save_dir ./checkpoints/U005 >> U005.log 2>&1
