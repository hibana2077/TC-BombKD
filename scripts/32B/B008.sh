#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=20:00:00
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
    --teachers videomae timesformer vivit \
    --converters ./checkpoints/BX007/converters_ep10.pt \
    --classes 51 \
    --frames 32 \
    --batch 8 \
    --epochs 50 \
    --lr 3e-4 \
    --save_dir ./checkpoints/BX008 >> BX008.log 2>&1
