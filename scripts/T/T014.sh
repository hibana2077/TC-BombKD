#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=02:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
# Train the student model on HMDB51 training set
python3 -m polyspace.train.train_student \
  --dataset hmdb51 \
  --root ./datasets/hmdb51 \
  --split train \
  --student vjepa2 \
  --classes 51 \
  --frames 16 \
  --batch 8 \
  --epochs 10 \
  --lr 1e-3 \
  --save_dir ./checkpoints/student >> T014.log 2>&1