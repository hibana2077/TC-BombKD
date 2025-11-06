#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=24GB
#PBS -l walltime=12:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
# python3 -m polyspace.data.featurize --dataset hmdb51 --root ./datasets/hmdb51 --split train --out ./features --student vjepa2 --teachers videomae timesformer vivit --batch 2 --workers 2 --frames 16 >> T002.log 2>&1
# python3 -m polyspace.train.inspect_features --features ./features/features_hmdb51_train.index.json --limit 100 --sample 3 >> T002.log 2>&1
python3 -m polyspace.train.train_fusion \
    --dataset ssv2 \
    --root ./datasets/ssv2 \
    --split train \
    --student vjepa2 \
    --teachers videomae \
    --converters ./checkpoints/S001/converters_ep6.pt \
    --classes 174 \
    --frames 16 \
    --batch 8 \
    --epochs 20 \
    --lr 3e-4 \
    --save_dir ./checkpoints/S002 >> S002.log 2>&1
