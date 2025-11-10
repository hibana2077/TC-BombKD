#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=32GB
#PBS -l walltime=02:40:00
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
    --root ./features/ssv2/features_ssv2_train.index.json \
    --split train \
    --student vjepa2 \
    --teachers videomae timesformer vivit \
    --converters ./checkpoints/S004/converters_ep10.pt \
    --classes 174 \
    --frames 16 \
    --batch 8 \
    --epochs 20 \
    --lr 3e-4 \
    --use_cached_features \
    --features_fp16 \
    --save_dir ./checkpoints/S005 >> S005.log 2>&1
