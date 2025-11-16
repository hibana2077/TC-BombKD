#!/bin/bash
#PBS -P kf09
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
    --features ./features/features_breakfast_train.index.json \
	--teachers videomaessv2 timesformerssv2 vivit \
    --converters ./checkpoints/B097/converters_ep10.pt \
    --classes 10 \
    --batch 8 \
    --epochs 50 \
    --lr 3e-4 \
    --features_fp16 \
    --save_dir ./checkpoints/B098 >> B098.log 2>&1
