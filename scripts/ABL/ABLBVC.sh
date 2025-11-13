#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=32GB
#PBS -l walltime=00:40:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -m polyspace.train.feature_cls \
    --task ar \
    --train ./features/features_breakfast_train.index.json \
    --test ./features/features_breakfast_test.index.json \
    --feature conv:videomae \
    --converters ./checkpoints/B097/converters_ep10.pt \
    --teachers videomae \
    --remap_by_order \
    --ar_models svm lr >> ABLBVC.log 2>&1