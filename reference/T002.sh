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
source /scratch/rp06/xx12345/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/xx12345/TC-BombKD/.cache"

cd ../..
python3 -m src.run train --data-root "./datasets/ucf101_subset/temp_download/extracted/UCF101_subset" --output-dir "./runs/videomae-ucf" --epochs 100 --batch-size 2 --no_tqdm >> T002.log 2>&1
