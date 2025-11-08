#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=8GB
#PBS -l walltime=00:40:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -m polyspace.vis.vis_ct_orig \
  --dataset hmdb51 \
  --root ./datasets/hmdb51 \
  --split train \
  --student vjepa2 \
  --teachers videomae timesformer vivit \
  --all_teachers \
  --marker_size 180 \
  --converters ./checkpoints/H057/converters_ep10.pt \
  --frames 16 \
  --per_class 25 \
  --max_classes 12 \
  --batch 8 \
  --seed 42 \
  --save_dir ./VIS/VU003 >> VU003.log 2>&1
