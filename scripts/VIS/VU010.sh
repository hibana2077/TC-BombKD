#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=64GB
#PBS -l walltime=00:40:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python -m polyspace.vis.vis_ct \
  --dataset breakfast \
  --root ./features/features_breakfast_train.index.json \
  --split train \
  --student vjepa2 \
  --teachers videomae timesformer vivit \
  --teacher timesformer \
  --converters ./checkpoints/B097/converters_ep10.pt \
  --per_class 50 \
  --max_classes 10 \
  --marker_size 200 \
  --legend_marker_size 20 \
  --class_selection lr \
  --save_dir ./VIS/VU010 >> VU010.log 2>&1