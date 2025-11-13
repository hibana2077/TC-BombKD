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
  --dataset ucf101 \
  --root ./datasets/UCF101 \
  --split train \
  --student vjepa2 \
  --teachers videomae timesformer vivit \
  --all_teachers \
  --marker_size 200 \
  --legend_marker_size 20 \
  --legend_font_size 14 \
  --legend_layout horizontal \
  --class_selection lr \
  --converters ./checkpoints/U010/converters_ep10.pt \
  --frames 16 \
  --per_class 50 \
  --max_classes 12 \
  --batch 8 \
  --seed 42 \
  --save_dir ./VIS/VU002 >> VU002.log 2>&1


# # Use Logistic Regression to select classes by accuracy
# python vis_ct_orig.py --dataset hmdb51 --root /path/to/data --class_selection lr --max_classes 10

# # Use Random Forest with custom legend settings
# python vis_ct_orig.py --dataset ucf101 --root /path/to/data \
#   --class_selection rf \
#   --legend_marker_size 12.0 \
#   --legend_font_size 14.0 \
#   --legend_layout horizontal

# # Traditional random selection (original behavior)
# python vis_ct_orig.py --dataset diving48 --root /path/to/data --class_selection random