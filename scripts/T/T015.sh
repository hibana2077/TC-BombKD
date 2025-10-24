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
# Evaluate only the student model on HMDB51 validation set
python3 -m polyspace.train.eval_downstream \
  --dataset hmdb51 \
  --root ./datasets/hmdb51 \
  --split validation \
  --student vjepa2 \
  --student_only \
  --classes 51 \
  --student_head ./checkpoints/student/selfres_ep4.pt \
  --frames 16 \
  --batch 8 >> T015.log 2>&1