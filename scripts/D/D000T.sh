#!/bin/bash
#PBS -P kf09
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=64GB
#PBS -l walltime=24:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -m polyspace.data.featurize \
  --dataset diving48 \
  --root ./datasets/Diving48 \
  --split test \
  --out ./features/diving48 \
  --student vjepa2div \
  --teachers vivit videomaessv2 timesformerssv2 \
  --batch 2 \
  --workers 2 \
  --student_frames 32 \
  --teacher_frames 32 16 16 \
  --shard_size 512 \
  --storage npy_dir \
  --fp16 \
  --no_tqdm \
  >> D000T.log 2>&1