#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=64GB
#PBS -l walltime=48:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
# 備註：若不加 --shard_size 仍可輸出單一 pkl；為避免記憶體爆炸，建議啟用分片與 fp16。
# 產出：features_ucf101_train.index.json + 多個 features_ucf101_train_shard_XXXXX.pkl
python3 -m polyspace.data.featurize \
  --dataset hmdb51 \
  --root ./datasets/hmdb51 \
  --split test \
  --out ./features/hmdb51 \
  --student vjepa2ssv2 \
  --teachers vivit videomaeg timesformerg \
  --batch 2 \
  --workers 2 \
  --student_frames 64 \
  --teacher_frames 32 16 16 \
  --shard_size 512 \
  --storage npy_dir \
  --fp16 \
  --no_tqdm \
  >> H000T.log 2>&1