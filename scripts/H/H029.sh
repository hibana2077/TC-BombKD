#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=00:40:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
# 備註：若不加 --shard_size 仍可輸出單一 pkl；為避免記憶體爆炸，建議啟用分片與 fp16。
# 產出：features_hmdb51_train.index.json + 多個 features_hmdb51_train_shard_XXXXX.pkl
echo "checkpoint: ep10" >> H029.log 2>&1
python3 -m polyspace.train.eval_downstream \
  --dataset hmdb51 \
  --root ./datasets/hmdb51 \
  --split test \
  --student vjepa2 \
  --teachers videomae \
  --converters ./checkpoints/H027/converters_ep10.pt \
  --fusion ./checkpoints/H028/fusion_ep10.pt >> H029.log 2>&1
echo "checkpoint: ep9" >> H029.log 2>&1
python3 -m polyspace.train.eval_downstream \
  --dataset hmdb51 \
  --root ./datasets/hmdb51 \
  --split test \
  --student vjepa2 \
  --teachers videomae \
  --converters ./checkpoints/H027/converters_ep9.pt \
  --fusion ./checkpoints/H028/fusion_ep9.pt >> H029.log 2>&1