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
# 備註：若不加 --shard_size 仍可輸出單一 pkl；為避免記憶體爆炸，建議啟用分片與 fp16。
# 產出：features_hmdb51_train.index.json + 多個 features_hmdb51_train_shard_XXXXX.pkl
python3 -m polyspace.train.eval_downstream \
  --dataset hmdb51 \
  --root ./datasets/hmdb51 \
  --split test \
  --student vjepa2 \
  --student_only \
  --fusion ./checkpoints/fusion_final/fusion_ep3.pt >> T013.log 2>&1