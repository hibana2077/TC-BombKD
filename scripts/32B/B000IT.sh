#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=12:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
# 備註：若不加 --shard_size 仍可輸出單一 pkl；為避免記憶體爆炸，建議啟用分片與 fp16。
# 產出：features_ssv2_train.index.json + 多個 features_ssv2_train_shard_XXXXX.pkl
python3 -m polyspace.train.inspect_features --features ./features/frames32/features_breakfast_train.index.json --limit 100 --sample 3 >> B000IT.log 2>&1