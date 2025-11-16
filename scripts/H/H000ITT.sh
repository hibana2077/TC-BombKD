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
# 產出：features_hmdb51_train.index.json + 多個 features_hmdb51_train_shard_XXXXX.pkl
python3 -m polyspace.train.inspect_features --features ./features/features_hmdb51_test.index.json --limit 0 --sample 3 >> H000ITT.log 2>&1
python3 -m polyspace.train.scan_labels --index ./features/hmdb51/features_hmdb51_test.index.json --report >> H000ITT.log 2>&1