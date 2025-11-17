#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
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
	--dataset ucf101 \
	--root ./datasets/UCF101 \
	--split train \
	--out ./features/ucf101 \
	--student vjepa2div \
	--teachers vivit videomaeg timesformerg \
	--batch 2 \
	--workers 2 \
	--student_frames 64 \
	--teacher_frames 32 16 16 \
	--shard_size 512 \
	--storage npy_dir \
	--fp16 \
	--no_tqdm \
	>> U000.log 2>&1