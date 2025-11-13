#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=02:30:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -m polyspace.train.train_converter \
	--features ./features/features_uav_train.index.json \
	--teachers videomae timesformer vivit \
	--d_in 1024 --d_out 768 \
	--kind b \
	--epochs 10 \
	--batch 32 \
	--workers 1 \
	--log_every 20 \
	--pin_memory \
	--loss_l2 0.1 \
	--loss_cos 0.8 \
	--loss_nce 0.1 \
	--loss_vic 0.0 \
	--loss_bar 0.0 \
	--loss_l1 0.0 \
	--save_dir ./checkpoints/UH082 >> UH082.log 2>&1
