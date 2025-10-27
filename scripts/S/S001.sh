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
# python3 -m polyspace.data.featurize --dataset hmdb51 --root ./datasets/hmdb51 --split train --out ./features --student vjepa2 --teachers videomae timesformer vivit --batch 2 --workers 2 --frames 16 >> T002.log 2>&1
# python3 -m polyspace.train.inspect_features --features ./features/features_hmdb51_train.index.json --limit 100 --sample 3 >> T002.log 2>&1
python3 -m polyspace.train.train_converter \
	--features ./features/features_hmdb51_train.index.json \
	--teachers videomae \
	--teacher_lens 1568 \
	--d_in 1024 --d_out 768 \
	--kind a \
	--epochs 10 \
	--batch 32 \
	--workers 1 \
    --log_every 20 \
    --pin_memory \
	--loss_l2 0.0 \
	--loss_cos 0.0 \
	--loss_nce 0.0 \
	--loss_vic 1.0 \
	--loss_bar 0.0 \
	--loss_l1 0.0 \
	--save_dir ./checkpoints/H042 >> H042.log 2>&1
