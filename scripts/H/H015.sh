#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=50GB
#PBS -l walltime=40:30:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -m polyspace.train.train_converter \
	--features ./features/hmdb51/features_hmdb51_train.index.json \
	--teachers vivit videomaeg timesformerg \
	--d_in 1408 --d_out 768 \
	--d_out_map "vivit=768,videomaeg=1280,timesformerg=768" \
	--kind c \
	--epochs 10 \
	--batch 8 \
	--workers 1 \
	--log_every 20 \
	--pin_memory \
	--loss_l2 0.2 \
	--loss_cos 0.0 \
	--loss_nce 0.0 \
	--loss_vic 0.0 \
	--loss_bar 0.0 \
	--loss_l1 0.0 \
	--save_dir ./checkpoints/H015/converter >> H015.log 2>&1

python3 -m polyspace.train.train_fusion \
	--features ./features/hmdb51/features_hmdb51_train.index.json \
	--teachers vivit videomaeg timesformerg \
	--converters ./checkpoints/H015/converter/converters_ep10.pt \
	--classes 51 \
	--batch 8 \
	--epochs 50 \
	--lr 3e-4 \
	--features_fp16 \
	--save_dir ./checkpoints/H015/fusion >> H015.log 2>&1

for ep in {1..50}; do
  echo "checkpoint: ep$ep" >> H015.log 2>&1
	python3 -m polyspace.train.eval_downstream \
		--features ./features/hmdb51/features_hmdb51_test.index.json \
		--teachers vivit videomaeg timesformerg \
		--converters ./checkpoints/H015/converter/converters_ep10.pt \
		--fusion ./checkpoints/H015/fusion/fusion_ep$ep.pt \
		--features_fp16 >> H015E.log 2>&1
done
