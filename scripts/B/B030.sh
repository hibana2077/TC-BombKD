#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=44GB
#PBS -l walltime=24:30:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -m polyspace.train.train_converter \
	--features ./features/breakfast/features_breakfast_train.index.json \
	--teachers videomaessv2 timesformerssv2 vivit \
	--d_in 1408 --d_out 768 \
	--kind c \
	--epochs 10 \
	--batch 8 \
	--workers 1 \
	--log_every 20 \
	--pin_memory \
	--loss_l2 0.10 \
	--loss_cos 0.60 \
	--loss_nce 0.10 \
	--loss_vic 0.10 \
	--loss_bar 0.10 \
	--loss_l1 0.00 \
	--save_dir ./checkpoints/B030/converter >> B030.log 2>&1

python3 -m polyspace.train.train_fusion \
    --features ./features/breakfast/features_breakfast_train.index.json \
	--teachers videomaessv2 timesformerssv2 vivit \
    --converters ./checkpoints/B030/converter/converters_ep10.pt \
    --classes 10 \
    --batch 8 \
    --epochs 50 \
    --lr 3e-4 \
    --features_fp16 \
    --save_dir ./checkpoints/B030/head >> B030.log 2>&1

for ep in {1..50}; do
  echo "checkpoint: ep$ep" >> B030.log 2>&1
  python3 -m polyspace.train.eval_downstream \
    --features ./features/breakfast/features_breakfast_test.index.json \
    --teachers videomaessv2 timesformerssv2 vivit \
    --converters ./checkpoints/B030/converter/converters_ep10.pt \
    --fusion ./checkpoints/B030/head/fusion_ep$ep.pt \
    --features_fp16 >> B030E.log 2>&1
done
