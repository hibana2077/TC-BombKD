#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=12:30:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -m polyspace.train.train_converter \
    --features ./features/features_ucf101_train.index.json \
    --teachers videomae timesformer vivit \
    --d_in 1024 --d_out 768 \
    --kind b \
    --epochs 10 \
    --batch 32 \
    --workers 1 \
    --log_every 20 \
    --pin_memory \
    --loss_l2 0.10 \
    --loss_cos 0.10 \
    --loss_nce 0.20 \
    --loss_vic 0.05 \
    --loss_bar 0.55 \
    --loss_l1 0.00 \
    --save_dir ./checkpoints/U060 >> U060.log 2>&1

python3 -m polyspace.train.train_fusion \
    --dataset ucf101 \
    --root ./features/features_ucf101_train.index.json \
    --split train \
    --student vjepa2 \
    --teachers videomae timesformer vivit \
    --converters ./checkpoints/U060/converters_ep10.pt \
    --classes 101 \
    --frames 16 \
    --batch 8 \
    --epochs 50 \
    --lr 3e-4 \
    --use_cached_features \
    --features_fp16 \
    --save_dir ./checkpoints/U060 >> U060.log 2>&1

for ep in {1..50}; do
  echo "checkpoint: ep$ep" >> U060.log 2>&1
  python3 -m polyspace.train.eval_downstream \
    --dataset ucf101 \
  --root ./features/features_ucf101_test.index.json \
    --split test \
    --student vjepa2 \
    --teachers videomae timesformer vivit \
    --converters ./checkpoints/U060/converters_ep10.pt \
    --fusion ./checkpoints/U060/fusion_ep$ep.pt \
    --features_fp16 \
    --use_cached_features \
    --advance-cls-head \
    --frames 16 >> U060E.log 2>&1
done
