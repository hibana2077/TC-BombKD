#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=24GB
#PBS -l walltime=00:10:00
#PBS -l wd
#PBS -l storage=scratch/rp06

# Train converters (translators T_k) on cached Stech features
# Inputs: features extracted with student+teachers (vjepa2 + videomae timesformer vivit)
# Output: ./checkpoints/converters/converters_ep*.pt

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..

FEATURES_DIR=./features/stech
CONV_DIR=./checkpoints/converters
mkdir -p "$CONV_DIR"

python3 -m polyspace.train.train_converter \
  --features ${FEATURES_DIR}/features_shanghaitech_train.index.json \
  --teachers videomae timesformer vivit \
  --d_in 1024 \
  --d_out 768 \
  --epochs 10 \
  --batch 128 \
  --workers 1 \
  --lr 3e-4 \
  --loss_l2 0.0 \
  --loss_cos 1.0 \
  --loss_nce 0.0 \
  --loss_vic 0.0 \
  --loss_bar 0.0 \
  --loss_l1 0.0 \
  --save_dir ${CONV_DIR} \
  --kind b \
  --shuffle shard \
  --log_every 50 >> VADS001.log 2>&1
