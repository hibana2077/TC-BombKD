#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=01:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

# Train VAD (fusion + projection head) using pre-trained converters and cached features
# Requires: converters checkpoint from VADS001 (e.g., converters_ep10.pt)
# Output: ./checkpoints/vad/vad_ep*.pt

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..

FEATURES_DIR=./features/stech
CONVERTER_CKPT=./checkpoints/converters/converters_ep10.pt
VAD_DIR=./checkpoints/vad
mkdir -p "$VAD_DIR"

python3 -m polyspace.train.train_vad \
  --dataset shanghaitech \
  --root ./datasets/Stech/shanghaitech \
  --split train \
  --student vjepa2 \
  --use_cached_features \
  --cached_features_path ${FEATURES_DIR}/features_shanghaitech_train.index.json \
  --converters_ckpt ${CONVERTER_CKPT} \
  --translator_kind b \
  --batch 64 \
  --epochs 20 \
  --lr 3e-4 \
  --proj_dim 128 \
  --margin 1.0 \
  --save_dir ${VAD_DIR} \
  --features_fp16 >> VADS002.log 2>&1
#   --freeze_translators
  
