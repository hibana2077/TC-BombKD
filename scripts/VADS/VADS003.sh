#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=6:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

# Evaluate VAD model on cached test features and compute AUC; save per-video scores

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..

FEATURES_TEST=./features/stech/features_shanghaitech_test.index.json
CKPT=./checkpoints/vad/vad_ep10.pt

python3 -m polyspace.train.eval_vad \
  --dataset shanghaitech \
  --root ./datasets/Stech/shanghaitech \
  --split test \
  --use_cached_features \
  --cached_features_path ${FEATURES_TEST} \
  --ckpt ${CKPT} \
  --batch 128 \
  --features_fp16 \
  --save_scores ./features/stech/vad_scores_test.json >> VADS003.log 2>&1
