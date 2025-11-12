#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=32GB
#PBS -l walltime=02:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
python3 -m polyspace.train.feature_cls \
  --task vad \
  --train ./features/stech/features_shanghaitech_train.index.json \
  --test ./features/stech/features_shanghaitech_test.index.json \
  --feature vjepa2 \
  --vad_models dbscan iforest >> ABLVADS001.log 2>&1