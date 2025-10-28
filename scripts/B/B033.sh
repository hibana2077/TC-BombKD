#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=24GB
#PBS -l walltime=02:40:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/TC-BombKD/.cache"
export HF_HUB_OFFLINE=1

cd ../..
for ep in {10..50}; do
  echo "checkpoint: ep$ep" >> B033.log 2>&1
  python3 -m polyspace.train.eval_downstream \
    --dataset breakfast \
    --root ./datasets/breakfast \
    --split test \
    --student vjepa2 \
    --teachers timesformer \
    --converters ./checkpoints/B031/converters_ep10.pt \
    --fusion ./checkpoints/B032/fusion_ep$ep.pt >> B033.log 2>&1
done
