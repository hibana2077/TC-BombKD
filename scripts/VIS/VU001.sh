#!/bin/bash
# Visualization job for pre/post converter embeddings
# Adapt queue directives as needed for local or cluster environment.

module load cuda/12.6.2 2>/dev/null || true
source /scratch/rp06/sl5952/TC-BombKD/.venv/bin/activate 2>/dev/null || true
export HF_HOME="${HF_HOME:-./.cache}"  # fallback local cache

python3 -m polyspace.vis.vis_ct_orig \
  --dataset ucf101 \
  --root ./datasets/UCF101 \
  --split train \
  --student vjepa2 \
  --teachers videomae timesformer vivit \
  --converters ./checkpoints/U025/converters_ep10.pt \
  --frames 16 \
  --per_class 25 \
  --max_classes 12 \
  --batch 8 \
  --seed 42 \
  --save_dir ./VU001 >> V_U_001.log 2>&1
