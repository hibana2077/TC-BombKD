# Fusion Training Modes

This document explains the two modes for training the fusion head in TC-BombKD.

## Overview

The fusion head training supports two modes:
1. **Video Mode** (default): Processes raw videos on-the-fly
2. **Cached Features Mode** (recommended): Uses pre-extracted features

## Mode Comparison

| Aspect | Video Mode | Cached Features Mode |
|--------|-----------|---------------------|
| **Speed** | ~10-20 it/s | ~100-500 it/s |
| **Setup** | No preprocessing needed | Requires feature extraction |
| **Disk Usage** | None (uses videos) | ~1-10 GB per dataset |
| **Memory Usage** | Higher (video decoding) | Lower (features only) |
| **Best For** | Quick experiments, 1-2 epochs | Multi-epoch training, hyperparameter search |

## Video Mode (Default)

### Usage
```bash
python -m polyspace.train.train_fusion \
    --dataset ucf101 \
    --root /path/to/UCF101 \
    --split train \
    --student vjepa2 \
    --teachers videomae timesformer vivit \
    --converters ./checkpoints/converters/converters_ep10.pt \
    --classes 101 \
    --frames 16 \
    --batch 4 \
    --epochs 5
```

### How it works
1. Loads raw videos from dataset
2. Decodes frames using PyAV/OpenCV
3. Extracts student features using student backbone (frozen)
4. Applies converters to generate teacher-like features (frozen)
5. Trains fusion head to combine features

### Computational cost per batch
- Video I/O: ~50-100ms
- Student inference: ~100-200ms (depends on model)
- Converter inference: ~20-50ms
- Fusion forward+backward: ~10-20ms
- **Total: ~200-400ms per batch**

## Cached Features Mode (Recommended)

### Step 1: Extract Features (One-time Setup)

```bash
python -m polyspace.data.featurize \
    --dataset ucf101 \
    --root /path/to/UCF101 \
    --split train \
    --student vjepa2 \
    --teachers videomae timesformer vivit \
    --out ./features/ucf101 \
    --batch 8 \
    --workers 4 \
    --frames 16 \
    --shard_size 1000 \
    --fp16
```

**Parameters:**
- `--shard_size`: Number of samples per shard (reduces RAM usage)
- `--fp16`: Store features in float16 (saves 50% disk space)
- `--workers`: Number of data loading workers

**Output:**
```
./features/ucf101/
├── features_ucf101_train.index.json
├── features_ucf101_train_shard_00000.pkl
├── features_ucf101_train_shard_00001.pkl
└── ...
```

**Time estimate:** ~30-60 minutes for UCF101 train split

### Step 2: Train Fusion with Cached Features

```bash
python -m polyspace.train.train_fusion \
    --dataset ucf101 \
    --root ./features/ucf101 \
    --split train \
    --student vjepa2 \
    --teachers videomae timesformer vivit \
    --converters ./checkpoints/converters/converters_ep10.pt \
    --classes 101 \
    --batch 32 \
    --epochs 20 \
    --use_cached_features
```

**Note:** You can use larger batch sizes (e.g., 32-64) since there's no video decoding overhead.

### How it works
1. Loads pre-extracted features from sharded PKL files
2. Features already contain:
   - Student features (from frozen backbone)
   - Teacher features (from frozen teachers)
   - Labels
3. Applies converters to student features (still frozen)
4. Trains fusion head to combine features

### Computational cost per batch
- Feature loading: ~5-10ms (from disk cache)
- Converter inference: ~20-50ms
- Fusion forward+backward: ~10-20ms
- **Total: ~40-80ms per batch (5-10x faster!)**

## When to Use Each Mode

### Use Video Mode When:
- Running quick experiments (1-2 epochs)
- Testing different architectures
- Don't want to store features on disk
- Dataset is small (<10K videos)

### Use Cached Features Mode When:
- Training for many epochs (>5)
- Running multiple experiments with same features
- Limited video I/O bandwidth
- Student model is slow (e.g., ViT-Large)
- Doing hyperparameter search on fusion head
- Dataset is large (>50K videos)

## Tips for Cached Features Mode

1. **Extract features with FP16** to save disk space:
   ```bash
   --fp16
   ```

2. **Use sharding** for large datasets to reduce RAM:
   ```bash
   --shard_size 1000
   ```

3. **Extract validation features** separately:
   ```bash
   python -m polyspace.data.featurize --split validation ...
   ```

4. **Reuse features** for multiple experiments:
   - Train fusion with different teacher combinations
   - Try different fusion architectures
   - Tune hyperparameters

5. **Check disk space** before extraction:
   - Estimate: `num_videos * num_frames * feature_dim * 4 bytes * (1 + num_teachers)`
   - Example: 10K videos × 16 frames × 768 dim × 4 bytes × 4 models ≈ 2 GB
   - With FP16: ~1 GB

## Example Workflow

```bash
# 1. Extract features once
python -m polyspace.data.featurize \
    --dataset ucf101 --root /data/UCF101 --split train \
    --student vjepa2 --teachers videomae timesformer vivit \
    --out ./features --shard_size 1000 --fp16

# 2. Train multiple fusion experiments (fast!)
for lr in 1e-3 3e-4 1e-4; do
    python -m polyspace.train.train_fusion \
        --dataset ucf101 --root ./features \
        --student vjepa2 --teachers videomae timesformer vivit \
        --converters ./ckpt/converters.pt --classes 101 \
        --use_cached_features --lr $lr --epochs 20 \
        --save_dir ./fusion_lr${lr}
done
```

## Troubleshooting

### Issue: "Cached features do not contain labels"
- **Cause:** Features were extracted with old version of featurize.py
- **Solution:** Re-extract features with updated version

### Issue: Out of memory during feature extraction
- **Cause:** Shard size too large or too many workers
- **Solution:** Reduce `--shard_size` or `--workers`

### Issue: Slow feature loading in cached mode
- **Cause:** Disk I/O bottleneck
- **Solution:** 
  - Use SSD instead of HDD
  - Reduce `--workers` in dataloader
  - Increase `--shard_size` to reduce file count

### Issue: Different results between modes
- **Cause:** Both modes should give same results (student/converters are frozen)
- **Solution:** Check that:
  - Same random seeds are used
  - Same number of frames
  - Features were extracted with correct models
