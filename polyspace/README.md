# PolySpace (single backbone multi-space alignment)

This folder implements the core idea from docs/abs.md:

- One student backbone (e.g., V-JEPA 2) produces clip-level features
- Lightweight converters translate student features to multiple teacher spaces (VideoMAE, TimeSformer, ViViT)
- A residual-gated fusion head aggregates translated features with the student residual path for classification

## Structure

- data/: datasets (HMDB51, Diving48, SSv2) and feature extraction
- models/: backbones, converters (Procrustes/Residual MLP), fusion head
- losses/: L2, Cosine, InfoNCE, VICReg, Barlow Twins, CKA meter
- train/: train converters, train fusion classifier, evaluate
- utils/: metrics, Procrustes closed-form, CKA viz helper

## Quickstart

1. Prepare datasets per docs/new_dataset.md
1. Extract features

```bash
python -m polyspace.data.featurize --dataset hmdb51 --root ./datasets/hmdb51 --split train --out ./features --student vjepa2 --teachers videomae timesformer vivit --batch 2 --workers 2 --frames 16
```

1. Train converters T_i

```bash
python -m polyspace.train.train_converter --features ./features/features_hmdb51_train.json --teachers videomae timesformer vivit --d_in 768 --d_out 768 --epochs 10 --batch 128 --lr 1e-3 --save_dir ./checkpoints/converters
```

1. Train fusion head

```bash
python -m polyspace.train.train_fusion --dataset hmdb51 --root ./datasets/hmdb51 --split train --student vjepa2 --teachers videomae timesformer vivit --converters ./checkpoints/converters/converters_ep10.pt --classes 51 --frames 16 --batch 4 --epochs 5 --lr 3e-4 --save_dir ./checkpoints/fusion
```

1. Evaluate

```bash
python -m polyspace.train.eval_downstream --dataset hmdb51 --root ./datasets/hmdb51 --split validation --student vjepa2 --teachers videomae timesformer vivit --converters ./checkpoints/converters/converters_ep10.pt --fusion ./checkpoints/fusion/fusion_ep5.pt --frames 16 --batch 4
```

Notes:

- If transformers models are unavailable, a lightweight Identity backbone will be used as a fallback so the pipeline remains runnable.
- FLOPs reporting is stubbed; integrate fvcore or ptflops for precise FLOPs.

## Requirements

See project-level requirements.txt. You may need GPU for reasonable speed.
