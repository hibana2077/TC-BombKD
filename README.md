# TC-BombKD

This repository implements PolySpace: a single-backbone multi-space alignment and residual-gated fusion framework for video understanding.

Key pieces:

- polyspace/data: dataset indexers and feature extraction utilities
- polyspace/models: V-JEPA 2 backbone and teacher wrappers + converters + fusion head
- polyspace/losses: alignment/contrastive losses (VICReg, Barlow Twins, InfoNCE)
- polyspace/train: training scripts for converters and fusion classifier
- reference/: useful inspection scripts and Barlow Twins reference
- docs/: dataset instructions and method abstract

Quickstart (PowerShell):

1) Install dependencies

```pwsh
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

2) Prepare datasets following `docs/new_dataset.md`.

3) Offline feature extraction (example for HMDB51 train split):

```pwsh
python - << 'PY'
from polyspace.data import HMDB51Dataset, extract_features

ds = HMDB51Dataset('./datasets/hmdb51', split='train')
extract_features(ds, out_dir='./features/hmdb51/train', frame_count=32, size=224)
PY
```

4) Train converters to align V-JEPA 2 to teacher spaces:

```pwsh
python -m polyspace.train.train_converter --feat_dir ./features/hmdb51/train --out ./runs/converters --dim 1024 --epochs 5 --w_l2 1.0 --w_vic 1.0 --use_procrustes_init
```

5) Train residual-gated fusion + classifier:

```pwsh
python -m polyspace.train.train_fusion --feat_dir ./features/hmdb51/train --converters_dir ./runs/converters --out ./runs/fusion --dim 1024 --classes 51 --epochs 10
```

6) Evaluate:

```pwsh
python -m polyspace.train.eval_downstream --feat_dir ./features/hmdb51/validation --checkpoint ./runs/fusion/fusion_cls.pt
```

Notes:
- The provided wrappers use Hugging Face models; ensure GPU memory is sufficient or switch to CPU.
- For Diving48 and SSv2, follow `docs/new_dataset.md` for structure and labels.
