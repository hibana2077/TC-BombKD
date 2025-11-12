# Test on HMDB51

## Individual Results

| Code | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Notes |
|------|-------------------|-------------------|--------|
| H003 | 66.74 | 92.35 | Early experiment |
| H004 | 70.67 | 93.63 | Early experiment |
| H005 | 73.03 | 94.49 | Early experiment |
| H006 | 72.89 | 93.42 | Early experiment |

## Experiment Results by Configuration

| Code Range | Type | Loss Weights | Top-1 Acc (%) | Top-5 Acc (%) | Teacher Models |
|------------|------|--------------|---------------|---------------|----------------|
| H001-H006 | A | [0.0, 0.0, 0.03, 0.03, 0.0, 0.0] | 73.03 | 94.49 | VideoMAE (K400) |
| H007-H011 | A | [0.1, 0.8, 0.0, 0.0, 0.0, 0.1] | 75.18 | 95.71 | VideoMAE (K400) |
| H012-H016 | A | [0.0, 0.0, 0.0, 0.1, 0.1, 0.8] | 74.25 | 95.64 | VideoMAE (K400) |
| H017-H021 | A | [0.0, 0.0, 0.1, 0.1, 0.1, 0.7] | 74.75 | 95.14 | VideoMAE (K400) |
| H022-H026 | A | [0.0, 0.0, 0.2, 0.1, 0.1, 0.6] | 75.46 | 95.06 | VideoMAE (K400) |
| H027-H031 | A | [0.0, 0.0, 0.2, 0.2, 0.1, 0.5] | 75.82 | 94.92 | VideoMAE (K400) |
| H032-H036 | A | [0.0, 0.9, 0.0, 0.0, 0.1, 0.0] | 76.04 | 95.21 | VideoMAE (K400) |
| H037-H041 | A | [0.0, 0.0, 0.5, 0.5, 0.0, 0.0] | 77.97 | 95.42 | VideoMAE (K400) |
| H042-H046 | A | [0.0, 0.0, 0.0, 1.0, 0.0, 0.0] | 76.90 | 94.99 | VideoMAE (K400) |
| H047-H051 | A | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | 75.75 | 95.14 | VideoMAE (K400) |
| H052-H056 | A | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | **82.19** | 95.71 | Multi-teacher (A converter) |
| H057-H061 | A | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | 79.33 | 95.49 | Multi-teacher (A converter) |
| H062-H066 | B | [0.0, 0.0, 0.0, 1.0, 0.0, 0.0] | 80.33 | 94.28 | Multi-teacher (B converter) |
| H067-H071 | B | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD | Multi-teacher (B converter) |
| H072-H074 | C | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD | Multi-teacher (C converter) |
| H075-H077 | C | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | TBD | TBD | Multi-teacher (C converter) |
| H078-H080 | A | [0.0, 0.9, 0.1, 0.0, 0.0, 0.0] | TBD | TBD | Multi-teacher (A converter) |
| H081-H083 | A | [0.0, 0.85, 0.0, 0.1, 0.0, 0.05] | TBD | TBD | Multi-teacher (A converter) |
| H084-H086 | B | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD | Multi-teacher (B converter) |
| H087-H089 | B | [0.0, 0.8, 0.1, 0.1, 0.0, 0.0] | TBD | TBD | Multi-teacher (B converter) |
| H090-H092 | C | [0.0, 0.9, 0.0, 0.1, 0.0, 0.0] | TBD | TBD | Multi-teacher (C converter) |
| H093-H095 | C | [0.0, 0.8, 0.2, 0.0, 0.0, 0.0] | TBD | TBD | Multi-teacher (C converter) |
| H096-H098 | A | [0.05, 0.9, 0.0, 0.0, 0.0, 0.05] | TBD | TBD | Multi-teacher (A converter) |
| H099-H101 | A | [0.0, 0.7, 0.2, 0.1, 0.0, 0.0] | TBD | TBD | Multi-teacher (A converter) |
| H102-H104 | B | [0.0, 0.75, 0.0, 0.25, 0.0, 0.0] | TBD | TBD | Multi-teacher (B converter) |
| H105-H107 | C | [0.0, 0.7, 0.0, 0.2, 0.1, 0.0] | TBD | TBD | Multi-teacher (C converter) |

## Loss Weights Configuration

Loss weights are configured as: [l2, cosine, nce, vicreg, barlow_twins, l1]

## Teacher Model Configurations

### Type A (H001-H061)

- **H001-H031**: VideoMAE pre-trained on Kinetics400
- **H052-H061**: Multi-teacher setup with A-type converters
  - VideoMAE pre-trained on Kinetics400
  - TimeSformer pre-trained on SSv2  
  - ViViT pre-trained on Kinetics400

### Type B (H062-H071)

- Multi-teacher setup with B-type converters
- Teachers: VideoMAE, TimeSformer, ViViT pre-trained on Kinetics400 and SSv2

### Type C (H072-H077)

- Multi-teacher setup with C-type converters
- Teachers: VideoMAE, TimeSformer, ViViT pre-trained on Kinetics400 and SSv2
