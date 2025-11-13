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
| H001-H003 | A | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | **82.19** | 95.71 | Multi-teacher (A converter) |


## Loss Weights Configuration

Loss weights are configured as: [l2, cosine, nce, vicreg, barlow_twins, l1]

