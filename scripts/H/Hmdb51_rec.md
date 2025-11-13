# Test on HMDB51

## Experiment Results by Configuration

| Code Range | Type | Loss Weights | Top-1 Acc (%) | Top-5 Acc (%) | Teacher Models |
|------------|------|--------------|---------------|---------------|----------------|
| H001-H003 | A | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | 82.19 | 95.71 | Multi-teacher (A converter) |


## Loss Weights Configuration

Loss weights are configured as: [l2, cosine, nce, vicreg, barlow_twins, l1]