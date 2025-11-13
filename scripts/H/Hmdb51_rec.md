# Test on HMDB51

## Experiment Results by Configuration

| Code Range | Type | Loss Weights | Top-1 Acc (%) | Top-5 Acc (%) | Teacher Models | ADV cls head |
|------------|------|--------------|---------------|---------------|----------------|---|
| H001-H003 | A | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | 82.19 | 95.71 | Multi-teacher (A converter) | N |
| H004-H006 | A | [0.0, 0.8, 0.2, 0.0, 0.0, 0.0] | - | - | Multi-teacher (A converter) | Y |
| H007-H009 | A | [0.0, 0.7, 0.0, 0.3, 0.0, 0.0] | - | - | Multi-teacher (A converter) | Y |
| H010-H012 | B | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | - | - | Multi-teacher (B converter) | Y |
| H013-H015 | B | [0.0, 0.5, 0.5, 0.0, 0.0, 0.0] | - | - | Multi-teacher (B converter) | Y |
| H016-H018 | B | [0.0, 0.3, 0.4, 0.3, 0.0, 0.0] | - | - | Multi-teacher (B converter) | Y |
| H019-H021 | C | [0.0, 0.2, 0.0, 0.6, 0.2, 0.0] | - | - | Multi-teacher (C converter) | Y |
| H022-H024 | C | [0.0, 0.1, 0.0, 0.0, 0.9, 0.0] | - | - | Multi-teacher (C converter) | Y |
| H025-H027 | C | [0.0, 0.25, 0.25, 0.25, 0.25, 0.0] | - | - | Multi-teacher (C converter) | Y |


## Loss Weights Configuration

Loss weights are configured as: [l2, cosine, nce, vicreg, barlow_twins, l1]