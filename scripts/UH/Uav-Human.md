# UAV-Human Dataset Score

| Code | loss weight | Top1 Acc (%) | Top5 Acc (%) |
|------|-------------|---------------|---------------|
| UH001-UH003 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | TBD | TBD |
| UH004-UH006 | [0.0, 0.0, 0.0, 1.0, 0.0, 0.0] | TBD | TBD |
| UH007-UH009 | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | TBD | TBD |
| UH010-UH012 | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |
| UH013-UH015 | [0.1, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |
| UH016-UH018 | [0.1, 0.0, 1.0, 0.0, 0.0, 0.0] | TBD | TBD |
| UH019-UH021 | [0.1, 0.0, 0.0, 1.0, 0.0, 0.0] | TBD | TBD |
| UH022-UH024 | [0.1, 0.0, 0.0, 0.0, 1.0, 0.0] | TBD | TBD |

## Loss Weights

[L2, COS, NCE, VIC, BAR, L1]

## Teacher Models

- UH001-UH012: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-A converters.
- UH013-UH024: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-B converters.