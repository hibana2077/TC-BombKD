# UAV-Human Dataset Score

| Code | loss weight | Top1 Acc (%) | Top5 Acc (%) |
|------|-------------|---------------|---------------|
| UH001-UH003 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | 19.51 | 42.42 |
| UH004-UH006 | [0.0, 0.0, 0.0, 1.0, 0.0, 0.0] | TBD | TBD |
| UH007-UH009 | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | TBD | TBD |
| UH010-UH012 | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |
| UH013-UH015 | [0.1, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |
| UH016-UH018 | [0.1, 0.0, 1.0, 0.0, 0.0, 0.0] | TBD | TBD |
| UH019-UH021 | [0.1, 0.0, 0.0, 1.0, 0.0, 0.0] | TBD | TBD |
| UH022-UH024 | [0.1, 0.0, 0.0, 0.0, 1.0, 0.0] | 24.68 | 49.44 |
| UH025-UH027 | [0.0, 0.9, 0.1, 0.0, 0.0, 0.0] | 36.50 | 62.46 |
| UH028-UH030 | [0.0, 0.9, 0.0, 0.1, 0.0, 0.0] | TBD | TBD |
| UH031-UH033 | [0.0, 0.9, 0.0, 0.0, 0.1, 0.0] | TBD | TBD |
| UH034-UH036 | [0.0, 0.9, 0.0, 0.0, 0.0, 0.1] | TBD | TBD |
| UH037-UH039 | [0.0, 0.8, 0.2, 0.0, 0.0, 0.0] | TBD | TBD |
| UH040-UH042 | [0.0, 0.7, 0.3, 0.0, 0.0, 0.0] | TBD | TBD |
| UH043-UH045 | [0.0, 0.6, 0.4, 0.0, 0.0, 0.0] | TBD | TBD |
| UH046-UH048 | [0.0, 0.5, 0.5, 0.0, 0.0, 0.0] | TBD | TBD |

## Loss Weights

[L2, COS, NCE, VIC, BAR, L1]

## Teacher Models

- UH001-UH012: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-A converters.
- UH013-UH024: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-B converters.
- UH025-UH036: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-C converters.
- UH037-UH048: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-A converters and different loss weight combinations.