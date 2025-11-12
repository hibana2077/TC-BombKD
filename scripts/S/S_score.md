# UAV-Human Dataset Score

| Code | loss weight | Top1 Acc (%) | Top5 Acc (%) |
|------|-------------|---------------|---------------|
| S001-S003 | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | TBD | TBD |
| S004-S006 | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | TBD | TBD |
| S007-S009 | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | TBD | TBD |
| S010-S012 | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |
| S013-S015 | [0.0, 1.0, 0.1, 0.0, 0.0, 0.0] | TBD | TBD |
| S016-S018 | [0.0, 1.0, 0.0, 0.1, 0.0, 0.0] | TBD | TBD |
| S019-S021 | [0.0, 1.0, 0.0, 0.0, 0.1, 0.0] | TBD | TBD |
| S022-S024 | [0.1, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |


## Loss Weights

[L2, COS, NCE, VIC, BAR, L1]

## Teacher Models

- S001-S003: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-A converters.
- S004-S006: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-B converters.
- S007-S009: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-C converters.
- S010-S012: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-A converters and different loss weight combinations.