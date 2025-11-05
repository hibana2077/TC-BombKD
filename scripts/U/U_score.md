# Score records for U(UCF101) scripts

| Code | loss weight | Top1 Acc (%) | Top5 Acc (%) |
|------|-------------|---------------|---------------|
| U001-U003 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | 89.82 | 98.81 |
| U004-U006 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | 88.32 | 97.91 |
| U007-U009 | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | TBD | TBD |
| U010-U012 | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |
| U013-U015 | [0.0, 0.0, 0.0, 1.0, 0.0, 0.0] | TBD | TBD |
| U016-U018 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | TBD | TBD |

## Loss Weights

[L2, COS, NCE, VIC, BAR, L1]

## Teacher Models

- U001-U003: VideoMAE pre-trained on Kinetics400
- U004-U006: TimeSformer pre-trained on ssv2
- U007-U018: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2