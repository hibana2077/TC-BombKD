# Score records for U(UCF101) scripts

| Code | loss weight | Top1 Acc (%) | Top5 Acc (%) |
|------|-------------|---------------|---------------|
| U001-U003 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | 89.82 | 98.81 |
| U004-U006 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | 88.32 | 97.91 |
| U007-U009 | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | 91.83 | 98.19 |
| U010-U012 | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | 92.55 | 99.17 |
| U013-U015 | [0.0, 0.0, 0.0, 1.0, 0.0, 0.0] | 91.78 | 99.33 |
| U016-U018 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | 91.99 | 99.28 |
| U019-U021 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | 91.78 | 98.36 |
| U022-U024 | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | 89.69 | 98.26 |
| U025-U027 | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | 90.80 | 98.26 |
| U028-U030 | [0.0, 0.0, 0.0, 1.0, 0.0, 0.0] | 86.94 | 97.78 |
| U031-U033 | [0.1, 0.9, 0.0, 0.0, 0.0, 0.0] | 90.38 | 98.39 |
| U034-U036 | [0.0, 0.9, 0.1, 0.0, 0.0, 0.0] | 89.14 | 98.15 |
| U037-U039 | [0.0, 0.9, 0.0, 0.1, 0.0, 0.0] | 85.78 | 97.67 |
| U040-U042 | [0.0, 0.9, 0.0, 0.0, 0.1, 0.0] | 89.53 | 98.18 |
| U043-U045 | [0.0, 0.9, 0.0, 0.0, 0.0, 0.1] | 88.53 | 98.28 |
| U046-U048 | [0.02, 0.98, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |

## Loss Weights

[L2, COS, NCE, VIC, BAR, L1]

## Teacher Models

- U001-U003: VideoMAE pre-trained on Kinetics400
- U004-U006: TimeSformer pre-trained on ssv2
- U007-U018: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2, with Type-A converters.
- U019-U024: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2, with Type-B converters.
- U025-U045: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2, with Type-C converters.
- U046-U048: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2, with Type-C converters and different loss weight combinations.