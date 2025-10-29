# Score records for U(UCF101) scripts

| Code | loss weight | Top1 Acc (%) | Top5 Acc (%) |
|------|-------------|---------------|---------------|
| B001-B003 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | 89.82 | 98.81 |
| B004-B006 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | TBD | TBD |

## Loss Weights

[L2, COS, NCE, VIC, BAR, L1]

## Teacher Models

- B001-B003: VideoMAE pre-trained on Kinetics400
- B004-B006: TimeSformer pre-trained on ssv2