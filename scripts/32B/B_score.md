# Score records for B(Breakfast) with 32 frames scripts

| Code | loss weight | Top1 Acc (%) | Top5 Acc (%) |
|------|-------------|---------------|---------------|
| B001-B003 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | 67.38 | 98.17 |
| B004-B006 | [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] | 53.35 | 92.68 |
| B007-B009 | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |

## Loss Weights

[L2, COS, NCE, VIC, BAR, L1]

## Teacher Models

- B001-B006: VideoMAE pre-trained on Kinetics400
- B007-B009: {VideoMAE pre-trained on Kinetics400 | TimeSformer pre-trained on ssv2 | ViViT pre-trained on Kinetics400}