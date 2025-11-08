# Test on Diving48

## Loss Weights

[l2, cosine, nce, vicreg, barlow_twins, l1]

## Different Settings

| Code | Loss Weights | Top1 Acc (%) | Top5 Acc (%) |
|------|--------------|--------------|--------------|
| D001-D003 | [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] | 23.65 | 60.56 |
| D004-D006 | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | 58.93 | 91.78 |
| D007-D009 | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | 52.13 | 90.46 |
| D010-D012 | [0.0, 0.0, 0.0, 1.0, 0.0, 0.0] | 19.70 | 52.99 |
| D013-D015 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | 23.25 | 57.72 |
| D016-D018 | [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] | 23.20 | 59.24 |
| D019-D021 | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | 41.88 | 84.37 |
| D022-D024 | [0.2, 0.2, 0.2, 0.2, 0.1, 0.1] | 41.12 | 84.52 |
| D025-D027 | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |
| D028-D030 | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | TBD | TBD |
| D031-D033 | [0.0, 0.0, 0.0, 1.0, 0.0, 0.0] | TBD | TBD |
| D034-D036 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | TBD | TBD |
| D037-D039 | [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] | TBD | TBD |
| D040-D042 | [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] | TBD | TBD |
| D043-D045 | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |

## Teacher Models

- D001-D024: {VideoMAE pre-trained on Kinetics400 | TimeSformer pre-trained on ssv2 | ViViT pre-trained on Kinetics400} with Type-A converters.
- D025-D039: {VideoMAE, TimeSformer, ViViT} pre-trained on Kinetics400 and ssv2 with Type-B converters.
- D040-D045: {VideoMAE, TimeSformer, ViViT} pre-trained on Kinetics400 and ssv2 with Type-C converters.