# Test on Diving48

## Loss Weights

[l2, cosine, nce, vicreg, barlow_twins, l1]

## Different Settings

| Code | Loss Weights | Top1 Acc (%) | Top5 Acc (%) |
|------|--------------|--------------|--------------|
| D001-D003 | [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |
| D004-D006 | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |
| D007-D009 | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | TBD | TBD |
| D010-D012 | [0.0, 0.0, 0.0, 1.0, 0.0, 0.0] | TBD | TBD |
| D013-D015 | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | TBD | TBD |
| D016-D018 | [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] | TBD | TBD |
| D019-D021 | [0.1, 0.1, 0.1, 0.1, 0.1, 0.5] | TBD | TBD |
| D022-D024 | [0.2, 0.2, 0.2, 0.2, 0.1, 0.1] | TBD | TBD |
| D025-D027 | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD |
| D028-D030 | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | TBD | TBD |

## Teacher Models

- D001-D024: {VideoMAE pre-trained on Kinetics400 | TimeSformer pre-trained on ssv2 | ViViT pre-trained on Kinetics400} with Type-A converters.
- D025-D030: {VideoMAE, TimeSformer, ViViT} pre-trained on Kinetics400 and ssv2 with Type-B converters.
