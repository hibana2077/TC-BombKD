# UAV-Human Dataset Score

| Code | Type | loss weight | Top1 Acc (%) | Top5 Acc (%) |
|------|------|-------------|---------------|---------------|
| UH001-UH003 | A | [0.0, 0.0, 0.0, 0.0, 1.0, 0.0] | 19.51 | 42.42 |
| UH004-UH006 | A | [0.0, 0.0, 0.0, 1.0, 0.0, 0.0] | 29.79 | 56.23 |
| UH007-UH009 | A | [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] | 32.90 | 58.31 |
| UH010-UH012 | A | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | 31.85 | 57.89 |
| UH013-UH015 | B | [0.1, 1.0, 0.0, 0.0, 0.0, 0.0] | 33.32 | 59.07 |
| UH016-UH018 | B | [0.1, 0.0, 1.0, 0.0, 0.0, 0.0] | 30.77 | 57.24 |
| UH019-UH021 | B | [0.1, 0.0, 0.0, 1.0, 0.0, 0.0] | 34.50 | 60.15 |
| UH022-UH024 | B | [0.1, 0.0, 0.0, 0.0, 1.0, 0.0] | 24.68 | 49.44 |
| UH025-UH027 | C | [0.0, 0.9, 0.1, 0.0, 0.0, 0.0] | 36.50 | 62.46 |
| UH028-UH030 | C | [0.0, 0.9, 0.0, 0.1, 0.0, 0.0] | 17.91 | 38.60 |
| UH031-UH033 | C | [0.0, 0.9, 0.0, 0.0, 0.1, 0.0] | 31.63 | 59.21 |
| UH034-UH036 | C | [0.0, 0.9, 0.0, 0.0, 0.0, 0.1] | 35.15 | 60.98 |
| UH037-UH039 | A | [0.0, 0.8, 0.2, 0.0, 0.0, 0.0] | 34.15 | 59.62 |
| UH040-UH042 | A | [0.0, 0.7, 0.3, 0.0, 0.0, 0.0] | 32.71 | 58.33 |
| UH043-UH045 | A | [0.0, 0.6, 0.4, 0.0, 0.0, 0.0] | 35.33 | 61.04 |
| UH046-UH048 | A | [0.0, 0.5, 0.5, 0.0, 0.0, 0.0] | 32.45 | 58.10 |
| UH049-UH051 | A | [0.1, 0.4, 0.5, 0.0, 0.0, 0.0] | 27.67 | 53.42 |
| UH052-UH054 | A | [0.0, 0.4, 0.5, 0.1, 0.0, 0.0] | 28.06 | 54.43 |
| UH055-UH057 | A | [0.1, 0.3, 0.5, 0.1, 0.0, 0.0] | 0.76 | 3.73 |
| UH058-UH060 | A | [0.0, 0.3, 0.6, 0.1, 0.0, 0.0] | 26.39 | 51.75 |
| UH061-UH063 | A | [0.05, 0.35, 0.55, 0.05, 0.0, 0.0] | 27.20 | 52.90 |
| UH064-UH066 | A | [0.1, 0.5, 0.3, 0.1, 0.0, 0.0] | 28.02 | 55.55 |
| UH067-UH069 | A | [0.05, 0.4, 0.45, 0.1, 0.0, 0.0] | 27.81 | 54.34 |
| UH070-UH072 | A | [0.1, 0.45, 0.4, 0.05, 0.0, 0.0] | 17.78 | 37.88 |
| UH073-UH075 | A | [0.0, 0.45, 0.45, 0.05, 0.05, 0.0] | 34.21 | 60.88 |
| UH076-UH078 | A | [0.1, 0.4, 0.4, 0.1, 0.0, 0.0] | 26.13 | 52.59 |

## Loss Weights

[L2, COS, NCE, VIC, BAR, L1]

## Teacher Models

- UH001-UH012: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-A converters.
- UH013-UH024: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-B converters.
- UH025-UH036: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-C converters.
- UH037-UH048: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-A converters and different loss weight combinations.
- UH049-UH078: {VideoMAE, TimeSformer, Vivit} pre-trained on Kinetics400 and ssv2 with Type-A converters and extended loss weight combinations.