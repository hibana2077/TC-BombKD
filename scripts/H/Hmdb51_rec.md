# Test on HMDB51

## Experiment Results by Configuration

| Code Range | Type | Loss Weights | Top-1 Acc (%) | Top-5 Acc (%) | ADV cls head |
|------------|------|--------------|---------------|---------------|---|
| H001 | A | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD | N |
| H002 | B | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD | N |
| H003 | C | [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD | N |
| H004 | A | [0.1, 0.1, 0.2, 0.05, 0.55, 0.0] | TBD | TBD | N |
| H005 | B | [0.1, 0.1, 0.2, 0.05, 0.55, 0.0] | TBD | TBD | N |
| H006 | C | [0.1, 0.1, 0.2, 0.05, 0.55, 0.0] | TBD | TBD | N |
| H007 | A | [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD | N |
| H008 | B | [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD | N |
| H009 | C | [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD | N |
| H010 | A | [0.0, 0.9, 0.0, 0.0, 0.1, 0.0] | TBD | TBD | N |
| H011 | C | [0.0, 0.9, 0.0, 0.0, 0.1, 0.0] | TBD | TBD | N |
| H012 | A | [0.0, 0.5, 0.2, 0.0, 0.3, 0.0] | TBD | TBD | N |
| H013 | C | [0.0, 0.5, 0.2, 0.0, 0.3, 0.0] | TBD | TBD | N |
| H014 | A | [0.2, 0.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD | N |
| H015 | C | [0.2, 0.0, 0.0, 0.0, 0.0, 0.0] | TBD | TBD | N |
| H016 | A | [0.0, 0.0, 0.1, 0.0, 0.0, 0.0] | TBD | TBD | N |
| H017 | C | [0.0, 0.0, 0.1, 0.0, 0.0, 0.0] | TBD | TBD | N |
| H018 | A | [0.1, 0.1, 0.2, 0.05, 0.55, 0.0] | TBD | TBD | Y |
| H019 | C | [0.1, 0.1, 0.2, 0.05, 0.55, 0.0] | TBD | TBD | Y |


## Loss Weights Configuration

Loss weights are configured as: [l2, cosine, nce, vicreg, barlow_twins, l1]