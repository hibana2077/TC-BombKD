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

## Loss Weights Configuration

Loss weights are configured as: [l2, cosine, nce, vicreg, barlow_twins, l1]