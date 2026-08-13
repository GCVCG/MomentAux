# Detection (PASCAL VOC, ResNet-18 + single-level FCOS at stride 8)

Delta is aux minus baseline. AP50 is the headline; cells where BOTH
arms fall below 1.0 AP50 are UNINTERPRETABLE by the floor rule and are
marked. fg_acc and fg_iou are conditioned on ground-truth foreground and
do not floor. G is the same measure under a frozen trunk.

| pct | D AP50 | D fg_acc | D fg_iou | G fg_acc | G fg_iou | base fg_acc | branch | floor |
|---:|---:|---:|---:|---:|---:|---:|:--|:--|
| 1% | -0.04 | -0.23 | -0.0003 | -0.16 | +0.0017 | 15.31 | below | UNINTERPRETABLE |
| 2% | -0.00 | -0.23 | -0.0001 | -0.37 | -0.0006 | 21.57 | below | UNINTERPRETABLE |
| 5% | -0.08 | +0.11 | -0.0048 | +0.34 | +0.0130 | 28.89 | below |  |
| 10% | +0.56 | -0.32 | +0.0032 | +0.08 | +0.0073 | 36.53 | inside-bracket |  |
| 25% | -0.41 | +0.01 | +0.0017 | -0.47 | +0.0011 | 46.27 | above |  |
| 100% | -0.14 | +0.64 | +0.0031 | -- | -- | 64.15 | above |  |

## Excluded runs

- vocdet_none_5pct/seed2: regression branch dead (fg_iou 0.0000, fg_acc 27.86) -- excluded from Delta

## Why there is no law column

Delta = G + readout requires G on the same metric as Delta. The probe's
AP50 floors at exactly the fractions whose baseline fg_acc sits below the
crossing bracket, so no cell below the crossing is resolvable, and the
cells above it have Delta and G both near zero. Detection therefore
contributes no resolvable law cell -- the same structural outcome as
Pascal-Context on the dense side, and for the same reason.
