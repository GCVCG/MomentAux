# Selection re-scored on the CIFAR-100 validation carve-out (block E)

Generated 2026-08-24 02:11 by `analysis/selection_val_rescore.py`. Every surviving checkpoint of the CIFAR-100 sweeps that selected the reference configuration is re-scored (last.pt, fp32, eval transform) on VAL of `data/valcarve/cifar100.json` -- images no cell trained on and no selection looked at -- and each selection is re-run there.

## Verdict

- Pre-registered prediction: **no axis flips on the validation carve-out**. Falsifier: any axis flips.
- Axes re-scored: 24. Axes whose numeric winner changes on val: **5** ['tap_10pct', 'lambda_const_25pct', 'endpoint_1pct', 'endpoint_2pct', 'endpoint_5pct'].
- **Scored on the letter: band MISSED; falsifier FIRED.**
- Scored on substance: of the 10 axes whose test-scored winner led by more than 2 SEM (target_5pct, target_10pct, loss_2pct, loss_10pct, lambda_const_2pct, lambda0_1pct, lambda0_2pct, lambda0_15pct, lambda0_25pct, headnorm_r50_10pct), **0 flip on val**. Every nominal flip sits on an axis whose test-scored margin was itself below 1 SEM -- a statistical tie at selection time, where the 'winner' is seed noise on either split. This resolvable/tie distinction was NOT part of the pre-registration; it qualifies the fired falsifier, it does not undo it.
- Scope caveat (recorded in advance): this BOUNDS the selection defect, it does not remove it -- the checkpoints were still selected on test.

## Validation sets

- @1%: 11200 VAL images
- @2%: 11200 VAL images
- @3%: 11200 VAL images
- @5%: 11200 VAL images
- @7%: 11200 VAL images
- @10%: 11200 VAL images
- @15%: 10579 VAL images (621 removed as overlapping the 15% training subset)
- @25%: 9352 VAL images (1848 removed as overlapping the 25% training subset)

## Val-vs-test shift (val-scored minus recorded test accuracy, per checkpoint)

- All 255 checkpoints: mean +0.33, median +0.35, sd 0.53, 25% negative.
- @1%: n=32, mean +0.00 (sd 0.24)
- @2%: n=27, mean -0.44 (sd 0.39)
- @3%: n=9, mean -0.06 (sd 0.30)
- @5%: n=50, mean +0.34 (sd 0.44)
- @7%: n=9, mean +0.58 (sd 0.35)
- @10%: n=92, mean +0.53 (sd 0.45)
- @15%: n=12, mean +0.71 (sd 0.40)
- @25%: n=24, mean +0.71 (sd 0.38)

B1 (2026-08-16) measured a ~-0.3 systematic val-vs-test shift on a mixed 8-dataset cell set; on these CIFAR-100 sweep cells the shift is +0.33 overall and FRACTION-DEPENDENT (negative at 2%, +0.5..+0.7 at 10-25%), so the sign of B1's pooled number does not transfer to this population. Within an axis the shift is shared by every arm, which is why it moves winners only where the margin is itself noise-sized.

Identity check (re-scored test vs recorded final_test_acc): 255 checkpoints, max |gap| 0.02, 0 over 0.5.

### Reading the five nominal flips

- `tap_10pct`: the test margins among layer2 / layer3 / layer2+3+4 are 0.02-0.04 points -- the recorded finding was 'tap depth is FLAT across the first three stages', and layer3 was chosen on that plateau (confirmed at 1%, which does not flip). Val reproduces the plateau with a different noise ordering (winner layer2 over layer3 by 0.28 +-0.34, under 1 SEM); the cliff at layer4 reproduces: layer3 beats layer4 by 1.89 on test and 1.58 on val.
- `lambda_const_25pct`: 0.1-vs-0.3 at +0.23 +-0.58 on test, +0.12 +-0.68 the other way on val -- a tie in both directions.
- `endpoint_{1,2,5}pct`: per-fraction endpoint margins are 0.03-0.12 points on BOTH splits, under their SEMs. The actual endpoint decision (weight_final 0 over 0.1) was made on the envelope -- primarily the 10% cell (+0.21 test / +0.14 val, no flip) and the structural neutrality argument at 100% (not re-scorable here) -- not on any of these three near-tie fractions.

## Per-axis decisions

Margins are winner-minus-arm on each split; SEM is seed-paired where the arms share seed labels (p) and independent otherwise (i). `Delta` is vs the axis baseline on the same split.

### Target family @5% (lambda=0.3 constant)

n_val=11200; baseline abl5_none test 25.36 / val 25.36; recorded selection: magnitude.  
**Winner on test: magnitude** (margin over steerable +1.63 +-0.43); **winner on val: magnitude** (margin over steerable +1.77 +-0.56) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| magnitude | 3 | 28.54 +-0.27 | 29.18 +-0.28 | +3.18 | +3.82 | -0.00 +-0.00p | -0.00 +-0.00p |
| structure | 3 | 26.44 +-0.10 | 26.92 +-0.36 | +1.07 | +1.56 | -2.10 +-0.35p | -2.26 +-0.25p |
| steerable | 3 | 26.91 +-0.34 | 27.41 +-0.29 | +1.55 | +2.05 | -1.63 +-0.43p | -1.77 +-0.56p |
| rotinv | 3 | 26.80 +-0.19 | 27.04 +-0.20 | +1.43 | +1.67 | -1.74 +-0.44p | -2.15 +-0.42p |
| invariants | 3 | 24.93 +-0.18 | 25.35 +-0.05 | -0.43 | -0.02 | -3.61 +-0.34p | -3.84 +-0.28p |
| gabor(k5 edges) | 3 | 25.92 +-0.11 | 26.16 +-0.31 | +0.56 | +0.79 | -2.62 +-0.26p | -3.03 +-0.04p |
| random-fixed | 3 | 25.24 +-0.15 | 25.53 +-0.21 | -0.12 | +0.17 | -3.30 +-0.41p | -3.65 +-0.30p |
| hog | 3 | 26.45 +-0.18 | 26.61 +-0.18 | +1.08 | +1.24 | -2.09 +-0.43p | -2.58 +-0.20p |
| fitnets-teacher | 3 | 24.87 +-0.30 | 25.07 +-0.21 | -0.49 | -0.29 | -3.67 +-0.34p | -4.11 +-0.23p |

### Target family @10% (lambda=0.3 constant)

n_val=11200; baseline abl10_none test 40.28 / val 40.49; recorded selection: magnitude.  
**Winner on test: magnitude** (margin over structure +1.03 +-0.17); **winner on val: magnitude** (margin over structure +0.88 +-0.27) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| magnitude | 3 | 42.99 +-0.03 | 43.53 +-0.17 | +2.71 | +3.05 | -0.00 +-0.00p | -0.00 +-0.00p |
| structure | 3 | 41.96 +-0.16 | 42.65 +-0.23 | +1.68 | +2.17 | -1.03 +-0.17p | -0.88 +-0.27p |
| steerable | 3 | 41.19 +-0.31 | 41.86 +-0.24 | +0.91 | +1.38 | -1.80 +-0.33p | -1.67 +-0.31p |
| rotinv | 3 | 41.02 +-0.33 | 41.10 +-0.31 | +0.74 | +0.61 | -1.97 +-0.33p | -2.43 +-0.31p |
| invariants | 3 | 37.21 +-0.19 | 37.51 +-0.06 | -3.07 | -2.97 | -5.78 +-0.18p | -6.02 +-0.12p |
| gabor(k5 edges) | 3 | 40.19 +-0.24 | 40.75 +-0.20 | -0.09 | +0.26 | -2.80 +-0.24p | -2.79 +-0.31p |
| random-fixed | 3 | 40.32 +-0.13 | 40.62 +-0.23 | +0.04 | +0.14 | -2.67 +-0.10p | -2.91 +-0.31p |
| hog | 3 | 41.55 +-0.13 | 42.12 +-0.27 | +1.27 | +1.63 | -1.43 +-0.11p | -1.42 +-0.14p |
| fitnets-teacher | 3 | 40.34 +-0.09 | 40.91 +-0.20 | +0.06 | +0.42 | -2.65 +-0.06p | -2.62 +-0.15p |

### Loss form @2% (magnitude, lambda=0.3)

n_val=11200; baseline abl2_none test 14.17 / val 13.43; recorded selection: mse.  
**Winner on test: mse** (margin over cosine +1.14 +-0.24); **winner on val: mse** (margin over cosine +1.01 +-0.26) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| mse | 3 | 15.30 +-0.25 | 14.65 +-0.17 | +1.13 | +1.21 | -0.00 +-0.00p | -0.00 +-0.00p |
| cosine | 3 | 14.16 +-0.10 | 13.64 +-0.14 | -0.01 | +0.21 | -1.14 +-0.24p | -1.01 +-0.26p |

### Loss form @10% (magnitude, lambda=0.3)

n_val=11200; baseline abl10_none test 40.28 / val 40.49; recorded selection: mse.  
**Winner on test: mse** (margin over cosine +2.35 +-0.39); **winner on val: mse** (margin over cosine +2.74 +-0.37) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| mse | 3 | 42.99 +-0.03 | 43.53 +-0.17 | +2.71 | +3.05 | -0.00 +-0.00p | -0.00 +-0.00p |
| cosine | 3 | 40.64 +-0.38 | 40.79 +-0.35 | +0.36 | +0.30 | -2.35 +-0.39p | -2.74 +-0.37p |

### Tap depth @10% (magnitude, lambda=0.3)

n_val=11200; baseline abl10_none test 40.28 / val 40.49; recorded selection: layer3.  
**Winner on test: layer2+3+4** (margin over layer2 +0.02 +-0.31); **winner on val: layer2** (margin over layer3 +0.28 +-0.34) -> **FLIP**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| layer2 | 3 | 43.01 +-0.34 | 43.82 +-0.38 | +2.73 | +3.33 | -0.02 +-0.31p | -0.00 +-0.00p |
| layer3 | 3 | 42.99 +-0.03 | 43.53 +-0.17 | +2.71 | +3.05 | -0.05 +-0.06p | -0.28 +-0.34p |
| layer4 | 3 | 41.10 +-0.59 | 41.95 +-0.59 | +0.82 | +1.46 | -1.93 +-0.61p | -1.87 +-0.61p |
| layer2+3+4 | 3 | 43.03 +-0.05 | 43.43 +-0.19 | +2.75 | +2.94 | -0.00 +-0.00p | -0.38 +-0.34p |

### Tap depth @1% (magnitude, lambda_0=2.0 -> 0)

n_val=11200; baseline abl1_none test 8.95 / val 8.83; recorded selection: layer3.  
**Winner on test: layer3** (margin over layer1 +0.04 +-0.21); **winner on val: layer3** (margin over layer1 +0.19 +-0.19) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| layer1 | 3 | 10.77 +-0.17 | 10.98 +-0.20 | +1.82 | +2.15 | -0.04 +-0.21p | -0.19 +-0.19p |
| layer2 | 3 | 10.55 +-0.05 | 10.57 +-0.10 | +1.60 | +1.74 | -0.26 +-0.01p | -0.60 +-0.16p |
| layer3 | 3 | 10.81 +-0.05 | 11.17 +-0.08 | +1.85 | +2.34 | -0.00 +-0.00p | -0.00 +-0.00p |

### Constant lambda @10%

n_val=11200; baseline abl10_none test 40.28 / val 40.49; recorded selection: 0.5.  
**Winner on test: 0.5** (margin over 1.0 +0.16 +-0.17); **winner on val: 0.5** (margin over 1.0 +0.07 +-0.33) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| 0.05 | 3 | 41.13 +-0.26 | 41.13 +-0.25 | +0.85 | +0.65 | -2.39 +-0.36p | -3.12 +-0.33p |
| 0.1 | 3 | 41.65 +-0.19 | 41.86 +-0.15 | +1.37 | +1.37 | -1.87 +-0.07p | -2.40 +-0.11p |
| 0.3 | 3 | 42.99 +-0.03 | 43.53 +-0.17 | +2.71 | +3.05 | -0.53 +-0.11p | -0.72 +-0.08p |
| 0.5 | 3 | 43.51 +-0.13 | 44.26 +-0.12 | +3.23 | +3.77 | -0.00 +-0.00p | -0.00 +-0.00p |
| 1.0 | 3 | 43.35 +-0.15 | 44.19 +-0.42 | +3.07 | +3.70 | -0.16 +-0.17p | -0.07 +-0.33p |

### Constant lambda @2%

n_val=11200; baseline abl2_none test 14.17 / val 13.43; recorded selection: 2.0.  
**Winner on test: 2.0** (margin over 1.0 +0.63 +-0.18); **winner on val: 2.0** (margin over 1.0 +1.02 +-0.04) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| 0.3 | 3 | 15.30 +-0.25 | 14.65 +-0.17 | +1.13 | +1.21 | -2.13 +-0.45p | -2.87 +-0.06p |
| 0.5 | 3 | 16.30 +-0.24 | 15.48 +-0.10 | +2.13 | +2.04 | -1.13 +-0.40p | -2.04 +-0.02p |
| 1.0 | 3 | 16.80 +-0.03 | 16.49 +-0.10 | +2.63 | +3.06 | -0.63 +-0.18p | -1.02 +-0.04p |
| 2.0 | 3 | 17.43 +-0.21 | 17.51 +-0.11 | +3.26 | +4.08 | -0.00 +-0.00p | -0.00 +-0.00p |

### Constant lambda @25%

n_val=9352; baseline abl25_none test 61.35 / val 62.01; recorded selection: 0.1.  
**Winner on test: 0.1** (margin over 0.3 +0.23 +-0.58); **winner on val: 0.3** (margin over 0.1 +0.12 +-0.68) -> **FLIP**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| 0.02 | 3 | 60.79 +-0.08 | 61.51 +-0.28 | -0.56 | -0.51 | -1.16 +-0.17p | -1.11 +-0.31p |
| 0.05 | 3 | 61.37 +-0.17 | 61.60 +-0.21 | +0.02 | -0.41 | -0.58 +-0.23p | -1.01 +-0.26p |
| 0.1 | 3 | 61.95 +-0.13 | 62.49 +-0.23 | +0.60 | +0.48 | -0.00 +-0.00p | -0.12 +-0.68p |
| 0.3 | 3 | 61.72 +-0.45 | 62.61 +-0.46 | +0.37 | +0.60 | -0.23 +-0.58p | -0.00 +-0.00p |

### Schedule vs constant @10% (the reference-configuration choice)

n_val=11200; baseline abl10_none test 40.28 / val 40.49; recorded selection: cos 1.0->0.  
**Winner on test: cos 1.0->0** (margin over cos 1.0->0.1 +0.21 +-0.18); **winner on val: cos 1.0->0** (margin over cos 1.0->0.1 +0.14 +-0.20) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| const 0.3 | 3 | 42.99 +-0.03 | 43.53 +-0.17 | +2.71 | +3.05 | -1.33 +-0.18p | -1.54 +-0.23p |
| const 0.5 | 3 | 43.51 +-0.13 | 44.26 +-0.12 | +3.23 | +3.77 | -0.81 +-0.07p | -0.82 +-0.30p |
| const 1.0 | 3 | 43.35 +-0.15 | 44.19 +-0.42 | +3.07 | +3.70 | -0.97 +-0.23p | -0.88 +-0.04p |
| cos 1.0->0.1 | 3 | 43.82 +-0.04 | 44.73 +-0.23 | +3.54 | +4.24 | -0.50 +-0.18p | -0.35 +-0.20p |
| cos 1.0->0 | 10 | 44.03 +-0.16 | 44.86 +-0.14 | +3.75 | +4.38 | -0.00 +-0.00p | -0.00 +-0.00p |

### Schedule start lambda_0 @1% (cosine -> 0)

n_val=11200; baseline abl1_none test 8.95 / val 8.83; recorded selection: 2.0.  
**Winner on test: 2.0** (margin over 1.0 +0.45 +-0.16); **winner on val: 2.0** (margin over 1.0 +0.83 +-0.17) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| 1.0 | 10 | 10.35 +-0.07 | 10.34 +-0.08 | +1.40 | +1.51 | -0.42 +-0.16p | -0.79 +-0.17p |
| 2.0 | 3 | 10.81 +-0.05 | 11.17 +-0.08 | +1.85 | +2.34 | -0.00 +-0.00p | -0.00 +-0.00p |

### Schedule start lambda_0 @2% (cosine -> 0)

n_val=11200; baseline abl2_none test 14.17 / val 13.43; recorded selection: 2.0.  
**Winner on test: 2.0** (margin over 1.0 +0.64 +-0.27); **winner on val: 2.0** (margin over 1.0 +1.12 +-0.07) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| 1.0 | 3 | 16.67 +-0.13 | 16.13 +-0.09 | +2.50 | +2.69 | -0.64 +-0.27p | -1.12 +-0.07p |
| 2.0 | 3 | 17.31 +-0.19 | 17.25 +-0.02 | +3.14 | +3.81 | -0.00 +-0.00p | -0.00 +-0.00p |

### Schedule start lambda_0 @15% (cosine -> 0)

n_val=10579; baseline abl15_none test 50.09 / val 50.75; recorded selection: 0.3.  
**Winner on test: 0.3** (margin over 1.0 +0.39 +-0.07); **winner on val: 0.3** (margin over 1.0 +0.65 +-0.23) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| 1.0 | 3 | 52.64 +-0.15 | 53.26 +-0.06 | +2.55 | +2.51 | -0.39 +-0.07p | -0.65 +-0.23p |
| 0.3 | 3 | 53.03 +-0.10 | 53.91 +-0.23 | +2.94 | +3.16 | -0.00 +-0.00p | -0.00 +-0.00p |

### Schedule start lambda_0 @25% (cosine -> 0)

n_val=9352; baseline abl25_none test 61.35 / val 62.01; recorded selection: 0.3.  
**Winner on test: 0.3** (margin over 1.0 +0.72 +-0.11); **winner on val: 0.3** (margin over 1.0 +0.51 +-0.24) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| 1.0 | 3 | 61.51 +-0.12 | 62.52 +-0.11 | +0.16 | +0.51 | -0.72 +-0.11p | -0.51 +-0.24p |
| 0.3 | 3 | 62.23 +-0.15 | 63.03 +-0.14 | +0.88 | +1.02 | -0.00 +-0.00p | -0.00 +-0.00p |

### Head-norm on ResNet-18 @10% (lambda_0=1.0 -> 0)

n_val=11200; baseline abl10_none test 40.28 / val 40.49; recorded selection: tie (adopted always-on as free).  
**Winner on test: head_norm** (margin over no head_norm +0.42 +-0.29); **winner on val: head_norm** (margin over no head_norm +0.14 +-0.28) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| no head_norm | 10 | 44.03 +-0.16 | 44.86 +-0.14 | +3.75 | +4.38 | -0.25 +-0.29p | -0.09 +-0.28p |
| head_norm | 6 | 44.45 +-0.28 | 45.00 +-0.27 | +4.17 | +4.52 | -0.00 +-0.00p | -0.00 +-0.00p |

### Head-norm on ResNet-50 @10%

n_val=11200; baseline r50_none_10pct test 40.65 / val 40.90; recorded selection: head_norm, lambda_0=1.0.  
**Winner on test: head_norm, lambda_0=1.0** (margin over no head_norm, lambda_0=0.3 +1.51 +-0.37); **winner on val: head_norm, lambda_0=1.0** (margin over no head_norm, lambda_0=0.3 +1.86 +-0.47) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| no head_norm, lambda_0=1.0 | 3 | 39.98 +-1.83 | 41.14 +-1.80 | -0.67 | +0.24 | -4.60 +-2.02p | -4.20 +-1.98p |
| no head_norm, lambda_0=0.3 | 3 | 43.07 +-0.20 | 43.49 +-0.29 | +2.42 | +2.59 | -1.51 +-0.37p | -1.86 +-0.47p |
| head_norm, lambda_0=1.0 | 3 | 44.58 +-0.19 | 45.35 +-0.20 | +3.93 | +4.45 | -0.00 +-0.00p | -0.00 +-0.00p |

### Schedule endpoint @1% (lambda_0=1.0 cosine)

n_val=11200; baseline abl1_none test 8.95 / val 8.83; recorded selection: -> 0.  
**Winner on test: -> 0.1** (margin over -> 0 +0.08 +-0.11); **winner on val: -> 0** (margin over -> 0.1 +0.05 +-0.03) -> **FLIP**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| -> 0 | 10 | 10.35 +-0.07 | 10.34 +-0.08 | +1.40 | +1.51 | -0.04 +-0.11p | -0.00 +-0.00p |
| -> 0.1 | 3 | 10.43 +-0.09 | 10.29 +-0.17 | +1.48 | +1.46 | -0.00 +-0.00p | -0.08 +-0.03p |

### Schedule endpoint @2% (lambda_0=1.0 cosine)

n_val=11200; baseline abl2_none test 14.17 / val 13.43; recorded selection: -> 0.  
**Winner on test: -> 0** (margin over -> 0.1 +0.09 +-0.30); **winner on val: -> 0.1** (margin over -> 0 +0.11 +-0.19) -> **FLIP**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| -> 0 | 3 | 16.67 +-0.13 | 16.13 +-0.09 | +2.50 | +2.69 | -0.00 +-0.00p | -0.11 +-0.19p |
| -> 0.1 | 3 | 16.58 +-0.21 | 16.24 +-0.11 | +2.41 | +2.80 | -0.09 +-0.30p | -0.00 +-0.00p |

### Schedule endpoint @3% (lambda_0=1.0 cosine)

n_val=11200; baseline abl3_none test 18.47 / val 18.35; recorded selection: -> 0.  
**Winner on test: -> 0.1** (margin over -> 0 +0.12 +-0.16); **winner on val: -> 0.1** (margin over -> 0 +0.00 +-0.20) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| -> 0 | 3 | 22.15 +-0.29 | 22.17 +-0.09 | +3.68 | +3.82 | -0.12 +-0.16p | -0.00 +-0.20p |
| -> 0.1 | 3 | 22.27 +-0.22 | 22.18 +-0.13 | +3.80 | +3.82 | -0.00 +-0.00p | -0.00 +-0.00p |

### Schedule endpoint @5% (lambda_0=1.0 cosine)

n_val=11200; baseline abl5_none test 25.36 / val 25.36; recorded selection: -> 0.  
**Winner on test: -> 0.1** (margin over -> 0 +0.03 +-0.41); **winner on val: -> 0** (margin over -> 0.1 +0.12 +-0.23) -> **FLIP**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| -> 0 | 10 | 30.51 +-0.12 | 31.13 +-0.13 | +5.15 | +5.77 | -0.01 +-0.41p | -0.00 +-0.00p |
| -> 0.1 | 3 | 30.54 +-0.40 | 31.01 +-0.48 | +5.17 | +5.65 | -0.00 +-0.00p | -0.23 +-0.23p |

### Schedule endpoint @7% (lambda_0=1.0 cosine)

n_val=11200; baseline abl7_none test 32.09 / val 32.40; recorded selection: -> 0.  
**Winner on test: -> 0** (margin over -> 0.1 +0.08 +-0.29); **winner on val: -> 0** (margin over -> 0.1 +0.24 +-0.07) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| -> 0 | 3 | 36.96 +-0.33 | 37.75 +-0.19 | +4.87 | +5.35 | -0.00 +-0.00p | -0.00 +-0.00p |
| -> 0.1 | 3 | 36.88 +-0.05 | 37.51 +-0.14 | +4.79 | +5.11 | -0.08 +-0.29p | -0.24 +-0.07p |

### Schedule endpoint @10% (lambda_0=1.0 cosine)

n_val=11200; baseline abl10_none test 40.28 / val 40.49; recorded selection: -> 0.  
**Winner on test: -> 0** (margin over -> 0.1 +0.21 +-0.18); **winner on val: -> 0** (margin over -> 0.1 +0.14 +-0.20) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| -> 0 | 10 | 44.03 +-0.16 | 44.86 +-0.14 | +3.75 | +4.38 | -0.00 +-0.00p | -0.00 +-0.00p |
| -> 0.1 | 3 | 43.82 +-0.04 | 44.73 +-0.23 | +3.54 | +4.24 | -0.50 +-0.18p | -0.35 +-0.20p |

### Schedule endpoint @15% (lambda_0=1.0 cosine)

n_val=10579; baseline abl15_none test 50.09 / val 50.75; recorded selection: -> 0.  
**Winner on test: -> 0** (margin over -> 0.1 +0.32 +-0.27); **winner on val: -> 0** (margin over -> 0.1 +0.30 +-0.47) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| -> 0 | 3 | 52.64 +-0.15 | 53.26 +-0.06 | +2.55 | +2.51 | -0.00 +-0.00p | -0.00 +-0.00p |
| -> 0.1 | 3 | 52.32 +-0.26 | 52.97 +-0.48 | +2.23 | +2.22 | -0.32 +-0.27p | -0.30 +-0.47p |

### Schedule endpoint @25% (lambda_0=1.0 cosine)

n_val=9352; baseline abl25_none test 61.35 / val 62.01; recorded selection: -> 0.  
**Winner on test: -> 0.1** (margin over -> 0 +0.28 +-0.23); **winner on val: -> 0.1** (margin over -> 0 +0.10 +-0.37) -> **no flip**

| arm | n | test mean | val mean | Delta test | Delta val | test margin to winner | val margin to winner |
|---|---|---|---|---|---|---|---|
| -> 0 | 3 | 61.51 +-0.12 | 62.52 +-0.11 | +0.16 | +0.51 | -0.28 +-0.23p | -0.10 +-0.37p |
| -> 0.1 | 3 | 61.79 +-0.25 | 62.62 +-0.36 | +0.44 | +0.61 | -0.00 +-0.00p | -0.00 +-0.00p |

## Not re-scorable

- lambda0 @100% (sched0 / sched03 / sched01) and constant lambda @100%: 100% cells trained on every train image; no validation carve-out exists (the carve-out is drawn from the unused portion of train).

## Inventory

| cell | config | finals | last.pt found | scored | status | trees |
|---|---|---|---|---|---|---|
| abl10_none | configs/ablations_full/abl10_none.yaml | 10 | 10 | 10 | {'ok': 10} | runs |
| abl15_none | configs/diagnostics/abl15_none.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| abl1_none | configs/ablations_full/abl1_none.yaml | 10 | 10 | 10 | {'ok': 10} | runs |
| abl25_none | configs/diagnostics/abl25_none.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| abl2_none | configs/diagnostics/abl2_none.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| abl3_none | configs/diagnostics/abl3_none.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| abl5_none | configs/ablations_full/abl5_none.yaml | 10 | 10 | 10 | {'ok': 10} | runs |
| abl7_none | configs/diagnostics/abl7_none.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxgab_10pct_l30 | configs/diagnostics/auxgab_10pct_l30.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxhog_10pct | configs/diagnostics/auxhog_10pct.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxhog_5pct | configs/diagnostics/auxhog_5pct.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxinv_10pct_l30 | configs/diagnostics/auxinv_10pct_l30.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_10pct_cos | configs/diagnostics/auxmag_10pct_cos.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_10pct_l05 | configs/diagnostics/auxmag_10pct_l05.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxmag_10pct_l10 | configs/diagnostics/auxmag_10pct_l10.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxmag_10pct_l100 | configs/diagnostics/auxmag_10pct_l100.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxmag_10pct_l30 | configs/diagnostics/auxmag_10pct_l30.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxmag_10pct_l30_tap2 | configs/diagnostics/auxmag_10pct_l30_tap2.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxmag_10pct_l30_tap234 | configs/diagnostics/auxmag_10pct_l30_tap234.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxmag_10pct_l30_tap4 | configs/diagnostics/auxmag_10pct_l30_tap4.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxmag_10pct_l50 | configs/diagnostics/auxmag_10pct_l50.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxmag_10pct_sched | configs/diagnostics/auxmag_10pct_sched.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_10pct_sched0 | configs/diagnostics/auxmag_10pct_sched0.yaml | 10 | 10 | 10 | {'ok': 10} | runs |
| auxmag_15pct_sched | configs/diagnostics/auxmag_15pct_sched.yaml | 3 | 3 | 3 | {'ok': 3} | runs_bscpull |
| auxmag_15pct_sched0 | configs/diagnostics/auxmag_15pct_sched0.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_15pct_sched03 | configs/diagnostics/auxmag_15pct_sched03.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_1pct_sched | configs/diagnostics/auxmag_1pct_sched.yaml | 3 | 3 | 3 | {'ok': 3} | runs_bscpull |
| auxmag_1pct_sched0 | configs/diagnostics/auxmag_1pct_sched0.yaml | 10 | 10 | 10 | {'ok': 10} | runs |
| auxmag_1pct_sched2 | configs/diagnostics/auxmag_1pct_sched2.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_1pct_sched2_tap1 | configs/diagnostics/auxmag_1pct_sched2_tap1.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_1pct_sched2_tap2 | configs/diagnostics/auxmag_1pct_sched2_tap2.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_25pct_l02 | configs/diagnostics/auxmag_25pct_l02.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_25pct_l05 | configs/diagnostics/auxmag_25pct_l05.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_25pct_l10 | configs/diagnostics/auxmag_25pct_l10.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxmag_25pct_l30 | configs/diagnostics/auxmag_25pct_l30.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxmag_25pct_sched | configs/diagnostics/auxmag_25pct_sched.yaml | 3 | 3 | 3 | {'ok': 3} | runs_bscpull |
| auxmag_25pct_sched0 | configs/diagnostics/auxmag_25pct_sched0.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_25pct_sched03 | configs/diagnostics/auxmag_25pct_sched03.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_2pct_cos | configs/diagnostics/auxmag_2pct_cos.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_2pct_l100 | configs/diagnostics/auxmag_2pct_l100.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_2pct_l200 | configs/diagnostics/auxmag_2pct_l200.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_2pct_l30 | configs/diagnostics/auxmag_2pct_l30.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxmag_2pct_l50 | configs/diagnostics/auxmag_2pct_l50.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 2, 'ok(aux-head-skipped)': 1} | runs |
| auxmag_2pct_sched | configs/diagnostics/auxmag_2pct_sched.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_2pct_sched0 | configs/diagnostics/auxmag_2pct_sched0.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_2pct_sched2 | configs/diagnostics/auxmag_2pct_sched2.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_3pct_sched | configs/diagnostics/auxmag_3pct_sched.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_3pct_sched0 | configs/diagnostics/auxmag_3pct_sched0.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_5pct_l30 | configs/diagnostics/auxmag_5pct_l30.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxmag_5pct_sched | configs/diagnostics/auxmag_5pct_sched.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxmag_5pct_sched0 | configs/diagnostics/auxmag_5pct_sched0.yaml | 10 | 10 | 10 | {'ok': 10} | runs |
| auxmag_7pct_sched | configs/diagnostics/auxmag_7pct_sched.yaml | 3 | 3 | 3 | {'ok': 3} | runs_bscpull |
| auxmag_7pct_sched0 | configs/diagnostics/auxmag_7pct_sched0.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxrand_10pct_l30 | configs/diagnostics/auxrand_10pct_l30.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxrand_5pct_l30 | configs/diagnostics/auxrand_5pct_l30.yaml | 3 | 3 | 3 | {'ok(legacy-keys-remapped)': 3} | runs |
| auxrot_10pct_l30 | configs/diagnostics/auxrot_10pct_l30.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxste_10pct_l30 | configs/diagnostics/auxste_10pct_l30.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxstr_10pct_l30 | configs/diagnostics/auxstr_10pct_l30.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxteach_10pct | configs/diagnostics/auxteach_10pct.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| auxteach_5pct | configs/diagnostics/auxteach_5pct.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| grid_c100_r18_axinvariantsL3_l03_e17c19_5pct | configs/grid/grid_c100_r18_axinvariantsL3_l03_e17c19_5pct.yaml | 3 | 3 | 3 | {'ok': 3} | runs,runs_bscpull |
| grid_c100_r18_axmomentscatL3_l03_d020d0_5pct | configs/grid/grid_c100_r18_axmomentscatL3_l03_d020d0_5pct.yaml | 3 | 3 | 3 | {'ok': 3} | runs,runs_bscpull |
| grid_c100_r18_axrotinvL3_l03_a9b216_5pct | configs/grid/grid_c100_r18_axrotinvL3_l03_a9b216_5pct.yaml | 3 | 3 | 3 | {'ok': 3} | runs,runs_bscpull |
| grid_c100_r18_axsteerableL3_l03_db5093_5pct | configs/grid/grid_c100_r18_axsteerableL3_l03_db5093_5pct.yaml | 3 | 3 | 3 | {'ok': 3} | runs,runs_bscpull |
| grid_c100_r18_axstructureL3_l03_26d33f_5pct | configs/grid/grid_c100_r18_axstructureL3_l03_26d33f_5pct.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| r18_aux_10pct_hn | configs/diagnostics/r18_aux_10pct_hn.yaml | 6 | 6 | 6 | {'ok': 6} | runs |
| r50_aux_10pct | configs/diagnostics/r50_aux_10pct.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| r50_aux_10pct_hn | configs/diagnostics/r50_aux_10pct_hn.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| r50_aux_10pct_l03 | configs/diagnostics/r50_aux_10pct_l03.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
| r50_none_10pct | configs/diagnostics/r50_none_10pct.yaml | 3 | 3 | 3 | {'ok': 3} | runs |
