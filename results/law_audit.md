# Sign-law audit (machine-generated: python analysis/audit_law_paired.py)

This is the canonical audit and the source of every law number in the paper.
Uncertainty is seed-paired. Regenerate with the command in the title line.
Scope excludes 100% cells: the probe-ceiling rule refuses the G/readout split
where the evaluation's labels are the cell's own (Section: statistical protocol).

==============================================================
law-scope cells with >=3 seed-matched arms : 958
SEM(paired)/SEM(independent) median        : 0.601  (independent overstates in 82%)
inside crossing bracket (no prediction)    : 94
unresolved (|readout| <= 2.0 SEM)            : 409
RESOLVABLE (these test the law)            : 455
  sign as predicted                        : 393 (86.4%)
  wrong side                               : 62
  Wilson 95% CI                            : [82.9, 89.2]
  majority-sign baseline                   : 75.6%  (always predict the commoner sign)
==============================================================

THRESHOLD SENSITIVITY
  >1.0 SEM :  509/629  = 80.9%  [77.7, 83.8]
  >1.5 SEM :  446/531  = 84.0%  [80.6, 86.9]
  >2.0 SEM :  393/455  = 86.4%  [82.9, 89.2]
  >2.5 SEM :  343/384  = 89.3%  [85.8, 92.0]
  >3.0 SEM :  304/342  = 88.9%  [85.1, 91.8]

BY FLANK
  below crossing :  298/314  = 94.9%  [91.9, 96.8]
  above crossing :   95/141  = 67.4%  [59.3, 74.6]

CLUSTERED (one vote per dataset,backbone,fraction)
  192/235 = 81.7%  [76.3, 86.1]   (455 cells, 235 groups)

LEAVE-ONE-DATASET-OUT (bracket re-estimated without that dataset)
  cifar10      [ 23.3, 50.1]   17/20  = 85.0%
  cifar100     [ 29.5, 71.0]  129/134 = 96.3%
  cub          [ 30.0, 55.5]   35/37  = 94.6%
  dtd          [ 29.6, 54.4]   25/26  = 96.2%
  eurosat      [ 27.4, 49.7]   26/35  = 74.3%
  food101      [ 28.5, 50.1]   41/47  = 87.2%
  pathmnist    [ 27.2, 43.3]   10/19  = 52.6%
  stl10        [ 27.5, 50.7]   10/11  = 90.9%
  tin          [ 32.2, 54.5]   54/62  = 87.1%
  POOLED HELD-OUT: 347/391 = 88.7%  [85.2, 91.5]

WHAT THE READOUT DEPENDS ON (variance explained)
  baseline accuracy alone (5-point bins) : R^2 = 0.268
  + dataset   on the residual            : R^2 = 0.133  (20 levels)
  + backbone  on the residual            : R^2 = 0.003  (7 levels)
  + fraction  on the residual            : R^2 = 0.015  (10 levels)
  residual SD at fixed baseline          : 1.95 points
