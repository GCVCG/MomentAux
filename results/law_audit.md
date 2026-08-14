# Sign-law audit (machine-generated: python analysis/audit_law_paired.py)

This is the canonical audit and the source of every law number in the paper.
Uncertainty is seed-paired. Regenerate with the command in the title line.

==============================================================
law-scope cells with >=3 seed-matched arms : 1009
SEM(paired)/SEM(independent) median        : 0.590  (independent overstates in 82%)
inside crossing bracket (no prediction)    : 98
unresolved (|readout| <= 2.0 SEM)            : 450
RESOLVABLE (these test the law)            : 461
  sign as predicted                        : 395 (85.7%)
  wrong side                               : 66
  Wilson 95% CI                            : [82.2, 88.6]
==============================================================

THRESHOLD SENSITIVITY
  >1.0 SEM :  511/643  = 79.5%  [76.2, 82.4]
  >1.5 SEM :  448/539  = 83.1%  [79.7, 86.0]
  >2.0 SEM :  395/461  = 85.7%  [82.2, 88.6]
  >2.5 SEM :  345/388  = 88.9%  [85.4, 91.7]
  >3.0 SEM :  305/344  = 88.7%  [84.9, 91.6]

BY FLANK
  below crossing :  293/311  = 94.2%  [91.0, 96.3]
  above crossing :  102/150  = 68.0%  [60.2, 74.9]

CLUSTERED (one vote per dataset,backbone,fraction)
  200/247 = 81.0%  [75.6, 85.4]   (461 cells, 247 groups)

LEAVE-ONE-DATASET-OUT (bracket re-estimated without that dataset)
  cifar10      [ 22.9, 50.1]   18/21  = 85.7%
  cifar100     [ 27.1, 66.6]  125/130 = 96.2%
  cub          [ 29.5, 57.0]   35/37  = 94.6%
  dtd          [ 29.5, 56.3]   25/27  = 92.6%
  eurosat      [ 24.8, 50.1]   25/33  = 75.8%
  food101      [ 27.0, 54.2]   38/42  = 90.5%
  pathmnist    [ 23.8, 44.3]   11/20  = 55.0%
  stl10        [ 24.4, 53.7]    6/7   = 85.7%
  tin          [ 32.1, 50.5]   55/67  = 82.1%
  POOLED HELD-OUT: 338/384 = 88.0%  [84.4, 90.9]

WHAT THE READOUT DEPENDS ON (variance explained)
  baseline accuracy alone (5-point bins) : R^2 = 0.255
  + dataset   on the residual            : R^2 = 0.128  (21 levels)
  + backbone  on the residual            : R^2 = 0.010  (9 levels)
  + fraction  on the residual            : R^2 = 0.020  (11 levels)
  residual SD at fixed baseline          : 1.95 points
