# Sign-law audit (machine-generated: python analysis/audit_law_paired.py)

This is the canonical audit and the source of every law number in the paper.
Uncertainty is seed-paired. Regenerate with the command in the title line.
Probe file: linear_probe_last.json   (matched-epoch corpus: the released one, every G from last.pt)
Scope excludes 100% cells: the probe-ceiling rule refuses the G/readout split
where the evaluation's labels are the cell's own (Section: statistical protocol).

==============================================================
law-scope cells with >=3 seed-matched arms : 955
SEM(paired)/SEM(independent) median        : 0.541  (independent overstates in 86%)
inside crossing bracket (no prediction)    : 93
unresolved (|readout| <= 2.0 SEM)            : 377
RESOLVABLE (these test the law)            : 485
  sign as predicted                        : 408 (84.1%)
  wrong side                               : 77
  Wilson 95% CI                            : [80.6, 87.1]
  majority-sign baseline                   : 78.4%  (always predict the commoner sign)
==============================================================

THRESHOLD SENSITIVITY
  >1.0 SEM :  502/635  = 79.1%  [75.7, 82.0]
  >1.5 SEM :  456/553  = 82.5%  [79.1, 85.4]
  >2.0 SEM :  408/485  = 84.1%  [80.6, 87.1]
  >2.5 SEM :  362/425  = 85.2%  [81.5, 88.2]
  >3.0 SEM :  323/371  = 87.1%  [83.3, 90.1]

BY FLANK
  below crossing :  320/337  = 95.0%  [92.1, 96.8]
  above crossing :   88/148  = 59.5%  [51.4, 67.0]

CLUSTERED (one vote per dataset,backbone,fraction)
  194/245 = 79.2%  [73.7, 83.8]   (485 cells, 245 groups)

LEAVE-ONE-DATASET-OUT (bracket re-estimated without that dataset)
  cifar10      [ 22.7, 59.8]   10/13  = 76.9%
  cifar100     [ 27.1, 71.7]  131/151 = 86.8%
  cub          [ 26.1, 71.8]   41/43  = 95.3%
  dtd          [ 25.2, 71.8]   25/26  = 96.2%
  eurosat      [ 23.2, 58.5]   24/30  = 80.0%
  food101      [ 24.4, 71.8]   39/40  = 97.5%
  pathmnist    [ 23.9, 50.5]    8/18  = 44.4%
  stl10        [ 24.7, 71.8]    8/9   = 88.9%
  tin          [ 27.8, 71.8]   55/59  = 93.2%
  POOLED HELD-OUT: 341/389 = 87.7%  [84.0, 90.6]

WHAT THE READOUT DEPENDS ON (variance explained)
  baseline accuracy alone (5-point bins) : R^2 = 0.300
  + dataset   on the residual            : R^2 = 0.151  (20 levels)
  + backbone  on the residual            : R^2 = 0.010  (7 levels)
  + fraction  on the residual            : R^2 = 0.012  (10 levels)
  residual SD at fixed baseline          : 1.71 points
