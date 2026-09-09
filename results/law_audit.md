# Sign-law audit (machine-generated: python analysis/audit_law_paired.py)

This is the canonical audit and the source of every law number in the paper.
Uncertainty is seed-paired. Regenerate with the command in the title line.
Probe file: linear_probe_last.json   (matched-epoch corpus: the released one, every G from last.pt)
Scope excludes 100% cells: the probe-ceiling rule refuses the G/readout split
where the evaluation's labels are the cell's own (Section: statistical protocol).

==============================================================
law-scope cells with >=3 seed-matched arms : 991
SEM(paired)/SEM(independent) median        : 0.543  (independent overstates in 86%)
inside crossing bracket (no prediction)    : 103
unresolved (|readout| <= 2.0 SEM)            : 394
RESOLVABLE (these test the law)            : 494
  sign as predicted                        : 415 (84.0%)
  wrong side                               : 79
  Wilson 95% CI                            : [80.5, 87.0]
  majority-sign baseline                   : 78.7%  (always predict the commoner sign)
==============================================================

THRESHOLD SENSITIVITY
  >1.0 SEM :  513/651  = 78.8%  [75.5, 81.8]
  >1.5 SEM :  463/563  = 82.2%  [78.9, 85.2]
  >2.0 SEM :  415/494  = 84.0%  [80.5, 87.0]
  >2.5 SEM :  368/432  = 85.2%  [81.5, 88.2]
  >3.0 SEM :  328/376  = 87.2%  [83.5, 90.2]

BY FLANK
  below crossing :  327/344  = 95.1%  [92.2, 96.9]
  above crossing :   88/150  = 58.7%  [50.7, 66.2]

CLUSTERED (one vote per dataset,backbone,fraction)
  197/250 = 78.8%  [73.3, 83.4]   (494 cells, 250 groups)

LEAVE-ONE-DATASET-OUT (bracket re-estimated without that dataset)
  cifar10      [ 22.7, 59.8]   10/13  = 76.9%
  cifar100     [ 27.1, 71.7]  134/154 = 87.0%
  cub          [ 26.1, 71.8]   41/43  = 95.3%
  dtd          [ 25.2, 71.8]   25/26  = 96.2%
  eurosat      [ 23.2, 58.5]   24/30  = 80.0%
  food101      [ 24.4, 71.8]   39/40  = 97.5%
  pathmnist    [ 23.9, 50.1]    8/19  = 42.1%
  stl10        [ 24.7, 71.8]    8/9   = 88.9%
  tin          [ 27.8, 71.8]   55/59  = 93.2%
  POOLED HELD-OUT: 344/393 = 87.5%  [83.9, 90.4]

WHAT THE READOUT DEPENDS ON (variance explained)
  baseline accuracy alone (5-point bins) : R^2 = 0.294
  + dataset   on the residual            : R^2 = 0.146  (23 levels)
  + backbone  on the residual            : R^2 = 0.015  (8 levels)
  + fraction  on the residual            : R^2 = 0.013  (10 levels)
  residual SD at fixed baseline          : 1.69 points
