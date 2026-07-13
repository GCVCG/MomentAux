# MomentStem study — project conventions

Fixed Gabor/Zernike moment filters ("MomentStem") in front of CNN backbones.
Standalone work, deliberately decoupled from the BMVC/MomentsNeRF paper.
Read PORTING.md before touching momentstem/stem.py — it records what was
ported vs corrected and why.

## Non-negotiables

- **The recipe is frozen**: SGD m=0.9, lr 0.1, wd 5e-4, cosine, 200 epochs,
  batch 128, crop+flip only. Headline cells never deviate; diagnostic cells
  that do must say so in their config name (e.g. `diag10e400_*`) and are
  never mixed into headline tables.
- **Filter banks are pinned**: tests/test_bank_regression.py holds numeric
  fingerprints. Changing any bank invalidates every existing run — do it
  only as a loud, deliberate decision with re-pinned constants.
- **Subsets are committed**: data/subsets/*.json. Every stem sees identical
  indices. Regenerate only via scripts/make_subsets.py (deterministic).
- **One entry point**: `python train.py --config <cell>.yaml --seed N`.
  `python analysis/aggregate.py` regenerates every table from runs/.
- **Aggregation groups by config NAME** (cell), not by parsed fields —
  variants sharing a stem differ in stem_kwargs.
- No hand-rolled metrics (torchmetrics + sklearn cross-check in tests).
- HWC->CHW crossings need content tests, not just shape (tests/test_tensor_layout.py).

## Compute

- All experiments run on the LOCAL RTX 3090 (user decision 2026-07-11;
  don't wait for the cluster). Long runs: detach with `setsid nohup`,
  write a `*_wave.log` in repo root, end with a `*_COMPLETE` marker line.
- turing cluster (`ssh amughrabi@turing.ub.edu`): repo mirror at
  /mnt/beegfs/amughrabi/projects/MomentsCNNEncoder, venv at
  /mnt/beegfs/amughrabi/envs/momentstem. GPU jobs need
  `--partition=gpu --qos=gpu` (default QOS has zero GPUs). **H100 only —
  the H200 is reserved for the user's other work.** Max 2 running jobs.
- CIFAR-100-C lives at data/CIFAR-100-C (local) and
  /mnt/beegfs/amughrabi/data/CIFAR-100-C (turing).

## State of findings (2026-07-13)

- Champion family "MomentStem-G": RGB passthrough + 9 calibrated Gabor
  kernels, 12ch, zero trainable params (`stem: moments-cat`,
  `stem_calibrate: true`, `stem_kwargs: {use_zernike: false}`).
  KERNEL SIZE IS A REGIME KNOB (full envelopes, 3 seeds/cell):
  - k11 Δ vs baseline: +1.3@1%, +1.8@2%, +1.9@3%, +1.9@5%, +0.6@7%,
    −1.4@10%, −2.4@15%, −1.6@25%, ~0@100%.
  - k5 (`stem_kernel_size: 5`, 225 fixed weights) Δ: +0.8@1%, +1.0@2%,
    +1.5@3%, +2.4@5%, +1.4@7%, −0.1@10%, −0.9@15%, −0.8@25%, ~0@100%.
  k11 wins at 1–3% (coarser prior), k5 wins at 5–15% and nearly erases
  the mid-data penalty band. Champion re-pin (single k5 vs per-regime
  choice) awaits user decision. k7 is not the midpoint — it's worse than
  both at 10% (39.31).
- Design space closed: pyramid/luma banks no better than the random k11
  bank at 5%, worse at 10%; MultiMaskPool readout (Zernike/random/learned
  masks) fails end-to-end under the frozen recipe despite +0.4–1.5 in
  linear probes — Zernike is dead at every placement tried (stem, readout).
- Falsified mechanisms (documented negatives, keep them dead): channel-scale
  imbalance (fixed by calibration, that was v1's real bug), ZCA/collinearity
  (calibrate_zca exists, didn't help), step budget (deficit persists at
  800 epochs), prior-as-init (gabor-learn loses the low-data gain entirely
  AND deepens the mid-data deficit), prior-as-warmup (stem_unfreeze_epoch
  at the overtake point changes nothing: -1.18/-2.38 vs fixed -1.40/-2.37
  at 10/15%). The benefit is constitutively tied to fixedness + low data.
- Surviving account: prior-shaped features commit during the high-LR phase;
  beneficial when data can't estimate RGB statistics (≤5%), costly at
  10–25%, harmless at 100%. conv1 usage ratio (logged per epoch in
  metrics.csv) tracks pruning but pruning conv1 doesn't recover accuracy.
- H2 (capacity substitution) dead; H3 (CIFAR-C robustness) null at 100%,
  tracks clean gain at 5%.
