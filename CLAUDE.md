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

## State of findings (2026-07-16)

- GENERALIZATION (2026-07-16). The champion (λ0=1.0 cosine→0, magnitude target,
  tap layer3, MSE) transplants with NO retuning across dataset and depth.
  DATASET AXIS — CIFAR-10, ResNet-18, 3 seeds, champion config verbatim:
    1% 39.35±0.26 → 45.96±0.18  = +6.61   (largest gain the method has produced
                                           on ANY dataset; beats C100's +5.30)
    5% 69.05±0.17 → 73.46±0.19  = +4.41
   10% 80.71±0.98 → 81.80±0.19  = +1.09
  DEPTH AXIS — CIFAR-100 @10%, λ0=1.0, 3 seeds: R18 +4.14±0.27, R34 +3.95±0.22
  (λ transfers across depth untouched). R50 λ0=1.0 −0.67±2.59 (UNSTABLE, see
  degeneracy below); R50 λ0=0.3 +2.42±0.28 (stable). So the exception is
  BLOCK TYPE (bottleneck), not depth.
- *** THE PREDICTIVE LAW: gain tracks THE DEFICIT THE DATA LEAVES, not the data
  FRACTION. C10@10% (baseline 80.71 — net already solved the low-level feature
  problem) gives only +1.09, though "10%" is the peak band on C100 (+4.14,
  baseline ~41). Gain rises monotonically as baseline accuracy falls, ACROSS
  datasets with different class counts and per-class counts. This was stated as
  a FALSIFIABLE PREDICTION BEFORE the C10 cells ran and HELD at all 3 points —
  it is the redundancy account's first out-of-sample confirmation, not a fit.
  Corollary (also held): C10@1% is 50 img/class = 10x richer per class than
  C100@1% (5/class), so it sits in the "schedule works" regime, not C100's
  "prior must never relax" regime — read PER-CLASS COUNT, not percentage.
- *** MECHANISM TRACED — the aux SCALE DEGENERACY (this is why R50 broke).
  ‖W·f − t‖² is INVARIANT under (f → f/c, W → c·W): SGD can minimise the aux by
  COLLAPSING the tapped features and INFLATING the head, learning nothing.
  Measured (trace() probe, 300 steps): R50 λ0=1.0 → layer3 std 0.596→0.051
  (12x collapse), ‖W_aux‖ 1.64→11.2 (7x), CE STALLS AT CHANCE 4.605 for 200+
  steps. Healthy comparators: R18 λ0=1.0 (L3 0.59→0.40, W→3.05, CE→3.32);
  R50 λ0=0.3 (L3 0.60→0.51, W→2.24, CE→3.52). Bottleneck blocks are more
  collapsible (1x1 projections), hence the R50-only failure.
  TWO FIXES DERIVED FROM THE MECHANISM (both work): (a) head_norm — project
  ‖W‖ back to its init after every step, removing the degenerate direction
  (CE 3.54, L3 0.287); (b) cosine loss — scale-free (CE 3.57, L3 0.585, NO
  collapse). Derived only AFTER three blind guesses failed (below).
- FALSIFIED FIXES for the R50 instability (keep them dead — all three were
  guesses made before tracing, and tracing is what actually solved it):
  (a) magnitude/GradNorm balancing — NO-OP; aux magnitudes are already
  identical across backbones, so there was nothing to balance.
  (b) fp32 / disabling AMP — WORSE, not better (NaN@epoch 6 vs @27). Not a
  precision bug.
  (c) BatchNorm on the tap — MUCH worse (NaN@21): BN DIVIDES BY the collapsed
  0.04 std, so it is a 25x AMPLIFIER of the exact pathology, not a cure.
- BREAKTHROUGH — MomentAux: moments as a SOFT TRAINING PRIOR on a vanilla
  backbone (momentstem/aux.py). Deployed model is a plain ResNet (RGB→logits,
  identical FLOPs to baseline, +0 inference params); during training only, an
  aux head taps layer3 and is regressed (MSE·λ + CE) onto the fixed magnitude
  moment maps. This is the FIRST placement that SCALES WITH DATA — positive at
  EVERY scale. CHAMPION λ=0.3 (magnitude target, layer3), FULL envelope
  (3 seeds): Δ +0.81@1%, +1.13@2%, +1.76@3%, +3.31@5%, +3.19@7%, +2.81@10%,
  +2.24@15%, +0.46@25%, +0.00@100%. Positive everywhere, peak in the 5–10%
  band, exactly neutral at 100%. Beats the BEST forward-path stem at 5–25%
  (e.g. @10% aux +2.81 vs k5 −0.09; @15% +2.24 vs −0.87). Only extreme low
  data (1–3%) still favors the forward-path magnitude specialist (+2.5/+3.5).
  λ knee: gain peaks λ≈0.5 (@10% +3.33), high-data safety holds through λ=0.3
  (+0.00@100%), breaks after (−0.36@100% at λ=0.5).
  STRUCTURE CONTROL (decisive): aux target = random-fixed maps gives ≈0
  (+0.01@5%, +0.14@10%) vs magnitude +3.31/+2.81 → moment gain +3.30/+2.67.
  So it is the MOMENT STRUCTURE, not "any aux regression / deep supervision".
  DESIGN SWEEP @10%: tap layer2 +2.83 ≈ layer3 +2.81; layer4 +0.92 (too late);
  multi-layer [2,3,4] +2.85 (no gain over layer3); gabor-k5 target −0.24
  (oriented edges are a BAD aux target — phase-invariant magnitude far better).
  → tap=layer3 and target=magnitude are SETTLED.
  TARGET SWEEP @10% (all energy families as aux targets, λ=0.3): magnitude
  +2.81 > structure +1.78 > steerable +1.01 > rotinv +0.84 > gabor −0.24 >
  invariants −2.97. The aux setting RESCUES features that were catastrophic
  forward-path (rotinv −5.64→+0.84, structure −5.20→+1.78) but magnitude still
  wins decisively. Best aux target = a MILD, INFO-PRESERVING nonlinearity;
  raw edges and heavily-processed invariants are bad targets.
  LOSS FORM: MSE ≫ cosine (@10% +2.81 vs +0.46; @2% +1.13 vs −0.01). The
  magnitude scale of the moment maps matters — cosine discards it. SETTLED.
  λ IS A DATA-REGIME KNOB (as kernel size was forward-path). Low data wants a
  MUCH stronger prior (no high-data over-reg risk there): @2% λ.3 +1.13,
  λ.5 +2.13, λ1.0 +2.63, λ2.0 +3.26 (≈ the forward-path specialist's +3.53).
  *** CHAMPION (2026-07-15): cosine λ SCHEDULE 1.0→0.0, magnitude target,
  tap layer3, MSE. ("prior dominates early, data takes over" inside one run;
  λ reaching EXACTLY 0 makes late training pure CE, so 100% neutrality is
  structural, not tuned.) FULL envelope (3 seeds), Δ vs baseline:
    1% +1.49 | 2% +2.50 | 3% +3.68 | 5% +5.30 | 7% +4.87 | 10% +4.14 |
    15% +2.55 | 25% +0.25 | 100% +0.08
  Positive at EVERY scale; peak +5.30@5%. Beats aux-fixed-λ=0.3 everywhere
  except 25% (+0.25 vs +0.46, tie-ish). vs BEST forward-path stem: champion
  WINS 3–15% decisively (@10% +4.14 vs −0.09; @7% +4.87 vs +1.44; @5% +5.30
  vs +3.34) and ties at 100%. CROSSOVER ~3%: below it the HARD input prior
  still wins (1% +2.55, 2% +3.53 fwd-path vs +1.49/+2.50 champion) — at
  extreme scarcity you want the prior strong THROUGHOUT, and the schedule's
  decay costs you (fixed λ=2.0 @2% gives +3.26 > schedule's +2.50).
  weight_final tuning: →0.0 dominates →0.1 (@10% +4.14 vs +3.64; @100% +0.08
  vs −0.19; @5% +5.30 ≈ +5.31; @3% +3.68 ≈ +3.80).
  *** FINAL RULE (2026-07-15): ALWAYS cosine-decay λ to EXACTLY 0 (that is what
  makes high-data neutrality structural); the λ_START is the DATA-REGIME KNOB
  (exactly as kernel size was forward-path). Best-per-regime (3 seeds):
    1% λ0=2.0 +1.91 | 2% λ0=2.0 +3.14 | 3% λ0=1.0 +3.68 | 5% λ0=1.0 +5.30 |
    7% λ0=1.0 +4.87 | 10% λ0=1.0 +4.14 | 15% λ0=0.3 +2.94 | 25% λ0=0.3 +0.97 |
    100% λ0=0.1 +0.24 (±0.10, ~2.4σ — small but real)
  POSITIVE AT EVERY SCALE. λ0 must be matched to the regime: a single λ0=1.0
  is best only at 3–10%; it UNDERSHOOTS at 1–2% (2.0 start: +2.50→+3.14@2%)
  and OVERSHOOTS at 15–100% (0.3 start: +2.55→+2.94@15%, +0.25→+0.97@25%;
  0.1 start @100%: +0.08→+0.24).
  λ HAS AN INTERIOR OPTIMUM PER REGIME, not a monotone "smaller at scale":
  @25% constant λ 0.02 −0.47 | 0.05 +0.11 | 0.10 +0.69 | 0.30 +0.46 — a
  TOO-WEAK aux is WORSE THAN NONE (competing gradient, no payoff).
  Gentle SCHEDULE beats best constant λ at high data (@25% +0.97 vs +0.69).
  1–2% REMAINS the forward-path magnitude stem's (+2.55/+3.53 vs +1.91/+3.14):
  at extreme scarcity a prior that never relaxes beats any decaying one.
  DEFENSIBILITY CONTROLS (all λ=0.3, vs magnitude +3.31@5% / +2.81@10%):
  - random-fixed target: +0.01@5%, +0.14@10% → NOT "any aux signal".
  - learned teacher / FitNets (frozen same-data backbone's layer3 features):
    −0.36@5%, +0.16@10% → NOT "just distillation"; a LEARNED target that costs
    a whole extra model does ~NOTHING while the free hand-crafted moment gives
    +3.3/+2.8.
  - HOG target (MaskFeat's descriptor): +1.22@5%, +1.37@10% → hand-crafted
    descriptors do help, but the MOMENT is ~2x better.
  → The moment structure is unambiguously the source of the gain, and
  phase-invariant energy specifically is the right descriptor.
  KEY INSIGHT: forward-path moments are a hard constraint (penalty band);
  aux-loss moments are a soft prior that shapes features without occupying
  input bandwidth, so abundant data overrides them instead of paying for them.
  Nearest prior art (deep research): Bhattarai 2023 (joint HOG aux, but
  segmentation), MaskFeat (HOG target, but SSL pretraining), FitNets/Mostajabi
  (intermediate-feature regression, but LEARNED targets). Novel intersection:
  fixed hand-crafted MOMENT target × joint aux+CE from scratch × vanilla deploy
  × provably scales. TODO controls: learned-teacher (FitNets) + HOG targets;
  ciFAIR-100 re-eval of low-data points; cross-backbone/dataset.
- Champion family "MomentStem-G" (forward-path, superseded as the scaling
  answer but still the low-data accuracy leaders): RGB passthrough + 9
  calibrated Gabor
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
- "Make them work AFTER 5%" — FALSIFIED for fixed nonlinear features
  (momentstem/energy.py, EnergyStem). Hypothesis: a prior encoding what the
  mid-data net can't self-learn (phase/rotation invariance, 2nd-order stats)
  could survive the penalty band. 10% band (3 seeds): magnitude −1.74,
  rotinv −5.64, structure −5.20 vs baseline — ALL worse than the linear k11
  Gabor stem (−1.40), the invariant/2nd-order ones catastrophically so
  (they discard raw signal the net wants at mid-data). The penalty band is
  agnostic to linear-vs-nonlinear and to which invariance: any fixed
  pre-committed extra channel costs accuracy at 10%+. Strongly supports the
  pure-low-data-statistics-estimation account. steerable/invariants
  (principled refinements) built + tested; not run at the band.
- BUT energy-magnitude is a NEW LOW-DATA CHAMPION (3 seeds, tight σ):
  Δ vs baseline +2.55@1%, +3.34@5% — beats BOTH Gabor stems at both points
  (prev best k11 +1.35@1%, k5 +2.35@5%). Phase-invariant complex-Gabor
  quadrature energy is a better low-data prior than oriented edges. Still
  −1.74@10% (a low-data specialist, not a band fix). rotinv also net-positive
  at low data (+0.97@1%, +0.45@5%) then collapses (−5.64@10%); structure
  ~null (+0.47@1%, −0.27@5%). Magnitude FULL envelope now mapped (3 seeds;
  100% 2 seeds): Δ vs baseline +2.55@1%, +3.53@2%, +2.99@3%, +3.34@5%,
  +0.78@7%, −1.74@10%, −4.76@15%, −7.34@25%, −3.89@100%. Beats BOTH Gabor
  stems at 1–5%; k5 retakes at 7%. KEY: unlike the linear stems, magnitude
  does NOT wash to ~0 at 100% — it stays −3.89 (phase-invariance permanently
  discards raw signal), and its band penalty grows far faster (−7.34@25%).
  A sharp ≤5% specialist, harmful everywhere above. (100% only 2 seeds,
  ±1.70 — large cost confirmed, exact value not nailed.)
  THREE-REGIME champion map: energy-magnitude best ≤5% (max low-data acc),
  k5 best 7–25% (never-lose all-rounder), k11 ~wash at 100%.
- steerable (angular-harmonic energy, principled rotinv) RUN at low data
  (3 seeds): Δ +1.87@1%, +1.69@2%, +1.81@3%, +1.50@5%, −1.05@7%, −4.82@10%.
  Beats its crude rotinv form (+0.97@1%→+1.87) and beats the Gabor stems at
  1%, but LOSES to magnitude at every point (mag +2.55/+3.53/+2.99/+3.34)
  and crosses negative earlier (−1.05@7% vs mag +0.78). Verdict: rotation-
  invariance is a genuine low-data prior but phase-invariant energy is
  strictly better. magnitude is the confirmed low-data champion; invariants
  (structure-tensor eigen, refines the NULL structure type) not worth running.
  Configs: enmag_/enrot_/enstr_/enste_/eninv_*pct.yaml.
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
