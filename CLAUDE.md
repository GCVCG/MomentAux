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
  write a `logs/*_wave.log` (PID to `logs/*.pid`), end with a `*_COMPLETE`
  marker line. All wave logs live in `logs/` — never the repo root.
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
    1% 39.35±0.26 → 45.96±0.18  = +6.61 ±0.18  (the 1–2% band holds the largest
    2% 51.34±1.20 → 58.49±0.16  = +7.14 ±0.70   gains the method has produced on
    5% 69.05±0.17 → 73.46±0.19  = +4.41 ±0.15   ANY dataset; both beat C100's
   10% 80.71±0.98 → 81.80±0.19  = +1.09 ±0.57   peak of +5.30)
  (±  on Δ = SEM of the difference of two 3-seed means, not the seed σ.)
  *** "C10 peaks at ≤1%" IS WITHDRAWN (2026-07-17) — and do NOT replace it with
  "peaks at 2%". Filling in 2% made it NOMINALLY the higher cell (+7.14 vs
  +6.61) but the difference is +0.53 ± 0.72 = 0.73σ: NOT DISTINGUISHABLE. The
  1–2% band is one FLAT PLATEAU at +6.6..+7.1; which point is the summit is
  UNRESOLVED. The σ comes entirely from the c10_none_2pct BASELINE (±1.20,
  seeds 52.48/51.95/50.20 — one low seed); the aux cell is ±0.16.
  This is the SAME failure mode as the retracted law and the 100% cell: a
  difference read off 3 seeds whose σ cannot support it. Caught BEFORE it was
  committed this time, which is the only reason it is not a fourth retraction.
  CONSEQUENCE FOR THE SYNTHESIS — the prediction "C10's true peak is 0.5%, so
  +6.61@1% is the RISING LIMB not the summit" is NOT supported, and is
  nominally CONTRADICTED. Above the claimed ~25 img/class threshold mechanism
  (a) alone should govern (less data → more gain), so 1% (50/cls) ought to
  BEAT 2% (100/cls). It does not. Either (b) is GRADED and still biting at
  50–100/cls, or the plateau is real. UNDERPOWERED — do not adjudicate at 3
  seeds. Deepening both cells to ~10 seeds is cheap (600 / 1400 steps) and is
  the way to settle it; until then state ONLY "gain plateaus across 1–2%, then
  decays (+4.41@5%, +1.09@10%)".
  BACKBONE AXIS — CIFAR-100 @10%, ONE config λ0=1.0 + head_norm, 3 seeds:
    R18 44.49±1.04 vs 40.18 = +4.31 | R34 44.06±0.27 vs 40.11 = +3.95 (no hn)
    R50 44.58±0.32 vs 40.65 = +3.93 (hn REQUIRED)
  *** UNIVERSAL λ: one λ0=1.0 gives +3.9..+4.3 on ALL THREE backbones — spread
  inside seed noise, NO per-family retuning. The old R50 exception was BLOCK
  TYPE (bottleneck), not depth, and head_norm CLOSES it.
  head_norm is not a stability patch, it RECOVERS gain: R50 λ0=0.3 (the tuned
  fallback) only reached +2.42 — weakening the prior suppressed the collapse AND
  the signal. Per-seed proof it is systematic, not a rare bad seed: R50 no-hn
  {41.22, 36.37, 42.35} → hn {44.45, 44.95, 44.35}. The BEST no-hn seed is BELOW
  the WORST hn seed; σ 3.18 → 0.32 (10x). Every seed was being taxed.
  *** head_norm IS A SAFE ALWAYS-ON DEFAULT (SETTLED 2026-07-16, 6 seeds). At 3
  seeds it looked mean-neutral but σ 0.33 → 1.04, unresolvable. At 6 seeds R18+hn
  = 44.45±0.70 (Δ +4.27) vs +4.14±0.33: mean unchanged, and σ REGRESSED 1.04 →
  0.70. F=4.32 on df(5,2), nowhere near significance. The 3-seed σ was one seed
  (43.36) landing low. Seeds: 43.36, 44.70, 45.40, 44.29, 44.83, 44.14.
  => ONE CONFIG FOR EVERY BACKBONE: λ0=1.0 cosine→0, magnitude, tap layer3, MSE,
  head_norm ON. Free on R18/R34; the difference between +3.93 and −0.67±3.18 on R50.
  LESSON: a σ from 2-3 seeds is nearly uninformative in EITHER direction.
- *** RETRACTED (2026-07-16) — "gain tracks THE DEFICIT THE DATA LEAVES, not the
  data fraction". I committed this as a law; it is FALSE and the retraction is
  itself informative. Two independent failures:
  (a) NOT MONOTONE — C100 falsifies it alone. @1% the baseline is 8.90 (the
  LARGEST deficit in the study, 89 pts of headroom) and the gain is +1.49, the
  SMALLEST non-zero gain in the envelope. The curve is UNIMODAL, peak at 5%.
  Below 25 img/class a BIGGER deficit gives a SMALLER gain — the exact opposite.
  Holds with best-per-regime λ0 too (1% λ0=2.0 → +1.91 < 5% +5.30), so it is
  NOT a λ artifact.
  (b) NO CROSS-DATASET x-AXIS — and it is STRUCTURALLY UNIDENTIFIABLE, not a
  missing variable. C10 and C100 are BOTH EXACTLY 50,000 images, so a given %
  fixes total images AND (frozen recipe, drop_last=True) total STEPS, while
  per-class count differs by exactly 10x. Matching per-class count NECESSARILY
  unmatches total data/steps by 10x. You cannot match both from this pair.
  The pair I built the claim on (C100@10% vs C10@1%: baseline 40.18 vs 39.35,
  both 50 img/class, +4.14 vs +6.61) differs 7800 vs 600 STEPS — 13x. I
  presented a compute-mismatched pair as a controlled comparison.
- STEP COUNT is a first-order confound in EVERY cross-fraction claim: the frozen
  recipe ties steps to data (1% = 600 steps, 10% = 7800, 100% = 78000). Same 10%
  data at 800 epochs (diag10e800_none) = 45.49 vs 40.18 at 200 — training longer
  buys +5.31, MORE than the method's whole effect at that cell (+4.14). Δ per
  cell is still valid (baseline gets identical steps); the SHAPE of Δ vs "data"
  is really Δ vs "data AND compute jointly".
  Corollary: sub-1% CIFAR-10 cells are IMPOSSIBLE under the frozen recipe —
  0.5% = 250 imgs = 1 batch/epoch = 200 steps; 0.1% = 50 imgs = 0 batches
  (drop_last=True) = empty loader, would not train at all.
- WHAT SURVIVES the retraction (state it THIS narrowly):
  (a) RIGHT FLANK ONLY: gain decays to ~0 as data becomes sufficient. Holds on
  both datasets. The one prediction that genuinely held — C10@10% would give a
  SMALL gain (+1.09) despite "10%" being C100's peak band, where percentage
  alone predicted ~+4 — was a right-flank call. Real, but one correct call was
  over-generalized into a two-sided law.
  (b) MATCHED-% IS THE CLEAN COMPARISON (identical images, steps, recipe; only
  class granularity differs): C100 peaks at 5%; C10 is ALREADY AT ITS PLATEAU
  by 1–2% and decaying by 5%. At matched data AND compute, the 100-class task
  needs MORE data before the prior pays off than the 10-class task. First
  confound-free cross-dataset statement we have, and it SURVIVES the peak-
  location withdrawal above: it only needs C10's plateau to sit LEFT of C100's,
  which holds however 1% vs 2% resolves (C10 is already decaying at 5%, where
  C100 peaks). Do NOT restate it as "C10 peaks at ≤1%".
- *** SUPERCLASS FORK ANSWERED (2026-07-16) — GAIN FOLLOWS TOTAL DATA/COMPUTE,
  NOT PER-CLASS COUNT. cifar100super = CIFAR-100's IMAGES + its 20 coarse labels,
  reusing C100's committed subset indices (byte-identical images, identical steps,
  per-class x5). Fork stated in ADVANCE: ~+0.25 (per-class) vs ~+5.30 (data/compute).
  RESULT @5% (3 seeds): 42.66±0.68 → 48.51±0.40 = **+5.84**. Unambiguous.
  DECISIVE PAIR — same per-class count, 23x different gain:
    super@5%  2500 imgs / 3800 steps / 125-per-cls → +5.84
    C100@25% 12500 imgs / 19400 steps / 125-per-cls → +0.25
  Same data+compute, 5x different per-class → SAME gain:
    C10@5% (250/cls) +4.41 | super@5% (125/cls) +5.84 | C100@5% (25/cls) +5.30
- *** THE SYNTHESIS (2026-07-16) — TWO MECHANISMS, ONE PER FLANK. One cell
  refuses to fit the above: at 500 imgs / 600 steps, C10@1% = +6.61 (50/cls) but
  C100@1% = +1.49 (5/cls) — IDENTICAL data AND compute, 4.4x different gain. So
  granularity matters, but ONLY at the bottom. The account that fits every cell:
  (a) the prior's FEATURE benefit tracks TOTAL DATA/COMPUTE (how much the data
      cannot teach itself) → governs the RIGHT flank: prior goes redundant
      (+0.25@25%, +0.15@100%);
  (b) REALISING it needs >=~25 img/class so the classifier can define a boundary
      → governs the LEFT flank. Measured INDEPENDENTLY by the linear probe:
      realization 41% at 5/cls vs 84% at 25/cls.
  Explains the UNIMODAL curve peaking near 25 img/class, and predicts C10's true
  peak is 0.5% (25/cls) — unrunnable under the frozen recipe (250 imgs = 1
  batch/epoch), so +6.61@1% is the RISING LIMB, not the summit.
  *** tin@1% PREDICTION HELD (2026-07-17) — mechanism (b) SURVIVES its first
  out-of-family falsification test, on a dataset the synthesis was not built on
  (Tiny-ImageNet, 200 classes, 64x64). Predicted IN ADVANCE and in writing:
  "SUPPRESSED ~+1.5 DESPITE 2x C100@1%'s images; if it lands ~+5, (b) is WRONG".
  LANDED: 5.22±0.32 → 6.95 = +1.73 (aux is 1 seed so far; seeds 1-2 running).
  2x the images and 2.3x the steps of C100@1% bought ~nothing, because per-class
  stayed at 5. NOT the ~+5 that would have killed it.
  *** THE MATCHED TRIPLE (2026-07-17) — the strongest evidence for (b) yet, and
  the MIRROR IMAGE of the superclass pair. All three cells have IDENTICAL total
  images (1000) and IDENTICAL steps (1400 = 7 batches x 200), same champion
  λ0=1.0 config; ONLY per-class count differs:
      100 img/cls  (C10@2%,   10-way)  51.34 → 58.49  = +7.15 ±0.70
       10 img/cls  (C100@2%, 100-way)  14.17 → 16.67  = +2.50 ±0.16
        5 img/cls  (tin@1%,  200-way)   5.22 →  6.95  = +1.73 ±0.18 (1 seed)
  MONOTONE in per-class count at perfectly fixed data AND compute. The
  C10-vs-C100 leg is **6.5σ** (+4.65 ±0.72) — not a noise story. (tin leg
  +0.77 ±0.24 pending its remaining seeds.)
  WHY THIS IS NOT A CONTRADICTION of "gain follows total data/compute, NOT
  per-class count" (the superclass result): the two pin down DIFFERENT flanks.
    superclass pair: same per-class (125), 23x different gain → per-class ALONE
      does not determine gain.
    this triple:     same data+compute (1000/1400), 4.1x different gain →
      data+compute ALONE does not determine gain either.
  Together they prove BOTH factors are real and NEITHER suffices — which is
  exactly the two-mechanism synthesis, now with a decisive experiment per side.
  RECONCILED BY THE ~25 img/cls THRESHOLD, which both datasets now support:
    at 2500 imgs, EVERY cell is at/above 25/cls (25/125/250) → all realized, and
      the spread is flat-ish: C100@5% +5.30, super@5% +5.84, C10@5% +4.41.
    at 1000 imgs, the cells STRADDLE it (5/10/100) → steep: +1.73/+2.50/+7.15.
  So per-class count matters ENORMOUSLY at 1000 imgs and BARELY at 2500 — which
  is what a threshold predicts, and is why neither factor alone ever fit.
  CAVEAT — DO NOT over-read the triple: at fixed total data, per-class count and
  CLASS COUNT are PERFECTLY anti-correlated (1000/10=100, 1000/100=10,
  1000/200=5), so this triple CANNOT separate "5 examples can't define a
  boundary" from "200-way is simply a harder task". tin also differs in
  RESOLUTION (64x64). What breaks that tie is the superclass cell: at 2500 imgs,
  20-way (super, +5.84) vs 100-way (C100, +5.30) — 5x different class count,
  SAME gain. So class count per se is NOT the driver at that scale.
  LIVE PREDICTION STILL OPEN: stl@10% = 500 imgs / 50-per-cls / 600 steps →
  should be ~+6.6 (tests whether 96x96 resolution changes anything).
- ciFAIR-100 (2026-07-16) — the low-data gains are NOT memorised train/test
  duplicates. CIFAR-100's test set has 927/10000 near-duplicates of train images;
  ciFAIR replaces exactly those. The Δ is stable at every point:
    Δ CIFAR → ciFAIR:  1% +1.88→+1.68 | 5% +5.40→+5.36 | 10% +4.10→+4.15 |
                     100% +0.18→+0.28
  Both cells eat identical contamination, so it cancels. The ABSOLUTE drop grows
  with data (−0.65@1% → −2.86@100%) because contamination is against the FULL
  train set — at 1% most duplicate sources are not even in the 500-image subset.
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
  *** TAP DEPTH IS **NOT** A REGIME KNOB (2026-07-16) — the first knob in this
  study that isn't. TAP SWEEP @1% (λ0=2.0, 3 seeds, baseline 8.90): layer1
  10.77±0.29 = +1.87 | layer2 10.55±0.09 = +1.65 | layer3 10.81±0.09 = +1.91.
  layer3 is BEST at 1% just as at 10%; the whole spread is ~seed noise, and
  layer1 (earliest possible tap, full 32x32, target unpooled) does NOTHING.
  So the optimum does NOT move with data regime: a broad plateau layer1≈layer2≈
  layer3 with layer4 the only cliff, at BOTH ends of the range. Contrast kernel
  size (forward-path) and λ0 (aux), which ARE regime knobs.
  NOTE ON THE REASONING: "SETTLED" above was originally concluded from a SINGLE
  regime (10%) in a study whose recurring finding is that nothing is settled from
  one regime — bad justification that happened to reach the right answer. It has
  now survived its first out-of-regime test on evidence.
  CONSEQUENCE: the LEFT-FLANK COLLAPSE IS REAL, not a tap-depth artifact. The
  hypothesis "@1% layer3 is too late, the data can't estimate even early layers"
  is FALSIFIED — tapping at layer1 recovers nothing. Surviving account: at 5
  img/class the CLASSIFIER, not the features, is the bottleneck (5 examples
  cannot define a boundary however good the representation), which explains both
  why tap depth is irrelevant there AND why the forward-path stem still wins
  (+2.55 vs aux +1.91): it changes what the CLASSIFIER SEES rather than only
  shaping intermediate features.
  *** CONFIRMED BY LINEAR PROBE (2026-07-16, analysis/linear_probe.py). Probing
  the FROZEN penultimate features on the FULL train set (removes the classifier's
  data bottleneck, so only feature quality is measured), 3 seeds:
    @1%  baseline  8.90 e2e → probe 26.81±0.24
         aux λ0=2.0 10.81 (+1.91) → probe 31.51±0.37  (features +4.70)
         aux λ0=1.0 10.39 (+1.49) → probe 30.68±0.64  (features +3.87)
         aux tap1   10.77 (+1.87) → probe 31.74±0.13  (features +4.93)
         aux tap2   10.55 (+1.65) → probe 30.98±0.53  (features +4.17)
    @5%  baseline 25.23 e2e → probe 37.02±0.30
         aux λ0=1.0 30.53 (+5.30) → probe 43.33±0.54  (features +6.31)
  *** THE PRIOR IMPROVES FEATURES AT BOTH SCALES BY COMPARABLE AMOUNTS
  (+4.70@1% vs +6.31@5%). What differs is REALIZATION:
    @5%: 5.30 realized / 6.31 available = 84%
    @1%: 1.91 realized / 4.70 available = 41%
  So the LEFT FLANK IS NOT THE PRIOR FAILING — it is the CLASSIFIER failing to
  cash it in. At 5 img/class the moment prior does ~3/4 of the feature work it
  does at 25 img/class and <half of it reaches the logits.
  Cross-checks that this is real, not a story: (i) the probe TRACKS ACCURACY
  monotonically (λ0=2.0 > λ0=1.0 on BOTH probe and e2e); (ii) it INDEPENDENTLY
  REPRODUCES the tap null (tap1≈tap2≈tap3 on features, as on accuracy); (iii) it
  explains WHY tapping earlier couldn't help — there was never a feature deficit
  to fix (@1% features are already +4.70 better).
  CAVEATS: the probe trains its head on 50k labels the cell never saw → it is a
  DIAGNOSTIC, NEVER a headline cell. The baseline's own 8.90→26.81 gap is not a
  finding (any head with 100x labels does better); the finding is the aux-vs-
  baseline DELTA under identical probing.
  ACTIONABLE: @1% there is +4.70 of feature gain sitting UNCLAIMED. Anything
  that relieves the CLASSIFIER bottleneck (not a better prior) should cash more
  of it in. The left flank is an opportunity, not a ceiling.
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
    100% λ0=0.1 +0.15 (0.77σ — NOT distinguishable from zero; see correction)
  *** CORRECTED 2026-07-16: this cell was "+0.24 (±0.10, ~2.4σ — small but
  real)" on a 2-SEED baseline (78.43±0.07). Third seed -> baseline 78.52±0.16,
  aux 78.67±0.12, so Δ=+0.15 at 0.77σ. NEUTRAL, not positive. The method is
  unharmed — neutrality at full data is what the structural λ→0 schedule
  PREDICTS — but do NOT say "positive at every scale"; say "positive up to 25%,
  neutral at 100% by construction".
  POSITIVE AT EVERY SCALE UP TO 25% (100% is neutral, see above).
  λ0 must be matched to the regime: a single λ0=1.0
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
