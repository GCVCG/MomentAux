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
  /mnt/beegfs/amughrabi/projects/MomentsCNNEncoder (rsync copy, NOT a git
  repo — sync code with rsync, pull run results back the same way), venv at
  /mnt/beegfs/amughrabi/envs/momentstem. GPU jobs need
  `--partition=gpu --qos=gpu` (default QOS has zero GPUs). Max 2 running
  jobs (= one per GPU). **2026-07-19 user decision: the next-steps campaign
  runs on turing using BOTH GPUs (H100 + H200)** — this supersedes the old
  "H200 reserved" rule for this campaign. Wave sbatch scripts pin the venv
  python explicitly (submission-env-independent); markers/logs follow the
  local conventions but live in slurm/logs/.
- CIFAR-100-C lives at data/CIFAR-100-C (local) and
  /mnt/beegfs/amughrabi/data/CIFAR-100-C (turing).
- **`num_workers` IS PART OF THE REPRODUCIBILITY CONTRACT** (verified 2026-07-17,
  not assumed). Data ORDER is worker-count-independent (the sampler draws from
  `generator=gen` in the main process), but AUGMENTATION IS NOT: PyTorch seeds
  each worker's torch RNG to `base_seed + worker_id`, and RandomCrop/
  RandomHorizontalFlip draw from it. Measured nw=8 vs nw=2, same seed: labels
  identical in every batch; pixels identical for batches 0..nw-1 then diverge
  from batch index == nw onward (round-robin hands batch i to worker i, so the
  first nw batches match trivially — a 2-batch test shows a FALSE match, which
  is exactly the mistake to avoid here).
  CONSEQUENCE: changing num_workers re-draws the augmentation stream. This is
  UNBIASED (still crop+flip from the same distribution — it acts like a
  different augmentation seed), so a Δ across a worker-count boundary is valid,
  but seeds are NOT paired across it and the cell is NOT byte-reproducible.
  RULE: keep num_workers constant WITHIN a cell; never change it on a cell that
  already has completed seeds. (Known crossing: c10_none_2pct is nw=8 while
  c10_aux_2pct is nw=2 — Δ valid, exact reproduction needs the recorded value.
  Every run's num_workers is stored in its final.json `config`.)
  Contention is real and worth managing: at load 113 (16 cores, 76 workers) the
  same cells ran 44 min/seed vs 5.4 min/seed at low load — ~5x. These jobs are
  DATALOADER-bound, not GPU-bound; nw=2 with a few concurrent runs beats nw=8.

## Active campaign (2026-07-18, user-approved "run the whole needed experiments")

- xspace wave (running): Phase B tin20+tin@1% deepening to 10 seeds, Phase C
  tin 2x2 crossing. Phase A (C100 crossing) answered: FULL INVARIANCE.
- frontier2 wave (running, cheap->expensive): tin@2% pair+probes; tin20b
  (disjoint class draw, the tin20 control) pair+probes; c10_aux_10pct_l03 +
  tin_aux_5pct_l20 (per-regime lambda vs the law); diagvit pair (ViT-tiny,
  FIRST ATTENTION BACKBONE, AdamW diag, tap blocks.8 with the new token->
  spatial adapter in aux.py); then tin@25% and tin@100% LAST (the 100% pair
  is ~a week of GPU and can be killed without collateral).
- PREDICTIONS RECORDED BEFORE RESULTS (no-unmeasured-G rule respected):
    tin@2%: envelope-based +1.3..+2.3 (G at 2000 is unmeasured -- the probe
      afterward measures it; no G-based number).
    tin20b: granularity account => +3.0..+5.0 (tin20's +4.07 within noise);
      a materially different value means tin20's class DRAW mattered.
    c10_aux_10pct_l03: +0.3..+1.1 and NOT above champion's +1.09 -- G(C10,
      5000)=0.65 is measured and exhausted; per-regime lambda cannot
      manufacture gain past G's ceiling (the C10@15% rescue lesson).
    tin_aux_5pct_l20: +2.2..+3.3 (C100 low-baseline precedent: lambda0=2.0
      beats 1.0 below the crossing).
    diagvit: qualitative only -- underfit baseline (<=30) => ConvNeXt-like
      rescue (+5..+12); well-trained baseline => champion-like (+3..+5).
    tin@25%/100%: qualitative only (G unmeasured): envelope stays flat
      (<=+2.2); 100% ~ neutral (+-0.5).
- LANDINGS SCORED (2026-07-18, later same day):
    tin@2%: **+1.81 ±0.19** (9.56±0.11 -> 11.37±0.31) — IN BAND (+1.3..+2.3).
      G(tin,2000) = **+3.47 ±0.24** — slots exactly into the monotone-falling
      G(tin) curve: 4.33@1000 -> 3.47@2000 -> 2.67@5000 -> 1.70@10000.
      readout(base 9.56) = −1.66, negative far below the crossing: SIGN LAW
      16th clean cell. tin envelope now 1.60/1.81/2.13/1.65 at 1/2/5/10% —
      flat, exactly as falling-G predicts.
    tin20b: **+5.27 ±0.66** (43.83±0.84 -> 49.10±0.78) — 0.2σ from tin20's
      10-seed +5.42±0.41: the CLASS DRAW IS IRRELEVANT, granularity account
      CONFIRMED on a disjoint draw. (Nominally 0.27 above the recorded band
      top +5.0, but the band was anchored on tin20's 3-seed +4.07, which
      itself moved to +5.42 — both draws agree with each other.)
      *** TENSION TO WATCH: G(tin20b,1000) = +2.37 ±0.59 vs G(tin20) =
      +5.23 ±0.90 — 2.7σ apart while the e2e deltas are 0.2σ apart. Either
      one probe measurement is off (both are 3-seed, tin20's σ 0.90) or the
      G/readout SPLIT differs between draws while the sum matches — which
      would be a coincidence the law does not predict. Phase C re-probes
      tin20 at 10 seeds and adjudicates. readout(tin20b) = +2.90 at base
      43.83: sign correct (17th cell), magnitude pending the tension.
    c10_aux_10pct_l03: **+1.10 ±0.57** (80.71 -> 81.81±0.15) — IN BAND
      (+0.3..+1.1, at top edge) and =champion's +1.09 exactly. PREDICTION
      HELD: per-regime λ0 cannot beat G's ceiling; G(C10,5000)=0.65 is
      exhausted by EITHER λ0. (Nominal Δ>G again — readout positive at
      base 80.7, consistent with the mapped positive branch +0.44@80.7.)
    tin_aux_5pct_l20: FINAL **+1.90 ±0.12** (21.05 -> 22.95±0.08) — MISSED
      LOW (band +2.2..+3.3), ≤ champion's +2.13 (1.4σ below, n.s.). The
      C100 "λ0=2.0 wins below the crossing" precedent does NOT transplant
      to tin. Third straight demonstration that the λ0 knob cannot push Δ
      past what G supplies (C10@15% rescue, c10_l03, now tin_l20).
    diagvit: **+13.26 ±0.33** (16.47±0.50 -> 29.74±0.29, C100@10%) — the
      LARGEST GAIN IN THE STUDY, first attention backbone. Qualitative
      prediction (underfit baseline => ConvNeXt-like rescue) CONFIRMED,
      nominal value just above the +5..+12 band. ViT-tiny from scratch at
      5000 imgs is exactly the underfit regime (base 16.5 vs R18's 40.2);
      the prior rescues it even harder than ConvNeXt (+10.57). σ pattern
      again: baseline 0.50 -> aux 0.29. NOT a headline row (AdamW diag).
      Mechanism is now demonstrated on conv-SGD, conv-AdamW, and
      attention-AdamW.
    tin@1% DEEPENED to 10 seeds (xspace Phase B): 5.30±0.22 -> 6.80±0.19 =
      **+1.49 ±0.09** (3-seed said +1.60±0.20 — mild regression to mean,
      same story). The 5/cls universal floor stands: C100@1% +1.48, tin@1%
      +1.49.
    tin@25%: baseline 49.19±0.21 done; aux 2/3 seeds, partial Δ +0.02 —
      tracking NEUTRAL, inside the qualitative "flat ≤+2.2" band and
      consistent with G(tin) falling through ~1 at 25k images. Score when
      seed 2 lands.
- H3-for-aux ANSWERED from existing robustness.json (2026-07-18): CIFAR-100-C
  mCE delta -2.87@5% / -2.56@10% / +0.02@100% vs clean gains -5.3/-4.1/-0.3
  (err): corruption robustness TRACKS the clean gain (~55-60% of it), no
  extra benefit, no cost, neutral at 100%. Same verdict as forward-path H3.
- aggregate.py regenerated (results/summary.md|tex current).

## Turing campaign (2026-07-19, user-approved "all points deserve
## investigation, use turing, both GPUs")

- Four thrusts approved: (1) left-flank readout experiments, (2) diagvit
  envelope, (3) fine-grained transplant (CUB-200 class), (4) SSL matched-
  compute comparison. (1)+(2) launched first (code-ready); (3)+(4) queued
  behind them (need dataset/SSL scaffolding).
- NEW CODE: `head: cosine` config option (CosineClassifier in backbones.py,
  Gidaris-style s*cos(theta), learnable s init 16, ResNets only, diag-only
  guard in train.py; linear_probe.py plumbs it for ckpt loading — probe
  extraction itself never touches fc). Smoke-tested; full suite 101 passed.
- diagcos wave (H200): parent cells VERBATIM + head:cosine, both pair
  members get the head — diagcos_c100_{none,aux}_1pct (parents abl1_none /
  auxmag_1pct_sched0) and diagcos_tin_{none,aux}_1pct. 3 seeds + full-label
  probes.
  PREDICTIONS RECORDED IN ADVANCE: the study's own evidence (Q7.3 —
  e2e realizes exactly what a same-budget LINEAR probe realizes; tap sweep —
  the left flank is label-information-limited, not architecture-limited)
  predicts the NULL: Δ_cos ≈ Δ_linear.
    diagcos_c100_1pct: Δ +1.1..+1.9 (linear-head champion +1.49).
    diagcos_tin_1pct:  Δ +1.1..+1.9 (linear-head champion +1.49±0.09).
    G on cosine-head ckpts ≈ unchanged (C100@500 ~3.9, tin@1000 ~4.2-4.5):
      the head shapes readout, not features.
  FALSIFIER: Δ ≥ +2.5 on either dataset => head expressivity IS part of the
  left-flank penalty, reopening the readout-design axis (NCM/prototype heads,
  and the "+4.7 unclaimed feature gain" becomes partially cashable). A big
  move in G would instead say the cosine constraint changed FEATURE learning
  (feature-norm regularization) — a different, also-interesting outcome.
- diagvit wave (H100): diagvit_{none,aux}_{1,5,25}pct on C100 (10% already
  measured: +13.26±0.33, base 16.47). 3 seeds + probes. AdamW => diag-only.
  PREDICTIONS (qualitative — G(vit) unmeasured, no G-based numbers):
    @5%: large rescue, Δ ≥ +8 (baseline deep-underfit).
    @1%: positive but SUPPRESSED vs 5%/10% (sign law: readout negative at
      very low baselines; all vit baselines sit below the ~30 crossing, so
      Δ < G everywhere, suppression strongest at 1%).
    @25%: positive; smaller than +13.26 iff the baseline rises materially
      above 16.5 (Δ should fall monotonically as baseline underfit-ness
      shrinks). If base@25% stays ≤ ~25, the rescue may stay ≥ +10.
    Probes: G(vit) expected LARGER than R18's G at matched cells (part of
      the rescue is a feature deficit attention cannot self-learn here).
  FALSIFIER for "the same law governs attention": readout = Δ − G coming out
  POSITIVE at baselines far below 30 (conv sign law would not transfer).
- *** HEAD-FORMS ANSWERED AT THE PROBE LEVEL (2026-07-20, local wave,
  analysis/head_forms.py, all seeds, 5 draws/seed): THE READOUT PENALTY IS
  LABEL-INFORMATION, NOT HEAD EXPRESSIVITY. Matched 5/cls budget on frozen
  ckpt features, aux-vs-baseline gap by head form:
    C100@1%: linear +1.74±0.05 | cosine +1.57±0.05 | ncm +1.47±0.09
    tin@1%:  linear +1.48±0.06 | cosine +1.49±0.05 | ncm +1.50±0.04
  The GAP is head-form-INVARIANT (tin exactly so; C100 spread ~0.27, small).
  Cosine/NCM lift ABSOLUTE accuracy ~+0.3-0.6 on BOTH members (a generic
  few-shot head effect) but read the SAME aux gap — few-shot head tricks do
  not unlock the unclaimed feature gain. STRONG PRIOR that the e2e diagcos
  cells land at the null (Δ_cos ≈ Δ_linear), as predicted in advance.
- tin@100% LANDED (2026-07-20, frontier2 COMPLETE): 66.17±0.25 ->
  65.75±0.58 = **−0.42 ±0.36** — inside the qualitative band (~neutral
  ±0.5), 1.2σ from zero. THE TIN ENVELOPE IS CLOSED: +1.49/+1.81/+2.13/
  +1.65/+0.10/−0.42 at 1/2/5/10/25/100% — falling G(tin) all the way to a
  mildly-negative full-data cell (mirrors C10@100% −0.26: λ0=1.0 verbatim
  above the crossing; per-regime λ0 would likely restore exact neutrality,
  same as C10's story — not worth the GPU to confirm).
- *** diagvit@10% PROBED (2026-07-20, local; linear_probe.py gained a ViT
  extraction path — timm ViT global_pool is the STRING "token", CLS token +
  fc_norm): G(vit, C100@10%) = **+14.85 ±0.53** (probe 24.87±0.87 ->
  39.72±0.29) — ~2.3x the largest conv G ever measured (6.35). Implied
  readout = +13.26 − 14.85 = **−1.59** at base 16.47: NEGATIVE below the
  ~30 crossing, exactly as the conv-derived sign law requires (21st clean
  cell, first attention cell). The ViT rescue is a G effect: attention at
  this scale has a huge feature deficit the prior fills. The running
  diagvit wave tests this across 1/5/25%.
- PHASE 2 QUEUED (2026-07-20, submitted as PENDING behind the running
  waves; MaxSubmitPU=128 so queuing ahead is allowed, MaxJobsPU=2 enforces
  one-per-GPU). Both use --gres=gpu:1 (whichever GPU frees first).
  NEW INFRA: data.py CUB200 loader (CUB-200-2011 @64px squash-resize,
  5994/5794 split, stats pinned from the train split, subsets extended —
  data/subsets/cub_25pct.json committed, 1594 imgs ~8/cls);
  scripts/simclr_pretrain.py (NT-Xent, two-view, subset-images-only);
  train.py `init_from` (diag-only, {seed}-templated, strict=False minus
  classifier). All smoke-tested; suite 101 green after changes.
  cub wave (slurm/cub_wave.sbatch): cub_{none,aux}_{25,100}pct, champion
  verbatim, 3 seeds + probes. PREDICTIONS RECORDED IN ADVANCE (G(cub)
  unmeasured -> banded by tin/tinsuper precedent):
    cub@100% (30/cls, ~9200 steps): Δ +1.5..+4.5 — fine-grained weak
      supervision is where the prior does its most feature work (tin20/
      tinsuper ckpt-set effect), but 6k images sits on the falling-G right
      flank. Baseline guess (NOT a prediction): ~15-25.
    cub@25% (~8/cls): Δ +0.5..+2.0 (near the universal ~+1.5 5-per-cls
      floor; readout penalty deep on the left flank).
    FALSIFIERS: Δ(100%) ≥ +5 => fine-grained tasks carry genuinely richer
      G than any 64px dataset measured (reopens the fine-grained axis as a
      headline direction); Δ(100%) ≤ 0 => the fine-grained hypothesis dies.
    PROBE CAVEAT recorded now: at cub@100% probe labels == cell labels, so
    the G/readout SPLIT is not interpretable there (probe-ceiling rule);
    the aux-vs-base probe gap under identical probing still is.
  ssl wave (slurm/ssl_wave.sbatch): per seed SimCLR 200ep pretrain on the
  committed 5% subset images -> frozen recipe from that init
  (diagssl_simclr_5pct); plus diagssl_none400_5pct = 400-epoch plain
  baseline (the 2x-compute control; step-count confound lesson).
  PREDICTIONS RECORDED IN ADVANCE:
    diagssl_simclr_5pct: 26..28.5 (above baseline 25.23, BELOW champion
      30.53 despite 2x compute — SimCLR is data-starved at 2500 imgs).
    diagssl_none400_5pct: ~27-29 (diag10e800 precedent: longer training
      alone buys points) — the SSL init must beat THIS, not the 200ep
      baseline, to claim any SSL-specific value.
    FALSIFIER: diagssl_simclr_5pct ≥ 30.5 => SSL matches the prior at 2x
      compute and the "cheapest prior" positioning needs rewording.
- *** DIAGCOS LANDED (2026-07-20, wave COMPLETE, 6h48): THE E2E NULL
  CONFIRMED ON C100, AND TIN CAME IN BELOW EVEN THE NULL.
    diagcos_c100_1pct: Δ_cos = **+1.60 ±0.17** — IN BAND (+1.1..+1.9),
      0.6σ from linear champion +1.49. Exactly the predicted null.
    diagcos_tin_1pct:  Δ_cos = **+0.92 ±0.15** — MISSED LOW (band
      +1.1..+1.9): ~3σ BELOW the linear champion +1.49±0.09. Baseline
      identical to linear's (5.30 = 5.30); the cosine head cost the AUX
      cell ~0.6. G probes say features are unchanged (C100 3.25±0.18 vs
      3.88±0.39; tin 4.27±0.26 vs 4.19-4.33 — dead on), so the loss is a
      READOUT-level training effect at 200-way/5-shot, not a feature one.
  VERDICT: head expressivity buys nothing anywhere (head_forms predicted
  this) and e2e cosine training slightly HURTS at fine label spaces. The
  readout-design axis is now closed from both ends: the left-flank
  unclaimed gain is label-information-limited, full stop.
- *** DIAGVIT ENVELOPE LANDED (2026-07-20, wave COMPLETE, 2h23; probes
  included). Full pairs + G at every point (3 seeds):
    @1%:  6.44±0.19 ->  7.80±0.21 = **+1.35 ±0.16** | G +5.89±0.75,
      readout −4.54 @ base 6.4
    @5%: 12.25±0.83 -> 21.60±0.51 = **+9.35 ±0.56** | G +13.17±0.54,
      readout −3.82 @ base 12.3  (band "Δ ≥ +8" HIT)
    @10% (prior): +13.26±0.33 | G +14.85±0.53, readout −1.59 @ 16.5
    @25%: 28.50±0.63 -> 42.17±0.57 = **+13.67 ±0.49** | G +14.03±0.29,
      readout −0.36 @ base 28.5
  @1% suppressed as predicted; @25% MISSED the soft "Δ falls as base
  rises" call — because G(vit) does NOT fall: 5.89/13.17/14.85/14.03 at
  0.5/2.5/5/12.5k imgs, a rising-then-plateau curve ~14-15 with NO right
  flank through 25%. Same error class as ever: extrapolating an unmeasured
  G. THE LAW ITSELF IS PERFECT ON ATTENTION: Δ = G + readout holds at all
  four points, readout monotone (−4.54/−3.82/−1.59/−0.36), negative below
  the ~30 crossing everywhere — sign law now 25 clean cells, 5 on ViT.
  ViT-tiny's feature deficit is HUGE and constant-ish (~14) across 2.5-12.5k
  images: the prior is worth ~2.3x more to attention-at-small-scale than to
  any conv backbone measured.
- *** SSL WAVE LANDED — THE FALSIFIER FIRED (2026-07-20, wave COMPLETE,
  0h58). diagssl_simclr_5pct = **34.41 ±0.17** (seeds 34.56/34.22/34.45)
  vs predicted 26..28.5 — blew through the ≥30.5 falsifier: SimCLR
  pretrain on the SAME 2500 subset images + frozen recipe BEATS the
  champion aux (30.53) by **+3.9** at 2x compute. The compute control
  diagssl_none400_5pct = **27.34 ±0.25** landed IN its band (27-29):
  plain 2x compute buys +2.1, so the SSL-specific value is **+7.1**.
  Frontier at C100@5%: baseline 25.23 (1x) < none400 27.34 (2x) <
  champion aux 30.53 (~1.02x) < simclr-init 34.41 (2x).
  CONSEQUENCE (per the recorded falsifier): the "cheapest prior"
  POSITIONING NEEDS REWORDING — MomentAux's claim narrows to its
  ~zero marginal cost (+2% compute vs +100%) and possible orthogonality;
  it is NOT the accuracy frontier at 2x. What survives untouched: every
  within-recipe finding (the law, G curves, sign law) — none of them
  claimed SSL-superiority. OPEN QUESTIONS -> ssl2 wave (queued):
  is the prior REDUNDANT with SimCLR features (combo cell), and does
  SSL's win survive at near-matched compute (50ep pretrain)?
  ssl2 wave PREDICTIONS RECORDED IN ADVANCE:
    diagssl_simclraux_5pct (simclr init + champion aux, 2.02x): the
      fwd-combo precedent and "prior substitutes for supervision" both
      predict REDUNDANCY: 34.4 + (+0..+1.5) over simclr-init alone.
      FALSIFIER: ≥ +2.5 over 34.41 => the two priors stack (moment
      structure carries information SimCLR cannot learn from 2500 imgs)
      — would reopen aux-on-SSL as a headline direction.
    diagssl_simclr50_5pct (1.25x): 30..34 (SSL gains are front-loaded);
      the frontier question is whether it clears champion 30.53 at near-
      matched compute. Below 30.5 => the champion keeps a genuine
      low-overhead niche; above => SSL dominates from ~1.25x on.
    Probes (G under identical probing vs abl5_none ckpts): if SimCLR's
      +9.2 is FEATURE-side, G(simclr ckpts) ≈ +9..+12 (≫ champion's
      6.35); the law then demands readout(base 34.4) ≈ +0.5..+1.5
      (positive branch, decaying). A small G with big Δ would violate
      the law and be the bigger news.
- tinsem wave QUEUED (2026-07-20, user: "no reason not to queue it" — the
  Q6.9j semantic-vs-arbitrary caveat gets its adjudication). tinsem =
  tinsuper with ONLY the block-of-10 sort key changed: WordNet hypernym-
  path order (scripts/make_tin_semantic_order.py, committed in
  data/subsets/tin_semantic_order.json — groups audit as coherent: vehicles,
  dogs+cats, insects, food...) vs lexicographic wnid. Byte-identical pixels,
  identical 20x50 structure at tin@1%'s committed subset. 10 seeds;
  tinsuper DEEPENED 3->10 in the same wave (power-matched reference);
  probes in both spaces.
  PREDICTIONS RECORDED IN ADVANCE:
    PRIMARY FORK — G_200(tinsem ckpts) on the common 200-way stick:
      H-semantic (coherence carries the tin20-vs-tinsuper G_200 gap):
        ≈ 3.0-3.1 (tin20/tin20b's band), above tinsuper's 2.55.
      H-arbitrary (sort key irrelevant): ≈ 2.55 (tinsuper's value); the
        3.08-vs-2.55 hint was noise or tin20's pixel population.
    BOTH branches: G_200(tinsem) stays WELL below tin@1%'s 4.19 (the
      fine-vs-coarse main effect, 3.8σ+, must persist). If it lands ~4.2
      the coarse-training G cut itself was an artifact — major surprise.
    e2e: baseline should rise well above tinsuper's 14.08 (semantically
      coherent groups are learnable; if base stays ~14 the semantic sort
      failed to create visual coherence and the fork is void — audit).
      Δe2e is NOT independently predicted: the law demands Δ = G_20own +
      readout(base), with readout sign set by base vs the ~30 crossing
      (tinsuper hit this identity exactly; tinsem must too — 21st/22nd
      clean cells).

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
  *** tin@1% PREDICTION HELD NUMERICALLY (2026-07-17, final 3 seeds: 5.22±0.32
  → 6.82±0.15 = +1.60 ±0.20) — predicted in advance: "SUPPRESSED ~+1.5 DESPITE
  2x C100@1%'s images; if it lands ~+5, (b) is WRONG". Landed 0.5σ from the
  prediction, 16σ from the falsifier. BUT the confirmation is likely HOLLOW —
  see the DATASET-CONTROL CRISIS below: tin gives small gains at EVERY per-class
  count measured, so "suppressed because 5/cls" is not yet distinguishable from
  "tin is a low-gain dataset". The prediction hit the right number for possibly
  the wrong reason. Do NOT cite tin@1% alone as evidence for (b).
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
  CAVEAT ESCALATED TO A CONFOUND (2026-07-17) — the triple is NOT decisive
  after all; see the DATASET-CONTROL CRISIS below. At FIXED per-class count,
  changing only the dataset moves Δ by 6.0-8.1σ, so the triple's three datasets
  confound per-class count with dataset identity. Q6.9b's monotonicity is real
  as DATA but its attribution to per-class count was an over-read — same error
  class as the retracted law.
  *** stl@10% PREDICTION HELD (2026-07-17, 3 seeds): 41.58±1.31 → 47.51±0.60 =
  +5.92 ±0.83 vs predicted ~+6.6, within noise of C10@1%'s +6.62. RESOLUTION IS
  IRRELEVANT to the prior (3x linear res, ImageNet content; k11's coverage
  fraction does not matter). stl@50% (=C10@5% mirror, 250/cls/2500 imgs): +3.42
  at 1 seed, tracking C10@5%'s +4.41. Kills every "kernel scale must match
  object scale" account.
  *** THE DATASET-CONTROL CRISIS (2026-07-17) — the tin envelope broke the
  threshold story:
  (1) ~25 img/cls does NOT transfer: crossing 5->25/cls buys +3.81 on C100
      (+1.48->+5.30) but +0.53 on tin (+1.60->+2.13). tin@5% vs C100@5%, both
      25/cls: 8.1σ apart.
  (2) At fixed per-class, dataset identity moves Δ: 50/cls C10@1% +6.62 vs
      C100@10% +4.14 (6.0σ); 25/cls C100@5% +5.30 vs tin@5% +2.13 (8.1σ).
  WHAT SURVIVES — the left flank transplants PERFECTLY at matched per-class AND
  matched images: 5/cls gives +1.48 (C100@1%) vs +1.60 (tin@1%) across 2x
  images/classes/resolution — a universal ~+1.5 floor; 50/cls at 500 imgs gives
  +6.62 (C10) vs +5.92 (stl) across 3x resolution. Something per-class-like is
  real; the FIXED-COUNT threshold is what died.
  RIGHT FLANK: no single covariate collapses datasets (at 5000 imgs: +1.09 C10 /
  +2.13 tin / +3.19 super / +4.14 C100; neither %, imgs, baseline, nor headroom
  orders all four). Redundancy onset is task-specific.
  *** THE FORK RESOLVED BY THE PROBE (2026-07-17, FINDINGS Q6.9f): REALIZATION
  IS UNIVERSAL, THE FEATURE GAIN IS DATASET-DEPENDENT. Decomposing Δ = R x G
  (R = e2e/probe-gap, G = full-label probe gap):
    5/cls:  R = 38% (C100@1%, G +3.88) vs 37% (tin@1%, G +4.33)
    25/cls: R = 83% (C100@5%, G +6.35) vs 80% (tin@5%, G +2.67)
  R matches across datasets AT BOTH COUNTS; G varies 2.4x at 5%. The law:
  Δ = R(labels/cls) x G(dataset, images), R universal, G task-specific.
  tin@5% = 0.80 x 2.67 = +2.14 (measured +2.13) — realization was never broken
  on tin; the prior just provides less there. G is NON-MONOTONE in data and
  dataset-specific in shape: C100 3.88@500 -> 6.35@2500; tin 4.33@1000 ->
  2.67@5000. The e2e peak sits where G peaks (R saturates early).
  *** AMENDED same day (all 10 probes done): the 50/cls probes BREAK the
  multiplicative form — R(50) measured 113%/137%/126% (C100@10%/C10@1%/stl@10%),
  the "88-95%" band was WRONG. Correct form is ADDITIVE: Δe2e = G + readout(k),
  readout MONOTONE in labels/cls and crossing zero between 25 and 50:
    5/cls: −2.40,−2.73 | 25/cls: −1.06,−0.54 | 50/cls: +0.48,+1.80,+1.23.
  Positive readout at ≥50/cls means aux features are EASIER TO READ with few
  labels (self-realization C10@1%: base 72.9% vs aux 78.2% of own probe
  ceiling) — the prior confers BOTH better features (G) AND more label-
  efficient readout. Q7.3's "gains are linearly shallow" is the same property.
  Component-wise transplant: C10@1% vs stl@10% at matched 500/50cls — G 4.81
  vs 4.70, readout +1.80 vs +1.23. G(C100): 3.88@500 -> 6.35@2500 -> 3.66@5000
  — G PEAKS AT 5%, exactly where the e2e envelope peaks: the envelope shape IS
  the feature-gain curve. Universal-R at 5/25 cls stands (38/37%, 83/80%).
  *** super@1% LANDED BETWEEN THE BRANCHES (2026-07-17): 22.50±0.63 →
  24.59±0.31 = +2.09 ±0.40. 2.5σ below additive-granularity (+3.1), 1.5σ above
  dataset-effect (+1.5). vs C100@1% (+1.48, byte-identical pixels): coarsening
  5->25/cls bought +0.61 ±0.42 — granularity effect REAL-LOOKING BUT MODEST,
  ~1/5 of the original threshold story. Suspect assumption now measurable:
  G=3.88 was measured under 100-way FINE CE and transplanted to 20-way coarse
  training; if coarse CE makes the aux target more redundant, G_coarse < 3.88
  and the shortfall is G's, not readout's. probe3 wave measuring G_coarse on
  super1/super5 pairs (super2 on landing).
  *** SUPER PROBES ANSWERED IT (2026-07-17, FINDINGS Q6.9h): G IS LABEL-SPACE-
  INVARIANT (coarse vs fine on identical pixels: 4.07 vs 3.88 @500 imgs; 5.44
  vs 6.35 @2500, 1.8σ) — G is a property of (pixels, images, aux config) only.
  The READOUT term follows TASK PERFORMANCE, not label budget: sign flips at
  baseline ≈ 30-35% in EVERY label space (9/9 probed cells: negative below
  base ~25, positive above ~39). Within-space it rises with per-class
  (20-way: −1.98@25 → +0.40@125) but the crossing count is space-dependent.
  "LABELS PER CLASS" WAS A PROXY for baseline height — that is why it looked
  universal on C100/tin and then failed to transfer.
  REVISED LAW: Δe2e = G(pixels, images) + readout(task performance).
  *** super@2% LANDED IN BAND (2026-07-17): 29.84±0.08 -> 34.93±0.34 =
  +5.08 ±0.20 vs predicted +4.3..+5.0. DATASET-EFFECT BRANCH DEAD (13σ).
  Relabeling alone (10->50/cls, byte-identical pixels) bought +2.58. Granularity
  pattern complete: +0.61 @500 imgs, +2.58 @1000, ~0 @2500+ (readout saturated).
  TENSION RESOLVED same day by the super2 probes: G(1000)=+5.33 (steep rise
  500->1000 then flat: 4.07/5.33/5.44 — the interpolation was the error),
  readout(29.84)=−0.25 (mildly negative, as the sign law requires). 10/10
  probed cells fit the sign law; crossing pinned to base ∈ [29.8, 39.4],
  likely just above 30.
  ALSO LANDED: C10@25% = −0.83±0.20 (predicted −0.5..0 — overshoot confirmed,
  slightly beyond band; second negative C10 cell). stl@50% 3-seed = +3.18±0.32
  vs C10@5% mirror +4.41 (3.5σ: G(stl) falls faster with data than G(C10); the
  500-img pair matched component-wise, the 2500-img pair splits).
  ConvNeXt-tiny+AdamW (DIAG ONLY, FINAL 3 seeds): 23.31±1.26 -> 33.88±0.37 =
  +10.57 ±0.76 — largest gain in the study, first non-ResNet + non-SGD. The
  baseline is underfit (23 vs R18's 40) — read as "prior rescues underfit
  modern backbone", NOT a headline row. Note baseline σ 1.26 vs aux σ 0.37:
  the variance-reduction pattern again, at its most extreme.
  *** tin@10% LANDED IN BAND (2026-07-17, 3 seeds): 33.60±0.27 → 35.24±0.38 =
  +1.65 ±0.27 vs original band +1.5..+2.2 (0.6σ below the conditional floor).
  TIN ENVELOPE COMPLETE AND FLAT: +1.60@1% / +2.13@5% / +1.65@10% — never
  above +2.2 at any scale, exactly as falling-G(tin) predicts.
  *** tin@10% PROBED (2026-07-17): G=+1.70, readout(33.60) = **−0.06** vs the
  advance demand "≈ 0 ± 0.5 at the crossing" — the sharpest sign-law hit yet.
  SIGN LAW NOW 11/11; crossing pinned to baseline ∈ [33.6, 39.4]. G(tin)
  monotone falling: 4.33@1000 -> 2.67@5000 -> 1.70@10000 (band said 1.8-2.4;
  landed a hair below). tin is simply a LOW-G DATASET; its readout behaves
  exactly like every other dataset's.
  *** G CURVES COMPLETED (2026-07-18, probe4 wave, all four datasets):
    G(C10):  4.81@500  5.52@1000  3.97@2500  0.65@5000 — peaks at 1000 (2%)
    G(stl):  4.70@500  4.98@1000  3.20@2500
    G(C100): 3.88@500  5.33@1000c 5.44@2500c/6.35f  3.66@5000
    G(tin):  4.33@1000 2.67@5000  1.70@10000
  In EVERY dataset the e2e envelope peak sits at the G peak — the envelope
  shape IS the feature-gain curve (readout saturates/flattens early). The two
  *** CORRECTED BY THE AUDIT (2026-07-19, analysis/audit_law.py): "every
  dataset" was an overclaim — it holds on C10 (both peak @2%) and C100 (both
  @5%) where readout is flat across the curve, but NOT on tin: with the
  completed curves, G(tin) peaks at 1% (4.19) while the envelope peaks at 5%
  (+2.13), because the left-flank readout penalty spans 2.6 points
  (−2.69@1% -> −0.05@10%) and opposes falling G. Correct statement: envelope
  peak = G peak WHERE readout is ~flat; on tin the flat envelope is falling
  G x rising readout. Δ = G + readout needs no correction — the shorthand did.
  recorded G estimates from the miss post-mortem: G(C10,1000)~5.6 HIT (5.52);
  G(stl,1000)~3.1-3.4 MISSED (4.98) — so the stl@20% e2e shortfall was the
  READOUT term, not G.
  READOUT POSITIVE BRANCH MAPPED on ceiling-free C10 (50k-image probe):
  +1.80@39.4 -> +1.14@51.7 -> +0.44@69.1 -> +0.44@80.7 — positive throughout,
  DECAYING toward zero with sufficiency, not growing. Sign law now 15 clean
  cells, zero violations.
  *** tin20 ANSWERED — TIN'S FLAT ENVELOPE IS A GRANULARITY ARTIFACT
  (2026-07-18, FINDINGS Q6.9d). The within-tin control: 20 of tin's 200
  classes (every 10th sorted wnid), 1000 imgs / 50-per-cls / 1400 steps /
  20-way at 64x64, calibration byte-identical to tin@1%'s. Predicted from
  measured quantities BEFORE the runs: +3.8..+5.5 if granularity governs, ~+2
  if tin is a low-G pixel property. LANDED: 41.47±1.50 -> 45.53±0.50 =
  **+4.07 ±0.92** — in band, 2.5x the largest gain tin ever produced (+2.13),
  2.2σ from the pixel-property branch. Moving ONLY the label space took tin
  from +1.60 to +4.07.
  *** DEEPENED TO 10 SEEDS (2026-07-18, xspace Phase B): 40.69±1.00 ->
  46.11±0.82 = **+5.42 ±0.41** — STRENGTHENED, not regressed. The 3-seed
  baseline's high seed (43.20, seed 0) was the outlier; 7 new baseline seeds
  all landed 39.7-41.1 while the aux cell stayed ~46. Still in the +3.8..+5.5
  granularity band (now at its TOP); the gap vs tin@1%'s +1.60 is 8.4σ.
  tin20 is now the 3.4x loud version of tin's suppression, not 2.5x.
  PROBED: G(tin20-pixels, 1000) = +5.23 ±0.90 ≈ G(tin, 1000) = 4.33 ±0.30 —
  G is class-subset-invariant within noise; the e2e gain is G-driven exactly
  as the branch requires. readout = −1.16 ±1.29: TOO NOISY to score against
  the sign law (3-seed σ 1.50 on both the aux probe and baseline e2e); not
  counted for or against. NOTE vs super: tin20's images DIFFER from tin@1%'s
  (class subset), unlike super's byte-identical pixels — the control is
  dataset/resolution/pipeline-matched, not image-matched.
  CONSEQUENCE: "tin is a low-G dataset" is WRONG as a pixel statement — the
  200-WAY LABEL SPACE, not the pixels, suppressed tin's envelope. G measured
  under a fine label space is NOT the pixels' G ceiling. The law's G term is
  G(pixels, images, LABEL SPACE) after all — label-space-invariance (Q6.9h)
  held between 100-way fine and 20-way coarse on C100, but breaks between
  200-way and 20-way on tin (4.33 vs 5.23 at matched images, 0.9σ — weak
  evidence, needs seeds; the e2e gap +1.60 vs +4.07 at 2.6σ is the loud
  version).
  *** 2x2 TRAIN-SPACE x PROBE-SPACE CROSSING LAUNCHED (2026-07-18, xspace
  wave) — user decision: resolve WHERE label-space invariance breaks. The
  tin20-vs-tin G comparison (5.23 vs 4.33, 0.9σ) confounds three things:
  training label space, probe label space, and (on tin) pixel population.
  DESIGN: probe BOTH checkpoint sets under BOTH label spaces.
    Phase A (C100 pixels, BYTE-IDENTICAL images — no pixel confound):
      {abl1, super1} ckpts x {cifar100 100-way, cifar100super 20-way} probes.
      NOTE this also patches a hole in Q6.9h: the original "invariance" showed
      G_finetrained-fineprobed ≈ G_coarsetrained-coarseprobed — it never
      crossed them, so training-space and probe-space effects could cancel.
    Phase B: deepen tin20 + tin@1% cells to 10 seeds (probe σ 1.50 at 3).
    Phase C: tin 2x2 at 10 seeds.
  KNOWN CORNERS: G_200(tin@1%ckpt)=4.33±0.30, G_20(tin20ckpt)=5.23±0.90,
  G_fine(abl1ckpt)=3.88, G_coarse(super1ckpt)=4.07.
  *** PHASE A ANSWERED — H-INVARIANCE WINS ON C100 (2026-07-18). The full
  2x2 on byte-identical pixels (G = aux-minus-baseline probe gap, 3 seeds):
                        fine probe        coarse probe
    fine-trained          +3.88 ±0.39       +3.71 ±0.31
    coarse-trained        +3.37 ±0.34       +4.07 ±0.51
  All four corners in 3.4-4.1, pairwise within ~1σ; interaction +0.87 ±0.79
  (1.1σ, n.s.). NO probe-stick effect, NO training-space effect: on C100
  pixels G is fully invariant to both label spaces (100<->20-way), and the
  Q6.9h hole (diagonal-only comparison) is patched by the crossing.
  CONSEQUENCE: if the tin 2x2 at 10 seeds preserves the 5.23-vs-4.33 shift,
  it is NOT a label-space artifact — it is either 200-way/5-per-cls TRAINING
  specifically, or tin20's different PIXEL POPULATION.
  HYPOTHESES RECORDED IN ADVANCE:
    H-invariance (Q6.9h fully right): all four G's agree within noise at
      matched pixels — C100 crossing lands 3.8-4.1 in all cells; tin shift
      was noise/pixels.
    H-probe-stick: G depends on the PROBE space — off-diagonal cells move
      toward the probe space's diagonal value (e.g. G_200(tin20ckpt) ≈ 4.3).
    H-training: G is a property of the CHECKPOINTS — each ckpt set carries
      its G across probe spaces (G_200(tin20ckpt) ≈ 5.2, G_20(tin@1%ckpt)
      ≈ 4.3). On tin this is still training-space-OR-pixel-population
      (entangled by design); the C100 crossing has no such entanglement, so
      Phase A cleanly separates H-probe-stick from H-training.
  *** PHASE B+C ANSWERED — BOTH EFFECTS ARE REAL ON TIN, AND THEY CANCEL ON
  THE DIAGONAL (2026-07-18, 10 seeds every corner, xspace COMPLETE):
                        200-way probe     20-way probe
    tin@1%-trained        +4.20 ±0.15       +6.64 ±0.47
    tin20-trained         +3.08 ±0.16       +4.50 ±0.58
    probe-space effect (20-way − 200-way): +1.93 ±0.39 = 5.0σ
    ckpt-set effect (tin@1% − tin20):      +1.63 ±0.39 = 4.2σ
    interaction: +1.02 ±0.78 (1.3σ, n.s.)
  NO recorded hypothesis wins outright: H-invariance is FALSIFIED on tin
  (two ~5σ main effects) yet the DIAGONAL is invariant (4.20 vs 4.50, 0.5σ)
  — the two effects OPPOSE and cancel there, which is precisely the
  cancellation hole the crossing was built to catch (on tin it is real; on
  C100 Phase A showed no effects at all). The 3-seed "5.23 vs 4.33 G shift"
  is GONE at 10 seeds: G_20(tin20ckpt) settled to 4.50±0.58.
  CONSEQUENCES:
  (1) THE GRANULARITY EFFECT ON e2e IS A READOUT EFFECT, NOT A G EFFECT.
      At matched images (1000), diagonal G is matched (4.20 vs 4.50) while
      Δe2e differs 3.6x (+1.49 vs +5.42). The difference is carried by
      readout: −2.71 at base 5.30 (tin@1%) vs +0.92 at base 40.69 (tin20)
      — both signs exactly as the sign law requires (18 clean cells now).
      Relabeling tin 200-way -> 20-way moved Δ by moving the BASELINE
      through the readout crossing, not by changing feature gain. "tin is
      a low-G dataset" dies COMPLETELY: G(tin,1000)≈4.2-4.5 ≈ C100's 5.33
      ballpark; the flat tin envelope = falling G right flank + readout
      penalty at 200-way baselines.
  (2) MEASURED G IS NOT PROBE-SPACE-INVARIANT ON TIN (it was on C100):
      coarse 20-way probes report ~+1.9 MORE aux-vs-baseline gap than
      200-way probes on the SAME checkpoints. G numbers are comparable
      ONLY within a fixed probe space; cross-dataset G comparisons keep
      their probe spaces attached (all recorded G curves are same-space,
      so the curves stand).
  (3) The ckpt-set effect (+1.63: tin@1%-trained pairs carry more G under
      EITHER probe) stays entangled between training label space and pixel
      population by design. tin20b's 3-seed G=2.37 vs tin20's 10-seed 4.50
      (2.6σ) hints the pixel DRAW moves measured G while e2e stays put
      (+5.27 vs +5.42, 0.2σ) — 3-seed probe, not adjudicated; do not build
      on it without deepening tin20b's probes.
  *** FOLLOWUP2 WAVE ANSWERED BOTH LOOSE ENDS (2026-07-19):
  (A) PROBE-BUDGET CONTROL — the probe-space effect SURVIVES, STRENGTHENED.
      Suspected confound: the 200-way probe trains on 100k images vs the
      20-way probe's 10k. Measured via --shots on all four corners: G_200
      RISES with probe budget (2.56@25 -> 3.06@50 -> 3.43@100 -> 3.99@250
      -> 4.19@500/cls on tin@1% ckpts), so at MATCHED 10k total the gap
      WIDENS: G_200@50/cls=3.06±0.17 vs G_20=6.70±0.45 = 7.6σ (tin20 ckpts:
      3.4σ). The budget confound worked in the OPPOSITE direction — coarse
      probes genuinely read more aux-vs-baseline gap on tin; the Q6.9j
      label-space interpretation stands, now confound-controlled. Mechanistic
      corollary: the FINE probe is label-hungry — the feature gap is fully
      visible to a 20-way probe at 25/cls but needs 500/cls to saturate the
      200-way probe. Same shape as the e2e readout penalty at fine spaces.
  (B) TIN20B DEEPENED TO 10 SEEDS — THE G TENSION WAS 3-SEED NOISE. e2e
      +4.85±0.38 (vs tin20 +5.42±0.41, 1.0σ); G_20 = 3.94±0.42 (vs tin20
      4.58±0.60, 0.9σ — the 3-seed 2.37-vs-5.23 gap is GONE); readout +0.91
      at base 44.16 vs tin20's +0.84 at 40.69 — the two disjoint draws now
      agree on EVERY quantity: e2e, G, and readout (sign law 19th cell).
      Class draw is irrelevant, full stop; "pixel draw moves measured G" is
      dead. New off-diagonal G_200(tin20b)=2.54±0.17: the ckpt-set effect
      REPLICATES on the b draw (vs G_200(tin@1%ckpts)=4.19: 7.3σ) — fine-
      TRAINED checkpoint pairs carry a larger measured gap under either
      probe, on two independent class draws. (tin20-vs-tin20b within-space
      2.3σ — borderline, noted, not built on.)
    tin@25% FINAL: **+0.10 ±0.28** (49.19±0.21 -> 49.29±0.44) — NEUTRAL,
      inside the qualitative band (flat ≤+2.2). The tin envelope right
      flank reaches zero by 25k images, exactly as falling G(tin) predicts:
      +1.49/+1.81/+2.13/+1.65/+0.10 at 1/2/5/10/25%.
- *** tin@100% LANDED — THE CAMPAIGN IS COMPLETE (2026-07-19, frontier2
  COMPLETE marker): 66.17±0.25 -> 65.75±0.58 = **−0.42 ±0.36** — inside the
  qualitative band (neutral ±0.5), 1.2σ from zero, mirroring C10@100%'s
  −0.26 (champion λ0=1.0 verbatim; per-regime λ0=0.1 would likely restore
  exact neutrality as on C100 — not worth the GPU). THE TIN ENVELOPE IS
  CLOSED: +1.49/+1.81/+2.13/+1.65/+0.10/−0.42 at 1/2/5/10/25/100% — falling
  G(tin) all the way down, left flank readout-suppressed. Every cell of the
  user-approved remaining-experiments program has now landed and been scored
  against its pre-registered band.
- auxmag3 SCORED (2026-07-19, both cells + probes, 3 seeds):
    @1%: +1.63 ±0.09 — IN BAND (+1.3..+1.7), 1.1σ from champion's +1.49:
      no added value, exactly as the redundancy account predicted.
      G = 4.52 ±0.18 vs champion 4.19 (1.9σ, below the G>=5.0 falsifier).
    @5%: **+2.72 ±0.19 — NOMINALLY ABOVE the band (+1.8..+2.4) and past the
      e2e falsifier** (+0.59 ±0.26 above champion = 2.3σ). BUT the G probe
      shows NO feature-level difference (2.83 vs 2.67, 0.6σ) — an e2e excess
      with flat G would have to be a readout effect the bank should not
      touch, so 3-seed noise is the leading account (the study's oldest
      lesson). NOT ADJUDICATED: deepening BOTH 5% aux cells to 10 seeds
      (auxmag3deep wave, launched immediately — GPU freed by frontier2).
      If +0.59 survives 10v10 (~5σ at that power), the octave DOES carry
      value at mid-data and the bank-design axis reopens; if it shrinks,
      the redundancy account stands everywhere.
- LEVELS REANALYSIS of the 2x2 (2026-07-19, from existing probe JSONs):
  under the common 200-way stick, the ckpt-set effect is a BASELINE effect:
  base 19.27 (tin@1%-trained) vs 23.71/23.98 (tin20/tin20b-trained) while
  aux-under-fine 23.46 ≈ base-under-coarse 23.71 — the prior almost exactly
  closes the weak-supervision deficit ("the prior substitutes for
  supervision"). But 20-way-trained ckpts saw only 20 classes' PIXELS:
  label space and pixel population still entangled on tin. C100's crossing
  (training space does nothing on byte-identical pixels) suggests PIXELS.
- auxmag3 LAUNCHED (2026-07-19, user asked "why wouldn't a wider bank add
  value?" -> the one non-redundant widening gets its test): energy-magnitude3
  = the committed 8-pair bank + ONE LOWER OCTAVE (sigma=4, k=17 so the
  envelope fits), 12 pairs, orientations unchanged (extra orientations are
  near-linear combinations -- only the octave is a new direction). Run on tin
  (64x64 makes the octave physically meaningful), champion config otherwise.
  Bank pinned in test_bank_regression (additive; committed banks untouched).
  PREDICTIONS RECORDED IN ADVANCE (redundancy account = no added value):
    auxmag3_tin_1pct: +1.3..+1.7 (champion +1.49 +-0.09; shared baseline)
    auxmag3_tin_5pct: +1.8..+2.4 (champion +2.13 +-0.18)
    G_200(auxmag3@1% ckpts) ~ 4.2 (champion's 4.19). FALSIFIER: e2e >= +0.5
    above champion on either cell, or G >= 5.0 -- either says the new octave
    carries supervisory value the 2-octave bank misses, reopening the
    bank-design axis.
- tinsuper LAUNCHED (2026-07-19): tin's images relabeled sorted-wnid//10 ->
  20 coarse groups; tin@1%'s COMMITTED SUBSET via SUBSET_ALIAS (byte-
  identical pixels, all 200 fine classes at 5/cls = 50/coarse-cls, 1400
  steps). The tin mirror of cifar100super; disentangles the ckpt-set effect.
  PREDICTIONS RECORDED IN ADVANCE:
    PRIMARY FORK — G_200(tinsuper ckpts) on the common 200-way stick:
      H-pixels (C100's training-space-invariance transplants; pixel
        population carried the ckpt effect): ≈ 4.2 (tin@1%'s value).
      H-label-space (training label space carried it): ≈ 2.5-3.1
        (tin20/tin20b's value).
    e2e: BOTH branches predict a granularity boost over tin@1%'s +1.49 via
      readout (baseline moves toward/past the crossing); band +2.5..+6.5,
      wide because G_20 on these ckpts is unmeasured in tinsuper's own
      coarse space. Any Δ >= +2.5 = relabeling-alone reproduces the
      granularity effect on byte-identical tin pixels (mirror of super@2%'s
      +2.58); Δ ~ +1.5 would instead say tin20's pixel POPULATION, not its
      label space, carried the e2e granularity effect — which would
      contradict the readout account and be a major surprise.
  *** tinsuper LANDED (2026-07-19) — FORK ANSWERED: THE TRAINING LABEL
  SPACE, NOT THE PIXEL POPULATION, CARRIES THE CKPT-SET EFFECT; AND THE
  e2e BAND MISSED IN THE MOST INSTRUCTIVE WAY POSSIBLE.
    THE FORK (primary, hit cleanly): G_200(tinsuper ckpts) = **+2.55
    ±0.41** — 0.0σ from tin20b's 2.54, 1.2σ from tin20's 3.08, 3.8σ from
    the H-pixels branch (4.2). On BYTE-IDENTICAL pixels, switching the
    training loss from 200-way fine CE to 20-way coarse CE (even an
    ARBITRARY coarse partition) cuts the measured aux-vs-baseline gap
    from 4.19 to 2.55. The "prior substitutes for supervision" account is
    now label-space-based, not pixel-based: fine-grained weak supervision
    (5/cls, 200-way) is where the prior has the most feature work to do.
    ALSO: C100's training-space invariance does NOT transplant to tin —
    the H-pixels prediction leaned on it and lost.
    e2e: **+1.01 ±0.17** (14.08±0.29 -> 15.09±0.06) — MISSED the
    +2.5..+6.5 band, BELOW even the ~+1.5 "surprise" branch. Post-mortem:
    the band assumed relabeling raises the baseline toward the crossing
    (as semantic coarsening did: tin20 base 40.7, super base 29.8). The
    POSITIONAL groups are visually incoherent, so the baseline stayed at
    14.08 — far below the crossing — and the law then PREDICTS no boost:
    readout(14.08) = −0.36 (mildly negative, sign law 20th clean cell),
    and G_20own(+1.37) + readout(−0.36) = +1.01 EXACTLY. So the miss
    falsifies "label COUNT alone buys the granularity gain" (my data.py
    comment said this — wrong) and confirms the revised law's actual
    form the hard way: a cell ENGINEERED to have few classes but low
    task performance gets NO readout boost. Granularity helps e2e only
    insofar as it raises baseline task performance.
    CAVEAT for the fork: tinsuper's coarse groups are arbitrary while
    tin20's are semantic (real wnids) — G_200 2.55 vs 3.08 (1.2σ) hints
    semantic coherence may matter a little within coarse-trained; not
    adjudicated. The fine-vs-coarse main effect (4.19 vs 2.5-3.1) is 3.8σ+
    and stands regardless.
  *** PROBE-CEILING RULE (2026-07-18): the Δ = G + readout decomposition is
  trustworthy ONLY while the probe holds far more labeled data than the cell.
  stl's probe has just 5000 imgs (500/cls); at stl@50% the baseline's probe
  uplift is +0.28 — the cell has ~reached its own probe ceiling and 'readout'
  loses meaning (stl@20% −0.62±0.84, stl@50% −0.02 are EXCLUDED as
  measurements, not counterexamples). CIFAR/tin probes (50k/100k) are safe at
  every % measured.
  *** C10 ENVELOPE COMPLETE (2026-07-17, champion λ0=1.0 verbatim, 3 seeds):
    1% +6.62 | 2% +7.14 | 3% +5.38 | 5% +4.41 | 7% +2.21 | 10% +1.09 |
    15% −0.66 | 25% −0.83 | 100% −0.26±0.18
  Shape: positive 1-10%, crosses zero at 10-15%, dips negative at 15-25%, then
  RECOVERS toward zero at 100% (−0.26, within noise of neutral). The recovery
  fits the overshoot account: at 15-25% there are too few post-decay steps to
  undo the early λ=1.0 shaping; at 100% (78k steps, most after λ≈0) the damage
  washes out. NOTE this is λ0=1.0 verbatim — C100's 100% neutrality headline
  used λ0=0.1; the matched C100 λ0=1.0-sched cell gave +0.08. On C10 the
  champion is NEVER positive above 10%: per-regime λ0 (0.3 at 15%+) is not
  optional on easy datasets.
  *** TWO PREDICTIONS MISSED (2026-07-17), same root cause — extrapolating
  unmeasured G curves:
  (1) stl@20% = +4.36 ±0.56 vs predicted +5.5..+7.0. The C10@2% mirror
      (+7.14) is now 3.1σ away: at 1000 imgs / 100-per-cls the C10<->stl
      transplant SPLITS, though it was component-wise perfect at 500 imgs.
      Same pattern as the 2500-img split (stl@50% +3.18 vs C10@5% +4.41):
      G(stl) FALLS with data from 500 imgs on, while G(C10) rises 500->1000.
      G shape is dataset-specific and cannot be extrapolated — the same error
      as the super2 interpolation, repeated. Estimated G(stl,1000) ≈ 3.1-3.4
      (falling from 4.70@500) vs G(C10,1000) ≈ 5.6.
  (2) c10@15% λ0=0.3 rescue = −0.13 ±0.29 vs predicted +0.3..+1.0. The rescue
      restored NEUTRALITY, not gain (champion λ0=1.0: −0.66). The prediction
      anchored on C100@15%'s +2.94 — but C10@15% (baseline 85.65) is past
      G(C10)'s zero: weakening λ0 removes the shaping COST, and there is no
      feature gain left to unlock. LESSON: per-regime λ0 protects from harm
      but cannot manufacture gain past G's zero-crossing.
  *** Q9.4 SETTLED (2026-07-18, 10 seeds/cell): THE 1-2% BAND IS A FLAT
  PLATEAU. Δ(1%)=+6.37±0.15, Δ(2%)=+6.66±0.27; difference +0.29±0.31 = 0.95σ,
  95% CI [−0.31,+0.89] — any summit is bounded below ~1 point. Both deltas
  SHRANK from their 3-seed values (+6.62→+6.37, +7.14→+6.66): regression to
  the mean, exactly as the 3-seed warning predicted. Headline C10 1-2% cells
  now cite the 10-seed numbers.
  *** VARIANCE-REDUCTION ANSWERED NEGATIVE (2026-07-18, 10v10): at 1% the
  direction REVERSES (aux σ 0.38 > baseline 0.28, F=0.55); at 2% F=2.56 vs
  critical ~4.03 — not significant. The 3-seed pooled pattern (12/18 pairs,
  p=0.073) DOES NOT REPLICATE when powered. What remains real: the prior
  rescues INSTABILITY (R50 no-hn σ 3.18→0.32, ConvNeXt 1.26→0.37, the 2pct
  baseline's low seed) — a statement about failure modes, not routine seed
  noise. Do not claim variance reduction as a method property.
  num_workers preserved per cell (audited: none_1/aux_1/none_2 nw=8, aux_2 nw=2).
  *** Q7.3 SETTLED (2026-07-17): shots dose-response. e2e realizes EXACTLY what
  a same-label-budget LBFGS probe realizes (e2e +1.91 ≈ 5-shot gap +2.08; e2e
  +5.30 ≈ 25-shot gap +5.31). Left flank = label scarcity at readout, NOT an
  optimization failure. And R(k) is not universal: @5%-features' gain is 81%
  visible to a 5-shot head; @1%-features' gain needs hundreds of labels.
  *** COMBO ANSWERED NEGATIVE (2026-07-17): fwd stem + aux at C100 1-2% =
  +1.73/+2.98 — ≈ aux alone (+1.90/+3.14), 2.9σ WORSE than fwd stem alone
  (+2.54/+3.52) @1%. NOT additive; the aux constraint is redundant when the
  moments are already in the input. aux.py's "may be additive" is falsified.
  *** C10@15% = −0.66 ±0.22 (3σ) — the champion's FIRST negative cell. The
  zero-crossing is DATASET-DEPENDENT (C100 crosses ~25%, C10 between 10-15%).
  "Positive up to 25%" is a C100 statement. λ->0 makes END-of-training pure CE
  but does NOT guarantee neutrality — early-phase shaping can still cost at
  sufficiency. PREDICTIONS RECORDED: C10@15% λ0=0.3 => +0.3..+1.0; C10@25%
  λ0=1.0 (running) => −0.5..0.
  SEED-VARIANCE NOTE: aux cells' σ < baseline σ in 12/18 pairs; exact sign-flip
  test on 3v3 pairs p=0.073 — SUGGESTIVE ONLY, do not claim. Q9.4's 10-seed
  cells will power it properly.
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
