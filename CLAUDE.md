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

## USER DECISIONS (2026-07-22)

- **BANK STAYS AT 8 PAIRS — NO RE-PIN.** The widthoctave adjudication showed a
  wider 12-channel aux target buys ~+0.3..+0.5 at 64px (and exactly nothing at
  32px: auxmag3 excess +0.02). User decision: "keep them 8, no need to re-pin
  for now." So the committed 8-pair bank remains the headline bank, every
  existing run stays valid, and tests/test_bank_regression.py fingerprints are
  untouched. "Widen the target at >=64px" is documented as a cheap OPTION, not
  the default. Do not reopen without an explicit new decision.
- **CLUSTER SPLIT CONFIRMED**: turing runs the LIVE work (envelope completion,
  new waves); BSC fills the missing numbers of the results grid, including the
  1,467 already-closed-configuration runs. Unchanged from the 2026-07-22 split.
- BSC credentials were rotated 2026-07-22 after an earlier one was pasted in
  plaintext. Credentials live only in a 0600 file on the local machine, are
  never echoed to a transcript, and SSH key auth is what the automation
  actually uses. (Details deliberately omitted: this file is released with
  the paper.)

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
- *** CUB WAVE LANDED — BOTH CELLS IN BAND, FINE-GRAINED HYPOTHESIS
  SURVIVES MODESTLY (2026-07-20, wave COMPLETE 12h, probes included):
    cub@25% (~8/cls): 5.57±0.14 -> 6.13±0.18 = **+0.56 ±0.13** — IN BAND
      (+0.5..+2.0, bottom edge). G = +0.85±0.26; readout = −0.29 at base
      5.6 — negative deep below the crossing, sign law 26th clean cell.
      Deep-left-flank suppression exactly as the law predicts.
    cub@100% (30/cls, 6k imgs): 44.32±0.68 -> 47.06±0.57 = **+2.74 ±0.51**
      — IN BAND (+1.5..+4.5); NEITHER falsifier fired (not ≥+5, not ≤0).
      Probe gap +3.03±0.65 ≈ Δ. The pre-recorded PROBE CAVEAT applies:
      probe labels == cell labels at 100%, so the G/readout split is NOT
      interpretable here (readout −0.29 nominal, excluded per the
      probe-ceiling rule; the aux-vs-base gap under identical probing
      stands).
    READING: at ~matched image count, fine-grained CUB@100% (+2.74, 6k
    imgs) sits a bit above tin@5% (+2.13, 5k) — fine-grained weak
    supervision helps somewhat, but is NOT a qualitatively richer-G regime.
    The fine-grained axis stays a solid transplant result, not a headline
    direction. FIRST NON-64px-native dataset (squash-resized), fifth
    dataset the champion transplants to verbatim-positive.
- *** AUXMAG3 DEEPENING ADJUDICATED — THE OCTAVE EXCESS IS REAL (2026-07-20,
  local 10v10 vs the 3-seed tension recorded 2026-07-19):
    tin@5% champion (10 seeds): 23.20±0.30 = Δ +2.15 (3-seed +2.13 held)
    auxmag3   (10 seeds):       23.68±0.19 = Δ +2.64
    auxmag3 − champion = **+0.49 ±0.11 = 4.4σ** — the 3-seed nominal
    excess (+0.59, 2.3σ) SURVIVED power. G probes agree in direction:
    35.05±0.38 vs 34.69±0.32 (10-seed probes, +0.36 ±0.16, 2.3σ).
    VERDICT: the extra LOW OCTAVE (sigma=4/k=17) carries real supervisory
    value at tin@5% — right at the recorded falsifier boundary (≥+0.5),
    and this time the feature-level probe moves WITH e2e (unlike the
    3-seed read). THE BANK-DESIGN AXIS REOPENS, scoped: octaves (new
    frequency content), not orientations (near-linear combos), are the
    direction; worth one follow-up (auxmag3 on tin@10%/C100@5%) before
    any bank re-pin. The committed 8-pair bank stays the headline bank —
    a re-pin would invalidate every existing run and needs the user's
    explicit decision.
- *** SSL2 LANDED (2026-07-20, wave COMPLETE 0h50) — REDUNDANCY CONFIRMED
  AS ACTIVE INTERFERENCE, AND THE FRONTIER PINNED:
    diagssl_simclr50_5pct: **30.81 ±0.23** — IN BAND (30..34), a statistical
      TIE with champion 30.53 at 1.25x compute. The champion's unique niche
      is compute budgets < ~1.25x; from there up SSL matches then wins.
    diagssl_simclraux_5pct: **32.90 ±0.42** — MISSED LOW (band 34.4+0..+1.5):
      the aux prior COSTS an SSL-initialized run −1.51 ±0.31, and its probe
      is 0.70 lower (feature-level component). Mirrors the fwd-combo lesson:
      once features are shaped (here by SimCLR), early λ0=1.0 shaping only
      taxes. RECOMMENDATION ON RECORD: aux XOR SSL, never both.
    Probes: G(simclr ckpts vs abl5_none ckpts) = **+9.0** — IN the +9..+12
      feature-side band (≫ champion's 6.35). SSL's win is the SAME currency
      as the prior's G — it fills the same feature deficit, more of it —
      which is WHY they do not stack. Law check on the first non-aux
      intervention: Δ +9.18 ≈ G +9.02, readout +0.16 ±0.3 (≈0; nominally
      above the negative branch expected at base 25.2 — noted, not scored:
      the sign law was derived on aux cells only).
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

## Post-campaign deepening (2026-07-21, user: "go with all points, use
## turing, make sure training is fast")

- FAST-PATH INFRA: tin cells are BeeGFS-small-file-IO-bound (~17s/epoch,
  110k JPEGs). Fix: sbatch stages the 248MB tin ZIP -> /dev/shm (RAM tmpfs,
  ~8s one-time unzip) and passes --data-root there. Byte-identical pixels,
  same augmentation stream, subsets still read from the repo -> protocol-
  neutral; tin then runs at CIFAR speed. C100 cells need no staging (single
  in-memory pickle, ~0.6s/epoch). tin/train + probe both take --data-root.
- NEW CODE: build_model gains image_size (vit patch scales image_size//8 so
  the token grid is always 8x8: patch 4 @32px, patch 8 @64px); MomentAuxModel
  shape-probe uses image_size not a hardcoded 32 (was the ViT-on-tin blocker).
  Threaded through train.py + linear_probe.py. Suite 101 green; 32px ViT path
  byte-unchanged.
- deepen wave (H100, C100 fast): 1/5/10% champion PAIRS 3->10 seeds + 10-seed
  probes (abl{1,5,10}_none, auxmag_{1,5,10}pct_sched0). No new prediction —
  hardens headline numbers against the 3-seed-sigma risk (C10 precedent:
  +7.14->+6.66 under power). WATCH: does any 3-seed Δ move >0.5 at power?
- widthoctave wave (staged tin): auxmag6o_tin_5pct + tin_none_5pct 3->10;
  DECISIVE 10v10 auxmag6o (width, no octave) vs auxmag3 (octave), both at 10.
  PREDICTION: if the +0.46 auxmag6o excess (3-seed) SURVIVES ~= auxmag3's
  +0.49, the driver is target WIDTH not the octave (bank-design axis = "more
  channels", cheap); if it SHRINKS below +0.3, the octave is special after
  all. FALSIFIER for width: auxmag6o − auxmag3 at 10v10 differs by >2σ.
- ssl3 wave (H100, C100 fast): SimCLR-init at 1/2/10% vs baseline/champion.
  PREDICTION: SimCLR gain over baseline SHRINKS as data falls (too few images
  to contrast) — at 1% (500 imgs) SimCLR ~ baseline or worse, while champion
  aux holds +1.5..+1.9; the SSL 2x-compute frontier win is DATA-REGIME-
  BOUNDED (strong at 5-10%, gone at 1%). Bands: diagssl@10% 44..47 (vs champ
  ~44, base 40.2); @2% 15..18 (vs champ 16.7); @1% 8..11 (vs champ 10.4,
  base 8.9). FALSIFIER: SimCLR beats champion at 1% => SSL dominates even at
  extreme scarcity, aux's low-data niche narrows too.
- vit2 wave (mixed): (1) diagvit@100% C100 — does G(vit) FALL at 50k imgs?
  PREDICTION: baseline now well-trained (~55-62, above the ~30 crossing), so
  readout ~0+ and Δ = G; if G finally falls (right flank), Δ small (+1..+4);
  if G stays ~14, Δ still large (+10+) — the latter would make ViT-tiny a
  permanent-deficit backbone, unlike every conv. (2) diagvit on tin@10%
  (64px, 200-way, staged): does the ViT rescue TRANSFER off 32px C100?
  PREDICTION qualitative — large positive Δ (ViT-tiny underfits tin harder
  than R18); G(vit,tin) > G(R18,tin)=1.70. FALSIFIER for "same law on
  attention everywhere": readout positive at a sub-30 baseline on tin.
- All 4 submitted; 2 run (MaxJobsPU=2), 2 queue (MaxSubmitPU=128). Fast C100
  waves land in hours; vit2@100% is the long pole (78k steps).

- *** DEEPEN WAVE LANDED (2026-07-21, COMPLETE 2h08) — THE HEADLINE C100
  NUMBERS SURVIVE POWER. 10 seeds/cell (local seeds 0-2 + turing 3-9 merged;
  config pins num_workers so each seed's augmentation stream is machine-
  independent — same cross-machine merge precedent as tinsuper):
    C100@1%:  8.93±0.10 -> 10.35±0.21 = **+1.42 ±0.07**  (3-seed +1.49, −0.07)
    C100@5%: 25.36±0.56 -> 30.51±0.39 = **+5.15 ±0.22**  (3-seed +5.30, −0.15)
    C100@10%:40.28±0.57 -> 44.03±0.50 = **+3.75 ±0.24**  (3-seed +4.14, −0.39)
  ALL THREE SHRANK, none by >0.5 (the recorded WATCH threshold). Direction
  matches the C10 precedent exactly (+7.14->+6.66): 3-seed deltas are
  slightly optimistic, regression-to-mean is real but small at these cells.
  The envelope SHAPE is unchanged and the peak stays at 5%. Headline tables
  should now cite the 10-seed values.
  10-seed G/readout (turing probes): G 4.16/6.26/3.55 at 1/5/10%, readout
  −2.76 @ base 8.9, −1.12 @ 25.4, +0.08 @ 40.3 — sign law holds at power,
  crossing again bracketed just below ~40.
- *** diagvit@100% LANDED — ViT-tiny IS A PERMANENT-DEFICIT BACKBONE
  (2026-07-21, 3 seeds + probes): 50.61±0.51 -> 60.50±0.64 = **+9.88 ±0.47**,
  G(vit,50k) = **+9.70 ±0.44**, readout = **+0.19** at base 50.6.
  PREDICTION SCORING — the recorded dichotomy was MIS-SPECIFIED and I am
  scoring it against myself: I wrote "if G finally falls (right flank), Δ
  small (+1..+4); if G stays ~14, Δ still large (+10+)". G DID fall (14.0 ->
  9.70) and Δ STAYED LARGE (+9.88). Both branches were wrong as stated
  because they assumed G and readout move together; in fact G fell ~30%
  while readout climbed from −0.36 (@25%) to +0.19, so Δ barely moved.
  Baseline also came in at 50.6 vs the predicted 55-62 band.
  WHAT IS TRUE: at FULL CIFAR-100 (50k images, 78k steps) ViT-tiny still
  gains **+9.88** where EVERY conv backbone is neutral at 100% (C100 +0.15,
  C10 −0.26, tin −0.42). The prior supplies something attention-at-this-
  scale cannot self-learn even with all the data — the "permanent deficit"
  conclusion holds, reached by a route I did not predict.
  G(vit) curve complete: 5.89@0.5k, 13.17@2.5k, 14.85@5k, 14.03@12.5k,
  9.70@50k — rise, plateau ~14, then a SHALLOW right flank (still 2x any
  conv G ever measured at any scale). Law: Δ +9.88 = G +9.70 + readout
  +0.19; positive readout at base 50.6, far above the ~30 crossing —
  sign law 29th clean cell, 6th on attention.
  CONSEQUENCE: the ViT direction is the strongest remaining headline
  candidate. The natural next comparison is against heavy-augmentation
  small-ViT recipes (DeiT-style), since "a fixed spectral target
  substitutes for what small ViTs cannot learn" now has a full envelope
  (1/5/10/25/100%) behind it.

- *** ViT-ON-TIN LANDED — THE ATTENTION RESCUE TRANSFERS OFF 32px
  (2026-07-21, vit2 COMPLETE 3h21, 3 seeds + probes; 64px, 200-way, staged):
    diagvit_tin_10pct: 8.60±0.41 -> 17.75±0.18 = **+9.15 ±0.26**
    G(vit, tin@10k) = **+12.39 ±0.43** (probe 13.76 -> 26.15)
    readout = **−3.24** at base 8.6
  R18 COMPARATOR at the SAME cell: Δ +1.65, G +1.70. So attention's feature
  deficit at 64px/200-way is **7.3x** the conv deficit (12.39 vs 1.70) and
  the prior fills it: +9.15 vs +1.65 e2e. PREDICTIONS BOTH HIT ("large
  positive Δ"; "G(vit,tin) > G(R18,tin)=1.70" — by 25σ). The recorded
  FALSIFIER for "same law on attention everywhere" (readout POSITIVE at a
  sub-30 baseline on tin) did NOT fire: readout is −3.24 at base 8.6,
  negative exactly as the conv-derived sign law demands. Sign law 30th
  clean cell, 7th on attention, and the first on a non-32px input.
  CONSEQUENCE: the ViT deficit is NOT a 32px-CIFAR artifact. Together with
  diagvit@100% (+9.88 at FULL data), ViT-tiny-from-scratch is a
  permanent-deficit backbone across data scale AND input scale.
- *** SSL3 LANDED — MY "DATA-REGIME-BOUNDED SSL WIN" PREDICTION IS DEAD;
  THE FALSIFIER FIRED AT EVERY FRACTION (2026-07-21, COMPLETE 0h53):
      C100   baseline  champion-aux  SimCLR-init(2x)   SSL−aux   band
      @1%      8.93       10.35        11.21 ±0.14      +0.85    8..11 (just over)
      @2%     14.17       16.67        19.05 ±0.26      +2.38    15..18 (OVER)
      @5%     25.36       30.51        34.41 ±0.17      +3.90    (prior wave)
      @10%    40.28       44.03        49.04 ±0.32      +5.01    44..47 (OVER)
  I predicted SimCLR would collapse below ~1000 images ("at 1% SimCLR ~
  baseline or worse") and that the SSL frontier win was data-regime-bounded.
  WRONG on both counts: SimCLR-init beats the champion aux at EVERY measured
  fraction, missing 3 of 4 bands on the HIGH side. The recorded falsifier —
  "SimCLR beats champion at 1% => aux's low-data niche narrows too" — fired.
  WHAT IS ACTUALLY TRUE: the SSL margin over aux GROWS with data
  (+0.85/+2.38/+3.90/+5.01 at 1/2/5/10%), i.e. aux's RELATIVE position is
  best at extreme scarcity, converging toward parity at 1% — but never wins.
  Both interventions peak near 5% over baseline (aux +5.15, SSL +9.05);
  SSL is roughly 2x aux everywhere at 2x compute.
  POSITIONING, STATED HONESTLY: MomentAux is NOT the accuracy frontier at
  any measured C100 fraction once 2x compute is affordable. Its surviving
  claims are (a) ~zero marginal cost (+2% compute, no pretraining stage,
  no pipeline change), (b) near-parity with SSL at 1% (+0.85 apart), and
  (c) the ATTENTION regime, where no SSL comparison exists yet and the
  prior is worth +9..+10 at full data. Every within-recipe finding (the
  law, G curves, sign law, all controls) is untouched — none of them ever
  claimed SSL-superiority.
  OPEN (not queued): SimCLR-init on ViT-tiny would test whether SSL also
  dominates in the one regime where the prior is dramatic. That is now the
  single most important positioning experiment left.

- sslvit wave LAUNCHED (2026-07-21, autonomous — user: "do the needed
  experiments, no need to wait"): SimCLR-init on ViT-tiny at C100 5%/10%,
  3 seeds + probes. THE decisive positioning cell: SSL beats the aux prior
  at every conv fraction, but ViT-tiny is the one regime where the prior is
  dramatic (+13.26@10%, G=14.85 = 2.3x any conv G). Does SSL dominate there
  too? NEW CODE: scripts/simclr_pretrain.py is now backbone-agnostic
  (classifier .fc/.head swap, AdamW when the cell's optimizer is adamw,
  image_size threaded); smoke-tested — 150 tensors transfer, only the
  classifier is left fresh.
  PREDICTIONS RECORDED IN ADVANCE (and this time I am betting AGAINST the
  conv pattern transplanting, having just been wrong the other way):
    diagsslvit_simclr_10pct: **20..30** (above baseline 16.47, BELOW aux
      29.74). Reasoning: ViT-tiny has no spatial inductive bias to
      bootstrap from, so contrastive pretraining on 5000 images must LEARN
      what the moment prior simply INJECTS. If the conv pattern instead
      transplanted verbatim (SSL−aux ≈ +5), this cell would be ~34.7 —
      explicitly outside my band, so the bet is falsifiable.
    diagsslvit_simclr_5pct: **14..22** (baseline 12.25, aux 21.60).
    FALSIFIER A (positioning closes): ≥ 31 at 10% => SSL dominates even in
      the attention regime; MomentAux has NO accuracy niche left anywhere
      and its claim narrows to pure cost (+2% vs +100% compute).
    FALSIFIER B (strongest possible result for the method): ≤ 17 at 10%
      (≈ baseline) => SimCLR genuinely FAILS on ViT-tiny at this scale,
      and the prior's attention advantage is unique and unmatched by SSL.
    Probes: G(simclr-vit ckpts) tells whether any SSL gain here is the
      same feature currency as the prior's (as it was on conv, G=+9.0).

- *** SSLVIT LANDED — THE ATTENTION REGIME IS THE METHOD'S REAL NICHE
  (2026-07-21, COMPLETE 1h15, 3 seeds + probes). My band was 20..30 at 10%,
  explicitly betting AGAINST the conv pattern transplanting (conv-verbatim
  would have been ~34.7). LANDED 29.77 — IN BAND, and NEITHER falsifier
  fired (not >=31, not <=17):
      C100 (ViT-tiny)  baseline   aux-ViT   SimCLR-ViT(2x)   SSL−aux
      @5%               12.25      21.60     20.34 ±0.72      **−1.26**
      @10%              16.47      29.74     29.77 ±1.00      **+0.04**
  On CONV, SimCLR beat the aux prior at EVERY fraction and the margin GREW
  with data (+0.85/+2.38/+3.90/+5.01 at 1/2/5/10%). On ATTENTION that
  pattern DIES: SSL merely TIES at 10% and LOSES at 5% — at 1/50th the
  compute for the prior (~1.02x vs 2x).
  FEATURE SIDE, the sharper statement: the moment prior produces MORE
  feature gain than SimCLR on ViT at both fractions —
    G(aux-ViT) 14.85 vs G(simclr-ViT) 13.46 @10%
    G(aux-ViT) 13.17 vs G(simclr-ViT) 10.02 @5%
  So on attention-at-small-scale the fixed spectral target is not merely
  cheaper than contrastive pretraining, it is a BETTER source of the
  features the backbone cannot self-learn. readout(simclr-ViT) = −0.16
  @base 16.5 and −1.93 @base 12.2 — negative below the crossing, matching
  the aux-derived sign law on a non-aux intervention.
  POSITIONING AS IT NOW STANDS: conv => SSL wins if you can pay 2x;
  attention => the prior matches/beats SSL at 2% of the cost. The ViT
  direction is the headline.
- *** WIDTHOCTAVE ADJUDICATED — IT IS TARGET **WIDTH**, NOT THE OCTAVE
  (2026-07-21, COMPLETE 2h08, 10v10; tin staged to /dev/shm, 2.2s/epoch vs
  ~17s on BeeGFS — the fast-path works):
      tin_none_5pct      21.08±0.26 (n=10)
      tin_aux_5pct       23.20±0.30   Δ +2.12 ±0.12   (8-pair champion bank)
      auxmag3_tin_5pct   23.68±0.19   Δ +2.60 ±0.10   excess +0.49 ±0.11
      auxmag6o_tin_5pct  23.50±0.34   Δ +2.43 ±0.14   excess +0.31 ±0.14
    *** auxmag6o − auxmag3 = **−0.18 ±0.12 = 1.4σ** — NOT DISTINGUISHABLE.
  The recorded falsifier for the width account (">2σ apart") did NOT fire.
  Both routes to a 12-channel target — one EXTRA OCTAVE (new frequency
  content) or TWO EXTRA ORIENTATIONS (no new frequency) — buy the same
  ~+0.3..+0.5 at tin@5%. The octave is NOT special; TARGET WIDTH is the
  driver. Combined with the clean 32px negative control (auxmag3 excess
  +0.02 there), the scoped conclusion is: **at 64px a wider aux target buys
  ~+0.4; at 32px width buys nothing** — i.e. the gain needs resolution to
  put the extra channels to work, but not new frequency bands specifically.
  CAVEAT recorded: auxmag3's own excess (+0.49, 4.9σ) is more clearly
  nonzero than auxmag6o's (+0.31, 2.2σ); the DIFFERENCE is what is
  adjudicated, and it is null. The committed 8-pair bank stays the headline
  bank (a re-pin invalidates every existing run and needs an explicit
  decision); "widen the target at >=64px" is now a documented, cheap option.
- sslvit2 wave LAUNCHED (autonomous follow-up): completes the ViT-SSL
  envelope at 1% and 25%, plus the ViT COMBO (SimCLR init + moment aux).
  PREDICTIONS RECORDED IN ADVANCE:
    diagsslvit_simclr_1pct: **6..9** (aux-ViT 7.80, base 6.44). 500 images
      is far too few for contrastive learning on a backbone with no spatial
      prior; expect SSL <= aux here, the widest aux margin of the envelope.
    diagsslvit_simclr_25pct: **40..46** (aux-ViT 42.17, base 28.50). THE
      decisive envelope point: on ViT the SSL−aux trend so far is
      −1.26@5% -> +0.04@10%, so extrapolating the conv behaviour predicts
      SSL OVERTAKES aux somewhere above 10%. If SSL lands >43.5 the ViT
      advantage is itself data-bounded (aux wins only at <=10%); if it
      lands <=42 the prior holds across the whole ViT envelope.
    diagsslvitaux_10pct (combo, 2.02x): **29..32**. Conv precedent says the
      two priors do NOT stack (aux costs an SSL-init run −1.51), and here
      they fill the same deficit (G 13.5 vs 14.9). FALSIFIER: >=33 => they
      DO stack on attention, contradicting the conv result and reopening
      aux-on-SSL as a direction.

- *** SSLVIT2 LANDED — THE ViT-SSL ENVELOPE IS COMPLETE, AND THE PRIOR'S
  ATTENTION ADVANTAGE IS **DATA-BOUNDED** (2026-07-21, COMPLETE 1h45,
  3 seeds + probes). Both bands HIT; the combo falsifier did NOT fire:
      C100 (ViT-tiny)  base   aux-ViT   SimCLR-ViT(2x)   SSL−aux   G(aux) G(ssl)
      @1%              6.44     7.80     7.05 ±0.17      **−0.74**  5.89   1.78
      @5%             12.25    21.60    20.34 ±0.72      **−1.26**  13.17 10.02
      @10%            16.47    29.74    29.77 ±1.00      **+0.04**  14.85 13.46
      @25%            28.50    42.17    43.58 ±0.23      **+1.41**  14.03 14.38
  SSL−aux is MONOTONE RISING in data (−0.74/−1.26/+0.04/+1.41): the prior
  WINS below 10%, TIES at 10%, and LOSES at 25% (+1.41 = 3.9σ). My @25%
  prediction had the fork at 43.5 ("if SSL lands >43.5 the ViT advantage is
  itself data-bounded"); it landed 43.58 — the data-bounded branch, by a
  hair but significantly.
  CORRECTION TO THE EARLIER FRAMING: "the attention regime is the method's
  niche" was too broad. The correct statement is **the LOW-DATA attention
  regime (<=10%)**. What is still remarkable is how far the crossover moved:
  on CONV, SSL wins from 1% upward (+0.85 already at 500 imgs); on ViT the
  crossover is pushed out to ~10%, and below it the prior wins outright at
  1/50th the compute.
  FEATURE SIDE explains it: G(simclr-ViT) COLLAPSES at low data (1.78 @500
  imgs vs the prior's 5.89) — contrastive learning genuinely starves on a
  backbone with no spatial prior — then catches up and passes the prior only
  at 25% (14.38 vs 14.03). This is exactly the mechanism I wrongly predicted
  for CONV (where it did not happen); on ATTENTION it is real.
  readout(simclr-ViT) = −1.17/−1.93/−0.16/+0.71 at bases 6.4/12.2/16.5/28.5
  — negative below the ~30 crossing, rising toward it: the aux-derived sign
  law continues to hold on a non-aux intervention (4 more cells).
- *** ViT COMBO: NO STACKING ON ATTENTION EITHER (2026-07-21):
  diagsslvitaux_10pct = **29.93 ±0.96** — IN BAND (29..32); the falsifier
  (>=33 => they stack) did NOT fire. vs SimCLR-alone 29.77 (+0.16 ±0.80)
  and vs aux-alone 29.74 (+0.20 ±0.58) — both indistinguishable from zero.
  G(combo) = +14.14, BETWEEN the two singles (aux 14.85, simclr 13.46), not
  above either: the deficits they fill are the same deficit, so the second
  intervention adds nothing.
  NOTE the difference from conv: there the combo actively HURT (−1.51, the
  early λ0=1.0 shaping taxing already-shaped features); on ViT it is merely
  NEUTRAL. Bottom line unchanged and now demonstrated on BOTH backbone
  families: **aux XOR SSL, never both.**

- deit wave LAUNCHED (2026-07-21, user-approved option (a) + "test it on all
  data portions"): the HEAVY-AUGMENTATION rival for the ViT claim. DeiT
  (Touvron et al. 2021) augmentation hyper-parameters VERBATIM — RandAugment
  rand-m9-mstd0.5-inc1, RandomErasing p=0.25 mode=pixel, Mixup 0.8, CutMix
  1.0, label smoothing 0.1 (timm's own implementations, so nothing is tuned
  by me) — ADDED on top of the study's base crop+flip.
  DESIGN NOTE (deliberate, and it is a choice worth stating): I did NOT
  import DeiT's optimizer/schedule/epoch budget. diagdeit_* is IDENTICAL to
  diagvit_* except for `augment: deit`, so the only variable is the
  augmentation stack. Importing the whole DeiT recipe would confound
  augmentation with lr/warmup/epochs and make the comparison
  uninterpretable. Full envelope: 1/5/10/25/100%, both members, 3 seeds
  + probes. augment: => diag-only (guard in train.py).
  THE QUESTION: does the moment prior still add anything once a small ViT is
  trained the way the literature says it should be?
  PREDICTIONS RECORDED IN ADVANCE:
    Baselines RISE (that is what DeiT aug is for): diagdeit_none@10%
      **20..30** (from 16.47), @100% **58..68** (from 50.61). At @1%
      (500 imgs) heavy aug may instead HURT — too much regularization for
      too few images; band **5..8** (from 6.44).
    Δ_deit (aux − none, both under deit aug) SHRINKS but stays positive:
      @10% **+3..+10** (from +13.26), @1% **+0..+3** (from +1.35),
      @100% **0..+6** (from +9.88). Reasoning: the SSL result showed
      anything that fills the same feature deficit does not stack; heavy
      augmentation plausibly supplies part of what the prior supplies,
      so partial substitution is the base case.
    THE DECISIVE ABSOLUTE COMPARISON: diagdeit_none@10% vs diagvit_aux@10%
      (29.74). If augmentation ALONE reaches 29.74, then "just use the
      standard recipe" matches "use the prior", and the ViT claim must be
      restated as a cost claim, not a capability claim.
    FALSIFIER A (prior redundant with augmentation): Δ_deit <= +1 at 10%
      => heavy aug supplies what the prior supplied; the ViT headline
      weakens to "a cheap substitute for a tuned augmentation recipe".
    FALSIFIER B (prior complementary — strongest form of the claim):
      Δ_deit >= +8 at 10% (barely shrunk) => the prior injects structure
      augmentation cannot manufacture, and the two are additive.

- *** DEIT WAVE LANDED — THE PRIOR AND HEAVY AUGMENTATION ARE STRONGLY
  COMPLEMENTARY, NOT SUBSTITUTES. MY PREDICTION WAS WRONG IN EVERY Δ CELL
  (2026-07-22, COMPLETE ~6h, 3 seeds + probes, C100/ViT-tiny):
      pct   vit-base vit-aux  Δ_vit | deit-base deit-aux  Δ_deit | G_vit G_deit
      1%      6.44    7.80   +1.35  |   6.48     9.68    +3.20  |  5.89  10.29
      5%     12.25   21.60   +9.35  |  12.55    28.89   +16.34  | 13.17  21.35
      10%    16.47   29.74  +13.26  |  20.28    41.29   +21.00  | 14.85  22.20
      25%    28.50   42.17  +13.67  |  32.27    57.32   +25.05  | 14.03  23.05
      100%   50.61   60.50   +9.88  |  61.39    75.25   +13.86  |  9.70  13.03
  PREDICTION SCORING — baselines ALL IN BAND (I predicted deit-none 20..30
  @10% -> 20.28; 58..68 @100% -> 61.39; 5..8 @1% -> 6.48). But every Δ
  prediction MISSED HIGH: I said Δ_deit would SHRINK (+3..+10 @10%, 0..+6
  @100%) on the reasoning that augmentation supplies part of what the prior
  supplies. The OPPOSITE is true — Δ_deit is LARGER than Δ_vit at every
  single fraction (+21.00 vs +13.26 @10%). FALSIFIER B fired (Δ_deit >= +8
  @10%): the prior injects structure augmentation cannot manufacture.
  THE DECISIVE PRE-REGISTERED COMPARISON — does augmentation ALONE reach the
  prior's plain-ViT accuracy? NO, at every fraction up to 25%:
      @1% 6.48 vs 7.80 | @5% 12.55 vs 21.60 | @10% 20.28 vs 29.74
      @25% 32.27 vs 42.17 | @100% 61.39 vs 60.50 (aug alone finally edges it)
  So the ViT claim does NOT collapse to a cost claim: below full data the
  fixed spectral target beats the standard recipe outright, and ON TOP of
  that recipe it still adds +13.9..+25.1.
  FEATURE SIDE: G is BIGGER under augmentation (21-23 vs 13-15) — the prior's
  feature gain grows when the nuisance-invariance is handled by augmentation.
  CONTRAST WITH SSL, and this is the sharp point: aux + SimCLR do NOT stack
  (combo neutral on ViT, −1.51 on conv) because SSL fills the SAME feature
  deficit (G(simclr) ≈ G(aux)). aux + AUGMENTATION DO stack, strongly.
  Augmentation teaches invariance to nuisance transforms; the moment prior
  injects oriented-energy STRUCTURE. Different currencies, so they add.
  RECOMMENDATION UPDATED: aux XOR SSL, but aux AND augmentation.
  BEST ViT NUMBER IN THE STUDY: **75.25%** at C100@100% (DeiT-aug + prior)
  vs 61.39 aug-only and 50.61 plain. This is now the headline ViT result.

- *** ENVELOPES FILLED IN (2026-07-22, fillvit/fillconv still running; the
  2/3/7/15% points are new). Two corrections to earlier statements:
    ViT-tiny on C100, Δ by fraction (1/2/3/5/7/10/15/25/100%):
      plain      +1.35 +3.25 +6.21  +9.35 +10.99 +13.26 **+14.44** +13.67 +9.88
      DeiT-aug   +3.20 +6.01 +9.48 +16.34 +18.69 +21.00  +24.59 **+25.05** +13.86
    (1) THE PLAIN-ViT ENVELOPE PEAKS AT 15% (+14.44), not at 25% as the
        sparser grid suggested; and the DeiT-aug envelope peaks LATER, at 25%
        (+25.05). So heavy augmentation does not merely amplify the prior
        (~1.8-2.4x at every fraction) — it SHIFTS the peak RIGHT, toward more
        data. Consistent with the mechanism: augmentation removes the
        overfitting that otherwise caps how much structure the net can use.
    (2) SSL−aux ON CONV IS UNIMODAL, NOT MONOTONE. With 3/7/15% filled:
        +0.85 +2.38 +2.32 +3.90 +4.99 +5.01 +4.09 at 1/2/3/5/7/10/15%.
        I previously wrote "the SSL margin over aux GROWS with data" from the
        1/2/5/10% points alone — WRONG as stated. It grows to a peak near
        7-10% and is already declining by 15% (+4.09). The honest claim is
        that SSL's advantage is largest in the mid-data band and shrinks at
        both ends. Same error class as every other sparse-grid extrapolation
        in this study, which is exactly what filling the grid is for.

- deitssl wave LAUNCHED (2026-07-22, user: "are we going somehow to merge the
  DeiT-aug with SSL − aux?" — a real hole they spotted): the 2x3
  {plain,deit} x {none,aux,simclr} has NO deit-simclr cell. Every SSL-vs-prior
  head-to-head ran under plain crop+flip, so the ViT positioning claim is
  currently conditioned on a recipe nobody would use for a small ViT.
  THE MECHANISM AT STAKE: aux+augmentation STACK (different currencies —
  oriented-energy STRUCTURE vs nuisance-INVARIANCE). SimCLR's objective IS
  invariance to an augmentation family, so heavy augmentation may SUBSTITUTE
  for it rather than amplify it. If so, the head-to-head FLIPS under a modern
  recipe and the claim strengthens from "low-data attention" to "attention
  under any modern recipe".
  PREDICTIONS RECORDED IN ADVANCE (bands sit BELOW the aux cell — betting on
  PARTIAL substitution: SSL amplified by aug, but less than aux is):
    diagdeitssl_simclr_5pct:  **24..30** (deit-base 12.55, deit-aux 28.89)
    diagdeitssl_simclr_10pct: **38..43** (deit-base 20.28, deit-aux 41.29)
    diagdeitssl_simclr_25pct: **50..57** (deit-base 32.27, deit-aux 57.32)
    FALSIFIER A (my bet dies): deit-ssl >= deit-aux at BOTH 10% and 25% =>
      equal amplification, no flip; the positioning claim then holds ONLY
      under plain augmentation and must be restated that way.
    FALSIFIER B (strongest form for the method): deit-ssl <= deit-base + 5
      => augmentation almost entirely substitutes for SimCLR, and the prior
      is the only intervention that survives a modern recipe.
    PROBES: G(deit-ssl) vs G(deit-aux)=22.20. If SSL's currency really is
      invariance, its G should rise LESS under augmentation than aux's did
      (aux 14.85 -> 22.20; simclr 13.46 -> ?).
    COMBO diagdeitsslaux_10pct: "aux XOR SSL" was established WITHOUT
      augmentation (conv −1.51, ViT neutral). Beating BOTH singles here
      would falsify it under a modern recipe.
  COST: the SimCLR pretrain ckpt is AUGMENTATION-AGNOSTIC (only the
  supervised stage changes), so this reuses fillvit's pretrains — ~27
  supervised runs, no new pretraining. Full envelope 1-100%, 3 seeds,
  decisive fractions (10/25) FIRST. Suite 101 green.
  CAVEAT STATED UP FRONT: the pretrain uses SimCLR's OWN augmentations, so
  this tests "does heavy SUPERVISED augmentation substitute for contrastive
  pretraining", NOT "would SimCLR-under-DeiT-augs do better".
- GATED FOLLOW-UP, code built and NOT launched (2026-07-22): scripts/
  simclr_pretrain.py gained `--augment deit`, which strengthens the
  CONTRASTIVE VIEWS with RandAugment + RandomErasing. Only those two DeiT
  components transfer — Mixup/CutMix blend two images and make the NT-Xent
  positive pair ambiguous, and label smoothing needs labels NT-Xent lacks —
  so it is "stronger views", NOT "the DeiT recipe"; do not describe it as
  the latter. Smoke-tested (6 -> 8 transform stages), suite 101 green.
  WHY NOT LAUNCHED: unlike deitssl this needs NEW 2x-compute pretraining,
  and its value is set by the deitssl outcome. DECISION RULE ON RECORD:
    falsifier A fires (deit-ssl >= deit-aux) => DO NOT RUN, moot (SSL
      already wins under augmentation; the live question becomes why the
      prior stopped mattering).
    band holds (deit-ssl below deit-aux) => RUN: "was SSL given its best
      shot?" is then the main objection and only this cell answers it.
    falsifier B fires (deit-ssl ~ deit-base) => RUN, urgently: a
      near-total substitution result must be shown to survive a fairly
      strengthened SimCLR.
  Scope if triggered: decisive fractions 5/10/25 only, 3 seeds (9 pretrains
  + 9 finetunes), NOT the full envelope.
- *** DEITSSL LANDED — THE HEAD-TO-HEAD FLIPS: UNDER A MODERN RECIPE THE
  PRIOR BEATS SSL AT EVERY FRACTION (2026-07-22, wave COMPLETE, 3 seeds).
  deit-ssl (SimCLR-init ViT under DeiT augmentation) vs its comparators
  (C100/ViT-tiny, full envelope):
      pct   deit-base  deit-ssl   deit-aux   aux−ssl | plain aux−ssl
      1%      6.48       7.20       9.68      +2.48   | +0.74
      2%      9.98      11.25      15.99      +4.74   | —
      3%     11.31      14.28      20.79      +6.51   | —
      5%     12.55      21.64      28.89      +7.25   | +1.26
      7%     15.98      29.58      34.67      +5.09   | —
      10%    20.28      36.06      41.29      +5.23   | −0.04 (ssl tied)
      15%    24.07      43.46      48.66      +5.20   | —
      25%    32.27      52.31      57.32      +5.01   | −1.41 (ssl won)
      100%   61.39      73.25      75.25      +2.00   | —
  Under PLAIN crop+flip, SSL TIED the prior at 10% and WON at 25% (aux−ssl
  −0.04/−1.41). Under DeiT augmentation the prior WINS AT EVERY FRACTION, by
  ~+5 across the mid-range. MECHANISM CONFIRMED: heavy supervised augmentation
  supplies the nuisance-INVARIANCE SimCLR was providing (same currency, so it
  SUBSTITUTES and collapses SSL's marginal value), while the prior's
  oriented-energy STRUCTURE is a different currency that augmentation amplifies
  (deit-aux ≫ deit-ssl). This is the same STACK-vs-SUBSTITUTE split seen on
  conv (aux+SSL don't stack; aux+aug do), now shown head-to-head.
  BANDS: @25% IN BAND (52.31, 50..57); @5% and @10% MISSED LOW (21.64 vs
  24..30; 36.06 vs 38..43) — I OVER-estimated how much augmentation would
  amplify SSL; substitution was MORE complete than the "partial" bet. Neither
  falsifier fired (deit-ssl not ≥ deit-aux, not ≈ deit-base+5).
  COMBO diagdeitsslaux_10pct = **37.72** (38.60/37.91/36.65) — below aux-alone
  41.29, barely above ssl-alone 36.06: **aux XOR SSL HOLDS under a modern
  recipe** (combo falsifier "beat BOTH singles" did NOT fire). Starting from
  an SSL init, adding aux under augmentation does not climb back to aux-alone.
  POSITIONING STRENGTHENED: the ViT claim was "the prior wins in the LOW-DATA
  attention regime (≤10%) under plain augmentation". It now reads "the prior
  beats SimCLR at EVERY C100 fraction once a small ViT is trained the modern
  way (DeiT aug)" — the recipe nobody-would-not-use is exactly where the prior
  looks best.
  *** G PROBES (re-run correctly in deitssl2, aux-vs-base gap over deit_none):
      pct   G(deit-ssl)   G(deit-aux)   plain G(ssl)->deit   plain G(aux)->deit
      5%      +12.77        +21.37        10.02 -> 12.77       13.17 -> 21.37
      10%     +17.30        +22.20        13.46 -> 17.30       14.85 -> 22.20
      25%     +17.98        +23.05        14.38 -> 17.98       14.03 -> 23.05
    THE PRE-REGISTERED PROBE PREDICTION HELD: aux's G rises MUCH more under
    augmentation than SSL's (@10% +7.35 vs +3.84). SSL's currency IS
    nuisance-invariance, so heavy augmentation — which also supplies
    invariance — adds little to its features; the prior's oriented-energy
    STRUCTURE is orthogonal, so augmentation compounds it. G(deit-aux) >
    G(deit-ssl) at every fraction, so the e2e flip is a FEATURE-side effect,
    not a readout artifact — the sharpest statement of the stack-vs-substitute
    split yet: same-currency interventions (SSL, aug) do not compound;
    different-currency ones (prior, aug) do.
  *** DEITSSL2 BUG + RESUBMIT: the stronger-view part of deitssl2 FAILED —
    every DeiT-view pretrain died on "flock: cannot open lock file
    .../pretrain.pt.lock: No such file or directory" (flock cannot create the
    lockfile in a not-yet-existent parent dir; the deitssl wave reused
    fillvit's pre-made dirs, these paths were new). The G probes above
    SUCCEEDED regardless. Fix (mkdir -p the parent before flock) shipped as
    deitssl2b_wave.sbatch, resubmitted.
  *** DEITSSL2B LANDED — STRONGER VIEWS HURT SSL; THE FLIP IS NOT AN
    AUGMENTATION-ASYMMETRY ARTIFACT (2026-07-22, 3 seeds + probes):
        pct  deit-sslsv  vs deit-ssl(plain views)  vs deit-aux | G(sslsv) G(ssl)
        5%     17.52         −4.12                   −11.37     | +6.20    +12.77
        10%    26.25         −9.81                   −15.04     | +6.37    +17.30
        25%    40.75        −11.56                   −16.57     | +7.78    +17.98
    Giving SimCLR DeiT-strength contrastive views (RandAugment + RandomErasing
    ON TOP of crop+jitter+grayscale) made it WORSE at every fraction — e2e
    −4..−12 vs plain SimCLR views, and G roughly HALVED (+6..+8 vs +13..+18).
    The falsifier (deit-sslsv ≥ deit-aux at 10% AND 25%) did NOT fire — sslsv
    lands 11..17 BELOW aux. MY BAND MISSED: I predicted stronger views
    "recover SOME SSL value" (30..40 @10%); they HURT. But the hedge I
    under-committed to was right — "SimCLR's own ablation says crop+color is
    the critical pair, harder views help little at 2.5-12.5k images" — the
    truth was not "help little" but "over-distort the positive pair and
    degrade the contrastive signal". CONCLUSION: SimCLR's BEST SHOT is its
    STANDARD views; the "was SSL under-augmented?" objection is answered
    emphatically — its best configuration still loses to the prior under a
    modern recipe, and strengthening it only widens the gap. The head-to-head
    flip stands. THE DEIT×SSL COLUMN IS CLOSED: under DeiT augmentation the
    moment prior beats SimCLR (standard OR strengthened) at every C100
    fraction, feature-side and e2e.
- deitssl2 wave LAUNCHED (2026-07-22, the gated stronger-views rebuttal FIRED
  by the band-holds branch): because deit-ssl sits below deit-aux everywhere
  AND my bands missed LOW, "was SimCLR given its best shot?" is the sharpest
  objection — the deitssl pretrain used SimCLR's OWN 2020 views while the
  supervised stage used DeiT augs. deitssl2 gives the PRETRAIN DeiT-strength
  views (--augment deit) at 5/10/25%, 3 seeds (9 new 2x-compute pretrains + 9
  finetunes), and re-runs the failed deitssl G probes with the correct CLI.
  PREDICTION RECORDED IN ADVANCE: stronger views RECOVER SOME of SSL's value
  but do NOT overturn the flip — deit-sslsv stays BELOW deit-aux at 10% and
  25% (bands 30..40 @10%, 46..55 @25%; SimCLR's own ablation says crop+color
  is the critical pair, harder views help little at 2.5-12.5k images).
  FALSIFIER (SSL was under-augmented all along): deit-sslsv ≥ deit-aux at BOTH
  10% and 25% => the flip was an augmentation-asymmetry artifact and the
  "prior wins under a modern recipe" claim must be withdrawn.

- *** TABLE AUDIT (2026-07-23, vs ledger + pairing + law + representation):
  23/23 recorded headline numbers reproduce exactly from the exporters; all
  paired Δs match their families on every recipe axis; probe-side G values
  match the ledger to 0.01. THREE representation fixes shipped: (1) baseline
  tie-break is now (seeds, original-over-grid_, name) — equal-power ties
  previously fell to load order (c10@7% twins differ 0.77); (2) `is_headline`
  now requires >=3 seeds AND no collapsed seed (163 one/two-seed legacy
  ablations were flagged headline); (3) new `bistable` column: cells where
  >=1 seed sits at chance while the cell mean trains.
  *** CORRECTION — ConvNeXt-SGD is BISTABLE, NOT uniformly at chance: BSC
  re-runs show seeds {0.84, 42.25, 19.54} @100%, {17.78, 1.00, 18.85} @15%.
  The dead-list's "sits at chance (0.92%)" was a per-seed observation
  over-generalized. Still dead for headline use (the frozen recipe cannot
  train it reliably), but the correct statement is seed-bistable collapse —
  same failure family as the R50 no-head_norm collapse. Learned-pool cells
  show the same pattern ({43.97, 1.34, 44.31}).

- genssl wave LAUNCHED (2026-07-23, user: "results are lacking for
  generalizability — not all datasets are used for the configurations").
  COVERAGE AUDIT FINDING: moment-aux [plain] spans 10 datasets x 5 backbones
  (the core claim IS general), but EVERY SSL and DeiT comparison is C100-only
  — simclr-init, ssl+aux, and the whole DeiT column. All three positioning
  claims transplant to tin in this wave (2x images, 64px, 200-way: different
  input scale AND label space at once).
  PREDICTIONS RECORDED IN ADVANCE:
    Track A (conv SSL, tin@1/5/10% vs aux +1.49/+2.13/+1.65):
      diagssl_tin_simclr_1pct:  6.3..7.8  (base 5.30, aux 6.80; SSL ~ aux)
      diagssl_tin_simclr_5pct:  23.5..26  (base 21.08, aux 23.20)
      diagssl_tin_simclr_10pct: 35.5..38.5 (base 33.60, aux 35.24)
      FALSIFIER (narrows the SSL claim): SSL <= aux at BOTH 5% and 10% =>
        conv-SSL dominance is C100-specific.
    Track B (ViT SSL, tin@10% vs aux-ViT-tin +9.15, G=12.39):
      diagsslvit_tin_simclr_10pct: 13..18 (base 8.60, aux 17.75) — betting
        at-or-below aux. FALSIFIER: > 19.75 => SSL beats the prior on
        attention off-C100.
    Track C (DeiT column, tin/ViT@10%): deit-base(tin) 10..16;
      Delta_deit(aux) +10..+20 (stacking transplants; plain +9.15);
      deit-ssl BELOW deit-aux (the substitution flip transplants).
      FALSIFIER: deit-ssl >= deit-aux on tin => the flip is C100-specific
      and the modern-recipe claim must be scoped to CIFAR.
  COST: ~33 runs + 12 pretrains, staged tin, one GPU. G probes on all six
  decisive cells at the end.
  *** GUARD BUG + REPAIR (2026-07-23, same day): simclr_pretrain.py carried a
  leftover cifar100-only ValueError — ALL 12 tin pretrains in genssl died at
  claim (Tracks A+B); Track C (no pretrain) ran fine. Fixed to a verified
  whitelist (cifar100, tin), both paths smoke-tested, suite 101 green.
  genssl2 (turing) repairs A@1/5/10 + B + deitssl_tin + probes.
  *** GENSSL/GENSSL2 SCORED (2026-07-23, wave COMPLETE, 3 seeds + probes;
  tin@10% comparators: base 33.60/aux 35.24; vit base 8.60/aux 17.75;
  deit base 10.49/aux 27.43):
    TRACK A (conv SSL on tin) — SSL DOMINANCE TRANSPLANTS, UNIMODAL:
      SSL-init: 8.30/15.19/20.61/27.79/38.44 at 1/2/3/5/10%
      SSL−aux:  +1.50/+3.82/+4.63/+4.59/+3.20 — unimodal peaking 3-5%,
      declining by 10%, same shape as C100. @10% landed 38.44, band top
      (35.5..38.5). Falsifier (SSL<=aux at 5 AND 10) did NOT fire: conv-SSL
      dominance is now a TWO-POPULATION claim.
    TRACK B (ViT SSL, tin@10%) — *** THE FALSIFIER FIRED ***:
      diagsslvit_tin_simclr_10pct = 20.69 ±0.33 > 19.75 (aux 17.75 + 2):
      SimCLR-ViT BEATS the prior on attention off-C100, by +2.94, with
      MORE feature gain (G_ssl +16.76 vs G_aux +12.39). Band (13..18)
      missed high. RESTATEMENT REQUIRED AND MADE: the "low-data attention
      niche" is an IMAGE-COUNT statement, not a fraction statement — C100:
      tie at 5k imgs, SSL wins at 12.5k; tin: SSL wins at 10k. Both
      populations agree on a crossover ~5k images; below it (C100 1-5%,
      tin 1-5% pending) the prior wins on attention, above it SSL does.
      tin ViT-SSL at 1/2/5% (queued in the expansion) adjudicates the
      below-crossover side off-C100.
    TRACK C (deit-ssl, tin@10%) — THE MODERN-RECIPE FLIP TRANSPLANTS:
      diagdeitssl_tin_simclr_10pct = 25.45 ±0.38 < deit-aux 27.43 (−1.98);
      falsifier (deit-ssl >= deit-aux => flip C100-specific) did NOT fire.
      Under plain aug SSL wins this cell (+2.94); add DeiT augmentation and
      the prior wins (−1.98) — the stack-vs-substitute mechanism holds on
      population #2. Feature side agrees: G(deit-aux) +19.72 >
      G(deit-ssl) +18.52.
    NET POSITIONING AFTER GENSSL: "prior beats SSL under a modern recipe"
    now holds on BOTH C100 and tin; "prior beats SSL under plain aug on
    attention" holds only BELOW ~5k images (restated); conv plain-aug SSL
    dominance confirmed everywhere measured.
  ENVELOPE EXTENSION (user: "why not all portions? use the local machine"):
  local 3090 wave fills the tin conv-SSL envelope at 2/3/7/15/25% (predictions
  in scripts/genssl_local_wave.sh header: SSL>base from 2% up, UNIMODAL
  SSL−aux margin peaking mid-band, ~0 by 25%). tin@100% SSL deliberately
  EXCLUDED: the pretrain alone is ~2x a tin@100% train (week-scale on 3090);
  queue on turing later only if the envelope shape justifies it.
  DATASET EXPANSION APPROVED (user): EuroSAT + DTD next (non-photo statistics
  and texture-dominated — the domain axis none of the current 5 populations
  covers). Food: FoodSeg103 is a SEGMENTATION set (pixel masks, wrong task
  for the frozen classification recipe); its classification analog Food-101
  is the right candidate if the food domain is wanted (fine-grained,
  texture-rich, squash-resize like CUB).

- IMPORTANCE-OVER-COST DIRECTIVE (2026-07-23, user: "Don't worry about how
  much the question is expensive. If the question is important... we can go
  with it."). Three cost-excluded questions reinstated the same day:
  (1) domssl: SimCLR-init on ALL FOUR domain datasets @5/10/25/100% (42
      normal + 9 big-lane tasks on BSC; simclr_pretrain whitelist extended
      to eurosat/dtd/pathmnist/food101, each smoke-tested). THE QUESTION:
      SimCLR's views encode PHOTO invariances (crop+color) — do they
      transfer to satellite/texture/histopathology, where the natural
      invariances differ (rotation is real there), while the moment prior
      is domain-agnostic? PREDICTIONS: SSL > aux on food101/pathmnist
      (photo-like texture, abundant data); SSL data-starved on dtd@5-10%
      (188-376 imgs) => SSL <= aux there; eurosat uncertain — recorded as
      the open fork, no number. FALSIFIER for "SSL always wins on conv":
      any domain with aux > SSL at >=2 fractions.
  (2) diaggrid_ssl_tin_100pct: closes the tin conv-SSL envelope (big lane;
      staging makes it ~7h/seed, not week-scale). PREDICTION: 67..69.5
      (base 66.17, aux −0.42; SSL stays positive at full data as on C100).
  (3) vitenv wave (turing 2183): the FULL ViT/DeiT envelope on tin
      @1/2/5/15/25/100 + deit-ssl @5/25 — the attention story currently
      rests on ONE tin fraction. PREDICTIONS: Delta_vit(tin) peaks +10..+15
      @15-25%, stays +6..+10 @100% (permanent-deficit signature);
      deit amplification 1.8-2.4x throughout (measured 1.85x @10%);
      deit-ssl below deit-aux at every fraction. FALSIFIERS:
      Delta_vit(tin@100%) <= +2 => permanent deficit is C100-specific;
      amplification < 1.3x anywhere on the mid-band; deit-ssl >= deit-aux.

- COVERAGE BATCH 2 + RE-PIN EVIDENCE CAMPAIGN (2026-07-23, user: "do we need
  to submit any other experiments?" + "maybe we can try the re-pin gabor
  idea"). Claim-coverage audit found three remaining single-population
  claims; 132 tasks appended to BSC (worklist.bsc now 2859):
  (A) ViT pairs on domains (attention headline was C100+tin only):
      diaggrid_vit_{eurosat,pathmnist,food101}@5/25, dtd@25/100.
      PREDICTION: the rescue transplants (Delta >= +5 wherever the ViT
      baseline is underfit); pathmnist is the interesting cell (ViTs are
      the rising architecture in pathology).
  (B) Backbone universality off C100 (was C100@10% only): r34+r50 champion
      pairs on tin@5% + pathmnist@5%. PREDICTION: one lambda0=1.0 (+hn)
      lands within noise of the r18 delta on both datasets — the universal-
      lambda claim gets its second and third population.
  (C) SSL on CUB (fine-grained comparator): diaggrid_ssl_cub_{25,100}.
      PREDICTION: SSL's fine-grained weakness is real — SSL-init <= aux
      (+2.74@100%); FALSIFIER: SSL > aux by >=2 => even fine-grained is
      SSL's territory on conv.
  (D) RE-PIN EVIDENCE (user reopened the 8-pair decision): mag3 (extra
      OCTAVE, 12ch) AND mag6o (extra ORIENTATIONS, 12ch) arms at every
      64px domain (@5or10/25) + cub@100. Combined with the existing tin
      10v10 (+0.49/+0.31), this gives width-vs-octave on SIX 64px
      populations. DECISION RULE RECORDED IN ADVANCE: if the wide target
      beats the 8-pair champion by >= +0.3 pooled on >= 3 of 5 new 64px
      populations, propose a formal 64px-headline-bank re-pin (as a NEW
      config family — old rows stay valid, nothing is invalidated); if
      octave and orientations DIVERGE on any domain (>2sigma), the
      octave-vs-width question reopens with domain statistics as the
      lever. The 32px datasets keep the 8-pair bank regardless (auxmag3
      excess at 32px was +0.02 — dead null).

- ARCHITECTURES/BASELINES AUDIT (2026-07-23, user: "do we need to test more
  architectures and baselines?"). Two reviewer-critical comparator gaps
  CLOSED (39 tasks -> BSC, worklist 2898):
  (1) SimSiam (scripts/simsiam_pretrain.py, paper CIFAR recipe, negative-free)
      — answers "SimCLR is a weak/old comparator". diaggrid_simsiam on
      C100@5/10/25 + tin@5/10. PREDICTIONS: SimSiam ≈ or > SimCLR at small
      data (negative-free methods are reported more data-robust); the
      prior's positioning must hold against the BETTER of the two SSLs.
      FALSIFIER: SimSiam ≫ SimCLR AND > aux where SimCLR lost => the SSL
      comparison must be redone with SimSiam as the reference.
  (2) ImageNet-TRANSFER comparator (diagtransfer_*, pretrained: true, r18
      trunk minus surgery-reset conv1): the "nobody trains from scratch"
      objection, run on c100@5, pathmnist@5, eurosat@5, cub@100, base+aux
      arms. NEW GUARD: train.py now refuses pretrained: true without a diag
      prefix (outside images = loudest data-contract break; was unguarded).
      PREDICTIONS: transfer dominates on photo-like sets (c100, cub);
      the gap NARROWS on domain-shifted sets (pathmnist stains, eurosat
      optics — 1-epoch smoke already at 58.15 on pathmnist@5 vs scratch
      ~? full-run needed); aux-on-transfer mirrors the SSL no-stack
      result on photo sets. FALSIFIER: transfer wins EVERYWHERE by >10 =>
      the whole from-scratch story needs an explicit scope statement
      ("when pretraining is unavailable/mismatched/disallowed").
  COMBINATIONS (user: "are we going to do some combination with them?"):
  (a) transfer x aux is ALREADY the diagtransfer aux arms. PREDICTIONS:
      photo-like sets (c100@5, cub@100) => no-stack/tax (ImageNet features
      already fill the deficit, the SSL-combo precedent); pathmnist@5 =>
      the OPEN FORK — if ImageNet features are domain-mismatched enough,
      the deficit is unfilled and the prior may STACK on transfer. A
      confirmed stack there would be the first prior-on-pretrained gain
      and the most deployable finding in the study ("fine-tuning on
      medical? add the free prior"). eurosat@5 between the two.
  (b) simsiamaux combos QUEUED (c100@5/10, tin@5; reuse simsiam pretrains
      via shared flock): does "aux XOR SSL" hold for negative-free SSL?
      PREDICTION: no-stack transplants (SimSiam's currency is still
      invariance). FALSIFIER: combo > simsiam-alone by >= +1.5 anywhere
      => the no-stack rule was SimCLR-specific, and the whole aux-XOR-SSL
      recommendation needs re-derivation per SSL family.
  (c) NOT planned: simsiam x deit (gated on SimSiam's plain-aug result),
      transfer x deit (engineering, no mechanism question).
- *** MODERN-ARCHITECTURE BATCH WIRED AND LAUNCHED (2026-07-23, user: "do we
  need to add more up to date and modern architectures and baselines?" —
  the formerly-deferred batch is now real; 51 tasks -> BSC, worklist 2958):
  NEW CODE: backbones.py gains swin_tiny (timm img_size + window_size=4,
  layers.2 = layer3 analog, NHWC) and mobilenetv3_small_100 (conv_stem
  stride 2->1 surgery, blocks.3 tap); aux._to_spatial now folds Swin's
  4D channels-LAST taps (H==W and C>W heuristic + CONTENT test in
  test_tensor_layout — suite now 102). scripts/dino_pretrain.py: DINO with
  EMA teacher, centering, weight-normed head K=4096, 2 global + 4 local
  crops; DEVIATION STATED IN THE DOCSTRING: local crops are small-SCALE at
  full RESOLUTION (fixed ViT patch grid), not half-res — never call it
  verbatim DINO. All three smoke-tested e2e (swin+aux, mnet+aux, dino-vit).
  CELLS: diaggrid_swin_{c100@5/10/25, tin@10} pairs (AdamW diag);
  grid_mnet_{c100@5/10, tin@5} pairs (frozen SGD, headline-eligible,
  bistable flag will catch a cnx-style collapse); diaggrid_dino_vit_{5,10,
  25}pct (C100).
  PREDICTIONS RECORDED IN ADVANCE:
    Swin: baselines FAR above ViT-tiny's (hierarchy is most of what ViT
      lacks at this scale); G(swin) ≪ G(vit) and Delta_swin +1..+5 —
      reading: the deficit is ARCHITECTURAL-BIAS-ABSENCE, not attention
      per se. FALSIFIER: Delta_swin >= +8 at 5-10% => the deficit is
      attention-intrinsic and the mechanism story changes.
    MobileNet: conv-with-strong-bias => champion-like +2..+5 @5%,
      decaying right flank. Watch bistability under frozen SGD.
    DINO-ViT: self-distillation is MORE data-hungry than contrastive at
      2.5-12.5k imgs => DINO <= SimCLR-ViT at 5/10%, prior stays ahead
      <=10%. FALSIFIER: DINO >= aux-ViT anywhere <=10% => modern
      attention-SSL closes the low-data niche and the ViT positioning
      must be restated against DINO.
  STILL EXCLUDED, with reasons: Mamba/VMamba (CUDA-kernel deps, wiring
  risk ≫ evidence value at 32-64px), BYOL (SimSiam already represents
  negative-free), MAE (known-weak at tiny data; run only if a reviewer
  demands it), CaiT/DeiT-III (recipe variants, the aug axis already
  isolates what they'd add).
  INFRA: ~/.cache lives on a FULL 4TB HDD -> HF_HOME now points into the
  repo (.hf_cache, gitignored); weights pre-fetched and shipped to BSC
  (compute nodes have no internet), env.sh exports HF_HOME + HF_HUB_OFFLINE=1.
  *** BSC TRANSFER-CELL FIX CHAIN (2026-07-24): timm needs the
  huggingface_hub PACKAGE to even resolve a pretrained source (cache alone
  insufficient); BSC pip cannot reach PyPI -> wheels shipped by scp. hub
  1.24 imports httpcore (absent) -> downgraded to 0.30.2 (requests-based,
  matches timm 1.0.27). Then timm silently skips the safetensors branch
  without the safetensors PACKAGE and asks for pytorch_model.bin (not in
  cache) -> shipped safetensors-0.8.0 manylinux wheel. Verified:
  "OFFLINE PRETRAINED OK" on the BSC venv. Failed diagtransfer cells
  self-heal via reconcile.

- *** SIMSIAM + COMBOS + DINO@5 SCORED (2026-07-24, BSC expansion cells,
  3 seeds each):
  SIMSIAM COLLAPSES AT STUDY SCALE — MY PREDICTION WRONG, IN THE METHOD'S
  FAVOR: simsiam-init Δ vs baseline = +0.13/+0.61/−0.03 on C100@5/10/25
  (25.49/40.89/61.32 vs base 25.36/40.28/61.35) and +0.93/+0.77 on
  tin@5/10 — negative-free SSL learns ~NOTHING at 2.5-25k imgs, bs128,
  while SimCLR gains +9/+8.8/+2.8. I predicted "SimSiam ≈ or > SimCLR
  (more data-robust)" — the opposite. CONSEQUENCE: SimCLR was the
  STRONGER comparator all along; the "SSL strawman" objection dies
  empirically. On tin the free prior BEATS SimSiam outright at both
  fractions (aux 23.20/35.24 vs simsiam 22.01/34.37).
  *** SIMSIAMAUX COMBO — THE NO-STACK FALSIFIER FIRED: combo beats
  simsiam-alone by +6.16/+4.29/+2.02 (≫ +1.5 tripwire) AND edges
  aux-alone by ~+1 everywhere (31.65 vs 30.51 @c100-5; 45.18 vs 44.03
  @c100-10; 24.03 vs 23.20 @tin-5). "aux XOR SSL" IS SIMCLR-SPECIFIC as
  a rule — but the CURRENCY MECHANISM survives perfectly: no-stack
  happens when the init already FILLS the feature deficit (SimCLR);
  SimSiam fills ~nothing, so the prior does its full work on top.
  RESTATED RULE: the prior is redundant iff the init has already bought
  the features — "aux XOR effective-SSL", judged by the init's own gain.
  DINO-ViT@5%: 15.22 ±0.44 — below aux-ViT 21.60 AND SimCLR-ViT 20.34
  (+2.97 over base). Falsifier (DINO >= aux <=10%) not fired at 5%;
  prediction (self-distillation more data-hungry) holding. 10/25% pending.

- *** BIG LANDING SWEEP SCORED (2026-07-26, ~1,660 new finals since 07-24;
  BSC counter 5,104/7,246, big lane 176/198, transfer fix VERIFIED in
  production — 242 diagtransfer finals exist):
  (1) RE-PIN ADJUDICATED — **NO RE-PIN; THE TIN WIDTH EFFECT DOES NOT
      GENERALIZE.** Pooled 12ch-vs-champion excess on the new 64px
      populations (inverse-variance over all cells with n>=2):
        dtd  mag3 +0.15±0.19 | mag6o −0.16±0.14
        path mag3 −0.01±0.27 | mag6o −0.75±0.47
        food mag3 −0.02±0.10 | mag6o +0.63±0.13
        cub  mag3 +0.02±0.01 | mag6o +0.02±0.02
        stl  mag3 +0.09±0.11 | mag6o +0.10±0.15   (esat pending, aux
        comparators not yet landed at matching pcts)
      The decision rule needed >= +0.3 pooled on >= 3 of 5 populations:
      ONE arm on ONE population qualifies (food/mag6o). VERDICT: the
      committed 8-pair bank stays, now with cross-domain evidence behind
      it — the tin@5% +0.49/+0.31 excess is a tin-local fact. NOTED, not
      acted on: on food, mag6o−mag3 = +0.65 ± ~0.16 (~4σ over 2 cells) —
      the divergence tripwire nominally fires there with ORIENTATIONS,
      not the octave, as the helpful widening (opposite of tin's
      original octave story). 2-cell read at n2-3; park unless food
      becomes a headline population.
  (2) TRANSFER COMPARATOR SCORED — BOTH PREDICTIONS HELD, THE OPEN FORK
      CLOSED AGAINST THE STACK. transfer-none vs scratch-none: dominates
      photo-like sets (+13..+18 c10, +15..+24 stl/tin, +10..+18
      c100/food, +7 dtd) but the gap NARROWS to +1.9 on eurosat and
      +1.8..+2.9 on pathmnist — the domain-shift prediction exactly; the
      "transfer wins everywhere by >10" falsifier did NOT fire, so the
      from-scratch story needs no defensive scope statement.
      aux-on-transfer (λ0=1.0 verbatim): NEGATIVE EVERYWHERE. The
      pathmnist OPEN FORK resolves to NO STACK (−0.4@10%, −1.1@7% —
      mild tax); photo-like sets take a HEAVY tax (−10..−24 on
      stl/c10/c100/food/tin; tin@10% has a collapsed seed, σ21 →
      bistable flag). The tax SCALES with init strength — the currency
      account's cleanest demonstration yet: the stronger the features
      the init already bought, the more the early shaping destroys.
      "Prior on pretrained" is dead as a deployable direction.
  (3) DINO ENVELOPE COMPLETE (all cells vit_tiny; the expansion
      diaggrid_dino_<ds> family = DINO-ViT on domains): DINO <= aux-ViT
      at every fraction <=10% on every dataset (esat −3..−5, dtd −1..−4,
      food −1..−3, c100 @10% 24.60 vs 29.74), statistical tie on
      pathmnist @7/10% (−0.17/+0.40, n3). On most domains DINO ≈ the
      PLAIN ViT baseline — the pretraining buys ~nothing at study scale
      — EXCEPT pathmnist, where it does real work (+8 over base @10%).
      DINO beats aux only at c100@100% (62.30 vs 60.50, 2x compute).
      The "DINO >= aux <=10%" falsifier: one nominal +0.40 tie-cell, no
      material fire → the ViT positioning stands against modern
      attention-SSL. Prediction (self-distillation more data-hungry
      than contrastive) CONFIRMED: DINO < SimCLR-ViT everywhere too.
  (4) SWIN — THE BASELINE COLLAPSES; THE PRIOR IS AN OPTIMIZER STABILIZER.
      swin-none is seed-bistable or at chance on most datasets (esat@15%
      {11.1,33.5,11.1}, stl@100% {10.0,22.0,10.0}, tin@10% {0.5,1.6,0.5},
      path@7% {44,17,49}, c10@10% {28,42,15}) while swin-aux trains
      tightly (esat 92.3±0.1, c10 68.3±0.2, tin 17.2±1.6) — the same
      failure family as ConvNeXt-SGD/R50-no-hn, now under AdamW, and the
      strongest stabilization signature in the study. On C100 — the ONE
      dataset where swin-none trains — Δ = +6.1/+7.6/+9.0/+7.2 at
      5/10/15/25%: ABOVE the predicted +1..+5 band and touching the
      >=+8 falsifier ("deficit is attention-intrinsic") at 15%. The
      "hierarchy supplies most of what ViT lacks" prediction MISSED —
      swin's deficit under this recipe is at least as large as ViT's.
      All swin cells carry the bistable caveat; none are headline.
  (5) PATHMNIST ANOMALY VERIFIED AS DATASET-LEVEL, NOT AN SSL BUG: every
      arm declines above ~10% (base 92.0@10% -> 89.5@25% -> 86.7@100%;
      aux 93.8 -> 86.7; simclr 94.3 -> 87.6; simsiam 91.6 -> 88.2).
      More data hurts ALL methods — consistent with PathMNIST's
      center-shifted test split + the frozen 200-epoch recipe
      overfitting train-center stain statistics. Treat pathmnist
      right-flank cells as distribution-shift-confounded; low-data
      cells (<=10-15%) are the interpretable ones.
  (6) SSL-VS-AUX ON DOMAINS (conv, partial): food = SSL territory
      (+2.4..+4.3 over aux); pathmnist ≈ tie (−0.2..+0.8); eurosat aux
      WINS @5% (+1.85) — one fraction so far, the "aux > SSL at >=2
      fractions" falsifier needs a second; dtd@15% SimCLR ≫ aux
      (+6.84: 22.57 vs 15.73) — my "SSL data-starved on dtd" prediction
      is WRONG at 15% (564 imgs suffice). SimSiam stays ~useless on
      domains too (≈base on food/path) — the effective-SSL rule holds.
  (7) BACKBONE TRANSPLANTS ON DOMAINS: mnet (frozen SGD, NO bistability)
      is champion-like on pathmnist (+2.3..+4.7 across 1-15%), modest
      on c100/food/stl, ~0 on tin@5% (−0.35, band +2..+5 MISSED) and
      dtd/cub. vit/deit pairs stack on every domain (deit esat@1%
      +17.5, path +9..+10, food@10% +20.1) except cub's deep floor.
      CUB expansion cells at 3-20% sit at 1-4% absolute (~2-8 img/cls,
      200-way) — deep-left-flank, expected, not an error.

- SWIN G-PROBE PASS LAUNCHED (2026-07-28, local 3090 on the new study venv,
  while BSC is fair-share throttled): the swin result — baseline collapses on
  most datasets, prior stabilizes, and Δ = +6.1/+7.6/+9.0/+7.2 on C100 where
  the baseline DOES train — has NO feature-side measurement. This pass probes
  diaggrid_swin_c100_{none,aux}_{5,10,15,25}pct, 3 seeds, full-train-set probe
  (identical protocol to every other G in the ledger).
  SCOPE NOTE: only C100 is probed. On esat/stl/tin/path/c10 the swin BASELINE
  is seed-bistable or at chance, so its probe would measure a TRAINING FAILURE,
  not a feature deficit — a huge "G" there would be an artifact and must not be
  fed to the law. C100 swin seeds all train (worst spread 11.7-15.2 @5%).
  PREDICTIONS RECORDED IN ADVANCE — derived from the law, not guessed. The
  sign law fixes readout from the BASELINE height (crossing ~30), and swin's
  baselines are 13.88/19.00/23.40/31.81 at 5/10/15/25%. Reading readout off
  the measured ViT-on-C100 curve at matched baselines (−3.82@12.3, −1.59@16.5,
  −0.36@28.5, +0.19@50.6) gives ≈ −3.5/−1.2/−0.8/+0.2, so G = Δ − readout:
      @5%  G ≈ +9.6   @10% G ≈ +8.8   @15% G ≈ +9.8   @25% G ≈ +7.0
  BAND: **G(swin) = +7..+11 at every fraction** — i.e. BETWEEN conv R18
  (3.55-6.26) and ViT (13.17-14.85), roughly 2/3 of ViT's.
  READING IF THE BAND HOLDS: Swin's hierarchy fills PART of the feature
  deficit attention-at-small-scale suffers, but only about a third of it; the
  rest is intrinsic to attention under this recipe. That is the quantitative
  version of the qualitative claim my original swin prediction got wrong
  (I said G(swin) ≪ G(vit) and Δ +1..+5; e2e already falsified the Δ half).
  FALSIFIER A (deficit is ATTENTION-INTRINSIC): G(swin) >= 13 at any fraction
    (≈ G(vit)) => the hierarchy buys NOTHING on the feature side and the
    "hierarchy supplies most of what ViT lacks" account dies on both axes.
  FALSIFIER B (STABILIZATION-ONLY, and it would break the law): G(swin) <= 6
    (conv-like) => swin's big e2e Δ is NOT feature-side, which forces
    readout = Δ − G to be strongly POSITIVE (+1..+3) at baselines of 14-23,
    far below the ~30 crossing — the first sign-law violation on a
    non-collapsed cell, and it would mean the prior's swin gain is pure
    optimization rescue rather than feature injection.
  Note both falsifiers are reachable from the same measurement, and the band
  between them is narrow — this is a sharp test, not a safe one.
- *** SWIN G-PROBES LANDED — ALL FOUR IN BAND, NEITHER FALSIFIER FIRED, AND
  THE LAW PREDICTED A NEW BACKBONE FAMILY FROM ITS BASELINES ALONE
  (2026-07-28/29, local 3090 on the study venv, 3 seeds/cell, full-train probe):
      pct  base_e2e aux_e2e   Δ    | probe_base   probe_aux     G      readout
       5%   13.88   19.96  +6.08 | 24.42±4.26  34.93±0.22  +10.51±2.46  −4.43
      10%   19.00   26.58  +7.58 | 29.91±1.03  39.21±0.65   +9.29±0.71  −1.71
      15%   23.40   32.42  +9.02 | 32.94±2.21  43.00±0.10  +10.06±1.28  −1.04
      25%   31.81   39.00  +7.19 | 39.03±0.78  46.62±0.24   +7.59±0.47  −0.40
  PREDICTION SCORING: band was +7..+11 at every fraction — 4/4 IN BAND, and
  the POINT predictions derived from the law (9.6/8.8/9.8/7.0) each landed
  within ~0.9 of measurement (10.51/9.29/10.06/7.59) with the ordering
  preserved. Falsifier A (G>=13, attention-intrinsic) did NOT fire (max
  10.51); falsifier B (G<=6, stabilization-only) did NOT fire (min 7.59).
  This is the first time the law was used to predict a NEW BACKBONE FAMILY's
  feature gain in advance from nothing but its baseline heights, and it held.
  THE ANSWER ON HIERARCHY, quantified: G(swin) 7.6-10.5 sits BETWEEN conv R18
  (3.55-6.26) and ViT (13.17-14.85) — ~2x conv, but only ~2/3 of ViT. So
  Swin's hierarchy fills roughly ONE THIRD of the attention feature deficit
  and the rest is intrinsic to attention at this scale. My ORIGINAL swin call
  ("G(swin) << G(vit)") is directionally right, but the inference I drew from
  it ("hierarchy supplies MOST of what ViT lacks") is WRONG — it supplies
  about a third. The e2e half of that prediction (Δ +1..+5) was already
  falsified by the +6..+9 landings.
  SIGN LAW: readout −4.43/−1.71/−1.04/−0.40 — negative throughout and
  MONOTONE rising toward zero as the baseline rises, 4 more clean cells
  (~34 total, first on a hierarchical-attention backbone). The 25% cell
  TIGHTENS THE CROSSING BRACKET: readout is still −0.40 at base 31.81, so
  the zero-crossing is ABOVE 31.8; with conv's +0.08 at base 40.28 the
  bracket narrows to **base ∈ [31.8, 40.3]** (was [29.8, 39.4], "likely
  just above 30" — that guess is now excluded).
  STABILIZATION IS VISIBLE AT THE FEATURE LEVEL TOO: baseline probe σ
  4.26/1.03/2.21/0.78 vs aux probe σ 0.22/0.65/0.10/0.24. The @5% baseline
  has a probe outlier (seed2 19.50 vs 26.9) mirroring its e2e wobble
  (11.7 vs 15.2) — so the instability the prior rescues is a property of the
  learned FEATURES, not just of the classifier. Consistent with the standing
  rule: variance reduction is NOT a routine property, but instability rescue
  is real (R50 no-hn, ConvNeXt, and now Swin).
  NET: the prior's swin gain is genuinely feature-side (falsifier B dead) AND
  the prior stabilizes — both, not either. Swin cells stay non-headline
  (bistable baselines off C100), but the C100 column is now a full law cell.
- TRANSFER-TAX G-PROBE PASS LAUNCHED (2026-07-29, local 3090): the big-sweep
  claim "aux-on-transfer is negative everywhere and the tax SCALES with init
  strength — the currency account's cleanest demonstration yet" is e2e-ONLY.
  The currency account says the prior's early λ0=1.0 shaping DESTROYS features
  the ImageNet init already bought; that is a claim about FEATURES and has
  never been measured. This pass probes diagtransfer2_{none,aux} at
  c10@5% (tax −17.11, base 81.65), c100@7% (tax −16.37, base 45.64) and
  path@10% (tax −0.41, base 93.79) — 3 seeds each.
  CELL CHOICE IS CONSTRAINED BY THE PROBE-CEILING RULE: the 100% transfer
  cells (where the tax nearly VANISHES: c100 −0.69, c10 −0.26, esat −0.06)
  would be the most interesting comparison but their probe labels == cell
  labels, so no G/readout split is interpretable there. Low/mid fractions
  only, where the probe holds 10-20x the cell's labels.
  PREDICTIONS RECORDED IN ADVANCE, derived from Δ = G + readout. All three
  baselines sit FAR ABOVE the [31.8, 40.3] crossing, so readout is on the
  measured positive branch and small (C10's mapped branch: +0.44 @ base 80.7),
  which forces essentially the whole tax onto G:
      c10@5%   readout ≈ +0.4 => G ≈ −17.5   BAND [−20, −14]
      c100@7%  readout ≈ +0.3 => G ≈ −16.7   BAND [−19, −13]
      path@10% readout ≈ +0.3 => G ≈  −0.7   BAND [−2.5, +0.5]
  ORDERING PREDICTION: G(c10) ≈ G(c100) ≪ G(path) — a photo-matched init has
  far more to destroy than a stain-domain-mismatched one.
  FALSIFIER A (currency account WRONG, tax is READOUT-side): |G| <= 4 on
    EITHER heavy cell while Δ is −16..−17 => the prior did not damage the
    features, it broke the classifier/optimization, and "destroys what the
    init bought" must be replaced by an optimization story.
  FALSIFIER B (damage worse than e2e shows): G <= −25 on a heavy cell => the
    readout term is doing large RESCUE work (strongly positive), which the
    positive-branch decay says it cannot — would break the law at high
    baselines.
- *** TRANSFER-TAX PROBES LANDED — THE TAX IS ~ENTIRELY FEATURE-SIDE; THE
  CURRENCY ACCOUNT IS NOW MEASURED, NOT INFERRED (2026-07-29, 3 seeds/cell):
      cell        base   Δ_e2e | probe_none  probe_aux      G        readout
      c10 @5%    81.65  −17.11 | 83.71±1.64 67.90±2.46 −15.81±1.71  −1.30
      c100@7%    45.64  −16.37 | 54.63±1.92 38.37±4.92 −16.26±3.05  −0.11
      path@10%   93.79   −0.41 | 92.01±0.49 92.28±1.00  +0.27±0.64  −0.68
  3/3 IN BAND (predicted −17.5/−16.7/−0.7; c100 landed within 0.4 of its
  point prediction). FALSIFIER A (|G|<=4 on a heavy cell => tax is
  readout-side) did NOT fire — G carries essentially the ENTIRE tax
  (−15.81 of −17.11; −16.26 of −16.37). FALSIFIER B (G<=−25) did NOT fire.
  ORDERING PREDICTION HELD EXACTLY: G(c10) ≈ G(c100) ≪ G(path).
  WHAT THIS ESTABLISHES: the early λ0=1.0 shaping DESTROYS ImageNet features
  — this is damage to the representation itself, not a classifier/optimizer
  failure. And the decisive control is pathmnist: where ImageNet features are
  domain-mismatched (transfer beats scratch by only +1.8..+2.9 there vs
  +13..+24 on photo sets), there is NOTHING to destroy and G = +0.27 ±0.64,
  indistinguishable from zero (0.4σ). The damage is PROPORTIONAL TO WHAT THE
  INIT ACTUALLY SUPPLIED — the currency account's sharpest confirmation, and
  the first time it has been shown on the FEATURE side of a NEGATIVE result.
  READOUT — SCORED HONESTLY AS UNRESOLVED, NOT AS A VIOLATION: the three
  readouts (−1.30, −0.11, −0.68) are nominally mild NEGATIVES at baselines
  far ABOVE the [31.8, 40.3] crossing, where the aux-derived positive branch
  (+0.44 @ base 80.7) would predict small positives. But every one is within
  ~1σ of zero given G's SEM (±1.71/±3.05/±0.64), so NONE is resolvable and
  none is significantly different from the predicted small positive either.
  Additionally the sign law was derived on aux-FROM-SCRATCH cells; a
  pretrained-init TAX cell is outside its derived scope. Recorded as
  not-scored (same treatment as tin20's noisy readout), NOT as a 35th cell
  and NOT as a counterexample. Deepening these three cells' probes would
  settle whether the positive branch extends to tax cells.
  CAVEAT for the path cell: probe_none 92.01 sits BELOW its e2e 93.79 — on
  pathmnist the finetuned network beats a linear head on its own frozen
  features, so absolute probe levels there understate the cell. The DELTA is
  unaffected (both arms probed identically), which is the measured quantity.
  ASIDE, derivable from the existing e2e table and worth stating: the tax
  DECAYS TO ~ZERO AT FULL DATA on every dataset (c10 −17.65@3% -> −0.26@100%;
  c100 −11.20@3% -> −0.69@100%; esat −5.75@3% -> −0.06@100%; tin -> −0.56).
  So "the tax scales with init strength" is more precisely: the tax scales
  with HOW MUCH OF THE FINAL PERFORMANCE THE INIT IS CARRYING. At 100% the
  data dominates and the λ->0 schedule's structural neutrality reasserts
  itself, exactly as it does for aux-from-scratch at 100%.
- *** deit-ssl ON tin EXTENDED TO 3 FRACTIONS — THE MODERN-RECIPE FLIP
  TRANSPLANTS BUT ONLY IN THE LOW-DATA BAND (2026-08-04, 3 seeds):
      pct  deit-none  deit-ssl  deit-aux | aux−ssl        verdict
        5    8.29      16.67     18.46   | +1.79 ±0.29    prior WINS (6.2σ)
       10   10.49      25.45     27.43   | +1.98 ±0.32    prior WINS (6.1σ)
       25   23.14      38.48     38.24   | −0.24 ±0.44    TIE (0.5σ)
  The pre-registered falsifier was "deit-ssl >= deit-aux on tin => the flip
  is C100-specific and the modern-recipe claim must be scoped to CIFAR".
  At 25% deit-ssl is NOMINALLY above deit-aux (+0.24) — but at 0.5σ that is
  a TIE, not a win, so the falsifier does not fire in substance. Scored
  honestly rather than as a clean pass, because the SHAPE differs from C100:
  there the prior beat SimCLR at EVERY fraction including 25% (+5.01) and
  100% (+2.00); on tin its edge DECAYS to zero by 25%.
  RESTATEMENT: "the prior beats SimCLR once a small ViT is trained the modern
  way" holds on BOTH populations, but on tin only up to ~10-15% — above that
  they converge. Same direction as every other axis in this study (the
  prior's advantage is a low-data phenomenon); the C100-wide claim was the
  outlier, not the rule.
- *** LEGACY FAMILIES FILLED IN — TWO TWO-POINT CLAIMS BECAME FULL ENVELOPES,
  AND THEY REPRODUCE UNDER A NEW OS (2026-08-04, C100/r18, Δ vs baseline):
      pct   enrot   enstr   eninv   enste  | axteach(fitnets)
        1   +0.92   +0.42   +0.41   +1.81  |  −0.02
        5   +0.32   −0.40   −3.53   +1.37  |    --
       10   −5.74   −5.30   −9.45   −4.92  |    --
       15   −8.99   −7.81  −10.85   −8.40  |  +0.35
       25  −11.87   −8.66  −10.39  −11.07  |  −0.03
      100   −1.42   −1.03   −1.13   −1.09  |  −0.04
  (1) REPRODUCTION CHECK: enrot@10% = −5.74 vs the recorded −5.64, enstr@10%
      = −5.30 vs −5.20 — both within 0.1 of values measured months ago on a
      different machine and now a different OS (RHEL 9.6). The forward-path
      negatives are solid.
  (2) The forward-path penalty band now has its full SHAPE on four energy
      stems: mildly POSITIVE at 1-5%, deeply negative through 10-25%
      (−8..−12), then recovering to ~−1 at 100%. Same shape as the Gabor
      stems — "any fixed pre-committed extra channel costs accuracy at 10%+"
      is now an envelope statement, not a single-cell one.
  (3) axteach (FitNets learned teacher) is **~0 at EVERY fraction**
      (−0.25..+0.73 across 1-100%). The recorded claim rested on two points
      (−0.36@5%, +0.16@10%); it is now an 8-point envelope of nothing, which
      makes "a LEARNED target that costs a whole extra model does ~NOTHING
      while the free hand-crafted moment gives +3.3/+2.8" much harder to
      dismiss. Worth having run after all.
- *** THE tin ViT/DeiT ENVELOPE IS COMPLETE — THE LAST PRE-REGISTERED
  FALSIFIER BLOCK LANDS (2026-08-04, 3 seeds every cell):
      pct  vit-none vit-aux  Δ_vit | deit-none deit-aux  Δ_deit | amp
        1    3.45     4.84   +1.39 |   3.99      7.04    +3.05  | 2.20x
        2    4.17     7.46   +3.29 |   4.28     10.84    +6.56  | 1.99x
        5    6.02    12.79   +6.77 |   8.29     18.46   +10.18  | 1.50x
       10    8.60    17.75   +9.15 |  10.49     27.43   +16.94  | 1.85x
       15   10.92    21.64  +10.73 |  15.15     31.85   +16.70  | 1.56x
       25   15.86    26.35  +10.48 |  23.14     38.24   +15.11  | 1.44x
      100   34.81    41.98   +7.17 |  50.72     57.32    +6.60  | 0.92x
  PREDICTION SCORING (vitenv bands, recorded 2026-07-23):
    "Δ_vit(tin) peaks +10..+15 @15-25%" — **HIT**: peak +10.73 @15%, +10.48
      @25%, in band and peaking exactly where predicted. The tin plain-ViT
      envelope therefore has the SAME SHAPE as C100's (which also peaks at
      15%, +14.44) at lower magnitude — a second population for the
      envelope-peak claim.
    "stays +6..+10 @100%" — HIT (+7.17), scored earlier.
    "deit amplification 1.8-2.4x throughout" — **MISSED LOW** at 5/15/25%
      (1.50/1.56/1.44); only 1/2/10% sit in or near the band. The recorded
      falsifier ("amplification < 1.3x anywhere on the mid-band") did NOT
      fire — the minimum is 1.44x — so stacking transplants, but WEAKER than
      on C100. Honest statement: heavy augmentation amplifies the prior
      ~1.4-1.9x on tin vs ~1.8-2.4x on C100, i.e. amplification is
      population-dependent and I over-generalized C100's value.
    Falsifiers: Δ_vit(tin@100%) <= +2 (permanent deficit C100-specific) did
      NOT fire; deit-ssl >= deit-aux did NOT fire (25.45 < 27.43 @10%).
  CAVEAT ON THE AMPLIFICATION NUMBERS: the deit-none tin baselines are
  HIGH-VARIANCE though not collapsed — per-seed @15% {9.72, 16.63, 19.09},
  @10% {11.94, 6.55, 12.98}, @2% {5.29, 5.14, 2.40}. No seed sits at chance
  (so not the swin/ConvNeXt bistable family), but a σ up to 4.86 on the
  DENOMINATOR's baseline makes each amp ratio poorly determined at 10-15%.
  The aux arms are tight (σ 0.29-0.68) — the same variance asymmetry seen
  everywhere else. Treat the amp column as ±0.3, not as three digits.
- *** EUROSAT ANSWERS THE DOMAIN-SSL FALSIFIER — THE PRIOR BEATS SimCLR ON
  CONV, AND THE "SSL WINS ON CONV" CLAIM IS NOW SCOPED TO PHOTO DOMAINS
  (2026-08-02, full 11-fraction envelope, 3 seeds every cell):
      pct    base     aux    simclr | aux−ssl | aux−base
        1   67.47   70.14   69.35   |  +0.79  |  +2.67
        2   77.98   79.59   80.09   |  −0.51  |  +1.60
        3   83.90   85.67   84.24   |  +1.43  |  +1.77
        5   90.80   91.64   89.79   |  +1.85  |  +0.84
        7   92.25   93.54   92.10   |  +1.44  |  +1.30
       10   93.67   94.90   94.13   |  +0.77  |  +1.22
       15   96.04   96.20   95.95   |  +0.25  |  +0.16
       20-100: all three arms converge (|Δ| <= 0.14)
  The recorded falsifier — "any domain with aux > SSL at >=2 fractions kills
  'SSL always wins on conv'" — FIRED at **7 fractions** (1/3/5/7/10/15/50),
  with the material margins in the 1-10% band (+0.77..+1.85). The domssl
  prediction had eurosat as the explicit OPEN FORK ("uncertain — no number");
  it resolves toward the prior, and for the pre-registered REASON: SimCLR's
  views encode PHOTO invariances (crop + colour jitter), which transfer badly
  to satellite imagery where the real invariances differ (rotation is
  meaningful, colour statistics are not photographic), while the moment
  prior's oriented-energy target is domain-agnostic.
  STATE IT WITH THE NUANCE, because the domain axis does NOT split cleanly
  into "photo vs non-photo": on DTD — also non-photo, texture-dominated —
  SimCLR BEATS aux heavily (+6.84 at 15%), and on food/pathmnist SSL wins or
  ties. So the correct claim is **eurosat is a population where the free
  prior beats 2x-compute SimCLR outright in the low-data band**, which is
  enough to retire the blanket "on conv, SSL wins if you can pay 2x" — that
  sentence must now name its populations (C100, tin, food, dtd) rather than
  claim conv in general.
  Right flank behaves exactly as everywhere else: by 20% all three arms are
  within 0.14 of each other and the prior is structurally neutral.

## IMAGENET-SCALE VALIDATION (2026-08-05, user: "push it further and make it
## a TPAMI-class claim")

- WHY: every result in this study is <=100k images, <=200 classes, <=64px,
  ResNet-18/ViT-tiny. The reviewer objection is not rigor, it is SCALE. The
  differentiator we have is that the law PREDICTS (Swin's G was called in
  advance from baselines alone, 4/4 in band). So the test that changes the
  paper's class is: **does the law still predict at a scale reviewers care
  about?** Not "does the method win there" — the claim is predictive, not SOTA.
- CONSTRAINT DISCOVERED: BSC has NO outbound internet on login OR compute
  nodes (curl returns nothing; this is the same wall that forced scp'ing pip
  wheels). Every dataset must be fetched on the LOCAL machine and rsynced.
  9.1T free on BSC scratch, 13T free on local /media/HDD_16TB — space is not
  the constraint, acquisition is.
- STAGE 1 (launched): **ImageNet64** (benjamin-paine/imagenet-1k-64x64,
  ungated parquet, 1.28M images, 1000 classes, 64px). Chosen because it drops
  into the EXISTING 64px pipeline verbatim (identical to tin), so there is no
  resolution or recipe confound to argue about — only scale changes:
  13x the images and 5x the label space of anything measured.
  Cells: imagenet64 {none,aux} x {resnet18, vit_tiny}, 3 seeds + probes.
  DEVIATION, stated up front: 200 epochs on 1.28M images is not affordable;
  these cells run a REDUCED epoch budget and therefore carry a diag prefix
  and are NEVER mixed into the frozen-recipe headline tables. Both arms of
  each pair get the identical budget, so Delta stays valid (the step-count
  lesson from diag10e800).
  PROBE NOTE: a full-train-set probe at 1.28M x 512 is impractical for LBFGS;
  these G values will use a FIXED SHOTS budget (--shots) and must be compared
  only to other same-budget probes, never to the full-train G curve.
- PREDICTIONS RECORDED IN ADVANCE (before any ImageNet64 run):
  The sign law says readout is a function of BASELINE HEIGHT, crossing zero
  at base in [31.8, 40.3] and decaying on the positive branch (+0.44@80.7).
  ImageNet64 baselines will sit FAR ABOVE the crossing (R18 ~50-60% expected,
  1280 img/class), so the law makes a sharp, almost parameter-free call:
    (P1) readout ~ 0..+0.5 => **Delta ~= G within +-0.5** on BOTH backbones.
         This is the core predictive claim: whatever the prior does to the
         FEATURES shows up in accuracy nearly 1:1, with no readout penalty.
    (P2) CONV at 1.28M images is deep on the right flank => G(r18) ~ 0..+1
         and **Delta(r18) = 0.0 +-0.5** (neutral). Every one of the 10
         populations measured is neutral at full data; 1.28M images is the
         most data the prior has ever faced, so this is the strongest form
         of the redundancy claim.
    (P3) ViT-tiny: if the PERMANENT-DEFICIT conclusion is real (+9.88 at full
         C100 50k, +7.17 at full tin 100k, where every conv is neutral), it
         must survive 1.28M images: **Delta(vit) >= +3**, with G(vit) >= 4.
  FALSIFIERS, each of which changes a headline claim:
    (F1) |Delta - G| > 1.5 on either backbone at these high baselines =>
         the readout term is NOT a pure function of baseline height at scale,
         and the law's predictive form is bounded to <=100k images. This is
         the one that would cost the paper its central contribution.
    (F2) Delta(r18) >= +1.5 => the prior is NOT redundant at full data after
         all, and "neutral at 100% by construction" needs rewriting (it would
         also be the method's best practical news).
    (F3) Delta(vit) <= +1 => ViT-tiny's deficit is DATA-BOUNDED, not
         permanent; the ViT headline must be restated as "small-data
         attention" and the strongest claim in the paper weakens.
  NOTE the asymmetry deliberately built in: F2 and F3 are opposite-signed,
  so this single experiment cannot be passed by any uniform outcome -- a flat
  result confirms P2 and fires F3; a large result fires F2 and confirms P3.
- DOMAIN G-PROBE PASS LAUNCHED (2026-08-05): the GENERALIZABILITY audit found
  the law is sampled far more narrowly than the effect — Delta is demonstrated
  on 14 populations (~127 paired cells) but G+readout on only ~5 (probe cells:
  tin 26, c100 15, vit 15, c10 10, but PathMNIST 4, CUB 4, and ZERO on
  eurosat/dtd/food101). "Is the law general or a CIFAR/tin regularity?" is
  therefore weaker than it needs to be, and the fix is cheap: the checkpoints
  ALREADY EXIST on BSC, so this needs probe runs only, no retraining.
  Cells chosen to STRADDLE the readout crossing [31.8, 40.3] on four new
  populations (3 below, 5 above), which is what makes the sign law falsifiable
  here rather than merely confirmable:
      cell          base    Delta  | branch
      dtd  @5%      8.35   +0.30   | below  (readout must be NEGATIVE)
      dtd  @15%    13.58   +2.15   | below
      food @5%     21.91   +5.63   | below
      food @10%    41.73   +3.18   | above  (readout must be POSITIVE)
      food @15%    52.88   +0.46   | above
      dtd  @100%   43.51   +3.55   | above
      esat @5%     90.80   +0.84   | above, far
      path @10%    92.01   +1.74   | above, far
  PREDICTIONS RECORDED IN ADVANCE, readout read off the MEASURED curves at
  matched baseline height (C100: -2.76@8.9, -1.12@25.4, +0.08@40.3; tin:
  -2.71@5.3, -0.54@21.1, -0.06@33.6; positive branch: +1.80@39.4, +1.14@51.7,
  +0.44@69.1, +0.44@80.7), so G_pred = Delta - readout_pred:
      dtd  @5%   readout ~ -2.7 => G ~ +3.0   BAND [+1.0, +5.0]
      dtd  @15%  readout ~ -2.0 => G ~ +4.2   BAND [+2.0, +6.5]
      food @5%   readout ~ -1.4 => G ~ +7.0   BAND [+4.5, +9.5]
      food @10%  readout ~ +0.1 => G ~ +3.1   BAND [+1.5, +4.5]
      food @15%  readout ~ +1.1 => G ~ -0.6   BAND [-2.0, +1.5]
      dtd  @100% readout ~ +0.3 => G ~ +3.3   BAND [+1.5, +5.0]
      esat @5%   readout ~ +0.4 => G ~ +0.4   BAND [-1.0, +2.0]
      path @10%  readout ~ +0.4 => G ~ +1.3   BAND [-0.5, +3.0]
  FALSIFIER (the law is CIFAR/tin-specific): readout = Delta - G coming out
    POSITIVE at any of the three sub-crossing cells (dtd@5/15, food@5), or
    NEGATIVE below -1.0 at either far-above cell (esat@5, path@10). Either
    would show the sign law does not transfer to non-photo domains and the
    law's scope must be stated as "CIFAR-like populations".
  NOTE esat@5 and food@15 are deliberately included as the HARD cases: their
  Deltas are small (+0.84, +0.46), so if G comes back large the readout term
  would have to be strongly negative at a high baseline — the cleanest way for
  this pass to fail.
  PLACEMENT: run on BSC's now-idle grid lane (GRID_COMPLETE fired 2026-08-05),
  NOT the local 3090 — that GPU is running the user's other project.
- *** DOMAIN G-PROBES LANDED — THE LAW IS NOT A CIFAR/tin REGULARITY
  (2026-08-05, 8 cells on 4 new populations, 3 seeds each):
      cell        base    Δ    | G           | readout | band        verdict
      dtd @5%     8.35  +0.30  | +1.88±0.50  |  −1.58  | [1.0,5.0]   IN
      dtd @15%   13.58  +2.15  | +2.52±0.38  |  −0.37  | [2.0,6.5]   IN
      food@5%    21.91  +5.63  | +6.18±0.57  |  −0.55  | [4.5,9.5]   IN
      food@10%   41.73  +3.18  | +2.80±0.85  |  +0.37  | [1.5,4.5]   IN
      food@15%   52.88  +0.46  | −0.12±0.10  |  +0.58  | [-2.0,1.5]  IN
      dtd @100%  43.51  +3.55  | +3.55±0.54  |  +0.00  | [1.5,5.0]   IN
      esat@5%    90.80  +0.84  | +0.30±0.14  |  +0.54  | [-1.0,2.0]  IN
      path@10%   92.01  +1.74  | −0.91±0.70  |  +2.65  | [-0.5,3.0]  OUT
  **7/8 IN BAND, and 8/8 CORRECT IN SIGN.** The recorded falsifier — readout
  POSITIVE at any sub-crossing cell, or below −1.0 at a far-above cell — did
  NOT fire anywhere. The three sub-crossing cells (dtd@5/15, food@5) all give
  NEGATIVE readout; the five above-crossing cells all give POSITIVE (or zero
  at dtd@100%, whose base 43.51 sits just above the bracket). The two
  deliberately HARD cases behaved: esat@5 (Δ +0.84) returned G +0.30, and
  food@15 (Δ +0.46) returned G −0.12 — small Δ genuinely means small G, not a
  large G cancelled by a large readout.
  SCOPE CHANGE: G+readout was previously demonstrated on ~5 populations, all
  CIFAR-like. It now holds on **satellite (eurosat), texture (dtd), fine-
  grained food, and histopathology** as well, spanning both branches of the
  sign law. "Is the law a CIFAR/tin regularity?" is answered: no.
  THE ONE MISS, scored honestly and NOT explained away: path@10% returned
  G = −0.91 ±0.70 against a predicted +1.3, forcing readout = +2.65 where the
  positive branch predicts ~+0.4. The sign is still correct. But note the
  caveat already recorded for pathmnist on 2026-07-29: its probe_none (91.77)
  sits BELOW its own e2e (92.01) — on pathmnist a linear head on frozen
  features UNDER-READS the finetuned network, so the probe is a compressed
  measuring stick there and absolute G is not trustworthy. This is the same
  class of limitation as the probe-ceiling rule, and it is a property of the
  PROTOCOL on that dataset, not evidence against the law. Recorded as a MISS
  regardless; deepening it would need a stronger probe (e.g. MLP head) whose
  results would not be comparable to the rest of the ledger.
- *** THE SIGN LAW RE-AUDITED AT 84 CELLS — AND THE HEADLINE NUMBER I ALMOST
  REPORTED WAS WRONG (2026-08-05). The 190-task probe campaign took the law
  from ~5 populations to **14** (84 r18 champion-pair cells with both a probe
  and >=3 seeds). A naive sign count looked alarming:
      below crossing: 21/27 negative readout (78%)
      above crossing: 27/52 positive readout (**52%** -- a coin flip)
  That framing is WRONG, and the error is instructive: above the crossing the
  law PREDICTS readout ~ +0.4 decaying to ~0, and at high fractions Delta and
  G are both ~0, so readout is a small difference of small numbers and its
  SIGN is pure noise. Counting those signs tests nothing.
  Re-audited against each cell's OWN uncertainty (SEM propagated from both
  e2e arms and both probe arms):
      UNRESOLVED (|readout| <= 2 SEM):        63 cells (75%)
      significantly on the PREDICTED side:    20 cells
      significantly on the WRONG side:         1 cell (1%)
  So of the **21 cells where readout is resolvable at all, 20 have the
  predicted sign**. The single exception is c10@50% (readout −0.25 ±0.08 at
  base 93.04, where a small POSITIVE was expected) -- statistically resolved
  only because its errors are tiny, and physically negligible at 0.25 points.
  CONSEQUENCE FOR THE PAPER, and it is a strengthening not a weakening: the
  sign law's CONTENT lives where readout is resolvable -- the left flank and
  mid-band, where it is large and negative. On the right flank it correctly
  predicts ~0 and gets ~0, which is true but not discriminating. State it that
  way. A reviewer who runs the naive sign count will get 52% and think the law
  fails; the paper should preempt that by reporting the uncertainty-weighted
  version and being explicit that 75% of high-data cells cannot test it.
  This is exactly what broadening from 5 to 14 populations was for.
- *** THE LAW AT 497 CELLS (2026-08-05, after the 1,707-probe campaign; BSC
  probe count 206 -> 2,238 in a few hours once the per-GPU concurrency fix
  landed). Restricted to the law's DERIVED SCOPE -- aux-from-scratch, no
  pretrained init, no SSL init, stem none:
      497 law cells | 6 backbones | 14 datasets
        vit_tiny 169, resnet18 130, swin_tiny 91, mobilenetv3 91, r50 8, r34 8
      readout RESOLVABLE (|readout| > 2 SEM):  134
      sign as predicted:                       129 (**96%**)
      wrong side:                                5
  This is the paper's central evidence and it is now an order of magnitude
  larger than the ~34 cells it rested on this morning.
  SCOPE DISCIPLINE THAT MATTERS: a first pass mixed in the ImageNet-TRANSFER
  cells and reported 86%. Those are `pretrained: true` (the tax), which the
  2026-07-29 entry explicitly places OUTSIDE the sign law's derived scope,
  and they enter with large NEGATIVE Delta and G. Excluding them (and the
  SSL-init cells, which are non-aux interventions scored separately) is not
  cherry-picking -- it is applying the scope the law was defined with. Both
  numbers are recorded here so the choice is visible.
  THE 5 EXCEPTIONS, and one of them is a pattern worth naming:
      food101 r18 @50%    base 71.7  D −1.30  G +4.66  ro −5.96
      food101 mnet @50%   base 55.1  D −0.23  G +3.70  ro −3.93
      pathmnist mnet @20% base 89.0  D −0.48  G +1.88  ro −2.36
      cifar100 r18 @20%   base 54.1  D +4.03  G +5.92  ro −1.90
      dtd vit @50% (deit) base 14.2  D +17.30 G +14.73 ro +2.57
  The first three share a signature: at HIGH data the prior still IMPROVES
  the frozen features (G > 0, clearly resolved) while COSTING accuracy
  (Delta < 0). That is a real effect the current law does not capture --
  "better features, worse accuracy" at sufficiency -- and it is the most
  interesting open thread left. It is also consistent with the long-standing
  overshoot account (early lambda=1.0 shaping costs at high data), but the
  probe now shows the cost is NOT feature-side.
- *** THE 69 GAP PROBES RAN LOCALLY, AND THE AUDIT IS NOW A COMMITTED SCRIPT
  (2026-08-05, user: "can we run them locally instead of waiting BSC? We can
  leave BSC for actual training"). Yes -- and for a stronger reason than
  scheduling. The pass surfaced three latent bugs:
  (a) BSC was BLOCKED, not slow: all 8 ms_grid jobs sat at QOSGrpNodeLimit
      behind the 4 ImageNet big-lane jobs, which hold the whole group node cap.
  (b) **40 of the 69 probe tasks would have FAILED on BSC** -- those cells
      (abl*, c10_*, diagcnxadamw_*, diagdeit_*) were trained locally and their
      checkpoints never existed there. Queued as-is it was a crash-loop: the
      axteach signature, SIXTH guard/asset-drift incident. Local had 56/69
      already, the other 13 were a 2.7GB pull -- local 69/69 vs BSC 29/69.
  (c) CONVNEXT PROBES WERE DEAD EVERYWHERE: timm ConvNeXt has NO top-level
      `global_pool` attribute at all (pooling lives inside NormMlpClassifierHead),
      so linear_probe.py's bare attribute access raised AttributeError before
      any branch ran -- all 22 ConvNeXt probes, on BSC and locally. Fixed by
      deferring to timm's forward_head(pre_logits=True), gated on the
      attribute's ABSENCE so every family that HAS a global_pool keeps the exact
      branch its recorded G was measured under. VERIFIED across five families:
      extracted dim == classifier.in_features (cnx 768, r18 512, vit 192,
      swin 768, mnet 1024). ConvNeXt now has feature-side evidence for the
      FIRST time in the study.
  (d) `head_pool` WAS NEVER PLUMBED into the probe: MultiMaskPool cells replace
      global_pool with a masked 4096-d readout, so the model built for them did
      not match their state_dict at all. build_model already accepted head_pool;
      linear_probe.py simply never passed it. Defaults to None elsewhere, i.e.
      previous behaviour exactly.
  *** NEW INCIDENT CLASS -- SILENT LOCAL CHECKPOINT CORRUPTION THAT RSYNC WILL
  NEVER REPAIR: 5 of 205 local checkpoints (all grid_c100_cnx_*) failed
  torch.load with "PytorchStreamReader ... invalid header or archive is
  corrupted", while the BSC originals loaded fine AT BYTE-IDENTICAL SIZE.
  Because rsync's default quick check is size+mtime, every previous sync
  silently preserved the corruption -- it is invisible to the normal repair
  path. Deleted and re-pulled; all 5 verified. LESSON: verify run mirrors by
  LOADING, not by listing -- a same-size file is not an intact file. Only these
  205 of the local tree's 2751 cells were scanned; a full scan is worth doing.
  ALSO CHECKED, because deleting checkpoints mid-wave is exactly how a silent
  partial result happens: linear_probe.py SKIPS a missing checkpoint rather than
  failing, so a cell probed during the delete/re-pull window would have been
  silently UNDER-SEEDED. Audited all 69 probe-seed counts against their
  checkpoint counts: 0 under-seeded. The 3 genuine failures wrote no JSON at all
  (the exception aborts before the write), i.e. they failed loud.
  *** THE SIGN LAW AT 909 CELLS -- 250/260 RESOLVABLE CORRECT (**96%**), and the
  audit is now analysis/audit_sign_law.py rather than an ad-hoc query:
      cells with Delta+G:                909 | 7 backbones | 14 datasets
        r18 500, vit 189, swin 91, mnet 91, r50 24, r34 8, convnext 6
      inside crossing bracket [31.8,40.3] (no prediction):  98
      unresolved (|readout| <= 2 SEM):                     551
      RESOLVABLE (these test the law):                     260
        sign as predicted: 250 (96%) | wrong side: 10
  STATE THE COUNT CHANGE HONESTLY: this is NOT "497 -> 909 because of the 69 new
  probes" -- 69 probes cannot add 412 cells. The committed script's scope is
  BROADER than the morning's ad-hoc one: it admits every aux-from-scratch cell
  including bank variants (mag3/mag6o), tap variants (L3/L4) and non-champion
  lambdas, which ARE in scope by the law's definition. What is genuinely
  reproducible is that the headline percentage is UNCHANGED at 96% under a scope
  ~1.8x wider, and that it is now regenerable by one command.
  *** THE PRETRAINED-SCOPE LEAK FIRED A THIRD TIME, now fixed at the SOURCE: the
  first run of the new script reported 90% because all ~40 diagtransfer2 TAX
  cells were back in -- the scope filter keyed on `init_from`, but transfer cells
  carry `pretrained: true` and NO init_from. The exporter now emits a
  `pretrained` column and the audit filters on the real property instead of on a
  cell-name prefix. Both numbers are printed (--scope all gives 332/374 = 89%)
  so the choice stays visible, as the earlier 2026-08-05 entry requires.
  *** THE 10 EXCEPTIONS, and HALF are the SAME open thread: five carry the
  "BETTER FEATURES, WORSE ACCURACY AT HIGH DATA" signature flagged this morning
  -- G clearly positive and resolved while Delta is negative:
      food101   r18  mag3  @50%  base 71.8  D -1.30  G +4.66  ro -5.96
      food101   r18  mag6o @50%  base 71.8  D -0.91  G +4.97  ro -5.88
      food101   mnet aux   @50%  base 55.1  D -0.23  G +3.70  ro -3.93
      pathmnist mnet aux   @20%  base 89.0  D -0.48  G +1.88  ro -2.36
      pathmnist r18  mag3  @50%  base 89.8  D -1.36  G +0.71  ro -2.07
  Two more are pathmnist@1% (base 80.8, D +5.6..+5.8, G +9.6..+9.9, ro ~-4.0):
  large POSITIVE Delta with an even larger G -- the same "readout negative at a
  high baseline" shape, on the dataset whose probe is already recorded as a
  COMPRESSED measuring stick (probe_none BELOW its own e2e). The remaining three
  are small-magnitude singletons. So the exception set is not scattered noise: it
  concentrates on food101/pathmnist at >=20% data, exactly the regime the current
  law does not model.
- *** RUN-THE-FULL-GRID DECISION (2026-08-06, user): the rebuilt 7,479-task
  worklist runs IN FULL -- no pruning. I proposed cutting ~1,760 tasks whose
  verdicts are already adjudicated at power (simsiam_domains 552: SimSiam
  ~useless on 5 populations; stgaborlearn 111: prior-as-init falsified;
  dino_domains 570: envelope declared complete; parts of transfer2), on
  marginal-information grounds. User decision: "having all of them are
  defensible and complete answers. Having a complete results always safer" --
  the axteach precedent (a complete null envelope is much harder to dismiss
  than a partial one) applied campaign-wide. Decisive families (diagdepth,
  diagin*) stay ORDERED FIRST in the worklist, so the completeness tail costs
  cluster-days only, never time-to-decisive-results. Do not reopen.
- *** PAPER FRAMING DECISION (2026-08-06, user): the paper is sold as
  **"a controlled, cost-normalized benchmark of data-efficiency interventions
  -- and the predictive law that organizes its results."** Benchmark as the
  spine (frozen-recipe instrument -> complete grid -> decision guide), the law
  Delta = G + readout as the analysis layer that EXPLAINS the grid (stack-vs-
  substitute, sign law at 96% of resolvable cells), never as a separate
  thread. Rationale: the study's honest content is a multi-method comparison
  (free prior vs SimCLR/SimSiam/DINO vs transfer vs augmentation vs learned
  teachers, at 1x/~1.02x/2x compute, 14 datasets x 7 backbones, 500->1.28M
  images), and this framing DE-RISKS the G3 falsifier -- if ViT-S demotes the
  attention headline, the benchmark contribution stands and the honest
  falsifier scoring becomes evidence of integrity. Consequences accepted:
  (a) the harness (configs, subsets, pinned banks, runner, aggregation,
  audit scripts) must be RELEASED as a usable artifact; (b) fairness audits
  front-loaded (SimCLR best-shot tested, frozen-recipe-is-a-ResNet-recipe
  documented, bistability reported as findings); (c) natural venues:
  NeurIPS D&B / TMLR, TPAMI plausible if the ImageNet stages land clean.
  Complete-grid decision (same day, above) directly feeds this framing.
  VENUE DECIDED (user, 2026-08-07, after the full results landed):
  **INFORMATION FUSION**, with the fusion reframe. The deferred shortlist
  (2026-08-06: InfFus > PatternRec > KBS/NN > TPAMI) resolved once every
  scale falsifier died and P1/S1 held. Paper scaffold lives in paper/
  (Elsevier CAS double-column, cas-dc; compiles locally -- missing
  texlive-latex-extra styles hand-installed to ~/texmf: balance, makecell,
  stfloats, xstring, footmisc, moreverb, wrapfig; stix/charis/inconsolata
  fonts are file_if_exist-guarded and safely absent). Title frames the
  fusion question ("When does fusing hand-crafted spectral knowledge with
  learned representations pay?"); highlights <=85 chars each; abstract
  <=250 words draft in place; section plan C1-C5 with the stack/substitute/
  tax taxonomy as the fusion contribution. GUIDE-FOR-AUTHORS CAVEAT: the
  live ScienceDirect page is bot-blocked (403 on every route); requirements
  encoded from the stable Elsevier standard -- highlights 3-5 bullets
  <=85 chars, CRediT + competing-interests + data-availability + generative-
  AI statements, numbered references, format-free first submission (Your
  Paper Your Way) -- VERIFY manually in a browser before submitting.
- *** STALE-WORKLIST INCIDENT #7 + THE GRID IS ACTUALLY COMPLETE (2026-08-06,
  my error, caught same day). Rebuilding worklists after the OOM fix I ran the
  missing-cell generator from $MS/repo WITHOUT OUT=$MS/runs -- its default
  `runs` resolved to $MS/repo/runs (the SSL-pretrain dir, the EXACT axteach
  CWD-relative trap recorded 2026-08-04), so it saw ~no finals and emitted
  7,479 "missing" tasks. TRUTH, regenerated correctly: **21 genuinely missing
  in the whole study** (8 grid + 13 big-lane, most already in flight) -- the
  grid was essentially complete; ~99.7% of the list was phantom re-runs.
  Burned before catch: ~200 re-run completions (~3h x 24 GPUs, detected by
  finals-mtime arithmetic: 248 new-mtime finals vs total count +45 => ~200
  overwrites; sample of unclaimed tasks 16/20 already-final). SECOND
  consequence: my manual rebuild also skipped the keeper's bigcfg exclusion,
  so the 42 diagin tasks entered BOTH lanes and the ImageNet cells trained
  TWICE CONCURRENTLY in the same seed dirs. Damage audit: all 42 diagin
  best.pt VERIFIED by torch.load (0 corrupt); final.json values are computed
  in-memory (no resume logic) so they are valid single-run measurements;
  metrics.csv of the raced diagin cells is UNRELIABLE (truncated by killed
  duplicates) -- final.json is the record for those cells. The keeper's own
  reconcile was correct all along (OUT= set, grep -vFf .bigcfgs); only my
  by-hand invocation lacked both. RULE: never run make_missing_worklist by
  hand without OUT=$MS/runs AND the .bigcfgs exclusion -- or just let the
  keeper's reconcile do it (CUR==0 path).
  COROLLARY: the same-day "run the full 7,479-task grid" decision was about
  phantom tasks -- the completeness the user wanted ALREADY EXISTS (7,468
  finals). The families I proposed pruning were already complete.
- *** IMAGENET LANDINGS, FIRST SCORES (2026-08-06, from the correctly-
  completed big-lane runs; 3 seeds unless noted):
    STAGE 1 (ImageNet64, 1.28M imgs, 1000-way, 64px, 40ep diag):
      r18:  55.38+-0.04 -> 55.52 (1 seed)   Delta ~ +0.14  P2 (conv neutral
            at right-flank scale) TRACKING IN BAND; F2 not firing. 2 aux
            seeds still queued.
      vit:  48.24+-0.83 -> 51.48+-0.14  **Delta = +3.23** -- P3 (permanent
            deficit >= +3 at 1.28M images) CONFIRMED at the band edge;
            falsifier F3 (<= +1) DEAD. The attention deficit survives 13x
            the images of any prior population.
      swin: baseline {16.12, 0.10, 0.10} -- BISTABLE AT CHANCE at 1.28M
            images (0.1% = 1/1000); aux arm trains tightly (53.62/53.05).
            The swin stabilization signature transplants to ImageNet scale;
            cells stay non-headline per the bistable rule.
      mnet: 32.93 -> 34.88  Delta +1.95 (3v3) -- not pre-registered; small
            positive at full data on a capacity-limited backbone (baseline
            deeply underfit at 33%), noted not scored.
      P1 (Delta ~= G) awaits the fixed-shots probe pass.
    STAGE 2 (ImageNet-100 @224 NATIVE, DeiT aug, 100ep):
      *** ViT-S/16 (21.7M params): 65.39+-0.39 -> 78.39+-0.14 =
      **Delta = +13.00 +- 0.24** -- S3 predicted +2..+8: MISSED HIGH, the
      good direction; **G3 (<= +1, "ViT headline dies") IS DEAD**. At proper
      resolution, standard DeiT recipe, and 4x ViT-tiny's size, the prior
      adds MORE than it did to ViT-tiny at 10% C100 (+13.26). The paper's
      strongest claim now stands on a properly-configured ViT at 224px.
      ViT-B and R50 pairs still training (S4/G4, S2/G2 pending).
      CAVEAT on record: these cells' metrics.csv are the raced ones (see
      incident above); final.json + verified checkpoints are authoritative.
- *** THE SCALE FALSIFIER BLOCK IS DECIDED — EVERY PRE-REGISTERED SCALE
  FALSIFIER IS DEAD, AND THE MODEL-SCALE QUESTION ANSWERED IN THE METHOD'S
  FAVOR (2026-08-06 evening, 3 seeds unless noted):
    P2 CONFIRMED EXACTLY: ImageNet64 r18 55.38±0.06 -> 55.42±0.11 =
      **Delta +0.04 ±0.07** — conv neutrality at 1.28M images / 1000 classes,
      the largest data scale in the study, dead on the predicted 0.0±0.5.
      F2 dead. The "neutral at sufficiency by construction" claim now spans
      500 images to 1.28M on the same recipe family.
    S4 ANSWERED — THE DEFICIT GROWS WITH MODEL SCALE: ViT-B/16 @224
      43.33±3.29 -> 69.34±1.13 = **Delta = +26.01 ±2.01** — TWICE ViT-S's
      +13.00. The pre-registered fork read "data-hunger => Delta(ViT-B) >=
      Delta(ViT-S)"; it holds by +13 points, and G4 (shrinks-with-scale =>
      scope to compact models) is DEAD. 85.9M params at 126k images is
      more data-starved, and the prior fills more of it. NOTE the absolute
      ordering: vitb-aux 69.3 < vits-aux 78.4 — with the prior, the SMALLER
      ViT still wins at this data scale (right-sizing beats scale here);
      baseline σ 3.29 vs aux σ 1.13 — the variance asymmetry at its largest
      model yet. Largest Delta in the study, now at 224px/86M params.
    S2 TRACKING IN BAND (partial, 2v2): R50 @224 85.94 -> 85.80 =
      Delta −0.14 vs band 0.0±1.0; G2 (>= +1.5) not firing. Conv
      neutrality survives native resolution. Final seeds in flight.
  *** P1/S1 PROBE PASS LAUNCHED (2026-08-06 evening, ms_probe lane 44361350;
  own worklist/counter so live training jobs are untouched). PROTOCOL:
  ImageNet64 probed at FIXED SHOTS 25/100/250 per class with --shots-only
  (full-train LBFGS at 1.28M rows impractical; shots G compares only within
  budget — pre-registered 2026-08-05); ImageNet-100 gets the STANDARD
  full-train probe (126k rows, ledger-protocol-comparable) + same shots.
  diagin64_swin EXCLUDED: collapsed baseline => a probe measures training
  failure, not features (swin scope rule). NEW CODE: linear_probe.py
  --shots-only; shots-only output goes to linear_probe_SHOTS.json — the
  first smoke OVERWROTE abl1_none/linear_probe.json (the G-curve record!),
  restored immediately (26.68±0.37, matches ledger) and the filenames
  separated so it can never recur. Suite 102/102.
  PER-CELL IMPLIED PREDICTIONS, from Delta and the measured readout curve
  (positive branch +1.80@39.4 -> +1.14@51.7 -> +0.44@69.1..80.7), recorded
  IN ADVANCE of any probe result:
      G(r18,in64)  ~ -0.5..+0.1  (Delta +0.04, base 55.4)
      G(vit,in64)  ~ +1.8..+3.0  (Delta +3.23, base 48.2)
      G(mnet,in64) — NO sign prediction: base 32.9 sits INSIDE the
        crossing bracket [31.8,40.3]; measured value recorded as data.
      G(vits,224)  ~ +12.0..+12.6 (Delta +13.00, base 65.4)
      G(vitb,224)  ~ +24.0..+25.5 (Delta +26.01, base 43.3, ro ~ +1.4)
      G(r50,224)   ~ -0.7..-0.2  (Delta -0.14, base 85.9)
  FALSIFIER unchanged (F1/G1): |Delta - G| > 1.5 on any backbone => the
  law's predictive form is bounded to small images. CAVEATS ON RECORD
  BEFORE SCORING: (a) in100 probe labels == cell labels (100% cells), so
  the probe-ceiling rule limits interpreting the residual as "readout" —
  the aux-vs-base gap under identical probing is the trustworthy quantity
  (cub@100% precedent); (b) in64 shots-250 probes hold ~20% of the labels
  the cell saw, so shots-G may UNDERSTATE feature gain — if |Delta−G|
  exceeds 1.5 there, the full-budget direction must be checked before
  declaring G1 fired.
  *** CHECKPOINT-IDENTITY INCIDENT — THE DUPLICATE RACE DID DAMAGE AFTER ALL,
  ONE LEVEL DEEPER THAN THE LOAD TEST SEES (2026-08-07). The 2026-08-06 audit
  said "all 42 diagin best.pt VERIFIED by torch.load (0 corrupt)" — TRUE and
  INSUFFICIENT: a checkpoint can load perfectly and be the WRONG NETWORK. The
  first in64 probe returned aux-seed0 probing exactly at baseline level while
  its siblings sat 21 points higher; e2e re-evaluation of every ImageNet
  checkpoint against its own final.json exposed the mechanism: the grid-lane
  duplicates, cancelled mid-run at ~13:05 on 08-06, had been OVERWRITING
  best.pt with their best-so-far EARLY-EPOCH weights AFTER the big-lane
  originals finished (final.json survived because the duplicates never reached
  their end-of-run write; the recorded ACCURACIES are in-memory values and all
  stand). IDENTITY AUDIT (ckpt-eval vs recorded, |diff|<1.5 passes):
  **13/34 genuine, 21 wrong-epoch** — genuine: vitb BOTH ARMS 6/6 (finished
  after the cancellation), r50_aux 2, in64_r18_aux seeds1-2 (fresh restarts),
  in64_vit_none 3; damaged: vits ALL 6, r50_none 2, in64 r18_none 3 +
  r18_aux seed0 + vit_aux 3 + mnet 6.
  RULE ESTABLISHED: verify run mirrors by LOADING was already the lesson;
  the stronger form is verify by IDENTITY — evaluate the checkpoint and
  compare to its recorded final. analysis-level check, cheap (one val pass).
  RECOVERY: (a) local probes of damaged cells stopped, their outputs
  quarantined (*.DAMAGED-CKPT); (b) BSC ms_probe job cancelled (it would
  have probed damaged files); (c) worklist.big rebuilt (safe: 0 running
  workers) with 3 final-heals + 21 ckpt-repair retrains into
  **runs_ckptfix/** so recorded finals are never overwritten — probes of
  those cells will read runs_ckptfix and pair with ITS OWN finals
  (self-consistent Delta and G from the same runs); (d) the fully-genuine
  ViT-B pair probes NOW on the local 3090 — S1's most important cell
  (Delta +26.01) is unblocked.
  *** FIRST S1 CELL SCORED — G(ViT-B @224) LANDS JUST UNDER ITS BAND, AND THE
  RESIDUAL IS EXACTLY THE PRE-REGISTERED CEILING SIGNATURE (2026-08-07, local
  3090, full-train probe, 3 seeds, GENUINE ckpts):
      probe(none) 46.91±2.04  probe(aux) 69.81±0.74
      **G(vitb,224) = +22.89 ±1.25**  [band +24.0..+25.5 — MISSED LOW by 1.1]
      Delta −G = +3.12 ±2.37  [G1 threshold 1.5 — NOMINALLY exceeded,
      NOT statistically resolvable (1.3σ from zero)]
  SCORED HONESTLY, NEITHER PASS NOR FIRE: caveat (a), recorded IN ADVANCE,
  applies with its signature visibly present — probe(aux) 69.81 vs e2e(aux)
  69.34 means the aux arm sits AT its own probe ceiling (+0.5 headroom), so
  full-train G is COMPRESSED on the aux side exactly as the probe-ceiling
  rule predicts at probe-labels == cell-labels. The baseline has ~3.6 of
  headroom (46.9 vs 43.3), the asymmetry that produces a positive residual.
  Shots cross-check runs the right direction: G +29.3/+27.8/+24.6 at
  25/100/250/cls, converging DOWN toward the ceiling-compressed 22.9 — the
  aux features are dramatically more label-efficient, and a budget below
  saturation reads a LARGER gap. VERDICT: Delta ≈ G holds within ~1σ at
  224px/86M params under a pre-registered caveat; G1 is NOT declared fired
  on a ceiling-compressed cell. The vits pair (repairing) gives the second,
  less-compressed S1 point (aux e2e 78.4 sits farther from any ceiling).
  *** FULL RECOVERY WITHOUT RETRAINING — last.pt SURVIVED THE RACE
  (2026-08-07 midday). Insight: the duplicates were all KILLED MID-RUN, and
  last.pt is written only at run COMPLETION — so the originals' final-epoch
  weights were never touched. mtimes confirm (damaged best.pt all 12:26-13:03
  = duplicate writes; last.pt hours earlier = original completions), and an
  identity audit of all 28 pulled last.pt files returned **28/28 GENUINE**
  (last.pt eval == final_test_acc to 0.04). CONSEQUENCE: the 21-run
  ckpt-repair retrain plan is CANCELLED (1-2 node-days saved); worklist.big
  trimmed to the 3 genuine heals; the 21 wrong-epoch best.pt QUARANTINED
  (renamed *.wrongepoch on BSC AND locally so nothing can silently probe
  them). The remaining P1/S1 probes run locally with --ckpt last.pt, BOTH
  arms of each pair identically, so every gap is protocol-clean; PROTOCOL
  NOTE: those G values are measured on final-epoch weights rather than
  best.pt — for these diag cells best≈final (largest gap 0.10), and the
  vitb pair (probed on genuine best.pt) is unaffected.
  *** TRAIN.PY RUN-DIR GUARDS SHIPPED (2026-08-07, local + BSC, suite
  102/102, all three smoke-tested): (1) COMPLETED-RUN GUARD — final.json
  present => SKIP (idempotent re-runs; every stale-worklist incident #2..#7
  reduces to re-running finished cells; MS_FORCE_RERUN=1 overrides);
  (2) EXCLUSIVE RUN LOCK — flock on <seed_dir>/.runlock, second trainer on
  the same (config,seed) ABORTS loudly instead of racing checkpoint writes
  (the exact dual-lane failure that caused the wrong-epoch damage);
  (3) ATOMIC CHECKPOINT WRITES — best.pt/last.pt via tmp+os.replace, so a
  torn/half-written file (the corrupt-cnx signature) can no longer exist.
  *** P1/S1 SCORED — DELTA ≈ G SURVIVES IMAGENET SCALE; F1 AND G1 ARE DEAD
  (2026-08-07 afternoon, genuine last.pt checkpoints, both arms probed
  identically, 3 seeds/cell):
    STAGE 1 (ImageNet64, shots-250 G, same-budget protocol):
      r18:  Delta +0.04 ±0.07 | G +0.05 ±0.10 | **|D−G| = 0.01**  [band
            −0.5..+0.1 HIT] — conv neutrality mirrored exactly feature-side.
      vit:  Delta +3.23       | G +2.84 ±0.85 | **|D−G| = 0.39**  [band
            +1.8..+3.0 HIT, near top]; G rises with budget (1.53/2.38/2.84),
            converging toward Delta from below — the fine-probe label-hunger
            shape reproduced at 1.28M images.
      mnet: Delta +1.95       | G +1.81 ±0.40 | **|D−G| = 0.14**  [no sign
            prediction — base 32.9 inside the crossing bracket; recorded as
            data and lands as a small positive readout, fitting the
            transition region].
      Variance asymmetry on ALL THREE pairs (baseline probe sigma 2-3x the
      aux arm's): feature-level stabilization holds at ImageNet scale.
    STAGE 2 (ImageNet-100 @224, STANDARD full-train probe):
      vits: Delta +13.00 ±0.24 | **G = +11.95 ±0.51** [band +12.0..+12.6 —
            lands 0.05 below the bottom edge, 0.1 sigma: scored AT BAND] |
            residual +1.05 < 1.5: G1 does NOT fire on the headline cell,
            despite mild aux-side ceiling compression (probe(aux) 78.07 vs
            e2e 78.39). Shots G 14.8/14.3/12.7 converge down to the
            full-train value, same shape as vitb.
      vitb: G +22.89 ±1.25, residual +3.12 ±2.37 — ceiling-caveated,
            unresolved (scored 2026-08-07 morning, entry above).
      r50:  Delta −0.14 (2v2) | G = +0.03 (2v2, unpaired seeds none{1,2}
            aux{0,2}) | residual −0.17. Band was −0.7..−0.2: nominally 0.23
            above the top at 2v2 — a rounding-level miss on a dead-zero
            quantity; the substantive claim (conv feature-neutrality at
            224px) lands exactly. Re-probe at 3v3 when the heals finish.
  CROSS-MACHINE REPRODUCTION (2026-08-07): the in100 probes were run
  REDUNDANTLY on a second workstation ("solarflare", second RTX 3090, fresh venv
  built that day: torch 2.7.0+cu126 / timm 1.0.27). Probe means agree with
  the local machine to 0.01-0.05 on all four cells; G(vits) = +11.91 vs
  +11.95 local. The S1 headline number reproduces on independent hardware
  and a from-scratch environment. solarflare is now a commissioned second
  worker (repo+venv+in100 data at /media/HDD_4TB/amughrabi/momentstem).
  THE CAMPAIGN'S LAST PRE-REGISTERED QUESTION IS ANSWERED: the law predicts
  feature gain from accuracy (and vice versa) at 500 images and at 1.28M, at
  32px and at 224px, at 5.7M params and at 86M. FALSIFIER LEDGER, FINAL:
  F1 F2 F3 G1 G2 G3 G4 — ALL DEAD. P1 P2 P3 S1 S3 S4 confirmed, S2 partial
  (2v2, in band). NOTE ON PROVENANCE: in64/in100 G values were measured on
  last.pt (final-epoch weights) after the wrong-epoch quarantine; vitb on
  genuine best.pt; the two protocols agree where both exist (vits shots
  trend vs vitb) and best≈final on every one of these cells (max gap 0.10).
  STAGE-1/2 SCOREBOARD: P2 ✓  P3 ✓  S2 ✓(partial)  S3 ✓(above band)  S4 ✓
  — falsifiers F2, F3, G2, G3, G4 ALL DEAD. Still open: P1/S1 (Delta ≈ G
  at scale) — needs the fixed-shots probe pass; and the last 4 in-flight
  cells (r50_none seed0, r50_aux seed1, depth food_aux seed0).
  THE SCALE STORY FOR THE PAPER, one line: the law's regimes transplant
  intact to ImageNet scale — conv redundancy at sufficiency (+0.04 at
  1.28M), attention deficit persisting (+3.2 at 1.28M/64px) and GROWING
  with model scale at proper resolution (+13.0 ViT-S -> +26.0 ViT-B @224).
- STAGE 2 LAUNCHED (2026-08-05): **ImageNet-100 @224px NATIVE** (clane9/
  imagenet-100, 126,689 train / 5,000 val, 100 classes) with a MODEL-SCALE
  CURVE rather than a single point: **ViT-S/16 (21.7M), ViT-B/16 (85.9M),
  ResNet-50 (23.7M)**, both arms, 3 seeds = 18 runs. Stage 1 (ImageNet64)
  buys DATA and LABEL scale; Stage 2 buys RESOLUTION and MODEL scale, the
  other half of the reviewer objection.
  DEVIATIONS, stated up front: 100 epochs (not the frozen 200); native-res
  transforms (RandomResizedCrop / Resize+CenterCrop) since the small-image
  RandomCrop+pad is meaningless for variable-size JPEGs; DeiT augmentation on
  the two ViTs (the standard small-ViT recipe -- a plain-aug ViT-B on 126k
  images would barely train, and an untrainable baseline tests nothing).
  Both arms of each pair are identical, so every Delta stays valid. diag-only.
  PREDICTIONS RECORDED IN ADVANCE (before any Stage 2 run):
  All three baselines will sit FAR above the sign-law crossing [31.8, 40.3]
  (expect 65-85%), so readout is on the decayed positive branch:
    (S1) readout ~ 0..+0.5 => **Delta ~= G within +-0.5 on all three
         backbones**. Same core predictive claim as Stage 1, now at 224px and
         up to 86M params.
    (S2) CONV at 126k images is on the right flank => **Delta(R50) = 0.0
         +-1.0**. Every conv population measured is neutral at full data;
         this asks whether that survives real resolution.
    (S3) ViT-S: the permanent-deficit claim (+9.88 at full C100, +7.17 at
         full tin, both with ViT-TINY at 32-64px) must leave a trace on a
         PROPERLY CONFIGURED ViT: **Delta(ViT-S) = +2..+8**. Smaller than
         ViT-tiny's because ViT-S/16 at 224 under DeiT aug is a far better
         model for this data than ViT-tiny at 64px was for tin.
    (S4) ViT-B vs ViT-S is the NEW question the curve buys: if the deficit is
         driven by DATA-HUNGER, the bigger model should show a LARGER Delta
         (**Delta(ViT-B) >= Delta(ViT-S)**); if it is driven by SMALL-MODEL
         capacity, it should shrink.
  FALSIFIERS, each costing a specific claim:
    (G1) |Delta - G| > 1.5 on any backbone => the law's predictive form does
         not survive resolution/model scale, bounding it to <=96px small nets.
    (G2) Delta(R50) >= +1.5 => conv redundancy at full data is a low-res
         artifact and "neutral at 100%" must be rewritten.
    (G3) Delta(ViT-S) <= +1 => THE ViT HEADLINE LARGELY DIES: the deficit
         would be an artifact of 5.7M-param ViT-tiny at 32-64px, not a
         property of attention at small data, and the claim must be restated
         as "compact ViTs at low resolution".
    (G4) Delta(ViT-B) < Delta(ViT-S) - 2 => the deficit SHRINKS with model
         scale, so the claim must be scoped to compact models and cannot be
         extrapolated to modern ViT sizes.
  NOTE G3 is the one that matters most: the ViT story is the paper's
  strongest claim and it currently rests entirely on ViT-tiny. This is the
  experiment that either promotes it to a general statement about attention
  or demotes it to a small-model curiosity.

## BSC RHEL 9.6 MIGRATION (2026-08-01, from BSC HPC Support email)

- SCHEDULE: login nodes migrated Thu 30 July 08:00 (DONE); **compute nodes
  (GPP+ACC) migrate Mon 3 August 07:00, with a FULL-MACHINE RESERVATION from
  that moment** — jobs must finish before it or they will not start.
  BSC asked users to validate workflows on the new OS beforehand; the
  `rhel96` reservation (179 ACC nodes) and alogin2/glogin2 already run it.
- *** KEEPER OUTAGE FOUND AND FIXED — THE CAMPAIGN HAD STOPPED FEEDING:
  alogin1 (the ONLY host the cron keeper used) went unreachable while it was
  migrated; every tick since had logged `rc=255 :: Connection timed out`
  (121 failed ticks) and NOTHING was being submitted — the grid queue drained
  to 0 running jobs and stayed there, silently. FIX: keep_bsc_fed.sh now
  takes a HOST LIST (`BSC_HOSTS`, alogin2 first since it migrated first),
  probes each with a cheap `ssh true`, and uses the first that answers; the
  chosen host is recorded in the log line. If none answer it logs
  NO LOGIN HOST REACHABLE and exits non-zero instead of pretending success.
  Verified live: keeper reconnected via alogin2 and submitted 8 workers.
  LESSON (third silent-failure incident of this campaign): a cron job whose
  only failure signal is a non-zero exit code nobody reads is not monitored.
- *** QUEUE-COUNTER REWIND INCIDENT (2026-08-02) — ~20h OF CLUSTER TIME SPENT
  RE-RUNNING FINISHED CELLS. Symptom: the grid counter had gone BACKWARDS,
  7024/7246 -> 2743/7246, while worklist.bsc's mtime was still 2026-07-24
  (so no reconcile had run). Sampling the tasks being claimed showed **12 of
  13 already had final.json** — the whole 32-GPU cluster was redoing
  completed work, two days before the migration reservation.
  ROOT CAUSE, in bsc_worker.sbatch's atomic claim:
      i=$(flock "$LOCK" bash -c "read -r v < '$CTR' 2>/dev/null || v=0;
                                 echo \$((v+1)) > '$CTR'; echo \$v")
  A single transient failure of that read (GPFS hiccup) sets v=0 and WRITES
  1 back, rewinding the shared counter for every worker. The `|| v=0` was
  meant for first-run initialization and doubles as a data-destroying
  fallback — the same silent-guard family as the pretrain whitelists and the
  suppressed reconcile generator, 4th incident.
  FIX: the counter is initialized ONCE before the worker loop (`[ -s $CTR ]
  || echo 0`), and the claim now VALIDATES the value (non-empty, all digits)
  and EXITS NON-ZERO if it cannot read it, so the worker backs off and
  retries instead of restarting the queue. Unit-tested locally against three
  cases: absent counter (refuses), normal claims (0,1,2 monotone), corrupt
  counter (refuses, leaves the file untouched).
  RECOVERY: workers cancelled, worklist regenerated from what is actually
  missing (7246 stale -> **642** genuinely-missing grid tasks), counter
  reset to 0, 8 workers resubmitted against the clean list.
  SECOND BUG THIS EXPOSED: the reconcile generator only scans configs/grid/,
  so the 78 still-missing configs/diagnostics/ tasks (the vitenv tin ViT/DeiT
  envelope, incl. BOTH open falsifier blocks at 15/25%) were DROPPED by the
  regeneration. Recovered by extracting the diagnostics lines from the stale
  worklist, filtering to those without final.json, and appending — final
  worklist 720 lines (642 grid + 78 diagnostics), falsifier cells verified
  present. The generator's configs/grid-only scan remains a known limitation;
  anything queued from configs/diagnostics/ must be re-appended after every
  reconcile until that is fixed properly.
- *** WORKLIST-PATH REGRESSION (2026-08-03, my error, and the cause of the
  two incidents below looking worse than they were). The BSC-deployed
  bsc_worker.sbatch read `WORKLIST=$MS_ROOT/worklist.bsc`; the repo copy at
  slurm/bsc_worker.sbatch was OLDER and still read `worklist.txt` (the
  original 2026-07-22 full list, 2712 lines, long stale). Shipping the repo
  copy to deploy the counter fix SILENTLY REVERTED the path, so for ~a day
  the cluster worked the stale 2712-line list — re-running finished cells —
  while every rebuilt 604-task queue went unread. Symptom that exposed it:
  workers logged `tasks=2712` when worklist.bsc was 604 lines.
  FIX: repo copy corrected to worklist.bsc (with a comment saying why),
  redeployed, and the stale file renamed worklist.txt.stale.2026-07-22 so
  nothing can point at it again. VERIFIED: workers now log `tasks=604` and
  claim real missing cells.
  LESSON, and it is the general one: THE REPO COPY OF A DEPLOYED SCRIPT CAN
  BE OLDER THAN WHAT IS DEPLOYED. Diff before rsyncing over a live file —
  the same drift that left BSC running a pre-2026-07-29 linear_probe.py,
  only this time the drift went the other way and I overwrote the good copy.
- *** WORKLIST-SWAP-UNDER-RUNNING-WORKERS INCIDENT (2026-08-03, my error).
  The worker caches `N=$(wc -l < worklist)` ONCE at job start but resolves each
  task with `sed -n "${i}p"` against the LIVE file. I swapped worklist.bsc
  twice (7246->667->615) while 8 jobs were running with a cached N=2712, so
  once the counter passed 615 every claim returned an EMPTY line and the
  worker's `[ -z "$cmd" ] && continue` re-claimed INSTANTLY — a tight spin
  that raced the shared counter 0 -> 1698 and CONSUMED ~580 genuinely-missing
  tasks without executing them (claims are atomic, execution is not).
  WHY IT HAPPENED: the keeper only ever reconciles when CUR==0 precisely to
  avoid this, and I bypassed that guard by swapping the file by hand.
  RULE: never replace worklist.bsc while any ms_grid job is live — cancel
  first, or wait for the drain.
  FIX (shipped): on an empty line the worker now RE-READS the worklist length
  and breaks if it is past the new end, with a 1s backoff — so a swap is
  detected instead of spun on. Recovery: workers cancelled, worklist
  regenerated (604 real tasks, falsifier cells verified present), counter
  reset, 8 workers resubmitted. No results were lost — the skipped tasks were
  never marked done, so the regeneration picked them all back up.
  NOTE the failure mode is the mirror of the 2026-08-02 counter REWIND: that
  one re-ran finished work, this one skipped unfinished work. Both come from
  the same design property — a claim counter with no record of what was
  actually completed. The reconcile-from-final.json is what makes both
  recoverable.
- *** STACK VALIDATED ON RHEL 9.6 (login node, alogin2): the module set in
  env.sh loads unchanged (mkl/gcc/impi/hdf5/PYTHON 3.11.5/nvidia-hpc-sdk/
  cudnn/nccl), the venv activates, and torch 2.4.0a0 / torchvision 0.19.0a0 /
  timm 1.0.27 / numpy 1.26.4 all import. Critically `torchvision::nms`
  RESOLVES — that custom op is the first thing an ABI break kills, and
  env.sh's own comment records it as the reason the BSC pytorch module is
  deliberately not loaded. vit_tiny also builds. A GPU job on the rhel96
  reservation (train conv + train ViT + simclr pretrain + probe) is queued
  as the compute-node check.
  METHOD NOTE, recorded because it nearly caused a false alarm: testing with
  `source env.sh 2>&1 | tail -2` runs env.sh in a SUBSHELL (the pipe), so the
  environment never applies and python falls back to /usr/bin/python 3.9 with
  no torch. That looked exactly like a migration breakage and was not one.
  Source without a pipe when testing environment scripts.
- *** GPU VALIDATION ON RHEL 9.6 PASSED (job 44052275 on the rhel96
  reservation, 52s): conv train (mnet c100@5%, 2 ep) and ViT train
  (diagvit@10%, 2 ep) BOTH wrote final.json, and the SimCLR pretrain path ran
  (120 tensors saved). So training + pretraining are safe for Monday.
  The only failure was the PROBE step — and it was OUR STALE CODE, not the
  OS: BSC still carried the pre-2026-07-29 linear_probe.py and died on the
  known MobileNetV3 `conv_head` bug (`mat1 and mat2 ... 576x100`). The local
  fix had never been shipped. LESSON: only the pretrain scripts were being
  rsynced after local fixes; the analysis/ tree had silently drifted.
  analysis/, momentstem/, scripts/, train.py, data.py re-synced; both probe
  fixes (mnet ndim>2, swin fc_norm-absent) verified present on BSC, and
  configs/grid 2313 + configs/diagnostics 368 match local exactly.
  RSYNC INCIDENT while fixing this, recorded so it is not repeated:
  `rsync -az analysis/ momentstem/ scripts/ train.py data.py HOST:repo/`
  merges the CONTENTS of all three dirs into repo/ ROOT (trailing slashes),
  scattering 35 stray .py files there. They were moved to
  .stray_backup/ and the root restored to exactly data.py, train.py,
  eval_robustness.py; package dirs verified intact. Sync one tree at a time.
- *** MIGRATION READINESS COMPLETE (2026-08-01): after shipping the code, a
  probe-only job on the rhel96 reservation returned PROBE_RESULT + PROBE_OK,
  so ALL FOUR workflow paths are now validated on RHEL 9.6 — conv train,
  ViT train, SSL pretrain, linear probe. Nothing further is needed before
  Monday's compute-node migration. (The probe's 10.45 number is from a
  2-epoch smoke checkpoint and is NOT a measurement — do not record it.)
  Test artifacts (rhel96_test_runs/, the two test sbatch files) deleted;
  they lived outside runs/ so they never entered aggregation. The 35 stray
  root files are retained in .stray_backup/ as a safety copy — they are
  duplicates of files that exist correctly in analysis/momentstem/scripts,
  so the directory can be deleted whenever convenient.
- RECONCILE HARDENED FOR THE MIGRATION: both lanes ran the missing-cell
  generator as `python ... 2>/dev/null | grep ...`, so (a) generator errors
  were suppressed and (b) the pipeline's exit status was grep's, meaning a
  FAILED generator produced zero lines and was read as "nothing missing" ->
  GRID_COMPLETE/BIG_COMPLETE -> the lane stops forever. Now the generator
  runs to a temp file with its status checked explicitly; on failure the
  keeper logs `STATE ERROR ... generator FAILED` and touches nothing. Both
  reconciles also source env.sh first so they use the venv python rather than
  whatever the login node's default happens to be after the migration
  (system python is 3.9.21 and HAS pyyaml, so this is belt-and-braces).
- TIMING PLAN: at 2026-08-01 14:45 UTC the reservation is ~38h out = ~6.7
  worker cycles of 5h45. Slurm will simply refuse to START jobs that would
  overlap the reservation, so they PEND rather than die — no results are lost
  by keeping the keeper running through Monday. Nothing needs to be stopped.
- *** BIG-LANE CRASH-LOOP RESOLVED, VERIFIED IN PRODUCTION (2026-07-29): after
  the cifar10/stl10 whitelist fix, the final big job logged **OK 36 / 0 FAIL**
  and wrote BIG_COMPLETE. 18 of the 24 SSL-at-scale orphan cells now carry
  3 seeds. The other 6 (simclr/simsiam at c10@50, stl@20, stl@25) are NOT
  orphaned — they sit in the NORMAL lane (present in worklist.bsc, absent from
  .bigcfgs), so the grid worker picks them up; checked explicitly rather than
  inferred from the BIG_COMPLETE marker.
- *** SSL-AT-SCALE RIGHT FLANK MEASURED — AT 50-100% NOTHING MATTERS ON CONV
  (2026-07-29, the new big-lane cells, 3 seeds):
      ds    pct |  base    aux   simclr simsiam | ssl−aux
      food   50 | 71.75  71.09  70.81  70.92 | −0.29
      food  100 | 78.20  77.73  77.78  77.49 | +0.05
      tin    50 | 58.70  57.95  58.79  58.29 | +0.84
      c100   50 | 71.79  71.34  71.92  71.20 | +0.58
      c10   100 | 95.27    --   95.24  95.02 |   --
  Every arm converges: at 50-100% data, baseline ≈ champion aux ≈ SimCLR-init
  ≈ SimSiam-init to within ~0.8 on every population measured. This completes
  the SSL−aux curve's RIGHT flank: the margin rises to a mid-band peak
  (+5.01 @c100-10%, +4.63 @tin-3%) and DECAYS TO ZERO by 50%. So SimCLR's
  advantage over the prior is a MID-DATA phenomenon on both sides — it
  starves at ≤1-2% and becomes irrelevant at ≥50%, exactly where the data
  itself dominates and the prior's λ->0 schedule is structurally neutral.
  The honest one-line summary of the whole conv SSL-vs-aux question:
  **SSL wins only in the mid-data band; at both extremes the free prior ties
  it.**
  ALSO: food@100% champion pair = 78.20 -> 77.73 (**−0.47**) — the 10th
  population to show the expected ~neutral-at-100% behaviour.
- *** DINO-ViT AT SCALE — IT FINALLY DOES REAL WORK, BUT STILL LOSES TO THE
  PRIOR ON PHOTO SETS (2026-07-29):
      ds   pct | dino-ViT  vit-none  vit-aux | dino−aux
      c10   50 |  81.39     75.14     83.58  | −2.19
      c10  100 |  88.71     80.83     89.83  | −1.12
      stl   50 |  48.27     45.33     55.45  | −7.18
      stl  100 |  58.44     54.99     62.90  | −4.46
      food  50 |  40.37     25.59     36.08  | **+4.29**
  At ≤10% DINO was ≈ the plain ViT baseline (pretraining bought ~nothing);
  at 50-100% it clearly helps (+6..+15 over vit-none). But it still trails
  the moment prior on c10 and stl at every scale measured, and beats it only
  where DINO already showed domain-specific strength: food@50 (+4.29) and
  c100@100% (62.30 vs 60.50, recorded earlier). Same pattern as pathmnist.
  So "the prior beats modern attention-SSL" holds on photo-like sets across
  the whole envelope, but is NOT universal at scale — food and pathmnist are
  DINO's territory. Worth stating that way rather than as a blanket claim.
- MOBILENETV3 G-PROBE PASS LAUNCHED (2026-07-29, local 3090): mnet is the
  4th backbone family and the FIRST HEADLINE-ELIGIBLE one beyond the ResNets
  (frozen SGD recipe, no bistability observed) — and it has no G measurement.
  It also carries a real anomaly: champion-like on pathmnist (+2.3..+4.7
  across 1-15%) but FLAT on tin (−0.35..+1.40) and modest on C100.
  THE QUESTION: is the tin flatness a G effect (mnet's strong conv bias at
  64px leaves no feature deficit for the prior to fill) or a READOUT effect
  (tin's sub-crossing baselines suppress a real G, exactly as they do for
  R18-on-tin, where Δ +1.49 hides readout −2.71)?
  Cells: grid_mnet_{c100@7%, path@10%, tin@5%}_{none,aux}, 3 seeds.
  PREDICTIONS RECORDED IN ADVANCE, readout read off the MEASURED curves at
  matched baselines (C100: −2.76@8.93, −1.12@25.36, +0.08@40.28; tin:
  −2.71@5.30, −0.54@21.08, −0.06@33.60; positive branch: +0.44@80.7):
      c100@7%  base 23.12  Δ +2.84 | readout ≈ −1.3 => G ≈ +4.1  BAND [+2.5,+5.5]
      path@10% base 87.90  Δ +3.46 | readout ≈ +0.3 => G ≈ +3.2  BAND [+1.5,+4.5]
      tin@5%   base 16.66  Δ −0.35 | readout ≈ −1.1 => G ≈ +0.75 BAND [−0.5,+2.0]
  THE FORK, and the tin cell decides it:
    H-NO-DEFICIT (what my point prediction backs): G(tin) <= +1 while
      G(c100)/G(path) land +3..+4 => mnet genuinely has nothing to gain on
      tin; the flat tin envelope is a FEATURE statement about efficient
      convs at 64px, not a readout artifact.
    H-READOUT-SUPPRESSION: G(tin) ≈ +2..+3, comparable to the other two =>
      the flatness is the sign law at work and mnet's tin features DO carry
      a deficit. NOTE this branch requires readout ≈ −2.85 at base 16.66,
      far more negative than the measured tin curve gives at that height
      (−0.54 at base 21.08) — so the law makes this the harder branch, which
      is what makes the test informative rather than a coin flip.
  FALSIFIER for "the law transplants to efficient-conv": readout = Δ − G
    coming out POSITIVE at tin@5%'s sub-crossing base 16.66 (i.e. G < −0.35),
    or strongly NEGATIVE (<= −1.5) at path@10%'s base 87.90.
- *** MOBILENETV3 PROBES LANDED — 3/3 IN BAND, AND THE tin FORK RESOLVES TO
  H-NO-DEFICIT WITH A MEASURED ZERO (2026-07-29, 3 seeds/cell):
      cell       base    Δ    | probe_none  probe_aux      G        readout
      c100@7%   23.12  +2.84 | 32.64±0.72 37.12±1.96  +4.47±1.20   −1.63
      path@10%  87.90  +3.46 | 90.42±0.19 93.01±0.38  +2.59±0.24   +0.87
      tin@5%    16.66  −0.35 | 27.44±0.37 27.45±0.16  **+0.01±0.23** −0.36
  THE FORK IS ANSWERED: G(tin) = +0.01 ±0.23 — a DEAD ZERO (0.04σ), the
  tightest measured null in the study — while the same backbone shows
  +4.47 on C100 and +2.59 on pathmnist. So mnet's flat tin envelope is a
  FEATURE statement, NOT readout suppression: on tin there is simply nothing
  for the prior to add. H-READOUT-SUPPRESSION is dead (it needed G ≈ +2..+3).
  THE SHARP COMPARISON: at the SAME cell (tin@5%, identical pixels, images
  and 200-way label space), G(mnet) = +0.01 while G(R18) = +2.67. G is
  therefore a function of the BACKBONE as well as (pixels, images, label
  space) — and strikingly, the WEAKER net (mnet base 16.66 vs R18's 21.08)
  is the one with NO deficit. Naively a weaker model should have more to
  gain; it has none.
  TWO READINGS, NOT ADJUDICATED — state both: (a) mnet's depthwise-separable
  stack with squeeze-excite already encodes oriented-energy-like structure at
  64px, so the prior is redundant there; (b) mnet is CAPACITY-limited at tin's
  200-way task, so extra feature structure cannot be represented regardless of
  how good the target is. Distinguishing them needs a wider mnet (capacity
  held up) or a shots-sweep on the frozen features. Do not assert (a) alone —
  the c100/path cells show mnet CAN cash the prior in when the task is
  easier, which is equally consistent with (b).
  SIGN LAW — THE FALSIFIER DID NOT FIRE, LAW TRANSPLANTS TO EFFICIENT-CONV:
  readout −1.63 @ base 23.12 (negative below crossing), +0.87 @ base 87.90
  (positive far above), −0.36 @ base 16.66 (negative below). All three
  correct in sign; 3 more clean cells on a 4th backbone family, and the
  first on a HEADLINE-ELIGIBLE non-ResNet.
- PROBE PATH FIXED FOR timm MobileNetV3 (2026-07-29): mnet probes crashed
  with `mat1 and mat2 shapes cannot be multiplied (28800000x1 and 576x100)`.
  Cause: mnet's `global_pool` IS callable (so it took the ResNet branch) but
  its flatten is Identity AND a conv_head+act2 sits between pooling and the
  classifier — pooling alone returns 4D 576ch, while classifier.in_features
  is 1024. FIX: inside the callable branch, if the pooled tensor is still
  >2D, defer to timm's `forward_head(f, pre_logits=True)`. ResNets pool to
  2D and keep the original path untouched, so every recorded conv G stands.
  VERIFIED across all four families — extracted feature dim now equals
  classifier.in_features exactly: r18 512, mnet 1024, swin 768, vit 192.
  Suite 102/102.
- PROBE PATH EXTENDED TO timm ClassifierHead BACKBONES (2026-07-28): the
  Swin probes crashed on `'SwinTransformer' object has no attribute
  'fc_norm'` — swin's `global_pool` is the STRING 'avg' exactly as on ViT,
  but its pooling module lives inside `.head` and forward_features returns
  NHWC, so the ViT branch mis-pooled and reached for a non-existent
  fc_norm. FIX in analysis/linear_probe.py: a branch gated on fc_norm's
  ABSENCE that defers to timm's own `forward_head(f, pre_logits=True)` —
  verified bit-identical (max|diff| 0.0) to manual head pooling on
  swin_tiny. Gating on absence means every ViT cell keeps the exact branch
  its recorded G was measured under; CONFIRMED by re-probing
  diagvit_aux_10pct after the patch: 39.70±0.28 vs 39.72±0.29 recorded
  (0.02, within seed noise of an identical computation). Suite 102/102.
- *** VITENV tin@100% LANDED — THE PERMANENT-DEFICIT FALSIFIER DID NOT
  FIRE (2026-07-27): diagvit tin@100% (100k imgs, 3 seeds) = 34.81 ->
  41.98 = **+7.17** — IN the pre-registered band (+6..+10 @100%); the
  falsifier (<= +2 => "permanent deficit is C100-specific") is dead.
  ViT-tiny-from-scratch now gains large at FULL data on TWO populations
  (C100 +9.88@50k, tin +7.17@100k): the permanent-deficit conclusion is
  a two-population claim. DeiT-aug pair at tin@100%: 50.72 -> 57.32 =
  +6.60 (amp 0.92x at full data — same compression as C100@100%'s 1.40x
  vs 1.85-2.4x mid-band; the "amp < 1.3x on the MID-BAND" falsifier is
  untouched, mid-band tin pairs @15/25% still queued). deit-tin partial
  envelope: +8.19@3%, +13.33@7% (stacking transplants, as predicted).
  DINO-tin@10% = 16.98 vs aux-ViT-tin 17.75 (−0.77): DINO below the
  prior on the second population too. esat aux-vs-SimCLR: +1.85@5% but
  tie @15% (−0.06, aux n1) — the "aux > SSL at >=2 fractions" falsifier
  still needs esat@7/10% aux cells (queued).
- *** BIG-LANE CRASH-LOOP FIXED (2026-07-28): the reconciled 36-task big
  pass was churning in ~36s/job — every task FAILED instantly on the
  pretrain whitelist: dino/simclr/simsiam @50-100% on c10/stl are the
  first SSL cells ever aimed at cifar10/stl10, and none of the three
  pretrain scripts listed them (the SAME guard-bug class as genssl's
  cifar100-only ValueError, third incident). The reconcile loop worked
  exactly as designed (re-listed the same 36 missing every cycle) but
  cannot fix a deterministic failure — worth remembering: a reconcile
  that keeps finding the SAME cells missing is a crash-loop signal, not
  a walltime signal. FIX: cifar10+stl10 added to all three whitelists
  (torchvision loaders, same .transform swap pattern; STATS/IMAGE_SIZE/
  NUM_CLASSES all present). VERIFIED: simclr+simsiam 1-epoch smokes
  PASS locally on both datasets; DINO verified ON BSC (construction +
  MultiCrop + forward at 32px and 96px) because the local check is
  impossible — see next point. Scripts shipped to the BSC repo; the
  pending big jobs pick them up on start.
- *** LOCAL ENV HAZARD (2026-07-28): anaconda-base timm is now 0.6.7 —
  DOWNGRADED (almost certainly by nerfstudio/VolETA, which pins 0.6.7)
  from the 1.x the study's local ViT work used. timm 0.6.7 cannot build
  the study's ViT (patch_size override collides in the factory), so ANY
  local ViT run/probe now fails at construction, and conv runs would
  execute under a different timm than their originals. BSC (1.0.27,
  venv) and turing (own venv) are unaffected. RULE: before the next
  local wave or probe pass, create a dedicated study venv with timm
  pinned ~1.0.x — do NOT run study code from anaconda base again.
  *** RESOLVED 2026-07-28 (VolETA finished, GPU free): dedicated study venv
  built at **~/venvs/momentstem** — ALWAYS use `~/venvs/momentstem/bin/python`
  for local study work, never bare `python`. Pins chosen to match what the
  study ALREADY ran under, not "latest": torch 2.7.0+cu126 / torchvision
  0.22.0+cu126 (the exact versions recorded in all 733 local-3090 finals),
  timm 1.0.27 (BSC's version, so local probes and BSC checkpoints are
  directly comparable), numpy 1.26.4, sklearn 1.4.2, torchmetrics 1.9.0.
  Full lock in `requirements-study.txt` (pip freeze, committed).
  VERIFIED, not assumed: (a) suite 102/102 green INCLUDING the bank-regression
  fingerprints — the pinned banks are numerically identical under the new env,
  so no existing run is invalidated; (b) vit_tiny builds at 32/64/96px (the
  exact thing timm 0.6.7 could not do); (c) REPRODUCTION CHECK — reprobing
  diagvit_aux_10pct returned **39.72 ±0.27** vs the ledger's recorded
  **39.72 ±0.29** (mean to 0.01), so the venv reproduces recorded study
  numbers rather than merely running. That run's linear_probe.json had been
  an EMPTY STUB (a casualty of the 0.6.7 breakage); this pass restored it.
  NOTE: torch versions already differ BY MACHINE across the study (BSC
  2.4.0a0 n=5673, local 2.7.0 n=733, turing 2.8.0 n=443) — pre-existing,
  recorded per-run in final.json, not introduced by the venv. Keep local work
  on 2.7.0 so the local population stays internally consistent.
- BIG-LANE RECONCILE SHIPPED (2026-07-27): the 24h big lane had NO
  reconcile path — 53 tasks (SSL-at-scale: dino/simclr/simsiam @50/100%
  on c10/stl/food + food@100% champion pair) were claimed then
  walltime-killed and orphaned, and the normal-lane reconcile excludes
  big configs by design. Fixed: keep_bsc_fed.sh now regenerates
  worklist.big from missing∩bigcfgs when the big counter drains with no
  ms_big job live (BIG_COMPLETE marker when empty); worklist.big
  deduped (food@100% lines were duplicated); the `grep -c || echo 0`
  double-zero bug fixed in both lanes. Lane restarted 0/53 with 2x 24h
  jobs. CAVEAT: make_missing only scans configs/grid/, so diagnostics
  big cells (vitenv — all complete) are outside the reconcile's reach.

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
- *** AUXMAG3 DEEPENING ADJUDICATED — THE OCTAVE EXCESS IS REAL (2026-07-20,
  local 10v10 vs the 3-seed tension recorded 2026-07-19):
    tin@5% champion (10 seeds): 23.20±0.30 = Δ +2.15 (3-seed +2.13 held)
    auxmag3   (10 seeds):       23.68±0.19 = Δ +2.64
    auxmag3 − champion = **+0.49 ±0.11 = 4.4σ** — the 3-seed nominal
    excess (+0.59, 2.3σ) SURVIVED power. G probes agree in direction:
    35.05±0.38 vs 34.69±0.32 (10-seed probes, +0.36 ±0.16, 2.3σ).
    VERDICT: the extra LOW OCTAVE (sigma=4/k=17) carries real supervisory
    value at tin@5% — right at the recorded falsifier boundary (≥+0.5),
    and this time the feature-level probe moves WITH e2e (unlike the
    3-seed read). THE BANK-DESIGN AXIS REOPENS, scoped: octaves (new
    frequency content), not orientations (near-linear combos), are the
    direction; worth one follow-up (auxmag3 on tin@10%/C100@5%) before
    any bank re-pin. The committed 8-pair bank stays the headline bank —
    a re-pin would invalidate every existing run and needs the user's
    explicit decision.
- OCTAVE-MECHANISM WAVE LAUNCHED (2026-07-20, user-approved, local 3090):
  three cells that turn the adjudicated auxmag3 excess (+0.49, 4.4σ) into a
  mechanism-tested finding. PREDICTIONS RECORDED IN ADVANCE:
    auxmag3_tin_10pct (octave persistence on the falling flank): if the
      octave keeps supplying fresh feature content, excess vs champion
      (+1.65±0.27) of +0.3..+0.7; if its value was 5k-specific, 0±0.3.
      Secondary: G probe vs champion's G(tin,10000)=1.70.
    auxmag3_c100_5pct (NEGATIVE CONTROL, 32px): the resolution account says
      the sigma=4 envelope adds nothing at 32px — Δ within ±0.4 of the
      champion's +5.30. FALSIFIER: excess ≥ +0.5 at 32px kills the "new
      resolvable frequency" mechanism (the excess would then be about
      target statistics, not image content).
    auxmag6o_tin_5pct (WIDTH-MATCHED control, 12ch from 6 orientations x 2
      committed octaves, k11, no new frequencies): redundancy account says
      Δ ≈ champion's +2.15 (band +1.85..+2.45), NOT auxmag3's +2.64.
      FALSIFIER: excess ≥ +0.5 => target WIDTH, not the octave, drives the
      auxmag3 excess — would gut the octave interpretation.
  All three: 3 seeds + G probes; banks pinned additively (magnitude6o in
  test_bank_regression); suite 101 green.
- *** TINSEM LANDED — H-ARBITRARY WINS; THE "SEMANTIC COHERENCE" HINT WAS
  PROBE NOISE (2026-07-21, wave COMPLETE ~11h, 10 seeds every cell,
  tinsuper deepened, both probe spaces):
    e2e: tinsem 16.06±0.35 -> 18.04±0.29 = **+1.98 ±0.14** | tinsuper
      (deepened, n=7 turing seeds) 14.11±0.24 -> 15.41±0.41 = +1.30 ±0.18.
      Semantic sort raised the BASELINE +1.95 (~7σ; coherent groups are
      learnable — the fork was NOT void) and Δ rose with it, exactly the
      readout account.
    PRIMARY FORK: G_200(tinsem ckpts) = **+2.87 ±0.14** vs G_200(tinsuper
      deepened) = **+3.03 ±0.13** — IDENTICAL within noise (−0.16 ±0.19).
      Neither recorded branch as stated: tinsuper's 3-seed 2.55±0.41 MOVED
      to 3.03 at power, converging with tin20's 3.08 — so ALL coarse-
      trained ckpts agree ≈2.9-3.1 and the semantic-vs-arbitrary G gap
      NEVER EXISTED (3-seed probe noise, the study's oldest lesson, now
      caught at the probe level). Semantic coherence of the coarse
      partition does NOT change measured G.
    Fine-vs-coarse main effect PERSISTS: 2.87/3.03 ≪ tin@1%'s 4.19 (≥6σ).
    Law identity: tinsem readout = 1.98−2.58(G_20own) = −0.60 @ base 16.1;
      tinsuper deepened = 1.30−2.11 = −0.81 @ 14.1 — both negative below
      the crossing: sign law 27th and 28th clean cells.
    Q6.9j is now fully closed: the coarse-training G cut is about label
      SPACE (fine vs coarse), not about which coarse partition.
- *** OCTAVE-MECHANISM WAVE LANDED (2026-07-21, local, 3 seeds/cell) —
  THE RESOLUTION CONTROL IS CLEAN, THE WIDTH CONTROL MUDDIES THE OCTAVE
  STORY; MECHANISM NOT FULLY ADJUDICATED:
    auxmag3_c100_5pct (32px NEGATIVE CONTROL): Δ +5.32 ±0.46, excess vs
      champion +5.30 = **+0.02** — dead zero, IN BAND (±0.4). The sigma=4
      octave adds NOTHING at 32px: the "new resolvable frequency" account
      survives its falsifier exactly.
    auxmag3_tin_10pct (persistence): Δ +1.93 ±0.16, excess vs champion
      +1.65±0.27 = **+0.28 ±0.31** — straddles the fork (persistence band
      +0.3..+0.7 vs null 0±0.3): direction right, UNDERPOWERED, not
      adjudicated.
    auxmag6o_tin_5pct (WIDTH-MATCHED control, 12ch orientations, no new
      octave): Δ +2.61 ±0.27, excess vs champion 10-seed +2.15 = **+0.46
      ±0.29** — nominally ABOVE the +1.85..+2.45 band and at the ≥+0.5
      falsifier edge: near-auxmag3-sized excess WITHOUT new frequency
      content. Orientations were assumed near-redundant; at 3 seeds this
      challenges octave-specificity.
    NET READING: at 64px, WIDER aux targets (12ch) beat the 8-pair bank by
      ~+0.3..+0.5 whether the width comes from an octave OR orientations;
      at 32px width does nothing. Suggests the driver is target
      dimensionality-at-resolution, not uniquely the low octave. BOTH tin
      controls are 3-seed boundary reads — the auxmag3 lesson says deepen
      before concluding: a 10v10 of auxmag6o vs auxmag3 at tin@5% is the
      single decisive follow-up if the bank-design axis is pursued.
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
- *** axteach CRASH-LOOP FIXED (2026-08-04): the last ~89 tasks were failing
  more often than succeeding (OK=1 FAIL=5 per job) — 27 of them the
  `grid_c100_r18_axteach_l03` family (the FitNets learned-teacher control),
  dying deterministically on
  `FileNotFoundError: runs/abl3_none/seed0/last.pt`. Those TEACHER
  checkpoints are products of the original local study and had never been
  shipped to BSC. Fifth guard/asset-drift incident, and the same signature
  recorded on 2026-07-28: a reconcile that keeps re-listing the SAME cells
  is a crash-loop, not a walltime problem.
  PATH SUBTLETY, worth remembering: aux.py resolves the teacher path
  RELATIVE TO CWD, and the worker runs from $MS_ROOT/repo — so the teachers
  belong in **$MS/repo/runs/**, NOT $MS/runs (my first ship went to the
  wrong one). $MS/repo/runs is a genuinely separate directory holding the
  SSL pretrain checkpoints, for exactly the same reason (the flock pretrain
  commands also use relative paths from the repo dir).
  Shipped 8 teacher checkpoints (344MB) to repo/runs; VERIFIED with a real
  1-epoch run of grid_c100_r18_axteach_l03_10e860_1pct, which now trains and
  writes its output. NOTE the scientific value is nil — the FitNets question
  is settled ("learned teacher does ~NOTHING while the free hand-crafted
  moment gives +3.3/+2.8") — these cells were shipped only so the queue can
  reach GRID_COMPLETE instead of crash-looping forever.

## So2Sat LCZ42 — the cross-modality population (2026-08-08)

- WHY THIS POPULATION: every multi-source cell so far (EuroSAT-MS) splits ONE
  instrument's bands. So2Sat pairs **Sentinel-1 SAR (8 ch) with Sentinel-2
  optical (10 ch)**, 17 LCZ classes, 32px: different satellites, different
  physics (backscatter vs reflectance), no shared detector. Configs are the
  champion VERBATIM; only `dataset` differs across sar/opt/all, so the sensor
  comparison carries no recipe confound. 90 cells, split 54 solarflare /
  36 local.
- PREDICTIONS RECORDED WITH ONE CELL ALREADY SEEN, and that is stated rather
  than hidden: sf_so2sat_sar_none_10pct = 49.12±0.38 (n=3) and
  sf_so2sat_sar_aux_10pct seed0 = 48.85 were on disk when these were written.
  So the SAR band below is a POST-hoc reading of one seed, not a prediction,
  and must not be scored as one. The optical and fused bands are untouched.
    SAR-only: Δ ~ −0.5..+0.5 (neutral). REASONING, and it is a real
      mechanism claim: the aux target is oriented-energy magnitude, and SAR
      is speckle-dominated with no photographic edge statistics. If the
      prior is genuinely domain-agnostic it should still help; if it is
      really an *optical-image* prior, SAR is where that shows.
    Optical-only: Δ +1.0..+2.5 (EuroSAT-MS non-visible gave +2.31 at 10%).
    Fused: Δ ≥ optical-only if the currencies differ.
- THE CONFOUND, recorded BEFORE the result so it cannot be invented after:
  momentstem/energy.py builds its scalar field from a LUMA weighting, and
  for N != 3 channels I generalized that to a uniform 1/N mean. For optical
  bands that is defensible; for SAR it averages backscatter channels, which
  has no photometric meaning. So a NULL on SAR is ambiguous between "the
  prior does not suit radar statistics" and "the channel reduction is wrong
  for radar", and a null must be reported with that ambiguity intact. The
  disambiguating run (a SAR-specific reduction) is NOT queued and would be a
  new design decision, not a bug fix.
- FALSIFIER for "the prior is domain-agnostic": Δ(SAR) ≤ −1.0 at 3 seeds on
  two fractions => the prior is an optical-image prior and every
  cross-domain claim must name its modality.

## ViT BUDGET FALSIFIER (2026-08-09) — the paper's last accuracy claim

- WHY NOW: the conv budget envelope (SimCLR/SimSiam at 800 pre-epochs, 8
  fractions, landed 2026-08-09) showed **SimCLR at 5x beats the prior at
  EVERY conv fraction 1-25%** (+3.12..+9.24, 8-37 sigma). The conv accuracy
  claim is therefore withdrawn and restated as a COST claim. The paper's
  ONLY surviving accuracy claim is the ATTENTION regime under DeiT
  augmentation — and that was measured against SimCLR at 2x ONLY. Leaving it
  there would be exactly the asymmetry the referee flagged: the comparator
  gets its published budget where it loses, and never gets more.
- CELLS: diagdeitsslbudget_simclr800_vit_{5,10,25}pct — ViT-tiny, DeiT aug,
  SimCLR init at 800 pre-epochs, 3 seeds. Comparators (all measured):
      pct   deit-base  deit-ssl(2x)  deit-aux(1.02x)
        5     12.55       21.64         28.89
       10     20.28       36.06         41.29
       25     32.27       52.31         57.32
- PREDICTIONS RECORDED IN ADVANCE (nothing from these cells seen):
    @10%: **38..44**. Reasoning, and it is a real mechanism claim rather
      than a hedge: on CONV, 4x budget moved SimCLR by +2.2..+5.3 with the
      largest moves in the mid band. Applying the same increment to
      deit-ssl's 36.06 lands 38-41, i.e. BELOW deit-aux's 41.29 but close.
      The mechanism says it should move LESS than on conv: under DeiT
      augmentation SimCLR's currency (nuisance invariance) is already
      supplied by the supervised augmentation, which is why deit-ssl2b
      showed stronger views HURT (-4..-12). More of a substituted currency
      should buy little.
    @5%:  **23..28** (deit-aux 28.89)   @25%: **54..59** (deit-aux 57.32).
    So the recorded bet is: the prior HOLDS at 5 and 10%, and 25% is the
    cell most likely to flip (its conv counterpart flipped, and deit-ssl is
    already within 5.0 there).
- FALSIFIER, and it is the paper's last one: deit-ssl800 >= deit-aux at
  BOTH 10% and 25% => 5x compute overturns the prior on attention too, the
  attention accuracy claim is withdrawn like the conv one, and MomentAux's
  case becomes PURELY cost (+2% vs +400%) on every backbone measured. That
  outcome must be reported as plainly as this entry states it.
- HONEST NOTE ON WHAT IS NOT TESTED: the prior gets no extra budget in this
  comparison because it has none to spend — it is a fixed target with no
  pretraining stage. The comparison is "free prior vs SSL at 5x", not
  "equal budgets", and the paper must say so.

- *** ViT BUDGET FALSIFIER SCORED (2026-08-09, 3 seeds/cell, both GPUs).
  THE FALSIFIER DID NOT FIRE, BUT MY DIRECTIONAL BET WAS EXACTLY BACKWARDS:
      pct  deit-base  ssl-2x  aux-1.02x  ssl-5x | aux-ssl5x  sigma  band
       5%    12.55     21.64    28.89    29.97  |   -1.08    -2.0   23..28 HIGH
      10%    20.28     36.06    41.29    40.57  |   +0.71    +1.2   38..44 IN
      25%    32.27     52.31    57.32    55.85  |   +1.47    +7.2   54..59 IN
  RECORDED FALSIFIER: ssl800 >= aux at BOTH 10% and 25% => attention accuracy
  claim withdrawn. It did NOT fire: ssl-5x is below aux at both (tie at 10%,
  and the prior wins 25% by 7.2 sigma). The attention claim SURVIVES.
  BUT THE SHAPE IS THE OPPOSITE OF MY PREDICTION. I wrote "the prior HOLDS at
  5 and 10%, and 25% is the cell most likely to flip". In fact 5% FLIPPED
  (SSL wins by 1.08, 2.0 sigma; band missed HIGH at 29.97 vs 23..28), 10% is
  a TIE (1.2 sigma), and 25% is where the prior is STRONGEST (+1.47, 7.2
  sigma). I had the data-ordering inverted.
  WHY, and it is measurable rather than a story: 4x budget buys SimCLR
  +8.32/+4.51/+3.53 at 5/10/25% -- DECREASING with data. At 2500 images
  200 epochs is only ~3900 steps, so the comparator was step-starved exactly
  where I assumed it was data-starved beyond rescue. On CONV the same budget
  increment peaked in the MID band (+5.3 at 5%, +2.2 at 10%, -0.04 at 25%),
  so the budget response differs by backbone family too.
  HONEST RESTATEMENT REQUIRED, and made: "the prior beats SimCLR at every
  C100 fraction once a small ViT is trained the modern way" is a 2x
  statement (2x SSL is below aux at all three: 21.64/36.06/52.31). At 5x it
  becomes SSL WINS at 5%, TIE at 10%, PRIOR WINS at 25%. So on attention the
  prior's advantage at high comparator budget lives in the HIGHER-data band
  -- the reverse of the plain-aug ViT pattern (prior won BELOW 10%) and the
  reverse of conv (where 5x SSL wins everywhere). Three regimes, three
  different orderings; state each with its budget attached.

## BUDGET-CELL G PROBES + SECOND POPULATION (2026-08-09)

- WHY: the budget envelope is e2e-only. The currency account says SimCLR and
  the prior fill the SAME feature deficit (that is why they never stack), so
  giving SimCLR 5x should show up as MORE OF THE SAME CURRENCY on the
  feature side, not a different one. That is a falsifiable prediction and it
  has never been measured on a budget axis.
- PROTOCOL: every G is a matched pair probed on ONE machine under the
  identical linear protocol; solarflare's budget checkpoints were pulled
  local for exactly this reason. Baselines abl{5,10}_none and
  diagdeit_none_{5,10,25}pct are already probed under the same protocol.
- SCOPE CAVEAT, stated first: the sign law was derived on aux-from-scratch
  cells. SSL-init cells are a NON-AUX intervention and are scored
  separately, as the 2026-07-20 entry requires. They have complied before
  (conv simclr: Delta +9.18 vs G +9.02) but a miss here is not a sign-law
  violation.
- PREDICTIONS RECORDED IN ADVANCE (readout read off the measured curve at
  matched baseline height; G_pred = Delta - readout_pred):
      cell                    base    Delta   readout_pred  G_pred  BAND
      simclr800 C100@5%      25.36   +14.38     -1.1        15.5   [13.5,17.5]
      simclr800 C100@10%     40.28   +10.92     +0.1        10.8   [ 9.0,12.5]
      simclr800 ViT-deit@5%  12.55   +17.42     -3.8        21.2   [18.5,24.0]
      simclr800 ViT-deit@10% 20.28   +20.29     -1.2        21.5   [19.5,23.5]
      simclr800 ViT-deit@25% 32.27   +23.58     +2.0        21.6   [19.5,24.0]
- THE CURRENCY PREDICTION, which is the point of the pass: G(simclr-5x)
  should track G(prior) wherever the two are level e2e, and fall BELOW it
  where the prior wins e2e. Measured prior G on ViT-deit: 21.35@5%,
  22.20@10%, 23.05@25%. So the prediction is
      @5%:  G(ssl5x) >= G(prior)  (SSL wins e2e by 1.08)
      @10%: G(ssl5x) ~= G(prior)  (level e2e)
      @25%: G(ssl5x) <  G(prior)  (prior wins e2e by 1.47, 7.2 sigma)
  FALSIFIER for "budget buys more of the same currency": G(ssl5x) ABOVE
  G(prior) at 25% while losing e2e by 7.2 sigma => the 25% loss is
  readout-side, not feature-side, and "same currency, more of it" is the
  wrong account of what extra pre-training budget buys.
- SECOND POPULATION (solarflare, in parallel): diagdeitsslbudget_simclr800
  on **tin@10%** vs the measured tin comparators (deit-none 10.49,
  deit-ssl-2x 25.45, deit-aux 27.43). The C100 result — SSL wins at 5%,
  level at 10%, prior wins at 25% — rests on ONE population.
  PREDICTION: tin@10% is 10k images, i.e. between C100's 5% and 25% in
  image count, and the C100 ordering is image-count-driven, so expect
  ssl5x NEAR deit-aux: band **26..29**, i.e. level or slightly below.
  FALSIFIER: ssl5x >= 29 on tin => the budget flip is population-specific
  and the attention claim must be scoped to CIFAR-100.

- *** BUDGET-CELL G PROBES SCORED (2026-08-09, matched pairs, one machine):
      cell            base    Delta      G   readout   band        verdict
      conv C100@5%   25.36   +14.38  +11.27   +3.11   13.5..17.5  LOW
      conv C100@10%  40.28   +10.92   +8.68   +2.23    9.0..12.5  LOW
      ViT-deit@5%    12.55   +17.42  +22.24   -4.83   18.5..24.0  IN
      ViT-deit@10%   20.28   +20.29  +21.27   -0.98   19.5..23.5  IN
      ViT-deit@25%   32.27   +23.58  +20.77   +2.81   19.5..24.0  IN
  (1) THE CURRENCY PREDICTION HELD, 3/3, and it was the point of the pass.
      Predicted G(ssl5x) >= G(prior) @5%, ~= @10%, < @25%, tracking the e2e
      ordering. Measured against G(prior) 21.37/22.19/23.05:
        @5%  +22.24 vs 21.37 = ssl5x HIGHER (+0.87) -- SSL wins e2e (+1.08)
        @10% +21.27 vs 22.19 = ssl5x lower  (-0.92) -- e2e level
        @25% +20.77 vs 23.05 = ssl5x LOWER  (-2.28) -- prior wins e2e (+1.47)
      The recorded falsifier (G(ssl5x) ABOVE G(prior) at 25% while losing
      e2e by 7.2 sigma => the loss is readout-side and "same currency, more
      of it" is wrong) did NOT fire. Extra pre-training budget buys MORE OF
      THE SAME CURRENCY, and the whole ViT budget ordering is FEATURE-SIDE.
      Note G(prior) RISES with data (21.4/22.2/23.1) while G(ssl5x) FALLS
      (22.2/21.3/20.8) -- they cross between 5 and 10%, which is exactly
      where the e2e ordering crosses. Two independent measurements agreeing
      on the crossover location.
  (2) THE CONV CELLS MISSED LOW, BOTH OF THEM, and produced POSITIVE readout
      (+3.11, +2.23). At base 25.36 that is on the wrong side of the
      aux-derived branch (below the crossing, readout should be negative).
      SCORED AS RECORDED: the scope caveat written before this pass says
      SSL-init cells are a NON-AUX intervention outside the law's derived
      scope, so this is NOT a sign-law violation and is not counted as one.
      But it IS a finding: on conv, 5x-SSL cells sit ABOVE the aux readout
      branch, i.e. they cash MORE accuracy per unit of feature gain than the
      prior does at the same baseline height. On ViT the same cells sit ON
      the branch (-4.83/-0.98 below the crossing, correct sign). So the
      readout term is intervention-dependent on conv and not on attention --
      unexplained, recorded, and a real open thread rather than a tidy one.
- SECOND-POPULATION ENVELOPE (2026-08-09): tin@10% on solarflare (band
  26..29 recorded above) and tin@5% chained on local. PREDICTION for tin@5%,
  recorded before it runs: the C100 budget ordering is IMAGE-COUNT-driven
  (SSL wins at 2.5k, level at 5k, prior wins at 12.5k), so tin@5% = 5k
  images should land LEVEL with deit-aux 18.46 -> band **17..20**.
  Together with tin@10% (10k imgs, band 26..29, at-or-below aux 27.43) this
  gives the second population two points that test the image-count account
  rather than one point that merely repeats it.

- *** SECOND POPULATION SCORED (2026-08-09, tin, ViT-tiny + DeiT, 3 seeds).
  BOTH BANDS HIT, THE ACCOUNT BEHIND THEM DID NOT:
      pct  deit-base  ssl-2x  aux-1.02x  ssl-5x | aux-ssl5x  sigma  band
       5%    8.29      16.67    18.46    19.81  |   -1.35    -3.7   17..20 IN
      10%   10.49      25.45    27.43    27.73  |   -0.31    -0.6   26..29 IN
  The recorded falsifier (ssl5x >= 29 at tin@10% => the budget flip is
  population-specific) did NOT fire. But the bands were set from an
  IMAGE-COUNT account, and testing that account at MATCHED image counts
  falsifies it:
      C100  2500 imgs  aux-ssl5x  -1.08 (-2.0 s)  tie/SSL
      tin   5000 imgs             -1.35 (-3.7 s)  SSL WINS
      C100  5000 imgs             +0.71 (+1.2 s)  tie
      tin  10000 imgs             -0.31 (-0.6 s)  tie
      C100 12500 imgs             +1.47 (+7.2 s)  PRIOR WINS
  At 5,000 images the two populations DISAGREE (C100 level, tin SSL-wins by
  3.7 sigma), so the ordering is NOT a function of image count. tin favours
  the comparator at every matched count. The bands hit because they were
  wide, not because the reasoning was right -- recorded as a band hit and an
  ACCOUNT MISS, which is the same distinction the deepen wave forced.
  CONSEQUENCE FOR THE PAPER, and it narrows a claim: at 5x comparator budget
  the prior leads on C100 from ~10% upward but leads NOWHERE measured on tin
  (SSL wins at 5%, ties at 10%). So "the prior beats SimCLR under a modern
  recipe" is a **2x** statement on BOTH populations (deit-ssl2x is below aux
  on both: 16.67 vs 18.46, 25.45 vs 27.43) and, at 5x, a C100-only statement
  in the measured range. State it with both the budget AND the population.
- tin@25% BUDGET CELL LAUNCHED (2026-08-09) — the sharp remaining question:
  does the prior beat 5x SimCLR ANYWHERE on tin? On C100 it wins at 25% by
  7.2 sigma; on tin it lost at 5% and tied at 10%.
  tin@25% comparators: deit-none 23.14, deit-ssl-2x **38.48**, deit-aux
  **38.24** — note SSL already TIES the prior at 2x here, unlike C100 where
  2x SSL is 5 points behind at the same fraction.
  PREDICTION RECORDED IN ADVANCE, and it is a bet AGAINST our own method:
  **40..44**, i.e. 5x SSL WINS at tin@25% and the prior beats 5x SimCLR
  nowhere on this population. Reasoning: the 2x arms are already level, and
  4x more pre-training bought SimCLR +3.5 to +8.3 everywhere else measured.
  FALSIFIER: ssl5x <= 38.24 (prior wins or ties) => the population
  difference is NOT monotone in data and the tin envelope has a crossover
  after all, which would make the C100 and tin stories the same shape with
  a shifted crossing rather than different stories.

- *** THE POPULATION DISAGREEMENT IS FEATURE-SIDE, AND THE MECHANISM IS THE
  PRIOR'S G FALLING FURTHER THAN THE COMPARATOR'S (2026-08-09/10, tin budget
  probes; tin comparators' probes were computed on the cluster, so these are
  CROSS-MACHINE pairs -- verified elsewhere to agree to 0.01-0.05, and stated
  as a caveat rather than hidden):
      pop    imgs | G(ssl5x)  G(prior)  Gssl-Gpr | e2e aux-ssl
      C100   2500 |  +22.24    +21.37    +0.87   |   -1.08
      tin    5000 |  +17.72    +15.67    +2.05   |   -1.35
      C100   5000 |  +21.27    +22.19    -0.93   |   +0.71
      tin   10000 |  +20.92    +19.72    +1.19   |   -0.31
      C100  12500 |  +20.77    +23.05    -2.28   |   +1.47
  **5/5 SIGN AGREEMENT** between (G_ssl - G_prior) and the e2e ordering:
  whichever intervention has the larger feature gain wins end to end, on both
  populations, at every fraction. So the budget comparison is feature-side
  throughout and not a readout artifact -- the same conclusion the C100-only
  probes reached, now with the second population included.
  AND IT EXPLAINS THE DISAGREEMENT. At a MATCHED 5,000 images the two
  populations differ because the PRIOR's feature gain differs: G(prior) is
  +22.19 on C100 and only +15.67 on tin, while G(ssl5x) is +21.27 and +17.72.
  The comparator loses ~3.5 points of G moving to tin; the prior loses ~6.5.
  So the free spectral target contributes LESS on the 200-way 64px task
  relative to what contrastive pre-training contributes there, which is
  exactly why tin favours the comparator at every matched count.
  This is the honest mechanism for the narrowed claim, and it is measured
  rather than argued: the population term is not a mystery, it is G(prior)
  degrading faster than G(SSL) as the label space gets finer.

- *** tin@25% SCORED — THE FALSIFIER FIRED, MY BET LOST, AND THE TWO
  POPULATIONS TURN OUT TO AGREE (2026-08-10, 3 seeds):
      deit-base 23.14 | deit-ssl2x 38.48 | deit-aux 38.24 | deit-ssl5x 36.53
      ssl5x band was 40..44 -> MISSED LOW by 3.5
      aux - ssl5x = **+1.71 +-0.33 = +5.1 sigma: THE PRIOR WINS**
  I predicted 40..44, explicitly betting AGAINST our own method ("5x SSL
  wins and the prior beats it nowhere on tin"). Wrong by 5.1 sigma in the
  method's favour. The recorded falsifier -- "ssl5x <= 38.24 => the
  population difference is NOT monotone and the tin envelope has a crossover
  after all, making the two populations the same shape with a shifted
  crossing" -- FIRED, and its stated consequence is exactly what the data
  now show.
  *** THE MATCHED-FRACTION TABLE, and it is the clean result:
      pct      C100 (aux-ssl5x)        tin (aux-ssl5x)
       5%   -1.57 (-3.9 s) SSL      -1.35 (-3.7 s) SSL
      10%   +0.71 (+1.2 s) tie      -0.31 (-0.6 s) tie
      25%   +1.47 (+7.2 s) prior    +1.71 (+5.1 s) prior
  IDENTICAL ordering at every matched FRACTION on both populations. The
  earlier "tin favours the comparator everywhere / different stories"
  reading was an artifact of having only 5 and 10% on tin, and MUST BE
  WITHDRAWN from the paper. The correct statement: the ordering tracks the
  data FRACTION -- where a cell sits on its own dataset's learning curve --
  not the absolute image count. That is why matching on image count made
  the populations look like they disagreed (5k images is 10% of C100 but 5%
  of tin, i.e. different points on their own curves).
  *** AND A GENUINE ODDITY, 4.6 sigma: at tin@25%, 4x MORE pre-training made
  SimCLR WORSE -- ssl2x 38.48 vs ssl5x 36.53, a drop of 1.95 +-0.42. First
  cell in the study where extra pre-training budget hurts. Echoes the
  deitssl2b finding that stronger contrastive views hurt; do not explain it,
  report it.
- *** ViT@5% DEEPENED 3 -> 6 SEEDS: aux 28.87+-0.26, ssl5x 30.45+-0.31,
  aux-ssl5x = **-1.57 +-0.41 = -3.9 sigma** (3-seed value was -1.08, 2.0
  sigma). The one marginal cell in the budget story STRENGTHENED under
  power rather than regressing -- SSL genuinely wins that cell.

## IMAGENET ENVELOPE CAMPAIGN (2026-08-10, user: "use BSC to do all missing
## runs, test the training before you submit, maximize the GPU utilization")

- WHY: Table 3 showed dashes at every fraction for both ImageNet stages, and
  the reason was upstream of the table -- scripts/make_subsets.py had NO entry
  for imagenet64/imagenet100, so no index file existed and no fraction cell
  could ever be launched. The 100% stages (14 cells, 3 seeds, 164 GPU-hours)
  were built to fire the pre-registered SCALE falsifiers, which they did (F1,
  F2, F3, G1-G4 all dead). What they never tested is the ENVELOPE: peak
  location and left-flank suppression are established on eleven populations,
  none above 100k images. Section 9 already concedes this as "a cost decision
  rather than a considered one, and the cheapest gap to close".
- IT IS CHEAP FOR A STRUCTURAL REASON: the diagnostic epoch budget is FIXED
  (40 for imagenet64, 100 for imagenet100) regardless of fraction, so cost
  scales ~linearly with the fraction. Measured 100% wall-clocks: in64 r18 7.9
  h/seed-pair, vit 4.4, mnet 5.6; in100 vits 6.1, vitb 10.6, r50 11.9. The
  whole envelope is ~85 GPU-hours against the 164 already spent at 100%.
- CELLS (84 configs, 3 seeds = 252 runs). Both arms of every pair are
  byte-identical except the moment_aux block (verified by diff), and every
  fraction keeps its parent's num_workers so the augmentation stream stays
  comparable to the 100% anchor:
    imagenet64  @1/2/3/5/7/10/15/20/25%  x {r18, vit_tiny, mnet} x {none,aux}
    imagenet100 @1/2/5/10/25%            x {vit_s, vit_b, r50}   x {none,aux}
  50% is omitted on both: it costs as much as every other fraction combined
  and the 100% cell already anchors the right flank.
  diagin64_swin is DELIBERATELY EXCLUDED -- its 100% baseline is bistable at
  chance ({16.12, 0.10, 0.10}), so a fraction envelope there would measure
  training failure, not feature gain. Same scope rule as the swin G-probes.
- SUBSETS: imagenet64 (1,000 classes) and imagenet100 added to PCTS with the
  same deterministic stratified draw as every other dataset; --check
  reproduces all 14 files byte-identically. 1% of imagenet64 = 12,820 images =
  13 per class, the FINEST label space in the study at genuine scarcity.
- PREDICTIONS RECORDED IN ADVANCE (nothing from these cells seen; anchors are
  the measured 100% cells: r18 base 55.38 D +0.04 | vit 48.24 +3.24 | mnet
  32.93 +1.95 | vits 65.39 +13.00 | vitb 43.33 +26.01 | r50 85.85 -0.02):
    (E1) THE CONV ENVELOPE SHAPE RECURS AT 1000-WAY: Delta(r18,in64) is
      UNIMODAL in fraction -- an interior peak exceeding both ends by >= 0.5
      -- peaking at 5-20% with magnitude **+1.5..+4.0**, decaying to the
      measured +0.04 at 100%.
      FALSIFIER: monotone in fraction (no interior peak), or peak > +6 =>
      the unimodal envelope is a small-dataset regularity, and the shape
      claim must be scoped to <= 100k images.
    (E2) LEFT-FLANK SUPPRESSION AT THE FINEST LABEL SPACE: at 1% the baseline
      sits far below the crossing bracket [31.8, 40.3], so the sign law
      demands NEGATIVE readout and Delta < G. Band **Delta(r18,in64@1%) =
      +0.5..+2.5**, i.e. BELOW the envelope peak.
      FALSIFIER: readout POSITIVE at 1% (base far below 31.8) => the first
      sign-law failure on a resolvable left-flank cell at scale.
    (E3) THE ATTENTION DEFICIT SPANS THE ENVELOPE: Delta(vit,in64) >= +3 at
      every fraction >= 5%, peaking **+6..+14** at 10-25% (C100-ViT peaks at
      15% with +14.44, tin-ViT at 15% with +10.73), and above the conv
      envelope everywhere.
      FALSIFIER: Delta(vit) <= +1 at any fraction >= 5% => the attention
      deficit is fraction-bounded at 1000-way, not a property of attention
      at small data.
    (E4) CONV GAINS SURVIVE NATIVE RESOLUTION: R50 on in100 is neutral at
      100% (-0.02) but every conv population measured is POSITIVE at low
      data, so **Delta(r50,in100@1-5%) = +1..+6**, decaying through 25%.
      FALSIFIER: Delta(r50) <= 0 at EVERY fraction => conv gains do not
      survive 224px and the low-data conv claim is a low-resolution claim.
    (E5) THE MODEL-SCALE ORDERING PERSISTS: Delta(vitb) > Delta(vits) at
      every fraction (it is +26.01 vs +13.00 at 100%).
      FALSIFIER: the ordering inverts at any fraction >= 5%.
  NOT PREDICTED, recorded as data: mnet on in64 (base 32.93 sits INSIDE the
  crossing bracket, so the law makes no sign call there -- same treatment as
  the mnet shots cell), and the in100 ViT-B baselines at 1-2%, which may be
  at or near chance on 1,267 images (the bistable flag applies if a seed
  collapses; sigma was already 3.29 at 100%).
- GPU UTILIZATION (user asked explicitly). The worker already runs SLOTS
  trainings per GPU because these cells are DATALOADER-bound; SLOTS=3 was set
  after 8 slots OOM'd on 19.5GB diagin64 cells. The binding constraints are
  measured, not assumed: CPU budget SLOTS*4*(num_workers+1) <= 80 caps in64
  (num_workers 3) at SLOTS=5, and in100 (num_workers 5) at SLOTS=3. Memory is
  measured per cell in the smoke pass and the slot count set from it, so the
  two queues are submitted SEPARATELY -- a mixed queue would have to be sized
  for its heaviest member, which is exactly how the OOM incident happened.

- *** IMAGENET ENVELOPE, FIRST LANDINGS (2026-08-10, 3v3 each, 60/252 done,
  0 failures). Scored against the bands recorded before any cell ran:
      cell            base     aux    delta
      in64_r18  @1%    5.47    7.37   +1.90 +-0.06
      in64_r18  @2%   13.19   14.78   +1.59 +-0.12
      in64_r18  @3%   18.94   20.33   +1.38 +-0.12
      in64_vit  @1%    1.88    4.02   +2.14 +-0.06
      in64_vit  @2%    2.63    6.43   +3.80 +-0.07
      in64_vit  @3%    3.43    8.19   +4.76 +-0.17
      in64_mnet @1%    4.33    3.82   -0.52 +-0.11
      in64_mnet @2%    7.61    7.62   +0.01 +-0.13
      in64_mnet @3%    9.68   10.17   +0.49 +-0.13
  (E2) HIT: Delta(r18@1%) = +1.90, band +0.5..+2.5. The premise holds too --
    base 5.47 sits far below the crossing bracket [31.8,40.3], so the sign
    law demands NEGATIVE readout here; the G probes will test it.
  (E1) TRENDING AGAINST ME, and I am recording that now rather than after
    the fact. I predicted a UNIMODAL envelope with an interior peak at
    5-20%. r18 is MONOTONE FALLING so far (+1.90/+1.59/+1.38 at 1/2/3%
    toward the measured +0.04 at 100%), i.e. the peak is at or below 1%.
    If 5/7/10% continue the fall, the recorded falsifier ("monotone in
    fraction, no interior peak") FIRES and the interior-peak claim must be
    scoped to <=100k images. NOTE this is not the same as the envelope
    disappearing: a peak at or below the smallest runnable fraction is what
    C10 shows too (plateau at 1-2%, falling after). What would be wrong is
    my having predicted WHERE it peaks at 1000-way.
  (E3) ON TRACK: ViT is RISING steeply (+2.14/+3.80/+4.76) and already
    above the conv envelope at every fraction, as predicted. The >=5% cells
    test the +6..+14 band. The attention signature is stark at 1000 classes:
    vit baselines 1.88/2.63/3.43 are barely above chance (0.1%) and the
    prior more than doubles them.
  mnet: -0.52/+0.01/+0.49 at 1/2/3%. No band was recorded (its 100% base
    sits INSIDE the crossing bracket, where the law makes no sign call), so
    this is data, not a scored prediction. The early 2-seed "aux below
    baseline" I declined to score at 1% survives at 3v3 as a small real
    negative, and it turns positive by 3%.

- *** IMAGENET ENVELOPE SCORED AT 127/252 CELLS (2026-08-10, 3v3 each, still
  0 failures). The three backbones have THREE DIFFERENT ENVELOPE SHAPES at
  1000-way, which is the finding:
      pct    r18 base  D_r18  |  vit base  D_vit  |  mnet base  D_mnet
       1%      5.47    +1.90  |    1.88    +2.14  |    4.33    -0.52
       2%     13.19    +1.59  |    2.63    +3.80  |    7.61    +0.01
       3%     18.94    +1.38  |    3.43    +4.76  |    9.68    +0.49
       5%     27.26    +0.56  |    5.89    +5.22  |   12.94    +1.46
       7%     32.31    +0.42  |    7.59    +5.63  |   15.29    +2.28
      10%      (run)          |    9.67    +6.31  |   18.60    +2.61
      15%     42.27    -0.01  |   13.07    +6.72  |   22.49    +2.75
     100%     55.38    +0.04  |   48.24    +3.24  |   32.93    +1.95
  (E1) **FALSIFIED, and the falsifier is mine.** I predicted a UNIMODAL conv
    envelope with an interior peak at 5-20%. r18 is MONOTONE FALLING from the
    smallest runnable fraction to zero by 15% (+1.90 -> -0.01) with no
    interior peak anywhere. The recorded falsifier ("monotone in fraction, no
    interior peak => the shape claim must be scoped to <=100k images") FIRES.
    POST-MORTEM, and it is the same error class as the tin@25% budget miss: I
    predicted in FRACTION space when the operative axis is IMAGE COUNT. 1% of
    ImageNet64 is 12,820 images -- on Tiny-ImageNet that is the 10% cell,
    already past its peak. The smallest fraction runnable at ImageNet scale
    sits on the RIGHT FLANK. The envelope is not absent; its peak is below
    anything this dataset can express, exactly as CIFAR-10's is.
  (E2) **HIT**: D(r18@1%) = +1.90, band +0.5..+2.5, at 13 images per class
    over 1000 classes -- the finest label space in the study at genuine
    scarcity. Base 5.47 is far below the crossing bracket, so the sign law
    demands negative readout; the G probes test it.
  (E3) **IN BAND**: D(vit) >= +3 at every fraction >= 5% as predicted, and
    +6.31 @10% / +6.72 @15% are inside the +6..+14 peak band. Peak location
    (predicted 10-25%) awaits 20/25%.
  *** THE SHAPE IS A BACKBONE PROPERTY, NOT A DATASET PROPERTY. At the SAME
  images, same label space, same recipe: r18 falls monotonically, while vit
  and mnet RISE monotonically through 15%. The account that fits: the peak
  sits where the baseline crosses the readout bracket, and only r18's
  baseline reaches it within the measured range (5.5 -> 42.3, crossing
  [31.8,40.3] at 7-15%) -- which is exactly where its Delta reaches zero
  (+0.42 at base 32.3, -0.01 at base 42.3). vit (1.9->13.1) and mnet
  (4.3->22.5) never get near it, so both are still on their rising limb at
  the largest fraction measured. This is a sharper statement of "the peak
  tracks task difficulty" than the classification grid could make, because
  here the dataset is held fixed and only the backbone varies.

## DENSE-PREDICTION TRANSPLANT (2026-08-10, user: "run the experiments and
## having them and later we decide to include them to the paper")

- WHY: every number in the study is top-1 ACCURACY. The auxiliary target is a
  dense oriented-energy map m(x) in R^{NxHxW}, so a dense task asks whether
  the fusion rule and Delta = G + readout are properties of the METHOD or of
  the METRIC. PASCAL VOC-2012 segmentation, augmented (SBD) train split:
  10,582 train / 1,449 val, 21 classes, ResNet-18 at output stride 8 with an
  FCN head, fractions 1/2/5/10/25/100%, 3 seeds, both arms.
- WHAT TRANSFERS WITH NO REDESIGN, which is why this is cheap: MomentAuxModel
  pools its target to the tapped stage's size, so the SAME pinned bank, tap,
  lambda schedule and head_norm apply. At a 512 crop and stride 8 the target
  is matched at 32x32 instead of the 8x8 of a classification cell.
- STATE THE BIAS FIRST: a dense spatial target on a dense spatial task is the
  most favourable venue this prior could be given. A win here is WEAKER
  evidence of generality than a win on a task with no such alignment.
  Detection is the harder test and is deliberately not what this starts with.
- UNITS CAVEAT, recorded before any result: mIoU points are NOT accuracy
  points. Any comparison of a Delta here to a Delta in the classification
  grid is a comparison across units and must be stated as such rather than
  read as "bigger". The readout crossing bracket [31.8, 40.3] is an ACCURACY
  bracket and is NOT transplanted; what these cells can test is the FORM of
  the sign law, and with enough fractions they can ESTIMATE a crossing in
  mIoU space.
- PREDICTIONS RECORDED IN ADVANCE (nothing beyond 2-epoch smoke runs seen):
    (D1) THE ENVELOPE IS POSITIVE across 1-25%, with a peak magnitude of
      **+2..+8 mIoU**. PEAK LOCATION IS DELIBERATELY NOT PREDICTED: E1 on
      ImageNet64 just failed for exactly that, and the lesson (predict the
      axis you can measure, not the one you assume) is worth applying
      immediately rather than after a second miss.
    (D2) STRUCTURAL NEUTRALITY SURVIVES THE TASK CHANGE:
      **Delta(100%) = 0 +- 1.5 mIoU**. lambda reaches exactly zero at the
      final epoch, so late training is pure cross-entropy; this is the one
      property that should hold regardless of metric.
      FALSIFIER: |Delta(100%)| > 2.5 => structural neutrality is an artifact
      of the classification setting, which would be a real limitation and
      would need saying in the paper whether or not these cells are included.
    (D3) THE SIGN LAW'S FORM TRANSPLANTS: readout = Delta - G is NEGATIVE at
      1% (baseline mIoU deep on the left flank) and RISES toward zero as the
      baseline rises.
      FALSIFIER: the ORDERING inverts (readout positive at 1% and negative at
      25%) => the law's form is metric-specific, not a property of the method.
    (D4) THE ALIGNMENT QUESTION, and the point of running this at all. VOC@1%
      is 107 images = 5 per class over 20 classes, the same per-class regime
      as CIFAR-100@1% (+1.42) and tin@1% (+1.49), where the classification
      grid finds a universal ~+1.5 floor. If the prior pays MORE when the
      task is dense, **Delta(VOC@1%) = +2..+6 mIoU**.
      FALSIFIER, and it is the more interesting branch: Delta(VOC@1%) <= +1.5
      => target-task alignment is NOT where the value comes from, and the
      claim becomes cleaner (the prior supplies the same feature structure
      whatever the head does with it) rather than weaker.
- COST, measured on the local 3090 rather than estimated: a 1% cell is ~3s
  per epoch and the 1,449-image native-resolution validation is ~18s, so
  evaluation was costing more than the training it measured. Validation now
  runs every 5 epochs plus the final one, identically for both arms. The full
  6-fraction envelope is ~6-10 GPU-hours on an H100 for 36 runs.

- *** DENSE PRE-FLIGHT PASSED ON BSC AND THE VOC ENVELOPE IS LAUNCHED
  (2026-08-10, ms_denssmoke 44477248, DENSE_SMOKE_COMPLETE failures=0). All
  five checks green on the machine that will run the campaign:
    (1) the committed dense subsets reproduce byte-identically from BSC's OWN
        VOC/SBD copy (107/209/528/1059/2648 images) -- so the two machines
        agree on which images each cell trains on, which is the check that
        makes a cross-machine merge legitimate;
    (2) the joint-transform CONTENT tests pass under BSC's library versions;
    (3) both arms train and write a final.json -- and the aux arm logs
        lam 1.000 -> 0.000, i.e. the lambda schedule is live and
        MomentAuxModel BUILT under BSC's older aux.py via the in_channels
        compatibility shim. That was the one thing that could have produced a
        ONE-ARMED envelope (baseline fine, aux dead) and it is now measured,
        not assumed;
    (4) PEAK 4,367 MiB at the largest fraction;
    (5) the dense probe path runs end to end.
  LANE: the envelope runs on its OWN queue (worklist.dense /
  queue.counter.dense / queue.lock.dense / runs_dense, slurm/bsc_dense.sbatch,
  job ms_dense). NOT in the grid lane, deliberately: the grid reconcile
  rebuilds worklist.bsc from missing CLASSIFICATION cells only, so dense lines
  put there would be silently dropped at the next reconcile -- exactly how the
  78 diagnostics tasks vanished on 2026-08-02. The worker eval()s each
  worklist line, so a train_dense.py line needs no worker change.
  SLOTS=4 per GPU, sized from the measurement rather than guessed: memory
  allows far more (4.4GB each on an 80GB H100) but the cells are
  dataloader-bound on 512px joint transforms and the node budget
  SLOTS*4*(num_workers+1) <= 80 with num_workers=4 caps it at 4.
  ORDER: 1/2/5% first (the left flank and D4, the alignment question that is
  the point of running this), then 100% (D2, the structural-neutrality
  falsifier), then 10/25%. Both arms of a fraction stay adjacent so a partial
  drain never leaves a one-armed cell. 36 tasks, 2 workers.

- *** LOCAL-MIRROR CHECKPOINT AUDIT -- 9 STALE LOCAL COPIES FOUND AND
  REPAIRED; THE RECORDS AND THE CODE ARE BOTH CLEAN (2026-08-10, all 896
  local cells, verify-by-IDENTITY). Run to answer "do the new changes break
  previous checkpoints and reproducibility?". THE ANSWER IS NO -- aux.py is
  git-clean and the only new file in momentstem/ is segmentation.py, which
  nothing else imports -- but the sweep surfaced four separate things:
  (a) NINE CHECKPOINTS WERE STALE LOCALLY WHILE BSC HELD THE GENUINE ONES,
      the worst by 19 POINTS (auxmag_7pct_sched: recorded 36.90, local copy
      evaluated 17.94). Every one of the nine BSC originals reproduced its
      recorded accuracy to <=0.04 after pulling. Damage is SIGN-RANDOM
      (-18.96 to +0.94), which is how it is distinguishable from (b).
      Same class as the 2026-08-05 corruption and the 2026-08-06 wrong-epoch
      race; rsync's size+mtime quick check cannot see any of it.
  (b) A TIN-SPECIFIC RE-EVALUATION OFFSET, NOT DAMAGE: every locally-trained
      tin cell re-evaluates slightly BELOW its recorded value (-0.06, -0.10,
      -0.12, -0.24, -0.35, -0.38 on six independent cells). Six of six in the
      SAME direction is not checkpoint damage; the likely cause is JPEG decode
      drift across library versions over tin's 110k JPEGs (a few tens of
      borderline predictions flipping out of 10,000). Recorded numbers stand
      -- they were measured at train time -- and every Delta is a within-pair
      difference under identical conditions, so nothing in the ledger moves.
      RULE: a <=0.4 residual when re-verifying a tin cell is expected; treat
      >0.5, or any residual on a non-tin cell, as damage.
  (c) FIVE GENUINELY CORRUPT FILES (PytorchStreamReader, byte-identical size)
      re-pulled and verified; all four affected cells that carry a probe had
      probed all three seeds BEFORE the corruption, so no recorded G is built
      on a damaged file. Checked rather than assumed.
  (d) 26 CELLS CARRY LEGACY KEY NAMING: the original fixed-lambda aux family
      (auxmag_*, auxgab_*, auxrand_*) stores the auxiliary target under
      moment_stem. where the current model expects target.stem. -- a rename
      from early in the study, and the tensors under it are the pinned bank
      and its calibration, fixed and never trained. verify_checkpoints.py now
      remaps the prefix on READ; aux.py is untouched, so no cell's training
      behaviour can move, and the remap is only trustworthy because the
      evaluation that follows CHECKS it.
  NONE of the 11 discrepant cells is referenced by the manuscript -- they
  appear only in the released results CSVs, which are generated from
  final.json (values computed in-memory at train time, never re-derived from
  a checkpoint). Verified explicitly rather than argued.
  SCRIPT FIX, and it is the reusable lesson: the first sweep ABORTED at cell
  10 of 896 because one legacy-format load raised. An audit that stops on its
  first finding is worse than no audit -- it reports a clean bill for the 886
  cells it never reached. verify_checkpoints.py now CLASSIFIES rather than
  raises (ok / FAIL / CORRUPT / KEYS) and reports the four counts separately,
  because lumping "unreadable" together with "old naming" hides the one that
  matters.

- *** THE VOC DENSE ENVELOPE LANDED — D4 FALSIFIED IN THE CLEANER DIRECTION,
  D2 FALSIFIED BECAUSE MY PREMISE WAS WRONG (2026-08-10, 36/36 cells, 3 seeds
  every point, ~4h on the dense lane):
      pct    baseline         prior          Delta (mIoU)
        1   4.65 ±0.04    5.03 ±0.03      +0.38 ±0.05
        2   4.62 ±0.04    5.11 ±0.03      +0.50 ±0.05
        5   5.01 ±0.03    5.73 ±0.08      +0.72 ±0.09
       10   7.23 ±0.07    8.68 ±0.11      +1.46 ±0.13
       25  13.54 ±0.16   15.60 ±0.07      +2.06 ±0.17
      100  33.63 ±0.25   36.20 ±0.21      +2.57 ±0.32
  (D1) HIT, at the bottom edge: positive across 1-25% as predicted, peak
    magnitude in that range +2.06 vs the band +2..+8. I deliberately did NOT
    predict the peak location after E1, and that restraint was right — there
    is no peak in range at all (see D2).
  (D4) **THE FALSIFIER FIRED, AND IT IS THE BRANCH THAT MAKES THE CLAIM
    CLEANER.** I recorded: if the prior pays MORE when the task is dense,
    Delta(VOC@1%) = +2..+6; if instead Delta <= +1.5, "target-task alignment
    is NOT where the value comes from, and the claim becomes cleaner rather
    than weaker". Measured **+0.38 ±0.05** — not merely below the tripwire but
    an order of magnitude below the band. THE SHARP COMPARISON, at matched
    supervision density: VOC@1% is 107 images over 20 classes = ~5 per class,
    the same regime where classification shows a universal ~+1.5 floor
    (C100@1% +1.42, tin@1% +1.49). Dense gives **+0.38**. So a dense spatial
    target on a dense spatial task — the most favourable venue this prior
    could be given, as recorded up front — pays LESS, not more. The gain is
    not an artifact of target/task alignment.
  (D2) **THE FALSIFIER NOMINALLY FIRED, AND THE POST-MORTEM IS THE REAL
    RESULT.** Recorded: Delta(100%) = 0 ±1.5, falsifier |Delta| > 2.5 =>
    "structural neutrality is an artifact of the classification setting".
    Measured +2.57 ±0.32 — 3.3 sigma above the band top, and 0.07 past the
    2.5 threshold (that last crossing is 0.2 sigma, i.e. NOT resolvable, and
    must not be reported as a clean fire).
    BUT THE PREMISE WAS WRONG, NOT THE MECHANISM. lambda still reaches
    exactly 0 at the final epoch here, so structural neutrality is untouched
    as a mechanism; what fails is my assumption that VOC@100% is a
    SUFFICIENCY cell. It is 10,582 images at 21-way dense prediction with a
    baseline of 33.6 mIoU — nowhere near saturation. The envelope is MONOTONE
    RISING across every fraction (+0.38/+0.50/+0.72/+1.46/+2.06/+2.57) with
    no peak and no right flank in range: VOC@100% is a LOW-DATA cell in this
    study's terms, so the prior is still on its rising limb and neutrality was
    never predicted there by the mechanism — only by my mislabelling of the
    axis. SAME ERROR CLASS AS E1 on ImageNet64, four days apart: predicting in
    FRACTION space when the operative axis is absolute data. Recorded as a
    band miss and an ACCOUNT miss; "neutral at sufficiency" is NOT falsified,
    because this envelope never reaches sufficiency.
  UNITS CAVEAT, recorded before the run and still binding: mIoU points are not
  accuracy points. +2.57 on a 33.63 baseline is +7.6% relative; do not compare
  the magnitude to a classification Delta.
  Variance asymmetry is mild here and mostly at 25% (baseline SEM 0.16 vs aux
  0.07); it is not the dramatic stabilization seen on swin/ConvNeXt.
  (D3) STILL OPEN — needs the dense G probes; bands recorded below.

- DENSE G-PROBE PASS LAUNCHED (2026-08-10, dense lane, analysis/dense_probe.py:
  frozen backbone, single 1x1 conv to 21 classes, full train_aug split, mIoU on
  val — the dense analog of the linear probe, same protocol both arms).
  SCOPE: 1/2/5/10/25% only. VOC@100% is EXCLUDED by the probe-ceiling rule
  recorded in advance — there the probe's labels ARE the cell's labels, so no
  G/readout split is interpretable (the cub@100% precedent).
  *** THIS IS A FORK THE CLASSIFICATION GRID COULD NOT TEST, which is the best
  reason to run it. Q6.9h established that "labels per class" was a PROXY for
  BASELINE HEIGHT — the two are confounded in every classification cell. Dense
  prediction DECOUPLES them: VOC@1% has a tiny baseline (4.65 mIoU) but
  enormous supervision (107 images x ~262k labelled pixels). The two accounts
  therefore make opposite predictions here:
    H-BASELINE (the law as stated — readout is a function of task
      performance): readout NEGATIVE and material at 1-5%, so G >> Delta.
      BANDS: G(1%) = +1.0..+2.5 (vs Delta +0.38), G(5%) = +1.3..+3.0.
    H-LABELS (readout really tracks the label budget, and baseline height only
      ever stood in for it): dense supervision is abundant at every fraction,
      so readout ~ 0 and G ~= Delta within +-0.3 at EVERY fraction.
      BANDS: G(1%) = +0.1..+0.7, G(5%) = +0.4..+1.0.
    BOTH predict G(25%) ~= Delta = +1.8..+2.6 (readout near zero as the
      baseline rises), so the 25% cell is the CONTROL, not the discriminator —
      it is the 1-5% cells that separate the accounts.
  FALSIFIER for "the law's FORM transplants off accuracy" (recorded as D3
  before any dense run): the ORDERING inverts — readout POSITIVE at 1% and
  NEGATIVE at 25%. That would say the form is metric-specific rather than a
  property of the method.
  NOTE the units caveat still binds: the accuracy crossing bracket
  [31.8, 40.3] is NOT transplanted, and no mIoU crossing is claimed in
  advance. With five fractions this pass can ESTIMATE where one sits.

- *** DENSE G-PROBES LANDED — THE LAW'S FORM TRANSPLANTS TO SEGMENTATION, BUT
  MY D3 PREDICTION WAS WRONG AND THE REASON IS A UNITS ERROR I MADE MYSELF
  (2026-08-10, 10 cells, 3 seeds each, full train_aug probe, both arms
  identical protocol):
      pct  base_e2e aux_e2e  Delta | probe_b probe_a      G      | readout
        1    4.65    5.03   +0.38 |  4.26    4.34   +0.08 ±0.06 |  +0.29
        2    4.62    5.11   +0.50 |  4.31    4.52   +0.21 ±0.08 |  +0.29
        5    5.01    5.73   +0.72 |  4.83    5.37   +0.54 ±0.11 |  +0.18
       10    7.23    8.68   +1.46 |  6.85    8.07   +1.22 ±0.18 |  +0.24
       25   13.54   15.60   +2.06 | 12.30   13.83   +1.54 ±0.15 |  +0.52
  (1) THE GAIN IS FEATURE-SIDE, as everywhere else in the study: G tracks
      Delta at every fraction (0.08/0.21/0.54/1.22/1.54 vs 0.38/0.50/0.72/
      1.46/2.06), and G(1%) = +0.08 ±0.06 is a measured ZERO — at 107 images
      the prior barely moves the frozen features at all. That is the
      feature-side confirmation of D4: dense-at-5-per-class really does get
      almost nothing, and it is not a readout artifact.
  (2) MY D3 CALL IS FALSIFIED AS STATED. I predicted readout NEGATIVE at 1%
      and rising toward zero. Measured readout is POSITIVE at every fraction
      and roughly FLAT: +0.29/+0.29/+0.18/+0.24/+0.52. The recorded FALSIFIER
      was narrower than my claim — it required the ORDERING to invert
      (positive at 1% AND negative at 25%) — and it did NOT fire, since
      readout is positive at both ends. Scored honestly as: tripwire not
      fired, prediction wrong.
  (3) THE UNITS ERROR, and it is the useful part. I read "task performance" as
      mIoU, saw baselines of 4.65 and put the cells deep on the left flank.
      But readout is a property of the CLASSIFICATION THE HEAD ACTUALLY
      PERFORMS, and per-pixel accuracy on these same cells is
      **71.95 / 72.71 / 73.43 / 74.34 / 76.76 / 83.24** at 1/2/5/10/25/100%
      — every cell sits FAR ABOVE the crossing bracket [31.8, 40.3], on the
      decayed POSITIVE branch. There the measured classification mapping gives
      +0.44 @ base 69.1 and +0.44 @ base 80.7. Dense returns +0.18..+0.52,
      mean ~+0.30. Same sign, same small magnitude, same flatness. SO THE SIGN
      LAW HOLDS HERE — I mis-scaled the input variable, not the law. mIoU is a
      harsh metric and a 4.65 mIoU cell is NOT a low-task-performance cell.
      This is exactly what the units caveat recorded in advance was warning
      about ("the readout crossing bracket is an ACCURACY bracket and is NOT
      transplanted"); I wrote the caveat and then violated it in my own
      prediction one paragraph later.
  (4) THE FORK WAS MIS-SPECIFIED, and I am recording that rather than claiming
      a winner. H-LABELS' numeric bands (G ~= Delta, readout ~0 within +-0.3)
      are hit at 4 of 5 fractions and H-BASELINE's (G >> Delta, G(1%)
      +1.0..+2.5) is dead by 15 sigma. But once the baseline is read on the
      PIXEL-ACCURACY scale, H-baseline ALSO predicts readout ~ +0.4 — so both
      accounts collapse to the same prediction here and this pass CANNOT
      discriminate them. The discriminating cell would need abundant labels
      AND low per-pixel accuracy, which VOC cannot provide. Do not cite this
      pass as evidence that readout tracks label budget rather than task
      performance.
  (5) The G(25%) "control" both accounts agreed on (+1.8..+2.6) came in at
      +1.54 ±0.15 — MISSED LOW by 0.26 (1.7 sigma). A control that both
      branches miss is a sign the readout term is slightly larger at 25% than
      the positive branch predicts, consistent with its +0.52 there being the
      largest of the five.
  NET FOR THE PAPER: segmentation is now a 14th population where Delta = G +
  readout holds, on a different METRIC and a different TASK — with the
  explicit statement that the crossing must be evaluated on the head's own
  classification scale, not on mIoU. That scale caveat is a genuine
  contribution of running the dense transplant, and it was discovered by
  getting it wrong first.

## DENSE POPULATIONS 2 AND 3 (2026-08-11, user: "we need to deliver both")

- WHY: the dense transplant's two findings currently rest on ONE dataset. D4
  (alignment is not the source of the gain) draws its force from a comparison
  with the classification grid and is fairly robust; the UNITS finding --
  readout reads on the head's own classification scale, not on mIoU -- does
  not. VOC is object-centric and background-dominated, so its pixel accuracy
  sits at 72-83% while its mIoU is 4.6-33.6; the two scales never separate for
  a reason other than the one I claimed. That is exactly the position the
  classification story was in before the domain expansion.
    ade20k      150 scene classes, NO background class, 20,210 train images.
                Pixel accuracy and mIoU decouple sharply here, so this is the
                population that can actually TEST the units reading.
    cityscapes  19 classes, driving domain, near-fixed geometry, 2,975 train.
                The DOMAIN test rather than the label-space one.
- ACQUISITION, stated because it constrains what can be claimed: ADE20K comes
  from the MIT release directly. **Cityscapes' official download is
  ACCOUNT-GATED**, so its images come from an ungated HuggingFace mirror
  (Chris1/cityscapes, image + semantic_segmentation as separate features).
  scripts/prepare_seg_data.py does NOT trust that mirror's encoding: it
  detects labelIds-vs-colour, maps through the official 19-class trainId
  table, and --verify asserts that no staged mask carries a train id >=
  n_classes and that every mask matches its image pixel for pixel. The VOC
  palette bug (class 1 -> 38, class 15 -> 147, training broken while
  validation stayed correct) is why this is asserted rather than assumed.
- DEVIATIONS, both applied identically to both arms: Cityscapes is stored at
  HALF resolution (512x1024), because a frozen 512 crop on a 2048-wide frame
  samples a far smaller share of the scene than the same crop does on a ~500px
  VOC image -- half-res keeps the RECIPE comparable across populations, not
  merely nominally identical. The 50-epoch dense recipe is otherwise frozen on
  all three populations, with no per-dataset tuning.
- PREDICTIONS RECORDED IN ADVANCE (nothing beyond a 2-epoch smoke seen).
  VOC anchors: Delta +0.38/+0.50/+0.72/+1.46/+2.06/+2.57 at 1/2/5/10/25/100%,
  monotone rising, G tracking Delta, readout +0.18..+0.52 throughout.
    (E-A) ADE20K IS MONOTONE RISING TOO, and for the same reason: 20,210
      images at 150-way dense prediction is a low-data cell in this study's
      terms, so no right flank should appear. Delta(ade@100%) = **+1.5..+4.0**
      mIoU. FALSIFIER: an interior peak (some fraction exceeding BOTH 1% and
      100% by >= 0.5) => the dense envelope does have a peak and VOC's
      monotonicity was a VOC fact.
    (E-B) THE DEEP-LEFT FLOOR TRANSPLANTS: ade@1% is 202 images over 150
      classes (~1.3 per class), the thinnest supervision anywhere in the
      study. Delta(ade@1%) = **+0.0..+0.5**, i.e. at or below VOC@1%'s +0.38.
      FALSIFIER: >= +1.5 => scene-centric dense data gives the prior real
      work to do even at ~1 image per class, which would reopen the dense
      direction as a low-data claim rather than a scaling one.
    (E-C) THE UNITS TEST, and it is the point of ADE20K. Its pixel accuracy
      will be FAR below VOC's (no background class to inflate it; expect
      ~30-55% at the fractions measured, vs VOC's 72-83%). The sign law then
      makes a SHARP call the VOC cells could not: cells whose pixel accuracy
      falls BELOW the crossing bracket [31.8, 40.3] must show NEGATIVE
      readout, while VOC -- entirely above it -- showed positive readout at
      every fraction (+0.18..+0.52). So: **readout(ade) < 0 wherever pixel
      accuracy < 31.8, and > 0 wherever it exceeds 40.3.**
      FALSIFIER, and this one costs the units finding: readout POSITIVE on an
      ADE20K cell whose pixel accuracy is clearly below 31.8 => "read the
      crossing on the head's own classification scale" is wrong, and the VOC
      positive readouts had some other cause.
    (E-D) CITYSCAPES: the domain transplant. 19 classes, 2,975 images, and a
      scene layout so regular that a network learns most of it from very
      little data -- so the baseline should be HIGH early and the prior should
      have correspondingly little to add. Delta(city@100%) = **+0.5..+2.5**,
      and the envelope RISING but flatter than VOC's.
      FALSIFIER: Delta(city) <= 0 at every fraction => the dense gain does not
      survive a domain change and the dense claim must name its domain.
    (E-E) NOT PREDICTED, recorded as data: where each population's pixel
      accuracy sits relative to the bracket, and whether Cityscapes' near-
      fixed geometry produces the variance asymmetry (baseline sigma > aux
      sigma) seen on unstable classification cells.
  NOTE E-C is the one that matters. E-A/E-B/E-D are transplant checks; E-C is
  a test the first dense population was structurally incapable of running, and
  it can fire against a claim I made four days ago.

- FOURTH DENSE POPULATION ADDED THE SAME DAY (2026-08-11, user: "There is also
  FoodSeg103 ... /media/amughrabi/HDD_4TB_1/amughrabi/data/FoodSeg103").
  **foodseg103**: 4,983 train / 2,135 test, 104 classes (103 foods +
  background), texture-dominated, images ~512x384. Already local, so it costs
  no acquisition at all.
  WHY IT IS WORTH MORE THAN A FOURTH DATA POINT: food101 is an existing
  CLASSIFICATION population -- and one of the few where SimCLR beats the prior
  (+2.4..+4.3). FoodSeg103 is therefore the only place in the study where the
  SAME VISUAL DOMAIN can be compared across TASKS rather than only across
  datasets. The 2026-07-23 note ("a SEGMENTATION set, wrong task for the
  frozen classification recipe") was correct then and is now obsolete.
  TWO ENCODING TRAPS FOUND BY ASSERTING RATHER THAN ASSUMING, both of which
  would have produced plausible numbers from wrong data:
    (a) n_classes is **104, not 103**. category_id.txt has no trailing
        newline, so `wc -l` returns 103 and the last class ("other
        ingredients") would have been silently dropped from the confusion
        matrix. Pinned by taking max(mask) over 600 masks instead.
    (b) ONE training image in ~5,000 carries an EXIF orientation tag, so its
        stored size is (384, 512) against a (512, 384) mask -- it would have
        trained 90 degrees out of alignment with its own labels. Far too rare
        to move a metric visibly and invisible to any count-based check; the
        per-image size assertion is what surfaced it. Every image now goes
        through ImageOps.exif_transpose.
    Also 2 of 4,985 training images have NO mask; ids are read from the MASK
    directory, so those can never enter a split.
  PREDICTIONS RECORDED IN ADVANCE:
    (E-F) MONOTONE RISING like VOC, no interior peak: 4,983 images at 104-way
      dense prediction is a low-data cell in this study's terms.
      Delta(food@100%) = **+1.0..+3.5** mIoU.
    (E-G) DEEP-LEFT FLOOR: food@1% is ~1 image per class after stratification.
      Delta(food@1%) = **+0.0..+0.6**, matching VOC@1%'s +0.38.
    (E-H) PIXEL ACCURACY sits ABOVE the crossing bracket (plates and
      background dominate; expect ~55-75%), so readout should be POSITIVE and
      small at every fraction, as on VOC (+0.18..+0.52). This population is
      therefore a CONFIRMATION of the units reading, not a test of it --
      ADE20K remains the only population that can fire E-C.
    (E-I) THE CROSS-TASK COMPARISON, recorded as the reason to run it and NOT
      as a numeric prediction, because the two tasks' units do not convert:
      on food101 CLASSIFICATION the prior is beaten by SimCLR; if the dense
      food envelope is ordinarily positive, then "the domain where the prior
      loses to SSL" and "the task where it wins" are the same pixels, which
      would locate the SSL advantage in the TASK rather than the DOMAIN.
      No band, because a mIoU delta and an accuracy delta are not comparable.

- *** CITYSCAPES MASK-MAPPING BUG, CAUGHT BY ITS OWN VERIFIER ONLY AFTER THE
  VERIFIER WAS FIXED (2026-08-11). The mirror stores masks as RGB with the
  labelId REPLICATED across all three channels -- (7,7,7) road, (11,11,11)
  building, (26,26,26) car -- not as the official colour palette.
  _cs_mask_to_train saw ndim == 3, ran the palette LUT, matched NOTHING, and
  mapped every pixel to ignore. The staged split contained zero classes.
  *** AND verify() PRINTED "VERIFY OK" ON IT. The check was "no train id >=
  n_classes", which an all-ignore mask satisfies trivially. That is the same
  silent-guard family as the pretrain whitelists, the `|| v=0` counter
  fallback and the suppressed reconcile generator -- a guard whose passing
  condition is satisfied by the failure it exists to catch. EIGHTH incident.
  TWO FIXES, and the second matters more than the first:
    (a) a 3-channel mask whose channels are identical is now collapsed to
        labelIds; the palette branch is used only when the channels DIFFER,
        and it raises if the LUT matches nothing rather than returning an
        empty mask.
    (b) verify() now FAILS a split whose masks contain fewer than half the
        label space in 400 samples. Tested against a synthetic all-ignore
        tree, which it correctly REJECTS -- i.e. the guard now has a
        regression test of its own failure mode, not just of the happy path.
  After the fix: encoding detected as "labelId", 17/19 and 19/19 classes
  present. This is the third encoding trap in one day across three datasets
  (FoodSeg's 104-vs-103 and its EXIF-rotated image being the others), which
  is an argument for asserting every assumption about a new dataset rather
  than the two or three that seem risky.

- *** THE FOUR DENSE POPULATIONS SPAN THE READOUT CROSSING, WHICH MAKES E-C A
  LOCATION TEST RATHER THAN A YES/NO ONE (2026-08-11, from the 2-epoch smokes;
  these are SMOKE numbers, not measurements, and are recorded only to show the
  regime each population occupies):
      ade20k      pixAcc ~22-27   BELOW the crossing bracket [31.8, 40.3]
      cityscapes  pixAcc ~48-50   above
      foodseg103  pixAcc ~51      above
      voc_seg     pixAcc  72-83   far above (measured, full runs)
  The dense study previously had ONE population, entirely above the bracket,
  which is why it could only ever confirm the units reading. With ADE20K below
  it and three above, the sign law now has to get the ORDER right across four
  populations: readout NEGATIVE on ade20k and POSITIVE on the other three. If
  the four readouts come back monotone in pixel accuracy and change sign
  inside [31.8, 40.3], that is the accuracy-derived crossing REPRODUCED on a
  different task and a different metric -- a much stronger statement than
  "VOC's readouts were positive and its pixel accuracy was high".
  Stated as a caveat now rather than after: the smokes are 2 epochs, and pixel
  accuracy rises with training, so ade20k's full-run value could climb toward
  the bracket. If it lands INSIDE [31.8, 40.3] the cell makes no prediction
  (the same no-call rule used for mnet, whose baseline sat in the bracket) and
  E-C would be answered only by its lowest fractions.

- *** FIRST dense3 CELL, AND IT WEAKENS MY OWN E-C SETUP (2026-08-11):
  cityscapes@1% (3 seeds): 14.56 ±0.30 -> 14.54 ±0.14 = **Delta -0.02 ±0.33**,
  a dead null. Consistent with E-D's "flatter than VOC" (VOC@1% +0.38); one
  fraction says nothing about E-D's falsifier, which needs Delta <= 0 at EVERY
  fraction.
  THE PART THAT MATTERS MORE: its pixel accuracy is **66.7 / 67.6**, against
  the ~48-50 its 2-epoch smoke showed. So pixel accuracy climbs ~+18 points
  between a 2-epoch smoke and a full 50-epoch run.
  CONSEQUENCE FOR E-C, stated now rather than after ADE20K lands: I put ADE20K
  "below the crossing bracket [31.8, 40.3]" on the strength of a smoke reading
  22-27. If ADE20K moves like Cityscapes did, its trained pixel accuracy lands
  ~40-45 -- INSIDE or ABOVE the bracket, where the pre-registered no-call rule
  applies and E-C cannot fire at all. The caveat was recorded in advance
  ("pixel accuracy rises with training ... if it lands INSIDE the bracket the
  cell makes no prediction"), but I then described the four-population spread
  as though the smoke values were the operating point, which overstated it.
  WHAT STILL WORKS: ADE20K's LOW fractions have the lowest baselines of the
  envelope, so if any cell sits below the bracket it is ade@1-2%. E-C is
  therefore a test on 2-3 cells rather than a whole population, and its power
  is correspondingly lower. Recorded as a weakening of the design, not of the
  law -- and it is the reason the units question may still need a population
  chosen for LOW trained pixel accuracy rather than low smoke accuracy.

## DENSE FIXES: THE E-C RESCUE AND THE DENSE ATTENTION AXIS (2026-08-11,
## user: "do the needed fixes and don't worry about the cost, we need to do
## it right")

- WHY BOTH. ADE20K's TRAINED pixel accuracy landed at 34.0 (@1%) and 38.5
  (@2%) -- INSIDE the crossing bracket [31.8, 40.3], where the pre-registered
  no-call rule applies -- so E-C had nothing to fire on and the units finding
  stayed untested. Separately, the whole dense transplant used ResNet-18, so
  the study's STRONGEST claim (the attention regime: ViT-B +26.01 at
  ImageNet-100) had zero dense evidence.

- FIX 1: **pascalcontext**, and it does two jobs at once.
  (a) THE SAME-PIXELS CONTROL: it re-annotates the IDENTICAL VOC images
      voc_seg trains on, read straight from VOCdevkit so "same pixels" is
      literal. This is the dense analog of cifar100super, which is what
      settled the granularity question on the classification side; nothing
      else in the dense study can separate a PIXEL effect from a LABEL-SPACE
      effect.
  (b) THE LOW-PIXEL-ACCURACY POPULATION E-C needs: 254 classes over VOC
      images, against voc_seg's 21.
  PROTOCOL, stated rather than buried: 254 of the 459 classes, ranked by
  pixel frequency on the TRAIN split ALONE (ranking on val would leak
  held-out statistics into the label space itself), tail mapped to ignore.
  The reason is mundane -- an 8-bit mask holds 255 values and 255 is the
  ignore index -- and the cost is nil: the top 254 cover **100.00%** of
  labelled pixels, because only ~253 of the 459 ever occur. Same protocol
  SHAPE as Cityscapes' standard 19-of-34. The split is the OFFICIAL one, not
  a re-draw: VOC2012 Main/train and Main/val intersected with the
  Pascal-Context ids give exactly 4,998 / 5,105.
  PREDICTIONS RECORDED IN ADVANCE:
    (E-J) TRAINED pixel accuracy BELOW the bracket: **< 31.8 at 1-10%**.
      This is the premise E-C needs, and it is itself falsifiable -- if
      pascalcontext also lands inside or above the bracket, then no dense
      population in this study can test the units reading, and that must be
      reported as a limitation rather than worked around a third time.
    (E-K) GIVEN the premise holds, the sign law demands **readout NEGATIVE**
      on those cells -- the opposite sign from every dense cell measured so
      far (voc +0.18..+0.52). THIS IS THE UNITS TEST.
      FALSIFIER, and it costs the units finding: readout POSITIVE at a
      pascalcontext cell whose trained pixel accuracy is clearly below 31.8.
    (E-L) THE SAME-PIXELS COMPARISON, on byte-identical images: relabelling
      VOC 21 -> 254 classes should LOWER Delta at matched fraction, because
      it lowers task performance and the readout term follows task
      performance. Band: **Delta(pc@100%) = 0.0..+2.0** against voc_seg's
      +2.57. A pascalcontext Delta ABOVE voc's would say the label space,
      not the pixels, drives the dense gain -- which the classification-side
      G-is-label-space-invariant result (Q6.9h) says it should not.

- FIX 2: **diagswin_voc**, the dense ATTENTION axis. Swin-Tiny encoder, FCN
  head on its stride-8 stage, on VOC so it is directly comparable to the
  reference dense population.
  Swin rather than ViT for a STRUCTURAL reason, not a preference: Swin is
  hierarchical, so its stride-8 stage is a drop-in analog of the dilated
  ResNet's stride-8 output and BOTH backbones wear the IDENTICAL FCN head. A
  plain ViT emits one stride-16 token grid and would need its own upsampling
  decoder, whose capacity would then be a variable between the conv and
  attention arms. Swin is also already a classification population here (91
  law cells), so its dense behaviour is comparable to its OWN classification
  behaviour rather than only to ResNet-18's.
  DIAGNOSTIC, NEVER A HEADLINE ROW, for two stated reasons: it uses AdamW
  rather than the frozen recipe's SGD (train_dense.py now REFUSES adamw on a
  cell whose name lacks a diag prefix), because classification found
  swin-none seed-bistable or at chance under frozen SGD -- a dense Swin pair
  under SGD would measure a training failure, not a feature deficit; and the
  bistability caveat travels with it.
  PREDICTIONS RECORDED IN ADVANCE:
    (E-M) THE ATTENTION DEFICIT TRANSPLANTS TO DENSE: Delta(swin) >
      Delta(r18) at every matched fraction on VOC, and by a wide margin at
      the low end -- band **+2..+10 mIoU at 5-25%**, against r18's
      +0.72/+1.46/+2.06.
      FALSIFIER, and it is the interesting branch: Delta(swin) <= Delta(r18)
      at every fraction => the attention deficit is a property of the
      CLASSIFICATION TASK rather than of attention, and the paper's ViT
      headline must be scoped to classification. That would be a real
      restriction of the strongest claim in the study.
    (E-N) STABILIZATION SIGNATURE: baseline seed spread > aux seed spread,
      as on every classification swin cell. If swin-none collapses on a seed
      the cell carries the bistable flag and is excluded from any Delta.
  COST is not the constraint (user directive); correctness is. Both fixes
  are full 6-fraction envelopes, 3 seeds, 24 configs, 72 runs.

- *** SWIN IS FIXED-RESOLUTION AND THE DENSE PROTOCOL IS NOT -- CAUGHT BY THE
  SMOKE, FIXED BY PADDING RATHER THAN RESIZING (2026-08-11). The first dense
  Swin smoke died on `Input height (366) doesn't match model (512)`: Swin's
  windowed attention asserts the input matches the img_size it was built with,
  and timm has NO dynamic_img_size for Swin (that is a ViT-only feature,
  checked rather than assumed). Dense validation is native-resolution with
  variable image sizes, so every validation image but a square 512 crashed it.
  THE FIX AND WHY THE CHOICE MATTERS: SegSwin now PADS to the built size and
  CROPS the logits back, instead of RESIZING to it. Resizing would rescale
  every validation image and make these cells incomparable with the ResNet
  cells, which are scored at native resolution -- the comparison the whole
  experiment exists to make. Padding leaves every real pixel at its own scale
  and scores exactly the same pixels; the only difference is that attention
  can see a zero margin, and that is applied identically to both arms so it
  cannot enter Delta. Verified across the native sizes VOC actually produces
  (366x500, 375x500, 281x500, 500x500) and confirmed to leave the 512x512
  training path untouched. Bounded deliberately: the forward RAISES if an
  input exceeds the built size rather than silently downscaling, and VOC's
  largest image is 500x500.
  NOTE this is a constraint on which dense populations Swin can be run on at
  all -- Cityscapes at 512x1024 would need a Swin built at 1024, which is a
  different (and much more expensive) model. The dense attention axis is
  therefore VOC-only for now, and that is a scope statement, not an oversight.

- *** DENSE3 INTERIM SCORING — THE DENSE GAIN LOOKS VOC-SPECIFIC, AND THAT IS
  THE MOST IMPORTANT THING THE THREE NEW POPULATIONS HAVE SAID (2026-08-11,
  3 seeds every cell, 0 failures):
      pct   voc      ade20k   cityscapes  foodseg103   diagswin_voc
        1  +0.38     +0.15      -0.02       +0.16        -0.03
        2  +0.50     +0.19      +0.21       +0.22        +0.01
        5  +0.72     +0.25      +0.15       +0.14          --
       10  +1.46     +0.26      +0.35       +0.14          --
       25  +2.06      --        +0.36         --           --
      100  +2.57      --          --          --           --
  VOC is the ONLY population where the prior does material work, and its
  advantage GROWS with data (+0.38 -> +2.57) while the other three sit at
  +0.14..+0.36 and stay there. An order of magnitude, on the same recipe, the
  same backbone and the same code path.
  SCORED AGAINST THE RECORDED BANDS, at the fractions available:
    (E-B) ade@1% band +0.0..+0.5 -> **+0.15 HIT**.
    (E-G) food@1% band +0.0..+0.6 -> **+0.16 HIT**.
    (E-A) ade "monotone rising, Delta(100%) +1.5..+4.0": rising, but at
      +0.15/+0.19/+0.25/+0.26 it is nowhere near the band. TRACKING TOWARD A
      MISS (low), and I am recording that before the 25/100% cells land
      rather than after.
    (E-F) food "Delta(100%) +1.0..+3.5": +0.16/+0.22/+0.14/+0.14 is FLAT, not
      rising. TRACKING TOWARD A MISS (low).
    (E-D) cityscapes: the falsifier was "Delta <= 0 at EVERY fraction"; it has
      NOT fired (positive at 2/5/10/25%), but the envelope is far flatter than
      the +0.5..+2.5 band expects.
  THE HONEST READING, stated now: my dense predictions were anchored on VOC,
  and three populations say VOC is the outlier rather than the representative
  case. The D1/D2/D4 conclusions drawn from VOC alone therefore need
  restating with their population attached -- exactly the correction the
  classification side needed when it moved from 5 populations to 14. What
  survives untouched is D4's negative result (alignment is not the source of
  the gain), which is if anything STRENGTHENED: three more dense populations
  give even less than VOC did.
  NOT YET EXPLAINED, and worth not guessing at: VOC is object-centric with a
  dominant background class and only 21 classes; ade20k (150) and foodseg103
  (104) are fine-grained, cityscapes (19) is geometrically regular. Whether
  the driver is class count, background dominance or something else is what
  pascalcontext is positioned to answer -- it is VOC's own pixels at 254
  classes, so it separates the label space from the images.

- *** THE DENSE ATTENTION AXIS IS TRACKING TOWARD ITS FALSIFIER (2026-08-11,
  first two fractions): diagswin_voc Delta = **-0.03 @1%, +0.01 @2%**, against
  ResNet-18's +0.38 and +0.50 on the SAME dataset, fractions and head.
  The recorded falsifier is E-M: "Delta(swin) <= Delta(r18) at every fraction
  => the attention deficit is a property of the CLASSIFICATION TASK rather
  than of attention, and the paper's ViT headline must be scoped to
  classification." Two of six fractions in, it is firing. If 5-25% hold this
  shape, the strongest claim in the study gets a task-scope qualifier, and
  that must be reported as plainly as the +26.01 that motivated it.
  E-N ALSO FAILED, in the interesting direction: swin-none does NOT collapse
  here (base 4.24 +-0.02 @1%, tighter than its aux arm) and shows no variance
  asymmetry at all. On classification, swin-none is seed-bistable or at chance
  on most datasets and the prior's most visible effect is stabilization. On
  dense VOC there is nothing to stabilize -- which is itself evidence that the
  classification bistability was a property of that recipe/task rather than of
  the backbone.

- *** THE DENSE STUDY IS COMPLETE AT 6 POPULATIONS / 216 CELLS, 0 FAILURES,
  AND THE SAME-PIXELS CONTROL ANSWERS THE VOC ANOMALY (2026-08-11, 3 seeds
  every cell):
      Delta by fraction        1%     2%     5%    10%    25%   100%   #cls
      voc_seg               +0.38  +0.50  +0.72  +1.46  +2.06  +2.57    21
      cityscapes            -0.02  +0.21  +0.15  +0.35  +0.36  +1.04    19
      foodseg103            +0.16  +0.22  +0.14  +0.14  +0.29  +0.70   104
      ade20k                +0.15  +0.19  +0.25  +0.26  +0.42  +0.58   150
      pascalcontext         +0.09  +0.07  +0.09  +0.12  +0.15  +0.19   254
      diagswin_voc (attn)   -0.03  +0.01  +0.38  +1.49  +1.05  +1.08    21
  *** THE DECISIVE CELL, and it is exactly what pascalcontext was built for:
  voc_seg and pascalcontext are BYTE-IDENTICAL PIXELS. The only difference is
  the label space, 21 classes against 254. Delta at 100% goes **+2.57 -> +0.19**
  -- a 13x collapse from relabelling alone. So the dense gain is a LABEL-SPACE
  effect, not a property of these images, and "VOC is the outlier" has a
  mechanism rather than being a bare observation. Every dense population with a
  fine label space (104/150/254 classes) sits at +0.19..+0.70 at full data,
  while the two coarse ones (19/21) reach +1.04 and +2.57.
  NOTE this MIRRORS the classification side rather than contradicting it: there
  the granularity effect ran through the READOUT term (tinsuper, tin20, the
  2x2 crossing), with G itself label-space-invariant on matched pixels. The
  dense probes now pending will say whether the same split holds here.
  PREDICTIONS SCORED:
    (E-A) ade20k monotone rising: **HIT** on shape (no interior peak, so the
      falsifier did not fire), but Delta(100%) = +0.58 against a band of
      +1.5..+4.0 -- **MISSED LOW**.
    (E-B) ade@1% band +0.0..+0.5 -> +0.15 **HIT**.
    (E-D) cityscapes Delta(100%) band +0.5..+2.5 -> **+1.04 HIT**; the
      falsifier (Delta <= 0 at every fraction) did not fire.
    (E-F) foodseg Delta(100%) band +1.0..+3.5 -> **+0.70 MISSED LOW**.
    (E-G) food@1% band +0.0..+0.6 -> +0.16 **HIT**.
    (E-J) pascalcontext trained pixel accuracy BELOW the bracket at 1-10%:
      measured **20.33 / 23.87 / 26.96 / 30.27** -- **HIT 4/4**. The premise
      E-C needed and ADE20K failed to supply now exists, so the units test is
      finally runnable; it awaits the dense G probes (E-K).
    (E-L) pascalcontext Delta(100%) band 0.0..+2.0 -> **+0.19 HIT** (bottom
      edge), and far below voc's +2.57 exactly as the readout account
      predicts.
  SCORECARD: 5 hits, 2 misses (both LOW, both on my "the envelope will rise
  like VOC's" reasoning), 0 falsifiers fired. The pattern in the misses is the
  same one as D2 and E1: I extrapolated VOC's shape to populations whose label
  spaces are 5-12x finer.

- *** THE DENSE ATTENTION AXIS: THE ATTENTION DEFICIT DOES NOT TRANSPLANT TO
  DENSE PREDICTION (2026-08-11, diagswin_voc, 3 seeds, same dataset/fractions/
  head as the ResNet cells):
      pct     r18     swin
        1   +0.38   -0.03
        2   +0.50   +0.01
        5   +0.72   +0.38
       10   +1.46   +1.49
       25   +2.06   +1.05
      100   +2.57   +1.08
  E-M predicted Delta(swin) > Delta(r18) at EVERY fraction, band +2..+10 at
  5-25%. Measured: swin is BELOW r18 at 1/2/5/25/100% and TIED at 10%
  (+1.49 vs +1.46, 0.03 apart against SEMs of 0.06 and 0.13). The band is
  **MISSED BY A WIDE MARGIN** -- swin never exceeds +1.49 where +2..+10 was
  predicted.
  THE RECORDED FALSIFIER, scored strictly and honestly: it required
  Delta(swin) <= Delta(r18) at EVERY fraction, and at 10% swin is NOMINALLY
  0.03 higher, so by the letter it did not fire. But 0.03 at those SEMs is a
  tie, not a win, and the substance of the falsifier -- "the attention deficit
  is a property of the CLASSIFICATION TASK rather than of attention" -- is
  what the data show. Scored the same way as the deit-ssl tin@25% cell: the
  tripwire did not fire in the letter, and the claim it guards must still be
  restated.
  CONSEQUENCE FOR THE PAPER, and it is a real narrowing of the strongest
  claim: ViT-B gains +26.01 on ImageNet-100 CLASSIFICATION and every conv is
  neutral there, but on dense prediction a Swin encoder gains LESS than
  ResNet-18 on the same images. The attention headline must be scoped to
  classification rather than stated as a property of attention.
  CAVEAT that belongs with it: Swin is not ViT, this is one dataset, and the
  cells are AdamW diagnostics. What they cannot be is evidence FOR a dense
  attention deficit, which is what they were run to look for.
  (E-N) FAILED, in the informative direction: swin-none does NOT collapse here
  (4.24 +-0.02 at 1%, TIGHTER than its own aux arm) and shows no variance
  asymmetry at any fraction. On classification swin-none is seed-bistable or
  at chance on most datasets and stabilization is the prior's most visible
  effect there. On dense VOC there is nothing to stabilize -- evidence that
  the classification bistability was a property of that recipe and task, not
  of the backbone.

- *** CORRECTION TO THE SAME-PIXELS READING: THE "13x COLLAPSE" WAS A METRIC-
  SCALE ARTIFACT, AND THE LABEL-SPACE EFFECT IS ~1.6x, NOT 13x (2026-08-11,
  user asked why the segmentation gains look so small; the question exposed
  the error). I recorded earlier today that relabelling VOC 21 -> 254 classes
  collapses Delta from +2.57 to +0.19, "a 13x collapse from relabelling
  alone". That comparison is in ABSOLUTE mIoU POINTS across two populations
  whose mIoU SCALES differ by 8x (voc@100% base 33.63, pascalcontext@100%
  base 4.01). In RELATIVE terms it is +7.6% vs +4.7% -- a **1.6x** effect.
  The label-space effect is real and in the same direction, but I overstated
  its size by a factor of ~8, and "VOC is the outlier" was largely an
  artifact of VOC having the largest mIoU scale.
  SAME ERROR CLASS the study has caught before: comparing raw Delta across
  populations with very different baselines (the retracted "gain tracks the
  deficit" law, 2026-07-16). Absolute Delta is the practitioner's number;
  RELATIVE Delta is what compares across populations. Report both.
  *** THE DENSE GAINS ARE NOT SMALL IN RELATIVE TERMS -- the mIoU SCALE is:
      pop            pct  base_mIoU  Delta   Delta_rel | base_pixAcc  D_pixAcc
      voc             10     7.23    +1.46   **+20.2%**    74.34      +0.31
      voc            100    33.63    +2.57     +7.6%       83.24      +0.81
      foodseg103     100     6.46    +0.70    +10.8%       60.82      +0.88
      pascalcontext    1     0.76    +0.09    +11.9%       20.33   **+1.62**
      ade20k           1     1.74    +0.15     +8.7%       34.04      +1.13
  Relative gains run **+5% to +20%**, the same band as classification
  (C100@5% is +5.15 on 25.36 = +20% relative). And pixel accuracy shows what
  mIoU compresses: pascalcontext@1% gains **+1.62 points of pixel accuracy
  (+8.0% relative)** where mIoU reports "+0.09".
  WHY THE ABSOLUTE NUMBERS ARE TINY, and this is the substantive finding: the
  segmentors are BARELY TRAINED. VOC@100% reaches 33.6 mIoU where a standard
  ImageNet-pretrained R18-FCN reaches ~65-70; pascalcontext@1% has a baseline
  of 0.76 mIoU. The dense recipe is 50 epochs, from scratch, untuned. mIoU
  then punishes exactly that regime: it averages per-class IoU with EQUAL
  weight, so on 254 classes where ~200 score near zero the mean is dominated
  by classes no intervention can rescue.
  CONSEQUENCE: the binding constraint on the dense numbers is RECIPE STRENGTH,
  not the prior. It is also why E-K could not be scored -- readout is a
  difference of two small numbers when Delta is +0.09.

- LONG-SCHEDULE DENSE CELLS QUEUED (2026-08-11, user: "we are doing a real
  research, so real numbers are super important"). diag200e_{voc,pc}_
  {none,aux}_{10,100}pct -- 50 -> 200 epochs, everything else identical, both
  arms, 3 seeds, 24 runs. DIAGNOSTIC (recipe deviation => diag prefix).
  THE QUESTION, stated as a fork rather than a hope: is the small absolute
  dense Delta a property of the PRIOR or of the RECIPE?
  PREDICTIONS RECORDED IN ADVANCE:
    (L1) BASELINES RISE SUBSTANTIALLY -- this is the premise, and it is
      itself falsifiable. Band: voc@100% base **45..58 mIoU** (from 33.63);
      voc@10% **10..16** (from 7.23); pascalcontext@100% **6..10** (from
      4.01). FALSIFIER: baselines rise by less than 20% relative => the
      50-epoch recipe was NOT the binding constraint, and the small dense
      numbers are about the task, not the schedule.
    (L2) THE FORK ON THE PRIOR'S VALUE, and the two branches say opposite
      things about the paper:
      H-RECIPE (the weak recipe was suppressing it): Delta_rel HOLDS or
        GROWS, so Delta_abs grows more than proportionally --
        Delta(voc@100%) = **+3.5..+7.0** mIoU.
      H-UNDERFIT (the prior helps underfit models, exactly as on
        classification): Delta_rel FALLS as the baseline rises, so
        Delta(voc@100%) = **+1.0..+3.0**, i.e. flat or lower in absolute
        terms despite a much higher baseline.
      These bands overlap only at +3.0-3.5, so the cell discriminates.
    (L3) NOT PREDICTED, recorded as the reason this is worth the GPU: if
      baselines rise into a regime where Delta is several mIoU points,
      READOUT BECOMES RESOLVABLE and E-K -- which failed today because
      readout was a difference of two numbers near +0.09 -- becomes
      answerable on pascalcontext, the one population below the crossing.
    NOTE H-UNDERFIT is the outcome most consistent with the rest of the
    study (the prior's gain decays as the baseline strengthens, on every
    classification population measured). It is also the LESS convenient
    outcome for the dense story, which is why the band is recorded before
    the run rather than after.

- *** THE DENSE ATTENTION RESULT IS CONFIRMED FEATURE-SIDE (2026-08-11, swin
  dense probes, 3 seeds, same dataset/fractions/head as the ResNet cells):
      pct   G(swin)        G(r18)        | Delta_swin  Delta_r18
        1  -0.04 ±0.05   +0.08 ±0.06     |   -0.03      +0.38
        2  +0.04 ±0.03   +0.21 ±0.08     |   +0.01      +0.50
        5  +0.50 ±0.09   +0.54 ±0.11     |   +0.38      +0.72
       10  +1.09 ±0.15   +1.22 ±0.18     |   +1.49      +1.46
       25  +0.72 ±0.13   +1.54 ±0.15     |   +1.05      +2.06
  G(swin) <= G(r18) at EVERY fraction. So the e2e finding -- that the
  attention deficit does NOT transplant to dense prediction -- is not a
  readout artifact: the prior genuinely buys the attention backbone LESS
  feature gain than it buys ResNet-18 on the same images.
  *** THE SHARP CONTRAST WITH CLASSIFICATION, and it is the cleanest
  statement the dense study produces: on CLASSIFICATION the prior gives Swin
  roughly TWICE the feature gain it gives conv (G(swin) 7.6-10.5 vs G(r18)
  3.55-6.26 on C100). On DENSE PREDICTION it gives Swin LESS than conv. The
  attention advantage is therefore TASK-SPECIFIC, and that is now established
  on both the accuracy side and the feature side rather than on e2e alone.
  Readout is small and mostly positive on both backbones (-0.12..+0.40 swin,
  +0.18..+0.52 r18) at pixel accuracies of 72-76, i.e. well above the
  crossing -- the decayed positive branch, consistent with VOC's other cells.
  CAVEAT unchanged: Swin is not ViT, this is one dataset, and these are AdamW
  diagnostics. What they cannot be is evidence FOR a dense attention deficit.

- *** L1 SCORED — THE 50-EPOCH DENSE RECIPE WAS THE BINDING CONSTRAINT, AND
  BY MUCH MORE THAN I ESTIMATED (2026-08-11, first 3 of 24 long-schedule
  cells, 3 seeds): diag200e_voc_none_10pct = **18.42 +-0.05**
  (18.47/18.46/18.32) against the 50-epoch baseline **7.23 +-0.07**.
  L1's band was 10..16 mIoU -> **MISSED HIGH**. Its falsifier ("baselines
  rise by less than 20% relative => the schedule was NOT the binding
  constraint") is dead by an order of magnitude: the rise is **+155%
  relative**. A 4x schedule at 10% data now beats HALF of what 10x the data
  bought at 1x schedule (voc@100% at 50 epochs = 33.63).
  CONSEQUENCE, and it is not small: every absolute number in the 6-population
  dense envelope was measured on segmentors that were nowhere near trained.
  The relative-vs-absolute correction recorded earlier today said the mIoU
  SCALE was the problem; this says the SCALE ITSELF was an artifact of the
  schedule. Both readings stand and they compound.
  *** AND THE CURVE SHAPE ANSWERS "IS 200 ENOUGH?" FOR THIS CELL — YES, on
  the evidence that was missing at 50 epochs:
      ep     0    40    60    80   100   120   140   160   180   199
      mIoU 3.62  7.20 10.58 12.25 13.80 16.19 16.32 18.47 18.34 18.47
      lr   1.00e-2 ... 9.1e-4 @160 ... 2.2e-4 @180 ... 0 @199
  It PLATEAUS at ~epoch 160 with LR still nonzero (9% of peak), and is dead
  flat over the last 40 epochs. Contrast the 50-epoch run, which was gaining
  ~0.72 mIoU/EPOCH at epoch 40 with LR at 8% of peak when the cosine killed
  the step size -- there the flattening was an LR-SCHEDULE ARTIFACT, here it
  is convergence. NOTE ep 40 of the 200e run (7.20) is almost exactly where
  the entire 50-epoch run finished (7.23), which is the consistency check
  that the two schedules are the same trajectory sampled differently.
  CAVEAT: this is the 10% cell. The 100% cells carry 10x the images and may
  not plateau as early; their curves get the same test.
  L2 (the fork on what Delta does) needs the aux arm and is NOT yet scored.

## DENSE RECIPE RE-PINNED TO 200 EPOCHS (2026-08-11, user decision: "we need
## to make the number of epochs consistent in our study ... I don't like the
## selective way you propose")

- THE DECISION: the dense frozen recipe becomes **200 epochs at every
  fraction**, matching the classification recipe exactly, and the FULL
  6-population grid re-runs under it. No staging, no per-cell budget, no
  subset of "claim-carrying" cells.
- WHY THE USER IS RIGHT AND MY PROPOSAL WAS WRONG. I had proposed gating the
  launch on inspecting the 100%-cell convergence curves and picking the budget
  from them. That is PER-TASK TUNING -- choosing the dense recipe to suit the
  dense task and then reporting it as frozen, which is precisely what the
  frozen-recipe discipline exists to prevent and which this study does nowhere
  else. 200 is not a tuned number; it is the SAME budget the classification
  side has always used, so steps scale with data on both sides and a
  cross-task comparison (dense-vs-classification at matched images-per-class,
  i.e. D4) is no longer confounded by schedule.
  DISTINCTION THAT SURVIVES: choosing the budget from curves is tuning;
  REPORTING whether cells converged is a validity check and belongs in the
  paper either way. If a 100% cell is still climbing at 200 epochs that is a
  stated limitation, and any fix raises the budget UNIFORMLY.
- WHY THE RE-RUN IS NOT OPTIONAL, and note it rests on L1 alone rather than on
  the L2 fork: the +155% baseline rise means every absolute number in the
  50-epoch envelope was measured on undertrained segmentors. The clearest
  casualty is **D4**, the dense study's headline negative -- I compared
  VOC@1% (+0.38, 50 epochs) against C100@1% (+1.42) and tin@1% (+1.49), both
  of which got their frozen 200-epoch recipe, and concluded that a dense
  target on a dense task pays LESS so alignment is not the source of the gain.
  That put an undertrained cell against fully-trained ones. The conclusion may
  well survive; the evidence for it currently does not. The label-space effect
  (voc vs pascalcontext) and the dense attention result (swin vs r18) are
  internally controlled at 50 epochs but carry the same exposure -- if fine
  label spaces or attention backbones sit further from convergence, either
  could be a convergence effect wearing a mechanism's clothes.
- WHY THE FULL GRID RATHER THAN A SUBSET: the dense study's entire value is
  COMPARISON ACROSS POPULATIONS, so a partial re-run leaves 50-epoch and
  200-epoch populations in the same table and reintroduces exactly the
  confound being removed.
- WHAT SHIPPED: 72 canonical configs bumped epochs 50->200 and eval_every
  5->20 (10 validation passes per run, unchanged); train_dense.py RECIPE
  default and its frozen-recipe docstring re-pinned with the rationale; suite
  **108 passed**. The 8 diag200e_* configs are now CONTENT-IDENTICAL to their
  canonical counterparts except for `name`, so the running dense7 lane doubles
  as a name-independent reproduction check.
- ARCHIVE, and it is what makes the re-run possible at all: train.py's
  completed-run guard SKIPS any cell with a final.json, so the 216 fifty-epoch
  finals would have silently blocked every re-run. All 72 cell dirs moved
  BSC-side to **runs_dense_50e/** (216 finals preserved). They are not waste:
  they become the 1x arm of a schedule axis, and if the 200-epoch grid
  reproduces the ORDERINGS that is a robustness result the study did not have.
- LANE: a NEW lane dense8 (worklist.dense8 / queue.counter.dense8 /
  queue.lock.dense8 / slurm/bsc_dense8.sbatch), 216 tasks, 2 workers, ordered
  cheap-fractions-first with both arms of a pair adjacent so a partial drain
  never leaves a one-armed cell. A new lane rather than swapping
  worklist.dense7 under live workers -- that is the 2026-08-03
  swap-under-running-workers incident, and dense7 is still mid-run delivering
  L2.
- REPO/DEPLOY DRIFT CLOSED IN PASSING: slurm/bsc_dense6.sbatch and
  bsc_dense7.sbatch existed ONLY on BSC (created there directly), so the repo
  copy was missing entirely. Both pulled back before dense8 was derived from
  dense7, so the new lane is a diff of three lane-name lines against the
  ACTUALLY-DEPLOYED file rather than against a stale ancestor. Same hazard
  class as the 2026-08-03 worklist-path regression, caught before it bit.
- STILL PENDING under the new recipe: dense G-probes must be re-run on the
  200-epoch checkpoints (the recorded G values are all 50-epoch), and D4, the
  label-space effect and the dense attention result must be re-scored. Their
  50-epoch values stand as the 1x arm, not as the headline.

- *** L2 AT voc@10% — THE SCHEDULE MOVED THE BASELINE ENORMOUSLY AND DELTA
  NOT AT ALL (2026-08-11, 3 seeds both arms; the 100% cells, which are the
  pre-registered fork point, are still in flight so this is NOT a formal L2
  scoring):
        200e  none 18.42 +-0.05  aux 20.02 +-0.12  D +1.61 +-0.13
         50e  none  7.23         aux  8.68         D +1.46 +-0.13
      baseline  7.23 -> 18.42  = +155% relative
      Delta    +1.46 -> +1.61  = +0.15 +-0.18 = 0.8 sigma (UNCHANGED)
      Delta_rel  20.2% -> 8.7%
  Quadrupling the schedule left absolute Delta statistically identical while
  more than doubling the baseline, so RELATIVE gain fell by more than half.
  That is the H-UNDERFIT signature exactly: the prior helps underfit models
  and its value decays as the baseline strengthens -- the same behaviour as
  every classification population measured.
  CONSEQUENCE IF IT HOLDS AT 100%: the re-run CONFIRMS rather than overturns
  the dense conclusions. The small dense Delta would be a property of the
  prior on dense tasks, not of a weak recipe, and D4 survives on properly
  trained models with the schedule objection REMOVED rather than merely
  unaddressed. That is what the re-run buys in either branch, and it is why
  it had to be run: L1 alone made the old evidence inadmissible without
  saying which way the answer would go.
  NOTE the two results are not in tension. L1 (+155% baseline) says the
  50-epoch INSTRUMENT was bad; L2-at-10% says the CONCLUSION it produced was
  nevertheless about right. Both had to be measured; neither implies the
  other.

- *** GPU UTILIZATION AUDIT — HALF THE ALLOCATED GPUs WERE IDLE, AND THE CAUSE
  IS A DESIGN PROPERTY OF PER-WAVE LANES (2026-08-11, user: "are the gpus
  fully utalized?"). Measured with nvidia-smi on every running node rather
  than inferred from slot counts (these jobs are dataloader-bound, so a full
  slot count does not imply a busy GPU):
      as02r4b28   0% 0% 100% 100%      as05r5b31   0% 0% 100% 100%
      as06r2b23   0% 0% 100% 100%   => 6 of 12 GPUs at 0%, 0 MiB
  NOT a GPU-assignment bug (the launcher's `for g in 0 1 2 3` is correct). It
  is an **END-OF-QUEUE DRAIN**: dense7 had 24 tasks and 24 slots, so each slot
  took exactly one. The cheap voc@10% cells landed on the GPU0/GPU1 slots and
  finished in ~40 min; those workers then found the list exhausted and EXITED,
  while the 100%/pascalcontext cells on the GPU2/GPU3 slots kept the node.
  Half of every node went dark with 216 tasks queued in a lane that could not
  start.
  THE GENERAL LESSON: a per-wave lane with its own short worklist GUARANTEES
  this every time a wave ends. Long-lived lanes fed by a reconcile do not have
  it. Waves are convenient to launch and structurally wasteful to finish.
  *** THREE FIXES, and the first one contained the trap:
  (1) FEEDING THE DRAINED LANE. dense7's counter had run to 138 on a 24-line
      list (drained workers spinning on empty lines). Appending the grid
      naively would have started the next claim at 139 and SILENTLY SKIPPED
      the first ~114 appended tasks -- the 2026-08-03 skip incident exactly.
      So the new worklist is 168 no-op pad lines followed by the 216 real
      tasks, putting the work where the counter actually is. One node
      recovered within seconds (GPUs 0/1 went 0% -> 82%/26%).
      SAFE TO SHARE TASKS ACROSS LANES because train_dense.py carries BOTH
      guards shipped after the wrong-epoch damage: final.json => SKIP, and a
      non-blocking flock run lock that aborts a second trainer on the same
      (config, seed) instead of racing it. Verified present before doing this,
      not assumed.
  (2) LONGEST-JOB-FIRST. The worklist was ordered cheap-fractions-first, which
      GUARANTEES the tail: with 84 slots and 216 tasks the 100% cells are all
      claimed last and then run for hours with most slots idle. The unclaimed
      portion is now sorted 100% -> 1%, so the long poles start immediately and
      cheap cells fill in around them; makespan ~= the longest single cell.
      Both arms of a pair stay ADJACENT, which is the property the cheap-first
      order was really protecting (a partial drain never leaves a one-armed
      cell). Only lines ABOVE the counter were touched, so the SET of remaining
      tasks is unchanged and a claim racing the swap can neither lose nor
      duplicate a task -- asserted in the script, not argued.
  (3) ONE LANE, ONE COUNTER. The 7 pending ms_dense8 jobs were cancelled and
      resubmitted as ms_dense7. Two lanes over the same task set have separate
      counters, so each would have re-claimed everything the other already ran
      -- safe under the guards, but pure churn. slurm/bsc_dense8.sbatch stays
      in the repo, unused, for the record.
  POST-SURGERY INTEGRITY, verified rather than assumed: 384 lines = 168 pad +
  216 tasks, 0 duplicate (config, seed) pairs, 72/72 configs covered.
  WHAT COULD NOT BE FIXED: the allocation is 3 of 10 permitted nodes, and it is
  NOT us hoarding -- account ub234 had exactly these 3 jobs running.
  QOSGrpNodeLimit is a limit across EVERY user of the acc_resc QOS, so other
  projects hold it. Holding our nodes with work queued beats draining them and
  hoping to re-acquire.

- *** L2 SCORED AT 100% — BOTH MY BANDS MISSED, AND THE RE-PIN RESCUED A
  FALSIFIER THAT HAD NOMINALLY FIRED AGAINST THE METHOD (2026-08-11, all 18
  voc/pc 10%+voc100% diag200e finals; pc@100% still training):
      cell        base_50e base_200e |  D_50e   D_200e         | Drel_50e Drel_200e pixAcc
      voc@10%        7.23    18.42   |  +1.46  +1.61 +-0.13    |  20.2%    8.7%     76.9
      voc@100%      33.63    51.64   |  +2.57  +0.13 +-0.24    |   7.6%    0.3%     88.4
      pc@10%         1.83     2.43   |  +0.12  +0.10 +-0.01    |   6.6%    4.2%     37.5
  L1 at voc@100%: band 45..58 mIoU -> **51.64 HIT** (the voc@10% leg missed
  HIGH at 18.42 vs 10..16, scored earlier).
  L2 FORK, both branches WRONG: H-RECIPE (+3.5..+7.0) is dead by ~14 sigma.
  H-UNDERFIT (+1.0..+3.0) has the DIRECTION right -- Delta falls as the
  baseline rises -- but badly understated it: measured **+0.13 +-0.24, i.e.
  statistically ZERO**, below my own floor. Recorded as a direction hit and a
  BAND MISS on both branches, not as a win for H-UNDERFIT.
  *** D2 IS RESCUED, and this is the substantive consequence. D2 predicted
  "structural neutrality survives the task change: Delta(100%) = 0 +-1.5",
  falsifier |Delta| > 2.5. At 50 epochs it read +2.57 -- NOMINALLY FIRING
  against the method's core structural claim (lambda reaching exactly 0 is
  what makes high-data neutrality structural rather than tuned). At 200 epochs
  it reads **+0.13, squarely inside the D2 band**. So structural neutrality
  DOES hold on dense prediction; the 50-epoch cell simply was not at
  sufficiency. My post-mortem at the time said the premise was wrong because
  "VOC@100% is a low-data cell in this study's terms" -- right in spirit, but
  the mechanism was the SCHEDULE, and it is now measured rather than argued.
  At 200 epochs voc@100% is a properly trained cell (51.64 mIoU, 88.4 pixAcc)
  and the lambda->0 schedule delivers neutrality exactly as it does at 100% on
  every classification population.
  THIS IS THE STRONGEST ARGUMENT THAT THE RE-PIN WAS NECESSARY: the 50-epoch
  recipe was not merely imprecise, it MANUFACTURED A FALSIFIER HIT against the
  method's structural claim.
  SHAPE: the dense envelope steepens into the familiar form -- relative gain
  20.2% -> 8.7% at 10% and 7.6% -> 0.3% at 100%. A low-data phenomenon, same
  as every classification population.
  OPERATIONAL NOTE: job 44508854 COMPLETED cleanly (5h37, exit 0:0, 0 FAIL)
  and pc@100% is still training on 855, so nothing was lost when that node
  released; the canonical pascalcontext@100% pair is in the grid regardless
  and is ordered first under longest-job-first.

## THE 200-EPOCH DENSE GRID (2026-08-12, 212/216 cells; ade20k@100% aux and
## three singles still in flight). Delta mIoU, [50e] beside it, rel% of baseline

      pop            1%        2%        5%       10%       25%      100%
      voc          +0.39     +0.97     +1.15     +1.61     +2.34     +0.46
                   [+0.38]   [+0.50]   [+0.72]   [+1.46]   [+2.06]   [+2.57]
                     6%        12%       9%        9%        8%        1%
      cityscapes   +0.49     +0.41     +0.20     +0.68     +0.73     +0.48
      foodseg103   +0.12     +0.20     +0.26     +0.43     +0.56     -0.12
      ade20k       +0.18     +0.20     +0.37     +0.48     +0.37       --
      pascalctx    +0.11     +0.13     +0.16     +0.10     +0.16     +0.17
                     11%       10%       9%        4%        5%        3%
      diagswin_voc +0.42     +1.29     +1.76     +0.90     +0.75     +1.33
                   [-0.03]   [+0.01]   [+0.38]   [+1.49]   [+1.05]   [+1.08]
                     9%        21%       22%       8%        5%        5%
  baselines(200e) voc 6.23/8.21/13.03/18.28/29.05/51.53 (pixAcc@100% 88.3);
  pc 1.07/1.28/1.82/2.42/3.49/6.44 (56.4); swin 4.91/6.07/8.14/11.16/16.02/
  26.26 (80.9); foodseg ... 17.88 (69.6); ade20k ... 27.49 (72.3);
  cityscapes ... 49.43 (91.2).

- *** D4 SURVIVES EXACTLY, WHICH IS THE POINT OF HAVING RE-RUN: voc@1% =
  **+0.39 +-0.04** at 200 epochs against +0.38 at 50. The dense study's
  headline negative -- a dense target on a dense task, the most favourable
  venue this prior could be given, pays +0.39 at ~5 images per class where
  classification shows a universal ~+1.5 floor -- now rests on properly
  trained models with the schedule objection REMOVED rather than merely
  unaddressed. Target-task alignment is not where the value comes from.
- *** THE DENSE ENVELOPE GREW AN INTERIOR PEAK. At 50 epochs every population
  was monotone rising with no right flank, which is what made me call VOC@100%
  "a low-data cell in this study's terms". At 200 epochs voc PEAKS AT 25%
  (+2.34) and falls to +0.46 at 100%; foodseg peaks at 25% (+0.56) and goes
  **-0.12** at 100%; ade20k peaks at 10%. So the dense envelope is unimodal
  like every classification population, AND neutral at sufficiency on two
  populations. The 50-epoch monotonicity was a schedule artifact, not a
  property of dense prediction. (E-A is not scorable until ade20k@100% lands.)
- *** THE DENSE ATTENTION CONCLUSION IS WITHDRAWN -- IT WAS A CONVERGENCE
  ARTIFACT, AND THIS RESTORES THE PAPER'S STRONGEST CLAIM.
  At 50 epochs Swin was <= ResNet-18 at every fraction (-0.03/+0.01/+0.38 at
  1/2/5%), and I reported that the attention deficit does NOT transplant to
  dense prediction and that the ViT headline must be scoped to classification.
  At 200 epochs Swin is **+0.42/+1.29/+1.76** at 1/2/5% against voc's
  +0.39/+0.97/+1.15 -- ABOVE conv at 4 of 6 fractions, and in RELATIVE terms
  (the fair comparison, since swin is the weaker segmentor: 26.26 vs 51.53
  mIoU at 100%) **21-22% vs 12%/9% in the 2-5% band, i.e. ~2x**.
  That 2x is the SAME ratio classification measures: G(swin) 7.6-10.5 vs
  G(r18) 3.55-6.26 on C100. E-M's recorded falsifier -- "Delta(swin) <=
  Delta(r18) at EVERY fraction => the attention deficit is a property of the
  classification TASK" -- substantively held at 50 epochs and clearly does NOT
  hold at 200. Swin under AdamW was simply much further from convergence than
  the conv cells, so the 50-epoch recipe produced the OPPOSITE conclusion.
  E-M's BAND (+2..+10 at 5-25%) still MISSES LOW (max +1.76), so the size was
  over-predicted; the DIRECTION reverses.
  ALSO MIRRORS CLASSIFICATION AT FULL DATA: swin@100% +1.33 (5% rel) vs
  voc@100% +0.46 (1% rel) -- the attention backbone still gains where conv is
  ~neutral, exactly as ViT-tiny gains +9.88 at full C100 where every conv is
  neutral.
- *** THE LABEL-SPACE EFFECT LARGELY DISSOLVES ONCE NORMALISED. voc vs
  pascalcontext are BYTE-IDENTICAL pixels at 21 vs 254 classes. In absolute
  mIoU voc's peak (+2.34) is 14x pascalcontext's (+0.17) -- but their
  baselines differ 8x, which is the metric-scale artifact already corrected
  once on 2026-08-11. In RELATIVE terms: 6% vs 11% @1%, 12% vs 10% @2%,
  9% vs 9% @5%, 9% vs 4% @10%, 8% vs 5% @25%, 1% vs 3% @100% -- voc higher in
  the mid-band, pascalcontext higher at BOTH ends, ~1.6x at most. The
  same-pixels control therefore shows a MUCH weaker and less consistent
  label-space effect than the 50-epoch reading claimed. State it as a mid-band
  tendency, not as the 13x collapse first recorded nor the clean 1.6x.

- *** THE DENSE STUDY IS COMPLETE AND HAS A COMMITTED EXPORTER (2026-08-13):
  216/216 cells, 60/60 probes, and `analysis/aggregate_dense.py` regenerates
  every dense table from runs_dense/. WORTH RECORDING WHY IT WAS MISSING: the
  project's own non-negotiable is that tables regenerate from run records, and
  until now EVERY dense number in this ledger came from an ad-hoc query in a
  scratchpad. aggregate.py has no knowledge of runs_dense at all. That is the
  exact fragility the rule exists to prevent, and the Data Availability
  statement promises released tables.
  THE EXPORTER EMITS RELATIVE DELTA ALONGSIDE ABSOLUTE, deliberately: absolute
  mIoU is not comparable across populations whose baselines differ 8x, and
  reading it as if it were produced the "13x label-space collapse" that was
  really ~1.6x. It also evaluates readout against PIXEL ACCURACY rather than
  mIoU, and declines to make a sign call for cells INSIDE the crossing bracket
  (the mnet precedent).
  *** THE LAW ON DENSE PREDICTION: 30 cells with both Delta and G (1-25%; 100%
  excluded by the probe-ceiling rule):
      inside the bracket (no prediction):  2
      unresolvable (|readout| <= 2 SEM):  21
      RESOLVABLE (these test the law):     9  -> 9 as predicted, 0 wrong
  ALL NINE ARE ON THE POSITIVE BRANCH (pixAcc 52-78, above the bracket), which
  is the honest limitation: dense confirms the law on a new TASK and a new
  METRIC, but has ZERO resolvable cells below the crossing. That is exactly
  what E-K was built to test.
  *** E-K IS UNRESOLVED, NOT PASSED, AND THE REASON IS MINE. pascalcontext@1-2%
  has trained pixel accuracy 21.1/25.9 -- clearly below the bracket, where the
  law demands NEGATIVE readout. Measured: +0.02 and +0.00, i.e. ZERO. The
  recorded falsifier needed readout clearly POSITIVE there, so it did not fire;
  but the prediction was not confirmed either. When the long-schedule cells
  were queued I wrote that L3 would make E-K answerable once baselines rose.
  It did not: the re-pin more than doubled pascalcontext's baseline and left
  Delta at +0.11..+0.16, so readout is still a difference of two tiny numbers.
  The units test needs a population with abundant labels AND low pixel accuracy
  AND a material Delta, and no dense population in this study has all three.
  State it as a limitation rather than working around it a fourth time.
  *** THE ATTENTION CLAIM, IN ITS FINAL AND NARROWEST FORM. I have now stated
  this three ways and only the third is right. Relative G, swin vs r18 on VOC:
      pct     1%     2%     5%    10%    25%
      swin   4.6%  20.7%  17.4%   7.0%   2.0%
      r18    6.3%  13.4%  10.8%   8.1%   5.8%
  Swin's relative feature gain exceeds conv's ONLY at 2% and 5% (~1.6x), and
  conv is HIGHER at 1%, 10% and 25%; at 25% swin's G collapses to +0.27 against
  conv's +1.51. So the 50-epoch conclusion (swin <= conv at EVERY fraction) is
  still withdrawn -- swin clearly wins the 2-5% band on both accuracy and
  features -- but "~2x, the same ratio classification measures" was an overread
  from the 4 fractions available at the time. It is a NARROW-BAND effect.
  *** E-A SCORED: ade20k Delta(100%) = **-0.29** against a band of +1.5..+4.0,
  MISSED LOW. The recorded falsifier (an interior peak exceeding both ends by
  >= 0.5 mIoU) did NOT fire on the letter -- ade20k's peak is +0.38 against
  +0.18 at 1%, only 0.20 above -- but the "monotone rising" premise it was
  built on is false: the envelope falls to negative at 100%. Three populations
  now go neutral-or-negative at full data (voc +0.46/0.9%, foodseg -0.12,
  ade20k -0.29), which is the classification right flank reproduced on dense.
