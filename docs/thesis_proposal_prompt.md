# Prompt — Build a Bachelor Thesis Proposal (for Claude Code, run inside this repo)

Copy everything below the line into Claude Code while your working directory is
the `MomentsCNNEncoder` repository. It will read the codebase, do literature
research, and produce a complete, defensible thesis proposal.

---

You are helping me write a **bachelor thesis proposal** that continues an
existing research codebase. Work in this repository. Be rigorous, cite real
literature, and ground every methodological claim in the empirical results
already in this repo. Do not invent results — where you propose new experiments,
label them as proposed.

## Step 1 — Understand what already exists (read before writing)

Read these first and summarize them back to me in your own words before drafting:
- `CLAUDE.md` — the project conventions and the full "State of findings" ledger
  (frozen recipe, falsified mechanisms, the MomentAux breakthrough).
- `PORTING.md` — provenance of the moment filter banks.
- `momentstem/stem.py` (fixed Gabor/Zernike forward-path stem),
  `momentstem/energy.py` (nonlinear energy stems incl. `magnitude`),
  `momentstem/aux.py` (**MomentAux** — the soft training-only prior on a vanilla
  backbone; this is the intended centerpiece of the thesis).
- `train.py`, `momentstem/backbones.py`, `analysis/aggregate.py` and
  `results/summary.md` — how cells are run and aggregated.

## Step 2 — The scientific story so far (the thesis builds on this)

The project studies fixed image-moment filters (Gabor / Zernike / complex-Gabor
energy) as an inductive prior for CNNs under a strictly frozen training recipe
(SGD 0.9, lr 0.1, wd 5e-4, cosine, 200 epochs, ResNet-18, CIFAR-100), swept over
data fractions 1%–100% with 3 seeds/cell. Established, reproducible findings:

1. **Forward-path moments are a hard constraint.** A fixed moment stem prepended
   to the backbone helps only at very low data (≤5%) and *necessarily* costs
   accuracy in a mid-data "penalty band" (10–25%), washing to ~0 at 100%.
   Kernel size is a regime knob (k11 best 1–3%, k5 best 5–25%). Every attempt to
   soften this failed: learnable-from-init, freeze-then-unfreeze, ZCA, richer
   banks, multi-mask readout — all documented negatives.
2. **Phase-invariant energy is the best low-data forward-path feature.**
   `energy-magnitude` (complex-Gabor quadrature energy) reaches +2.5 to +3.5 pts
   at 1–5% but is a sharp specialist: it *never recovers*, staying −3.9 at 100%.
3. **MomentAux — the breakthrough this thesis extends.** Instead of feeding
   moments into the forward path, deploy a **vanilla ResNet** (RGB→logits, zero
   inference overhead) and, *during training only*, attach a small head that taps
   an intermediate layer and regresses onto the fixed moment maps (MSE × λ added
   to cross-entropy). The moments **shape the representation** instead of
   **occupying the input**. Result (magnitude target, λ=0.3): positive-or-neutral
   at **every** data scale — e.g. +2.81 at 10% (where every forward-path stem
   lost) and +0.00 at 100%. λ has a clean knee: gain peaks near λ≈0.5, the
   "never-hurts" high-data safety holds through λ=0.3. This is the first
   placement that *scales with data* and is deployable at zero cost.

Mechanistic account to foreground: forward-path priors compete for input
bandwidth and get committed during the high-LR phase (irreversible); an
auxiliary soft prior is a regularizer that abundant data simply overrides.

## Step 3 — Research the literature (use web search/deep research)

Situate MomentAux and find the closest prior art. Cover, with real citations:
- Fixed/structured filter banks in CNNs: Gabor-CNNs, **scattering transforms**
  (Bruna & Mallat), **steerable/equivariant CNNs** (Cohen & Welling; Weiler),
  wavelet/PCA-Net.
- Auxiliary/self-supervised feature-prediction losses, **feature distillation**,
  hint/attention transfer (FitNets, Romero; attention transfer, Zagoruyko),
  and distilling *hand-crafted* targets into networks.
- Inductive-bias-vs-data trade-offs; **priors that wash out with data**; low-data
  / small-sample image classification; regularization theory relevant to why a
  soft prior helps at low data and is harmless at high data.
- Position clearly: what is genuinely novel here (a fixed-moment *auxiliary*
  prior with a vanilla deployed model that provably scales with data), and what
  is adjacent.

## Step 4 — Produce the proposal

Write `docs/thesis_proposal.md` (and a LaTeX version `docs/thesis_proposal.tex`
if straightforward). ~6–10 pages. Sections:
1. **Title** (2–3 options) and one-paragraph abstract.
2. **Problem & motivation** — the deployability gap: fixed priors help low-data
   but hurt at scale; we want moments that scale.
3. **Background & related work** — from Step 3, with a comparison table placing
   MomentAux against scattering/steerable/distillation/aux-loss lines.
4. **Preliminary results** — summarize the existing findings above as the
   foundation (cite the repo's `results/` and CLAUDE.md; present the MomentAux
   scaling curve and the λ knee).
5. **Research questions / hypotheses** — sharp and falsifiable.
6. **Proposed methodology & experiments** (feasible on ONE RTX 3090 in a
   bachelor-thesis timeframe). Draw from, and prioritize among:
   - *Design space of MomentAux*: moment target (Gabor vs magnitude vs Zernike
     vs steerable), tap layer(s), multi-layer aux, aux-loss form (MSE / cosine /
     contrastive), spatial vs pooled target, λ schedule/annealing.
   - *Generalization of the scaling claim*: other backbones (ResNet-50, a small
     ViT), other datasets (CIFAR-10, STL-10, Tiny-ImageNet / ImageNet-100
     subsets), other data-fraction sweeps.
   - *Mechanism / analysis*: why soft-prior scales and hard-input doesn't —
     representation-similarity (CKA) between aux and baseline features, when the
     aux loss goes slack, feature-learning dynamics across the high-LR phase.
   - *Robustness & transfer*: CIFAR-100-C corruptions (repo has the harness),
     OOD, linear-probe transfer.
   - *Beyond classification* (stretch): detection/segmentation aux prior.
7. **Evaluation plan & baselines** — the existing controls (none / learned /
   random-fixed) plus forward-path stems as the contrast; metrics, seeds, and
   the frozen-recipe discipline.
8. **Timeline** (~4–6 months, milestone chart) and **risk/mitigation**.
9. **Expected contributions** — a deployable, scale-safe way to inject
   hand-crafted priors; an empirical + mechanistic account; open code.

## Constraints & style
- Honor the repo's discipline: frozen recipe, 3 seeds/cell, aggregate by config
  name, report negatives as plainly as positives.
- Scope experiments to a single RTX 3090 (each CIFAR-100 cell ~6–35 min).
- Distinguish clearly between **established results** (cite the repo) and
  **proposed work**. No fabricated numbers.
- Ask me before finalizing if any scoping decision is genuinely open (e.g.
  which datasets to commit to).
