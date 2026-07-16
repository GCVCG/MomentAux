# Prompt — Deep Research Survey (standalone, no repo access needed)

Copy everything below the line into Claude (deep-research mode). It is
self-contained — it embeds the facts a fresh model needs — and produces a cited
literature survey to underpin the thesis proposal.

---

Produce a **rigorous, well-cited literature survey** (a deep-research report)
that situates a specific new method and identifies its closest prior art and its
genuine novelty. Prioritize peer-reviewed and arXiv sources; give full citations
and links; flag where evidence is thin or contested.

## The method to situate (context)

A research project studies **fixed image-moment filters** — Gabor filters,
Zernike moments, and complex-Gabor quadrature *energy* (phase-invariant) — as an
inductive prior for CNN image classifiers, under a strictly fixed training recipe
(ResNet-18, CIFAR-100, SGD, cosine, 200 epochs), swept across training-set
fractions from 1% to 100% with 3 seeds per point.

Key empirical findings the survey must speak to:
1. When the moments are placed **in the forward path** (as extra input channels
   prepended to the backbone), they help only at very low data (≤5%) and
   *necessarily hurt* in a mid-data "penalty band" (10–25%), becoming neutral at
   full data. This appears intrinsic: the fixed channels compete for input
   bandwidth and get committed during the high-learning-rate phase; making them
   learnable or unfreezing them later removes the benefit or deepens the harm.
2. A new method, **MomentAux**, instead deploys a **plain CNN** (no moment
   channels, zero inference overhead) and uses the moments only as a
   **training-time auxiliary target**: a small head taps an intermediate layer
   and is regressed (MSE, weight λ, added to cross-entropy) onto the fixed moment
   feature maps of the input. This makes the moments a **soft prior that shapes
   the representation** rather than a hard input constraint. Empirically it is
   **positive or neutral at every data scale** (e.g. +2.8 points at 10% data,
   ~0 at 100%), i.e. it *scales with data* and never hurts — unlike the
   forward-path version.

## What to survey (organize into themed sections, each with a synthesis)

1. **Structured / fixed filter banks in deep vision.** Gabor-CNNs and learnable-
   Gabor layers; **scattering transforms** (Bruna & Mallat; Oyallon); wavelet
   and PCANet-style fixed features; where fixed front-ends help and where they
   are known to cap accuracy at scale.
2. **Equivariance & invariance as priors.** Steerable/group-equivariant CNNs
   (Cohen & Welling; Weiler & Cesa); harmonic networks; the trade-off between
   built-in invariance and learned features as data grows. Phase/complex-cell
   (energy model) representations.
3. **Auxiliary losses & feature-prediction training.** Deep supervision;
   auxiliary self-supervised heads; **knowledge / feature distillation** (Hinton;
   FitNets/Romero; attention transfer/Zagoruyko); distilling *hand-crafted or
   fixed* targets into a network's intermediate features. This is the closest
   family — pin down what is and isn't already done.
4. **Priors that "wash out" with data.** Bayesian/regularization framings where a
   prior dominates in the small-sample regime and is overridden by the
   likelihood/data as N grows; inductive-bias-vs-data-scale results (e.g. CNN vs
   ViT sample efficiency); anything formalizing "harmless-at-scale" priors.
5. **Low-data / small-sample image classification** methods and standard
   evaluation protocols (data-fraction sweeps, few-shot vs low-data distinction).

## Deliverable

- A structured report with the sections above, each ending in a short synthesis.
- A **comparison table**: for each closest method, note (representation used,
  where it lives — input/architecture/loss, deployed overhead, and whether it is
  reported to help or hurt as data scales).
- An explicit **novelty & gap statement**: precisely what about MomentAux (a
  *fixed hand-crafted moment* used as an *auxiliary feature-prediction target*
  with a *vanilla deployed model* that *provably scales with data*) is not
  covered by existing work, and which 2–3 papers a reviewer would most likely
  cite as the nearest neighbors.
- A short **reading list** (10–20 must-cite references) with one-line relevance
  notes.
