# Glossary & definitions

Every term of art in this study, defined once. Cross-referenced from
[FINDINGS.md](FINDINGS.md), [ARCHITECTURE.md](ARCHITECTURE.md), and
[VISUALS.md](VISUALS.md). Terms **bold** on first mention elsewhere are
defined here.

---

## The method

**Moment filters / the bank.** A fixed set of image filters inspired by
classical image *moments* (Gabor and Zernike families). The study's champion
bank is 8 **complex-Gabor quadrature pairs**: for each orientation/scale, an
*even* (cosine-phase) and an *odd* (sine-phase) kernel — two filters 90° out
of phase, like the real and imaginary parts of one complex filter. The banks
are numerically pinned by `tests/test_bank_regression.py`; changing one
invalidates every existing run. Rendered in
[VISUALS.md §1](VISUALS.md#1-the-prior-itself--the-gabor-quadrature-bank).

**Gabor filter.** A localized oriented sinusoid under a Gaussian envelope —
the standard model of early-visual-cortex simple cells. It responds to
edges/gratings of a particular orientation and spatial frequency at a
particular location.

**Magnitude map / energy map (the aux target).** For each quadrature pair,
convolve both kernels with the image's luma and take
`sqrt(even² + odd²)` per pixel. This is the local *oriented energy*:
how much structure of that orientation/scale is present, **regardless of
the phase** of the underlying edge (a bright-to-dark edge and a
dark-to-bright edge give the same response). One map per pair → an
8-channel spatial target. "Gabor-magnitude maps", "moment maps",
"energy-magnitude" and "the (aux) target" all refer to this. The target
sweep (FINDINGS §4) showed phase-invariant magnitude beats raw oriented
edges, rotation invariants, structure tensors, HOG, learned teachers, and
random fixed maps as an aux target.

**Calibration.** Once per run, the bank is fed a deterministic batch of
1024 training images (index order, eval transform, no labels) and each
channel is rescaled so all channels respond at comparable magnitude
(`stem_calibrate: true`). Identical for every cell/seed of a dataset, so
it can never leak label information.

**MomentStem (forward-path).** The *original* placement, now superseded:
the fixed bank runs in the network's input path (RGB + 9 filter channels
concatenated, conv1 widened). Helps ≤5% data, but pays the **penalty
band** — an accuracy *cost* at 10–25% data that no fixed forward-path
variant escapes — because a pre-committed channel occupies input bandwidth
abundant data wants back.

**MomentAux (the champion).** The moments as a **training-only auxiliary
loss** on a completely vanilla backbone. A 1×1-conv **aux head** taps an
intermediate layer (**tap**, settled: `layer3`) and is regressed onto the
avg-pooled magnitude maps with MSE. Total loss = `CE + λ(t) · MSE`. At
deployment the aux head is discarded: the shipped model is a plain ResNet —
identical FLOPs, **+0 inference parameters**. Implemented in
`momentstem/aux.py`.

**λ (lambda), λ0, λ_final, the λ schedule.** λ is the weight multiplying
the aux MSE term in the training loss — how hard the prior pushes relative
to cross-entropy. The champion uses a **cosine schedule**: λ starts at
**λ0** (`weight` in configs) and decays along a half-cosine to **λ_final**
(`weight_final`), which is **exactly 0** — so late training is pure CE and
neutrality at full data is *structural*, not tuned. **λ0 is the
data-regime knob**: best values are ~2.0 at 1–2% data, 1.0 at 3–10%, 0.3
at 15–25%, 0.1 at 100%. Three separate experiments showed λ0 cannot push
the gain past what **G** supplies — it protects from harm but cannot
manufacture feature gain.

**head_norm.** After every optimizer step, the aux head's weight matrix is
rescaled to its initial norm. This removes the **scale degeneracy**: the
aux loss ‖W·f − t‖² is invariant under (features → f/c, head → c·W), so
SGD can "solve" it by collapsing the tapped features and inflating the
head — which is exactly how ResNet-50 failed before the mechanism was
traced. head_norm is a safe always-on default (free on R18/R34, required
on R50).

**Tap.** The layer whose features the aux head reads (`layer3` for
ResNets; `stages.2` for ConvNeXt; `blocks.8` for ViT-tiny — matched by
depth fraction and spatial size). Tap depth is **not** a regime knob:
layer1≈layer2≈layer3 everywhere, layer4 is the only cliff.

---

## The measurement framework

**Cell.** One experimental condition = one config file = (dataset, subset
fraction, architecture, method variant), run at N seeds. Named like
`tin_aux_1pct` (dataset tin, MomentAux, 1% subset). **Pair** = a cell and
its baseline twin (`*_none_*`), differing only in the aux loss.

**Δ / Δe2e ("delta", end-to-end gain).** Mean test top-1 of the aux cell
minus its baseline, over seed pairs. The ± on Δ is the SEM of a difference
of means, not the per-seed σ.

**Envelope.** Δ as a function of data fraction for one dataset — e.g. tin's
envelope +1.49/+1.81/+2.13/+1.65/+0.10/−0.42 at 1/2/5/10/25/100%.

**The frozen recipe.** SGD momentum 0.9, lr 0.1, weight decay 5e-4, cosine
lr schedule, 200 epochs, batch 128, random-crop + horizontal-flip only.
Headline cells never deviate. Because epochs are fixed, **step count is
tied to data fraction** — 1% = 600 steps, 100% = 78,000 — which is why
naive cross-fraction comparisons confound data with compute.

**diag\* prefix.** A config whose name starts with `diag` deviates from the
frozen recipe (different optimizer, head, init, epochs...) and may never
enter a headline table. Enforced in `train.py` for AdamW, `head:`, and
`init_from`.

**Subsets.** The low-data fractions use *committed* index files
(`data/subsets/*.json`), drawn once, stratified per class, identical for
every method variant. `scripts/make_subsets.py --check` verifies they
reproduce byte-identically.

**Linear probe / G (feature gain).** Freeze a trained checkpoint, extract
penultimate features for the **full** train split, fit an L2-regularized
multinomial logistic regression (LBFGS) on them, evaluate on test
(`analysis/linear_probe.py`). **G** = probe accuracy of the aux checkpoint
minus probe accuracy of its baseline: how much better the aux model's
*features* are when the classifier bottleneck is removed. A diagnostic,
never a headline number (the probe sees labels the cell never had).

**readout.** The part of Δ the features don't explain:
`readout = Δe2e − G`. It measures how well the cell's *own* classifier
(trained on few labels) cashes in the feature gain.

**The law.** `Δe2e = G(pixels, images, training-label-space) +
readout(baseline task performance)`. G is a property of what the network
saw; readout depends only on how good the baseline is.

**The sign law / the crossing.** readout is *negative* when the baseline's
test accuracy is below ≈30–35% (few labels → the classifier can't exploit
better features, so realized gain < G) and *positive* above it (aux
features are easier to read — realized gain > G). 25 clean cells, zero
violations, including ViT.

**Shots / k-shot probe.** A probe trained on only k labels per class
(`--shots`), measuring how much of G is visible at a given label budget.
Q7.3: e2e realizes almost exactly what a same-budget linear probe realizes.

**Probe-ceiling rule.** The Δ = G + readout decomposition is only
meaningful while the probe has far more labels than the cell; when a cell
approaches its own probe ceiling (stl@50%, cub@100%), "readout" loses
meaning and is excluded.

**Realization R (historical).** An earlier multiplicative form Δ = R × G.
Falsified at 50 labels/class (R measured >100%) and replaced by the
additive law above. Appears only in dated FINDINGS entries.

**Granularity controls.** Datasets built to move *one* variable:
`cifar100super` (CIFAR-100's images, official 20 coarse labels),
`tin20`/`tin20b` (20 of tin's 200 classes, two disjoint draws),
`tinsuper` (all tin images, 20 *arbitrary* positional groups),
`tinsem` (same, but groups from the WordNet hypernym-path sort —
the semantic-vs-arbitrary control).

**G curve.** G as a function of image count for one dataset (fixed probe
space). The e2e envelope peak sits at the G peak wherever readout is flat.

**ckpt-set effect / probe-space effect.** From the 2×2 crossing on tin:
*what label space a checkpoint was trained under* changes its measured G
(fine-trained pairs carry more), and *what label space the probe uses*
changes measured G (coarse probes read more gap). G numbers are comparable
only within a fixed probe space.

---

## Infrastructure & conventions

**Wave.** One detached batch of runs (local `setsid nohup` script or slurm
job) named `<name>_wave`, logging to `logs/<name>_wave.log` (local) or
`slurm/logs/` (turing), ending with a `*_COMPLETE` marker line.

**Predictions on record.** Every wave's expected outcomes and falsifiers
are written into `CLAUDE.md` *before* launch; landings are scored against
those bands verbatim. This is what makes a "confirmed" here mean something.

**num_workers contract.** DataLoader worker count is part of
reproducibility: augmentation draws from per-worker RNG, so changing
`num_workers` re-draws the augmentation stream (unbiased, but seeds stop
being paired). Never change it on a cell with completed seeds.

**best.pt / final.json / metrics.csv.** Per run: best-accuracy checkpoint;
final summary (config, accuracies, param/FLOP accounting, environment);
per-epoch log. Since 2026-07-20 metrics.csv also logs `ce_loss`,
`aux_loss`, `lambda`, and `tap_std` (the collapse diagnostic) for aux runs.

**Observability tools** (all diagnostics; none feed headline tables):

| tool | question it answers |
|---|---|
| `analysis/linear_probe.py` | how good are the features (G), at what label budget (--shots), under which label space (--probe-dataset)? |
| `analysis/head_forms.py` | does the readout *head form* (linear/cosine/NCM) change the measured gap at matched labels? |
| `analysis/visualize_features.py` | what does the prior *do*: bank rendering, layer3-vs-target heatmaps + alignment r, t-SNE + silhouette, CAMs |
| `analysis/per_class_delta.py` | *which classes* gain or lose, by name, across seeds |
| `analysis/training_dynamics.py` | *when* in training the gap opens; λ/lr schedules; loss components; tap-std collapse check |
| `analysis/aggregate.py` | regenerates every results table from runs/ |
| `analysis/audit_law.py` | machine-checks every number in the law table (exit 1 on failure) |
| `analysis/eval_cifair.py`, `eval_robustness.py` | duplicate-contamination control; corruption robustness (CIFAR-100-C) |

**Aggregation rule.** Tables group by config *name* (the cell), never by
parsed fields — variants sharing a stem differ only in `stem_kwargs`.
