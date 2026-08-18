# A controlled, cost-normalized benchmark of data-efficiency interventions

**When does fusing hand-crafted spectral knowledge with learned
representations pay?**

This repository is the complete instrument, data and analysis behind the
paper of that name. It measures one fixed, hand-crafted source of knowledge,
a pinned bank of Gabor moment-energy targets injected only during training at
about 2% compute overhead, against the leading data-driven alternatives
(SimCLR, SimSiam and DINO self-supervised pre-training, ImageNet transfer,
DeiT-strength augmentation, and learned FitNets teachers) under **one frozen
recipe with committed data subsets**.

| | |
|---|---|
| experimental cells | 3,054<!--computeCells--> |
| training runs | 9,402<!--computeRuns--> (3,854<!--computeGpuHours--> GPU-hours) |
| datasets | 13, across six visual domains |
| backbone families | 9 (ResNet-18/34/50, MobileNetV3, ConvNeXt-T, ViT-tiny/S/B, Swin-T) |
| data scale | 150 to 1,281,167 images; 10 to 1000 classes |
| resolution | 32, 64, 96, 224 px |
| model scale | 5.7M to 86M parameters |

Everything is released: the harness, every configuration, the committed
subset indices, the pinned filter-bank fingerprints, **every per-run JSON
record, every training curve, every campaign log**, and the aggregated result
tables. See **[Released artifacts](#released-artifacts)**.

---

## The three claims

### 1. One law organizes the grid

For every paired cell we measure the end-to-end gain `Δ` and the
feature-level gain `G` from an identically-configured linear probe on frozen
features. Their difference behaves as a function of a single variable, the
**baseline accuracy** of the cell:

```
Δ  =  G  +  readout(base)
```

`readout` is strongly negative at low baselines (the features improve more
than a data-starved classifier can express), crosses zero in a measured
bracket of `[31.8, 40.3]` points, and decays back toward zero with
sufficiency. Audited scope-wide by `analysis/audit_law_paired.py`, which is
the canonical script: `readout = Δ − G` is formed **per seed**, because `Δ`
and `G` come from the same checkpoints and treating them as independent
overstates the uncertainty by a median factor of 1.8.

| | |
|---|---|
| cells with paired `Δ` and `G` | 1,724<!--auditAllPaired--> |
| in law scope (prior, from scratch, ≥3 seeds per arm) | 958<!--auditScope--> |
| inside the crossing bracket (no prediction made) | 94<!--auditBracket--> |
| unresolved (`|readout| ≤ 2·SEM`) | 409<!--auditUnresolved--> |
| **resolvable, these test the law** | **455<!--auditResolvable-->** |
| sign as predicted | **393<!--auditCorrect--> (86.4<!--auditRate-->%)** |
| wrong side | 62<!--auditWrong--> |

Two earlier figures for this table are superseded and we name them so nobody
cites them from an old copy. **96%** came from an independent-SEM audit
(`analysis/audit_sign_law.py`, still in the tree because it answers a
different question) whose uncertainty formula the paper withdrew. **78.9<!--auditPrevRate-->%**
came from before the scope filter was repaired: it tested the exported
`pretrained` field against the strings `true`/`1` while the exporter writes
`yes`, so the ImageNet-transfer cells leaked into an audit whose scope has
always excluded them. Those cells agree with the law 18.0<!--auditExclRate-->%
of the time on their own, which is why removing 50<!--auditExclResolvable-->
resolvable cells moved the headline as much as it did. The manuscript
discloses this in full.

The law is **predictive, not descriptive**. Registered in advance, it called
the feature gain of a backbone family it had never seen (Swin-T: predicted
band `+7..+11`, measured `10.5 / 9.3 / 10.1 / 7.6`, four of four in band),
eight cells on four unseen domains (seven in band, eight of eight correct in
sign), and the ImageNet-scale residual (`|Δ−G| ≤ 1.1` on five of six pairs).

### 2. Fusion outcomes follow the *currency* each source supplies

| fusion | currencies | outcome | evidence |
|---|---|---|---|
| prior + augmentation | structure + invariance | **stack** | `Δ` amplified 1.4–2.4×; `G` rises 14.9 → 22.2 |
| prior + effective SSL | structure ≈ invariance | **substitute** | combo ≤ best single; equal `G` |
| prior + ineffective SSL | structure + ~nothing | stack | full gain recovered on SimSiam |
| prior + ImageNet init | structure vs. mature features | **interference** | −16..−18 points at full auxiliary strength, vanishing below it; carried by `G`; null under domain shift |
| augmentation + SSL | invariance + invariance | substitute | the DeiT recipe collapses SimCLR's margin |

The practical corollary: a cheap linear probe of what each candidate source
contributes predicts whether combining them is worth anything, **before** any
joint training is run.

### 3. Attention carries a feature deficit that grows with model scale

Every convolutional backbone is neutral at data sufficiency, including at
1.28M images (`Δ = +0.04 ± 0.07`). Vision transformers are not:

| cell | baseline | with prior | Δ |
|---|---|---|---|
| ViT-tiny, ImageNet64, 1.28M imgs | 48.24 | 51.48 | **+3.23** |
| ViT-S/16, ImageNet-100 @224px, DeiT recipe, 100 ep | 65.39 | 78.39 | **+13.00** |
| ViT-B/16, ImageNet-100 @224px, DeiT recipe, 100 ep | 43.33 | 69.34 | **+26.01** |
| ViT-S/16, ImageNet-100, 200 ep | 79.47 | 83.99 | **+4.52** |
| ViT-B/16, ImageNet-100, 200 ep | 75.31 | 82.02 | **+6.71** |

The ordering holds at both budgets: doubling the schedule shrinks both gains
(the baselines catch up), but the larger model keeps the larger deficit
(+6.71 against +4.52 at 200 epochs, 2.4 SEM apart). Every pre-registered
scale falsifier (F1, F2, F3, G1, G2, G3, G4) is dead.

---

## The method under test: MomentAux

`momentstem/aux.py`. The moments are a **training-only soft prior**, never a
forward-path stem. The deployed model is a plain backbone, RGB to logits,
identical FLOPs, **zero extra inference parameters**. During training a 1×1
convolution taps the third stage and is regressed onto fixed phase-invariant
Gabor magnitude maps, with `λ` cosine-decayed to **exactly zero**, so
neutrality at full data is structural rather than tuned.

One configuration (`λ0 = 1.0`, magnitude target, third-stage tap, head-norm
on) transplants across every dataset and backbone without retuning.

**The controls are the contribution.** On CIFAR-100 at 10%:

| aux target | what it isolates | Δ |
|---|---|---|
| moment magnitude (ours) | phase-invariant energy | **+2.81** |
| HOG (MaskFeat descriptor) | would any hand-crafted descriptor do? | +1.37 |
| learned teacher (FitNets) | is a *learned* target better? (2× cost) | +0.16 |
| random fixed maps | is it just "any auxiliary regression"? | +0.14 |
| oriented edges (raw Gabor) | is it the filters, or their energy? | −0.24 |

A learned teacher costing an entire extra model does approximately nothing,
across a full eight-point envelope. The gain is the moment structure
specifically.

---

## Released artifacts

Attached to the tagged release (see the repository's Releases page). Sizes
are compressed; `SHA256SUMS` accompanies them.

| asset | contents |
|---|---|
| `run-records.tar.gz` | every run's `final.json` (config, accuracy, parameter and FLOP accounting, environment) and every probe record behind `G`: `linear_probe*.json` (full-train, fixed-shot, final-epoch and cross-label-space), `dense_probe.json`, `det_probe.json`, plus `robustness.json`, `cifair.json`, `head_forms_5shot.json` and `per_class_delta.json`. Records span `runs/`, `runs_turing/`, `runs_dense/` and `runs_det/`; per-seed records sit under `<tree>/<cell>/seed<N>/` and per-cell probes under `<tree>/<cell>/` |
| `training-curves.tar.gz` | per-epoch `metrics.csv` for 9,674 of the 10,030 released runs (the 356 without one, and why, are listed in `docs/ARTIFACTS.md`) |
| `result-tables.tar.gz` | the aggregated tables: `all_results.csv` (one row per cell), `results_by_portion.csv`, the combined workbook, `law_audit.md`, `summary.md`, `summary.tex`, the segmentation tables (`dense_results.csv`, `dense_law.csv`, `dense_summary.md`), the detection tables (`det_results.csv`, `det_summary.md`, `det_decompose.json`) and the per-analysis JSON records |
| `logs.tar.gz` | campaign and wave logs, including the cluster work-queue logs |
| `configs-and-subsets.tar.gz` | every cell configuration and the committed subset indices |

Every table and figure in the paper is generated from these assets by the
scripts in `analysis/` — five of them, listed in `docs/ARTIFACTS.md`, not one.
The per-run records are the raw evidence behind them, so any number in the
paper can be traced from the printed table back to the individual training
run that produced it.

**What is not released:** model checkpoints (223 GB) and the source image
datasets. All datasets are public and cited in the paper; the committed
subset indices reproduce the exact image selection without redistributing
images.

---

## Reproducing

```bash
pip install -r requirements-study.txt   # pinned lock for the study environment
python scripts/make_subsets.py --check  # verify committed subsets reproduce
python -m pytest tests/ -q              # contracts; must pass before training
```

```bash
python train.py --config configs/<cell>.yaml --seed N   # one cell, one seed
python analysis/linear_probe.py --run-dir runs/<cell>   # the feature gain G
python analysis/audit_law_paired.py                     # re-run the law audit
```

Regenerating the released tables takes five commands, not one, and
`aggregate.py` is **not** the one that writes the CSVs — it writes only
`summary.{md,tex}` while printing a long table, which is an easy way to
believe the CSVs are current when they are not. The full sequence, and the
reason it is a sequence, is in
[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md#regenerating-the-papers-tables-from-the-release).

Each run writes `metrics.csv` (per-epoch), `final.json` (config, accuracy,
`fvcore` parameter and FLOP accounting, environment) and checkpoints under
`runs/<cell>/seed<N>/`. No external services are involved.

### What makes a comparison in this repository trustworthy

- **The recipe is frozen.** SGD momentum 0.9, lr 0.1, weight decay 5e-4,
  cosine schedule, 200 epochs, batch 128, crop and flip only. Cells that must
  deviate carry a diagnostic namespace enforced by a guard in `train.py` and
  never enter a headline table. Both arms of such a pair share the deviation,
  so the paired difference stays valid.
- **Subsets are committed** under `data/subsets/`. Every intervention
  consumes byte-identical images.
- **Filter banks are pinned.** `tests/test_bank_regression.py` fingerprints
  their numerical values, so no measurement can drift with a library version.
- **`num_workers` is part of the contract.** PyTorch seeds each worker's
  augmentation RNG from a base seed plus worker index, so changing the count
  redraws the augmentation stream. It is pinned per cell and recorded per run.
- **Predictions were registered before results.** Every experimental wave was
  launched with numeric bands and explicit falsifiers recorded in `CLAUDE.md`
  before any result existed. Four major predictions missed; all four are
  reported in the paper.
- **Run-directory guards.** A completed run is a no-op, concurrent trainers
  on the same cell abort on an exclusive lock, and checkpoint writes are
  atomic.

---

## Documentation

- **[docs/index.md](docs/index.md)** — overview and headline results
- **[docs/GLOSSARY.md](docs/GLOSSARY.md)** — every term defined once: `λ` and
  its schedule, the Gabor bank and magnitude maps, `G`, readout, cells,
  pairs, envelopes, probes and their caveats
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — architecture and
  measurement diagrams
- **[docs/VISUALS.md](docs/VISUALS.md)** — the explainability figures, with
  how-to-read guides
- **[docs/FINDINGS.md](docs/FINDINGS.md)** — the question-by-question
  experimental record: every hypothesis, prediction, landing and retraction
- **[docs/ARTIFACTS.md](docs/ARTIFACTS.md)** — what is in each released
  asset and how to load it
- **[PORTING.md](PORTING.md)** — what was ported and what was corrected
- **`CLAUDE.md`** — the dense chronological working ledger, including every
  pre-registered prediction and every failure post-mortem

## Layout

```
momentstem/aux.py         MomentAux: the training-only prior under test
momentstem/stem.py        forward-path MomentStem (superseded; kept as a control)
momentstem/controls.py    learned / random-fixed / gabor-learn control stems
momentstem/backbones.py   timm backbones, small-image surgery, FLOP accounting
data.py                   14 dataset loaders, committed subsets, CIFAR-C
train.py                  the single training entry point (with run guards)
scripts/simclr_pretrain.py, simsiam_pretrain.py, dino_pretrain.py
analysis/linear_probe.py  the feature gain G (full-train and fixed-shot)
analysis/aggregate.py     runs/ -> markdown + LaTeX tables
analysis/audit_law_paired.py  scope-wide audit of the law (canonical)
analysis/audit_sign_law.py  the older independent-SEM audit, superseded
analysis/export_results_csv.py  the released result tables
configs/                  one YAML per cell (generated, committed)
tests/                    contracts: pinned banks, tensor layout, overhead,
                          subset determinism, metric reference
slurm/                    cluster work-queue: worker, big lane, probe lane
paper/                    the manuscript sources and figure generators
```

## Citation

```bibtex
@article{almughrabi2026fusing,
  author  = {AlMughrabi, Ahmad and Marques, Ricardo and Radeva, Petia},
  title   = {When does fusing hand-crafted spectral knowledge with learned
             representations pay? A controlled, cost-normalized benchmark
             and its organizing law},
  journal = {Information Fusion},
  year    = {2026}
}
```
