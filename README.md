# MomentStem: fixed moment filters as a prior for CNNs

**→ [docs/FINDINGS.md](docs/FINDINGS.md) is the record: every question asked,
its answer, the evidence, and its status** (settled / open / falsified /
retracted). `CLAUDE.md` is the denser chronological working ledger.

**Documentation** (all render directly on GitHub, mermaid included):
[docs/index.md](docs/index.md) — overview & headline results ·
[docs/GLOSSARY.md](docs/GLOSSARY.md) — **every term defined** (λ, the Gabor
magnitude maps, G, readout, cells, probes, waves...) ·
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — architecture + measurement
diagrams · [docs/VISUALS.md](docs/VISUALS.md) — the observability figures,
explained and embedded · `analysis/audit_law.py` — machine-verifies every
number in the law table from raw run files.

**Observability toolkit** (`analysis/`, all diagnostics): `linear_probe`
(feature gain G, shot-limited, cross-label-space), `head_forms`
(linear/cosine/NCM readouts at matched labels), `visualize_features`
(bank, target-alignment heatmaps, t-SNE, CAMs), `per_class_delta`
(which classes gain, by name, across seeds), `training_dynamics`
(when the gap opens; λ/lr schedules; loss components; collapse check),
`audit_law` (machine-checks the law table). See the
[GLOSSARY tool table](docs/GLOSSARY.md#infrastructure--conventions).

## Where the study actually landed

**MomentAux** (`momentstem/aux.py`): the moments are a **training-only soft
prior**, not a forward-path stem. The deployed model is a **plain ResNet** —
RGB→logits, identical FLOPs, **+0 inference params**. During training only, a
1×1-conv head taps `layer3` and is regressed onto fixed phase-invariant moment
maps (MSE·λ + CE), with λ cosine-decayed to **exactly 0** so that neutrality at
full data is structural rather than tuned.

It is **positive at every data scale** (CIFAR-100, ResNet-18, 3 seeds):

| data | 1% | 3% | 5% | 10% | 25% | 100% |
|---|---|---|---|---|---|---|
| **Δ top-1** | +1.91 | +3.68 | **+5.30** | +4.14 | +0.97 | +0.15 (n.s., neutral) |

and it transplants with no retuning across depth and dataset — one λ0=1.0
(+`head_norm`) gives +3.9…+4.3 on R18/R34/R50; CIFAR-10 gives +6.37@1% / +6.66@2% (10 seeds).

**The controls are the contribution.** A random target gives ≈0; a *learned*
FitNets teacher costing a whole extra model gives ≈0; HOG gives about half. The
gain is the moment structure specifically — see FINDINGS §4.

## How it got here (the negatives matter)

The original framing — fixed moments **in the forward path** — is **dead as a
scaling answer**, and the ledger keeps it dead. Such stems help below 5% but
*cost* accuracy at 10–25% (the "penalty band"), and that band survived every
attempt to explain it away: calibration, ZCA, an 8× step budget, prior-as-init,
prior-as-warmup, and nonlinear/rotation-invariant/2nd-order features alike. Any
fixed pre-committed input channel occupies bandwidth abundant data wants back.
Moving the prior **off the forward path** is what made it scale.

The original three hypotheses were answered, and two of them failed:
**H1** (data efficiency) holds only in the aux formulation; **H2** (capacity
substitution — R18+moments ≈ R34) is **dead**; **H3** (robustness gain under
CIFAR-C) is **null**.

See [PORTING.md](PORTING.md) for exactly what was ported from momentsnerf,
what changed, and one important finding about the original Zernike code.

## The controls ARE the contribution

**Aux targets** (`moment_aux.stem`, the live question — is it the moments?):

| target | what it isolates | Δ@10% |
|---|---|---|
| `energy-magnitude` | the method (phase-invariant moment energy) | **+2.81** |
| `random-fixed` | "any aux regression / deep supervision?" | +0.14 |
| `teacher` (FitNets) | "just distillation? is a LEARNED target better?" | +0.16 |
| `hog` (MaskFeat) | "would any hand-crafted descriptor do?" | +1.37 |

**Forward-path stems** (the original design; superseded — see FINDINGS §1):

| stem | what | what it isolates |
|---|---|---|
| `none` | vanilla backbone | control |
| `moments-sum` | fixed Gabor+Zernike, 3-ch output | pretrained-stem-compatible variant |
| `moments-cat` | fixed bank, 27-ch output (RGB+9 Gabor+15 Zernike) | headline variant |
| `energy-*` | fixed nonlinear energy (magnitude/rotinv/structure/steerable/invariants) | which invariance, if any, survives past 5% |
| `learned` | ONE plain conv, params/FLOPs matched to moments-cat within 2% | prior vs. depth/compute |
| `random-fixed` | moments-cat architecture, frozen random filters, matched norms | structure vs. fixedness |
| `gabor-learn` | moment-initialised, trainable | is fixing a feature or a bug? |

All under ONE recipe (deviations are a bug): SGD momentum 0.9, lr 0.1, wd
5e-4, cosine, 200 epochs, batch 128, crop+flip only, >=3 seeds per cell.

## Setup

```bash
pip install -r requirements.txt
python scripts/make_subsets.py --check   # verify committed subsets reproduce
python -m pytest tests/ -q               # must pass before any training
```

## Reproducing any number

```bash
python train.py --config configs/<cell>.yaml --seed N   # one run
python eval_robustness.py --run-dir runs/<cell>/seedN --cifar-c-root <dir>
python analysis/aggregate.py                            # regenerates ALL tables
```

Each run writes `metrics.csv` (per-epoch), `final.json` (config + accuracy +
fvcore param/FLOP accounting), and checkpoints under `runs/<cell>/seed<N>/`.
No external services involved.

## Cluster (turing)

```bash
python slurm/submit_grid.py --dry-run     # inspect the plan
python slurm/submit_grid.py               # full grid, 2 chained job streams
sbatch slurm/train.sbatch configs/<cell>.yaml 0 1 2   # one cell by hand
```

## Layout

```
momentstem/stem.py        MomentStem (fixed Gabor+Zernike buffers)
momentstem/controls.py    LearnedStem, random-fixed, gabor-learn, registry
momentstem/backbones.py   timm backbones + CIFAR surgery + fvcore accounting
data.py                   CIFAR-100/STL-10, committed subsets, CIFAR-C loader
train.py                  the single training entry point
eval_robustness.py        CIFAR-C evaluation (no retraining)
analysis/aggregate.py     runs/ -> markdown+LaTeX tables (mean+/-std, mCE)
configs/                  one YAML per cell (generated, committed)
tests/                    contracts: frozen filters, shapes+content, overhead
                          match, subset determinism, metric reference, layout
slurm/                    sbatch template + grid submitter (turing)
```

## Out of scope (v1)

NeRF/rendering, ImageNet-scale training, ViT backbones, hyperparameter
search, RandAugment. Scattering front-end (kymatio; Oyallon et al.) is the
closest published rival -- cited, optionally run later.
