# MomentStem: do fixed orthogonal-moment filters substitute for CNN capacity?

We previously showed (MomentsNeRF, BMVC 2026 submission) that fixed
Gabor+Zernike moment filters in front of a ResNet-34 encoder reach parity
with a ResNet-152 encoder inside a PixelNeRF pipeline. This repo tests that
claim on recognition tasks, where the encoder is isolated from view sampling
and volume rendering.

**Falsifiable hypotheses** (any may FAIL; failures get reported as plainly as
successes):

- **H1 (data efficiency):** backbone+moments beats vanilla backbone, and the
  gap grows as training data shrinks.
- **H2 (capacity substitution):** ResNet-18+moments approaches vanilla
  ResNet-34 (and R34+moments approaches R50) under an identical recipe.
- **H3 (robustness):** the gain is larger under corruption shift
  (CIFAR-10-C/100-C mCE) than in-distribution.

See [PORTING.md](PORTING.md) for exactly what was ported from momentsnerf,
what changed, and one important finding about the original Zernike code.

## The six stems (the controls ARE the contribution)

| stem | what | what it isolates |
|---|---|---|
| `none` | vanilla backbone | control |
| `moments-sum` | fixed Gabor+Zernike, 3-ch output | pretrained-stem-compatible variant |
| `moments-cat` | fixed bank, 27-ch output (RGB+9 Gabor+15 Zernike) | headline variant |
| `learned` | ONE plain conv, params/FLOPs matched to moments-cat within 2% | prior vs. depth/compute |
| `random-fixed` | moments-cat architecture, frozen random filters, matched norms | structure vs. fixedness |
| `gabor-learn` | moment-initialised, trainable | is fixing a feature or a bug? |

All under ONE recipe (deviations are a bug): SGD momentum 0.9, lr 0.1, wd
5e-4, cosine, 200 epochs, batch 128, crop+flip only, >=3 seeds per cell.

## Setup

```bash
pip install -r requirements.txt
python scripts/make_subsets.py     # once; subset JSONs are committed
python -m pytest tests/ -q         # must pass before any training
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
