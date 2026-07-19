# MomentStem — fixed moment priors for data-efficient CNNs

Fixed Gabor/Zernike moment filters as *training-time priors* on standard CNN
backbones. The champion method, **MomentAux**, regresses an intermediate
layer onto fixed moment-magnitude maps during training only — the deployed
network is a vanilla ResNet (identical FLOPs, +0 inference parameters) — and
is positive at every data scale up to 25%, neutral at 100% by construction.

## Documentation

- **[Architecture & measurement framework](ARCHITECTURE.md)** — mermaid
  diagrams: the MomentAux train/deploy split, the forward-path stem, the
  Δ = G + readout decomposition, the label-space control family, and the
  reproducibility spine.
- **[Reading the visuals](VISUALS.md)** — the observability figures
  (t-SNE clustering, layer3 heatmaps vs the moment target, CAMs, the Gabor
  bank) with how-to-read guides and interpretations.
- **[Findings](FINDINGS.md)** — the full question-by-question experimental
  record: every hypothesis, prediction, landing, and retraction.
- **[Porting notes](PORTING.md)** — what was ported vs corrected from the
  original MomentsNeRF code and why.

## Headline results (3–10 seeds, frozen recipe)

| dataset | best Δ (cell) | envelope shape |
|---|---|---|
| CIFAR-10 | +6.66 @2% | plateau 1–2%, decays, crosses zero 10–15% |
| CIFAR-100 | +5.30 @5% | unimodal, peak 5%, neutral @100% |
| STL-10 | +5.92 @10% | tracks CIFAR-10 at matched images |
| Tiny-ImageNet | +2.13 @5% | flat ≤ +2.2 (readout-suppressed; see findings) |

The law behind every cell: **Δe2e = G(features) + readout(task performance)**
— G measured by linear probes on frozen features, readout sign governed by
baseline accuracy (negative below ≈30%, positive above ≈34%; 20 cells, zero
violations). Machine-verified by `analysis/audit_law.py`.

## Reproducing

```bash
python train.py --config configs/diagnostics/<cell>.yaml --seed N
python analysis/aggregate.py          # regenerate all tables from runs/
python analysis/audit_law.py          # re-verify the law from raw files
python analysis/visualize_features.py --pair <none_cell> <aux_cell>
```

Subsets are committed (`data/subsets/`), filter banks are pinned by
regression tests, and every cell's `num_workers` is part of its
reproducibility contract.
