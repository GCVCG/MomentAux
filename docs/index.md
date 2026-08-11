# MomentStem — fixed moment priors for data-efficient CNNs

**Ahmad AlMughrabi**<sup>1,\*</sup>
([0000-0002-9336-3200](https://orcid.org/0000-0002-9336-3200)) ·
**Albert Clop**<sup>1</sup>
([0000-0002-0187-6288](https://orcid.org/0000-0002-0187-6288)) ·
**Benjamin Busam**<sup>2</sup>
([0000-0002-0620-5774](https://orcid.org/0000-0002-0620-5774)) ·
**Ricardo Marques**<sup>3</sup>
([0000-0001-8261-4409](https://orcid.org/0000-0001-8261-4409)) ·
**Petia Radeva**<sup>1</sup>
([0000-0003-0047-5172](https://orcid.org/0000-0003-0047-5172))

<sup>1</sup> Universitat de Barcelona ·
<sup>2</sup> Technical University of Munich ·
<sup>3</sup> Universitat Pompeu Fabra ·
<sup>\*</sup> corresponding author
([ahmad.almughrabi@ub.edu](mailto:ahmad.almughrabi@ub.edu))

Fixed Gabor/Zernike moment filters as *training-time priors* on standard CNN
backbones. The champion method, **MomentAux**, regresses an intermediate
layer onto fixed moment-magnitude maps during training only — the deployed
network is a vanilla ResNet (identical FLOPs, +0 inference parameters) — and
is positive at every data scale up to 25%, neutral at 100% by construction.

## Documentation

- **[Glossary & definitions](GLOSSARY.md)** — every term of art defined
  once: λ and its schedule, the Gabor quadrature bank and magnitude maps,
  G, readout, the law, cells/pairs/envelopes, probes and their caveats,
  the observability tool table.
- **[Architecture & measurement framework](ARCHITECTURE.md)** — mermaid
  diagrams: the MomentAux train/deploy split, the forward-path stem, the
  Δ = G + readout decomposition, the label-space control family, and the
  reproducibility spine.
- **[Reading the visuals](VISUALS.md)** — the observability figures
  (t-SNE clustering, layer3 heatmaps vs the moment target, CAMs, the Gabor
  bank) with how-to-read guides and interpretations.
- **[Findings](FINDINGS.md)** — the full question-by-question experimental
  record: every hypothesis, prediction, landing, and retraction.
- **[Released artifacts](ARTIFACTS.md)** — what is in each released asset
  (per-run records, training curves, result tables, campaign logs) and how
  to load it.
- **[Porting notes](PORTING.md)** — what was ported vs corrected from the
  original MomentsNeRF code and why.

## Headline results (3–10 seeds, frozen recipe)

| dataset | best Δ (cell) | envelope shape |
|---|---|---|
| CIFAR-10 | +6.66 @2% (10 seeds) | plateau 1–2%, decays, crosses zero 10–15% |
| CIFAR-100 | +5.15 @5% (10 seeds) | unimodal, peak 5%, neutral @100% |
| STL-10 | +5.92 @10% | tracks CIFAR-10 at matched images |
| Tiny-ImageNet | +2.12 @5% (10 seeds) | flat ≤ +2.2 (readout-suppressed) |
| ViT-tiny, CIFAR-100 | +14.44 @15% | large at *every* scale; +9.88 at 100% |
| ViT-S/16 @224px | +13.00 | ImageNet-100, standard DeiT recipe |
| ViT-B/16 @224px | **+26.01** | the deficit grows with model scale |

The law behind every cell: **Δ = G(features) + readout(baseline accuracy)**
— `G` measured by linear probes on frozen features, readout negative below
the measured crossing bracket `[31.8, 40.3]` and positive above it. Of the
511 cells whose readout is resolvable against its own uncertainty, **404
(79%) fall on the predicted side**. Machine-verified by
`analysis/audit_sign_law.py`.

The law is predictive: registered in advance it called an unseen backbone
family's feature gain from baseline heights alone (Swin-T, four of four in
band) and the ImageNet-scale residual (`|Δ−G| ≤ 1.1` on five of six pairs).

## Reproducing

```bash
python train.py --config configs/diagnostics/<cell>.yaml --seed N
python analysis/aggregate.py          # regenerate all tables from runs/
python analysis/audit_sign_law.py     # re-verify the law from raw files
python analysis/visualize_features.py --pair <none_cell> <aux_cell>
```

Subsets are committed (`data/subsets/`), filter banks are pinned by
regression tests, and every cell's `num_workers` is part of its
reproducibility contract.
