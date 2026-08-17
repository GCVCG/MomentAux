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

A controlled, **cost-normalized benchmark of data-efficiency interventions**,
and the measurable rule that organizes its results.

One frozen recipe and fixed image subsets are held constant while the
intervention varies: a free hand-crafted spectral prior (**MomentAux**),
self-supervised pre-training (SimCLR, SimSiam, DINO), ImageNet transfer,
DeiT-strength augmentation, and learned FitNets teachers — each reported
against a **declared multiple of baseline training compute**, and every
pairwise combination measured. 3,052<!--computeCells--> classification
configurations over 9,390<!--computeRuns--> runs, on 13 datasets and 9
backbones from 500 to 1.28M images and 32 to 224 px, plus segmentation and
detection transplants.

MomentAux itself regresses an intermediate layer onto fixed
moment-magnitude maps *during training only*: λ decays to exactly zero, so
the deployed network is byte-identical to the baseline (same FLOPs, +0
inference parameters) and neutrality at full data is structural rather than
tuned.

## Paper

Submitted to *Information Fusion*. The manuscript, its generated number file,
the response letter and the submission checker live in `paper/` and `docs/`;
`python paper/check_submission.py` verifies the package against the journal's
limits before it is assembled.

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
471<!--auditResolvable--> cells whose readout is resolvable against its own
seed-paired uncertainty, **402<!--auditCorrect--> (85.4<!--auditRate-->%)
fall on the predicted side** — 94.3<!--auditBelowRate-->% below the crossing,
where the account makes a strong prediction. Machine-verified by
`analysis/audit_law_paired.py`, which is the canonical audit; the older
`audit_sign_law.py` uses an independent-SEM formula the paper withdrew, and
its higher figure should not be quoted.

The law is predictive: registered in advance it called an unseen backbone
family's feature gain from baseline heights alone (Swin-T, four of four in
band) and the ImageNet-scale residual (`|Δ−G| ≤ 1.1` on five of six pairs).

## What the benchmark says about the alternatives

Costs are declared multiples of baseline training compute, which is the whole
point of the comparison: the prior is ~1.02×, self-supervised pre-training
2× at its published budget and 5× at four times that.

| regime | outcome |
|---|---|
| convolutional, SSL at 2× | prior competitive; SSL ahead in the mid-data band |
| convolutional, SSL at 5× | **SSL wins at every fraction 1–25%.** The prior's convolutional case is a *cost* case, not an accuracy case, and we say so without qualification |
| small ViT under the DeiT recipe, SSL at 2× | prior wins at every CIFAR-100 fraction |
| small ViT under the DeiT recipe, SSL at 5× | SSL at 5%, level at 10%, **prior at 25%** — the ordering flips with data, on both populations tested |
| prior + augmentation | **stack** (different currencies: structure vs. nuisance-invariance) |
| prior + effective SSL | **substitute** — the combination beats neither single arm |
| prior + ImageNet init | **tax**, up to −17 points, and the frozen-feature probe shows the damage is to the features themselves |

The practical rule that falls out: probe what each candidate source supplies
*before* combining them. Sources with the same currency do not add.

## Off classification

Every number above is top-1 accuracy, so the prior was transplanted to two
other tasks. Both headline results are negative, which is why they are worth
reporting.

- **Semantic segmentation**, 6<!--densePops--> populations (VOC, Cityscapes,
  FoodSeg103, ADE20K, Pascal-Context, and a Swin-T arm),
  216<!--denseCells--> runs at the same 200-epoch budget the classification
  recipe uses. The envelope keeps its shape and its structural neutrality at
  full data, but a dense target on a dense task — the venue most favourable
  to this prior — pays **less** than classification does at matched
  supervision density (+0.39<!--denseVocOneDelta--> mIoU at ~5 images per
  class, against a universal ≈+1.5 floor on classification). Target–task
  alignment is not where the value comes from.
- **Object detection** on the same VOC images is a **null** end to end and
  under the frozen-feature probe (`G(fg_acc)` = −0.16<!--detOneGFgAcc--> at
  1%, against a pre-registered falsifier of +1.5), so the null is the prior,
  not a weak head. A +0.84 AP50 gain was withdrawn after tracing it to one
  seed whose box-regression branch had collapsed.

Whatever the prior supplies is cashed fully by a whole-image classifier,
partly by a per-pixel one, and not at all by a coordinate regressor.

## Reproducing

```bash
python train.py --config configs/diagnostics/<cell>.yaml --seed N
python analysis/aggregate.py          # regenerate all tables from runs/
python analysis/audit_law_paired.py   # re-verify the law from raw files
python analysis/visualize_features.py --pair <none_cell> <aux_cell>
```

Subsets are committed (`data/subsets/`), filter banks are pinned by
regression tests, and every cell's `num_workers` is part of its
reproducibility contract.
