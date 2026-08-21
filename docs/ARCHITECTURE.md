# Architecture & measurement framework

Diagrams for the MomentStem study. GitHub renders the mermaid blocks natively.
Generated figures referenced here live in [viz/](viz/) (see
`analysis/visualize_features.py`).

## 1. MomentAux — the method under test (the paper's reference configuration)

The deployed network is a **vanilla ResNet** (RGB → logits, +0 inference
params). The moment prior exists only as a training-time auxiliary loss.

```mermaid
flowchart LR
    subgraph deploy["DEPLOYED PATH (train + inference)"]
        X[RGB image] --> C1[conv1] --> L1[layer1] --> L2[layer2] --> L3[layer3] --> L4[layer4] --> GAP[global avg pool] --> FC[fc] --> CE[cross-entropy]
    end
    subgraph train_only["TRAINING ONLY (discarded at deploy)"]
        X --> BANK["fixed Gabor quadrature bank
        (8 pairs, zero trainable params,
        calibrated once on 1024 images)"]
        BANK --> MAG["magnitude maps
        sqrt(even² + odd²)"]
        MAG --> POOL["avg-pool to tap resolution"]
        L3 -- tap --> HEAD["aux head: 1x1 conv
        (head_norm: ‖W‖ projected
        back to init each step)"]
        HEAD --> MSE["MSE"]
        POOL --> MSE
        MSE --> LAM["λ(t): cosine 1.0 → 0.0
        (prior dominates early,
        data takes over; λ=0 at end
        makes late training pure CE)"]
    end
    CE --> LOSS[total loss]
    LAM --> LOSS
```

Key settled design points: tap **layer3** (layer1≈layer2≈layer3 plateau,
layer4 cliff — tap depth is NOT a regime knob), target **magnitude** (beats
structure/steerable/rotinv/gabor/HOG/random/learned-teacher on CIFAR-100;
the ordering compresses on Tiny-ImageNet, where the surviving claim is the
margin over random fixed maps), loss **MSE**
(cosine discards the scale that matters), **λ0 is the data-regime knob**
(2.0 at 1–2%, 1.0 at 3–10%, 0.3 at 15–25%, 0.1 at 100%).

## 2. Forward-path MomentStem (the superseded placement)

```mermaid
flowchart LR
    X[RGB image] --> ID[identity passthrough 3ch]
    X --> G["9 calibrated Gabor kernels
    (k11 or k5, fixed)"]
    ID --> CAT[concat 12ch]
    G --> CAT
    CAT --> NET["ResNet (conv1 widened to 12ch)"] --> Y[logits]
```

Hard input constraint: wins only ≤5% data (energy-magnitude variant: +2.55@1%),
pays a **penalty band** at 10–25% that no variant escapes, washes out at 100%.
MomentAux exists because this placement cannot scale with data.

## 3. The measurement framework — Δ = G + readout

```mermaid
flowchart TD
    subgraph cells["Per cell: baseline & aux pairs (3–10 seeds, frozen recipe)"]
        B[baseline ckpts] & A[aux ckpts]
    end
    B --> E2E["Δe2e = aux − baseline
    (final_test_acc)"]
    A --> E2E
    B --> PROBE["linear probe on FROZEN features,
    FULL labeled train set
    (LBFGS, standardized, identical HPs)"]
    A --> PROBE
    PROBE --> GG["G = aux probe − baseline probe
    (feature gain the prior provides)"]
    E2E --> RO["readout = Δe2e − G
    (what the cell's classifier can cash in)"]
    GG --> RO
    RO --> LAW["SIGN LAW (seed-paired audit):
    readout < 0 below the crossing
    readout ≳ 0 above it
    crossing bracketed in [31.8, 40.3]
    455 resolvable cells, 393 (86.4%) as predicted"]
```

Guard rails: the decomposition is valid only while the probe holds far more
labels than the cell (probe-ceiling rule — stl@20/50% excluded); G is
comparable only within a fixed probe space (the tin 2×2 result); every claim
is machine-checked by `analysis/audit_law_paired.py` (the canonical audit);
`analysis/audit_law.py` is the older closure check and is kept for the record.

## 4. The label-space controls (what moved, what stayed fixed)

```mermaid
flowchart TD
    TIN["tin@1%: 1000 imgs, all 200 classes,
    5/cls, 200-way CE — Δ +1.49"]
    TIN20["tin20: 1000 imgs from 20 classes
    (wnids[::10]), 50/cls, 20-way CE — Δ +5.42"]
    TIN20B["tin20b: disjoint 20 classes
    (wnids[5::10]) — Δ +4.85"]
    TSUP["tinsuper: tin@1%'s EXACT pixels,
    labels = wnid//10 (20 arbitrary groups),
    50/coarse-cls — Δ +1.01"]
    TIN -- "change classes AND pixels" --> TIN20
    TIN20 -- "disjoint class draw:
    everything replicates" --> TIN20B
    TIN -- "change ONLY labels
    (byte-identical pixels)" --> TSUP
    TSUP -.-> V1["fork: G_200 = 2.55 ≈ coarse-trained
    ⇒ TRAINING LABEL SPACE carries the
    ckpt effect, not pixel population"]
    TIN20 -.-> V2["baseline 40.7 (semantic, coherent)
    ⇒ readout +0.84 ⇒ big Δ"]
    TSUP -.-> V3["baseline 14.1 (arbitrary, incoherent)
    ⇒ readout −0.36 ⇒ no boost:
    granularity helps ONLY via
    baseline task performance"]
```

## 5. Reproducibility spine

```mermaid
flowchart LR
    SUB["data/subsets/*.json
    (committed, stratified,
    SUBSET_ALIAS shares indices:
    super→cifar100, tinsuper→tin)"] --> TRAIN["train.py --config cell.yaml --seed N
    (frozen recipe: SGD 0.1, 200 ep,
    batch 128, crop+flip only)"]
    BANKS["pinned filter banks
    (tests/test_bank_regression.py)"] --> TRAIN
    TRAIN --> RUNS["runs/cell/seedN/final.json"]
    RUNS --> AGG["analysis/aggregate.py
    (groups by config NAME)"]
    RUNS --> PROBES["analysis/linear_probe.py
    (--probe-dataset, --shots)"]
    RUNS --> AUDIT["analysis/audit_law_paired.py
    (recomputes Δ, G, readout per seed;
    audits the sign law; writes law_audit.md)"]
    RUNS --> VIZ["analysis/visualize_features.py
    (t-SNE + silhouette, layer3 vs
    target heatmaps, CAM, the bank)"]
```
