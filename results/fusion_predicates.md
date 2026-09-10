# Fusion predicates, derived on every measured combination (block B-1)

Regenerate: `python analysis/fusion_predicates.py --build --score` (plus `--extract` for the checkpoint-based predicates and `--shots25` for the label-efficiency ones).

Context (see the CLAUDE.md wave-1/wave-2 entries, 2026-08-19/20): the published
Algorithm-1 predicate (REL, G-magnitude agreement) called 1 of 9 held-out pairs;
the strength-asymmetry rule derived from wave 1 scored 1/6 then 0/2 one family
over; fix-set overlap S(A,B) was at chance. This pass asks the question on every
measured combination at once, with leave-one-FAMILY-out as the honesty criterion
(family = source-type pair). OUTCOME LABELS use the recorded test-split exporter
values (results/all_results.csv), 2-SEM rule; 'INTERFERE' = SUBSTITUTE(cost).
Checkpoint predicates: last.pt (best.pt fallback), max 3 seeds, penultimate
features on a fixed stratified 2,000-image test subset.

## Derivation set: 134 pairs, 7 families

| family | n | STACK | SUBSTITUTE | INTERFERE | resolved | ckpt-complete |
|---|---|---|---|---|---|---|
| aug|mae | 5 | 3 | 2 | 0 | 3 | 5 |
| aug|simclr | 7 | 3 | 3 | 1 | 4 | 7 |
| prior|aug | 21 | 16 | 0 | 5 | 21 | 15 |
| prior|dino | 6 | 4 | 1 | 1 | 5 | 6 |
| prior|simclr | 13 | 1 | 4 | 8 | 9 | 13 |
| prior|simsiam | 5 | 4 | 1 | 0 | 4 | 5 |
| prior|transfer | 77 | 0 | 10 | 67 | 67 | 2 |

### Pairs

| pair | dA | dB | GA | GB | dC | GC | combo-best | sigma | outcome | REL | CKA(A,B) | nCKA | pc_r | S | leff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aug|mae:cifar100@1:vit_tiny | 0.08 | 0.06 | -1.89 | 0.59 | 1.02 | -0.28 | +0.94 | +2.3 | STACK | 1.31 | 0.27 | 0.55 | 0.29 | 0.59 | 1.93 |
| aug|mae:cifar100@2:vit_tiny | 1.07 | 1.46 | 0.98 | 2.57 | 2.18 | 3.11 | +0.72 | +1.2 | SUBSTITUTE | 0.62 | 0.27 | 0.58 | 0.16 | 0.58 | 1.91 |
| aug|mae:cifar100@5:vit_tiny | 0.30 | 8.21 | -0.22 | 10.44 | 9.02 | 11.63 | +0.81 | +0.3 | SUBSTITUTE | 1.02 | 0.25 | 0.50 | 0.25 | 0.59 |  |
| aug|mae:cifar100@10:vit_tiny | 3.81 | 13.04 | 3.57 | 13.65 | 18.06 | 18.75 | +5.02 | +8.7 | STACK | 0.74 | 0.30 | 0.58 | 0.37 | 0.70 | 0.75 |
| aug|mae:cifar100@25:vit_tiny | 3.77 | 14.15 | 4.82 | 13.97 | 21.97 | 21.08 | +7.82 | +5.8 | STACK | 0.65 | 0.37 | 0.67 | 0.31 | 0.82 | 1.51 |
| aug|simclr:cifar100@1:vit_tiny | 0.08 | 0.65 | -1.89 | 1.64 | 0.80 |  | +0.15 | +0.3 | SUBSTITUTE | 1.87 | 0.27 | 0.50 | 0.12 | 0.55 | 1.61 |
| aug|simclr:cifar100@5:vit_tiny | 0.30 | 8.09 | -0.22 | 9.90 | 9.39 | 12.86 | +1.30 | +0.9 | SUBSTITUTE | 1.02 | 0.25 | 0.48 | 0.33 | 0.57 |  |
| aug|simclr:cifar100@10:vit_tiny | 3.81 | 13.30 | 3.57 | 13.82 | 19.59 | 20.95 | +6.29 | +7.0 | STACK | 0.74 | 0.30 | 0.57 | 0.51 | 0.70 | 0.77 |
| aug|simclr:cifar100@25:vit_tiny | 3.77 | 15.08 | 4.82 | 14.24 | 23.81 | 22.51 | +8.73 | +27.0 | STACK | 0.66 | 0.36 | 0.64 | 0.36 | 0.80 | 1.56 |
| aug|simclr:eurosat@5:resnet18 | -7.68 | -1.01 | -2.91 | -2.31 | -1.66 |  | -0.65 | -1.4 | SUBSTITUTE | 0.21 | 0.83 | 0.88 | 0.71 | 0.90 | 2.61 |
| aug|simclr:food101@5:resnet18 | 9.46 | 9.56 | 11.26 | 7.84 | 14.05 |  | +4.49 | +4.2 | STACK | 0.30 | 0.49 | 0.64 | 0.50 | 0.70 | 0.65 |
| aug|simclr:food101@10:resnet18 | 9.26 | 5.45 | 9.50 | 3.51 | 5.84 |  | -3.42 | -4.4 | SUBSTITUTE(cost) | 0.63 | 0.51 | 0.69 | 0.61 | 0.95 | 0.72 |
| prior|aug:cifar100@1:vit_tiny | 1.40 | 0.08 | 5.00 | -1.89 | 3.28 | 8.17 | +1.88 | +6.6 | STACK | 1.38 | 0.25 | 0.43 | 0.23 | 0.48 | 1.80 |
| prior|aug:cifar100@2:vit_tiny | 3.24 | 1.07 | 6.76 | 0.98 | 7.08 | 13.40 | +3.84 | +17.2 | STACK | 0.86 | 0.27 | 0.49 | 0.19 | 0.55 | 1.90 |
| prior|aug:cifar100@3:vit_tiny | 6.22 | 1.34 | 10.10 | 1.29 | 10.82 | 17.23 | +4.60 | +18.7 | STACK | 0.87 | 0.26 | 0.46 | 0.28 | 0.54 | 1.11 |
| prior|aug:cifar100@5:vit_tiny | 9.35 | 0.30 | 12.96 | -0.22 | 16.62 | 21.29 | +7.27 | +18.5 | STACK | 1.02 | 0.24 | 0.43 | 0.26 | 0.54 |  |
| prior|aug:cifar100@7:vit_tiny | 10.99 | 1.63 | 13.75 | 1.48 | 20.32 | 23.19 | +9.33 | +23.0 | STACK | 0.89 | 0.26 | 0.48 | 0.37 | 0.60 | 0.96 |
| prior|aug:cifar100@10:vit_tiny | 13.27 | 3.81 | 15.01 | 3.57 | 24.82 | 25.88 | +11.55 | +50.5 | STACK | 0.76 | 0.29 | 0.54 | 0.46 | 0.67 | 0.67 |
| prior|aug:cifar100@15:vit_tiny | 14.45 | 2.84 | 15.45 | 3.43 | 27.43 | 27.43 | +12.98 | +30.8 | STACK | 0.78 | 0.32 | 0.59 | 0.35 | 0.72 | 1.44 |
| prior|aug:cifar100@25:vit_tiny | 13.67 | 3.77 | 13.88 | 4.82 | 28.82 | 27.53 | +15.15 | +44.1 | STACK | 0.65 | 0.36 | 0.63 | 0.31 | 0.79 | 1.47 |
| prior|aug:cifar100@100:vit_tiny | 9.89 | 10.78 | 9.82 | 10.99 | 24.64 | 23.89 | +13.86 | +49.0 | STACK | 0.11 | 0.43 | 0.73 | 0.61 | 1.09 | 0.34 |
| prior|aug:eurosat@5:resnet18 | 0.84 | -7.68 | 0.30 | -2.91 | -2.24 | -0.78 | -3.08 | -9.7 | SUBSTITUTE(cost) | 1.10 | 0.85 | 0.88 | 0.26 | 0.87 |  |
| prior|aug:eurosat@10:resnet18 | 1.23 | -0.13 | 0.62 | -0.01 | 0.68 | 0.46 | -0.55 | -5.5 | SUBSTITUTE(cost) | 1.02 | 0.85 | 0.88 | -0.61 | 0.96 |  |
| prior|aug:eurosat@25:resnet18 | -0.12 | 0.12 | -0.01 | 0.31 | -0.25 | 0.25 | -0.37 | -3.5 | SUBSTITUTE(cost) | 1.03 | 0.87 | 0.90 | 0.05 | 1.14 |  |
| prior|aug:food101@5:resnet18 | 5.63 | 9.46 | 6.27 | 11.26 | 13.78 | 15.92 | +4.32 | +11.3 | STACK | 0.44 | 0.60 | 0.74 | 0.61 | 0.74 | 0.32 |
| prior|aug:food101@10:resnet18 | 3.18 | 9.26 | 2.87 | 9.50 | 7.44 | 8.28 | -1.82 | -6.6 | SUBSTITUTE(cost) | 0.70 | 0.55 | 0.74 | 0.61 | 0.96 | 0.37 |
| prior|aug:food101@25:resnet18 | -0.69 | 3.05 | -0.71 | 4.05 | 0.62 | 2.29 | -2.43 | -8.0 | SUBSTITUTE(cost) | 1.18 | 0.51 | 0.70 | 0.31 | 1.15 | 0.33 |
| prior|aug:tin@1:vit_tiny | 1.28 | 0.54 | 4.77 | 1.03 | 3.56 | 9.41 | +2.28 | +24.1 | STACK | 0.78 |  |  |  |  |  |
| prior|aug:tin@2:vit_tiny | 3.29 | 0.11 | 6.79 | -1.21 | 6.67 | 13.01 | +3.38 | +39.0 | STACK | 1.18 |  |  |  |  |  |
| prior|aug:tin@5:vit_tiny | 6.77 | 2.27 | 10.22 | 1.74 | 12.44 | 17.30 | +5.67 | +22.2 | STACK | 0.83 |  |  |  |  |  |
| prior|aug:tin@15:vit_tiny | 10.72 | 4.23 | 11.96 | 4.33 | 20.93 | 21.86 | +10.21 | +23.5 | STACK | 0.64 |  |  |  |  |  |
| prior|aug:tin@25:vit_tiny | 10.49 | 7.28 | 10.62 | 7.65 | 22.38 | 21.82 | +11.89 | +29.2 | STACK | 0.28 |  |  |  |  |  |
| prior|aug:tin@100:vit_tiny | 7.17 | 15.91 | 7.17 | 15.06 | 22.51 | 21.28 | +6.60 | +4.9 | STACK | 0.52 |  |  |  |  |  |
| prior|dino:cifar10@10:vit_tiny | 14.02 | 10.86 | 11.97 | 8.55 | 13.90 |  | -0.12 | -0.5 | SUBSTITUTE | 0.29 | 0.65 | 0.96 | 0.87 | 1.45 | 0.10 |
| prior|dino:cifar100@5:vit_tiny | 9.35 | 2.97 | 12.96 | 4.13 | 10.37 | 13.76 | +1.02 | +3.1 | STACK | 0.68 | 0.30 | 0.79 | 0.75 | 0.92 | 0.01 |
| prior|dino:cifar100@10:vit_tiny | 13.27 | 8.13 | 15.01 | 8.24 | 14.19 | 15.02 | +0.92 | +5.2 | STACK | 0.45 | 0.34 | 0.87 | 0.85 | 1.05 | 0.12 |
| prior|dino:cifar100@25:vit_tiny | 13.67 | 12.47 | 13.88 | 12.01 | 14.99 |  | +1.32 | +3.8 | STACK | 0.13 | 0.41 | 0.90 | 0.90 | 1.15 | 0.07 |
| prior|dino:dtd@50:vit_tiny | 9.84 | 1.36 | 9.82 | 0.95 | 7.32 |  | -2.52 | -3.8 | SUBSTITUTE(cost) | 0.90 | 0.23 | 0.57 | 0.50 | 0.78 | 0.21 |
| prior|dino:food101@50:vit_tiny | 10.49 | 14.78 | 9.93 | 13.99 | 16.19 |  | +1.41 | +3.5 | STACK | 0.29 | 0.36 | 0.87 | 0.83 | 1.12 | 0.04 |
| prior|simclr:cifar10@2:resnet18 | 6.66 | 8.07 | 5.52 | 6.56 | 9.13 |  | +1.06 | +2.4 | STACK | 0.16 | 0.77 | 0.86 | 0.77 | 0.92 | 0.02 |
| prior|simclr:cifar100@5:resnet18 | 5.15 | 9.32 | 6.11 | 9.39 | 7.54 | 7.90 | -1.78 | -7.1 | SUBSTITUTE(cost) | 0.35 | 0.60 | 0.76 | 0.74 | 0.75 | 0.15 |
| prior|simclr:cifar100@10:resnet18 | 3.75 | 8.61 | 3.63 | 7.13 | 6.84 |  | -1.77 | -6.0 | SUBSTITUTE(cost) | 0.49 | 0.66 | 0.83 | 0.74 | 0.92 | 0.02 |
| prior|simclr:cifar100@25:resnet18 | 0.16 | 2.38 | 0.44 | 1.77 | 0.92 |  | -1.46 | -8.1 | SUBSTITUTE(cost) | 0.75 | 0.74 | 0.93 | 0.57 | 1.30 |  |
| prior|simclr:cifar100@10:vit_tiny | 13.27 | 13.30 | 15.01 | 13.82 | 13.46 | 14.41 | +0.16 | +0.2 | SUBSTITUTE | 0.08 | 0.36 | 0.86 | 0.90 | 1.07 | 0.10 |
| prior|simclr:cifar100@10:vit_tiny:deit | 21.01 | 15.78 | 22.31 | 17.38 | 17.44 |  | -3.57 | -6.0 | SUBSTITUTE(cost) | 0.22 | 0.57 | 0.88 | 0.92 | 0.98 | 0.04 |
| prior|simclr:eurosat@5:resnet18 | 0.84 | -1.01 | 0.30 | -2.31 | -0.67 | -1.13 | -1.51 | -8.0 | SUBSTITUTE(cost) | 1.13 | 0.88 | 0.92 | 0.60 | 1.08 |  |
| prior|simclr:eurosat@10:resnet18 | 1.23 | 0.46 | 0.62 | -0.25 | 0.35 | 0.03 | -0.88 | -7.5 | SUBSTITUTE(cost) | 1.40 | 0.90 | 0.93 | -0.42 | 1.05 |  |
| prior|simclr:eurosat@25:resnet18 | -0.12 | -0.10 | -0.01 | -0.12 | -0.24 | -0.11 | -0.14 | -1.6 | SUBSTITUTE | 0.92 | 0.94 | 0.96 | -0.51 | 1.26 |  |
| prior|simclr:food101@5:resnet18 | 5.63 | 9.56 | 6.27 | 7.84 | 8.06 | 6.52 | -1.50 | -1.2 | SUBSTITUTE | 0.20 | 0.51 | 0.71 | 0.83 | 0.78 | 0.33 |
| prior|simclr:food101@10:resnet18 | 3.18 | 5.45 | 2.87 | 3.51 | 3.03 | 1.16 | -2.42 | -4.2 | SUBSTITUTE(cost) | 0.18 | 0.63 | 0.85 | 0.71 | 1.05 | 0.35 |
| prior|simclr:food101@25:resnet18 | -0.69 | -0.53 | -0.71 | -0.53 | -1.87 | -1.78 | -1.34 | -9.4 | SUBSTITUTE(cost) | 0.25 | 0.76 | 0.96 | 0.53 | 1.59 | 0.17 |
| prior|simclr:stl10@10:resnet18 | 5.93 | 11.22 | 4.60 | 8.63 | 12.87 |  | +1.65 | +0.9 | SUBSTITUTE | 0.47 | 0.77 | 0.85 | 0.89 | 0.75 | 0.01 |
| prior|simsiam:cifar10@5:resnet18 | 4.41 | 0.05 | 4.03 | 0.04 | 4.86 |  | +0.45 | +3.6 | STACK | 0.99 | 0.83 | 0.92 | -0.21 | 1.16 |  |
| prior|simsiam:cifar100@5:resnet18 | 5.15 | 0.13 | 6.11 | 0.04 | 6.29 | 8.01 | +1.14 | +6.4 | STACK | 0.99 | 0.66 | 0.83 | 0.31 | 0.80 |  |
| prior|simsiam:cifar100@10:resnet18 | 3.75 | 0.61 | 3.63 | 0.32 | 4.90 | 4.90 | +1.15 | +3.3 | STACK | 0.91 | 0.71 | 0.90 | 0.46 | 1.10 |  |
| prior|simsiam:tin@5:resnet18 | 2.12 | 0.93 | 2.70 | 0.76 | 2.95 | 3.37 | +0.83 | +6.3 | STACK | 0.72 | 0.76 | 0.94 | 0.35 | 1.14 | 0.63 |
| prior|simsiam:tin@10:resnet18 | 1.64 | 0.77 | 1.63 | 0.68 | 2.22 |  | +0.58 | +2.0 | SUBSTITUTE | 0.58 | 0.73 | 0.96 | 0.53 | 1.35 | 0.30 |
| prior|transfer:cifar10@1:resnet18 | 12.78 | 6.36 | 9.69 | 4.94 | -4.23 | -8.54 | -17.01 | -27.1 | SUBSTITUTE(cost) | 0.49 |  |  |  |  |  |
| prior|transfer:cifar10@2:resnet18 | 14.09 | 6.66 | 11.81 | 5.52 | -6.94 | -8.11 | -21.03 | -11.0 | SUBSTITUTE(cost) | 0.53 |  |  |  |  |  |
| prior|transfer:cifar10@5:resnet18 | 11.87 | 4.41 | 10.17 | 4.03 | -3.37 | -4.77 | -15.24 | -17.4 | SUBSTITUTE(cost) | 0.60 | 0.74 | 0.86 | 0.61 | 1.05 |  |
| prior|transfer:cifar10@10:resnet18 | 9.32 | 1.09 | 8.26 | 0.61 | -4.64 | -4.98 | -13.96 | -18.4 | SUBSTITUTE(cost) | 0.93 |  |  |  |  |  |
| prior|transfer:cifar10@20:resnet18 | 3.76 | -0.77 | 3.80 | -0.65 | -1.25 | -0.85 | -5.01 | -4.7 | SUBSTITUTE(cost) | 1.17 |  |  |  |  |  |
| prior|transfer:cifar10@50:resnet18 | 1.00 | -0.68 | 1.01 | -0.59 | -0.16 | -0.20 | -1.16 | -5.7 | SUBSTITUTE(cost) | 1.58 |  |  |  |  |  |
| prior|transfer:cifar100@1:resnet18 | 3.61 | 1.40 | 6.26 | 4.20 | -1.49 | -6.13 | -5.10 | -15.6 | SUBSTITUTE(cost) | 0.33 |  |  |  |  |  |
| prior|transfer:cifar100@2:resnet18 | 5.07 | 2.50 | 6.55 | 5.05 | -2.96 | -5.79 | -8.03 | -26.2 | SUBSTITUTE(cost) | 0.23 |  |  |  |  |  |
| prior|transfer:cifar100@3:resnet18 | 8.14 | 3.68 | 9.17 | 5.71 | -3.07 | -5.03 | -11.21 | -13.2 | SUBSTITUTE(cost) | 0.38 |  |  |  |  |  |
| prior|transfer:cifar100@7:resnet18 | 15.39 | 4.87 | 14.57 | 4.83 | -1.46 | -3.95 | -16.85 | -9.5 | SUBSTITUTE(cost) | 0.67 | 0.58 | 0.77 | 0.53 | 0.75 |  |
| prior|transfer:cifar100@10:resnet18 | 18.00 | 3.75 | 16.40 | 3.63 | -0.18 | -1.10 | -18.18 | -13.7 | SUBSTITUTE(cost) | 0.78 |  |  |  |  |  |
| prior|transfer:cifar100@15:resnet18 | 15.36 | 2.55 | 13.67 | 2.72 | 0.14 | -1.41 | -15.22 | -4.6 | SUBSTITUTE(cost) | 0.80 |  |  |  |  |  |
| prior|transfer:cifar100@20:resnet18 | 11.10 | 0.62 | 9.77 | 0.37 | 0.75 | -0.87 | -10.35 | -4.5 | SUBSTITUTE(cost) | 0.96 |  |  |  |  |  |
| prior|transfer:cifar100@25:resnet18 | 8.48 | 0.16 | 8.09 | 0.44 | 0.67 | 0.00 | -7.81 | -3.8 | SUBSTITUTE(cost) | 0.95 |  |  |  |  |  |
| prior|transfer:cifar100@50:resnet18 | 3.06 | -0.65 | 3.24 | -0.21 | 0.97 | 1.34 | -2.09 | -6.1 | SUBSTITUTE(cost) | 1.06 |  |  |  |  |  |
| prior|transfer:cub@3:resnet18 | 0.26 | 0.02 | 2.45 | 0.76 | -0.28 | -1.64 | -0.54 | -3.6 | SUBSTITUTE(cost) | 0.69 |  |  |  |  |  |
| prior|transfer:cub@5:resnet18 | 0.49 | -0.08 | 2.73 | 0.72 | 0.42 | 0.79 | -0.07 | -0.2 | SUBSTITUTE | 0.74 |  |  |  |  |  |
| prior|transfer:cub@7:resnet18 | 0.69 | -0.06 | 3.16 | 0.75 | 0.39 | 0.63 | -0.30 | -1.1 | SUBSTITUTE | 0.76 |  |  |  |  |  |
| prior|transfer:cub@10:resnet18 | 1.10 | 0.33 | 3.16 | 0.63 | 0.32 | -0.02 | -0.78 | -4.9 | SUBSTITUTE(cost) | 0.80 |  |  |  |  |  |
| prior|transfer:cub@15:resnet18 | 1.17 | 0.35 | 4.86 | 0.78 | 0.17 | 0.31 | -1.00 | -2.5 | SUBSTITUTE(cost) | 0.84 |  |  |  |  |  |
| prior|transfer:cub@20:resnet18 | 2.01 | 0.10 | 5.70 | 0.43 | 0.54 | 0.39 | -1.47 | -4.1 | SUBSTITUTE(cost) | 0.92 |  |  |  |  |  |
| prior|transfer:cub@25:resnet18 | 3.30 | 0.56 | 6.41 | 0.83 | 0.95 | 0.99 | -2.35 | -2.1 | SUBSTITUTE(cost) | 0.87 |  |  |  |  |  |
| prior|transfer:cub@50:resnet18 | 8.82 | 1.82 | 11.89 | 1.87 | 0.91 | 1.55 | -7.91 | -8.9 | SUBSTITUTE(cost) | 0.84 |  |  |  |  |  |
| prior|transfer:dtd@5:resnet18 | 4.06 | 0.30 | 5.53 | 2.57 | -1.40 | -5.66 | -5.46 | -4.7 | SUBSTITUTE(cost) | 0.54 |  |  |  |  |  |
| prior|transfer:dtd@7:resnet18 | 5.87 | 1.69 | 7.22 | 2.68 | -3.56 | -6.43 | -9.43 | -11.8 | SUBSTITUTE(cost) | 0.63 |  |  |  |  |  |
| prior|transfer:dtd@10:resnet18 | 7.59 | 1.56 | 7.70 | 2.94 | -3.76 | -7.66 | -11.35 | -16.0 | SUBSTITUTE(cost) | 0.62 |  |  |  |  |  |
| prior|transfer:dtd@15:resnet18 | 9.24 | 2.15 | 8.37 | 2.66 | -1.79 | -4.84 | -11.03 | -6.6 | SUBSTITUTE(cost) | 0.68 |  |  |  |  |  |
| prior|transfer:dtd@20:resnet18 | 12.79 | 2.27 | 11.08 | 3.99 | -0.71 | -4.18 | -13.50 | -6.9 | SUBSTITUTE(cost) | 0.64 |  |  |  |  |  |
| prior|transfer:dtd@25:resnet18 | 13.80 | 3.12 | 12.26 | 4.44 | 1.64 | -1.18 | -12.16 | -3.7 | SUBSTITUTE(cost) | 0.64 |  |  |  |  |  |
| prior|transfer:dtd@50:resnet18 | 17.39 | 4.87 | 15.25 | 4.26 | -2.25 | -4.79 | -19.64 | -29.4 | SUBSTITUTE(cost) | 0.72 |  |  |  |  |  |
| prior|transfer:dtd@100:resnet18 | 13.46 | 3.55 | 13.90 | 4.13 | -1.58 | -1.51 | -15.04 | -5.0 | SUBSTITUTE(cost) | 0.70 |  |  |  |  |  |
| prior|transfer:eurosat@1:resnet18 | 9.12 | 2.47 | 2.93 | 2.11 | -17.19 | -14.61 | -26.31 | -7.4 | SUBSTITUTE(cost) | 0.28 |  |  |  |  |  |
| prior|transfer:eurosat@2:resnet18 | 5.48 | 1.61 | 0.33 | 0.96 | -1.67 | -6.24 | -7.15 | -10.5 | SUBSTITUTE(cost) | 0.66 |  |  |  |  |  |
| prior|transfer:eurosat@3:resnet18 | 3.09 | 1.77 | -0.22 | 0.53 | -2.67 | -5.40 | -5.76 | -4.0 | SUBSTITUTE(cost) | 1.42 |  |  |  |  |  |
| prior|transfer:eurosat@7:resnet18 | 1.34 | 1.29 | -0.32 | 0.55 | -2.32 | -3.36 | -3.66 | -6.1 | SUBSTITUTE(cost) | 1.58 |  |  |  |  |  |
| prior|transfer:eurosat@10:resnet18 | 1.70 | 1.23 | 0.62 | 0.62 | -1.00 | -1.83 | -2.70 | -10.6 | SUBSTITUTE(cost) | 0.00 |  |  |  |  |  |
| prior|transfer:eurosat@15:resnet18 | 0.65 | 0.16 | 0.23 | 0.17 | -0.96 | -1.32 | -1.61 | -16.6 | SUBSTITUTE(cost) | 0.26 |  |  |  |  |  |
| prior|transfer:eurosat@20:resnet18 | 0.42 | -0.13 | 0.35 | -0.23 | -0.60 | -0.76 | -1.02 | -8.5 | SUBSTITUTE(cost) | 1.66 |  |  |  |  |  |
| prior|transfer:eurosat@25:resnet18 | 0.31 | -0.12 | 0.28 | -0.01 | -0.30 | -0.29 | -0.61 | -4.8 | SUBSTITUTE(cost) | 1.04 |  |  |  |  |  |
| prior|transfer:eurosat@50:resnet18 | 0.30 | 0.10 | 0.08 | 0.04 | 0.08 | -0.04 | -0.22 | -1.2 | SUBSTITUTE | 0.50 |  |  |  |  |  |
| prior|transfer:eurosat@100:resnet18 | 0.06 | -0.14 | 0.07 | -0.21 | 0.00 | -0.07 | -0.06 | -0.6 | SUBSTITUTE | 1.33 |  |  |  |  |  |
| prior|transfer:food101@1:resnet18 | 3.78 | 1.42 | 7.95 | 2.99 | -1.04 | -5.77 | -4.82 | -6.8 | SUBSTITUTE(cost) | 0.62 |  |  |  |  |  |
| prior|transfer:food101@2:resnet18 | 5.53 | 2.98 | 9.12 | 4.76 | -0.26 | -2.63 | -5.79 | -10.1 | SUBSTITUTE(cost) | 0.48 |  |  |  |  |  |
| prior|transfer:food101@3:resnet18 | 8.27 | 4.12 | 11.59 | 5.63 | -1.73 | -4.27 | -10.00 | -15.2 | SUBSTITUTE(cost) | 0.51 |  |  |  |  |  |
| prior|transfer:food101@5:resnet18 | 8.00 | 5.63 | 10.33 | 6.27 | -2.54 | -3.12 | -10.54 | -21.1 | SUBSTITUTE(cost) | 0.39 |  |  |  |  |  |
| prior|transfer:food101@7:resnet18 | 10.91 | 6.72 | 12.68 | 7.44 | -2.90 | -2.64 | -13.81 | -6.4 | SUBSTITUTE(cost) | 0.41 |  |  |  |  |  |
| prior|transfer:food101@10:resnet18 | 8.54 | 3.18 | 8.63 | 2.87 | -3.45 | -3.64 | -11.99 | -7.8 | SUBSTITUTE(cost) | 0.67 |  |  |  |  |  |
| prior|transfer:food101@15:resnet18 | 4.87 | 0.47 | 4.62 | 0.00 | -4.58 | -4.73 | -9.45 | -3.0 | SUBSTITUTE(cost) | 1.00 |  |  |  |  |  |
| prior|transfer:food101@20:resnet18 | 3.23 | -0.21 | 3.30 | -0.38 | -3.61 | -3.39 | -6.84 | -3.9 | SUBSTITUTE(cost) | 1.12 |  |  |  |  |  |
| prior|transfer:food101@25:resnet18 | 1.55 | -0.69 | 1.91 | -0.71 | -1.52 | -1.39 | -3.07 | -8.4 | SUBSTITUTE(cost) | 1.37 |  |  |  |  |  |
| prior|transfer:food101@50:resnet18 | 0.45 | -1.11 | 0.52 | -0.99 | -1.98 | -1.86 | -2.43 | -3.2 | SUBSTITUTE(cost) | 1.53 |  |  |  |  |  |
| prior|transfer:pathmnist@1:resnet18 | 9.17 | 5.61 | 11.17 | 7.58 | 2.02 | 6.43 | -7.15 | -4.6 | SUBSTITUTE(cost) | 0.32 |  |  |  |  |  |
| prior|transfer:pathmnist@2:resnet18 | 8.63 | 6.37 | 9.18 | 6.91 | 5.17 | 6.28 | -3.46 | -3.3 | SUBSTITUTE(cost) | 0.25 |  |  |  |  |  |
| prior|transfer:pathmnist@3:resnet18 | 2.37 | 2.80 | 2.57 | 3.47 | -0.19 | 0.52 | -2.99 | -7.9 | SUBSTITUTE(cost) | 0.26 |  |  |  |  |  |
| prior|transfer:pathmnist@7:resnet18 | 2.96 | 1.71 | 3.10 | 0.99 | 1.84 | 2.69 | -1.12 | -1.6 | SUBSTITUTE | 0.68 |  |  |  |  |  |
| prior|transfer:pathmnist@10:resnet18 | 1.78 | 1.74 | 2.20 | 0.99 | 1.37 | 2.24 | -0.41 | -0.9 | SUBSTITUTE | 0.55 |  |  |  |  |  |
| prior|transfer:pathmnist@15:resnet18 | 1.67 | 1.14 | 1.86 | 1.34 | 1.14 | 1.40 | -0.53 | -1.3 | SUBSTITUTE | 0.28 |  |  |  |  |  |
| prior|transfer:pathmnist@20:resnet18 | 0.72 | -0.30 | 1.53 | 0.30 | -0.37 | 1.46 | -1.09 | -1.0 | SUBSTITUTE | 0.80 |  |  |  |  |  |
| prior|transfer:pathmnist@25:resnet18 | 1.76 | -0.18 | 1.33 | -1.02 | 0.36 | 0.00 | -1.40 | -1.4 | SUBSTITUTE | 1.77 |  |  |  |  |  |
| prior|transfer:pathmnist@50:resnet18 | 1.11 | -1.09 | 1.82 | -1.31 | -0.76 | -0.50 | -1.87 | -3.2 | SUBSTITUTE(cost) | 1.72 |  |  |  |  |  |
| prior|transfer:pathmnist@100:resnet18 | 1.52 | 0.01 | 1.70 | 0.15 | -0.57 | 0.08 | -2.09 | -2.4 | SUBSTITUTE(cost) | 0.91 |  |  |  |  |  |
| prior|transfer:stl10@3:resnet18 | 12.89 | 2.71 | 6.57 | 3.87 | -10.68 | -21.90 | -23.57 | -17.4 | SUBSTITUTE(cost) | 0.41 |  |  |  |  |  |
| prior|transfer:stl10@5:resnet18 | 16.67 | 2.52 | 9.09 | 4.35 | -6.53 | -18.10 | -23.20 | -7.0 | SUBSTITUTE(cost) | 0.52 |  |  |  |  |  |
| prior|transfer:stl10@7:resnet18 | 15.92 | 5.04 | 8.44 | 5.12 | 0.21 | -8.02 | -15.71 | -19.0 | SUBSTITUTE(cost) | 0.39 |  |  |  |  |  |
| prior|transfer:stl10@10:resnet18 | 16.43 | 5.93 | 8.68 | 4.60 | 1.52 | -5.86 | -14.91 | -7.4 | SUBSTITUTE(cost) | 0.47 |  |  |  |  |  |
| prior|transfer:stl10@15:resnet18 | 19.32 | 5.35 | 13.87 | 4.82 | -0.55 | -6.54 | -19.87 | -9.7 | SUBSTITUTE(cost) | 0.65 |  |  |  |  |  |
| prior|transfer:stl10@20:resnet18 | 22.02 | 4.36 | 17.39 | 4.89 | 0.18 | -4.77 | -21.84 | -7.7 | SUBSTITUTE(cost) | 0.72 |  |  |  |  |  |
| prior|transfer:stl10@25:resnet18 | 23.62 | 4.77 | 20.68 | 4.74 | -0.11 | -4.45 | -23.73 | -12.1 | SUBSTITUTE(cost) | 0.77 |  |  |  |  |  |
| prior|transfer:stl10@50:resnet18 | 17.38 | 3.17 | 17.27 | 3.14 | -2.77 | -5.19 | -20.15 | -24.5 | SUBSTITUTE(cost) | 0.82 |  |  |  |  |  |
| prior|transfer:stl10@100:resnet18 | 10.40 | 0.86 | 10.79 | 1.13 | -1.29 | -1.21 | -11.69 | -6.5 | SUBSTITUTE(cost) | 0.90 |  |  |  |  |  |
| prior|transfer:tin@1:resnet18 | 3.24 | 1.50 | 10.32 | 4.18 | -0.32 | -1.29 | -3.56 | -9.6 | SUBSTITUTE(cost) | 0.59 |  |  |  |  |  |
| prior|transfer:tin@2:resnet18 | 7.78 | 1.81 | 15.89 | 3.31 | -1.51 | -1.44 | -9.29 | -11.9 | SUBSTITUTE(cost) | 0.79 |  |  |  |  |  |
| prior|transfer:tin@5:resnet18 | 15.57 | 2.12 | 18.10 | 2.70 | -1.33 | -1.00 | -16.90 | -22.5 | SUBSTITUTE(cost) | 0.85 |  |  |  |  |  |
| prior|transfer:tin@10:resnet18 | 14.84 | 1.64 | 13.65 | 1.63 | -8.86 | -11.49 | -23.70 | -1.9 | SUBSTITUTE | 0.88 |  |  |  |  |  |
| prior|transfer:tin@20:resnet18 | 7.82 | 0.69 | 7.77 | 0.89 | 2.82 | 3.12 | -5.00 | -5.3 | SUBSTITUTE(cost) | 0.89 |  |  |  |  |  |
| prior|transfer:tin@25:resnet18 | 5.55 | 0.10 | 5.73 | 0.28 | 0.33 | 0.46 | -5.22 | -4.6 | SUBSTITUTE(cost) | 0.95 |  |  |  |  |  |
| prior|transfer:tin@50:resnet18 | 2.14 | -0.75 | 2.38 | -0.35 | -0.27 | -0.26 | -2.41 | -7.0 | SUBSTITUTE(cost) | 1.15 |  |  |  |  |  |

## Predicate scores: core (no transfer): STACK vs rest  (n=57, positives=['STACK'], n_pos=31)

| predicate | n | AUC (in-sample, oriented) | 95% CI | dir | thr (in-sample) | acc (in) | LOFO acc | LOFO n | LOFO AUC (pooled OOF) | per-family LOFO acc |
|---|---|---|---|---|---|---|---|---|---|---|
| G_max | 57 | 0.777 | [0.65,0.90] | >= | 10.000 | 0.75 | 0.58 | 57 | 0.777 | aug|mae:0.60(5) aug|simclr:0.71(7) prior|aug:0.67(21) prior|dino:0.67(6) prior|simclr:0.46(13) prior|simsiam:0.20(5) |
| G_absdiff | 57 | 0.774 | [0.64,0.89] | >= | 1.730 | 0.72 | 0.54 | 57 | 0.774 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.71(21) prior|dino:0.50(6) prior|simclr:0.31(13) prior|simsiam:0.40(5) |
| base_acc | 57 | 0.756 | [0.62,0.88] | <= | 31.050 | 0.75 | 0.72 | 57 | 0.756 | aug|mae:0.60(5) aug|simclr:0.71(7) prior|aug:0.90(21) prior|dino:0.83(6) prior|simclr:0.54(13) prior|simsiam:0.40(5) |
| leff_min | 44 | 0.742 | [0.58,0.88] | <= | 0.727 | 0.73 | 0.48 | 44 | 0.742 | aug|mae:0.75(4) aug|simclr:0.67(6) prior|aug:0.62(13) prior|dino:0.50(6) prior|simclr:0.20(10) prior|simsiam:0.20(5) |
| G_sum | 57 | 0.739 | [0.60,0.87] | >= | 2.710 | 0.72 | 0.53 | 57 | 0.739 | aug|mae:0.40(5) aug|simclr:0.71(7) prior|aug:0.57(21) prior|dino:0.67(6) prior|simclr:0.46(13) prior|simsiam:0.20(5) |
| is_vit | 57 | 0.733 | [0.62,0.84] | >= | 0.500 | 0.74 | 0.74 | 57 | 0.733 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.95(21) prior|dino:0.67(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) |
| cka_Abase | 51 | 0.725 | [0.57,0.86] | <= | 0.370 | 0.73 | 0.65 | 51 | 0.725 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.80(15) prior|dino:0.83(6) prior|simclr:0.62(13) prior|simsiam:0.20(5) |
| cka_AB | 51 | 0.717 | [0.55,0.86] | <= | 0.500 | 0.76 | 0.75 | 51 | 0.717 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.93(15) prior|dino:0.83(6) prior|simclr:0.85(13) prior|simsiam:0.20(5) |
| D_max | 57 | 0.711 | [0.57,0.84] | >= | 3.145 | 0.70 | 0.46 | 57 | 0.711 | aug|mae:0.40(5) aug|simclr:0.71(7) prior|aug:0.43(21) prior|dino:0.67(6) prior|simclr:0.38(13) prior|simsiam:0.20(5) |
| cka_Bbase | 51 | 0.708 | [0.55,0.85] | <= | 0.379 | 0.75 | 0.65 | 51 | 0.708 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.73(15) prior|dino:0.83(6) prior|simclr:0.69(13) prior|simsiam:0.20(5) |
| D_sum | 57 | 0.702 | [0.56,0.84] | >= | 2.795 | 0.72 | 0.56 | 57 | 0.702 | aug|mae:0.60(5) aug|simclr:0.71(7) prior|aug:0.62(21) prior|dino:0.67(6) prior|simclr:0.46(13) prior|simsiam:0.20(5) |
| cka_base_min | 51 | 0.696 | [0.53,0.85] | <= | 0.377 | 0.75 | 0.69 | 51 | 0.696 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.87(15) prior|dino:0.83(6) prior|simclr:0.69(13) prior|simsiam:0.20(5) |
| ncka_AB | 51 | 0.694 | [0.54,0.84] | <= | 0.678 | 0.69 | 0.57 | 51 | 0.694 | aug|mae:0.40(5) aug|simclr:0.71(7) prior|aug:0.87(15) prior|dino:0.17(6) prior|simclr:0.54(13) prior|simsiam:0.20(5) |
| S_AB | 51 | 0.663 | [0.50,0.82] | <= | 0.746 | 0.67 | 0.41 | 51 | 0.663 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.60(15) prior|dino:0.17(6) prior|simclr:0.31(13) prior|simsiam:0.20(5) |
| G_min | 57 | 0.663 | [0.51,0.81] | >= | 0.015 | 0.68 | 0.54 | 57 | 0.663 | aug|mae:0.60(5) aug|simclr:0.86(7) prior|aug:0.57(21) prior|dino:0.67(6) prior|simclr:0.38(13) prior|simsiam:0.20(5) |
| has_ssl_init | 57 | 0.662 | [0.55,0.78] | <= | 0.500 | 0.65 | 0.26 | 57 | 0.662 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.08(13) prior|simsiam:0.20(5) |
| D_min | 57 | 0.653 | [0.50,0.79] | >= | -0.035 | 0.68 | 0.51 | 57 | 0.653 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.57(21) prior|dino:0.67(6) prior|simclr:0.31(13) prior|simsiam:0.40(5) |
| D_absdiff | 57 | 0.644 | [0.50,0.78] | >= | 0.880 | 0.63 | 0.40 | 57 | 0.644 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.48(21) prior|dino:0.33(6) prior|simclr:0.31(13) prior|simsiam:0.20(5) |
| has_aug | 57 | 0.643 | [0.51,0.77] | >= | 0.500 | 0.65 | 0.46 | 57 | 0.643 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.76(21) prior|dino:0.33(6) prior|simclr:0.08(13) prior|simsiam:0.20(5) |
| leff_absdiff | 38 | 0.641 | [0.45,0.83] | >= | 0.499 | 0.71 | 0.53 | 38 | 0.641 | aug|mae:0.75(4) aug|simclr:0.50(6) prior|aug:0.82(11) prior|dino:0.33(6) prior|simclr:0.11(9) prior|simsiam:1.00(2) |
| REL | 57 | 0.502 | [0.34,0.66] | <= | 1.005 | 0.63 | 0.40 | 57 | 0.578 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.52(21) prior|dino:0.33(6) prior|simclr:0.23(13) prior|simsiam:0.20(5) |
| amp_soft | 57 | 0.551 | [0.39,0.70] | <= | 0.300 | 0.61 | 0.46 | 57 | 0.551 | aug|mae:0.80(5) aug|simclr:0.71(7) prior|aug:0.29(21) prior|dino:0.67(6) prior|simclr:0.31(13) prior|simsiam:0.60(5) |
| cka_base_absdiff | 51 | 0.597 | [0.43,0.75] | <= | 0.071 | 0.65 | 0.39 | 51 | 0.540 | aug|mae:0.60(5) aug|simclr:0.29(7) prior|aug:0.13(15) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:0.40(5) |
| amp_flag | 57 | 0.525 | [0.45,0.61] | <= | 0.500 | 0.56 | 0.37 | 57 | 0.525 | aug|mae:0.80(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.67(6) prior|simclr:0.08(13) prior|simsiam:0.60(5) |
| has_pretrained | 57 | 0.500 | [0.50,0.50] | >= | -0.000 | 0.54 | 0.35 | 57 | 0.500 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.24(21) prior|dino:0.67(6) prior|simclr:0.08(13) prior|simsiam:0.80(5) |
| readout_absdiff | 57 | 0.594 | [0.44,0.74] | >= | 0.150 | 0.63 | 0.33 | 57 | 0.319 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.24(21) prior|dino:0.67(6) prior|simclr:0.23(13) prior|simsiam:0.20(5) |
| pc_corr | 51 | 0.540 | [0.37,0.71] | <= | 0.518 | 0.65 | 0.47 | 51 | 0.297 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.60(15) prior|dino:0.17(6) prior|simclr:0.23(13) prior|simsiam:1.00(5) |
| pct | 57 | 0.549 | [0.40,0.70] | <= | 8.500 | 0.58 | 0.23 | 57 | 0.277 | aug|mae:0.20(5) aug|simclr:0.29(7) prior|aug:0.24(21) prior|dino:0.50(6) prior|simclr:0.08(13) prior|simsiam:0.20(5) |
| readout_absmax | 57 | 0.550 | [0.39,0.71] | >= | 0.200 | 0.61 | 0.35 | 57 | 0.270 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.29(21) prior|dino:0.67(6) prior|simclr:0.23(13) prior|simsiam:0.20(5) |
| G_ratio | 57 | 0.557 | [0.40,0.71] | <= | 0.366 | 0.63 | 0.47 | 57 | 0.268 | aug|mae:0.80(5) aug|simclr:0.57(7) prior|aug:0.48(21) prior|dino:0.33(6) prior|simclr:0.15(13) prior|simsiam:1.00(5) |

### Two-predicate rules (gate, then threshold), LOFO

| rule | LOFO acc | n | per-family |
|---|---|---|---|
| has_aug->STACK else REL | 0.47 | 57 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.76(21) prior|dino:0.33(6) prior|simclr:0.15(13) prior|simsiam:0.20(5) |
| has_aug->STACK else cka_AB | 0.46 | 57 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.76(21) prior|dino:0.33(6) prior|simclr:0.08(13) prior|simsiam:0.20(5) |
| has_aug->STACK else ncka_AB | 0.49 | 57 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.76(21) prior|dino:0.33(6) prior|simclr:0.23(13) prior|simsiam:0.20(5) |
| has_aug->STACK else pc_corr | 0.51 | 57 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.76(21) prior|dino:0.33(6) prior|simclr:0.31(13) prior|simsiam:0.20(5) |
| has_aug->STACK else G_min | 0.49 | 57 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.76(21) prior|dino:0.33(6) prior|simclr:0.23(13) prior|simsiam:0.20(5) |
| has_aug->STACK else readout_absmax | 0.49 | 57 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.76(21) prior|dino:0.33(6) prior|simclr:0.23(13) prior|simsiam:0.20(5) |
| amp_flag->STACK else REL | 0.46 | 57 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.52(21) prior|dino:0.33(6) prior|simclr:0.23(13) prior|simsiam:0.80(5) |
| amp_flag->STACK else cka_AB | 0.75 | 51 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.87(15) prior|dino:0.83(6) prior|simclr:0.85(13) prior|simsiam:0.40(5) |
| amp_flag->STACK else ncka_AB | 0.55 | 51 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.80(15) prior|dino:0.17(6) prior|simclr:0.54(13) prior|simsiam:0.40(5) |
| amp_flag->STACK else pc_corr | 0.47 | 51 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.60(15) prior|dino:0.17(6) prior|simclr:0.23(13) prior|simsiam:1.00(5) |
| amp_flag->STACK else G_min | 0.46 | 57 | aug|mae:0.40(5) aug|simclr:0.71(7) prior|aug:0.57(21) prior|dino:0.67(6) prior|simclr:0.08(13) prior|simsiam:0.40(5) |
| amp_flag->STACK else readout_absmax | 0.40 | 57 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.24(21) prior|dino:0.67(6) prior|simclr:0.23(13) prior|simsiam:1.00(5) |
| has_pretrained->not else REL | 0.40 | 57 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.52(21) prior|dino:0.33(6) prior|simclr:0.23(13) prior|simsiam:0.20(5) |
| has_pretrained->not else cka_AB | 0.75 | 51 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.93(15) prior|dino:0.83(6) prior|simclr:0.85(13) prior|simsiam:0.20(5) |
| has_pretrained->not else ncka_AB | 0.57 | 51 | aug|mae:0.40(5) aug|simclr:0.71(7) prior|aug:0.87(15) prior|dino:0.17(6) prior|simclr:0.54(13) prior|simsiam:0.20(5) |
| has_pretrained->not else pc_corr | 0.47 | 51 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.60(15) prior|dino:0.17(6) prior|simclr:0.23(13) prior|simsiam:1.00(5) |
| has_pretrained->not else G_min | 0.54 | 57 | aug|mae:0.60(5) aug|simclr:0.86(7) prior|aug:0.57(21) prior|dino:0.67(6) prior|simclr:0.38(13) prior|simsiam:0.20(5) |
| has_pretrained->not else readout_absmax | 0.35 | 57 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.29(21) prior|dino:0.67(6) prior|simclr:0.23(13) prior|simsiam:0.20(5) |

### Verdict

Best single-arm predicate out-of-family: **G_max** (LOFO pooled AUC 0.777, in-sample AUC 0.777 [0.65,0.90], LOFO accuracy 0.58 on 57 pairs). The pre-registered bar is LOFO AUC >= 0.8: **NOT REACHED**. No single-arm predicate separates STACK from SUBSTITUTE/INTERFERE out of family.

## Predicate scores: core, resolved only: STACK vs INTERFERE  (n=46, positives=['STACK'], n_pos=31)

| predicate | n | AUC (in-sample, oriented) | 95% CI | dir | thr (in-sample) | acc (in) | LOFO acc | LOFO n | LOFO AUC (pooled OOF) | per-family LOFO acc |
|---|---|---|---|---|---|---|---|---|---|---|
| base_acc | 46 | 0.867 | [0.73,0.97] | <= | 37.545 | 0.85 | 0.80 | 46 | 0.867 | aug|mae:1.00(3) aug|simclr:1.00(4) prior|aug:0.90(21) prior|dino:0.80(5) prior|simclr:0.56(9) prior|simsiam:0.50(4) |
| cka_Abase | 40 | 0.835 | [0.67,0.96] | <= | 0.630 | 0.80 | 0.65 | 40 | 0.835 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.80(15) prior|dino:0.80(5) prior|simclr:0.44(9) prior|simsiam:0.00(4) |
| cka_Bbase | 40 | 0.829 | [0.69,0.94] | <= | 0.379 | 0.80 | 0.60 | 40 | 0.829 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.73(15) prior|dino:0.80(5) prior|simclr:0.33(9) prior|simsiam:0.00(4) |
| is_vit | 46 | 0.820 | [0.69,0.93] | >= | 0.500 | 0.80 | 0.67 | 46 | 0.820 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.95(21) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:0.00(4) |
| cka_AB | 40 | 0.813 | [0.64,0.95] | <= | 0.500 | 0.82 | 0.68 | 40 | 0.813 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.93(15) prior|dino:0.80(5) prior|simclr:0.33(9) prior|simsiam:0.00(4) |
| cka_base_min | 40 | 0.808 | [0.64,0.94] | <= | 0.377 | 0.80 | 0.65 | 40 | 0.808 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.87(15) prior|dino:0.80(5) prior|simclr:0.33(9) prior|simsiam:0.00(4) |
| leff_min | 36 | 0.807 | [0.65,0.93] | <= | 0.990 | 0.78 | 0.58 | 36 | 0.807 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.85(13) prior|dino:0.40(5) prior|simclr:0.29(7) prior|simsiam:0.00(4) |
| G_max | 46 | 0.806 | [0.62,0.94] | >= | 2.235 | 0.80 | 0.70 | 46 | 0.806 | aug|mae:0.67(3) aug|simclr:0.75(4) prior|aug:0.90(21) prior|dino:0.80(5) prior|simclr:0.33(9) prior|simsiam:0.25(4) |
| G_absdiff | 46 | 0.789 | [0.64,0.91] | >= | 0.955 | 0.78 | 0.78 | 46 | 0.789 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.86(21) prior|dino:0.80(5) prior|simclr:0.44(9) prior|simsiam:1.00(4) |
| G_sum | 46 | 0.778 | [0.61,0.92] | >= | 2.660 | 0.80 | 0.78 | 46 | 0.778 | aug|mae:0.67(3) aug|simclr:0.75(4) prior|aug:0.90(21) prior|dino:0.80(5) prior|simclr:0.44(9) prior|simsiam:1.00(4) |
| ncka_AB | 40 | 0.752 | [0.60,0.89] | <= | 0.876 | 0.72 | 0.45 | 40 | 0.752 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.67(15) prior|dino:0.00(5) prior|simclr:0.22(9) prior|simsiam:0.00(4) |
| D_sum | 46 | 0.751 | [0.57,0.90] | >= | 2.795 | 0.78 | 0.74 | 46 | 0.751 | aug|mae:0.67(3) aug|simclr:0.75(4) prior|aug:0.86(21) prior|dino:0.80(5) prior|simclr:0.33(9) prior|simsiam:1.00(4) |
| S_AB | 40 | 0.749 | [0.59,0.88] | <= | 0.848 | 0.72 | 0.53 | 40 | 0.749 | aug|mae:1.00(3) aug|simclr:1.00(4) prior|aug:0.67(15) prior|dino:0.00(5) prior|simclr:0.33(9) prior|simsiam:0.25(4) |
| D_max | 46 | 0.748 | [0.57,0.89] | >= | 1.255 | 0.78 | 0.74 | 46 | 0.748 | aug|mae:0.67(3) aug|simclr:0.75(4) prior|aug:0.81(21) prior|dino:0.80(5) prior|simclr:0.44(9) prior|simsiam:1.00(4) |
| leff_absdiff | 30 | 0.741 | [0.55,0.90] | >= | 0.499 | 0.73 | 0.60 | 30 | 0.741 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.82(11) prior|dino:0.20(5) prior|simclr:0.17(6) prior|simsiam:1.00(1) |
| cka_base_absdiff | 40 | 0.731 | [0.55,0.89] | <= | 0.071 | 0.78 | 0.72 | 40 | 0.731 | aug|mae:1.00(3) aug|simclr:0.50(4) prior|aug:0.87(15) prior|dino:1.00(5) prior|simclr:0.44(9) prior|simsiam:0.50(4) |
| D_min | 46 | 0.695 | [0.50,0.86] | >= | -0.035 | 0.80 | 0.76 | 46 | 0.695 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.86(21) prior|dino:0.80(5) prior|simclr:0.33(9) prior|simsiam:1.00(4) |
| G_min | 46 | 0.692 | [0.51,0.85] | >= | 0.015 | 0.74 | 0.54 | 46 | 0.692 | aug|mae:0.67(3) aug|simclr:0.75(4) prior|aug:0.71(21) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:0.00(4) |
| readout_absdiff | 46 | 0.658 | [0.48,0.82] | >= | 0.135 | 0.74 | 0.74 | 46 | 0.658 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.81(21) prior|dino:0.80(5) prior|simclr:0.33(9) prior|simsiam:1.00(4) |
| has_aug | 46 | 0.655 | [0.50,0.80] | >= | -0.000 | 0.67 | 0.52 | 46 | 0.655 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.76(21) prior|dino:0.20(5) prior|simclr:0.11(9) prior|simsiam:0.00(4) |
| pct | 46 | 0.630 | [0.48,0.78] | <= | 100.000 | 0.67 | 0.61 | 46 | 0.630 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.62(21) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:1.00(4) |
| D_absdiff | 46 | 0.628 | [0.46,0.78] | >= | 0.020 | 0.67 | 0.54 | 46 | 0.628 | aug|mae:0.67(3) aug|simclr:0.50(4) prior|aug:0.57(21) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:1.00(4) |
| amp_soft | 46 | 0.541 | [0.35,0.73] | <= | 1.830 | 0.67 | 0.61 | 46 | 0.541 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.67(21) prior|dino:0.60(5) prior|simclr:0.11(9) prior|simsiam:1.00(4) |
| REL | 46 | 0.553 | [0.35,0.75] | <= | 1.005 | 0.72 | 0.61 | 46 | 0.538 | aug|mae:0.67(3) aug|simclr:0.75(4) prior|aug:0.71(21) prior|dino:0.80(5) prior|simclr:0.33(9) prior|simsiam:0.25(4) |
| has_pretrained | 46 | 0.500 | [0.50,0.50] | >= | -0.000 | 0.67 | 0.67 | 46 | 0.500 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.76(21) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:1.00(4) |
| amp_flag | 46 | 0.501 | [0.43,0.59] | <= | 1.000 | 0.67 | 0.65 | 46 | 0.468 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.76(21) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:0.75(4) |
| readout_absmax | 46 | 0.627 | [0.44,0.80] | >= | 0.200 | 0.72 | 0.48 | 46 | 0.362 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.29(21) prior|dino:0.80(5) prior|simclr:0.22(9) prior|simsiam:1.00(4) |
| pc_corr | 40 | 0.544 | [0.34,0.74] | <= | 0.518 | 0.68 | 0.50 | 40 | 0.248 | aug|mae:1.00(3) aug|simclr:0.50(4) prior|aug:0.60(15) prior|dino:0.00(5) prior|simclr:0.22(9) prior|simsiam:1.00(4) |
| G_ratio | 46 | 0.508 | [0.31,0.69] | >= | 0.007 | 0.67 | 0.46 | 46 | 0.237 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.48(21) prior|dino:0.60(5) prior|simclr:0.11(9) prior|simsiam:0.25(4) |
| has_ssl_init | 46 | 0.591 | [0.43,0.74] | <= | 1.000 | 0.67 | 0.43 | 46 | 0.209 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.24(21) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:1.00(4) |

## Predicate scores: all families: STACK vs rest  (n=134, positives=['STACK'], n_pos=31)

| predicate | n | AUC (in-sample, oriented) | 95% CI | dir | thr (in-sample) | acc (in) | LOFO acc | LOFO n | LOFO AUC (pooled OOF) | per-family LOFO acc |
|---|---|---|---|---|---|---|---|---|---|---|
| is_vit | 134 | 0.848 | [0.77,0.92] | >= | 0.500 | 0.89 | 0.89 | 134 | 0.848 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.95(21) prior|dino:0.67(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:1.00(77) |
| has_aug | 134 | 0.801 | [0.71,0.89] | >= | 0.500 | 0.85 | 0.85 | 134 | 0.801 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.76(21) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:1.00(77) |
| leff_min | 44 | 0.742 | [0.58,0.88] | <= | 0.727 | 0.73 | 0.48 | 44 | 0.742 | aug|mae:0.75(4) aug|simclr:0.67(6) prior|aug:0.62(13) prior|dino:0.50(6) prior|simclr:0.20(10) prior|simsiam:0.20(5) |
| base_acc | 134 | 0.730 | [0.64,0.81] | <= | 0.980 | 0.77 | 0.56 | 134 | 0.730 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.64(77) |
| cka_Abase | 53 | 0.730 | [0.58,0.86] | <= | 0.370 | 0.74 | 0.70 | 53 | 0.730 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.80(15) prior|dino:0.83(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:1.00(2) |
| cka_AB | 53 | 0.724 | [0.57,0.86] | <= | 0.500 | 0.77 | 0.75 | 53 | 0.724 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.93(15) prior|dino:0.83(6) prior|simclr:0.85(13) prior|simsiam:0.20(5) prior|transfer:1.00(2) |
| cka_Bbase | 53 | 0.720 | [0.56,0.86] | <= | 0.379 | 0.75 | 0.66 | 53 | 0.720 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.73(15) prior|dino:0.83(6) prior|simclr:0.69(13) prior|simsiam:0.20(5) prior|transfer:1.00(2) |
| G_max | 134 | 0.714 | [0.61,0.81] | >= | 12.820 | 0.77 | 0.58 | 134 | 0.714 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:0.70(77) |
| cka_base_min | 53 | 0.706 | [0.55,0.85] | <= | 0.377 | 0.75 | 0.70 | 53 | 0.706 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.87(15) prior|dino:0.83(6) prior|simclr:0.69(13) prior|simsiam:0.20(5) prior|transfer:1.00(2) |
| G_sum | 134 | 0.698 | [0.59,0.80] | >= | 20.805 | 0.78 | 0.31 | 134 | 0.698 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:0.22(77) |
| ncka_AB | 53 | 0.693 | [0.53,0.83] | <= | 0.678 | 0.70 | 0.58 | 53 | 0.693 | aug|mae:0.40(5) aug|simclr:0.71(7) prior|aug:0.87(15) prior|dino:0.17(6) prior|simclr:0.54(13) prior|simsiam:0.20(5) prior|transfer:1.00(2) |
| G_absdiff | 134 | 0.677 | [0.58,0.78] | >= | 15.940 | 0.77 | 0.34 | 134 | 0.677 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.25(77) |
| D_min | 134 | 0.659 | [0.55,0.77] | >= | 6.945 | 0.80 | 0.32 | 134 | 0.659 | aug|mae:0.40(5) aug|simclr:0.71(7) prior|aug:0.29(21) prior|dino:0.67(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:0.19(77) |
| S_AB | 53 | 0.659 | [0.50,0.81] | <= | 0.746 | 0.68 | 0.55 | 53 | 0.659 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.60(15) prior|dino:0.33(6) prior|simclr:0.69(13) prior|simsiam:0.20(5) prior|transfer:1.00(2) |
| D_sum | 134 | 0.656 | [0.55,0.76] | >= | 36.790 | 0.77 | 0.34 | 134 | 0.656 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:0.27(77) |
| leff_absdiff | 38 | 0.641 | [0.45,0.83] | >= | 0.499 | 0.71 | 0.53 | 38 | 0.641 | aug|mae:0.75(4) aug|simclr:0.50(6) prior|aug:0.82(11) prior|dino:0.33(6) prior|simclr:0.11(9) prior|simsiam:1.00(2) |
| has_ssl_init | 134 | 0.640 | [0.54,0.74] | >= | 1.000 | 0.77 | 0.11 | 134 | 0.640 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.08(13) prior|simsiam:0.20(5) prior|transfer:0.00(77) |
| D_max | 134 | 0.638 | [0.53,0.74] | >= | 23.620 | 0.77 | 0.40 | 134 | 0.638 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.35(77) |
| G_min | 134 | 0.626 | [0.51,0.74] | >= | 7.615 | 0.79 | 0.32 | 134 | 0.626 | aug|mae:0.40(5) aug|simclr:0.71(7) prior|aug:0.29(21) prior|dino:0.67(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:0.19(77) |
| readout_absdiff | 134 | 0.584 | [0.48,0.69] | >= | 9.410 | 0.77 | 0.31 | 134 | 0.584 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.19(77) |
| cka_base_absdiff | 53 | 0.620 | [0.46,0.77] | <= | 0.071 | 0.66 | 0.42 | 53 | 0.567 | aug|mae:0.60(5) aug|simclr:0.29(7) prior|aug:0.13(15) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:0.40(5) prior|transfer:1.00(2) |
| pct | 134 | 0.564 | [0.45,0.68] | <= | 1.000 | 0.77 | 0.54 | 134 | 0.564 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.60(77) |
| G_ratio | 134 | 0.541 | [0.43,0.66] | <= | 0.013 | 0.78 | 0.46 | 134 | 0.552 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.47(77) |
| D_absdiff | 134 | 0.546 | [0.44,0.67] | >= | 18.850 | 0.77 | 0.31 | 134 | 0.546 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.19(77) |
| REL | 134 | 0.519 | [0.41,0.63] | <= | 0.170 | 0.78 | 0.30 | 134 | 0.537 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.18(77) |
| amp_flag | 134 | 0.503 | [0.46,0.56] | >= | 1.000 | 0.77 | 0.22 | 134 | 0.500 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.04(77) |
| amp_soft | 134 | 0.580 | [0.47,0.69] | >= | 5.150 | 0.77 | 0.32 | 134 | 0.436 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.22(77) |
| pc_corr | 53 | 0.553 | [0.38,0.71] | <= | 0.518 | 0.66 | 0.49 | 53 | 0.330 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.60(15) prior|dino:0.17(6) prior|simclr:0.23(13) prior|simsiam:1.00(5) prior|transfer:1.00(2) |
| readout_absmax | 134 | 0.517 | [0.41,0.62] | >= | 8.110 | 0.77 | 0.28 | 134 | 0.214 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.24(21) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.14(77) |
| has_pretrained | 134 | 0.874 | [0.83,0.92] | <= | 0.500 | 0.81 | 0.15 | 134 | 0.126 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.24(21) prior|dino:0.67(6) prior|simclr:0.08(13) prior|simsiam:0.80(5) prior|transfer:0.00(77) |

## Predicate scores: core: INTERFERE vs rest  (n=57, positives=['SUBSTITUTE(cost)'], n_pos=15)

| predicate | n | AUC (in-sample, oriented) | 95% CI | dir | thr (in-sample) | acc (in) | LOFO acc | LOFO n | LOFO AUC (pooled OOF) | per-family LOFO acc |
|---|---|---|---|---|---|---|---|---|---|---|
| base_acc | 57 | 0.834 | [0.70,0.94] | >= | 41.655 | 0.82 | 0.77 | 57 | 0.834 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.95(21) prior|dino:0.67(6) prior|simclr:0.54(13) prior|simsiam:0.60(5) |
| is_vit | 57 | 0.790 | [0.67,0.89] | <= | 0.500 | 0.75 | 0.63 | 57 | 0.790 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:0.00(5) |
| cka_Abase | 51 | 0.788 | [0.62,0.91] | >= | 0.890 | 0.78 | 0.65 | 51 | 0.788 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.87(15) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:0.00(5) |
| cka_Bbase | 51 | 0.786 | [0.66,0.90] | >= | 0.422 | 0.75 | 0.59 | 51 | 0.786 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.67(15) prior|dino:0.67(6) prior|simclr:0.46(13) prior|simsiam:0.00(5) |
| cka_base_min | 51 | 0.766 | [0.60,0.90] | >= | 0.727 | 0.76 | 0.65 | 51 | 0.766 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.87(15) prior|dino:0.67(6) prior|simclr:0.46(13) prior|simsiam:0.00(5) |
| leff_min | 44 | 0.763 | [0.61,0.90] | >= | 1.673 | 0.80 | 0.75 | 44 | 0.763 | aug|mae:1.00(4) aug|simclr:0.83(6) prior|aug:0.85(13) prior|dino:0.83(6) prior|simclr:0.50(10) prior|simsiam:0.60(5) |
| cka_AB | 51 | 0.763 | [0.60,0.89] | >= | 0.840 | 0.78 | 0.65 | 51 | 0.763 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.73(15) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:0.00(5) |
| G_max | 57 | 0.748 | [0.59,0.88] | <= | 1.125 | 0.79 | 0.65 | 57 | 0.748 | aug|mae:0.80(5) aug|simclr:0.71(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:0.20(5) |
| cka_base_absdiff | 51 | 0.733 | [0.57,0.88] | >= | 0.071 | 0.78 | 0.75 | 51 | 0.733 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.87(15) prior|dino:1.00(6) prior|simclr:0.46(13) prior|simsiam:0.60(5) |
| S_AB | 51 | 0.724 | [0.59,0.85] | >= | 1.521 | 0.73 | 0.55 | 51 | 0.724 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.67(15) prior|dino:0.00(6) prior|simclr:0.46(13) prior|simsiam:0.20(5) |
| G_sum | 57 | 0.722 | [0.56,0.86] | <= | 2.260 | 0.79 | 0.67 | 57 | 0.722 | aug|mae:0.80(5) aug|simclr:0.57(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:0.80(5) |
| G_absdiff | 57 | 0.714 | [0.56,0.85] | <= | 0.910 | 0.79 | 0.74 | 57 | 0.714 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.86(21) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:0.80(5) |
| ncka_AB | 51 | 0.704 | [0.56,0.84] | >= | 0.878 | 0.73 | 0.59 | 51 | 0.704 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.67(15) prior|dino:0.50(6) prior|simclr:0.46(13) prior|simsiam:0.20(5) |
| D_sum | 57 | 0.700 | [0.54,0.85] | <= | 0.070 | 0.77 | 0.72 | 57 | 0.700 | aug|mae:0.80(5) aug|simclr:0.71(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:1.00(5) |
| leff_absdiff | 38 | 0.697 | [0.53,0.86] | <= | 0.007 | 0.76 | 0.66 | 38 | 0.697 | aug|mae:1.00(4) aug|simclr:0.83(6) prior|aug:0.82(11) prior|dino:0.17(6) prior|simclr:0.44(9) prior|simsiam:1.00(2) |
| D_max | 57 | 0.692 | [0.53,0.84] | <= | 1.255 | 0.77 | 0.68 | 57 | 0.692 | aug|mae:0.80(5) aug|simclr:0.57(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| D_min | 57 | 0.668 | [0.50,0.84] | <= | -0.125 | 0.81 | 0.77 | 57 | 0.668 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.86(21) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:1.00(5) |
| readout_absdiff | 57 | 0.662 | [0.50,0.82] | <= | 0.115 | 0.77 | 0.74 | 57 | 0.662 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:0.80(5) |
| pct | 57 | 0.657 | [0.52,0.79] | >= | 100.000 | 0.74 | 0.70 | 57 | 0.657 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.67(21) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| G_min | 57 | 0.653 | [0.49,0.81] | <= | -2.100 | 0.75 | 0.72 | 57 | 0.653 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| readout_absmax | 57 | 0.649 | [0.48,0.81] | <= | 0.055 | 0.75 | 0.75 | 57 | 0.649 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:1.00(5) |
| has_aug | 57 | 0.621 | [0.47,0.77] | <= | -0.000 | 0.74 | 0.74 | 57 | 0.621 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| D_absdiff | 57 | 0.581 | [0.42,0.73] | <= | 0.020 | 0.74 | 0.74 | 57 | 0.581 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| amp_soft | 57 | 0.521 | [0.34,0.69] | >= | 2.310 | 0.74 | 0.70 | 57 | 0.521 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.71(21) prior|dino:0.67(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| amp_flag | 57 | 0.514 | [0.42,0.59] | <= | -0.000 | 0.74 | 0.74 | 57 | 0.514 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| has_pretrained | 57 | 0.500 | [0.50,0.50] | >= | 0.000 | 0.74 | 0.74 | 57 | 0.500 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| REL | 57 | 0.570 | [0.39,0.75] | >= | 1.027 | 0.75 | 0.70 | 57 | 0.487 | aug|mae:0.80(5) aug|simclr:0.71(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| pc_corr | 51 | 0.519 | [0.33,0.70] | >= | 0.909 | 0.73 | 0.59 | 51 | 0.348 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.67(15) prior|dino:0.00(6) prior|simclr:0.38(13) prior|simsiam:0.80(5) |
| G_ratio | 57 | 0.540 | [0.37,0.71] | <= | 0.007 | 0.74 | 0.70 | 57 | 0.343 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:0.60(5) |
| has_ssl_init | 57 | 0.524 | [0.38,0.67] | >= | 1.000 | 0.74 | 0.74 | 57 | 0.235 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.76(21) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
