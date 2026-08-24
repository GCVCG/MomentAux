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

## Derivation set: 136 pairs, 7 families

| family | n | STACK | SUBSTITUTE | INTERFERE | resolved | ckpt-complete |
|---|---|---|---|---|---|---|
| aug|mae | 5 | 3 | 2 | 0 | 3 | 5 |
| aug|simclr | 8 | 4 | 3 | 1 | 5 | 7 |
| prior|aug | 22 | 17 | 0 | 5 | 22 | 15 |
| prior|dino | 6 | 4 | 1 | 1 | 5 | 6 |
| prior|simclr | 13 | 1 | 4 | 8 | 9 | 13 |
| prior|simsiam | 5 | 4 | 1 | 0 | 4 | 5 |
| prior|transfer | 77 | 0 | 10 | 67 | 67 | 2 |

### Pairs

| pair | dA | dB | GA | GB | dC | GC | combo-best | sigma | outcome | REL | CKA(A,B) | nCKA | pc_r | S | leff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aug|mae:cifar100@1:vit_tiny | 0.04 | 0.02 | -1.17 | 1.00 | 0.98 | 0.75 | +0.94 | +2.3 | STACK | 1.85 | 0.27 | 0.55 | 0.29 | 0.59 | 2.48 |
| aug|mae:cifar100@2:vit_tiny | 1.07 | 1.46 | 1.08 | 2.78 | 2.18 | 3.32 | +0.72 | +1.2 | SUBSTITUTE | 0.61 | 0.27 | 0.58 | 0.16 | 0.58 | 1.72 |
| aug|mae:cifar100@5:vit_tiny | 0.30 | 8.21 | 0.00 | 10.91 | 9.02 | 11.65 | +0.81 | +0.3 | SUBSTITUTE | 1.00 | 0.25 | 0.50 | 0.25 | 0.59 |  |
| aug|mae:cifar100@10:vit_tiny | 3.81 | 13.04 | 3.17 | 13.33 | 18.06 | 18.21 | +5.02 | +8.7 | STACK | 0.76 | 0.30 | 0.58 | 0.37 | 0.70 | 0.74 |
| aug|mae:cifar100@25:vit_tiny | 3.77 | 14.15 | 4.46 | 13.83 | 21.97 | 21.01 | +7.82 | +5.8 | STACK | 0.68 | 0.37 | 0.67 | 0.31 | 0.82 | 1.56 |
| aug|simclr:cifar100@1:vit_tiny | 0.04 | 0.61 | -1.17 | 1.78 | 0.76 |  | +0.15 | +0.3 | SUBSTITUTE | 1.66 | 0.27 | 0.50 | 0.12 | 0.55 | 2.39 |
| aug|simclr:cifar100@5:vit_tiny | 0.30 | 8.09 | 0.00 | 10.02 | 9.39 | 12.75 | +1.30 | +0.9 | SUBSTITUTE | 1.00 | 0.25 | 0.48 | 0.33 | 0.57 |  |
| aug|simclr:cifar100@10:vit_tiny | 3.81 | 13.30 | 3.17 | 13.46 | 19.59 | 20.47 | +6.29 | +7.0 | STACK | 0.76 | 0.30 | 0.57 | 0.51 | 0.70 | 0.77 |
| aug|simclr:cifar100@25:vit_tiny | 3.77 | 15.08 | 4.46 | 14.38 | 23.81 | 22.44 | +8.73 | +27.0 | STACK | 0.69 | 0.36 | 0.64 | 0.36 | 0.80 | 1.59 |
| aug|simclr:eurosat@5:resnet18 | -7.68 | -1.01 | -2.79 | -2.24 | -1.66 |  | -0.65 | -1.4 | SUBSTITUTE | 0.20 | 0.83 | 0.88 | 0.71 | 0.90 | 2.72 |
| aug|simclr:food101@5:resnet18 | 9.46 | 9.56 | 11.22 | 7.80 | 14.05 |  | +4.49 | +4.2 | STACK | 0.30 | 0.49 | 0.64 | 0.50 | 0.70 | 0.66 |
| aug|simclr:food101@10:resnet18 | 9.26 | 5.45 | 9.47 | 3.47 | 5.84 |  | -3.42 | -4.4 | SUBSTITUTE(cost) | 0.63 | 0.51 | 0.69 | 0.61 | 0.95 | 0.73 |
| aug|simclr:tin@10:vit_tiny | 1.89 | 12.09 | 1.39 | 16.76 | 16.85 | 19.91 | +4.76 | +13.4 | STACK | 0.92 |  |  |  |  |  |
| prior|aug:cifar100@1:vit_tiny | 1.36 | 0.04 | 5.89 | -1.17 | 3.24 | 9.12 | +1.88 | +6.6 | STACK | 1.20 | 0.25 | 0.43 | 0.23 | 0.48 | 2.52 |
| prior|aug:cifar100@2:vit_tiny | 3.24 | 1.07 | 7.23 | 1.08 | 7.08 | 13.56 | +3.84 | +17.2 | STACK | 0.85 | 0.27 | 0.49 | 0.19 | 0.55 | 1.71 |
| prior|aug:cifar100@3:vit_tiny | 6.22 | 1.34 | 10.43 | 1.43 | 10.82 | 17.35 | +4.60 | +18.7 | STACK | 0.86 | 0.26 | 0.46 | 0.28 | 0.54 | 0.96 |
| prior|aug:cifar100@5:vit_tiny | 9.35 | 0.30 | 13.17 | 0.00 | 16.62 | 21.35 | +7.27 | +18.5 | STACK | 1.00 | 0.24 | 0.43 | 0.26 | 0.54 |  |
| prior|aug:cifar100@7:vit_tiny | 10.99 | 1.63 | 13.43 | 1.15 | 20.32 | 22.87 | +9.33 | +23.0 | STACK | 0.91 | 0.26 | 0.48 | 0.37 | 0.60 | 1.03 |
| prior|aug:cifar100@10:vit_tiny | 13.27 | 3.81 | 14.83 | 3.17 | 24.82 | 25.37 | +11.55 | +50.5 | STACK | 0.79 | 0.29 | 0.54 | 0.46 | 0.67 | 0.66 |
| prior|aug:cifar100@15:vit_tiny | 14.45 | 2.84 | 14.99 | 2.84 | 27.43 | 26.99 | +12.98 | +30.8 | STACK | 0.81 | 0.32 | 0.59 | 0.35 | 0.72 | 1.58 |
| prior|aug:cifar100@25:vit_tiny | 13.67 | 3.77 | 14.03 | 4.46 | 28.82 | 27.51 | +15.15 | +44.1 | STACK | 0.68 | 0.36 | 0.63 | 0.31 | 0.79 | 1.50 |
| prior|aug:cifar100@100:vit_tiny | 9.89 | 10.78 | 9.70 | 10.90 | 24.64 | 23.93 | +13.86 | +49.0 | STACK | 0.11 | 0.43 | 0.73 | 0.61 | 1.09 | 0.34 |
| prior|aug:eurosat@5:resnet18 | 0.84 | -7.68 | 0.30 | -2.79 | -2.24 | -0.70 | -3.08 | -9.7 | SUBSTITUTE(cost) | 1.11 | 0.85 | 0.88 | 0.26 | 0.87 |  |
| prior|aug:eurosat@10:resnet18 | 1.23 | -0.13 | 0.55 | -0.01 | 0.68 | 0.47 | -0.55 | -5.5 | SUBSTITUTE(cost) | 1.02 | 0.85 | 0.88 | -0.61 | 0.96 |  |
| prior|aug:eurosat@25:resnet18 | -0.12 | 0.12 | 0.04 | 0.36 | -0.25 | 0.27 | -0.37 | -3.5 | SUBSTITUTE(cost) | 0.89 | 0.87 | 0.90 | 0.05 | 1.14 |  |
| prior|aug:food101@5:resnet18 | 5.63 | 9.46 | 6.17 | 11.22 | 13.78 | 15.90 | +4.32 | +11.3 | STACK | 0.45 | 0.60 | 0.74 | 0.61 | 0.74 | 0.33 |
| prior|aug:food101@10:resnet18 | 3.18 | 9.26 | 2.80 | 9.47 | 7.44 | 8.25 | -1.82 | -6.6 | SUBSTITUTE(cost) | 0.70 | 0.55 | 0.74 | 0.61 | 0.96 | 0.39 |
| prior|aug:food101@25:resnet18 | -0.69 | 3.05 | 0.08 | 5.34 | 0.62 | 3.83 | -2.43 | -8.0 | SUBSTITUTE(cost) | 0.99 | 0.51 | 0.70 | 0.31 | 1.15 |  |
| prior|aug:tin@1:vit_tiny | 1.28 | 0.54 | 4.85 | 1.07 | 3.56 | 9.52 | +2.28 | +24.1 | STACK | 0.78 |  |  |  |  |  |
| prior|aug:tin@2:vit_tiny | 3.29 | 0.11 | 7.75 | -0.49 | 6.67 | 13.85 | +3.38 | +39.0 | STACK | 1.06 |  |  |  |  |  |
| prior|aug:tin@5:vit_tiny | 6.77 | 2.27 | 10.62 | 1.93 | 12.44 | 17.60 | +5.67 | +22.2 | STACK | 0.82 |  |  |  |  |  |
| prior|aug:tin@10:vit_tiny | 9.15 | 1.89 | 12.39 | 1.39 | 18.83 | 21.11 | +9.68 | +45.7 | STACK | 0.89 |  |  |  |  |  |
| prior|aug:tin@15:vit_tiny | 10.72 | 4.23 | 11.95 | 4.24 | 20.93 | 21.79 | +10.21 | +23.5 | STACK | 0.65 |  |  |  |  |  |
| prior|aug:tin@25:vit_tiny | 10.49 | 7.28 | 10.46 | 7.43 | 22.38 | 21.70 | +11.89 | +29.2 | STACK | 0.29 |  |  |  |  |  |
| prior|aug:tin@100:vit_tiny | 7.17 | 15.91 | 7.17 | 15.13 | 22.51 | 21.22 | +6.60 | +4.9 | STACK | 0.53 |  |  |  |  |  |
| prior|dino:cifar10@10:vit_tiny | 14.02 | 10.86 | 11.41 | 8.03 | 13.90 |  | -0.12 | -0.5 | SUBSTITUTE | 0.30 | 0.65 | 0.96 | 0.87 | 1.45 | 0.13 |
| prior|dino:cifar100@5:vit_tiny | 9.35 | 2.97 | 13.17 | 4.09 | 10.37 | 13.87 | +1.02 | +3.1 | STACK | 0.69 | 0.30 | 0.79 | 0.75 | 0.92 | 0.03 |
| prior|dino:cifar100@10:vit_tiny | 13.27 | 8.13 | 14.83 | 7.90 | 14.19 | 14.77 | +0.92 | +5.2 | STACK | 0.47 | 0.34 | 0.87 | 0.85 | 1.05 | 0.15 |
| prior|dino:cifar100@25:vit_tiny | 13.67 | 12.47 | 14.03 | 12.06 | 14.99 |  | +1.32 | +3.8 | STACK | 0.14 | 0.41 | 0.90 | 0.90 | 1.15 | 0.07 |
| prior|dino:dtd@50:vit_tiny | 9.84 | 1.36 | 10.25 | 0.67 | 7.32 |  | -2.52 | -3.8 | SUBSTITUTE(cost) | 0.93 | 0.23 | 0.57 | 0.50 | 0.78 | 0.77 |
| prior|dino:food101@50:vit_tiny | 10.49 | 14.78 | 9.86 | 14.21 | 16.19 |  | +1.41 | +3.5 | STACK | 0.31 | 0.36 | 0.87 | 0.83 | 1.12 | 0.06 |
| prior|simclr:cifar10@2:resnet18 | 6.66 | 8.07 | 5.52 | 6.55 | 9.13 |  | +1.06 | +2.4 | STACK | 0.16 | 0.77 | 0.86 | 0.77 | 0.92 | 0.02 |
| prior|simclr:cifar100@5:resnet18 | 5.15 | 9.32 | 6.25 | 9.30 | 7.54 | 8.08 | -1.78 | -7.1 | SUBSTITUTE(cost) | 0.33 | 0.60 | 0.76 | 0.74 | 0.75 | 0.18 |
| prior|simclr:cifar100@10:resnet18 | 3.75 | 8.61 | 3.55 | 6.98 | 6.84 |  | -1.77 | -6.0 | SUBSTITUTE(cost) | 0.49 | 0.66 | 0.83 | 0.74 | 0.92 | 0.02 |
| prior|simclr:cifar100@25:resnet18 | 0.16 | 2.38 | 0.44 | 1.78 | 0.92 |  | -1.46 | -8.1 | SUBSTITUTE(cost) | 0.75 | 0.74 | 0.93 | 0.57 | 1.30 |  |
| prior|simclr:cifar100@10:vit_tiny | 13.27 | 13.30 | 14.83 | 13.46 | 13.46 | 14.14 | +0.16 | +0.2 | SUBSTITUTE | 0.09 | 0.36 | 0.86 | 0.90 | 1.07 | 0.11 |
| prior|simclr:cifar100@10:vit_tiny:deit | 21.01 | 15.78 | 22.20 | 17.30 | 17.44 |  | -3.57 | -6.0 | SUBSTITUTE(cost) | 0.22 | 0.57 | 0.88 | 0.92 | 0.98 | 0.04 |
| prior|simclr:eurosat@5:resnet18 | 0.84 | -1.01 | 0.30 | -2.24 | -0.67 | -1.14 | -1.51 | -8.0 | SUBSTITUTE(cost) | 1.13 | 0.88 | 0.92 | 0.60 | 1.08 |  |
| prior|simclr:eurosat@10:resnet18 | 1.23 | 0.46 | 0.55 | -0.26 | 0.35 | 0.00 | -0.88 | -7.5 | SUBSTITUTE(cost) | 1.47 | 0.90 | 0.93 | -0.42 | 1.05 |  |
| prior|simclr:eurosat@25:resnet18 | -0.12 | -0.10 | 0.04 | -0.05 | -0.24 | -0.07 | -0.14 | -1.6 | SUBSTITUTE | 1.80 | 0.94 | 0.96 | -0.51 | 1.26 |  |
| prior|simclr:food101@5:resnet18 | 5.63 | 9.56 | 6.17 | 7.80 | 8.06 | 6.42 | -1.50 | -1.2 | SUBSTITUTE | 0.21 | 0.51 | 0.71 | 0.83 | 0.78 | 0.33 |
| prior|simclr:food101@10:resnet18 | 3.18 | 5.45 | 2.80 | 3.47 | 3.03 | 1.18 | -2.42 | -4.2 | SUBSTITUTE(cost) | 0.19 | 0.63 | 0.85 | 0.71 | 1.05 | 0.34 |
| prior|simclr:food101@25:resnet18 | -0.69 | -0.53 | 0.08 | 0.59 | -1.87 | -0.67 | -1.34 | -9.4 | SUBSTITUTE(cost) | 0.86 | 0.76 | 0.96 | 0.53 | 1.59 |  |
| prior|simclr:stl10@10:resnet18 | 5.93 | 11.22 | 4.69 | 8.43 | 12.87 |  | +1.65 | +0.9 | SUBSTITUTE | 0.44 | 0.77 | 0.85 | 0.89 | 0.75 | 0.04 |
| prior|simsiam:cifar10@5:resnet18 | 4.41 | 0.05 | 3.97 | 0.06 | 4.86 |  | +0.45 | +3.6 | STACK | 0.98 | 0.83 | 0.92 | -0.21 | 1.16 |  |
| prior|simsiam:cifar100@5:resnet18 | 5.15 | 0.13 | 6.25 | 0.05 | 6.29 | 8.19 | +1.14 | +6.4 | STACK | 0.99 | 0.66 | 0.83 | 0.31 | 0.80 |  |
| prior|simsiam:cifar100@10:resnet18 | 3.75 | 0.61 | 3.55 | 0.20 | 4.90 | 4.81 | +1.15 | +3.3 | STACK | 0.94 | 0.71 | 0.90 | 0.46 | 1.10 |  |
| prior|simsiam:tin@5:resnet18 | 2.12 | 0.93 | 2.66 | 0.70 | 2.95 | 3.36 | +0.83 | +6.3 | STACK | 0.74 | 0.76 | 0.94 | 0.35 | 1.14 | 0.74 |
| prior|simsiam:tin@10:resnet18 | 1.64 | 0.77 | 1.70 | 0.69 | 2.22 |  | +0.58 | +2.0 | SUBSTITUTE | 0.59 | 0.73 | 0.96 | 0.53 | 1.35 | 0.32 |
| prior|transfer:cifar10@1:resnet18 | 12.78 | 6.36 | 9.47 | 4.81 | -4.23 | -8.95 | -17.01 | -27.1 | SUBSTITUTE(cost) | 0.49 |  |  |  |  |  |
| prior|transfer:cifar10@2:resnet18 | 14.09 | 6.66 | 11.71 | 5.52 | -6.94 | -7.93 | -21.03 | -11.0 | SUBSTITUTE(cost) | 0.53 |  |  |  |  |  |
| prior|transfer:cifar10@5:resnet18 | 11.87 | 4.41 | 10.99 | 3.97 | -3.37 | -4.82 | -15.24 | -17.4 | SUBSTITUTE(cost) | 0.64 | 0.74 | 0.86 | 0.61 | 1.05 |  |
| prior|transfer:cifar10@10:resnet18 | 9.32 | 1.09 | 8.32 | 0.65 | -4.64 | -4.97 | -13.96 | -18.4 | SUBSTITUTE(cost) | 0.92 |  |  |  |  |  |
| prior|transfer:cifar10@20:resnet18 | 3.76 | -0.77 | 3.82 | -0.64 | -1.25 | -0.85 | -5.01 | -4.7 | SUBSTITUTE(cost) | 1.17 |  |  |  |  |  |
| prior|transfer:cifar10@50:resnet18 | 1.00 | -0.68 | 1.05 | -0.43 | -0.16 | -0.23 | -1.16 | -5.7 | SUBSTITUTE(cost) | 1.41 |  |  |  |  |  |
| prior|transfer:cifar100@1:resnet18 | 3.63 | 1.42 | 6.05 | 4.16 | -1.47 | -6.15 | -5.10 | -15.6 | SUBSTITUTE(cost) | 0.31 |  |  |  |  |  |
| prior|transfer:cifar100@2:resnet18 | 5.07 | 2.50 | 6.63 | 5.14 | -2.96 | -5.60 | -8.03 | -26.2 | SUBSTITUTE(cost) | 0.22 |  |  |  |  |  |
| prior|transfer:cifar100@3:resnet18 | 8.14 | 3.68 | 9.15 | 5.75 | -3.07 | -4.99 | -11.21 | -13.2 | SUBSTITUTE(cost) | 0.37 |  |  |  |  |  |
| prior|transfer:cifar100@7:resnet18 | 15.39 | 4.87 | 12.33 | 4.81 | -1.46 | -3.93 | -16.85 | -9.5 | SUBSTITUTE(cost) | 0.61 | 0.58 | 0.77 | 0.53 | 0.75 |  |
| prior|transfer:cifar100@10:resnet18 | 18.00 | 3.75 | 16.38 | 3.55 | -0.18 | -1.21 | -18.18 | -13.7 | SUBSTITUTE(cost) | 0.78 |  |  |  |  |  |
| prior|transfer:cifar100@15:resnet18 | 15.36 | 2.55 | 13.97 | 2.73 | 0.14 | -1.51 | -15.22 | -4.6 | SUBSTITUTE(cost) | 0.80 |  |  |  |  |  |
| prior|transfer:cifar100@20:resnet18 | 11.10 | 0.62 | 9.93 | 0.54 | 0.75 | -0.62 | -10.35 | -4.5 | SUBSTITUTE(cost) | 0.95 |  |  |  |  |  |
| prior|transfer:cifar100@25:resnet18 | 8.48 | 0.16 | 8.08 | 0.44 | 0.67 | 0.06 | -7.81 | -3.8 | SUBSTITUTE(cost) | 0.95 |  |  |  |  |  |
| prior|transfer:cifar100@50:resnet18 | 3.06 | -0.65 | 3.15 | -0.31 | 0.97 | 1.23 | -2.09 | -6.1 | SUBSTITUTE(cost) | 1.10 |  |  |  |  |  |
| prior|transfer:cub@3:resnet18 | 0.26 | 0.02 | 1.66 | 0.35 | -0.28 | -1.82 | -0.54 | -3.6 | SUBSTITUTE(cost) | 0.79 |  |  |  |  |  |
| prior|transfer:cub@5:resnet18 | 0.49 | -0.08 | 3.01 | 0.15 | 0.42 | 1.16 | -0.07 | -0.2 | SUBSTITUTE | 0.95 |  |  |  |  |  |
| prior|transfer:cub@7:resnet18 | 0.69 | -0.06 | 3.28 | 0.79 | 0.39 | 1.00 | -0.30 | -1.1 | SUBSTITUTE | 0.76 |  |  |  |  |  |
| prior|transfer:cub@10:resnet18 | 1.10 | 0.33 | 3.95 | 1.38 | 0.32 | 0.68 | -0.78 | -4.9 | SUBSTITUTE(cost) | 0.65 |  |  |  |  |  |
| prior|transfer:cub@15:resnet18 | 1.17 | 0.35 | 4.96 | 0.60 | 0.17 | 0.18 | -1.00 | -2.5 | SUBSTITUTE(cost) | 0.88 |  |  |  |  |  |
| prior|transfer:cub@20:resnet18 | 2.01 | 0.10 | 5.81 | 0.65 | 0.54 | 0.50 | -1.47 | -4.1 | SUBSTITUTE(cost) | 0.89 |  |  |  |  |  |
| prior|transfer:cub@25:resnet18 | 3.30 | 0.56 | 6.53 | 0.84 | 0.95 | 0.90 | -2.35 | -2.1 | SUBSTITUTE(cost) | 0.87 |  |  |  |  |  |
| prior|transfer:cub@50:resnet18 | 8.82 | 1.82 | 11.91 | 1.93 | 0.91 | 1.72 | -7.91 | -8.9 | SUBSTITUTE(cost) | 0.84 |  |  |  |  |  |
| prior|transfer:dtd@5:resnet18 | 4.06 | 0.30 | 6.61 | 1.88 | -1.40 | -5.76 | -5.46 | -4.7 | SUBSTITUTE(cost) | 0.72 |  |  |  |  |  |
| prior|transfer:dtd@7:resnet18 | 5.87 | 1.69 | 7.50 | 3.72 | -3.56 | -6.10 | -9.43 | -11.8 | SUBSTITUTE(cost) | 0.50 |  |  |  |  |  |
| prior|transfer:dtd@10:resnet18 | 7.59 | 1.56 | 7.71 | 2.38 | -3.76 | -8.12 | -11.35 | -16.0 | SUBSTITUTE(cost) | 0.69 |  |  |  |  |  |
| prior|transfer:dtd@15:resnet18 | 9.24 | 2.15 | 8.41 | 2.52 | -1.79 | -5.12 | -11.03 | -6.6 | SUBSTITUTE(cost) | 0.70 |  |  |  |  |  |
| prior|transfer:dtd@20:resnet18 | 12.79 | 2.27 | 11.36 | 2.98 | -0.71 | -4.70 | -13.50 | -6.9 | SUBSTITUTE(cost) | 0.74 |  |  |  |  |  |
| prior|transfer:dtd@25:resnet18 | 13.80 | 3.12 | 12.43 | 4.70 | 1.64 | -1.81 | -12.16 | -3.7 | SUBSTITUTE(cost) | 0.62 |  |  |  |  |  |
| prior|transfer:dtd@50:resnet18 | 17.39 | 4.87 | 15.27 | 4.22 | -2.25 | -4.43 | -19.64 | -29.4 | SUBSTITUTE(cost) | 0.72 |  |  |  |  |  |
| prior|transfer:dtd@100:resnet18 | 13.46 | 3.55 | 13.81 | 3.55 | -1.58 | -1.86 | -15.04 | -5.0 | SUBSTITUTE(cost) | 0.74 |  |  |  |  |  |
| prior|transfer:eurosat@1:resnet18 | 9.12 | 2.47 | 2.84 | 1.59 | -17.19 | -11.85 | -26.31 | -7.4 | SUBSTITUTE(cost) | 0.44 |  |  |  |  |  |
| prior|transfer:eurosat@2:resnet18 | 5.48 | 1.61 | 0.22 | 0.87 | -1.67 | -6.22 | -7.15 | -10.5 | SUBSTITUTE(cost) | 0.75 |  |  |  |  |  |
| prior|transfer:eurosat@3:resnet18 | 3.09 | 1.77 | 0.00 | 0.68 | -2.67 | -5.35 | -5.76 | -4.0 | SUBSTITUTE(cost) | 1.00 |  |  |  |  |  |
| prior|transfer:eurosat@7:resnet18 | 1.34 | 1.29 | -0.44 | 0.55 | -2.32 | -3.38 | -3.66 | -6.1 | SUBSTITUTE(cost) | 1.80 |  |  |  |  |  |
| prior|transfer:eurosat@10:resnet18 | 1.70 | 1.23 | 0.52 | 0.55 | -1.00 | -1.74 | -2.70 | -10.6 | SUBSTITUTE(cost) | 0.05 |  |  |  |  |  |
| prior|transfer:eurosat@15:resnet18 | 0.65 | 0.16 | 0.28 | 0.18 | -0.96 | -1.28 | -1.61 | -16.6 | SUBSTITUTE(cost) | 0.36 |  |  |  |  |  |
| prior|transfer:eurosat@20:resnet18 | 0.42 | -0.13 | 0.32 | -0.23 | -0.60 | -0.78 | -1.02 | -8.5 | SUBSTITUTE(cost) | 1.72 |  |  |  |  |  |
| prior|transfer:eurosat@25:resnet18 | 0.31 | -0.12 | 0.36 | 0.04 | -0.30 | -0.23 | -0.61 | -4.8 | SUBSTITUTE(cost) | 0.89 |  |  |  |  |  |
| prior|transfer:eurosat@50:resnet18 | 0.30 | 0.10 | 0.11 | 0.05 | 0.08 | 0.02 | -0.22 | -1.2 | SUBSTITUTE | 0.55 |  |  |  |  |  |
| prior|transfer:eurosat@100:resnet18 | 0.06 | -0.14 | 0.04 | -0.22 | 0.00 | -0.15 | -0.06 | -0.6 | SUBSTITUTE | 1.18 |  |  |  |  |  |
| prior|transfer:food101@1:resnet18 | 3.78 | 1.42 | 7.99 | 3.04 | -1.04 | -5.94 | -4.82 | -6.8 | SUBSTITUTE(cost) | 0.62 |  |  |  |  |  |
| prior|transfer:food101@2:resnet18 | 5.53 | 2.98 | 9.06 | 4.80 | -0.26 | -2.66 | -5.79 | -10.1 | SUBSTITUTE(cost) | 0.47 |  |  |  |  |  |
| prior|transfer:food101@3:resnet18 | 8.27 | 4.12 | 11.61 | 5.64 | -1.73 | -4.14 | -10.00 | -15.2 | SUBSTITUTE(cost) | 0.51 |  |  |  |  |  |
| prior|transfer:food101@5:resnet18 | 8.00 | 5.63 | 10.21 | 6.17 | -2.54 | -3.21 | -10.54 | -21.1 | SUBSTITUTE(cost) | 0.40 |  |  |  |  |  |
| prior|transfer:food101@7:resnet18 | 10.91 | 6.72 | 12.68 | 7.39 | -2.90 | -2.73 | -13.81 | -6.4 | SUBSTITUTE(cost) | 0.42 |  |  |  |  |  |
| prior|transfer:food101@10:resnet18 | 8.54 | 3.18 | 8.55 | 2.80 | -3.45 | -3.66 | -11.99 | -7.8 | SUBSTITUTE(cost) | 0.67 |  |  |  |  |  |
| prior|transfer:food101@15:resnet18 | 4.87 | 0.47 | 4.75 | -0.11 | -4.58 | -4.74 | -9.45 | -3.0 | SUBSTITUTE(cost) | 1.02 |  |  |  |  |  |
| prior|transfer:food101@20:resnet18 | 3.23 | -0.21 | 3.28 | -0.65 | -3.61 | -3.43 | -6.84 | -3.9 | SUBSTITUTE(cost) | 1.20 |  |  |  |  |  |
| prior|transfer:food101@25:resnet18 | 1.55 | -0.69 | 2.95 | 0.08 | -1.52 | -0.22 | -3.07 | -8.4 | SUBSTITUTE(cost) | 0.97 |  |  |  |  |  |
| prior|transfer:food101@50:resnet18 | 0.45 | -1.11 | 6.37 | -0.24 | -1.98 | 3.99 | -2.43 | -3.2 | SUBSTITUTE(cost) | 1.04 |  |  |  |  |  |
| prior|transfer:pathmnist@1:resnet18 | 9.17 | 5.61 | 12.48 | 7.89 | 2.02 | 8.40 | -7.15 | -4.6 | SUBSTITUTE(cost) | 0.37 |  |  |  |  |  |
| prior|transfer:pathmnist@2:resnet18 | 8.63 | 6.37 | 10.31 | 7.23 | 5.17 | 7.28 | -3.46 | -3.3 | SUBSTITUTE(cost) | 0.30 |  |  |  |  |  |
| prior|transfer:pathmnist@3:resnet18 | 2.37 | 2.80 | 2.08 | 3.23 | -0.19 | 0.56 | -2.99 | -7.9 | SUBSTITUTE(cost) | 0.36 |  |  |  |  |  |
| prior|transfer:pathmnist@7:resnet18 | 2.96 | 1.71 | 2.19 | -0.05 | 1.84 | 1.68 | -1.12 | -1.6 | SUBSTITUTE | 1.02 |  |  |  |  |  |
| prior|transfer:pathmnist@10:resnet18 | 1.78 | 1.74 | 0.24 | -0.91 | 1.37 | 0.51 | -0.41 | -0.9 | SUBSTITUTE | 1.26 |  |  |  |  |  |
| prior|transfer:pathmnist@15:resnet18 | 1.67 | 1.14 | 0.15 | 0.27 | 1.14 | -0.76 | -0.53 | -1.3 | SUBSTITUTE | 0.44 |  |  |  |  |  |
| prior|transfer:pathmnist@20:resnet18 | 0.72 | -0.30 | 0.91 | -0.03 | -0.37 | 1.24 | -1.09 | -1.0 | SUBSTITUTE | 1.03 |  |  |  |  |  |
| prior|transfer:pathmnist@25:resnet18 | 1.76 | -0.18 | 0.58 | 0.41 | 0.36 | -0.56 | -1.40 | -1.4 | SUBSTITUTE | 0.29 |  |  |  |  |  |
| prior|transfer:pathmnist@50:resnet18 | 1.11 | -1.09 | 1.15 | -0.42 | -0.76 | 0.83 | -1.87 | -3.2 | SUBSTITUTE(cost) | 1.37 |  |  |  |  |  |
| prior|transfer:pathmnist@100:resnet18 | 1.52 | 0.01 | -1.03 | 0.72 | -0.57 | -0.15 | -2.09 | -2.4 | SUBSTITUTE(cost) | 1.70 |  |  |  |  |  |
| prior|transfer:stl10@3:resnet18 | 12.89 | 2.71 | 6.70 | 4.24 | -10.68 | -21.42 | -23.57 | -17.4 | SUBSTITUTE(cost) | 0.37 |  |  |  |  |  |
| prior|transfer:stl10@5:resnet18 | 16.67 | 2.52 | 9.02 | 4.27 | -6.53 | -18.13 | -23.20 | -7.0 | SUBSTITUTE(cost) | 0.53 |  |  |  |  |  |
| prior|transfer:stl10@7:resnet18 | 15.92 | 5.04 | 8.86 | 5.61 | 0.21 | -8.05 | -15.71 | -19.0 | SUBSTITUTE(cost) | 0.37 |  |  |  |  |  |
| prior|transfer:stl10@10:resnet18 | 16.43 | 5.93 | 8.79 | 4.69 | 1.52 | -5.78 | -14.91 | -7.4 | SUBSTITUTE(cost) | 0.47 |  |  |  |  |  |
| prior|transfer:stl10@15:resnet18 | 19.32 | 5.35 | 13.82 | 4.92 | -0.55 | -6.70 | -19.87 | -9.7 | SUBSTITUTE(cost) | 0.64 |  |  |  |  |  |
| prior|transfer:stl10@20:resnet18 | 22.02 | 4.36 | 17.43 | 4.98 | 0.18 | -4.63 | -21.84 | -7.7 | SUBSTITUTE(cost) | 0.71 |  |  |  |  |  |
| prior|transfer:stl10@25:resnet18 | 23.62 | 4.77 | 20.50 | 4.66 | -0.11 | -4.59 | -23.73 | -12.1 | SUBSTITUTE(cost) | 0.77 |  |  |  |  |  |
| prior|transfer:stl10@50:resnet18 | 17.38 | 3.17 | 17.44 | 3.20 | -2.77 | -4.90 | -20.15 | -24.5 | SUBSTITUTE(cost) | 0.82 |  |  |  |  |  |
| prior|transfer:stl10@100:resnet18 | 10.40 | 0.86 | 10.86 | 1.01 | -1.29 | -1.27 | -11.69 | -6.5 | SUBSTITUTE(cost) | 0.91 |  |  |  |  |  |
| prior|transfer:tin@1:resnet18 | 3.24 | 1.50 | 10.04 | 4.19 | -0.32 | -1.52 | -3.56 | -9.6 | SUBSTITUTE(cost) | 0.58 |  |  |  |  |  |
| prior|transfer:tin@2:resnet18 | 7.78 | 1.81 | 15.69 | 3.46 | -1.51 | -1.61 | -9.29 | -11.9 | SUBSTITUTE(cost) | 0.78 |  |  |  |  |  |
| prior|transfer:tin@5:resnet18 | 15.57 | 2.12 | 18.87 | 2.66 | -1.33 | -1.07 | -16.90 | -22.5 | SUBSTITUTE(cost) | 0.86 |  |  |  |  |  |
| prior|transfer:tin@10:resnet18 | 14.84 | 1.64 | 18.90 | 1.70 | -8.86 | -11.51 | -23.70 | -1.9 | SUBSTITUTE | 0.91 |  |  |  |  |  |
| prior|transfer:tin@20:resnet18 | 7.82 | 0.69 | 14.60 | 0.52 | 2.82 | 2.66 | -5.00 | -5.3 | SUBSTITUTE(cost) | 0.96 |  |  |  |  |  |
| prior|transfer:tin@25:resnet18 | 5.55 | 0.10 | 12.00 | 0.21 | 0.33 | 0.43 | -5.22 | -4.6 | SUBSTITUTE(cost) | 0.98 |  |  |  |  |  |
| prior|transfer:tin@50:resnet18 | 2.14 | -0.75 | 2.56 | -0.33 | -0.27 | -0.22 | -2.41 | -7.0 | SUBSTITUTE(cost) | 1.13 |  |  |  |  |  |

## Predicate scores: core (no transfer): STACK vs rest  (n=59, positives=['STACK'], n_pos=33)

| predicate | n | AUC (in-sample, oriented) | 95% CI | dir | thr (in-sample) | acc (in) | LOFO acc | LOFO n | LOFO AUC (pooled OOF) | per-family LOFO acc |
|---|---|---|---|---|---|---|---|---|---|---|
| G_absdiff | 59 | 0.793 | [0.67,0.90] | >= | 3.760 | 0.75 | 0.59 | 59 | 0.793 | aug|mae:0.60(5) aug|simclr:0.62(8) prior|aug:0.77(22) prior|dino:0.50(6) prior|simclr:0.31(13) prior|simsiam:0.60(5) |
| G_max | 59 | 0.788 | [0.65,0.90] | >= | 10.340 | 0.76 | 0.58 | 59 | 0.788 | aug|mae:0.60(5) aug|simclr:0.75(8) prior|aug:0.64(22) prior|dino:0.67(6) prior|simclr:0.46(13) prior|simsiam:0.20(5) |
| base_acc | 59 | 0.769 | [0.64,0.89] | <= | 31.050 | 0.76 | 0.73 | 59 | 0.769 | aug|mae:0.60(5) aug|simclr:0.75(8) prior|aug:0.91(22) prior|dino:0.83(6) prior|simclr:0.54(13) prior|simsiam:0.40(5) |
| leff_min | 43 | 0.752 | [0.59,0.90] | <= | 0.722 | 0.72 | 0.42 | 43 | 0.752 | aug|mae:0.75(4) aug|simclr:0.50(6) prior|aug:0.54(13) prior|dino:0.50(6) prior|simclr:0.11(9) prior|simsiam:0.20(5) |
| G_sum | 59 | 0.748 | [0.61,0.87] | >= | 2.875 | 0.73 | 0.54 | 59 | 0.748 | aug|mae:0.40(5) aug|simclr:0.75(8) prior|aug:0.59(22) prior|dino:0.67(6) prior|simclr:0.46(13) prior|simsiam:0.20(5) |
| is_vit | 59 | 0.740 | [0.62,0.85] | >= | 0.500 | 0.75 | 0.75 | 59 | 0.740 | aug|mae:0.60(5) aug|simclr:0.62(8) prior|aug:0.95(22) prior|dino:0.67(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) |
| cka_Abase | 51 | 0.725 | [0.57,0.86] | <= | 0.370 | 0.73 | 0.65 | 51 | 0.725 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.80(15) prior|dino:0.83(6) prior|simclr:0.62(13) prior|simsiam:0.20(5) |
| cka_AB | 51 | 0.717 | [0.55,0.86] | <= | 0.500 | 0.76 | 0.75 | 51 | 0.717 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.93(15) prior|dino:0.83(6) prior|simclr:0.85(13) prior|simsiam:0.20(5) |
| D_max | 59 | 0.714 | [0.57,0.84] | >= | 3.145 | 0.71 | 0.46 | 59 | 0.714 | aug|mae:0.40(5) aug|simclr:0.75(8) prior|aug:0.41(22) prior|dino:0.67(6) prior|simclr:0.38(13) prior|simsiam:0.20(5) |
| cka_Bbase | 51 | 0.708 | [0.55,0.85] | <= | 0.379 | 0.75 | 0.65 | 51 | 0.708 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.73(15) prior|dino:0.83(6) prior|simclr:0.69(13) prior|simsiam:0.20(5) |
| D_sum | 59 | 0.700 | [0.55,0.83] | >= | 2.795 | 0.73 | 0.63 | 59 | 0.700 | aug|mae:0.60(5) aug|simclr:0.75(8) prior|aug:0.59(22) prior|dino:0.67(6) prior|simclr:0.46(13) prior|simsiam:1.00(5) |
| cka_base_min | 51 | 0.696 | [0.53,0.85] | <= | 0.377 | 0.75 | 0.69 | 51 | 0.696 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.87(15) prior|dino:0.83(6) prior|simclr:0.69(13) prior|simsiam:0.20(5) |
| leff_absdiff | 36 | 0.622 | [0.42,0.82] | >= | 0.328 | 0.67 | 0.50 | 36 | 0.695 | aug|mae:0.75(4) aug|simclr:0.50(6) prior|aug:0.80(10) prior|dino:0.17(6) prior|simclr:0.12(8) prior|simsiam:1.00(2) |
| ncka_AB | 51 | 0.694 | [0.54,0.84] | <= | 0.678 | 0.69 | 0.57 | 51 | 0.694 | aug|mae:0.40(5) aug|simclr:0.71(7) prior|aug:0.87(15) prior|dino:0.17(6) prior|simclr:0.54(13) prior|simsiam:0.20(5) |
| S_AB | 51 | 0.663 | [0.50,0.82] | <= | 0.746 | 0.67 | 0.41 | 51 | 0.663 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.60(15) prior|dino:0.17(6) prior|simclr:0.31(13) prior|simsiam:0.20(5) |
| has_ssl_init | 59 | 0.661 | [0.54,0.78] | <= | 0.500 | 0.64 | 0.25 | 59 | 0.661 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.08(13) prior|simsiam:0.20(5) |
| D_absdiff | 59 | 0.661 | [0.51,0.80] | >= | 0.880 | 0.64 | 0.41 | 59 | 0.661 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.45(22) prior|dino:0.33(6) prior|simclr:0.31(13) prior|simsiam:0.20(5) |
| D_min | 59 | 0.652 | [0.50,0.80] | >= | -0.050 | 0.69 | 0.56 | 59 | 0.652 | aug|mae:0.60(5) aug|simclr:0.62(8) prior|aug:0.68(22) prior|dino:0.67(6) prior|simclr:0.31(13) prior|simsiam:0.40(5) |
| has_aug | 59 | 0.652 | [0.53,0.78] | >= | 0.500 | 0.66 | 0.47 | 59 | 0.652 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.77(22) prior|dino:0.33(6) prior|simclr:0.08(13) prior|simsiam:0.20(5) |
| G_min | 59 | 0.647 | [0.50,0.79] | >= | 0.695 | 0.69 | 0.64 | 59 | 0.647 | aug|mae:0.60(5) aug|simclr:0.88(8) prior|aug:0.82(22) prior|dino:0.83(6) prior|simclr:0.31(13) prior|simsiam:0.20(5) |
| REL | 59 | 0.513 | [0.35,0.67] | <= | 0.985 | 0.63 | 0.47 | 59 | 0.624 | aug|mae:0.60(5) aug|simclr:0.75(8) prior|aug:0.55(22) prior|dino:0.33(6) prior|simclr:0.31(13) prior|simsiam:0.20(5) |
| cka_base_absdiff | 51 | 0.597 | [0.43,0.75] | <= | 0.071 | 0.65 | 0.39 | 51 | 0.540 | aug|mae:0.60(5) aug|simclr:0.29(7) prior|aug:0.13(15) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:0.40(5) |
| has_pretrained | 59 | 0.500 | [0.50,0.50] | >= | -0.000 | 0.56 | 0.36 | 59 | 0.500 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.67(6) prior|simclr:0.08(13) prior|simsiam:0.80(5) |
| readout_absmax | 59 | 0.502 | [0.35,0.65] | >= | 0.395 | 0.59 | 0.32 | 59 | 0.468 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.27(22) prior|dino:0.67(6) prior|simclr:0.08(13) prior|simsiam:0.20(5) |
| amp_flag | 59 | 0.503 | [0.42,0.58] | >= | -0.000 | 0.56 | 0.32 | 59 | 0.389 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.67(6) prior|simclr:0.08(13) prior|simsiam:0.40(5) |
| amp_soft | 59 | 0.508 | [0.36,0.67] | <= | 0.665 | 0.63 | 0.31 | 59 | 0.337 | aug|mae:0.20(5) aug|simclr:0.38(8) prior|aug:0.23(22) prior|dino:0.50(6) prior|simclr:0.15(13) prior|simsiam:0.80(5) |
| readout_absdiff | 59 | 0.581 | [0.43,0.72] | >= | 0.160 | 0.61 | 0.32 | 59 | 0.322 | aug|mae:0.20(5) aug|simclr:0.50(8) prior|aug:0.32(22) prior|dino:0.50(6) prior|simclr:0.23(13) prior|simsiam:0.20(5) |
| pc_corr | 51 | 0.540 | [0.37,0.71] | <= | 0.518 | 0.65 | 0.47 | 51 | 0.297 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.60(15) prior|dino:0.17(6) prior|simclr:0.23(13) prior|simsiam:1.00(5) |
| pct | 59 | 0.543 | [0.41,0.69] | <= | 8.500 | 0.56 | 0.22 | 59 | 0.273 | aug|mae:0.20(5) aug|simclr:0.25(8) prior|aug:0.23(22) prior|dino:0.50(6) prior|simclr:0.08(13) prior|simsiam:0.20(5) |
| G_ratio | 59 | 0.555 | [0.40,0.70] | <= | 0.361 | 0.64 | 0.47 | 59 | 0.256 | aug|mae:0.60(5) aug|simclr:0.62(8) prior|aug:0.55(22) prior|dino:0.33(6) prior|simclr:0.08(13) prior|simsiam:1.00(5) |

### Two-predicate rules (gate, then threshold), LOFO

| rule | LOFO acc | n | per-family |
|---|---|---|---|
| has_aug->STACK else REL | 0.47 | 59 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.77(22) prior|dino:0.17(6) prior|simclr:0.15(13) prior|simsiam:0.20(5) |
| has_aug->STACK else cka_AB | 0.47 | 59 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.77(22) prior|dino:0.33(6) prior|simclr:0.08(13) prior|simsiam:0.20(5) |
| has_aug->STACK else ncka_AB | 0.51 | 59 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.77(22) prior|dino:0.33(6) prior|simclr:0.23(13) prior|simsiam:0.20(5) |
| has_aug->STACK else pc_corr | 0.53 | 59 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.77(22) prior|dino:0.33(6) prior|simclr:0.31(13) prior|simsiam:0.20(5) |
| has_aug->STACK else G_min | 0.53 | 59 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.77(22) prior|dino:0.33(6) prior|simclr:0.31(13) prior|simsiam:0.20(5) |
| has_aug->STACK else readout_absmax | 0.49 | 59 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.77(22) prior|dino:0.50(6) prior|simclr:0.15(13) prior|simsiam:0.00(5) |
| amp_flag->STACK else REL | 0.47 | 59 | aug|mae:0.40(5) aug|simclr:0.62(8) prior|aug:0.55(22) prior|dino:0.33(6) prior|simclr:0.31(13) prior|simsiam:0.60(5) |
| amp_flag->STACK else cka_AB | 0.79 | 52 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.94(16) prior|dino:0.83(6) prior|simclr:0.85(13) prior|simsiam:0.60(5) |
| amp_flag->STACK else ncka_AB | 0.62 | 52 | aug|mae:0.40(5) aug|simclr:0.71(7) prior|aug:0.88(16) prior|dino:0.17(6) prior|simclr:0.54(13) prior|simsiam:0.60(5) |
| amp_flag->STACK else pc_corr | 0.48 | 52 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.62(16) prior|dino:0.17(6) prior|simclr:0.23(13) prior|simsiam:1.00(5) |
| amp_flag->STACK else G_min | 0.69 | 59 | aug|mae:0.40(5) aug|simclr:0.75(8) prior|aug:0.91(22) prior|dino:0.67(6) prior|simclr:0.46(13) prior|simsiam:0.60(5) |
| amp_flag->STACK else readout_absmax | 0.41 | 59 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.36(22) prior|dino:0.50(6) prior|simclr:0.08(13) prior|simsiam:1.00(5) |
| has_pretrained->not else REL | 0.47 | 59 | aug|mae:0.60(5) aug|simclr:0.75(8) prior|aug:0.55(22) prior|dino:0.33(6) prior|simclr:0.31(13) prior|simsiam:0.20(5) |
| has_pretrained->not else cka_AB | 0.75 | 51 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.93(15) prior|dino:0.83(6) prior|simclr:0.85(13) prior|simsiam:0.20(5) |
| has_pretrained->not else ncka_AB | 0.57 | 51 | aug|mae:0.40(5) aug|simclr:0.71(7) prior|aug:0.87(15) prior|dino:0.17(6) prior|simclr:0.54(13) prior|simsiam:0.20(5) |
| has_pretrained->not else pc_corr | 0.47 | 51 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.60(15) prior|dino:0.17(6) prior|simclr:0.23(13) prior|simsiam:1.00(5) |
| has_pretrained->not else G_min | 0.64 | 59 | aug|mae:0.60(5) aug|simclr:0.88(8) prior|aug:0.82(22) prior|dino:0.83(6) prior|simclr:0.31(13) prior|simsiam:0.20(5) |
| has_pretrained->not else readout_absmax | 0.32 | 59 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.27(22) prior|dino:0.67(6) prior|simclr:0.08(13) prior|simsiam:0.20(5) |

### Verdict

Best single-arm predicate out-of-family: **G_absdiff** (LOFO pooled AUC 0.793, in-sample AUC 0.793 [0.67,0.90], LOFO accuracy 0.59 on 59 pairs). The pre-registered bar is LOFO AUC >= 0.8: **NOT REACHED**. No single-arm predicate separates STACK from SUBSTITUTE/INTERFERE out of family.

## Predicate scores: core, resolved only: STACK vs INTERFERE  (n=48, positives=['STACK'], n_pos=33)

| predicate | n | AUC (in-sample, oriented) | 95% CI | dir | thr (in-sample) | acc (in) | LOFO acc | LOFO n | LOFO AUC (pooled OOF) | per-family LOFO acc |
|---|---|---|---|---|---|---|---|---|---|---|
| base_acc | 48 | 0.875 | [0.74,0.97] | <= | 37.545 | 0.85 | 0.81 | 48 | 0.875 | aug|mae:1.00(3) aug|simclr:1.00(5) prior|aug:0.91(22) prior|dino:0.80(5) prior|simclr:0.56(9) prior|simsiam:0.50(4) |
| leff_min | 35 | 0.850 | [0.70,0.96] | <= | 0.977 | 0.83 | 0.69 | 35 | 0.850 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.92(13) prior|dino:0.40(5) prior|simclr:0.33(6) prior|simsiam:0.50(4) |
| cka_Abase | 40 | 0.835 | [0.67,0.96] | <= | 0.630 | 0.80 | 0.65 | 40 | 0.835 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.80(15) prior|dino:0.80(5) prior|simclr:0.44(9) prior|simsiam:0.00(4) |
| cka_Bbase | 40 | 0.829 | [0.69,0.94] | <= | 0.379 | 0.80 | 0.60 | 40 | 0.829 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.73(15) prior|dino:0.80(5) prior|simclr:0.33(9) prior|simsiam:0.00(4) |
| is_vit | 48 | 0.827 | [0.71,0.93] | >= | 0.500 | 0.81 | 0.69 | 48 | 0.827 | aug|mae:1.00(3) aug|simclr:0.80(5) prior|aug:0.95(22) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:0.00(4) |
| G_max | 48 | 0.820 | [0.66,0.94] | >= | 0.795 | 0.81 | 0.71 | 48 | 0.820 | aug|mae:0.67(3) aug|simclr:0.80(5) prior|aug:0.91(22) prior|dino:0.80(5) prior|simclr:0.44(9) prior|simsiam:0.00(4) |
| cka_AB | 40 | 0.813 | [0.64,0.95] | <= | 0.500 | 0.82 | 0.68 | 40 | 0.813 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.93(15) prior|dino:0.80(5) prior|simclr:0.33(9) prior|simsiam:0.00(4) |
| cka_base_min | 40 | 0.808 | [0.64,0.94] | <= | 0.377 | 0.80 | 0.65 | 40 | 0.808 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.87(15) prior|dino:0.80(5) prior|simclr:0.33(9) prior|simsiam:0.00(4) |
| G_absdiff | 48 | 0.798 | [0.66,0.92] | >= | 0.920 | 0.79 | 0.79 | 48 | 0.798 | aug|mae:1.00(3) aug|simclr:0.80(5) prior|aug:0.86(22) prior|dino:0.80(5) prior|simclr:0.44(9) prior|simsiam:1.00(4) |
| G_sum | 48 | 0.788 | [0.62,0.92] | >= | 2.790 | 0.81 | 0.77 | 48 | 0.788 | aug|mae:0.67(3) aug|simclr:0.80(5) prior|aug:0.91(22) prior|dino:0.80(5) prior|simclr:0.44(9) prior|simsiam:0.75(4) |
| ncka_AB | 40 | 0.752 | [0.60,0.89] | <= | 0.876 | 0.72 | 0.45 | 40 | 0.752 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.67(15) prior|dino:0.00(5) prior|simclr:0.22(9) prior|simsiam:0.00(4) |
| D_max | 48 | 0.752 | [0.58,0.89] | >= | 1.255 | 0.79 | 0.75 | 48 | 0.752 | aug|mae:0.67(3) aug|simclr:0.80(5) prior|aug:0.82(22) prior|dino:0.80(5) prior|simclr:0.44(9) prior|simsiam:1.00(4) |
| S_AB | 40 | 0.749 | [0.59,0.88] | <= | 0.848 | 0.72 | 0.53 | 40 | 0.749 | aug|mae:1.00(3) aug|simclr:1.00(4) prior|aug:0.67(15) prior|dino:0.00(5) prior|simclr:0.33(9) prior|simsiam:0.25(4) |
| D_sum | 48 | 0.747 | [0.57,0.90] | >= | 2.795 | 0.79 | 0.75 | 48 | 0.747 | aug|mae:0.67(3) aug|simclr:0.80(5) prior|aug:0.86(22) prior|dino:0.80(5) prior|simclr:0.33(9) prior|simsiam:1.00(4) |
| cka_base_absdiff | 40 | 0.731 | [0.55,0.89] | <= | 0.071 | 0.78 | 0.72 | 40 | 0.731 | aug|mae:1.00(3) aug|simclr:0.50(4) prior|aug:0.87(15) prior|dino:1.00(5) prior|simclr:0.44(9) prior|simsiam:0.50(4) |
| leff_absdiff | 28 | 0.728 | [0.52,0.90] | >= | 0.019 | 0.79 | 0.61 | 28 | 0.728 | aug|mae:1.00(3) aug|simclr:0.75(4) prior|aug:0.90(10) prior|dino:0.00(5) prior|simclr:0.20(5) prior|simsiam:1.00(1) |
| D_min | 48 | 0.689 | [0.50,0.85] | >= | -0.050 | 0.81 | 0.77 | 48 | 0.689 | aug|mae:1.00(3) aug|simclr:0.80(5) prior|aug:0.86(22) prior|dino:0.80(5) prior|simclr:0.33(9) prior|simsiam:1.00(4) |
| G_min | 48 | 0.671 | [0.49,0.84] | >= | -1.705 | 0.73 | 0.62 | 48 | 0.671 | aug|mae:0.67(3) aug|simclr:0.80(5) prior|aug:0.82(22) prior|dino:0.80(5) prior|simclr:0.22(9) prior|simsiam:0.00(4) |
| has_aug | 48 | 0.664 | [0.51,0.81] | >= | -0.000 | 0.69 | 0.54 | 48 | 0.664 | aug|mae:1.00(3) aug|simclr:0.80(5) prior|aug:0.77(22) prior|dino:0.20(5) prior|simclr:0.11(9) prior|simsiam:0.00(4) |
| D_absdiff | 48 | 0.646 | [0.49,0.80] | >= | 0.020 | 0.69 | 0.56 | 48 | 0.646 | aug|mae:0.67(3) aug|simclr:0.60(5) prior|aug:0.59(22) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:1.00(4) |
| readout_absdiff | 48 | 0.631 | [0.45,0.79] | >= | 0.050 | 0.71 | 0.69 | 48 | 0.631 | aug|mae:1.00(3) aug|simclr:0.80(5) prior|aug:0.77(22) prior|dino:0.60(5) prior|simclr:0.22(9) prior|simsiam:1.00(4) |
| pct | 48 | 0.626 | [0.47,0.77] | <= | 100.000 | 0.69 | 0.65 | 48 | 0.626 | aug|mae:1.00(3) aug|simclr:0.80(5) prior|aug:0.68(22) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:1.00(4) |
| REL | 48 | 0.572 | [0.38,0.75] | <= | 1.009 | 0.71 | 0.58 | 48 | 0.547 | aug|mae:0.67(3) aug|simclr:0.80(5) prior|aug:0.73(22) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:0.25(4) |
| amp_flag | 48 | 0.527 | [0.44,0.61] | >= | -0.000 | 0.69 | 0.69 | 48 | 0.527 | aug|mae:1.00(3) aug|simclr:0.80(5) prior|aug:0.77(22) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:1.00(4) |
| has_pretrained | 48 | 0.500 | [0.50,0.50] | >= | -0.000 | 0.69 | 0.69 | 48 | 0.500 | aug|mae:1.00(3) aug|simclr:0.80(5) prior|aug:0.77(22) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:1.00(4) |
| amp_soft | 48 | 0.543 | [0.34,0.73] | >= | -0.730 | 0.73 | 0.65 | 48 | 0.455 | aug|mae:0.67(3) aug|simclr:0.80(5) prior|aug:0.77(22) prior|dino:0.60(5) prior|simclr:0.11(9) prior|simsiam:1.00(4) |
| readout_absmax | 48 | 0.537 | [0.36,0.71] | >= | 0.150 | 0.69 | 0.46 | 48 | 0.339 | aug|mae:1.00(3) aug|simclr:0.80(5) prior|aug:0.27(22) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:1.00(4) |
| G_ratio | 48 | 0.537 | [0.36,0.71] | >= | -0.000 | 0.69 | 0.52 | 48 | 0.265 | aug|mae:1.00(3) aug|simclr:0.80(5) prior|aug:0.55(22) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:0.25(4) |
| pc_corr | 40 | 0.544 | [0.34,0.74] | <= | 0.518 | 0.68 | 0.50 | 40 | 0.248 | aug|mae:1.00(3) aug|simclr:0.50(4) prior|aug:0.60(15) prior|dino:0.00(5) prior|simclr:0.22(9) prior|simsiam:1.00(4) |
| has_ssl_init | 48 | 0.591 | [0.43,0.73] | <= | 1.000 | 0.69 | 0.44 | 48 | 0.207 | aug|mae:1.00(3) aug|simclr:0.80(5) prior|aug:0.23(22) prior|dino:0.80(5) prior|simclr:0.11(9) prior|simsiam:1.00(4) |

## Predicate scores: all families: STACK vs rest  (n=136, positives=['STACK'], n_pos=33)

| predicate | n | AUC (in-sample, oriented) | 95% CI | dir | thr (in-sample) | acc (in) | LOFO acc | LOFO n | LOFO AUC (pooled OOF) | per-family LOFO acc |
|---|---|---|---|---|---|---|---|---|---|---|
| is_vit | 136 | 0.855 | [0.78,0.92] | >= | 0.500 | 0.89 | 0.89 | 136 | 0.855 | aug|mae:0.60(5) aug|simclr:0.62(8) prior|aug:0.95(22) prior|dino:0.67(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:1.00(77) |
| has_aug | 136 | 0.810 | [0.72,0.89] | >= | 0.500 | 0.85 | 0.85 | 136 | 0.810 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.77(22) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:1.00(77) |
| leff_min | 43 | 0.752 | [0.59,0.90] | <= | 0.722 | 0.72 | 0.42 | 43 | 0.752 | aug|mae:0.75(4) aug|simclr:0.50(6) prior|aug:0.54(13) prior|dino:0.50(6) prior|simclr:0.11(9) prior|simsiam:0.20(5) |
| G_ratio | 136 | 0.525 | [0.41,0.64] | <= | -0.000 | 0.76 | 0.44 | 136 | 0.744 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.27(22) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.43(77) |
| base_acc | 136 | 0.739 | [0.65,0.82] | <= | 0.980 | 0.76 | 0.55 | 136 | 0.739 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.64(77) |
| cka_Abase | 53 | 0.730 | [0.58,0.86] | <= | 0.370 | 0.74 | 0.70 | 53 | 0.730 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.80(15) prior|dino:0.83(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:1.00(2) |
| cka_AB | 53 | 0.724 | [0.57,0.86] | <= | 0.500 | 0.77 | 0.75 | 53 | 0.724 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.93(15) prior|dino:0.83(6) prior|simclr:0.85(13) prior|simsiam:0.20(5) prior|transfer:1.00(2) |
| cka_Bbase | 53 | 0.720 | [0.56,0.86] | <= | 0.379 | 0.75 | 0.66 | 53 | 0.720 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.73(15) prior|dino:0.83(6) prior|simclr:0.69(13) prior|simsiam:0.20(5) prior|transfer:1.00(2) |
| G_max | 136 | 0.714 | [0.62,0.80] | >= | 12.925 | 0.76 | 0.57 | 136 | 0.714 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:0.70(77) |
| cka_base_min | 53 | 0.706 | [0.55,0.85] | <= | 0.377 | 0.75 | 0.70 | 53 | 0.706 | aug|mae:0.60(5) aug|simclr:0.57(7) prior|aug:0.87(15) prior|dino:0.83(6) prior|simclr:0.69(13) prior|simsiam:0.20(5) prior|transfer:1.00(2) |
| G_sum | 136 | 0.697 | [0.60,0.79] | >= | 21.915 | 0.76 | 0.33 | 136 | 0.697 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:0.27(77) |
| leff_absdiff | 36 | 0.622 | [0.42,0.82] | >= | 0.328 | 0.67 | 0.50 | 36 | 0.695 | aug|mae:0.75(4) aug|simclr:0.50(6) prior|aug:0.80(10) prior|dino:0.17(6) prior|simclr:0.12(8) prior|simsiam:1.00(2) |
| ncka_AB | 53 | 0.693 | [0.53,0.83] | <= | 0.678 | 0.70 | 0.58 | 53 | 0.693 | aug|mae:0.40(5) aug|simclr:0.71(7) prior|aug:0.87(15) prior|dino:0.17(6) prior|simclr:0.54(13) prior|simsiam:0.20(5) prior|transfer:1.00(2) |
| G_absdiff | 136 | 0.684 | [0.58,0.78] | >= | 17.200 | 0.76 | 0.43 | 136 | 0.684 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.42(77) |
| S_AB | 53 | 0.659 | [0.50,0.81] | <= | 0.746 | 0.68 | 0.55 | 53 | 0.659 | aug|mae:0.40(5) aug|simclr:0.57(7) prior|aug:0.60(15) prior|dino:0.33(6) prior|simclr:0.69(13) prior|simsiam:0.20(5) prior|transfer:1.00(2) |
| D_min | 136 | 0.656 | [0.55,0.76] | >= | 6.945 | 0.79 | 0.32 | 136 | 0.656 | aug|mae:0.40(5) aug|simclr:0.62(8) prior|aug:0.27(22) prior|dino:0.67(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:0.19(77) |
| D_sum | 136 | 0.655 | [0.55,0.75] | >= | 36.790 | 0.76 | 0.33 | 136 | 0.655 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:0.27(77) |
| D_max | 136 | 0.643 | [0.54,0.74] | >= | 23.620 | 0.76 | 0.39 | 136 | 0.643 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.35(77) |
| has_ssl_init | 136 | 0.640 | [0.54,0.73] | >= | 1.000 | 0.76 | 0.11 | 136 | 0.640 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.08(13) prior|simsiam:0.20(5) prior|transfer:0.00(77) |
| G_min | 136 | 0.615 | [0.50,0.72] | >= | 7.410 | 0.77 | 0.45 | 136 | 0.615 | aug|mae:0.40(5) aug|simclr:0.62(8) prior|aug:0.27(22) prior|dino:0.67(6) prior|simclr:0.77(13) prior|simsiam:0.20(5) prior|transfer:0.43(77) |
| cka_base_absdiff | 53 | 0.620 | [0.46,0.77] | <= | 0.071 | 0.66 | 0.42 | 53 | 0.567 | aug|mae:0.60(5) aug|simclr:0.29(7) prior|aug:0.13(15) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:0.40(5) prior|transfer:1.00(2) |
| D_absdiff | 136 | 0.561 | [0.45,0.67] | >= | 18.850 | 0.76 | 0.30 | 136 | 0.561 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.19(77) |
| pct | 136 | 0.561 | [0.45,0.68] | <= | 1.000 | 0.76 | 0.53 | 136 | 0.561 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.60(77) |
| REL | 136 | 0.520 | [0.41,0.63] | <= | 0.175 | 0.76 | 0.31 | 136 | 0.534 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.21(77) |
| amp_flag | 136 | 0.531 | [0.47,0.60] | >= | 1.000 | 0.76 | 0.19 | 136 | 0.531 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.00(77) |
| amp_soft | 136 | 0.618 | [0.52,0.72] | >= | 5.260 | 0.76 | 0.27 | 136 | 0.472 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.14(77) |
| pc_corr | 53 | 0.553 | [0.38,0.71] | <= | 0.518 | 0.66 | 0.49 | 53 | 0.330 | aug|mae:0.60(5) aug|simclr:0.43(7) prior|aug:0.60(15) prior|dino:0.17(6) prior|simclr:0.23(13) prior|simsiam:1.00(5) prior|transfer:1.00(2) |
| readout_absdiff | 136 | 0.536 | [0.43,0.65] | >= | 9.400 | 0.76 | 0.25 | 136 | 0.241 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.10(77) |
| has_pretrained | 136 | 0.874 | [0.83,0.92] | <= | 0.500 | 0.81 | 0.15 | 136 | 0.126 | aug|mae:0.60(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.67(6) prior|simclr:0.08(13) prior|simsiam:0.80(5) prior|transfer:0.00(77) |
| readout_absmax | 136 | 0.545 | [0.43,0.65] | <= | 0.060 | 0.76 | 0.28 | 136 | 0.126 | aug|mae:0.40(5) aug|simclr:0.50(8) prior|aug:0.23(22) prior|dino:0.33(6) prior|simclr:0.92(13) prior|simsiam:0.20(5) prior|transfer:0.16(77) |

## Predicate scores: core: INTERFERE vs rest  (n=59, positives=['SUBSTITUTE(cost)'], n_pos=15)

| predicate | n | AUC (in-sample, oriented) | 95% CI | dir | thr (in-sample) | acc (in) | LOFO acc | LOFO n | LOFO AUC (pooled OOF) | per-family LOFO acc |
|---|---|---|---|---|---|---|---|---|---|---|
| base_acc | 59 | 0.842 | [0.72,0.94] | >= | 41.655 | 0.83 | 0.78 | 59 | 0.842 | aug|mae:1.00(5) aug|simclr:0.75(8) prior|aug:0.95(22) prior|dino:0.67(6) prior|simclr:0.54(13) prior|simsiam:0.60(5) |
| leff_min | 43 | 0.810 | [0.67,0.94] | >= | 1.821 | 0.84 | 0.79 | 43 | 0.810 | aug|mae:1.00(4) aug|simclr:0.83(6) prior|aug:0.92(13) prior|dino:0.83(6) prior|simclr:0.56(9) prior|simsiam:0.60(5) |
| is_vit | 59 | 0.797 | [0.68,0.90] | <= | 0.500 | 0.76 | 0.64 | 59 | 0.797 | aug|mae:1.00(5) aug|simclr:0.75(8) prior|aug:0.77(22) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:0.00(5) |
| cka_Abase | 51 | 0.788 | [0.62,0.91] | >= | 0.890 | 0.78 | 0.65 | 51 | 0.788 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.87(15) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:0.00(5) |
| cka_Bbase | 51 | 0.786 | [0.66,0.90] | >= | 0.422 | 0.75 | 0.59 | 51 | 0.786 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.67(15) prior|dino:0.67(6) prior|simclr:0.46(13) prior|simsiam:0.00(5) |
| cka_base_min | 51 | 0.766 | [0.60,0.90] | >= | 0.727 | 0.76 | 0.65 | 51 | 0.766 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.87(15) prior|dino:0.67(6) prior|simclr:0.46(13) prior|simsiam:0.00(5) |
| cka_AB | 51 | 0.763 | [0.60,0.89] | >= | 0.840 | 0.78 | 0.65 | 51 | 0.763 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.73(15) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:0.00(5) |
| G_max | 59 | 0.758 | [0.59,0.89] | <= | 0.795 | 0.81 | 0.80 | 59 | 0.758 | aug|mae:0.80(5) aug|simclr:0.75(8) prior|aug:0.91(22) prior|dino:0.83(6) prior|simclr:0.54(13) prior|simsiam:1.00(5) |
| G_sum | 59 | 0.733 | [0.57,0.88] | <= | 2.305 | 0.80 | 0.69 | 59 | 0.733 | aug|mae:0.80(5) aug|simclr:0.62(8) prior|aug:0.77(22) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:0.80(5) |
| cka_base_absdiff | 51 | 0.733 | [0.57,0.88] | >= | 0.071 | 0.78 | 0.75 | 51 | 0.733 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.87(15) prior|dino:1.00(6) prior|simclr:0.46(13) prior|simsiam:0.60(5) |
| S_AB | 51 | 0.724 | [0.59,0.85] | >= | 1.521 | 0.73 | 0.55 | 51 | 0.724 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.67(15) prior|dino:0.00(6) prior|simclr:0.46(13) prior|simsiam:0.20(5) |
| G_absdiff | 59 | 0.721 | [0.58,0.85] | <= | 0.910 | 0.80 | 0.75 | 59 | 0.721 | aug|mae:1.00(5) aug|simclr:0.75(8) prior|aug:0.86(22) prior|dino:0.83(6) prior|simclr:0.31(13) prior|simsiam:1.00(5) |
| ncka_AB | 51 | 0.704 | [0.56,0.84] | >= | 0.878 | 0.73 | 0.59 | 51 | 0.704 | aug|mae:1.00(5) aug|simclr:0.71(7) prior|aug:0.67(15) prior|dino:0.50(6) prior|simclr:0.46(13) prior|simsiam:0.20(5) |
| D_sum | 59 | 0.700 | [0.53,0.85] | <= | 0.030 | 0.78 | 0.73 | 59 | 0.700 | aug|mae:0.80(5) aug|simclr:0.75(8) prior|aug:0.77(22) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:1.00(5) |
| D_max | 59 | 0.697 | [0.53,0.85] | <= | 1.255 | 0.78 | 0.69 | 59 | 0.697 | aug|mae:0.80(5) aug|simclr:0.62(8) prior|aug:0.77(22) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| leff_absdiff | 36 | 0.690 | [0.48,0.87] | <= | 0.019 | 0.83 | 0.81 | 36 | 0.690 | aug|mae:1.00(4) aug|simclr:0.83(6) prior|aug:0.90(10) prior|dino:0.83(6) prior|simclr:0.50(8) prior|simsiam:1.00(2) |
| D_min | 59 | 0.665 | [0.47,0.83] | <= | -0.125 | 0.81 | 0.78 | 59 | 0.665 | aug|mae:1.00(5) aug|simclr:0.75(8) prior|aug:0.86(22) prior|dino:0.83(6) prior|simclr:0.46(13) prior|simsiam:1.00(5) |
| pct | 59 | 0.653 | [0.51,0.79] | >= | 100.000 | 0.75 | 0.71 | 59 | 0.653 | aug|mae:1.00(5) aug|simclr:0.88(8) prior|aug:0.68(22) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| readout_absdiff | 59 | 0.639 | [0.48,0.79] | <= | 0.050 | 0.76 | 0.75 | 59 | 0.639 | aug|mae:1.00(5) aug|simclr:0.88(8) prior|aug:0.77(22) prior|dino:0.67(6) prior|simclr:0.46(13) prior|simsiam:1.00(5) |
| G_min | 59 | 0.633 | [0.45,0.79] | <= | -1.705 | 0.76 | 0.73 | 59 | 0.633 | aug|mae:1.00(5) aug|simclr:0.75(8) prior|aug:0.77(22) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| has_aug | 59 | 0.630 | [0.48,0.77] | <= | -0.000 | 0.75 | 0.75 | 59 | 0.630 | aug|mae:1.00(5) aug|simclr:0.88(8) prior|aug:0.77(22) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| D_absdiff | 59 | 0.597 | [0.45,0.74] | <= | 0.020 | 0.75 | 0.75 | 59 | 0.597 | aug|mae:1.00(5) aug|simclr:0.88(8) prior|aug:0.77(22) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| REL | 59 | 0.577 | [0.39,0.75] | >= | 1.855 | 0.75 | 0.71 | 59 | 0.577 | aug|mae:0.80(5) aug|simclr:0.75(8) prior|aug:0.77(22) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| G_ratio | 59 | 0.576 | [0.41,0.73] | <= | -0.000 | 0.75 | 0.75 | 59 | 0.576 | aug|mae:1.00(5) aug|simclr:0.88(8) prior|aug:0.77(22) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| amp_flag | 59 | 0.535 | [0.44,0.61] | <= | -0.000 | 0.75 | 0.75 | 59 | 0.535 | aug|mae:1.00(5) aug|simclr:0.88(8) prior|aug:0.77(22) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| has_pretrained | 59 | 0.500 | [0.50,0.50] | >= | 0.000 | 0.75 | 0.75 | 59 | 0.500 | aug|mae:1.00(5) aug|simclr:0.88(8) prior|aug:0.77(22) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| amp_soft | 59 | 0.563 | [0.37,0.74] | <= | -0.730 | 0.78 | 0.71 | 59 | 0.485 | aug|mae:0.80(5) aug|simclr:0.88(8) prior|aug:0.77(22) prior|dino:0.67(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
| readout_absmax | 59 | 0.563 | [0.41,0.72] | <= | 0.080 | 0.75 | 0.73 | 59 | 0.450 | aug|mae:1.00(5) aug|simclr:0.88(8) prior|aug:0.77(22) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:0.80(5) |
| pc_corr | 51 | 0.519 | [0.33,0.70] | >= | 0.909 | 0.73 | 0.59 | 51 | 0.348 | aug|mae:1.00(5) aug|simclr:0.86(7) prior|aug:0.67(15) prior|dino:0.00(6) prior|simclr:0.38(13) prior|simsiam:0.80(5) |
| has_ssl_init | 59 | 0.527 | [0.37,0.67] | >= | 1.000 | 0.75 | 0.75 | 59 | 0.233 | aug|mae:1.00(5) aug|simclr:0.88(8) prior|aug:0.77(22) prior|dino:0.83(6) prior|simclr:0.38(13) prior|simsiam:1.00(5) |
