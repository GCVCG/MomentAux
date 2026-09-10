# MomentAux benchmark: released data assets

Supplementary material for *When does fusing hand-crafted knowledge with learned representations pay? A cost-normalized benchmark of stacking, substitution and interference* (AlMughrabi, Busam, Marques, Radeva).

**These files are also published on GitHub.** They are the assets of the tagged release [`v1.0-benchmark`](https://github.com/GCVCG/MomentAux/releases/tag/v1.0-benchmark) of the study repository [https://github.com/GCVCG/MomentAux](https://github.com/GCVCG/MomentAux), which also holds the code that produced them (training harness, configs, subsets, pinned filter banks, exporters and audit scripts). `docs/ARTIFACTS.md` in that repository documents every asset in detail and how to regenerate the paper's tables from them; this file is the short form that travels with the bundle.

Built from repository commit `26a7217` (2026-09-10) by `python scripts/make_release_assets.py --out dist/`. Tarballs are byte-reproducible (sorted members, fixed mtimes), so the checksums below identify this exact build.

| asset | packed | files | contents |
|---|---:|---:|---|
| `configs-and-subsets.tar.gz` | 4.4 MB | 3,693 | every cell's config and the committed subset indices, so any cell can be re-run on byte-identical images |
| `logs.tar.gz` | 1.8 MB | 396 | campaign logs: what was submitted when, and what failed |
| `result-tables.tar.gz` | 1.6 MB | 187 | aggregated tables, the law audit, and the per-analysis JSON records |
| `run-records.tar.gz` | 1.9 MB | 26,294 | every run's authoritative record: config as executed, accuracy, FLOPs, environment, and the probes behind G |
| `training-curves.tar.gz` | 35.6 MB | 9,984 | per-epoch train/test accuracy, loss components, the lambda schedule |

## Which to open first

- `result-tables.tar.gz`: one row per experimental cell in `results/all_results.csv`; the same pivoted by data fraction; the Excel workbook with a column dictionary; `results/law_audit.md`, the sign-law audit verbatim; the segmentation and detection grids; and the per-figure JSON records. This answers almost every question.
- `configs-and-subsets.tar.gz`: the YAML of every cell and the committed subset indices, so any cell can be re-run on byte-identical images with `python train.py --config <cell>.yaml --seed N` from the repository.
- `run-records.tar.gz`: every run's `final.json` and probe record, needed to re-run the seed-paired audit (`python analysis/audit_law_paired.py`).
- `training-curves.tar.gz`: per-epoch `metrics.csv` for every run.
- `logs.tar.gz`: campaign logs, what was submitted when and what failed.

Model checkpoints (about 275 GB) are not released; every cell is re-trainable from the configs and subsets above.

## Verifying

```
sha256sum -c SHA256SUMS
```

SHA256SUMS:

```
53121b9c8810ed5594efd14eb6848e6f92a8f6c2366fdfee3505b9f751ba2beb  configs-and-subsets.tar.gz
8228707df007c04522e23f4a46ac3a338398f361c5e4e25064f091f51b4c17c3  logs.tar.gz
aac321eda2c5fabbb828ce3d1653fe74e06d89309ee8c2f4e9ac0e47c92fb2cd  result-tables.tar.gz
24c3140db8a6bdfdf318c682bf63d105d887cf9b254ad60ab6d8575ec2308d62  run-records.tar.gz
1955f9c910322da9bd241d651e2e30f5cff926a790cb17e2afa07dc98d13a107  training-curves.tar.gz
```
