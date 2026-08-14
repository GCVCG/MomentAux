# Released artifacts

Every number in the paper traces to a file in this release. The assets are
attached to the tagged GitHub release; `SHA256SUMS` accompanies them.

## What is in each asset

### `result-tables.tar.gz`

The aggregated tables. This is the right starting point for almost every
question.

| file | contents |
|---|---|
| `results/all_results.csv` | **one row per experimental cell**: dataset, backbone, intervention, data fraction, seed count, accuracy mean and standard deviation, probe accuracy and probe seed count, the paired baseline, `delta` and its standard error, `G` and its standard error, readout, and the scope flags (`aux_target`, `init_from`, `pretrained`, `stem`, `bistable`, `is_headline`) |
| `results/results_by_portion.csv` | the same measurements pivoted to configuration by data fraction |
| `results/MomentStem_results.xlsx` | both views in one workbook, with a column dictionary and the law audit |
| `results/law_audit.md` | the canonical sign-law audit, verbatim output of `analysis/audit_law_paired.py` (seed-paired uncertainty, threshold sensitivity, robustness partitions and the full exception list) |
| `results/summary.md`, `results/summary.tex` | the generated tables |

The scope flags matter. The law is defined over auxiliary-prior cells trained
from scratch, so reproducing the audit means filtering to `aux_target`
present, `init_from` empty, `pretrained` absent and `stem == none`.

**Test emptiness, not truthiness.** `pretrained` is written as the string
`yes` or left empty, so a boolean cast is wrong in both directions: an earlier
version of this snippet used `~df.pretrained.astype(bool)`, and because pandas
reads the empty cells as `NaN` and `NaN` casts to `True`, it selected **zero
rows**. The same mistake in the audit script — comparing against the strings
`true`/`1` when the exporter writes `yes` — is what let the ImageNet-transfer
cells leak into the audit and cost 6.6 points of the reported rate. It is
worth being pedantic about.

```python
import pandas as pd
df = pd.read_csv("results/all_results.csv")

law = df[df.aux_target.notna() & df.init_from.isna()
         & df.pretrained.isna() & (df.stem.fillna("none") == "none")
         & df.baseline_cell.notna() & df.base_acc.notna()]
law = law.assign(readout=law.delta - law.G)      # 1,237 cells
```

**That is not yet the paper's 1,009.** The audit forms `readout` *per seed*,
which needs four measurements from the same seed — both arms' accuracy and
both arms' probe — and drops any cell without at least three seeds common to
all four. That check needs the per-run records, not the summary table, so it
cannot be done from the CSV alone: 228 of the 1,237 cells fall out. Unpack
`run-records.tar.gz` alongside the tables and run

```bash
python analysis/audit_law_paired.py      # 1,009 in scope, 461 resolvable, 395 correct
```

which is the single command behind every law number in the paper. The
`delta_sem` and `G_sem` columns in the CSV are *independent* standard errors;
combining them in quadrature overstates the uncertainty on `readout` by a
median factor of 1.8, because `Δ` and `G` are measured on the same
checkpoints and are positively correlated. The paper reports the paired form
and the repository keeps the older independent-SEM script
(`analysis/audit_sign_law.py`) only because it answers a different question.

### `run-records.tar.gz`

Every run's raw record, under the original `runs/<cell>/seed<N>/` paths.

- `final.json` — the complete configuration as executed, final and best test
  accuracy, parameter and FLOP accounting from `fvcore`, the resolved
  `num_workers`, and the environment (PyTorch, `timm`, CUDA, host). This is
  the authoritative record for a cell's accuracy.
- `linear_probe.json` — the full-train-set linear probe, the measurement
  behind `G`.
- `linear_probe_shots.json` — fixed-shot probes, where used. **These are
  comparable only to other probes at the same shot budget**, never to the
  full-train curve.
- `robustness.json` — CIFAR-100-C corruption evaluation, where run.

### `training-curves.tar.gz`

Every run's `metrics.csv`: per-epoch train and test accuracy, loss
components, the `λ` schedule value, learning rate, and the `conv1` usage
ratio. Use these for training dynamics; use `final.json` for the reported
accuracy.

One caveat is recorded honestly: a small number of ImageNet cells were
briefly trained twice concurrently by a queue fault. Their `final.json`
values are computed in memory and are valid single-run measurements, but
their `metrics.csv` files were truncated by the killed duplicate. The
affected cells are the `diagin*` family.

### `logs.tar.gz`

Campaign and wave logs, including the cluster work-queue logs. These are
included because several of the paper's methodological points, notably the
guard and reconciliation failures reported as limitations, are only visible
here.

### `configs-and-subsets.tar.gz`

Every cell configuration (`configs/`) and the committed subset indices
(`data/subsets/`). Both are also in the repository; they are bundled so the
release is self-contained.

The subset indices are what make the comparison controlled: every
intervention at a given (dataset, fraction) consumes byte-identical images.
`scripts/make_subsets.py --check` verifies that they regenerate.

## What is not released, and why

- **Model checkpoints (223 GB).** Too large to distribute. Every cell
  retrains from its committed configuration and subset.
- **Source images.** All 14 datasets are public and cited in the paper. The
  subset indices reproduce the exact selection without redistributing images.

## Verifying an asset

```bash
sha256sum -c SHA256SUMS
tar -tzf run-records.tar.gz | head
```

## Regenerating the paper's tables from the release

Run all three. They write different things, and the one with the most
reassuring console output is not the one that writes the released CSVs:

```bash
tar -xzf run-records.tar.gz            # restores runs/<cell>/seed<N>/final.json
python analysis/aggregate.py           # -> results/summary.{md,tex}
python analysis/export_results_csv.py  # -> results/all_results.csv,
                                       #    results/results_by_portion.csv
python analysis/audit_law_paired.py    # -> the sign-law numbers in the paper,
                                       #    with the seed-paired uncertainty
```

`aggregate.py` prints a long table and writes only the summary files, so it
is easy to run it, see the new cells scroll past, and conclude the released
CSVs are current when they are not. We made exactly that mistake and record
it here rather than only fixing it.

`audit_law_paired.py` supersedes `audit_sign_law.py`. The older script forms
the readout's uncertainty as if the end-to-end gain and the feature gain
were independent; they come from the same checkpoints and are correlated,
which overstates the uncertainty by a median factor of 1.8 and admits too
few cells to the audit. Both are kept so the difference is inspectable, but
the paper reports the paired version.
