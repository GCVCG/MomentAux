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
| `results/law_audit.md` | the sign-law audit as printed in the paper |
| `results/summary.md`, `results/summary.tex` | the generated tables |

The scope flags matter. The law is defined over auxiliary-prior cells trained
from scratch, so reproducing the audit means filtering to `aux_target`
present, `init_from` empty, `pretrained` false and `stem == none`. The
released audit script does exactly this and prints both the in-scope and
all-cells numbers so the choice stays visible.

```python
import pandas as pd
df = pd.read_csv("results/all_results.csv")

law = df[df.aux_target.notna() & df.init_from.isna()
         & ~df.pretrained.astype(bool) & (df.stem.fillna("none") == "none")
         & (df.n_seeds >= 3) & (df.n_probe_seeds >= 3)]
law["readout"] = law.delta - law.G
```

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
