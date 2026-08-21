#!/usr/bin/env python3
"""Build the tagged release's assets, reproducibly, and verify what went in.

The v1.0 assets were assembled by hand. That is the same hazard as every
other hand-maintained number in this project: nobody can tell later whether
an asset matches the tree it claims to come from, and `docs/ARTIFACTS.md`
promises specific contents that nothing checks. This script builds all six
assets from the working tree, prints the counts that the manifest claims, and
writes SHA256SUMS.

    python scripts/make_release_assets.py --out dist/
    python scripts/make_release_assets.py --out dist/ --verify   # counts only

WHAT EACH ASSET IS FOR is documented in docs/ARTIFACTS.md; the mapping from
that manifest to the globs below is the thing to keep in step.

REPRODUCIBILITY. Members are added in sorted order with a fixed mtime and
uid/gid, and the gzip wrapper is stamped mtime=0 too, so building the same
tree twice gives byte-identical tarballs. That is what lets a reader check a
download against SHA256SUMS from the repo alone. Verify it the way it was
verified here -- build twice into different directories and diff SHA256SUMS.
"""
import argparse
import fnmatch
import hashlib
import gzip
import os
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH = (2026, 1, 1, 0, 0, 0)  # fixed, so tarballs are byte-reproducible

# asset -> (list of (root, pattern-predicate), human description)
def _under(*dirs):
    return lambda p: any(p.startswith(d + os.sep) or p == d for d in dirs)


# Run trees that must NEVER enter an asset, even though their names start
# "runs" and the predicates below key on that prefix.
#
# runs_dense_50e holds the WITHDRAWN 50-epoch segmentation generation. The
# dense recipe was re-pinned to 200 epochs (see the dense re-pin decision in
# CLAUDE.md) and every released dense number comes from the 200-epoch tree in
# runs_dense. The two generations reached OPPOSITE conclusions on the dense
# attention result, so shipping them side by side under near-identical cell
# names invites a reader to aggregate across both and get a table that looks
# complete, is internally consistent, and is wrong. This is not hypothetical:
# on 2026-08-16 the local runs_dense tree was found to hold 50 cells of the
# 50-epoch generation, and a regeneration would have blended them silently.
# Keep the archive out of the release; it is preserved on the cluster and
# locally under this name for provenance, not for distribution.
EXCLUDED_RUN_TREES = ("runs_dense_50e",)


def _in_runs(p):
    """True for a path under a SHIPPABLE run tree."""
    top = p.split(os.sep)[0]
    return top.startswith("runs") and top not in EXCLUDED_RUN_TREES


ASSETS = {
    "result-tables.tar.gz": (
        lambda p: p.startswith("results" + os.sep) and p.endswith(
            (".csv", ".md", ".tex", ".xlsx", ".json")),
        "aggregated tables, the law audit, and the per-analysis JSON records",
    ),
    "run-records.tar.gz": (
        # 2026-08-16/17. This predicate used to list four exact basenames and
        # silently shipped less than it promised, in three separate ways:
        #
        #   * "linear_probe_SHOTS.json" matched NOTHING -- every fixed-shot
        #     probe on disk is lowercase linear_probe_shots.json (70 files),
        #     and docs/ARTIFACTS.md documented the file as shipped. README.md
        #     promised "every linear_probe*.json" while 101 of 2,664 were left
        #     out (the shots probes, the LASTPT probes behind the ImageNet G
        #     values measured after the wrong-epoch quarantine, and the
        #     cross-label-space tin/tin20/cifar100/cifar100super probes).
        #   * dense_probe.json was absent, so every dense G and all 30 rows of
        #     results/dense_law.csv rested on records the release omitted.
        #   * runs_det was not even a collector root, so the whole detection
        #     task -- 36 finals and 30 det_probe.json -- shipped nowhere, while
        #     results/det_results.csv was released and the paper has a
        #     detection section.
        #
        # A GLOB is used for the probe family deliberately: an exact-name list
        # is what let a case typo delete 70 files from the release without any
        # visible failure. Adding a probe variant must not require editing this
        # tuple. The trade is that anything named linear_probe*.json ships, so
        # do not park scratch files under that name.
        lambda p: (os.path.basename(p) in (
            "final.json", "dense_probe.json", "det_probe.json",
            "robustness.json", "cifair.json", "head_forms_5shot.json",
            "per_class_delta.json")
            or fnmatch.fnmatch(os.path.basename(p), "linear_probe*.json")
        ) and _in_runs(p),
        "every run's authoritative record: config as executed, accuracy, "
        "FLOPs, environment, and the probes behind G",
    ),
    "training-curves.tar.gz": (
        lambda p: os.path.basename(p) == "metrics.csv" and _in_runs(p),
        "per-epoch train/test accuracy, loss components, the lambda schedule",
    ),
    "configs-and-subsets.tar.gz": (
        lambda p: (p.startswith("configs" + os.sep) and p.endswith((".yaml", ".yml")))
        or (p.startswith("data" + os.sep + "subsets") and p.endswith(".json")),
        "every cell's config and the committed subset indices, so any cell "
        "can be re-run on byte-identical images",
    ),
    "logs.tar.gz": (
        lambda p: p.startswith("logs" + os.sep) and p.endswith((".log", ".txt")),
        "campaign logs: what was submitted when, and what failed",
    ),
}


def walk(root):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        # Hidden FILES are skipped for the same reason hidden dirs are, which
        # the original only did for dirs: results/.bak_MomentStem_results.xlsx
        # (a stray 2026-08-04 backup) was being shipped inside
        # result-tables.tar.gz as if it were a released table -- a stale
        # duplicate of the workbook, i.e. exactly the two-generations hazard
        # EXCLUDED_RUN_TREES guards against, one directory over.
        for f in files:
            if f.startswith("."):
                continue
            full = os.path.join(base, f)
            yield os.path.relpath(full, ROOT)


def collect(pred):
    # runs_det added 2026-08-17: the detection task's records were unreachable
    # from here, so no predicate could ship them however it was written.
    # EXCLUDED_RUN_TREES still governs which "runs*" trees may ship.
    tops = ("results", "runs", "runs_turing", "runs_dense", "runs_det",
            "runs_ckptfix", "configs", "data", "logs")
    out = []
    for t in tops:
        d = os.path.join(ROOT, t)
        if os.path.isdir(d):
            out += [p for p in walk(d) if pred(p)]
    return sorted(set(out))


def build(name, pred, outdir, verify):
    members = collect(pred)
    total = sum(os.path.getsize(os.path.join(ROOT, p)) for p in members)
    print(f"  {name:30} {len(members):7,} files  {total/1048576:8.1f} MB raw")
    if verify:
        return None
    path = os.path.join(outdir, name)
    # mtime=0 on the GZIP WRAPPER as well as on the members. tarfile's "w:gz"
    # stamps the current time into the gzip header, so fixing only the member
    # mtimes still gives a different checksum on every build -- which the
    # reproducibility check below caught, after the docstring had already
    # claimed otherwise.
    with gzip.GzipFile(path, "wb", compresslevel=9, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        for p in members:
            info = tar.gettarinfo(os.path.join(ROOT, p), arcname=p)
            info.mtime = int(time.mktime(EPOCH + (0, 0, -1)))
            info.uid = info.gid = 0
            info.uname = info.gname = "momentaux"
            with open(os.path.join(ROOT, p), "rb") as fh:
                tar.addfile(info, fh)
    return path, len(members), total


REPO_URL = "https://github.com/GCVCG/MomentAux"
RELEASE_TAG = "v1.0-benchmark"
RELEASE_URL = f"{REPO_URL}/releases/tag/{RELEASE_TAG}"


def _git_head():
    try:
        import subprocess
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        d = subprocess.run(["git", "log", "-1", "--format=%cs"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        return h, d
    except Exception:
        return "unknown", "unknown"


def write_readme(outdir, rows, sums):
    """dist/README.md: what each asset is, its checksum, and where else it lives.

    Written by the builder rather than by hand so the counts, sizes and
    checksums it quotes are the ones of the files beside it. It exists because
    the assets double as the journal's supplementary material, and a reviewer
    opening a tarball without the repository needs the manifest in the bundle.
    """
    head, date = _git_head()
    lines = [
        "# MomentAux benchmark: released data assets",
        "",
        "Supplementary material for *When does fusing hand-crafted knowledge with "
        "learned representations pay? A cost-normalized benchmark of stacking, "
        "substitution and interference* (AlMughrabi, Clop, Busam, Marques, "
        "Radeva).",
        "",
        "**These files are also published on GitHub.** They are the assets of "
        f"the tagged release [`{RELEASE_TAG}`]({RELEASE_URL}) of the study "
        f"repository [{REPO_URL}]({REPO_URL}), which also holds the code that "
        "produced them (training harness, configs, subsets, pinned filter "
        "banks, exporters and audit scripts). `docs/ARTIFACTS.md` in that "
        "repository documents every asset in detail and how to regenerate the "
        "paper's tables from them; this file is the short form that travels "
        "with the bundle.",
        "",
        f"Built from repository commit `{head}` ({date}) by "
        "`python scripts/make_release_assets.py --out dist/`. Tarballs are "
        "byte-reproducible (sorted members, fixed mtimes), so the checksums "
        "below identify this exact build.",
        "",
        "| asset | packed | files | contents |",
        "|---|---:|---:|---|",
    ]
    for name, n, raw, packed, desc in rows:
        lines.append(f"| `{name}` | {packed/1048576:.1f} MB | {n:,} | {desc} |")
    lines += [
        "",
        "## Which to open first",
        "",
        "- `result-tables.tar.gz`: one row per experimental cell in "
        "`results/all_results.csv`; the same pivoted by data fraction; the "
        "Excel workbook with a column dictionary; `results/law_audit.md`, the "
        "sign-law audit verbatim; the segmentation and detection grids; and the "
        "per-figure JSON records. This answers almost every question.",
        "- `configs-and-subsets.tar.gz`: the YAML of every cell and the committed "
        "subset indices, so any cell can be re-run on byte-identical images "
        "with `python train.py --config <cell>.yaml --seed N` from the "
        "repository.",
        "- `run-records.tar.gz`: every run's `final.json` and probe record, "
        "needed to re-run the seed-paired audit "
        "(`python analysis/audit_law_paired.py`).",
        "- `training-curves.tar.gz`: per-epoch `metrics.csv` for every run.",
        "- `logs.tar.gz`: campaign logs, what was submitted when and what failed.",
        "",
        "Model checkpoints (about 275 GB) are not released; every cell is "
        "re-trainable from the configs and subsets above.",
        "",
        "## Verifying",
        "",
        "```",
        "sha256sum -c SHA256SUMS",
        "```",
        "",
        "SHA256SUMS:",
        "",
        "```",
        *sums,
        "```",
        "",
    ]
    with open(os.path.join(outdir, "README.md"), "w") as fh:
        fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--verify", action="store_true",
                    help="report what each asset would contain, build nothing")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"{'asset':30} {'files':>13}  {'size':>11}")
    built = {}
    for name, (pred, desc) in ASSETS.items():
        r = build(name, pred, args.out, args.verify)
        if r:
            path, n, raw = r
            built[name] = (path, n, raw, desc)
    if args.verify:
        return

    sums, rows = [], []
    for name in sorted(built):
        p, n, raw, desc = built[name]
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        sums.append(f"{h.hexdigest()}  {name}")
        rows.append((name, n, raw, os.path.getsize(p), desc))
        print(f"  {name:30} -> {os.path.getsize(p)/1048576:8.1f} MB packed")
    with open(os.path.join(args.out, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sums) + "\n")
    write_readme(args.out, rows, sums)
    print(f"\nwrote {len(built)} assets + SHA256SUMS + README.md to {args.out}")


if __name__ == "__main__":
    main()
