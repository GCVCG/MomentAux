"""Citation hygiene for the manuscript, as a check rather than an inspection.

Three rounds of review lost three attributions in round 2 and a fourth
(alain2016understanding, the linear-probe protocol that produces G) in round 3.
Every one was found by a referee reading the sources. The failure is not that
any individual citation was hard to notice; it is that nothing was CHECKING,
so each loss had to be discovered by hand, one at a time. This script turns
that class of defect into a build step.

It reports three things:

  UNCITED   an entry in refs.bib that no \\cite reaches. Either the claim it
            supported was rewritten and lost its attribution -- the exact
            round 3 defect -- or the entry is dead weight against the
            journal's hard 50-reference limit. Both need a decision.
  MISSING   a \\cite with no entry, which is an undefined reference at build
            time and is caught by LaTeX anyway; reported here for completeness.
  COUNT     the number of entries that will actually print, against the
            journal's limit. Exceeding it is a desk-reject risk, which is why
            restoring an attribution means trading one out, not appending.

    python paper/check_citations.py            # report
    python paper/check_citations.py --strict   # non-zero exit if anything is wrong
"""
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIMIT = 50   # Information Fusion: hard cap, non-compliance is a desk-reject risk


def bib_keys(path):
    """Keys defined in the .bib, in file order."""
    with open(path) as f:
        return re.findall(r"^@\w+\{\s*([^,\s]+)\s*,", f.read(), re.M)


def cited_keys(tex_files):
    """Every key reached by any \\cite variant, mapped to where it is used."""
    used = {}
    pat = re.compile(r"\\(?:cite|citep|citet|citealp|citeauthor|citeyear)"
                     r"\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}")
    for path in tex_files:
        with open(path) as f:
            body = f.read()
        # a % that is not \% starts a comment: a citation there does not print
        body = re.sub(r"(?<!\\)%.*", "", body)
        for m in pat.finditer(body):
            for key in m.group(1).split(","):
                key = key.strip()
                if key:
                    used.setdefault(key, set()).add(os.path.basename(path))
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    tex = sorted(glob.glob(os.path.join(HERE, "*.tex")) +
                 glob.glob(os.path.join(HERE, "sections", "*.tex")) +
                 glob.glob(os.path.join(HERE, "tables", "*.tex")))
    keys = bib_keys(os.path.join(HERE, "refs.bib"))
    used = cited_keys(tex)

    uncited = [k for k in keys if k not in used]
    missing = sorted(k for k in used if k not in keys)

    print(f"bib entries: {len(keys)}   cited: {len(keys) - len(uncited)}   "
          f"limit: {LIMIT}")
    for k in uncited:
        print(f"UNCITED  {k}")
    for k in missing:
        print(f"MISSING  {k}  (cited in {', '.join(sorted(used[k]))})")
    over = max(0, len(keys) - len(uncited) - LIMIT)
    if over:
        print(f"OVER LIMIT by {over}: trade an attribution out, do not append")

    bad = len(uncited) + len(missing) + over
    print("citations clean" if not bad else f"{bad} issue(s)")
    if args.strict and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
