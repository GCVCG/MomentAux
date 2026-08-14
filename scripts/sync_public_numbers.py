#!/usr/bin/env python3
"""Keep README.md and docs/index.md in step with the paper's generated numbers.

WHY THIS EXISTS. The manuscript stopped drifting when its recurring figures
moved into a generated macro file (paper/tables/numbers.tex, written by
scripts/make_paper_numbers.py). The public README and the project page did
not get the same treatment, and they drifted exactly as the paper had: the
README advertised "268 (96%)" and the site "404 (79%)" for the sign-law
audit long after the paper reported 395 of 461 (85.7%). Both were numbers the
paper had superseded -- the 96% came from the independent-SEM audit the paper
explicitly withdrew, and the 79% from before the scope filter was repaired.
Anyone reading the repository was therefore reading retracted figures.

HOW IT WORKS. A number in the markdown is tagged with an HTML comment naming
its macro:

    | **resolvable, these test the law** | **461**<!--auditResolvable--> |

`--check` verifies every tagged value against numbers.tex and exits non-zero
on a mismatch; `--fix` rewrites them. The tag is invisible in rendered
Markdown, so the pages read normally on GitHub and GitHub Pages.

    python scripts/sync_public_numbers.py --check
    python scripts/sync_public_numbers.py --fix

The macro file is the single source of truth, so regenerating it and running
--fix is the whole update path after new runs land.
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MACROS = os.path.join(ROOT, "paper", "tables", "numbers.tex")
TARGETS = [os.path.join(ROOT, "README.md"), os.path.join(ROOT, "docs", "index.md")]

# The macro file writes LaTeX thousands separators; Markdown wants commas.
TEX = {"{,}": ",", r"\%": "%", "~": " "}


def load_macros(path):
    out = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}\s*(?:%.*)?$",
                         open(path).read(), re.M):
        val = m.group(2)
        for a, b in TEX.items():
            val = val.replace(a, b)
        out[m.group(1)] = val
    return out


TAG = re.compile(r"(?P<value>[-+−]?[\d][\d,\.]*%?)(?P<tag><!--\s*([A-Za-z]+)\s*-->)")


def same(a, b):
    """Compare VALUE, not typography.

    Prose writes a gain as "+0.39" where the macro holds "0.39", and uses a
    typographic minus (U+2212) where the macro has a hyphen. Both are the same
    number, and a guard that reported them as drift would be trained away
    within a week. Anything else -- a different digit, a lost comma group --
    is a genuine mismatch and is reported.
    """
    norm = lambda v: v.replace("−", "-").lstrip("+")
    return norm(a) == norm(b)


def process(path, macros, fix):
    src = open(path).read()
    bad, fixed, unknown = [], 0, []

    def sub(m):
        nonlocal fixed
        name = m.group(3)
        if name not in macros:
            unknown.append(name)
            return m.group(0)
        want = macros[name]
        if same(m.group("value"), want):
            return m.group(0)
        bad.append((name, m.group("value"), want))
        if fix:
            fixed += 1
            # keep the author's leading sign if the macro carries none
            lead = m.group("value")[0] if m.group("value")[0] in "+-−" else ""
            keep = lead if want[0] not in "+-−" else ""
            return keep + want + m.group("tag")
        return m.group(0)

    new = TAG.sub(sub, src)
    if fix and new != src:
        open(path, "w").write(new)
    return bad, fixed, unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not (args.fix or args.check):
        ap.error("pass --check or --fix")

    macros = load_macros(MACROS)
    print(f"{len(macros)} macros loaded from paper/tables/numbers.tex")
    problems = 0
    for path in TARGETS:
        rel = os.path.relpath(path, ROOT)
        bad, fixed, unknown = process(path, macros, args.fix)
        tagged = len(re.findall(TAG, open(path).read()))
        if unknown:
            print(f"  {rel}: UNKNOWN MACRO {sorted(set(unknown))}")
            problems += len(set(unknown))
        if args.fix:
            print(f"  {rel}: {tagged} tagged values, {fixed} corrected")
        else:
            for name, got, want in bad:
                print(f"  {rel}: {name} is {got}, numbers.tex says {want}")
            problems += len(bad)
            print(f"  {rel}: {tagged} tagged values, {len(bad)} stale")
    if problems:
        print(f"\n{problems} problem(s). Run --fix, or regenerate numbers.tex first:")
        print("  python scripts/make_paper_numbers.py > paper/tables/numbers.tex")
        sys.exit(1)
    print("\npublic numbers match the paper")


if __name__ == "__main__":
    main()
