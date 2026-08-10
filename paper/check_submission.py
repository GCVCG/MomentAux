"""Verify the submission package against the journal's stated requirements.

WHY THIS EXISTS. Three submissions reached the referee without a compiled
PDF. Not because anyone forgot -- SUBMISSION_FILES.md listed `main.pdf` every
time -- but because `paper/.gitignore` excluded it, so every package assembled
from the repository silently dropped exactly the file the rule named. A
checklist that a human reads cannot catch a file that is invisible; a script
that looks for the file can.

It also re-derives the compliance numbers that SUBMISSION_FILES.md used to
state as prose. Those had drifted to "28 pages" while the manuscript was 32,
which is the same class of defect as every number the referee has caught in
the text: a figure written once and not re-checked.

Information Fusion, research article:
  10-35 pages, maximum 50 references, abstract <= 250 words,
  3-5 highlights of at most 85 characters, editable source AND a PDF.

    python paper/check_submission.py          # report
    python paper/check_submission.py --strict # non-zero exit on any failure
"""
import argparse
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES = (10, 35)
MAX_REFS = 50
MAX_ABSTRACT = 250
HIGHLIGHTS = (3, 5)
MAX_HIGHLIGHT = 85


def rendered(tex):
    """Strip the LaTeX a character count should not include."""
    t = tex.replace(r"$\times$", "\u00d7").replace(r"$\sim$", "~")
    t = re.sub(r"\\[a-zA-Z]+\s*\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\[a-zA-Z]+|[{}$\\]", "", t)
    return " ".join(t.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    fail = []

    def check(ok, msg):
        print(("  ok    " if ok else "  FAIL  ") + msg)
        if not ok:
            fail.append(msg)

    pdf = os.path.join(HERE, "main.pdf")
    main_tex = os.path.join(HERE, "main.tex")

    print("Package")
    check(os.path.exists(pdf), "compiled main.pdf is present")
    if os.path.exists(pdf):
        # A PDF older than its sources is worse than none: it looks complete.
        srcs = ([main_tex, os.path.join(HERE, "refs.bib")]
                + glob.glob(os.path.join(HERE, "sections", "*.tex"))
                + glob.glob(os.path.join(HERE, "tables", "*.tex"))
                + glob.glob(os.path.join(HERE, "figs", "*.pdf")))
        newest = max(os.path.getmtime(s) for s in srcs if os.path.exists(s))
        check(os.path.getmtime(pdf) >= newest,
              "main.pdf is at least as new as every source it is built from")
    # The rule that caused the three missing PDFs.
    gi = os.path.join(HERE, ".gitignore")
    ignored = os.path.exists(gi) and re.search(
        r"^\s*main\.pdf\s*$", open(gi).read(), re.M)
    check(not ignored, "main.pdf is not excluded by .gitignore")

    print("Journal limits")
    if os.path.exists(pdf):
        info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
        n = int(re.search(r"Pages:\s+(\d+)", info).group(1))
        check(PAGES[0] <= n <= PAGES[1], f"pages: {n} (limit {PAGES[0]}-{PAGES[1]})")

    bbl = os.path.join(HERE, "main.bbl")
    if os.path.exists(bbl):
        nref = len(re.findall(r"\\bibitem", open(bbl).read()))
        check(nref <= MAX_REFS, f"references: {nref} (limit {MAX_REFS})")

    src = open(main_tex).read()
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S)
    if abstract:
        words = len(rendered(abstract.group(1)).split())
        check(words <= MAX_ABSTRACT, f"abstract: {words} words (limit {MAX_ABSTRACT})")

    hl = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", src, re.S)
    if hl:
        items = [rendered(i) for i in re.findall(r"\\item\s+(.*)", hl.group(1))]
        check(HIGHLIGHTS[0] <= len(items) <= HIGHLIGHTS[1],
              f"highlights: {len(items)} (limit {HIGHLIGHTS[0]}-{HIGHLIGHTS[1]})")
        longest = max(len(i) for i in items) if items else 0
        check(longest <= MAX_HIGHLIGHT,
              f"longest highlight: {longest} chars (limit {MAX_HIGHLIGHT})")

    print("Files named in SUBMISSION_FILES.md")
    listed = open(os.path.join(HERE, "SUBMISSION_FILES.md")).read()
    for m in re.finditer(r"`([^`]+)`", listed):
        item = m.group(1)
        if "*" in item:
            check(bool(glob.glob(os.path.join(HERE, item))), f"{item} matches files")
        elif "/" in item or "." in item:
            if item.endswith((".tex", ".pdf", ".bib", ".cls", ".sty",
                              ".bst", ".txt", ".md", ".png")):
                check(os.path.exists(os.path.join(HERE, item)), f"{item} exists")

    print("\nsubmission ready" if not fail else f"\n{len(fail)} problem(s)")
    if args.strict and fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
