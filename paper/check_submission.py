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
# .25\textwidth under cas-dc; see the build-log check for why it is whitelisted.
CLASS_BOX_PT = "123.62721"


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
        # The graphical abstract and the highlights are typeset into main.pdf for
        # convenience but are SUBMITTED AS SEPARATE FILES, so they are not part of
        # the manuscript the page limit applies to. Counting them reported a false
        # failure at 36 pages while the manuscript itself was 34. Discount one page
        # per preliminary block that is actually typeset (\submissionmodetrue drops
        # both, and then nothing is discounted).
        head = open(main_tex).read()
        prelim = 0
        if re.search(r"^\s*\\submissionmodefalse", head, re.M):
            prelim = sum(bool(re.search(rf"\\begin{{{env}}}", head))
                         for env in ("graphicalabstract", "highlights"))
        m = n - prelim
        note = f"pages: {m} (limit {PAGES[0]}-{PAGES[1]})"
        if prelim:
            note += f"; main.pdf is {n} incl. {prelim} separately-submitted page(s)"
        check(PAGES[0] <= m <= PAGES[1], note)

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

        # highlights.txt is the file the JOURNAL actually receives (the guide
        # wants highlights as a separate editable upload), and in submission
        # mode the LaTeX block is not typeset at all -- so a drift between the
        # two is invisible in the compiled PDF. It had already happened: the
        # .txt still said "2,894-configuration" and carried a superseded fifth
        # bullet while main.tex said 2,978. Compare them on rendered text.
        hlf = os.path.join(HERE, "highlights.txt")
        if os.path.exists(hlf):
            # Resolve the generated macros first: highlights.txt must carry
            # literal values (it goes into a plain-text form field), so an
            # unexpanded \computeCells would look like a mismatch when it is
            # actually the point. Then fold the two spellings of the same
            # glyph, since the .txt is deliberately ASCII.
            nums = os.path.join(HERE, "tables", "numbers.tex")
            # The value pattern must allow ONE level of nesting: the counts are
            # written 2{,}978 for LaTeX's thin-space grouping, and a plain
            # [^}]* stops at the inner brace and yields "2,".
            defs = dict(re.findall(
                r"\\newcommand\{\\(\w+)\}\{((?:[^{}]|\{[^{}]*\})*)\}",
                open(nums).read())) if os.path.exists(nums) else {}
            raw = re.findall(r"\\item\s+(.*)", hl.group(1))
            expanded = []
            # Longest name first, so \computeCells is not eaten by \compute.
            order = sorted(defs, key=len, reverse=True)
            for it in raw:
                for k in order:
                    it = it.replace("\\" + k, defs[k].replace("{,}", ","))
                expanded.append(rendered(it))
            txt = [ln.strip().lstrip("-").strip()
                   for ln in open(hlf).read().splitlines()
                   if ln.strip().startswith("-")]

            def norm(xs):
                return [re.sub(r"\s+", " ", x.replace("\u00d7", "x")).strip()
                        for x in xs]
            check(norm(txt) == norm(expanded),
                  "highlights.txt matches the highlights block in main.tex")

    # ---- build log -------------------------------------------------------
    # A referee read the shipped log and found an overfull box we had not, then
    # reported it for five consecutive rounds. Boxes are now checked here, with
    # ONE whitelisted exception identified by its exact size.
    #
    # THE WHITELISTED BOX IS THE CLASS'S OWN. cas-common.sty sets the ARTICLE
    # INFO column as \hbox_to_wd:nn {\z@} {\box \g_stm_key_box}: a box of width
    # .25\textwidth deliberately placed in a zero-width hbox so the abstract can
    # flow beside it. .25\textwidth is 123.62721pt on this class, which is the
    # excess reported, and it cannot be silenced from main.tex because
    # \twocolumn's optional argument is typeset inside \@parboxrestore, whose
    # final act is \sloppy (\hfuzz=0.5pt). Whitelisting by exact size rather
    # than raising \hfuzz is deliberate: a blanket \hfuzz would also have hidden
    # the 3.01pt table overrun this check caught on its first run.
    log = os.path.join(HERE, "main.log")
    if os.path.exists(log):
        print("Build log")
        boxes = re.findall(r"(Overfull \\[hv]box \(([0-9.]+)pt too (?:wide|high)\))",
                           open(log, errors="replace").read())
        stray = [b for b, pt in boxes if pt != CLASS_BOX_PT]
        known = sum(1 for _, pt in boxes if pt == CLASS_BOX_PT)
        check(not stray, "no overfull boxes beyond the known class artifact"
                         + (f"; found {len(stray)}: {stray[0]}" if stray else
                            f" ({known} whitelisted)"))
        text = open(log, errors="replace").read()
        check("Citation" not in text or "undefined" not in text,
              "no undefined citations or references")

    # ---- generated macros ------------------------------------------------
    # Every macro in numbers.tex must be PRINTED somewhere. Unused macros were
    # raised in two review rounds, and they matter beyond tidiness: a macro
    # nothing prints cannot drift visibly, so it quietly stops being audited
    # while still looking like a checked number.
    macros = os.path.join(HERE, "tables", "numbers.tex")
    if os.path.exists(macros):
        print("Generated numbers")
        names = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", open(macros).read()))
        body = ""
        for f in [main_tex] + sorted(glob.glob(os.path.join(HERE, "sections", "*.tex"))):
            body += open(f).read()
        used = set(re.findall(r"\\([A-Za-z]+)", body))
        unused = sorted(names - used)
        check(not unused, "every generated macro is used"
                          + (f"; unused: {', '.join(unused)}" if unused else
                             f" ({len(names)} macros)"))

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
