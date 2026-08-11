# What to upload

Everything lives in `paper/`. There is no separate submission copy, because a
copy drifts from the source the moment either is edited.

The guide requires editable source files; a PDF alone is not accepted.

| Upload | File |
|---|---|
| Manuscript source | `main.tex`, `sections/*.tex`, `refs.bib` |
| Template files | `cas-dc.cls`, `cas-sc.cls`, `cas-common.sty`, `unsrtnat.bst` |
| Figures, separate files | `figs/*.pdf` |
| Compiled manuscript | `main.pdf` |
| Highlights, separate editable file | `highlights.txt` |
| Graphical abstract, separate file | `figs/graphical_abstract_submission.png` |
| Cover letter | `cover_letter.md` |

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`

Regenerate the graphical abstract in submission format (2.5:1, 1889x755,
above the 1328x531 minimum) with:

    GA_SUBMISSION=1 python figs/make_graphical_abstract.py

Verified by `python paper/check_submission.py`, which is the authority here
rather than this sentence. It checks the compiled PDF is present and not
older than its sources, that `main.pdf` is not excluded by `.gitignore` (it
was, which is why three submissions reached the referee without one), and
every limit the guide sets: pages 10-35, at most 50 references, abstract at
most 250 words, 3-5 highlights of at most 85 characters. Run it before
uploading; it exits non-zero with `--strict`.

Two companion checks are worth running with it:

    python paper/check_citations.py     # uncited entries, missing keys, cap
    python scripts/make_paper_numbers.py > paper/tables/numbers.tex

The second regenerates the numbers the prose cites by macro, so a figure in
the text cannot disagree with the table beside it. Suggested reviewers are
not requested by this journal.
