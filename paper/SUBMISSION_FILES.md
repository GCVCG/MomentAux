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

Verified against the journal guide on 2026-08-10: 28 pages (limit 10-35),
50 references (limit 50), 243-word abstract (limit 250), 5 highlights at
most 73 characters (limits 3-5 and 85). Suggested reviewers are not
requested by this journal.
