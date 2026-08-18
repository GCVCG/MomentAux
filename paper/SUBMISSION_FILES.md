# What to upload

Everything lives in `paper/`, with one deliberate exception noted in the
table below. There is no separate submission copy, because a copy drifts from
the source the moment either is edited.

THE RESPONSE LETTER IS PART OF THE UPLOAD. It lives in `docs/` rather than
`paper/` so that `paper/` holds only what LaTeX compiles, and that tidying is
exactly how it went missing from a submission once: the manifest listed only
files under `paper/`, so the assembled package silently dropped it and the
referee reported three rounds of revisions as invisible. `check_submission.py`
now fails if it is absent.

The guide requires editable source files; a PDF alone is not accepted.

| Upload | File |
|---|---|
| Manuscript source | `main.tex`, `sections/*.tex`, `refs.bib` |
| Template files | `cas-dc.cls`, `cas-sc.cls`, `cas-common.sty`, `unsrtnat.bst` |
| Figures, separate files | `figs/*.pdf` |
| Compiled manuscript | `main.pdf` |
| Highlights, separate editable file | `highlights.txt` |
| Graphical abstract, separate file | `figs/graphical_abstract_submission.pdf` |
| Cover letter | `cover_letter.md` |
| Response to the referee | `../docs/RESPONSE_TO_REFEREE.md` |

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`

The highlights and the graphical abstract are typeset into `main.pdf` only
when `\submissionmodefalse` is set in `main.tex`. It is ON for submission, so
the compiled manuscript is the article alone: the guide asks for both as
separate uploads, and typesetting them into the manuscript would both
duplicate a deliverable and spend two of the 35 permitted pages on it.

We upload the PDF graphical abstract rather than the PNG because the guide
names "TIFF, EPS, PDF or MS Office" as the preferred types; the PNG is
regenerated alongside it and is a valid fallback if the submission system
refuses vector input. Both are 2.5:1 and clear the 1328x531 minimum (the PDF
renders to 1890x756 at 300 dpi, and being vector is readable at any size).
Regenerate both with:

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
