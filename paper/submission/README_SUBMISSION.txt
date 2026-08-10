Information Fusion submission package
=====================================
The guide requires editable source files; a PDF alone is not accepted.

  main.tex, sections/, refs.bib   manuscript source (cas-dc, double column)
  cas-dc.cls, cas-sc.cls,
  cas-common.sty, unsrtnat.bst    template and bibliography style, bundled so
                                  the package builds without a local TeX tree
  figs/*.pdf                      all figures, separate files
  manuscript.pdf                  compiled manuscript, for reference
  highlights.txt                  separate highlights file (5 bullets, <=85 chars)
  graphical_abstract_submission.png
                                  separate graphical abstract, 1889x755 px
                                  (2.5:1, above the 1328x531 minimum at 300 dpi)
  cover_letter.md                 cover letter

Compile with:  pdflatex main && bibtex main && pdflatex main && pdflatex main

Verified against the journal guide (2026-08-10):
  pages 28 (limit 10-35) | references 50 (limit 50) | abstract 243 words
  (limit 250) | highlights 5 bullets, max 73 chars (limits 3-5, 85)
