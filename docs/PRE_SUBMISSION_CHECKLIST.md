# Pre-submission checklist

User decision, 2026-08-08: **the repository goes public when the paper is
submitted to Information Fusion**, not before. Everything below must happen
in the same sitting as the submission, because the manuscript already cites
the repository URL in three places and each becomes a dead link until the
switch is flipped.

## 1. Make the repository public

- [ ] `gh repo edit GCVCG/MomentAux --visibility public --accept-visibility-change-consequences`
- [ ] Confirm the three URLs in the paper resolve: title-page footnote,
      abstract, data-availability statement.

Already done in advance:

- [x] `LICENSE` (MIT, code) and `LICENSE-DATA` (CC BY 4.0, derived data),
      with an explicit statement that no image dataset is redistributed.
- [x] README rewritten around the benchmark, with the released-artifact table.
- [x] `docs/ARTIFACTS.md` describing every release asset.
- [x] GitHub About description and 12 topics set.
- [x] Release `v1.0-benchmark` with 39 MB of assets and `SHA256SUMS`.
- [x] Credential handling generalised in `CLAUDE.md`; lab workstation IP
      removed. What remains is the author's own username, university cluster
      hostnames and filesystem paths, which are ordinary for a research
      repository and carry no access on their own.

## 2. Mint an archival DOI

A bare GitHub link does not satisfy the ACM "Artifacts Available" badge and
is vulnerable to link rot; reviewers increasingly expect an archived copy.

- [ ] Link the GCVCG organisation to Zenodo and enable it for this repository.
- [ ] Re-publish (or re-tag) `v1.0-benchmark` so Zenodo captures it.
- [ ] Add the DOI to the paper's data-availability statement and to the
      README citation block.

## 3. Decide what `CLAUDE.md` does in public

It is a ~2000-line working ledger: every pre-registered prediction with its
band and falsifier, every landing scored against it, every retraction, and
every operational failure. Publishing it is unusual.

The argument for: it *is* the pre-registration record. The paper claims
predictions were registered before results, and this is the only artifact
that evidences it. It also contains four missed predictions and two
retracted laws, which is exactly the material a sceptical reader wants.

The argument against: it is candid about process in a way no reviewer
expects to read, and it records mistakes in detail.

Recommendation: keep it, and reference it from the reproducibility appendix
as the pre-registration trail. Read it once end to end first.

## 4. Final manuscript checks

- [ ] Generative-AI declaration: confirm the tool name and stated purpose,
      or delete the section if nothing is to be disclosed.
- [ ] Regenerate every table and figure from `results/` so the paper and the
      released CSVs cannot disagree.
- [ ] Re-run `analysis/audit_law_paired.py` and confirm the numbers in
      Tables 2 and 3 still match.
- [ ] Confirm no LaTeX source comment records venue-framing or internal
      instructions (this was caught once already).

## Going public (run in this order, at submission, not before)

1. `bash scripts/scrub_for_release.sh --check` — lists what carries a machine
   address, a home path or a cluster account. Read the list.
2. `bash scripts/scrub_for_release.sh` — rewrites them to environment
   variables. **Then read the diff by hand**; a sed pass is not a review.
   Do not run this while the cron keeper is still feeding the cluster, since
   it rewrites the scripts that keeper invokes.
3. Verified already, and worth re-running after any rebase:
   `git log --all -p | grep -E 'sshpass|BEGIN .* PRIVATE KEY|(PASS|TOKEN|SECRET|API_KEY)[[:space:]]*='`
   returned nothing across all 243 commits. No credential has ever been
   committed. The credential that was once pasted in a working transcript
   was rotated on 2026-07-22 and never entered the repository.
4. Decide on `CLAUDE.md`. Recommendation: **publish it**. It is the
   pre-registration record, and the paper claims predictions were recorded
   before results existed. That claim is worth much more with the ledger
   visible than with it withheld, including the entries where a prediction
   missed. It contains no credentials, by (3).
5. Flip the repository to public, then mint the Zenodo DOI from the tag and
   add the DOI to the paper's data-availability statement.

## Guide-for-authors compliance (VERIFIED against the downloaded guide, 2026-08-10)

Checked against the journal's own PDF, not a third-party summary. The
mandatory LaTeX requirements are enforced by desk rejection, so they are
listed first.

| Mandatory requirement | Limit | Ours | Status |
|---|---|---|---|
| Manuscript length, final formatted | 10-35 pages | 27 | MET |
| **References, research article** | **max 50** | **50** | **MET (was 74)** |
| Must use the journal LaTeX template | cas-dc | cas-dc | MET |
| Editable source files at submission (.tex, figures) | required | prepared | MET |
| PDF alone | not accepted | n/a | MET |

| Other requirement | Ours | Status |
|---|---|---|
| Abstract | 243 words (max 250) | MET |
| Highlights: 3-5 bullets, <=85 chars, separate file with "highlights" in the name | 5 bullets, 65-73 chars, `paper/highlights.txt` | MET |
| Graphical abstract, separate file | `figs/graphical_abstract_submission.png`, 1889x755 (2.5:1, >=1328x531) | MET |
| Artwork as separate files, logical names | figures are separate PDFs | MET |
| CRediT statement | present | MET |
| Declaration of competing interests | present | MET |
| Declaration of generative AI use | present, Elsevier wording | MET |
| Funding sources | in acknowledgments | MET |
| Cover letter | `paper/cover_letter.md` | MET |

NOT FOUND anywhere in the guide: any request for suggested reviewers, and
any structured-abstract or keyword-count requirement. Treated as not
required.

THE REFERENCE CUT (2026-08-10). The guide caps research articles at 50
references and we had 74. Twenty-four were removed, chosen by protecting
every load-bearing citation first: datasets we train on, backbones and
optimizers we run, comparators we run, the method's own lineage, the
measurements we use, the fusion-theory citations that carry the venue case,
and the references the reviewer specifically asked us to add. What was cut
came from the discretionary remainder, preferring one of each redundant
pair (`vonrueden2022informedpre` where the 2023 survey stays;
`wightman2021resnet` where `musgrave2020metric` stays) and single-use
decorative citations. Two swaps were made after inspection:
`newell2020useful` was retained because it states the standing criticism of
linear evaluation that our methodology has to answer, and
`kornblith2019better` was cut in its place.
