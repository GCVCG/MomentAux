# Referee report, round 4: Information Fusion

**Title:** When does fusing hand-crafted spectral knowledge with learned representations pay? A controlled, cost-normalized benchmark and its organizing law
**Round:** Revision 3
**Attached:** Manuscript source, cover letter, highlights, submission manifest, two new check scripts. **No response letter.** No compiled PDF, for the fourth time.

*Note on what I checked.* I diffed against the round 3 submission, recomputed all eight Wilson intervals and the audit partitions, traced the law-scope definition back through all four versions, checksummed every figure, verified that all 69 generated macros resolve, extracted the figure text to check for clipping, and ran the authors' own two check scripts. The revised audit reconciles exactly: 98 + 450 + 461 = 1,009, 395 + 66 = 461, 311 + 150 = 461 with 18 and 48 wrong. All eight confidence intervals are correct, the held-out partition sums (384 + 57 + 20 = 461), the compute components sum (3,080 + 82 + 356 = 3,518), and 75.3 minus 61.4 equals the 13.9 in Table 8.

---

## 1. Summary

This round adds a new task axis and repairs the two outstanding items from my previous report. Sections 9.1 to 9.5 transplant the prior to semantic segmentation across six populations and to object detection on VOC, and both headline findings are negative: a dense target on a dense task pays less than it does on classification, and detection is flat end to end and flat under the frozen-feature probe. All hard-coded numbers have been moved into a generated macro file so prose cannot drift from tables. The audit has been rescoped to aux-from-scratch cells, which moves the headline sign agreement from 79.1 percent to 85.7 percent.

## 2. Scope fit

Settled, as in my previous two reports. Nothing here changes it.

**Assessment axes.** Scope fit 4. Novelty and positioning 4. Technical soundness 4. Experimental rigor 4. Reproducibility 4, artifact still unverified. Clarity and presentation 3. Likely impact 4.

## 3. Strengths

The new task section is the most self-critical material in a paper that already had a lot of it. The authors state the bias before the result, noting that a dense spatial target on a dense spatial task is the most favourable venue the prior could be given and that a positive result there would be weak evidence. The headline is then negative: at roughly five images per class, where classification shows a universal floor near plus 1.5, segmentation returns plus 0.39 mIoU, and the frozen-feature probe agrees at plus 0.31, so it is not a readout artifact. The conclusion drawn is that the prior's value is not target-task alignment, which makes the mechanism claim narrower than it was.

The detection null is handled properly. Rather than reporting a flat end-to-end result and leaving the weak-head reading open, the authors probe the trunks and find G at minus 0.16 and minus 0.37 against a pre-registered falsifier of plus 1.5, then demonstrate the probe works in that regime by lifting fg_acc from 15.3 to 23.8. That is the right structure for a null: show the instrument is sensitive, then report that there is nothing to find. They also withdraw a plus 0.84 AP50 gain traced to a single seed whose regression branch collapsed, with the diagnostic evidence given, and note that the study's own seed-level checks would not have caught it because the cell still trains.

Two validity notes are volunteered that change conclusions against the authors. The dense grid was first run at 50 epochs and produced a monotone rising envelope with no right flank; at 200 epochs the VOC 10 percent baseline rises by 155 percent and the shape reverses. The authors report that an undertrained instrument produced the opposite scientific answer twice. Separately, they record that they registered the mIoU versus pixel-accuracy scale issue in advance and then violated it in their own prediction.

The macro system is the right structural response to a pattern I flagged last round. Sixty-nine numbers now come from one generated file, and I verified every macro used resolves and that the compute components, the ViT arithmetic and the dense counts all reconcile. The two round 3 blocking items are fixed: the 62.7 figure is back to 61.4 and consistent with Table 8, and the response letter problem was addressed by, unfortunately, deleting the letter.

The authors also diagnosed why three submissions arrived without a PDF. `paper/.gitignore` excluded `main.pdf`, so any package assembled from the repository silently dropped the one file the submission guide names, even though the manifest listed it every time. That is a good root-cause finding.

## 4. Blocking issues

**1. The headline number rose because the audit was rescoped, and this is nowhere disclosed.** Table 7's caption now says scope is aux-from-scratch and that cells carrying an ImageNet or self-supervised initialization are excluded. That restriction is new to this version. In-scope cells fall from 1,100 to 1,009, resolvable from 511 to 461, and the rate rises from 79.1 percent to 85.7 percent, which is the figure now in the abstract. The excluded cells are not a random sample of the audit: of the 50 that left the resolvable set, 41 were wrong and 9 were correct. Below the crossing, the number of correct cells is unchanged at 293 while 23 cells disappear, every one of them a wrong one, moving that flank from 87.7 to 94.2 percent.

I want to be equally clear that the rule itself looks legitimate. I traced the scope definition back through all four versions. Round 1 defined it as "in law scope (prior, from scratch, ≥3 seeds per arm)". Round 2 dropped the qualifier when the audit was rewritten for seed-paired uncertainty, and the count jumped from 989 to 1,075. Round 3 inherited that. So this round restores the original definition rather than inventing a convenient one, and round 2's own Section 3.7.5 already stated that self-supervised initializations lie outside the scope in which the branch was estimated. The most likely reading is that the scope filter was lost in the round 2 rewrite and has now been restored.

That reading does not make the silence acceptable. A number in the abstract moved by 6.6 points in the favourable direction, after two rounds in which it moved down, and the manuscript says nothing about it. There is no response letter to say it either. A referee who did not diff four versions would not know. *Necessary:* state the change explicitly, in the manuscript and in a response letter. Give the previous figure, the restored scope rule and why it is the correct one, the fact that it was applied in rounds 1 and 4 but not 2 and 3, and the correct-versus-wrong split of the excluded cells. Report what those excluded cells do under the law, since Section 3.7.5 already shows they depart from the branch and that is a finding rather than an embarrassment.

**2. The response letter has been deleted rather than written.** My previous report's first blocking issue was that the letter still answered only round 1 and contradicted the manuscript's numbers. It is now absent from the package entirely, and the cover letter does not mention the revision, the referee or any change. Three rounds of substantial work, including this round's rescoping, the new task section and two withdrawn results, are now invisible to the editor. This is the wrong resolution of that issue. *Necessary:* supply a response letter covering rounds 2, 3 and 4, with every number matching the manuscript.

## 5. Non-blocking issues

1. The regenerated Figure 3 has a clipped legend. Its canvas shrank from 248.4 by 212.4 points to 239.76 by 203.835, and the legend no longer fits: the extracted text reads "cannot test it: unresolved (450" with the closing parenthesis cut off. It is the only figure in the package with unbalanced parentheses in its text layer, so the others are clean. The counts themselves are correct and match Tables 7 and 8.
2. The variance decomposition changed with the rescoping, from R-squared 0.18 to 0.26, dataset 0.10 to 0.13, backbone 0.004 to 0.01, and residual SD 2.2 to 2.0 points. All four now propagate consistently through Sections 4.1 and 10, which is the macro system working. But the text around them still reads as though nothing moved, and a reader comparing versions would want a sentence.
3. Seven defined macros are unused: `camDelta`, `computeHnvl`, `computeHtwo`, `detFloorPcts`, `detFullDelta`, `detOneDelta`, `vitBestDelta`. Harmless, but `vitBestDelta` is the consistency check for the ViT arithmetic and would be better used in the text than left as a comment.
4. The ensemble analysis in Section 8.2 still uses a 50-epoch SimCLR checkpoint where the rest of the paper uses 200 and 800, and still does not say why. This is the third round I have raised it. Given that Section 6 is entirely about budget mattering, one sentence would close it.
5. The dense law block contributes 9 resolvable cells of which 9 are correct, all above the crossing. A 9 of 9 result is worth reporting but is thin, and the section says so. I would resist any temptation to fold these into the headline audit count.
6. Section 9.5 reports that Pascal-Context was built to supply the negative branch and returned plus 0.02 and plus 0.00, recorded as unresolved rather than as confirmation or falsification. That is the correct call and I note it approvingly, but the section would read better if the limitation appeared before the result rather than after.

## 6. Questions for the authors

1. Confirm that the aux-from-scratch scope was applied in round 1, lost in round 2, and restored now, and say what the audit rate is for the excluded initialization-bearing cells taken on their own.
2. Was `check_submission.py` run before this package was assembled?

## 7. Minor and editorial

Regenerate Figure 3 with a canvas that fits its legend. Supply a compiled PDF. Supply a response letter.

## Assessment of the revision

| Round 3 point | Status |
|---|---|
| B1. Response letter stale and contradicts manuscript | **Not addressed.** Letter deleted rather than rewritten. See blocking issue 2. |
| B2. 62.7 versus Table 8's 13.9 | **Addressed.** Now 61.4 via a generated macro, and 75.3 minus 61.4 equals 13.9. |
| NB1. GPU-hour components off by 2 | **Addressed.** 3,080 plus 82 plus 356 equals 3,518 exactly. |
| NB2. Residual SD 2.2 versus 2.1 | **Addressed.** Single macro, now 2.0 everywhere. |
| NB3. Alain and Bengio linear-probe citation | **Addressed.** Restored. |
| NB4. Small-data ViT remedies uncited | **Addressed well.** Now names the closest antecedent explicitly and states how it differs. |
| NB5. 50-epoch ensemble checkpoint | **Not addressed.** Third round. |
| NB6. Highlight 4 advertises the favourable budget | **Addressed.** Now states the ordering flips with data at 5x. |
| NB7. No compiled PDF | **Diagnosed but not fixed.** See below. |

The PDF item deserves its own note, because it is the sharpest thing in this package. The authors found the root cause, fixed `.gitignore` with a comment explaining why, wrote `check_submission.py` specifically to catch it, and documented in that script's docstring that three submissions had reached me without a PDF. I ran their script. It reports three failures, all of them the missing `main.pdf`. So the tool built to prevent this exact failure was either not run before packaging or was run and overridden, and the fourth submission arrives without a PDF anyway. I have now reviewed this manuscript four times without seeing it typeset, and I still cannot comment on page count, figure legibility at print size, or table overflow in the narrow CAS columns.

On the pattern across rounds: the corrections continue to be real and verifiable, and the new tooling is a genuine structural fix rather than another point repair. But the two things I flagged as habits rather than incidents both recurred. A number moved in the paper's favour without disclosure, which is the class of problem the first three rounds were about, even though this particular instance appears to be a legitimate restoration. And a process artifact went out unchecked despite a checker existing for it.

## 8. Confidential comments to the editor

The science continues to hold up. I recomputed everything checkable in this version and found no arithmetic error. The new segmentation and detection sections are honest work with negative headlines, and the detection null in particular is constructed the way a null should be.

Blocking issue 1 is the one I want you to look at directly, and I want to be careful in how I put it. I do not believe this is manipulation. I traced the scope rule to round 1's own table, where it was stated explicitly, and the round 2 text shows the authors already believed those cells were out of scope while their script was counting them. The restoration is almost certainly a bug fix. But the effect is that the abstract's headline rose 6.6 points, the excluded cells were 82 percent wrong against 13 percent in the retained set, and nothing anywhere in the package mentions it. With no response letter, an editor has no way to learn this except by diffing four submissions. That is not a state a manuscript should reach, whatever the intent behind it, and the fix is a paragraph rather than an experiment.

The deletion of the response letter is what makes it serious. I asked for the letter to be rewritten; removing it instead means that three rounds of concessions, including two withdrawn results this round, are now invisible to you. I would not accept this package without one.

Across four rounds this paper has never disputed a point, has published every number that went against it, and has twice run experiments I marked optional and reported that they cost it a claim. I remain confident there is no misconduct here. What I see is a group that fixes what is named and does not always generalize the naming, and this round is the clearest instance in both directions: they built a tool to generalize one class of fix, and then shipped without running it.

Salvageable immediately. Nothing outstanding requires computation.

## 9. Recommendation and confidence

**Minor revision.** Disclose the rescoping and its effect, supply a response letter covering rounds 2 to 4, regenerate Figure 3 so its legend fits, and include the compiled PDF. None of this requires new experiments. I do not need to see it again, but I would ask you to verify the disclosure paragraph yourself before acceptance.

**Confidence: 4 of 5.**

I could not verify the released artifact, the per-run records, the leave-one-dataset-out implementation, or the new segmentation and detection runs, and I have never seen this manuscript typeset. My finding on the rescoping rests on comparing four submitted sources and on the scope wording in round 1's own table, so it does not depend on the artifact. Whether the numbers come from the records they claim to remains the one thing I cannot establish from the paper, and it is why this is a 4 rather than a 5.
