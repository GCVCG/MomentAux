# Referee report, round two: Information Fusion

**Title.** When does fusing hand-crafted knowledge with learned representations pay? A controlled, cost-normalized benchmark and a measurable rule for stacking, substitution and interference

**Round.** Revision. Assessed against my round-one report. No response letter was supplied, so I reconstructed the point-by-point status from the manuscript itself.

---

## 1. Summary of the revision

The manuscript has grown by about 17 percent in the sections and gained a supplementary file. The growth is not padding. Seven new experimental blocks appeared, and four of them cut against the authors' own claims: a label-budget-matched probe over 30 cells and eight datasets, a fixed-step envelope, a doubled-schedule control on the headline ViT-B cell at six seeds per arm, the target ablation repeated off the selection set, the fusion arms repeated on two convolutional off-selection populations, a weight sweep on the transfer tax, and the full 92-cell tax population. The audit was rerun with eleven late-entering cells and moved from 85.7 to 85.4 percent. Several headline claims are now narrower than they were, and the abstract, highlights and decision guide were updated to match rather than left behind.

## 2. Point-by-point status

**Blocking issues from round one.**

| # | Point | Status |
|---|---|---|
| B1 | Readout's negative branch may be an artifact of the probe's label budget | **Addressed, with new evidence that confirms the objection.** Thirty champion cells across eight datasets re-evaluated at each cell's own per-class budget. Below the crossing the readout mean moves from $-1.66$ to $-0.19$, above it from $+0.10$ to $-0.13$, 19 of 30 cells lose more than half their magnitude and only 3 stay resolvable. Section 4.1 now states that Eq. 1 is true by construction and that the only empirical claim is the sign of the residual. Section 4.2 concludes that below the crossing the negative sign is substantially a statement about the ratio of evaluation labels to training labels. |
| B2 | Audit scored against a coin | **Addressed.** The majority-sign baseline of 74.1 percent is now reported, the coin p-value is called true and useless, and the result is written as 85.4 percent at 46.2 percent coverage. The coverage figure was not asked for and is the better of the two additions. |
| B3 | ViT-B headline may be stabilization | **Addressed, and the number changed.** Per-seed baselines are given (46.5, 39.9, 43.6), the collapse rule is checked explicitly, and the worst prior seed beats the best baseline seed by 22 points, so it is not stabilization. A doubled schedule then reduces the gain from $+26.01$ to $+6.71 \pm 0.90$ at six seeds per arm, and the estimate is shown moving with seed count. Table 12 now carries an SEM. See R1 below for what this does not settle. |
| B4 | Envelope confounded with step budget | **Addressed, and it changed a conclusion.** At a matched 600 steps the left flank survives, the registered falsifier did not fire, and the right flank disappears entirely: at 25 percent the matched-budget gain is $+5.92$ against $+0.16$, and the feature gain is $+7.40$ against $+0.44$. Neutrality at sufficiency is restated as sufficiency of optimization. See R2. |
| B5 | Pre-registration ledger unverifiable | **Addressed.** Section 3.8 now states the ledger is released as a version-controlled file with one timestamped commit per wave, band commits preceding result commits. I cannot check this, but it is now a specific claim an editor can check. |
| B6 | Section 6.3 takeaway contradicted Section 6.1 | **Addressed.** The takeaway now attributes the asymmetry claim to the decision-level fit alone and states that the sensor experiments are consistent with it but cannot establish it. |

**Non-blocking issues from round one.** N1 addressed with a new off-selection-set ablation that fired three of four registered falsifiers, and the random-target null is now correctly reduced to a 32 px statement with the margin over an uninformative target as the surviving claim. N2 addressed twice over: 92 transfer cells, all negative, median $-7.0$, worst $-26.3$, plus a $\lambda_0$ sweep showing the tax vanishes at or below 0.3, and the guide row now reads "the prior only at reduced $\lambda_0$" instead of "never". N3 partially addressed: the procedure still has one full prospective run, but the stack branch it contrasts against is now bounded by the new convolutional cells. N4 addressed for the grokking attribution, which is now cited; the cookbook and reality-check attributions still carry no citation, and one reference was dropped to stay at the cap of 50. N5 addressed on both counts: the compute figure is now reported as run-hours with the concurrency overstatement named, and the Table S1 margin at 1 percent reconciles now that the budget table reads $+2.27$. N6 addressed. N7 addressed, the dense Swin row is marked diagnostic. N8 addressed, Table 9 now carries a Tiny-ImageNet DeiT column. Most of the round-one editorial items are fixed.

**New evidence or relabelled evidence.** New. Every item above rests on runs that did not exist in the previous version, and the manuscript's headline claims moved as a result: the taxonomy's stack row is now a statement about attention backbones, since prior with augmentation stacks at one convolutional cell of six and is negative at the other five; the tax is now "not at full strength" rather than "never"; the target ablation's discriminating power is now identified as a CIFAR-100 property. Nothing was relabelled. The manuscript did not grow without resolving anything.

## 3. Remaining blocking issues

**R1. The model-scale conclusion is not yet supported at a budget where both models are trained.** Location: Section 5, Table 12, the highlights, the discussion and the conclusion. The conclusion that the attention deficit grows with model scale rests on ViT-S at $+13.00$ against ViT-B at $+26.01$, both at 100 epochs. The new control then shows the ViT-B baseline gains $+31.98$ from the schedule alone and the gain falls to $+6.71$ at 200 epochs. So the comparison is between one model and another that is demonstrably undertrained at the budget where the comparison is made, and at 200 epochs the ordering may invert. Section 5 says plainly that ViT-S was not rerun and that the ordering's survival is untested, which is the right disclosure, but the highlights, the subsection heading, the discussion and the conclusion still assert the trend. **Necessary**, and it is one cell: ViT-S/16 at 200 epochs, both arms, seeds matched to the ViT-B control. If the authors would rather not run it, the alternative is to withdraw the model-scale claim everywhere it appears and keep only the full-data ViT-B result with its two budgets stated.

**R2. Property (i) of Section 3.2 is falsified by the paper's own new experiment and still stands.** Location: Section 3.2, item (i), against the fixed-step paragraph in Section 4.3 and the repeat at Section 9.2. Section 3.2 states that because $\lambda(T)=0$ exactly, neutrality at sufficiency is structural rather than tuned, and that the right-hand end of every envelope in Table 6 is a prediction of the schedule. The new fixed-step result shows the right flank does not appear at matched steps at all, which means it is not a consequence of the schedule. The text calls this property load-bearing. Section 9.2 then reports that structural neutrality survives the change of task and metric, on the same reasoning. **Necessary**, and it is editing: restate (i) as the weaker true claim, that $\lambda(T)=0$ removes the prior's gradient by the end of training, and move the explanation of the right flank to the optimization account the new experiment supports.

## 4. Non-blocking issues

1. **A number that does not reconcile.** The fixed-step paragraph in Section 4.3 quotes the frozen-recipe CIFAR-100 series as $+1.42$, $+2.50$, $+5.15$, $+3.75$ and $+0.25$. Four match Table 6 exactly; the fifth is $+0.16$ there. The head-norm variant disclosed in Section 3.2 gives $+0.82$ at that cell, so it is not that either.
2. **The abstract is at 252 words** with macros expanded, against the guide's 250.
3. **The cover letter is out of step with the manuscript in three ways.** It lists three authors where the manuscript has five, omitting Albert Clop and Benjamin Busam. It states that the accuracy asymmetry between sources predicts which outcome happens, which is the claim Section 6.1 withdraws and Section 6.3 now explicitly declines to make. And it says the rule holds where the two sources are Sentinel-1 radar and Sentinel-2 optical, where the paper reports that fusion never pays. The manuscript is the more careful document of the two; the letter should be brought back to it.
4. **Page budget.** The sections grew from about 24,500 to about 28,600 words while the appendix shrank and a supplementary file absorbed some of it. Against a 35 page limit this is worth checking before upload rather than after.
5. **A pattern the paper now has the evidence to state and does not.** Four separate findings share one shape: the 50-epoch dense grid gave the opposite answer, SimCLR at 5 percent was step-starved rather than data-starved, the ViT-B gain falls by three quarters at a doubled schedule, and the envelope's right flank disappears at matched steps. Each is reported locally. Together they say that this study's largest effects are measured where the baseline has not finished training, and that a data-efficiency benchmark has to separate data from optimization to mean anything. That is a stronger and more transferable contribution than the sign law, and it is already paid for.
6. **Two amplification ratios still round oddly.** Table 9 gives 2.4 at 1 percent where the displayed values give 2.29, and Section 5.1's Tiny-ImageNet upper bound of 2.2 sits below the 2.31 the displayed values give. Compute both from the unrounded values or print one more digit.
7. Two attributions still carry no citation: the cookbook-level guidance for large-scale self-supervised use, and the reality-check and reproducibility claim at the end of Section 2.6.

## 5. Questions for the authors

1. What is ViT-S/16's gain at 200 epochs on ImageNet-100, both arms, seeds matched to the ViT-B control?
2. Which of $+0.16$ and $+0.25$ is the CIFAR-100 25 percent frozen-recipe gain, and does the fixed-step comparison change if it is the former?
3. Given the matched-budget result, do the authors want the sign law or the matched-budget prediction as the paper's headline account? The second predicts the end-to-end difference to 0.17 points across 30 cells without training the combination, which is a stronger and more usable claim than a sign at 46.2 percent coverage.

## 6. Minor and editorial

The round-one editorial items are fixed: the missing conjunction in Section 4.4, the unclosed clause in Section 4.2, the stray space in Section 4.6, the "missed low" phrasing, the "at most 1.6 times" claim contradicted by the 10 percent column, the "500 to 1.28M images" range now reading 150, and the Table 5 caption now explaining the twenty-one dataset identities rather than pointing at rows that were not there. One new slip: the fixed-step paragraph reads "That sharpens the claim, but it does change it", which parses but is likely not what was meant. Row shading is unchanged and still runs against the journal's table guidance; production will probably strip it.

English quality, reported separately: unchanged and good. The new material is written to the same standard as the old.

## 7. Confidential comments to the editor

This is the most responsive revision I have handled in some time, and I want to be specific about why rather than say so generically. Of my six blocking issues, five are resolved with experiments that did not exist before, and four of those experiments produced results that damage the authors' own claims: the readout term largely disappears once the probe's label budget is matched, the envelope's right flank disappears at matched steps, the headline ViT-B gain falls from 26.0 to 6.7 at a doubled schedule, and the stack row of the taxonomy holds at one convolutional cell of six. All four are in the main text with their registered falsifiers reported, including one that fired and one that is neither fired nor excluded. The authors also report that a three-seed version of the ViT-B control would have cleared a threshold that six seeds do not, and they keep an outlier seed that costs them the clearance. I would take that at face value.

The scope question I raised last round is unchanged in substance, though the fusion positioning is now better integrated and the sensor claims are correctly narrowed. It remains your call.

Two practical notes. The cover letter has drifted from the manuscript and currently asserts a claim the manuscript withdraws; it also drops two authors. If it goes forward as written it will misrepresent the submission. And I still cannot verify the released artifacts or the pre-registration ledger, which is now the main thing standing between this paper and full checkability.

## 8. Recommendation and confidence

**Major revision**, at the light end. The rubric puts it here rather than at minor revision only because R1 concerns a claim that appears in the highlights and the conclusion and that one experiment could overturn. If the authors run ViT-S at 200 epochs, or withdraw the model-scale claim and keep the two-budget ViT-B result, and fix R2, I would move to minor revision without needing to see anything else. Everything remaining is editing or a single cell.

**Confidence: 4 of 5.** More is checkable than last round, and every derived quantity I recomputed in the revised tables reconciles: the audit partitions, all five Wilson intervals, the coverage and majority-sign figures, the tax counts and every margin subtraction in the budget tables. I still cannot verify the ledger, the released artifacts, the compiled page count, or any number against the underlying run records, and I did not reassess the segmentation and detection sections at the depth I gave the classification grid.

**Assessment axes, with movement from round one.**

| Axis | Round 1 | Now | Note |
|---|---|---|---|
| Scope fit | 3 | 3 | Better integrated, sensor claims correctly narrowed, underlying question unchanged. |
| Novelty and positioning | 3 | 4 | The matched-budget prediction result is a genuinely new and usable contribution. |
| Technical soundness | 3 | 4 | The identity is now named as an identity, the null is honest, the label-budget confound is measured rather than argued. |
| Experimental rigor | 4 | 5 | Seven new blocks, four against interest, falsifiers reported as fired or not. |
| Reproducibility | 5 | 5 | Ledger release claimed, not yet verifiable. |
| Clarity and presentation | 3 | 3 | Prose improved, length increased, claim density still high. |
| Likely impact | 3 | 4 | The data-versus-optimization finding is more transferable than the sign law. |
