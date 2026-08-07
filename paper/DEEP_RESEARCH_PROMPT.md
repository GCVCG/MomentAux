# Deep research prompt: presentation and positioning standards for an Information Fusion benchmark paper

Copy everything below the line into Claude's deep research mode. Bring the
report back and it will be applied to `paper/`.

---

## Context you are researching for

I am preparing a submission to **Information Fusion** (Elsevier, CAS
double-column template `cas-dc`). The paper is a **controlled,
cost-normalized benchmark of data-efficiency interventions**, plus a
predictive empirical law that organizes its results.

Concretely:

- One frozen training recipe and committed data-subset indices, so that
  every intervention is measured on byte-identical images.
- 2,800 experimental cells, roughly 8,900 training runs, 14 datasets,
  7 backbone families, 500 to 1.28M images, 32 to 224 px, 5.7M to 86M
  parameters.
- Interventions compared at declared training-cost multiples: a fixed
  hand-crafted spectral auxiliary target (about 1.02x), self-supervised
  pre-training with SimCLR, SimSiam and DINO (2x), ImageNet transfer,
  DeiT-strength augmentation, and a learned-teacher (FitNets) control.
- An empirical law, `Delta = G + readout(base)`, decomposing each
  intervention's end-to-end accuracy gain into a feature-level gain `G`
  measured by a linear probe and a readout term that depends only on
  baseline accuracy. Registered in advance, the law predicted the feature
  gain of an unseen backbone family and of ImageNet-scale cells; it holds
  on 268 of 278 cells where the readout is resolvable against its own
  uncertainty.
- A fusion taxonomy: different-currency sources **stack**, same-currency
  sources **substitute**, and injecting a shaping prior into an already
  informed initialization **taxes** in proportion to what that
  initialization carries.

The science is done and the numbers are fixed. **I am not asking you to
evaluate the science.** I am asking for evidence-based guidance on
presentation, structure and positioning, so that the manuscript reads
like a top-tier benchmark paper and survives review at a high-impact
fusion venue.

## What I need from you

Research each of the six areas below and return **specific, actionable,
sourced recommendations**. Prefer primary sources: publisher author
guides, journal editorials, style manuals, accessibility standards,
reproducibility checklists, and highly cited exemplar papers that I can
imitate. For every recommendation, state the source and say plainly
whether it is a hard requirement, a strong convention, or a matter of
taste. Where sources disagree, say so rather than picking one silently.

### 1. Section signposting and reader navigation

I have added a short "roadmap" paragraph to every section that has
subsections, naming each subsection and what it contributes.

- Is this the accepted convention in high-impact empirical venues, or
  does it read as padding? Find evidence, ideally from editorials, referee
  guidance or writing manuals aimed at computer-science and engineering
  journals.
- What is the recommended **length and placement** of such a paragraph?
  Should it forward-reference by number ("Section 4.2 shows..."), by name,
  or both?
- Are there venues or editors that explicitly discourage it?
- Should the same treatment be applied to the appendix, or is that
  overkill?
- Is there a recommended pattern for the **opening sentence of a results
  section** in benchmark papers, and how do the best-regarded benchmark
  papers (for example the NeurIPS Datasets and Benchmarks track) handle
  the transition from method to results?

### 2. Table design for dense quantitative benchmarks

My tables are `booktabs`-based, in a narrow double-column format, and
several are dense grids of accuracy deltas across data fractions.
I have just applied three changes and want them validated or corrected:

- **Alternating row shading (zebra banding).** Is this recommended or
  discouraged in print scientific typography? `booktabs` documentation is
  famously opinionated about rules and spacing; what does it, and
  comparable style guidance, say about background shading? What grey level
  is safe for print, for greyscale printing, and for accessibility
  contrast standards? Does it interact badly with `\multirow`,
  `\multicolumn` or footnote rows, and what is the accepted workaround?
- **Better-direction arrows** in column headers (up for higher is better,
  down for lower is better). Where did this convention originate, how
  widely is it used, and what is the accepted way to define it: in the
  caption, in a legend row, or once globally? Is it acceptable when a
  column is a signed difference where neither direction is simply better?
- **Stating the sort order in the caption.** Is this an actual expectation
  anywhere, or my own invention? What do style guides say a table caption
  must contain, and in what order? Find concrete guidance on
  self-contained captions for tables in engineering journals.
- What are the current best practices for making dense numeric tables
  readable: column alignment on the decimal point, significant figures,
  when to bold the best result, whether to report standard errors inline
  or in a released file, and how to mark cells that were not measured?
- Is there evidence on **tables versus figures** for this kind of
  multi-factor grid, that is, when a heatmap or small-multiple plot beats
  a numeric table for a reviewer?

### 3. Graphical abstracts

Information Fusion accepts a graphical abstract. Mine is a three-panel
figure: the fused components and their cost, a scatter of the law, and a
taxonomy of outcomes.

- What are Elsevier's **current, exact** specifications: minimum and
  preferred pixel dimensions, aspect ratio, file formats, font size
  floor, and whether text-heavy diagrams are acceptable?
- What distinguishes an effective graphical abstract from a decorative
  one? Find published guidance or studies on graphical abstract design,
  including any evidence on readership or citation effects.
- Is **all-uppercase** labelling advisable, or does it hurt legibility?
  Find typographic evidence rather than opinion.
- Should a graphical abstract carry a legend and axis labels, or be purely
  schematic?
- Is it acceptable, or expected, for the graphical abstract to reuse a
  panel from a figure in the paper?

### 4. Reproducibility and artifact release

I am releasing the full harness, every configuration, the committed
subset indices, all per-run JSON records, logs, and the complete result
tables as CSV, with the repository cited in the paper.

- What do the leading reproducibility checklists require, and what do
  they merely encourage? Cover at least the ACM artifact badging scheme,
  the NeurIPS reproducibility and datasets-and-benchmarks checklists, and
  any Elsevier or Information Fusion specific policy.
- Where should a repository link appear: abstract, footnote on page one,
  a data-availability statement, or all three? Is a bare GitHub link
  acceptable, or is an archived DOI (Zenodo, figshare, Software Heritage)
  now expected? What do publishers say about link rot?
- What licence choices are conventional for code plus derived result
  tables, and what are the pitfalls when the underlying datasets have
  their own licences and cannot be redistributed?
- What should a repository README contain to satisfy an artifact
  evaluator? Is there a template or checklist worth following?
- How should compute and **carbon** reporting be presented? Find current
  guidance and any venue that requires it, plus the accepted methodology
  for estimating emissions from GPU-hours, and what caveats an honest
  estimate must state.

### 5. Positioning a benchmark plus law at a fusion venue

- What does Information Fusion's scope and recent published record
  suggest it rewards? Find its aims and scope, recent editorials, and
  representative recent benchmark or survey-plus-experiment papers.
- How do successful benchmark papers frame their contribution so that it
  is not dismissed as "just experiments"? Find concrete framing patterns
  from well-cited benchmark papers.
- Our organizing law is empirical, not theoretical. How have other
  empirical scaling-law or regularity papers defended that status, and
  what language did they use to state scope and limitations without
  weakening the contribution?
- What are the common reviewer objections to large benchmark papers, and
  what pre-emptive material best defuses them?

### 6. Language and typography conventions

- What is current best practice on **em dashes, forward slashes and
  arrows in running prose** in scientific English? I have removed em
  dashes and most slashes; confirm whether that matches accepted style, or
  whether I have over-corrected and damaged readability.
- Elsevier journals: is British or American spelling required, expected,
  or free, and must it merely be internally consistent?
- What are the conventions for first use of abbreviations in an abstract
  versus the body, and for abbreviations in keywords, in Elsevier
  journals?
- Any current guidance on the **generative AI disclosure statement**:
  exact required wording, placement, and what must be disclosed.

## Output format

Return a report with one section per numbered area above. In each:

1. A short verdict on what I have already done: keep, change, or drop.
2. Numbered, concrete recommendations, each tagged
   **[requirement]**, **[strong convention]** or **[taste]**.
3. The sources for each, with links.

Finish with a single prioritized list of the ten changes that would most
improve the manuscript's chance of acceptance, ordered by impact per unit
of effort. Be blunt. If something I have done is wrong or pointless, say
so directly and say what to do instead.
