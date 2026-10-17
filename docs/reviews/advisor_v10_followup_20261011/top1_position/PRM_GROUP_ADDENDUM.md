# Saved PRM and authentic source-group extension

Authentic PRM rewards and canonical source groups became available after the
initial standalone-export diagnostics. This addendum supersedes the original
limitations concerning those two assets. Raw LR scores, token lengths and
repetition features are still outside this analysis. No model was fitted and
no evaluation population was held out from development.

## Main observations

PRM's independently replayed overall within-answer AUC is **0.801180**, higher
than L-SML's **0.779193** and IU-PCR's **0.780459**. However, selecting each
answer's single riskiest step gives **57.78%** top1-on-error for PRM, versus
**59.85%** L-SML and **59.52%** IU-PCR. On this development population the paired
source-group intervals for top1 differences are +0.48 to +3.65 percentage
points for L-SML and +0.13 to +3.35 for IU-PCR. AUC ranking and top1 localization
therefore support different comparisons. These intervals are descriptive;
they do not correct development selection or the multiple reported contrasts.

The position figure uses the **same 4,691 answers on both ends**, rather than
comparing an unrestricted mean from 6,030 answers with a restricted mean from
4,691. On these retained answers PRM drops from unrestricted AUC **0.793161**
to same-bin AUC **0.736105**. L-SML goes from **0.773633** to **0.746166**;
IU-PCR from **0.773980** to **0.744826**. The overall same-bin differences versus
PRM have 95% intervals crossing zero: L-SML **[-0.00155, +0.02199]**; IU-PCR
**[-0.00295, +0.02057]**. There is no overall superiority result under this
restriction. Plain averaging has same-bin AUC **0.749183**.

The more informative evidence remains **Simplicity**:

- Redundancy: same-bin AUC **0.840228** L-SML, **0.828520** IU-PCR, **0.624247**
  PRM, on **675 answers**. The paired L-SML-minus-PRM interval is
  **[+0.18734, +0.24484]**, IU-PCR-minus-PRM **[+0.17461, +0.23303]**.
- Circularity: **0.730513 / 0.739789 / 0.692017**, on **518 answers**. The paired
  L-SML-minus-PRM interval is **[+0.00564, +0.07201]**, IU-PCR-minus-PRM
  **[+0.01545, +0.08026]**.

This strengthens the case for investigating a complementary simplicity
signal. It does not establish complete position-confound removal. Relative
position bins are coarse, remaining within-bin differences persist, and the
pair composition and eligible population differ from the original metric.
Repetition and token-length controls, frozen confirmation and an independent
population remain required.

## Bootstrap contract

Use the supplied `question_source_components_v2` mapping exactly; do not
extract groups from row-ID strings. All 6,969 exported rows align to **707
canonical groups**. Each of 5,000 draws with seed **20261011** samples 707
whole groups with replacement. Every answer inherits its group's multiplicity.
Numerator and denominator are recomputed for each metric's eligible answers,
so the original metric remains **equal-answer weighted**, not equal-group
weighted. Groups lacking eligible rows contribute zero, and all bootstrap
denominators were positive.

Five contrasts were fixed before the extension was run: L-SML/IU-PCR versus
PRM, L-SML/IU-PCR versus plain averaging, and L-SML versus DS-filter averaging.
These are reported for within-answer AUC, top1-on-error and same-bin AUC over
all answers, redundancy and circularity: **45 paired percentile 95% intervals**.
No correction for multiple comparisons is applied. The reused group draws
are identified by their SHA-256 checksum and lexicographic group order.

## Verification and outputs

PRM rewards have higher values for safer steps; risk is `1 - reward`.
Every row's identity, step count, type/category and error annotation matches
the score export. All saved native labels exactly equal `reward >= 0.5`,
including **40 exact midpoint ties**. Native threshold labels do not determine
the AUC or top1 metrics here. PRM AUC and top1 exactly reproduce the existing
reported references. Previously saved diagnostics reproduce unchanged.

All pairwise AUC calculations are checked independently against rank-sum
arithmetic. For every one of the 45 contrasts, the first bootstrap draw is
also checked by explicitly repeating answer rows according to whole-group
multiplicity, independently of the grouped numerator/denominator calculation.
Input SHA-256 hashes are unchanged. Full checks and provenance are in
`PRM_GROUP_EXTENSION.json`.

`PRM_GROUP_AGGREGATES.csv` contains metric means and answer/group counts.
`SOURCE_GROUP_CONTRASTS.csv` contains all 45 paired intervals.
`SAME_ELIGIBLE_FIGURE_DATA.csv` stores the plotted values.
`SAME_ELIGIBLE_POSITION.svg` and `.png` are the standalone scientific figure.

Run from this directory with existing dependencies; the extra inputs default
to the durable sibling `../sources` directory:

```sh
MPLCONFIGDIR=/private/tmp/thinkfix-mpl-cache \
/Users/osegev/Desktop/hallucination_detection/.venv/bin/python \
  extend_prm_groups.py \
  --package-root /Users/osegev/Desktop/hallucination_detection/docs/reviews/thinkfix_rescore_review_20261011 \
  --output-root .
```
