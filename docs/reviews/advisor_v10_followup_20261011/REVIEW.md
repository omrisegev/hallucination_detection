# Advisor v10 review and saved-score follow-ups — 2026-10-11

Reviewed branch `claude/advisor-page-v6-official-prmscore`, frozen at
`53ef62e36e80b90de30b83d3fee33e1fb85b82c3` (remote head checked again after the runs).
Research exports remain frozen at `080877eb9a8369341d647f4115a3b475c040b46e`.
The three requested files were read: `CODEX_NOTES_2026-10-11.md`,
`advisor_review_v10.html`, and `add_advisor_views.py`; their metric sources and
the linked `PAGE_REVIEW_TASKS_2026-10-11.md` at `8996379` were also inspected.

## Assessment

Claude's additions improve the presentation: retain the bank/method comparison,
the PRMBench-style leaderboard and radar, and the restored emphasis on
Continuous L-SML and IU-PCR. The relevant research story is a complementary
Simplicity signal and competitive development performance, rather than a claim
that the newly assembled pipelines outperform everything else.

Two corrections matter before presenting v10:

1. **My earlier MS/Sensitivity calculation was wrong.** The official PRMBench
   table uses valid-step F1 alone for multi-solution consistency (MS), and averages
   PS, DR and MS for Sensitivity. Full Sensitivity is **69.34 L-SML / 69.54 IU-PCR /
   75.51 PRM-7B**. The earlier ~54.7 calculation incorrectly halved MS; ~59.8 is a
   different PS+DR-only diagnostic. Use the full convention throughout the main
   presentation and name any partial diagnostic explicitly. Primary reference:
   [official table generator](https://github.com/ssmisya/PRMBench/blob/561ab8aa7fb4104df6906e54231697ddd78a390f/mr_visualize/eval_res_view/draw_tabs/form_latex_strs.ipynb).
2. **“Decision is the bottleneck” is a hypothesis, not an established result.**
   The new fair threshold comparison below does not support solving the project
   by changing the global threshold alone. Clean-answer discrimination, error
   count allocation and ranking all remain relevant.

## What the new CPU runs establish

All runs use saved scores: **6,969 answers / 94,203 steps**. Pooled PRMScore is
macro-F1 from aggregate confusion counts on **6,211 noncontrol answers /
83,371 steps**. Synthetic controls are excluded from that pooled score but
included in the original global threshold population and reported separately.
Out-of-range annotations in 100 answers retain the canonical original membership
semantics; no labels were repaired.

### Frozen decision rules and uncertainty

The independently replayed N16 Q80 scores are **65.381 L-SML**, **65.359 IU-PCR**,
**64.741 plain average**, **65.252 DS-filter average**. PRM native 0.5 is **65.457**.
All reproduce the saved reference to within 1e-12. The PRM reward is a validity
score; error flags use **reward < 0.5**. Its 40 exact midpoint rewards remain valid.

Using **707 authenticated canonical source groups**, 5,000 paired whole-group
resamples give these pointwise 95% intervals, in PRMScore percentage points:

- L-SML minus plain: **+0.639 [+0.390, +0.892]**.
- IU-PCR minus plain: **+0.618 [+0.382, +0.849]**.
- L-SML minus DS-filter: **+0.129 [-0.060, +0.324]**.
- L-SML minus native PRM: **-0.076 [-0.767, +0.636]**.
- IU-PCR minus native PRM: **-0.098 [-0.789, +0.611]**.

The continuous methods improve over plain averaging in this development
comparison. Their difference from native PRM is unresolved; an interval spanning
zero is not evidence of equivalence. These intervals account for source
dependence, not prior method selection or multiple comparisons. They are not
untouched confirmation results. Each resample recomputes pooled F1 from confusion
counts; it does not average per-answer F1.

### Three different oracle questions

- **Best global threshold, actual ranking:** L-SML **65.690** (+0.309 points),
  IU-PCR **65.550** (+0.191). Applying the same oracle search to PRM gives
  **68.314**, versus its native **65.457** (+2.857). Thus changing our global
  operating point alone does not establish an advantage over a similarly
  optimized comparator. All searches include every tie-preserving cut and the
  reject-nothing boundary. No threshold is selected for deployment.
- **True error count per answer, actual ranking:** L-SML **71.730**, IU-PCR
  **71.675**. This supplies unavailable label information, including zero errors
  on clean answers; it is a diagnostic of count allocation, not a proposed method.
- **Perfect error-first ranking, original Q80 count per answer:** L-SML
  **81.822**, IU-PCR **81.876**. Considerable ranking headroom also remains.
  These ceilings change different information and are not additive effects or
  a causal decomposition of the bottleneck.

![Paired threshold diagnosis](paired_threshold_curves.png)

The true step error prevalence is **13.96% including controls**, and **15.77%
excluding controls**. Neither number proves that flagging the same proportion
would optimize macro-F1. `ANSWER_Z_Q80` uses one global quantile over answer-z
scores; it does **not** impose a 20% quota on each answer.

The saved frozen-G R2 reconstruction keeps the global flag budget and improves
synthetic-control false-flagging from **99.47% / 99.34% to 58.58%**. However,
pooled PRMScore falls slightly to **65.332 / 65.296**. This is a concrete tradeoff,
not an overall improvement. The frozen-G reconstruction matches the exported
R2 arithmetic but is **not a certified replay of the separately registered
`R2_COUNT` input contract**. Do not relabel it as such.

### Ranking, top1 and position

PRM within-answer AUC is **0.801180**, above **0.779193 L-SML / 0.780459 IU-PCR**.
Top1-on-error supports a different comparison: **59.85% / 59.52% versus 57.78%
PRM**, on 6,035 answers with at least one in-range error. Group-paired top1
differences are **+2.071 points [0.478, 3.654]** and **+1.740 [0.133, 3.346]**.
Present both metrics; do not summarize them as a universal winner.

Five fixed coarse relative-position bins retain **4,691 answers / 17,885
error-correct pairs**. Equal-answer same-bin AUC is **0.7462 L-SML / 0.7448
IU-PCR / 0.7361 PRM / 0.7492 plain / 0.7489 DS-filter / 0.5340 position-only**.
The overall continuous-minus-PRM intervals cross zero, and plain/DS averaging
remain competitive. The figure compares unrestricted and restricted AUC on the
same retained answers, avoiding a change of population between endpoints.

![Same eligible population position diagnosis](top1_position/SAME_ELIGIBLE_POSITION.png)

The interesting evidence is subtype-specific. Redundancy same-bin AUC is
**0.8402 L-SML / 0.8285 IU-PCR / 0.6242 PRM** on 675 answers; circularity is
**0.7305 / 0.7398 / 0.6920** on 518 answers. The group-paired differences are
positive for both methods in both subtypes. This supports further investigation
of a complementary simplicity signal. Coarse bins leave residual position
differences; text repetition and token-length controls are still needed.

The corrected LR cache supports a concrete disagreement inventory: on 6,035
error-containing answers, L-SML alone selects an error in **489** cases, LR alone
in **371**, both in **3,123**, neither in **2,052**. IU-PCR corresponding counts
are **461 / 363 / 3,131 / 2,080**. IDs and selected steps are saved for future
case inspection; this is not a new fitted combination. The cache lacks full LR
step scores. Its observed hash is pinned and ordering was validated against three
independent frozen method arrays, but no historical expected cache hash was found.

### Masking versus prefix correction

The old-to-new decline mostly occurs with prefix correction. For N16 L-SML,
reported top1 is **62.20 old → 59.98 prefix-corrected → 59.85 with mask1**;
IU-PCR is **60.93 → 59.30 → 59.52**. AUC is **0.7958 → 0.7799 → 0.7792** and
**0.7924 → 0.7806 → 0.7805**, respectively. Thus mask1 is slightly negative on
these AUCs and L-SML top1, slightly positive on IU-PCR top1. This is not evidence
that masking generally helps, nor that it explains the full drop. The rationale
is to suppress a suspected special-token artifact; it should be assessed with
a matched, label-free ablation. The currently available flat arrays are already
masked and cannot independently settle that ablation. After masking, top1 picks
step 0 in only **1.81% / 1.66%** of all L-SML/IU-PCR answers.

## Recommended v10 changes

- **Keep the new views reproducible.** `add_advisor_views.py:150–159` patches
  literal HTML anchors and rejects an already patched input. The original
  `build_presentation.py` does not include these additions. Integrate them into
  the canonical builder, or specify and verify one complete build pipeline.
- **Unify metric text and sources.** The new leaderboard uses full Sensitivity,
  but the older graph and `official_prmscore_v1.py` category export still omit MS
  from the mean. Fix both; also fix “20% of every answer” in prose/docstrings.
- **Label published references properly.** Paper rows are reported literature
  values, not reruns. Show the o1-mini evaluation size of 394 next to that row.
  Preserve published category means instead of recomputing them from rounded
  subcategory scores: several reconstructed values differ by 0.1 points.
- **Add the existing negative external control.** On Hard2Verify, step index
  alone reaches 0.483 Balanced F1, above the best panel cell 0.433 and locked
  L-SML 0.421. Socratic is different: index 0.580 versus locked L-SML 0.628.
  These task-specific Balanced-F1 values must not be mixed with PRMBench AUC or
  pooled PRMScore. The current generalization narrative needs this qualification.
- **Use a short central story.** Lead with Continuous L-SML and IU-PCR; show
  overall score plus top1, then the simplicity comparison and the main failure
  modes. Keep a compact leaderboard/radar, and move full bank-by-method matrices
  to the appendix. Nested bank means are descriptive, not independent replicates.
- **Fix provenance/footer.** The current footer still calls v10 a local
  replacement with the published artifact unchanged. Record the actual builder,
  data and publication version separately.

Earlier suggestions that remain relevant: paired source-group uncertainty
(now run), valid/error F1 decomposition (**0.8765 / 0.4311** L-SML versus
**0.9156 / 0.3935** PRM), worked redundancy/circularity examples, position/length/
repetition controls, and the external step-index comparison. The proposed MoM
prevalence direction remains research, not a validated remedy. Global prevalence
alone cannot identify clean answers; learned groups do not guarantee conditional
independence, and zero pairwise correlation does not establish it. Test both
channel and grouped-vote conditional dependence before using estimated prevalence
to set quotas.

The earlier research-code findings also remain relevant. GitHub blob checks on
the v10 commit confirm that the restricted-LR weights script, nested-bank runner
and self-generation evaluator are unchanged from the original audited research
commit. Restricted LR subsets still need feature selection inside each outer
training fold; the entropy gate still needs corrected answer-level inputs; the
self-generation evaluator's “small point difference plus CI containing zero”
must not be presented as an equivalence test. These do not invalidate the frozen
N16 score arithmetic or the separate full-LR reference. Details and concrete
fixtures are in the [original research audit](../thinkfix_rescore_review_20261011/REVIEW.md#findings-requiring-action).
Its old MS/Sensitivity objection is superseded by the correction above.

## Completed work, remaining work and reproduction

Completed locally: exact Q80/frozen-R2 replay, scalar threshold curves with a fair
PRM oracle, per-answer/type count diagnostics, top1/position diagnostics, authentic
source-group paired intervals, and LR top-step disagreement. Inputs were retrieved
read-only from size-inspected Drive assets; hashes and identity joins were checked.
The original v10 page, research sources and unrelated dirty checkout are unchanged.
This follow-up package is local and uncommitted; it has not been published to
GitHub or Claude.

Still pending: integrate the presentation fixes; replay exact registered
`R2_COUNT` and `ENTROPY_GATE_Q80`; corrected channel/type and conditional group-vote
analyses; repetition/token-length controls and worked text examples; safe feature
selection and frozen confirmation. Corrected FEATURES is 53,985,301 bytes and was
located with a tracked hash, but not needed/downloaded for these runs. A split
over these already inspected scores can support internal validation only; fitting
or normalization on the full population cannot be called untouched transfer.

Run with the existing repository environment, without package installation:

```sh
MPLCONFIGDIR=/private/tmp/advisor-v10-mpl .venv/bin/python -B docs/reviews/advisor_v10_followup_20261011/decision_analysis.py
MPLCONFIGDIR=/private/tmp/advisor-v10-mpl .venv/bin/python -B docs/reviews/advisor_v10_followup_20261011/paired_decision_analysis.py
MPLCONFIGDIR=/private/tmp/advisor-v10-mpl .venv/bin/python -B docs/reviews/advisor_v10_followup_20261011/lr_disagreement.py
```

Top1/position reproduction and independent arithmetic verification are documented
in `top1_position/README.md` and `top1_position/PRM_GROUP_ADDENDUM.md`. Use the
durable `sources/dev_run_patched.py` when reproducing the initial protocol check.
Source download provenance is archived in `sources/README.md`; its temporary
discovery paths are historical, while analysis inputs are preserved here.

Independent final review completed after the work. No major arithmetic or
interpretation defect was found. Two minor provenance/reproduction issues were
corrected and the affected extension regenerated. See
[independent review receipt](INDEPENDENT_REVIEW.md) for the exact checks and limits.
