# N16 top-ranked error and position diagnostics — 2026-10-11

**Two stages are preserved.** The initial diagnostics below used the standalone
export only. Authentic saved PRM rewards and canonical source groups were
subsequently supplied; `PRM_GROUP_EXTENSION.json` and
`SOURCE_GROUP_CONTRASTS.csv` extend the analysis with independently replayed
PRM scores and paired source-group confidence intervals. Their availability
supersedes the initial asset limitations concerning PRM and groups. Neither
stage fits a model or creates a held-out evaluation.

This runs follow-up tasks on the saved **prefix-corrected + mask1** N16 risk
scores. It does not fit a method, change an input, select a winner, or evaluate
a held-out source group. Presentation branch inspected: `53ef62e3`; research
inputs originate from `080877eb9a8369341d647f4115a3b475c040b46e`. Input,
canonical-script and diagnostic-script SHA-256 values are in `RESULTS.json`.

## Findings

- **Top1-on-error:** L-SML 59.85% (3,612 / 6,035), IU-PCR 59.52%, plain
  average 58.82%, DS-filter average 60.00%. These independently reproduce the
  existing corrected+mask1 table. Old replay values were 62.20%, 60.93%,
  61.13%, 61.13%, respectively; prefix correction accounts for most of the
  change. Historical PRM 57.78% and LR 57.90% are reported references only:
  their raw scores are absent from this export. These development figures
  do not establish a statistically significant advantage.
- **First-step selection is uncommon:** top1 is step 0 in 1.81% of all L-SML
  answers and 1.66% of IU-PCR answers, or 1.38% and 1.29% of answers with an
  actual in-range error. Step-0 selection by itself does not show a strong
  first-step dominance pattern in the already-masked export. This cannot
  settle whether masking is beneficial: no unmasked step scores are present.
- **Within-answer position-only risk** (later steps riskier, matching the
  pinned canonical scorer) has AUC 0.6617. The first/last-step rule is a weak
  localizer: top1-on-error is 18.91% when always selecting the last step.
- **Signal remains in a fixed coarse position diagnostic.** Split relative
  midpoint `(position + 0.5) / n_steps` into five equal bins. Within each
  answer compare errors with correct steps only when both are in the same
  bin, then average each eligible answer's AUC equally. Of 6,030 unrestricted
  AUC-eligible answers, 4,691 retain pairs (17,885 pairs total). The resulting
  AUC is 0.7462 L-SML, 0.7448 IU-PCR, 0.7492 plain average, 0.7489 DS-filter,
  and 0.5340 position-only. On these same eligible answers unrestricted AUCs
  are 0.7736, 0.7740, 0.7660, 0.7745 and 0.6558.
- **The simplicity signal survives this diagnostic.** Redundancy has
  same-bin AUC 0.8402 L-SML / 0.8285 IU-PCR, compared with 0.5246 position-only
  (675 / 758 eligible). Circularity has 0.7305 / 0.7398, compared with 0.5097
  position-only (518 / 758 eligible). The PRM cannot be evaluated under this
  restriction without its raw score array. Thus this checks the existence
  of signal beyond coarse position, not a controlled superiority claim.

The small overall ranking advantage of continuous fusion over plain averaging
is absent in this diagnostic; L-SML retains a redundancy advantage over plain
averaging (0.8402 vs 0.8234). These are descriptive observations, not grounds
for changing the frozen methods.

## Interpretation limits and integrity

Five bins were fixed in advance. They retain position differences inside each
bin; eligibility and error/correct pair composition change. The primary
quantity averages answers equally. The separately reported pooled-pair AUC
uses a different weighting and must not be substituted for it. This is not
proof that position confounding was removed.

Metadata contains 100 answers with at least one out-of-range error annotation.
The canonical original 1-based membership rule is retained: an annotation
outside the existing steps labels no step. Those annotations are counted in
`RESULTS.json`, never repaired. There are 6,035 answers with in-range errors,
6,030 with both classes, and 934 without an in-range error (including 758
synthetic correct controls).

Exact maximum and the canonical 8×machine-epsilon tie tolerance selected the
same step in every checked answer and method. All ties choose the earlier
step. Direct pairwise AUC matches an independent rank-sum calculation exactly;
`independent_check.py` additionally validates all counts and AUCs through
Python sorted-order/bisect arithmetic and integer bin assignments.

Authentic source-group IDs, raw PRM/LR scores, step token lengths and answer
text/repetition features are absent. **No source-group confidence interval,
held-out claim, token-length control or repetition control was computed.**
The dataset IDs are not assumed to encode source grouping.

## Outputs and reproduction

`SUMMARY.csv` contains full and position-restricted aggregate quantities.
`BY_TYPE.csv` adds error-type groups. `TOP1_STEP0_STRATA.csv` separates every
type by any-error/no-error and by the true step-0 label. `POSITION_BINS.csv`
contains per-bin eligibility and the two explicit aggregation schemes.
`HISTORICAL_TOP1.csv` separates recomputed methods from reported references.
`INDEPENDENT_CHECK.json` records the independent arithmetic validation.

Use the existing project interpreter; no dependencies were installed:

```sh
/Users/osegev/Desktop/hallucination_detection/.venv/bin/python \
  run_top1_position.py \
  --package-root /Users/osegev/Desktop/hallucination_detection/docs/reviews/thinkfix_rescore_review_20261011 \
  --pinned-run /Users/osegev/Desktop/hallucination_detection/docs/reviews/advisor_v10_followup_20261011/sources/dev_run_patched.py \
  --output-root .
/Users/osegev/Desktop/hallucination_detection/.venv/bin/python \
  independent_check.py \
  /Users/osegev/Desktop/hallucination_detection/docs/reviews/thinkfix_rescore_review_20261011 .
```

The pinned script is read only for protocol verification and hashed; none of
its imports or fitting/inference code is executed. On another machine, obtain
that exact Git blob before reproducing and check its recorded hash.
