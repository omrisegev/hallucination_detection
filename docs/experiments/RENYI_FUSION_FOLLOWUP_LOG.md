# Renyi / varentropy fusion follow-up experiment log

This is the short, append-only decision log for the ordered follow-up ladder.
Detailed protocols and machine-readable results live in their linked folders.
Every experiment stops for review before the next one begins.

## Experiment 1 — probability normalization, retained mass and preprocessing

- Date opened: 2026-09-14
- Status: **COMPLETE / REVIEW PASS**
- Question: did probability-head conditioning, discarded tail mass, or
  within-answer feature standardization cause the standalone-to-fusion loss?
- Fixed readout: Top10 token mean per official step.
- Fixed gate: mean entropy, training-fold quantile 0.3.
- Protocol: `PROBABILITY_NORMALIZATION_ABLATION_V1.md`
- Results: proper raw-head and conditional-head escort-varentropy agree to
  `8.73e-10`; four frozen score references replay to `7.05e-12`. `VE1 q50`
  gains .004678 PRMB within AUROC over q15 under the corrected interval while
  losing 0.285 PB points. A coarse tail bucket adds no useful signal. Removing
  within-answer centering restores roughly .04-.06 pooled/PRMScore without
  changing the registered within-answer ordering controls.
- Decision: `REJECT_P_TO_Q_AS_CAUSE`; promote `VE1 q50` as the support-width
  candidate; retain `VE.75 q15` for ProcessBench; do not add a scalar tail
  bucket; do not center every answer separately in calibrated fusion.
- Report: `../../results/probability_normalization_ablation_v1/REPORT.md`
- Frozen scores SHA256:
  `402af091e82a10147de659cac22befbd7f8fd3a5c8c3427082935368ab50daef`.

## Planned later experiments

1. Gate feature/readout optimization, beginning with Top10 mean.
2. Locator feature-bank expansion with H1, one top-1 view and retained mass.
3. Sparse position-varying simplex fusion.
4. One frozen new-model confirmation.

No later experiment is authorized by the execution of Experiment 1.

## Experiment 1B — benchmark-uniform support and weighting

- Date opened/completed: 2026-09-14
- Status: **COMPLETE / REVIEW PASS**
- Question: can one q15/q50/multiscale representation and one weighting rule be
  used unchanged across ProcessBench and PRMBench?
- Protocol: `UNIFORM_MULTISCALE_FUSION_V1.md`
- Uniformity: one feature definition and one shared cross-fitted PB+PRMB model;
  no cell-, model- or benchmark-specific weights or method selection.
- Results: the frozen worst-regret rule selects q15 raw equal after per-view
  Top10 (`PB=.366201`, `PRMB-within=.753436`). q50 raw gains only .001407
  within while losing .634 PB points; q15+q50 raw adds only .000312 within and
  loses .368 PB points. Global scaling removes useful natural scale ratios.
  The supervised simplex zeroes both low-alpha q15 views and collapses the
  eight-view bank to q50 `VE.75/VE1`, then loses on the localization endpoints.
  Centering leaves PB/within invariant but costs .006-.012 fold-pooled AUROC and
  .0075-.0096 PRMScore.
- Decision: `SELECT_Q15_RAW_EQUAL_UNIFORMLY_ON_DEVELOPMENT`; this supersedes
  Experiment 1's benchmark-specific wording about retaining q15 for PB and q50
  for PRMB. Do not add static q15+q50 duplication; retain natural scale ratios;
  reject this fixed global step-BCE simplex; preserve answer-level means.
- Report: `../../results/uniform_multiscale_fusion_v1/REPORT.md`
- Frozen OOF scores SHA256:
  `2e87a3d11f4ee77594b64775457b1e99e864efb7d5eaca8350f9688b4adc14f4`.

## Updated continuation after Experiment 1B

1. Gate feature/readout optimization, beginning with Top10 mean and the q15
   locator held fixed.
2. Locator expansion with H1 and one top-1 view only under a separately frozen
   uniform protocol; do not reopen q50 or a scalar tail bucket by default.
3. Sparse position-varying fusion only with an objective aligned to the actual
   localization endpoints, not the rejected global step-BCE surrogate.
4. One frozen new-model confirmation.

## Finalist replay after Experiment 1B

- Date opened/completed: 2026-09-14
- Status: **COMPLETE / REVIEW PASS**
- Question: what does the one deployable specification selected by Experiments
  1/1B score when replayed by itself against the original static and
  position-temporal starting points?
- Protocol: `SELECTED_Q15_FINALIST_REPLAY_V1.md`
- Frozen finalist: q15 `{H0lim, VE0, VE0.75, VE1}`, label-free orientation,
  per-view Top10, raw equal step fusion, no centering/scaling, no supervised
  weights, no q50 duplication, and one method for every benchmark. The entropy
  q=.3 gate and PRMScore q=.8 calibration remain unchanged.
- Results: PB all-8 36.6201%, PRMB within .753436, fold-pooled .722708,
  PRMScore .634412. Versus the original fusion-before-Top10, PB is +.453pp and
  within is +.002321 (family-wise 98.333% CI [.001322,.003367]); PRMScore is
  essentially unchanged (-.000392). Versus the original local shrinkage +
  position method, PB is +.701pp, within +.006055
  [.003317,.008850], and PRMScore +.047003.
- Decision: `FREEZE_Q15_RAW_PER_VIEW_TOP10_AS_CURRENT_DEVELOPMENT_FINALIST`.
  The tested position-varying standardized fusion remains mechanism evidence,
  not the current winner. The next separate experiment may change the gate;
  this replay does not.
- Report: `../../results/selected_q15_finalist_replay_v1/REPORT.md`
- Frozen score SHA256:
  `3bd5c5b95474d75366b97b012b018d168cacafb3ccea178d26170207e220c7e6`.

## Experiment 2 — gate feature/readout selection

- Date opened/completed: 2026-09-14
- Status: **COMPLETE / REVIEW PASS**
- Fixed components: q15 raw per-view-Top10 locator and other-fold q=.3 gate
  quantile rule.
- Search: 11 label-free token signals x 3 whole-answer readouts, one definition
  across all eight PB cells.
- Result: `tail15_mass__token_mean` leads at PB 36.8818%, versus 36.6107% for
  recalculated entropy mean. Its mean detector AUC is .792172 versus .742301.
  H1 exactly matches native entropy; Hinf and raw `-log p1` are weaker. Contrary
  to the locator, gate Top10 readouts lose at the fixed q=.3 operating point.
  Selected-minus-baseline is +.271pp with post-selection descriptive 95% CI
  [-1.079,+1.594].
- Decision: `SELECT_TAIL15_TOKEN_MEAN_AS_NEXT_GATE_CANDIDATE_ON_DEVELOPMENT`;
  do not call the F1 gain confirmed.
- Report: `../../results/gate_feature_readout_selection_v1/REPORT.md`
- Frozen detectors SHA256:
  `e3ccc87503df2629834348ab0e65727f8a43ccefd784eccf76588a21ce99a36a`.

## Integration checkpoint 2 — q15 finalist plus selected gate

- Date opened/completed: 2026-09-14
- Status: **COMPLETE / REVIEW PASS**
- Purpose: same-benchmark cumulative composition check requested by Omri; no
  independent holdout claim.
- Result: original static + entropy 36.1674%; q15 finalist + entropy 36.6201%;
  original static + tail15 36.5937%; integrated q15 + tail15 36.8818%.
  Thus the selected gate adds +.262pp over the current finalist and all selected
  decisions add +.714pp over the original. Family-wise 98.75% intervals for
  both comparisons cross zero.
- Decision: `RETAIN_TAIL15_MEAN_AS_NEXT_GATE_CANDIDATE; DO_NOT_YET_REPLACE_ENTROPY`.
  The next bounded question is the q=.3 operating point; after any threshold
  choice, rerun the complete integrated algorithm again.
- Report: `../../results/integrated_q15_tail15_gate_replay_v1/REPORT.md`

## Experiment 2B — full math-panel answer-gate development

- Date opened/completed: 2026-09-14
- Status: **COMPLETE / REVIEW PASS; NUMERICAL WINNER NOT PROMOTED**
- Population: all 18,614 answers in the 15 historical GSM8K/MATH500 cells.
- Search actually run: 11 token signals x 11 whole-answer readouts = 121
  singles; near-duplicate removal; forward equal-mean fusion; nonnegative
  epsilon-pruned simplex; and sparse logistic fusion. One q from `.05-.95` was
  selected uniformly by equal-family clean/error macro-F1.
- Numerical result: the three-feature equal mean (`VE1 Top10`, q15 raw4 final
  quarter, entropy Top10) scores F1 .641310 / AUROC .803491 at q=.45, versus
  .633562 / .786633 for the best single (`VE1 Top10`).
- Decision revision: do not promote the three-feature arm. The +.775pp math F1
  gain does not justify a new feature-selection/fusion layer. Its frozen PB
  transfer and PB q=.40 integration replay remain valid diagnostics, but the
  q=.40 candidate is superseded and must not be described as preferred.
- Protocol: `MATH_GATE_DEVELOPMENT_V1.md`.
- Machine-readable results: `../../results/math_gate_development_v1/`.

## Experiment 2C — simplified two-arm gate choice

- Date opened/completed: 2026-09-14
- Status: **COMPLETE / REVIEW PASS**
- Candidates only: native H1 entropy Top10; exact frozen q15 static token fusion
  followed by one whole-answer Top10.
- Exactness: the latter reconstructs frozen
  `original_static_fusion_before_top10` after the registered per-step Top10
  readout with maximum discrepancy zero. It is not the selected
  per-view-Top10 locator, where Top10 precedes fusion.
- Math result: entropy Top10 F1 .633271 / AUROC .791377 / AUPRC .775099;
  token-fusion Top10 .632755 / .787228 / .759507. Both select q=.45.
- Decision: `SELECT_ENTROPY_TOP10_Q45_AS_SIMPLE_TOTAL_ANSWER_CANDIDATE`.
  The three-feature gate is not promoted.
- Protocol: `SIMPLE_GATE_CHOICE_V1.md`.

## Frozen transfer checkpoint 2C — entropy Top10 q=.45

- Date completed: 2026-09-14
- Status: **COMPLETE / REVIEW PASS**
- Transfer: math-selected method and q applied unchanged to all eight PB cells;
  no PB method, fusion or q selection. The within-cell mid-rank transform is
  label-free.
- Answer detection: family-macro F1 .693567 / AUROC .792620, versus .649999 /
  .742301 for existing entropy mean q=.3.
- Localization with the same frozen q15 locator: 35.5339% versus 36.6201% for
  existing entropy mean q=.3; delta -1.086pp, conservative 98.75% paired CI
  [-2.946,+.769]pp.
- Error trade-off: 698 clean false alarms removed and 151 added, but 797 errors
  newly closed and 328 reopened; 85 exact localizations gained and 338 lost.
- Decision: retain entropy Top10 q=.45 for total-answer detection development,
  but retain mean entropy q=.3 when gating the current localization pipeline.
  The answer-level objective and exact-localization objective are not aligned
  at the transferred operating point.
- Report: `../../results/simple_gate_choice_v1/REPORT.md`.

## Experiment 2D — leading distinct simple gates as gates

- Date opened/completed: 2026-09-14
- Status: **COMPLETE / REVIEW PASS**
- Question: did simplifying 2C accidentally return to the initial entropy
  family, and do the other leading math singles behave better when used as
  actual gates for the frozen q15 locator?
- Frozen candidates: VE1 Top10 q=.45, entropy/H1 Top10 q=.45, Hinf Top10
  q=.45, audited q15 static token-fusion Top10 q=.45, and missing-tail15 mass
  Top10 q=.40. H1 and raw top-1 duplicates were not repeated.
- Transfer contract: every q came from the 15-cell math panel; no PB feature,
  readout, fusion or q calibration; one definition across all eight PB cells.
- Winner on both displayed PB objectives: tail15 Top10, answer family-macro F1
  .697932 / AUROC .799571 and localization 36.6736%. Existing entropy mean q=.3
  is .649999 / .742301 and 36.6201%. Localization delta +.054pp, family-wise
  99% CI [-1.632,+1.804]pp.
- Other PB localizations: q15 raw4 fusion 35.5691%, entropy 35.5339%, Hinf
  35.0077%, VE1 34.9234%. All improve answer detection over the baseline, so
  answer-level F1 alone does not choose the correct localization gate.
- Decision: `RETAIN_TAIL15_TOP10_Q40_AS_NEXT_DISTINCT_SIMPLE_CANDIDATE`;
  do not call the localization improvement confirmed. Historical tail15 mean
  q=.3 at 36.8818% remains a PB-developed diagnostic rather than this frozen
  math-to-PB transfer.
- Protocol: `LEADING_GATE_TRANSFER_V1.md`.
- Report: `../../results/leading_gate_transfer_v1/REPORT.md`.

## Experiment 2E — tail15 Top10 versus mean under one protocol

- Date opened/completed: 2026-09-14
- Status: **COMPLETE / REVIEW PASS**
- Frozen change: identical raw missing-top15-mass signal and frozen q15
  locator; only the answer readout changes. Top10 uses math-selected q=.40 and
  mean uses math-selected q=.45. No PB readout or q calibration.
- Math answer F1: Top10 .632415 versus mean .625927.
- PB answer detection: Top10 F1 .697932 / AUROC .799571; mean .691500 /
  .792172.
- PB localization: Top10 36.6736%; mean 36.6064%; Top10-minus-mean +.067pp,
  family-wise 98.333% paired CI [-.984,+1.149]pp.
- Trade-off: Top10 obtains 1,058 exact error localizations with 751 clean false
  alarms; mean obtains 1,016 with 642.
- Decision: `RETAIN_TAIL15_TOP10_AS_READOUT_CANDIDATE`; the difference is not
  confirmed. The historical tail15-mean q=.3 PB-developed row at 36.8818%
  exceeds both, pointing to the operating-point objective as the next question.
- Next bounded experiment: select a localization-cost-aware q for frozen
  tail15 Top10 on development and immediately replay the complete integrated
  method; external confirmation still follows only after that specification is
  frozen.
- Protocol: `TAIL15_READOUT_HEADTOHEAD_V1.md`.
- Report: `../../results/tail15_readout_headtohead_v1/REPORT.md`.

## Experiment 2F — localization-aware q and cumulative freeze

- Date opened/completed: 2026-09-14
- Status: **COMPLETE / REVIEW PASS — DEVELOPMENT FROZEN**
- Fixed components: q15 per-view-Top10 natural-unit locator; raw missing-top15
  mass token signal; whole-answer Top10 readout; within-cell label-free midrank
  percentile.
- Search: one uniform q over `.01-.99` for all eight ProcessBench cells,
  selected by official all-eight exact-localization macro-F1. No feature,
  readout, locator or per-benchmark specialization was permitted.
- Selection: q=.33, PB 37.4749%. The nearby q=.31-.35 values range only from
  37.3969% to 37.4749%, so the exact hundredth is treated as a development
  freeze rather than a stable external optimum.
- Cumulative comparison: starting static locator + entropy mean q=.3 is
  36.1674%; locator-only update 36.6201%; gate-only update 36.8759%; complete
  q15 + tail15 Top10 q=.33 is 37.4749%. Final-minus-start is +1.307pp with
  family-wise 99% grouped CI [-.484,+3.067]pp.
- Answer detection: final family-macro F1 .702180 / AUROC .799571 / AUPRC
  .863489, versus entropy-mean .649999 / .742301 / .831461.
- PRMB locator: within .753436 / fold AUROC .722708 / pooled OOF .722305 /
  PRMScore .634412. The gate is not applied to PRMB.
- Interpretation: this completes the requested same-development integration,
  not external generalization. The frozen method requires new-model/data
  confirmation.
- Protocol: `TAIL15_LOCALIZATION_Q_V1.md`.
- Report: `../../results/tail15_localization_q_v1/REPORT.md`.
- Original-plan audit: `../../results/tail15_localization_q_v1/PLAN_AUDIT.md`.
