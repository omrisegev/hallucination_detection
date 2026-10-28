# Pre-registration: between-group estimators (HEM, spectral start) and the next-step PRM view (CPU, 2026-10-28)

Written and committed before the first run. Development data; labels only score.

## Why

Omri, 2026-10-26/28:
- Of DS-EM (Dawid–Skene 1979) and HEM (Jaffe, Fetaya and Nadler 2016), use the more recent one: HEM.
- Try the SML eigenvector as the EM starting point (spectral initialization, as in Zhang, Chen, Zhou and Jordan 2014).
- Also try plain SML between groups.
- Test the next-step PRM reward, found by the PRM-features session
  (`docs/research_notes/PRM_MODEL_NEW_FEATURES_2026-10-26.md`, branch `claude/prm-new-features-2026-10-26`) to be
  nearly uncorrelated with the PRM and N16 on Socratic Qwen-7B (r .16, R² .11), as its own CHAIN group.

## Cells

PRMBench, Socratic-PRMBench and Hard2Verify × qwen_prm_7b, qwen_prm_72b, pure_prm, rlhflow_prm: 12 cells.
Math-Shepherd is out of the line (decision 2026-10-26) and is not run.

Hard2Verify populations:
- Part A uses the 177-answer common set for both fitting and scoring, as in `h2v_holdout/`.
- Part B uses CHAIN's exported recipe: fit on all loaded answers, score on the 177 (`prm_boost_math_rigor_v1/common.py`).

## Part A: between-group estimators in the binary L-SML pipelines

**Sets.** Frozen, no re-selection:
- SUB5 = q15_H1 + chosen_surprisal + top15_turnover + dominant_freq16 + bocpd_p0 + digit_spread;
- N16 = all 16 channels.

**Versions.**
- "+PRM": stage 1's `CellData` (PRM-anchored orientation, GAP votes).
- "alone": stage 2's `TeleCell` (stored orientation, no PRM).

**Common steps.** Groups come from Algorithm 1 on the vote covariance (K ≥ 3, else invalid). Scoring is continuous:
the weights are applied to answer-z values, then GAP and the official metric.

**Arms.** Only the estimator changes.

| Arm | Within groups | Between groups | Start |
|---|---|---|---|
| E0 `DS_MAJ` (reference; = stages 1–2) | SML eigenvector | two-coin DS-EM (`ds_em_votes`) | soft majority |
| E1 `DS_SPEC` | SML eigenvector | two-coin DS-EM | spectral |
| E2 `HEM_SPEC` (**primary**, Omri's choice) | HEM emissions | HEM | spectral, plus the 4 fixed extra starts of `fit_em`; best likelihood |
| E3 `HEM_MAJ` (= stage 1's secondary) | HEM emissions | HEM | majority, plus the same extra starts |
| E4 `SML_ONLY` | SML eigenvector | SML eigenvector of the vote covariance of [PRM vote, group votes], non-negative | none (no EM) |

**The spectral start.**
- v is the leading eigenvector of the zero-diagonal covariance of all input votes on the fitting pool, with its sign
  set so that sum(v) > 0.
- s = votes · v, then q₀ = expit(s / sd(s)), clipped to [.05, .95] for DS-EM.
- In `fit_em`, the spectral object only replaces the majority start.

**Label switching.**
- +PRM: the PRM anchors it (as now).
- Alone: the majority rule (DS-EM, as stage 2). For HEM, flip all between-group weights if fewer than half of the
  groups have a positive weight.

**HEM weights.**
- Linearized and non-negative, exactly as `stage1/analyze.py` `hem_weights`.
- The +PRM PRM column is its own group.
- The alone version uses the same function with the anchor replaced by the majority switch.

**Parity before reading results.**
- E0 must reproduce `stage1/SEARCH.npz` (+PRM) and `stage2/SEARCH.npz` (alone) `ps_fused` for SUB5 and N16 on the
  8 PRMBench/Socratic cells, within 1e-9.
- E0 must reproduce SUB5 in `h2v_holdout/H2V_SUB4_SUB5.json` on Hard2Verify.
- Any miss stops the run.

## Part B: the next-step PRM view in CHAIN_N16

**View.** For step k of an answer, next = the PRM risk z(logit(1 − reward)) at step k+1 of the same answer, over
scored (non-empty) steps.
- The last step has no next step. It gets the answer's mean of the other next values, which is 0 after answer-z.
- Orientation is fixed in advance: higher next-step risk means step k more likely wrong. The sign of its correlation
  with z(PRM) is audited and reported, but never changed.

**Arms.**
- `CHAIN_N16`: the exported recipe, recomputed. It must equal STATUS_MAIN's GAP value in all 12 cells within 1e-6;
  otherwise stop.
- `CHAIN_N16_NEXT`: the same telemetry groups. The cross stage is L-SML over [z(PRM), group scores…, z(next)], with
  every column its own group (`lsml_continuous_guarded`). The anchor flip is on z(PRM).

**Reported per cell:**
- the next-step view's cross weight share;
- its correlation with z(PRM) and with relative position.

## Endpoints

- **Primary:** the official metric under GAP (PRMBench PRMScore, `E.official`; external 0–100).
- **Secondary:** within-answer AUC.
- **Paired bootstrap:** 2,000 draws, seed 20261028; PRMBench resampled by source group, external sets by answer.
- **Contrasts:**
  - Part A: each arm − E0, in each set, version and cell.
  - Part B: CHAIN_N16_NEXT − CHAIN_N16, and CHAIN_N16_NEXT − PRM (STATUS "PRM same rule").
- **Context only, no contrast:** the PRM, CHAIN_N16, TELE_N16 and each arm's own plain mean.

## Reading rule (fixed now)

- **Gains and losses.** An arm "gains" in a cell only when its paired 95% interval excludes 0. Report counts of
  gains, losses and ties out of 12 per set and version.
- **E2 is preferred over E0** if it gains in at least 4 of the 24 (set × version × cell) comparisons and loses in
  none. Otherwise the reading is "no measurable difference" or "worse", as the counts say.
- **The next-step view helps** if CHAIN_N16_NEXT − CHAIN_N16 gains in at least 4 of 12 cells and loses in none.
- **Socratic** is read separately: does the view reduce the loss to strong PRMs (Qwen-7B, Qwen-72B, PURE)?
- **Nothing is tuned on these outputs.**

## Limits stated in advance

- All three benchmarks were inspected before, and SUB5 was selected on PRMBench and Socratic labels. This is
  development evidence, not a held-out test.
- Every estimator here assumes conditional independence between groups given y. Stage 0 and Step 481 showed that
  assumption is violated, and that even Oracle-CI loses to strong PRMs on Socratic. No arm is expected to remove that
  gap.
- The next-step view partly tracks the neighbouring step's label and position (r .23 on Socratic Qwen-7B).
