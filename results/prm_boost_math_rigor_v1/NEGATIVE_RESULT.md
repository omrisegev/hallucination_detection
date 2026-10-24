# Negative result — pre-registered label-free anchored improvement gate (prm-boost-anchored-gate-v1)

**Hypothesis (one sentence):** a label-free statistic G(w), computed from CHAIN_N16's own cross-stage views by
anchoring on the PRM (third-order cumulant route, `FRAMING_THEORY.md` P5), predicts per cell whether CHAIN_N16 beats
the PRM alone in within-answer AUC.

**Date / step:** 2026-10-24, Step 469 [PRM math rigor]. Pre-registration committed first: `6b7162f5`,
`GATE_PREREG.json`.

**Benchmark / population:**
- **Primary (10 cells).** PRMBench and Socratic-PRMBench × 5 PRMs.
  - PRMBench: 94,203 steps (`stage1/STAGE1_DIAGNOSTICS.csv`, `n_steps_eval`). Labels: `prmbench_meta.json`, raw
    1-based error steps.
  - Socratic: 26,052 steps. Labels: `socratic_gold.json`.
- **Secondary (5 cells).** Hard2Verify on the 177-answer common set, 1,703 steps.
- **PERM control.** 5 PRMBench cells.
- No folds: the gate is label-free and fitted per cell (transductive), like CHAIN itself.

## Arms compared
Arms are the four decision rules. "Within-AUC" is the mean within-answer AUC of the arm each rule picks, over the 5
cells (`stage2/POSTHOC_NUMBERS.json`). ProcessBench was excluded by the pre-registration (degenerate within-AUC,
out of focus).

| arm | what it is | PRMB within-AUC | PB macro-F1 | vs reference |
|---|---|---|---|---|
| reference (incumbent) | always fuse: CHAIN_N16 everywhere (today's method) | .8098 (Socratic .6209) | not run | — ; primary correct 6/10 |
| simple control | always PRM | .7411 (Socratic .7402) | not run | primary correct 4/10 |
| simple control | the review's gate: rank-one 'complete' loadings of the whole off-diagonal, G on the point estimate. It said fuse in 15/15 cells | .8098 (Socratic .6209) | not run | identical to the reference; 6/10 |
| candidate | anchored cumulant gate, 200-answer bootstrap, 90% interval. **It abstained in 15/15 cells**, so it fell back to the PRM everywhere | .7411 (Socratic .7402) | not run | primary correct **4/10**: pre-registered verdict **fail** (`stage2/GATE_SUMMARY.json`) |

**Win/loss record:** candidate vs reference over the 10 primary cells, by correct choice: 4–6
(`stage2/POSTHOC_NUMBERS.json`).
- **Primary endpoint** (share of correct choices). Difference −0.2; paired cell-bootstrap CI [−0.8, +0.4].
  Includes 0: yes. The pre-registered bar was ≥ 9/10.
- **Not pre-registered, for context.** Mean within-AUC of the picked arm: +.0253, CI [−.0478, +.1089]. Includes 0:
  yes. The large Socratic losses of CHAIN drive it.
- **PERM negative control.** "Not fuse" in 5/5 (`stage2/GATE_SUMMARY.json`). Uninformative here, since the gate
  never chose fuse.

## Three plausible reasons it failed
1. **The anchor condition does not hold on these views.** P5 assumes that, within each class, the PRM's noise is
   independent of the telemetry groups once position/length are removed. The measured mean |within-class
   correlation| of the residualized PRM with the group scores is .13–.35 (max .20–.59) across the 15 cells
   (`stage1/STAGE1_DIAGNOSTICS.csv`, `anchor_within_corr_*`). The telemetry is read from the PRM's own fine-tuned
   backbone, so shared noise is plausible. In synthetic data, shared PRM–telemetry noise already lowered this
   gate's accuracy from .793 to .640 (`synthetic/run_v1/SUMMARY.json`, `shared_heavy/cumulant`).
2. **The third-order structure contradicts the model.** With p < ½, every telemetry view should show positive
   C_0k, κ(0,0,k) and κ(0,k,k). Only 21/44 views do on the gate's own inputs. With per-answer SD scaling removed,
   17/44 do (`stage2/POSTHOC_CUMULANT_SIGNS.csv`, post hoc). The violation is therefore not caused by answer-z.
   Without enough sign-consistent views the estimator returns "anchor loading not identified" or loadings above
   the view variance, so it abstains (`stage2/GATE_RESULTS.csv`, `reason`).
3. **The covariance among views is mostly within-class structure, not error signal.**
   - Within-class covariance makes up .82–1.00 of the off-diagonal norm in every cell
     (`stage1/STAGE1_DIAGNOSTICS.csv`, `within_class_share_of_offdiag`). Moment-based identification has little
     signal to work with.
   - Even the synthetic study, where the model holds, abstained in 44% of PRMBench-size regimes under heavy tails
     (`synthetic/run_v1/SUMMARY.json`).

## Verdict
- [ ] closes the DIRECTION (the idea cannot work under this access/contract)
- [x] closes this IMPLEMENTATION only (roster, lambda, readout, bank); the idea stays open

What is closed: the third-order anchored estimator applied to the PRM and its *own backbone's* telemetry.
What stays open, for two reasons:
- **The improvement condition itself holds.** With labels plugged into the same formula, it predicts the sign of
  CHAIN − PRM in 13/15 cells, with correlation .979 between predicted and actual differences
  (`stage1/STAGE1_SUMMARY.json`).
- **The failure traces to a measured assumption.** The anchor's noise is correlated with the telemetry. A
  different anchor could satisfy it.

The result is not a finding that the gate idea is irrelevant. What fails, at this stage of the work, is
label-free identification with these views.

**Implementation note, found after the run.**
- `FRAMING_THEORY.md` P5(g) lists "an anchored covariance C_0k ≤ 0" as an abstain condition.
- `fusion_theory.gate_decision`, which the pre-registration points to, only counts such views
  (`n_negative_anchor_cov`); it does not abstain on them.
- This did not change the outcome: all 15 cells abstained for other reasons (`stage2/GATE_RESULTS.csv`, `reason`).
- A successor pre-registration should make the two match.

## What would reopen it
An anchor view whose within-class correlation with the telemetry groups is near zero on the Stage 1 check. For
example, the PRM head with telemetry from a *different* model: the standalone base-model telemetry on PRMBench,
`results/prm_standalone_telemetry_v1`. A declared labeled calibration split (Pareto Testing, `ADVISOR_TOOLS.md`)
would also reopen it, but that is a different access level.

**Source files:**
- Results: `stage2/GATE_RESULTS.csv`, `stage2/GATE_SUMMARY.json`, `stage2/POSTHOC_NUMBERS.json`,
  `stage2/POSTHOC_CUMULANT_SIGNS.csv`, `stage1/STAGE1_DIAGNOSTICS.csv`, `stage1/STAGE1_SUMMARY.json`,
  `synthetic/run_v1/{SUMMARY,LEVEL_SUMMARY}.json`, `GATE_PREREG.json`. There is no REPORT.html.
- Commands, all run with `COLLECTION_DATA=/Users/osegev/Desktop/hd_collection_data` and the
  `/Users/osegev/.venvs/hd_math_rigor` interpreter:
  - `python stage2_gate.py stage2`
  - `python stage1_diagnostics.py stage1`
  - `python posthoc_gate_diagnostics.py stage2`
- n_checked / n_total: 15 / 15 in-focus cells plus 5 / 5 PERM cells. The primary endpoint covers 10 / 10.
  Inputs are parity-checked: `stage0/PARITY_SUMMARY.json`, 90/90 metric rows and 15/15 refits.
