# Pre-declaration: redundancy errors under the SLIDER protocol, PRMBench and Socratic (2026-10-26)

**Status.**
- Requested by Omri on 2026-10-26.
- Written before any number below was computed. Not committed.
- Descriptive and in-sample: SUB5, S_R and C3 were found or derived on these benchmarks.

## Question

Which component adds the most on redundancy errors, the error type Socratic-PRMBench reports as hardest and SLIDER
targets? The candidates are the telemetry, the original PRM boosting, SUB5 or the rule set. Does any of them beat
SLIDER?

## SLIDER itself cannot be run

- Step-RRI (arXiv 2610.00571) has no code release.
- It needs language-model inference to estimate a partial-information decomposition. GPU and new inference are not
  approved.
- The only SLIDER number is its paper's PRMBench redundancy accuracy, 61.9. Its protocol is under-specified.
- On Socratic there is no SLIDER number. Our methods are compared with the PRM, position and each other.

## Cells

PRMBench and Socratic × {Qwen-7B, Qwen-72B, PURE, RLHFlow}. Math-Shepherd is dropped (Omri's decision of 2026-10-26).

## Step subsets

- **PRMBench:** the `redundency` class, all steps of those answers.
- **Socratic:** each of premise_redundancy, decomposition_redundancy, regather_redundancy and
  integration_redundancy, plus the four pooled.

## Arms (all label-free scores)

**References:**
- the PRM (stage-1 reference scale);
- relative position (later = riskier);
- CHAIN_N16 and TELE_N16 as stored, the original label-free PRM-boosting bank.
- PRMBench only:
  - LSML_CONT, the original 3-channel PRM boosting; its channels were chosen with PRM-7B labels;
  - CHAINRES_N16 and TELERES_N16, the stored position/length-residualized controls.

**Full bank, both representations** (stage-1 / stage-2 L-SML machinery on all 16 channels):
- ALL16 + PRM, raw and position-neutral;
- ALL16 alone, raw and position-neutral.

**Sets** (scores saved by `eval8.py run`): SUB5, S_R and C3 × {raw, position-neutral} × {+PRM, alone}.

## Metrics

**SLIDER protocol, as in `collection_competitors.slider`.**
- A logistic regression on the score with balanced class weights is fitted on a random half of the subset's
  answers. Accuracy and balanced accuracy are measured on the other half; mean over 10 splits.
- Labels set only this one-dimensional threshold, as in SLIDER's own evaluation.

**Label-free:**
- GAP-flag accuracy and balanced accuracy on the same steps;
- within-answer AUC.

**Parity check:** CHAIN_N16 and the PRM must reproduce `COMP_SLIDER.csv` on PRMBench.

## Predictions

- **R1.** On PRMBench redundancy, every telemetry-containing arm beats SLIDER's 61.9 accuracy and the PRM's balanced
  accuracy.
- **R2.** On Socratic, the telemetry beats the strong PRMs on regather_redundancy in balanced accuracy, and does not
  beat them on integration_redundancy.
- **R3.** Position neutralization changes the PRMBench redundancy result by less than 2 points of balanced accuracy
  for SUB5 + PRM. Position is weak on PRMBench redundancy.

All arms are reported. No arm is selected after the fact.
