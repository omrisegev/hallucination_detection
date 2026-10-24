# Pre-registration: label-free improvement gate for PRM boosting (2026-10-24)

**This file is committed before any gate statistic is computed on real data, and before the label-using
Stage 1 diagnostics run.** Machine-readable copy: `results/prm_boost_math_rigor_v1/GATE_PREREG.json`. It is read
by `stage2_gate.py`, which cannot run without it.

## Question

Can a label-free statistic predict, cell by cell, whether CHAIN_N16 beats the PRM alone in within-answer AUC? The
statistic uses only CHAIN's own inputs: the PRM risk and the telemetry group scores of one benchmark × PRM cell.
It answers review task 1 / W3, and the theory is in `FRAMING_THEORY.md` P2 and P5.

## What is already known (disclosure)

- **The 15 outcome signs are already known** (`results/prm_boost_collection_v1/tables/STATUS_MAIN.csv`).
  CHAIN > PRM holds in 9/15: PRMBench 5/5, Socratic 1/5, Hard2Verify 3/5.
- The label-using rule "telemetry-alone ≥ PRM" predicts all 15. It is a ceiling, not a method.
- **The gate's estimator was chosen on synthetic data only**, by a rule written before that run:
  - script: `results/prm_boost_math_rigor_v1/synthetic/select_gate_estimator.py`
  - result: `synthetic/run_v1/SUMMARY.json`
- No gate quantity and no Stage 1 diagnostic has been computed on real data.
- Stage 0 (parity) has run. It reproduces the exported scores and the published metrics and computes nothing new:
  `stage0/PARITY_SUMMARY.json`, 90/90 metric rows exact, 15/15 refits within 2.1e-7.

## Fixed design

| Item | Fixed choice |
|---|---|
| Inputs | CHAIN_N16's cross-stage views VC = [answer_z(PRM risk), answer_z(group scores)] and CHAIN's effective weights w, rebuilt by `common.chain_inputs` (parity-checked) |
| Estimation matrix | VC residualized within answers on `covariate_basis`: relative position, position², position³, plus log step tokens on PRMBench |
| Estimator | `fusion_theory.anchored_loadings`, anchored on view 0 (the PRM), method **`cumulant`**: u_0² = median_k C_0k·κ(0,0,k)/κ(0,k,k), then u_k = C_0k/u_0. The synthetic selection chose it (see the table below) |
| Statistic | G(w) = SNR(w) − SNR(PRM), Σ = C − uu' (`fusion_theory.gate_statistic`) |
| Decision | `fusion_theory.gate_decision`: 200 answer-bootstrap resamples, 90% interval, seed 20261024. Fuse iff the lower bound > 0. Otherwise use the PRM ('prm' or 'abstain') |
| Truth | sign of within-AUC(CHAIN_N16) − within-AUC(PRM) on the collection's evaluation sets (Hard2Verify: the 177-answer common set) |
| Primary cells | PRMBench and Socratic × 5 PRMs (10 cells) |
| Secondary cells | Hard2Verify × 5 (181 answers; many abstentions expected from the synthetic study) |
| Excluded | ProcessBench: its within-answer AUC is degenerate (position alone 1.000), and it is out of focus |

## Synthetic selection (done before this file was committed; no real data)

**Estimator** (`synthetic/run_v1/SUMMARY.json`).
- Setup: heavy-tailed noise, answer effects, a position covariate that also loads on the PRM, and a telemetry-wide
  latent nuisance; sizes like Hard2Verify, Socratic and PRMBench.
- Action accuracy, with abstain counted as PRM, averaged over the three sizes:

| method | heavy-tailed | Gaussian | shared PRM–telemetry noise |
|---|---|---|---|
| cumulant | **.793** | .737 | .640 |
| cumulant_w01 | .793 | .737 | .651 |
| cca | .739 | .824 | .599 |
| rank (point estimate) | .707 | .749 | .554 |
| fa | .633 | .642 | .436 |

- The pre-declared rule chose `cumulant`: tied with `cumulant_w01`, and equal abstain at the PRMBench size.
- Under heavy tails, when it decides it is always right (accuracy 1.00 at every size). But it abstains often:
  97% at the Hard2Verify size, 57% at Socratic, 44% at PRMBench.

**Interval level** (`synthetic/run_v1/LEVEL_SUMMARY.json`).
- Heavy-tailed action accuracy:
  - point estimate .739
  - level .5: .783
  - level .7: .782
  - level .8: .793
  - level .9: .793
- The pre-declared rule breaks ties within .02 toward the more conservative level, so it keeps **0.90**.

**What this predicts for the real test.** Many PRMBench cells may abstain. Abstention falls back to the PRM, and
the PRM is the wrong choice on PRMBench. The primary bar below may therefore not be met. The tiers are fixed now,
before any real-data computation.

## Success criteria

- **Primary.** The gate picks the better arm in the 10 primary cells. For reference, "always fuse" gets 6/10 and
  "always PRM" gets 4/10.
  - **Pass:** ≥ 9 of 10.
  - **Partial:** 7–8 of 10. Better than both trivial rules, but not decisive.
  - **Fail:** ≤ 6.
- **Negative control.** On the five PRMBench cells with the PERM construction, the gate does **not** choose fuse in
  **≥ 4 of 5**. PERM moves each answer's telemetry block to another answer with the same step count
  (`common.perm_rows`, the ablation5 seed; reproduced to 9e-8 of the exported CHAINPERM_N16). Fusion loses there:
  CHAIN .52–.59 vs PRM .66–.80.
- **Reported in every case, with no claim attached:**
  - Hard2Verify correct choices and abstentions;
  - all 15 cells together;
  - the review's rank-one gate under the same scoring;
  - G for the Fisher weights.

## What happens afterwards

- **No tuning after seeing the results.** A failure of the primary criterion is written with `/negative-result`.
- Any changed estimator, decision rule or covariate basis is a new pre-registration with a new ID.
- A pass is development evidence on data the project has inspected many times. It is not untouched confirmation.
  It earns one thing only: the right to build the Stage 3 candidate (anchored Fisher weights + this gate), which
  gets its own pre-registration with a non-inferiority bar against CHAIN on PRMBench.
