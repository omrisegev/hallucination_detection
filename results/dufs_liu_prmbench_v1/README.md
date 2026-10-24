# DUFS-LIU on PRMBench localization (2026-10-24/25, Claude)

Omri's questions (2026-10-24):
1. Does a label-using calibration identify better whether DUFS-LIU-PCR is on a nuisance manifold or on the
   hallucination manifold?
2. Can DUFS-LIU-PCR replace continuous L-SML for hallucination localization on PRMBench?
3. Can the PRM's own pseudo-labels overcome the DUFS-LIU-PCR problems instead of labels?
4. Does the picture change with more channels (56 instead of 16)?

Pre-registration: `PREREG.md` (commit `cbbb140f`, before any score; amendment 1 for the 56-channel bank in commit
`83141318`, before any 56-channel score). CPU only, no inference. PRMBench is development data: nothing here is
untouched confirmation.

## Bottom line

1. **DUFS-LIU cannot replace continuous L-SML on PRMBench.** Over the PRM and the N16 bank it loses to CHAIN_N16
   on all 5 PRMs: within-answer AUC -0.013 to -0.028 and GAP PRMScore -0.74 to -1.75. Every 95% source-group CI is
   below zero (`CONTRASTS.csv`, 5 PRMs x 6,969 answers). Here it is the same model as plain IU-PCR: the weight cosine
   is 0.9998-1.0000 on 5 of 5 PRMs, and DUFS adds -0.0000 to +0.0011 within-AUC.
2. **The bottleneck is the two-component subspace, not the graph.** IU-PCR and DUFS-LIU fuse only inside the two
   leading principal components of the 17 views. Those components hold about half of the channel variance, but only
   5-40% of the PRM channel's variance and 49-80% of the variance of the in-sample supervised error score
   (`SUBSPACE_POSTHOC.csv`, post hoc). So the PRM ends with a 2-7% weight share, against 13-24% in CHAIN (`RESULTS.csv`).
   The gates and the Laplacian only reweight inside that subspace. Every gate variant (DUFS, labels, PRM
   pseudo-label) leaves the weights at cosine >= 0.997 with plain IU-PCR (`scores/FIT_*.json`).
3. **Label calibration identifies little through this machinery.**
   - Labels as graph gates (TA-LIU, primary): the pre-registered "helps" rule passes (CI > 0 on 3 of 5 PRMs), but the
     change is -0.0005 to +0.0016 within-AUC. It is the same with 64 labeled answers or with all of them, and the
     PRMScore change is -0.08 to +0.09 (no CI excludes 0).
   - Labels as the fusion target (RHO-LIU): +0.002 to +0.012 within-AUC on 3 of 5 PRMs.
   - Both lose to logistic regression trained on the same labels, by 0.022-0.057 within-AUC (efficiency rule: 0 of 5).
   - The labels themselves carry the information. Logistic regression with only 64 labeled answers is already level
     with CHAIN on PRMScore (-0.20 to +0.96), and with 512 it beats CHAIN by +0.9 to +2.0 PRMScore. The DUFS-LIU
     structure cannot use them.
   - Part of the logistic-regression edge is PRMBench's step-position prior. With step position and length removed
     from the scores, logistic regression on all labels minus CHAIN is +0.010 / +0.014 / +0.013 / -0.002 / +0.012
     (`DIAGNOSTICS.csv`).
4. **The PRM's pseudo-labels do not close the gap.**
   - As graph gates (PA-LIU, primary): -0.0004 to +0.0030 within-AUC. The pre-registered rule passes (CI > 0 on 3 of
     5, PRMScore point >= 0 on 3 of 5), but the gain is negligible next to the 0.013-0.028 deficit to CHAIN.
   - As the target (PRHO-LIU): mixed, from +0.002 (Qwen-7B) to -0.016 (RLHFlow).
5. **With 56 channels (Qwen3-8B telemetry, Qwen-7B PRM) DUFS-LIU gets worse and logistic regression gets better.**
   - N56 minus N16: DUFS-LIU -0.016 within-AUC on telemetry alone and -0.024 with the PRM. Logistic regression with
     512 or all labels +0.010 to +0.020. CHAIN +0.001 [-0.002, +0.004].
   - At N56, DUFS-LIU plus PRM falls below the PRM alone (-0.034 within-AUC, -1.80 PRMScore) and below CHAIN
     (-0.037, -1.21).
   - The two-component subspace now holds 40% of the supervised score (64% at N16).
   - Labels or pseudo-labels in the graph or in the target change it by at most 0.0007 (`n56/VERDICT.json`).

**What it means for the manifold question.** On PRMBench steps, what DUFS-LIU-PCR fuses toward is decided by the
variance-chosen two-component subspace of the channels, the dominant shared-uncertainty direction of the telemetry.
It is not decided by the DUFS graph. Step position is not that direction: it has R^2 0.01-0.10 on the two components.
A calibration of the graph, by labels or by the PRM, cannot leave that subspace. A label-calibrated target can reach
only the subspace's own ceiling (RHO-LIU on all labels reaches the alignment the subspace allows: correlation with the
supervised score 0.68-0.90). Continuous L-SML avoids the trap without labels because CHAIN does not project onto the
top-variance components and gives the PRM its own group.

### Five PRMs, PRM-backbone N16: within-answer AUC / GAP PRMScore

| arm | access | Qwen-7B | Qwen-72B | PURE | RLHFlow | Math-Shepherd |
|---|---|---|---|---|---|---|
| PRM alone (better scale) | PRM | .801 / 68.01 | .795 / 69.49 | .763 / 66.24 | .662 / 59.39 | .684 / 60.21 |
| CHAIN_N16 (continuous L-SML) | label-free | **.838 / 70.40** | **.843 / 71.09** | **.816 / 69.51** | **.775 / 66.25** | **.778 / 66.39** |
| IU-PCR, PRM + N16 | label-free | .821 / 69.66 | .824 / 69.11 | .797 / 68.69 | .761 / 65.47 | .750 / 65.01 |
| DUFS-LIU, PRM + N16 | label-free | .821 / 69.66 | .825 / 69.34 | .797 / 68.64 | .762 / 65.47 | .750 / 65.06 |
| PA-LIU (PRM pseudo-label gates) | PRM pseudo-label | .822 / 69.76 | .828 / 69.55 | .797 / 68.58 | .764 / 65.76 | .750 / 64.97 |
| PRHO-LIU (PRM pseudo-label target) | PRM pseudo-label | .823 / 70.09 | .827 / 69.99 | .791 / 68.43 | .745 / 64.76 | .748 / 64.21 |
| TA-LIU, 512 labeled answers | labels | .822 / 69.68 | .827 / 69.43 | .797 / 68.61 | .762 / 65.47 | .750 / 64.98 |
| RHO-LIU, 512 labeled answers | labels | .823 / 69.86 | .837 / 70.98 | .797 / 68.66 | .766 / 66.06 | .751 / 64.58 |
| logistic regression, 64 answers | labels | .833 / 70.30 | .847 / 71.44 | .817 / 69.30 | .787 / 67.21 | .790 / 66.54 |
| logistic regression, 512 answers | labels | .846 / 71.34 | .859 / 72.29 | .834 / 70.38 | .805 / 68.26 | .807 / 67.64 |

Label arms are cross-fitted (5 source-group folds; 5 calibration draws for 64 and 512). Bold = best label-free arm.

### Qwen3-8B telemetry, 16 versus 56 channels (Qwen-7B PRM): within-answer AUC / GAP PRMScore

| arm | N16 | N56 |
|---|---|---|
| continuous L-SML, telemetry only | .779 / 67.57 | .771 / 66.13 |
| DUFS-LIU, telemetry only | .780 / 67.61 | .764 / 66.09 |
| logistic regression (all labels), telemetry only | .802 / 67.77 | .822 / 69.82 |
| PRM alone | .801 / 68.01 | .801 / 68.01 |
| CHAIN (continuous L-SML), PRM + telemetry | .803 / 68.57 | .804 / 67.41 |
| DUFS-LIU, PRM + telemetry | .791 / 68.17 | .768 / 66.20 |
| PA-LIU / TA-LIU (512) / RHO-LIU (all labels), PRM + telemetry | .791 / 67.67-68.25 | .768 / 66.20-66.22 |
| logistic regression (512 answers), PRM + telemetry | .834 / 69.74 | .844 / 70.76 |

## Design in one paragraph

DUFS-LIU fuses channels in two steps that can each follow a nuisance instead of the error: the DUFS gates choose the
coordinates of a sample graph (whose Laplacian smooths the weights), and IU-PCR picks the latent it fuses toward from
the channels' shared covariance, inside the two leading principal components. The experiment feeds three kinds of
outside information into those two places and compares the result with continuous L-SML (CHAIN_N16) on the same
inputs: (a) calibration labels (cross-fitted by source group, 64 or 512 answers, or all training folds), either as
graph gates (TA-LIU, the existing target-anchored construction) or as the fusion target (RHO-LIU: `rho = Cov(x, y)`);
(b) the PRM's own GAP flags as a pseudo-label, in the same two places (PA-LIU, PRHO-LIU); (c) nothing (registered
DUFS-LIU and plain IU-PCR). Logistic regression on the same calibration labels is the attribution control.

## Files

| file | content |
|---|---|
| `PREREG.md` | questions, arms, metrics, success rules, parity gates, amendment 1 |
| `run.py` | PRM-backbone N16 (5 PRMs): `parity`, `fit <prm>`, `evaluate`, `talius_parity` |
| `evaluate_dufs.py` | metrics, bootstrap contrasts, manifold diagnostics, verdicts for `run.py` |
| `run_n56.py`, `evaluate_n56.py` | amendment 1: Qwen3-8B telemetry, banks N16 and N56, Qwen2.5-Math-PRM-7B |
| `PARITY.csv`, `n56/PARITY.csv`, `TALIU_SPLIT_PARITY.json` | parity gates (all pass) |
| `RESULTS.csv`, `CONTRASTS.csv`, `DIAGNOSTICS.csv`, `GATES.csv`, `WEIGHTS.csv`, `SUBSPACE_POSTHOC.csv`, `VERDICT.json` | PRM-backbone N16 outputs |
| `n56/RESULTS.csv`, `n56/CONTRASTS.csv`, `n56/DIAGNOSTICS.csv`, `n56/SUBSPACE_POSTHOC.csv`, `n56/VERDICT.json` | amendment 1 outputs |
| `scores/`, `n56/scores/` | per-step scores (`SCORES_*.npz`) and fit records (`FIT_*.json`: weights, gates, flips) |

Helpers: `spectral_utils/anchored_liu.py` (external-target LIU solve, registered DUFS-LIU, split call) and
`spectral_utils/bank_rebuild.py` (nested-bank rebuild and the CHAIN construction, ports with parity gates). Tests:
`tests/test_anchored_liu.py` (8 pass).

## Reproduce

```
export COLLECTION_DATA=/Users/osegev/Desktop/hd_collection_data
PY=/Users/osegev/Desktop/hallucination_detection/.venv/bin/python
$PY results/dufs_liu_prmbench_v1/run.py parity
for p in qwen_prm_7b pure_prm rlhflow_prm math_shepherd; do $PY results/dufs_liu_prmbench_v1/run.py fit $p; done
TALIU_SPLIT=1 $PY results/dufs_liu_prmbench_v1/run.py fit qwen_prm_72b
$PY results/dufs_liu_prmbench_v1/run.py talius_parity
$PY results/dufs_liu_prmbench_v1/run.py evaluate
cd results/dufs_liu_prmbench_v1 && $PY run_n56.py parity
for b in N16 N56; do for s in tele prm; do $PY run_n56.py fit $b $s; done; done
$PY run_n56.py evaluate
```

Implementation note. The canonical `laplacian_iu_path` computes an ARPACK algebraic-connectivity diagnostic on every
call. On a connected 94,203-node graph it ran for over 15 minutes per call (Qwen-72B cell), so that cell and every
amendment-1 TA-LIU fit use the same fits split into their parts (same gates, same graph, same projected solve, no
diagnostic). The weights are identical: unit tests at 1e-10, and on real data the split calls reproduce every saved
canonical score of the Qwen-7B cell with difference 0.0 (`TALIU_SPLIT_PARITY.json`).

## Caveats

- Development data; the label arms use labels and are not label-free methods; the PRM pseudo-label arms read no
  PRMBench label, but the PRMs were trained by their authors with labels on other data.
- The nuisance diagnostics name step position and step length only.
- The 56-channel test uses the Qwen3-8B telemetry with one PRM, not the five PRM backbones.
