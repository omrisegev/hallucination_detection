# Pre-registration: AUTO-F, AUTO's partition with block-CCA/Fisher weights (2026-10-25)

ID `prm-boost-auto-f-v1`. JSON: `results/prm_boost_math_rigor_v1/AUTO_F_PREREG.json`. Code: `stage9_auto_f.py`.
**This file, the JSON and the code are committed before the first run.**

## Why

Omri prefers AUTO to CHAIN. AUTO has no hand-picked "PRM in its own group" prior, and so it resolves the W7
contradiction.

AUTO loses to CHAIN in 5/5 PRMBench PRMs (Step 475). It absorbs the PRM into a telemetry group, and inside that group
the L-SML eigenvector shrinks the weights toward equality (P3). The PRM's least-squares share falls to .005–.07.

AUTO-F keeps AUTO's partition and changes only the weights. Under the block model that L-SML already assumes, the
consistent weights are Fisher weights. The Ising aggregation paper (arXiv 2601.22336) supports the same principle
for class-independent dependence: the rule stays linear, but shared failure modes are discounted instead of counted
repeatedly.

## The AUTO-F estimator (label-free, one pass, no new inference)
1. **Partition.** AUTO's own partition of [z(PRM), N16]: free K, or forced K for part B.
2. **Normalization.** Every column is answer-z'd (P1). Constant columns are dropped, as AUTO drops them.
3. **Group composites.** Each group gets one composite by **block-vs-rest CCA**, with a ridge of 1e-3 × the mean of
   the diagonal.
   - Under the block model, the composite's direction is Ψ_b⁻¹u_b.
   - A singleton is its own composite.
   - Each composite is oriented to correlate positively with z(PRM), and standardized.
4. **Cross weights.** A one-factor ML factor analysis (sklearn) runs on the composites.
   - The loadings are oriented so that the PRM's composite loads positively.
   - Negative loadings are set to 0.
   - Weights are w = λ/ψ, with ψ floored at .01.
5. **Score.** fused = Σ w_b z_b, oriented with z(PRM), read out with GAP. With fewer than 3 composites, the score
   falls back to AUTO's; this is declared, and the fallbacks are counted.

## Endpoints (PRMBench, 5 PRMs)

Paired bootstrap, 2000 draws, using the collection's source-group weights.

- **E1. AUTO-F improves AUTO.** The within-AUC CI of AUTO-F − AUTO is above 0, in ≥ 4/5 PRMs.
- **E2. AUTO-F is not inferior to CHAIN.** In ≥ 4/5 PRMs, both hold for AUTO-F − CHAIN:
  - the within-AUC point estimate is ≥ −.005;
  - the GAP PRMScore point estimate is ≥ −0.5.
- **Verdict.**
  - Switching to AUTO-F **passes** iff E1 and E2 both hold. Otherwise it **fails**, and the failure is recorded with
    `/negative-result`.
  - **E3**, the W8 endpoint, is reported as its own result: for K = 2..8, the range of the least-squares PRM share is
    smaller for AUTO-F than for CHAIN with forced K, in ≥ 4/5 PRMs.

## Diagnostic: is a class-dependent (quadratic) direction worth pursuing? (labels, not a method)
- **Features.** [z(PRM), answer-z'd non-constant N16 channels].
- **Models.** LDA against QDA (reg 0.01), with 5-fold GroupKFold by source group.
- **Metric.** Within-AUC of the out-of-fold log-odds.
- **Rule.** "Worth pursuing" if QDA − LDA ≥ .01 in ≥ 3/5 PRMBench PRMs; otherwise "not supported".
- **Caveat.** A quadratic gain can also come from nuisance interactions, so it is a lead and not proof of the Ising
  mechanism.

## Reported without a criterion
- Socratic and Hard2Verify.
- The AUTO-F weights, loadings, number of composites, number of negative loadings, and the size of the PRM's group.
- The PRMScore range over K.

Nothing is tuned after the results.
