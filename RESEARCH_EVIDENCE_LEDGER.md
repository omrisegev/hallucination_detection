# Research evidence ledger

This ledger distinguishes results that transferred to the real hallucination
features from mechanisms demonstrated only in synthetic data.  A method is
marked **retained** only when its supported component remains part of the next
experiment; synthetic success alone is never recorded as a real-data win.

## Evidence entering hierarchical-active v2

| component or claim | strongest evidence | status | consequence |
|---|---|---|---|
| confidence-oriented U-PCR is the incumbent | fixed-stable real macro 0.7735 over 24 cells; fixed orientation stays within -0.06pp of per-cell `sign(rho)` | **retained** | every new head is anchored on U-PCR and must beat it directly |
| fixed feature orientation removes the polarity failure mode | consensus and historical EPR anchors agreed in 288/288 fixed-schema comparisons | **retained** | no correctness-derived per-cell polarity is allowed |
| low-dimensional PCR protects the useful head | adding the inverse tail cost 3.28pp; full inverse SDSF cost 3.32pp | **retained** | corrections stay two-dimensional at the target |
| bootstrap SDSF stabilizes SDSF | +1.80pp over current SDSF, 23W/1L | **mechanism only** | useful stabilization, but not a replacement for U-PCR |
| bootstrap SDSF improves the incumbent | -2.91pp versus SU-PCR, 2W/22L | **rejected on real cells** | do not reopen full-inverse SDSF |
| raw pair-product covariance solves dependency-aware rho | every sealed v5 solver worsened full-rho error; GLS damaged retained coordinates | **rejected synthetically** | do not try more raw GLS estimators for the same equations |
| per-cell six-direction trusted-label correction | -0.36pp versus U-PCR at 20 labels, CI [-0.64, -0.05] | **rejected on real cells** | reduce target correction dimension and share information across cells |
| per-cell two-direction trusted-label correction | safer than six directions; -0.15pp at 20 labels and approximately tied by 80 labels | **retained control** | use as the local labelled baseline, not yet as an improvement |
| U-PCR pseudo-label self-training | at 20 labels, pseudo+gold lost to anchored-6 in all 24 cells by 3.97pp | **rejected on real cells** | no pseudo-label arm in v2 |
| dependency correction can repair a biased U-PCR head | anchored-6 gained +0.79pp on sparse pairs and +30.71pp on a correlated weak block | **mechanism only** | preserve a shared-correction synthetic positive control |
| anti-redundancy feature selection | DPP -8.08pp (0W/24L); decorrelation -5.98pp against matched random subsets | **rejected on real cells** | do not treat diversity as an automatic benefit |
| a label-derived feature subset has real headroom | label-handed oracle +2.25pp; half-split oracle transfers 84% of the gain | **oracle evidence** | labels can help, but the target is non-unique and unstable |
| fixed choices transfer across cells/families | numerous LOCO/LOFO feature and shape choices were flat or negative | **unsupported** | v2 excludes the entire target family and treats transfer as a falsifiable mechanism |

## Current cycle

The registered protocol is `SPEC_HIERARCHICAL_ACTIVE_SPECTRAL_V2.md`.  The
confirmatory decision is **STOP_AND_REVISE**.

| component or claim | v2 evidence | status | consequence |
|---|---|---|---|
| a same-domain LOFO correction transfers | pooled-only -1.44pp vs U-PCR, CI [-2.01, -0.92], 1W/23L | **rejected on real cells** | stop cross-family linear correction for this feature bundle |
| broad pooling fixes the hierarchy | pooled-all -0.96pp, 3W/21L; less harmful but uses more donor labels | **rejected on real cells** | QA/math membership is not a useful correction hierarchy |
| active acquisition improves a local two-score head at 20 labels | active-local minus uniform-local -0.13pp, CI [-0.30, +0.04] | **unsupported** | do not claim generic active-learning gain |
| active acquisition can reject a bad transferred prior | hybrid-active minus hybrid-uniform +0.78pp, CI [+0.53, +1.08], 23W/1L | **retained safety mechanism** | informative labels should be used to test/shrink external priors, not assumed to improve U-PCR |
| hierarchical-active combined method improves U-PCR | -0.19pp, CI [-0.41, +0.02], 8W/16L | **rejected on real cells** | do not promote the v2 candidate |
| shared correction is learnable when truly shared | +47.63pp in the sealed shared-correction meta-world | **mechanism only** | harness is capable of detecting the intended effect; real transfer failure is informative |
| a selector over U-PCR and v2 can yield a major gain | perfect cell-level switch ceiling +0.12pp | **closed as headline** | safety gating cannot reach the contribution bar on these candidates |

## Current best-supported direction

Keep stable confidence-oriented U-PCR as the fusion rule.  Move the next
semi-supervised experiment to the feature/subset channel, where the existing
label-handed oracle shows +2.25pp of room and half-split selection retains 84%
of it.  Do not spend another cycle on pseudo-label feedback, inverse spectral
tails, raw pair-covariance weighting, or cross-family linear correction unless
new features or an independent supervision source change the information set.
