# Pre-label stop: hurdle HMM v1 (2026-11-02)

**What happened.** The pre-registered synthetic stop rule (`docs/experiments/HURDLE_HMM_PREREG_2026-11-02.md` §7)
failed. As the rule says, no real-data fit was run and no real label was read.

**Synthetic result** (`SYNTHETIC_RECOVERY.csv`, 18 fits, data drawn from the model itself):

| π | δ | median π̂ over 3 seeds | rule |
|---|---|---|---|
| .10 | 2 | .158 | fail |
| .60 | 2 | .452 | fail |
| .95 | 2 | 1.000 (at the bound) | pass |
| .10 / .60 / .95 | 1 | .407 / .221 / .858 | reported only (all miss by > .09) |

**Cause** (found by my own check and by the independent review, `REVIEW_STAGE1.md`):
- The likelihood is correct: two independent re-implementations agree to ≤ 1e-13 with the same quadrature.
- The fit is the global optimum of the anchored likelihood. With the TRUE clean law, π̂ is recovered within .03 in
  the three failing draws (.085 / .595 / .571).
- So the anchor is the cause. The truncated-normal fit on the bottom 30% has sd(μ̂0) .11–.19 at CRV and
  ProcessBench sizes. At δ ≈ 1–2, a .1–.2 error in μ0 moves π̂ a lot.
- A second defect: the 20-node quadrature centred on the prior is inaccurate on long answers (error in
  P(A=1|x) up to .13 at n = 60, .28 at n = 107). This must be fixed in any re-run.

**Reading.** Prevalence is identified by this model only if the clean law is known to about .05 sd. A label-free
low-tail anchor does not give that precision at our sample sizes. This agrees with T3 and P6 (theory note
2026-11-01): at d ≈ 1, the decision layer is ill-conditioned without calibration.

**Not closed.** The model class is not closed. Two re-designs need an amendment and a new synthetic test:
(A) anchor the clean law with a small labelled set (about 50 answers, in line with Step 487), adaptive quadrature;
(B) keep it label-free and treat (μ0, s0) as free parameters with the anchor as a prior. B is expected to be weak at
δ = 1.
