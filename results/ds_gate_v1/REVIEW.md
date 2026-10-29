# Independent review of ds_gate_v1 stage E1 (2026-10-29)

Scope: PROTOCOL.json, ds_gate.py, e1_check.py, test_synthetic.py. e1_check.py was NOT run. Checks run: the
synthetic test (passes), a brute-force E-step check, a monotonicity/bias check with an exact (a, b) M-step, and
label-free data checks (answer lengths, PRM score orientation vs reward / native flag, CRV `include` mask).
Throwaway scripts: scratchpad `chk1.py`-`chk4.py` (not in the repo).

## What is correct (confirmed)

- E-step likelihood of a has-error answer: sum over y with >= 1 error of prod pi^y (1-pi)^(1-y) L(y) equals
  prod_i(pi_i e^l1 + (1-pi_i) e^le) - prod_i (1-pi_i) e^le = e^M - e^N; dividing by 1 - prod(1-pi) is right.
- Step marginal under the constraint: {y_i = 1} implies {>= 1 error}, so P(y_i=1 | >=1, v) = P_unc(y_i=1 | v) /
  P_unc(>=1 | v) = P_unc * e^M / (e^M - e^N). The prior normalizer cancels. Code line 89 is right.
- q = P(has-error | v) * P(y=1 | has-error, v) (line 91) is right.
- M-step expected counts: E[z y] = q, E[z(1-y)] = W - q, E[1-z] = 1 - W; rho, psi, phi, phi0 updates (lines 97-102)
  are the correct closed forms. Newton for (a, b): gradient X'(q - W mu), negative Hessian X' diag(W mu(1-mu)) X:
  right for the objective WITHOUT the normalizer.
- Brute-force enumeration on 200 random small cells (1-5 steps, K=4): max |difference| in log-likelihood, P(has-error)
  and q = 2.1e-14.
- Labels: c.err, c.keep_gold, c.cls are read only after fit_hds (e1 lines 44-47). The AUTO reference is label-free.
- Populations: ProcessBench fit pool = every step (e1 line 32; sub5_all.load already sets keep = ones and keeps the
  prefix in keep_gold); evaluation on keep_gold. CRV: `include` is True on 100% of steps in both gold files
  (1,500 / 1,319 answers), so c.keep is label-free there. PRMBench evaluated = cls != 'correct'.
- PRM orientation (label-free check): Spearman(PRM_Q80, reward) = -1.000 on all 4 PRMBench PRMs;
  corr(PRM_Q80, native_error_flag) = +0.47 to +0.64; on the 12 ProcessBench/CRV cells corr('PRM', native invalid)
  = +0.27 to +0.68. Higher = more likely error holds everywhere. The AUTO reference uses the same logit arm
  (stage8 common.prm_arm), as in auto_new.py.

## Findings

### B1 - blocker (needs owner decision): the (a, b) M-step ignores the normalizer -> not a GEM, biased fixed point
ds_gate.py lines 103-109 (and PROTOCOL step3.fit, which froze this choice). Dropping
-sum_a w_a log(1 - prod(1-pi)) from Q means the update does not guarantee Q increases, so EM can decrease the
likelihood and converges to a point that is not the MLE: the logistic fit tracks the rate CONDITIONED on >= 1 error,
which overstates pi, and rho/psi/calibration absorb the error.
Measured on synthetic data from the model (same EM, start 0, chk2.py):

| data | (a,b) M-step | likelihood decreases | final loglik | rho (truth .65) | a, b (truth -2.5, 2.0) | ECE |
|---|---|---|---|---|---|---|
| test_synthetic default (3-15 steps) | as coded | 3 (worst -3.44 nats) | -83102.1 | .626 | -2.27, 2.13 | .0227 |
| same | with normalizer | 0 | -83026.1 | .640 | -2.50, 2.03 | .0106 |
| 1-4 steps | as coded | 0 | -26662.1 | .594 | -1.40, 3.01 | .0472 |
| same | with normalizer | 0 | -26219.1 | .636 | -2.51, 2.03 | .0095 |

Real cells: median 6-11 steps, 0.5-5.2% of answers have <= 3 steps, so they resemble the default row: the
implementation alone costs ~.01-.025 of the .05 D1 budget and doubles ECE against the .05 D3 bar. E1 would partly
test the shortcut, not the model. Also "likelihood monitored" (protocol) is not implemented: no trace or count of
decreases is kept or reported (fit_hds lines 126-131).
Fix: maximise the full Q in (a, b): add the term with gradient -w_a * P0_a/(1-P0_a) * sum_{j in a} pi_j x_j
(P0_a = prod(1-pi_j)); 2 parameters, so Newton with a numeric Hessian, L-BFGS or Nelder-Mead is cheap (chk2.py uses
Nelder-Mead). Record the log-likelihood trace and assert non-decrease (e.g. ll_t >= ll_{t-1} - 1e-6 * n) in
fit_hds; add that assert and an (a, b) recovery check to test_synthetic.py (it currently prints a, b but does not
test them). This changes a frozen protocol line, so it needs Omri's explicit amendment before E1.

### S1 - should-fix: numerical floor in log(e^M - e^N)
ds_gate.py line 79 (and 89). The floor min(N - M, -1e-12) is wrong when P(>= 1 error | has-error, votes) < 1e-12.
Extreme case (psi = 1-1e-4, all-zero votes): code P(has-error) = 1.7e-10 vs exact 8.5e-24; q given has-error
5e-14 vs exact 1.0 (one-step), .5, .33. Practical impact on the real cells is probably small (GAP gives every voter
>= 1 flag per answer, no one-step answers), but the fix is cheap and verified to match brute force (chk4.py):
D = sum_i logaddexp(0, t1 - t0) (= M - N >= 0); lnum = N + log(expm1(D)) (use D + log1p(-exp(-D)) for large D);
corr = 1 / (-expm1(-D)); lnorm = log(-expm1(Lz0)).

### S2 - should-fix: verdict passes vacuously when cells are invalid or missing
e1_check.py lines 84-85. The PRMBench part of D4 (`all(... for r in get('prmbench'))`) and D5
(`all(... for r in rows)`) only see valid cells; a K < 3 cell silently drops out and an empty list passes.
D1/D3/D4-ProcessBench are safe (counts fall). Fix: pass the fits list to verdict, count an invalid cell as a fail
for every criterion that covers it, and write n_invalid per benchmark into E1_VERDICT.json.

### S3 - should-fix (owner to settle the reading): D5 length reference
e1_check.py lines 23, 85. The protocol says "below that of every past estimator (>= 0.78 for Q80)". The code
compares only with the constant .78. The GAP count is a past estimator, and its per-cell length Spearman is already
computed (line 71) but unused. The literal reading is |rho_s(sum q, n)| < min(.78, |rho_s(GAP count, n)|) per
cell. Also, the .78 could not be traced: results/count_structure_check_v1/README.md has no such number and there is
no OUTPUT.txt. It is a PRMBench number being applied to ProcessBench and CRV cells. Cite the source and document
the reading in the verdict.

### S4 - should-fix (declare, or remove): PRMBench pool uses the control classification
e1_check.py lines 32-33: pool = c.keep & (len > 1); on PRMBench c.keep = (cls != 'correct'), which drops the 758
unperturbed originals. Those are exactly the answers known to be clean, so this is label-correlated metadata.
Judgment: this is not a leak into the EM fit. fit_hds sees all 6,969 answers, and the evaluated set is chosen after
the fit. The metadata only affects the step-0 cubic, the orientation signs and the vote covariance used for
grouping, so the effect is probably small. The project has precedent for it: the stage1 pool, and the sub5_all
docstring declares it. Acceptable only if declared. The protocol calls step 0 "label-free", however, and
pool = (len > 1) over all answers costs nothing and makes the access genuinely label-free on every benchmark. Prefer
that, or state the exception in the result.

### Notes
- N1 PRM scale: e1 uses the logit arm (PRM_Q80 / PRM) for every PRM. The Step 481 probe (via stage1.prm_arm) used
  the 1-reward scale for qwen_prm_7b, qwen_prm_72b and pure_prm (STATUS_META), and the project's 'PRM' GAP rows use
  that scale too. GAP flags change under this nonlinear transform, so the PRM votes and signs are not those of
  Step 481. e1 is internally consistent with the AUTO reference, so the fix is to declare it.
- N2 Constant mask (ds_gate line 35): it is computed on the raw channels, while the probe computed it on the cleaned
  ones. ds_gate follows the protocol text better: a raw-constant channel otherwise votes on the subtracted cubic,
  which is pure position. Column 0 (the PRM) is never masked: if the PRM is constant inside an answer,
  GAP k = 1 and the stable sort flags step 1. Mask it as well, or count how often it happens.
- N3 ds_gate line 32: if a pooled correlation is NaN, np.sign gives NaN, `sign == 0` does not catch it, and the
  whole column becomes NaN before topk_flags. Assert that the values are finite.
- N4 ECE bins (e1 line 53; also test_synthetic): quantile edges plus digitize do not give equal-count bins when
  P(has-error) ties or saturates near 0/1. The protocol asks for 10 equal-count bins. Use
  np.array_split(np.argsort(w, kind='stable'), 10).
- N5 fit_hds line 130 stops on |delta ll| / n_steps < tol. A decrease can also stop it, which matters only while B1
  stands. The "mean log-likelihood" is per step, not per answer; state which.
- N6 PRMBench: the protocol's D1 text compares rho_hat with the share of EVALUATED answers that have an error
  (about 1.0), but rho is fitted on all 6,969 answers. The code reports both, and the verdict excludes PRMBench
  from D1/D3, so this is fine. Few or no evaluated answers are clean, so D3 and closed-clean there are near
  degenerate (NaN-safe: a NaN fails). The PRMBench part of D4 is implemented at threshold rho_hat and required on
  all 4 PRMs. The protocol states neither the threshold nor the count, so document the reading.
- N7 D1 and D3 read "ProcessBench and CRV" as three benchmarks (PB, crv_arith, crv_gsm8k), each needing >= 3 of 4.
  That is a reasonable reading; document it.
- N8 ProcessBench D5: the true count covers only the prefix (0/1), while sum q covers the whole answer. This is
  consistent with the protocol but should be interpreted as a has-error correlation, not a count correlation.
- N9 The label-switch check compares only psi with phi. Also report rho at its clip bounds, and mean(phi0) > mean(psi)
  (an answer-level swap or degenerate state), in E1_FITS.json.
- N10 The m_step Newton update uses an unclipped pi while e_step clips it, and has no step-halving. This is minor.
  An exact-objective fix (B1) should evaluate the same clipped pi.

Verdict: fix first. B1 needs an approved protocol amendment (full (a, b) M-step plus a monotonicity assert). Also fix S2, and settle S3 and S4 before running E1.
