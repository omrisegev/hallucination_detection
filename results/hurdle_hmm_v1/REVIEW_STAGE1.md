# Independent review, Stage 1 (before any real-data fit): answer-level hurdle HMM

Reviewer: independent subagent, 2026-11-02. No label array of the real bundle was read. `evaluate.py` was not run on
real data; its functions were exercised only on synthetic fake cells (`review/metrics_check.py`).
My scripts and outputs are in `results/hurdle_hmm_v1/review/`.

## Verdict: FAIL (do not run the real-data fit)

The pre-registered synthetic stop rule (PREREG section 7) **fails as written**. The PREREG says: stop and report
before any real-data fit. I traced the failure to the stage-A anchor. I also found a quadrature error that matters
at real answer lengths. The likelihood algebra itself is correct.

## 1. Synthetic stop rule (SYNTHETIC_RECOVERY.csv, complete, 18 runs)

Rule: |π̂ − π| ≤ .05 for every δ = 2 setting, median over 3 seeds.

| π | δ | π̂ by seed | median π̂ | \|median − π\| | rule |
|---|---|---|---|---|---|
| .10 | 2 | .070 / .158 / .394 | .158 | .058 | **fail** |
| .60 | 2 | .428 / .452 / .701 | .452 | .148 | **fail** |
| .95 | 2 | 1.000 / 1.000 / .951 | 1.000 (bound, expit(8)) | .050 | pass, on the bound |
| .10 | 1 | .231 / .407 / .773 | .407 | .307 | (reported only) |
| .60 | 1 | .053 / .221 / .859 | .221 | .379 | (reported only) |
| .95 | 1 | .858 / .885 / .488 | .858 | .092 | (reported only) |

Under the "every run" reading it fails worse (π̂ = .394 at π = .10). Either reading: **FAIL**.
At δ = 1, run (π .10, seed 0) ends with δ̂ = .018 = exp(−4), the lower bound.

**Cause: the anchor, not the optimizer.** (`review/synthetic_optimum_check.py`, `review/oracle_anchor_check.py`,
`review/ORACLE_ANCHOR_CHECK.txt`)

| Draw | Stage-A anchor (μ0, s0); truth (0, 1) | π̂ with the anchor | π̂ with the true clean law | NLL at truth vs at fit |
|---|---|---|---|---|
| π .10, δ 2, seed 2 | (−.127, .933) | .394 | .085 | 13.196 vs 13.159 (fit is lower) |
| π .60, δ 2, seed 0 | (+.199, 1.061) | .428 | .595 | — |
| π .60, δ 2, seed 1 | (+.142, 1.058) | .452 | .571 | — |

- The fit is the global optimum of the anchored likelihood. A restart from the true θ returns the same point.
- With the true (μ0, s0), π is recovered within .03 in all three draws.
- Error steps are < 1% of the anchor region, so contamination is small. The problem is extrapolation error.

**Anchor spread on pure clean data** (`review/anchor_variance.py`, 60 replicates each; truncated-normal MLE on the
bottom 30%):

| Steps | sd of μ̂0 (f = 0) | sd of μ̂0 (f = .3) | sd of ŝ0 |
|---|---|---|---|
| 9,000 (CRV size) | .153 | .194 | .05–.06 |
| 14,000 (synthetic size) | .114 | .124 | .04 |
| 26,000 (ProcessBench size) | .093 | .110 | .03 |
| 94,000 (PRMBench size) | .045 | .042 | .016 |

At real separations (D3: d ≈ 0.8–1.15), a μ0 error of .1–.2 is 10–25% of δ. The error state absorbs it, so π̂ moves.

## 2. Re-implementation of the likelihood and posteriors (`review/reimpl_likelihood.py`, `review/quadrature_check.py`)

Two routes written from the PREREG text, not from `hurdle_hmm.py`:
- R1: brute-force sum over all 2^n step labels (n ≤ 8), with the "at least one error" conditioning by explicit
  exclusion of y = 0 and division by 1 − (1 − e1)(1 − h)^(n−1).
- R2: a 2-state log-space forward-backward minus the all-clean path (n up to 107).
- Integration over b: (a) the same 20-node Gauss–Hermite rule, to isolate the algebra; (b) an accurate integral
  (adaptive `quad_vec`, and an 8,001-point grid over ±10τ).

**Algebra (same Gauss–Hermite rule), max |difference|:**

| Set | Answers | log-lik | P(A=1\|x) | P(y_i=1\|x) |
|---|---|---|---|---|
| R1, n = 1–8, random θ, padded batches of 5 (incl. n = 1) | 125 | 3.6e-15 | 3.0e-15 | 4.0e-15 |
| R2, n = 8–60, padded batches of 3 | 90 | 7.1e-14 | 2.0e-14 | 8.7e-14 |
| Outliers (x = ±25, 30, 40) | 3 | 1.1e-13 | 4.3e-14 | 4.2e-14 |

The anchor parameterization (σ0² = (1−f)s0², τ² = f s0²), the 3-state chain, the conditioning, P(ever error),
padding, and the posterior formula are all correct.

**Quadrature (20-node GH vs accurate integral), data drawn from the model, μ0 = 0, s0 = 1, 12 answers per row,
max |difference|:**

| f | n | log-lik | P(A=1\|x) | P(y_i=1\|x) |
|---|---|---|---|---|
| .1 | 107 | .053 | .031 | .021 |
| .3 | 15 | .006 | .003 | .002 |
| .3 | 30 | .079 | .023 | .021 |
| .3 | 60 | .340 | .131 | .114 |
| .3 | 107 | 1.37 | .280 | .241 |
| .5 | 15 | .136 | .038 | .040 |
| .5 | 30 | .565 | .121 | .113 |
| .7 | 8 | .219 | .066 | .074 |
| .7 | 30 | 1.17 | .352 | .265 |

Full table: `review/QUADRATURE_CHECK.csv`. Reason: the rule places nodes on the prior of b. For long answers the
posterior of b is much narrower than the node spacing. The log-lik error has a mean sign (negative at large f), so it
can bias f̂. Real lengths: PRMBench median 11, max 107; ProcessBench max 47; CRV GSM8K max 36.
The existing unit test uses the same 20-node rule in its brute force, so it cannot detect this.

## 3. Other checks

| Item | Result |
|---|---|
| Anchor = truncated-normal MLE on x < pooled q.30 quantile | Matches PREREG. Uses all steps (the 11-01 purity check used `step_mask` steps; minor) |
| Fit subsample | `rng(0).choice(A, min(A, 2000))`, then trimmed; anchor on all steps; scoring on all answers. Matches |
| `run_fit.py` label access | Reads only `offsets`, `step_raw_fused`, `step_z_fused`. No label key |
| `gap_counts` | Identical text to `results/error_count_label_free_v1/banks_robustness.py:9-16` |
| `topk_flags` | Same behaviour as `common_ec.topk_flags` (stable argsort, ties by position) |
| `evaluate.macro_f1` | 0 difference from my implementation, 2,000 random cases; equals `common_ec.prmscore`/100 |
| P1, P2, P3 values | max \|diff\| 1.0e-16 against mine, 480 synthetic cells |
| Benchmark and line verdict | 30/30 agree with my implementation under the "separate benchmarks" reading |
| Bundle | Unchanged since f446884e (`git diff` empty) |

## 4. Consistency with earlier findings

| Prior finding | Addressed? |
|---|---|
| Answer-z removes the answer level (T1) | Yes: raw pooled scores |
| Shared answer factor pulled answer-level DS to π ≈ .5 | Partly: b integrated out, but b is assumed independent of A. If erroneous answers have a higher level, the model must explain it with error steps. Not declared |
| Step 485: unanchored HMM error state took .40–.55 of steps | Intended fix is the anchor; section 1 shows the anchor is too noisy to hold π |
| Step 486: the gain over GAP was a prevalence effect (SHARE beat MDS) | **Ignored.** No prevalence control is planned beside P2 |
| D3 / v1 selftest: gates worked at synthetic d = 2.3–3.0, real d ≈ 1 | **Repeated.** The required synthetic regime is δ = 2; δ = 1 (the real regime) is "reported, not required", and it fails badly |
| ProcessBench post-error steps unlabelled but scored (D2) | Declared; P1 failure predicted; no mitigation |
| CRV-Arith right skew (D2) | Declared; no mitigation |
| Step 486: degenerate "almost every step is an error" fits | No validity rule or boundary report. π̂ = 1.000 (bound) already occurs in synthetic data |

## Must-fix (before any real-data fit)

1. **Stop rule fired.** Per PREREG section 7, stop and report. Any redesign is a new, dated amendment, with the
   synthetic test re-run.
2. **Anchor.** The truncated-normal tail fit has sd(μ̂0) ≈ .11–.19 at CRV/ProcessBench sizes, and this alone
   explains the synthetic failure. Fix the clean law's estimation (for example, estimate μ0 and s0 inside stage B
   with the tail fit as a start or prior, or show a lower-variance anchor), then re-test.
3. **Quadrature.** Replace the prior-centred 20-node rule with an adaptive rule (nodes centred and scaled at each
   answer's posterior of b), or show accuracy at n = 107. Add a unit test against a dense integral, not the same rule.
4. **Synthetic regime.** Require recovery at δ = 1 (d ≈ 1, the real separation in D3), not only δ = 2. Otherwise a
   pass says nothing about the real cells.
5. **`evaluate.py` scope.** It globs every `*.npz` in the fit folder. With secondary fits in the same folder, they
   enter the benchmark counts (tested: ProcessBench `n_cells` = 8, so "≥ 3 of 4" is no longer the rule). Restrict to
   the 16 PRIMARY cells and assert n_cells = 4 per benchmark.
6. **Line rule wording.** "P1 and P2 both pass on ≥ 2 of 4 benchmarks" can mean the same benchmarks or any. The code
   uses any (separately). The readings differed in 2 of 30 synthetic verdicts. Fix one reading in the PREREG.
7. **Prevalence control (Step 486).** Add, as a fixed descriptive row beside P2, a label-free control that uses only
   the share (for example, flag pooled raw scores above the 1 − ŝ quantile, ŝ = model-implied step share) and
   "flag none". On CRV (3–7% error steps) GAP flags at least one step in every answer, so any rule that flags less
   may beat GAP on macro-F1 without using the HMM.
8. **Freeze.** `PROTOCOL.json` does not exist yet. Create it (hashes of PREREG, module, scripts) before the real run.

## Should-fix

- State P3's role in the line verdict (now computed, unused).
- Declare a validity/boundary report: δ, σ1, f, π at bounds; `converged`. Add them to SYNTHETIC_RECOVERY.csv.
- PREREG says TELE16_sa is "the only PRMBench cells in the bundle"; `SUB5_sa_prmbench_*` also exist. Justify TELE16.
- PRMBench bootstrap should resample source groups (as in Steps 485-486), not answers.
- `run_fit.py` skips cells whose output exists; check the protocol hash instead of silently reusing old fits.
- `maxiter=300`, the bounds, and `np.round` (round-half-to-even) for HH-K are not in the PREREG; list them.
- Compare synthetic π̂ to the realised share of A = 1 as well as to π (realised .123 at π = .10, seed 2).

## Files

- `review/reimpl_likelihood.py`, `review/REIMPL_LIKELIHOOD.csv`: algebra and quadrature comparison.
- `review/quadrature_check.py`, `review/QUADRATURE_CHECK.csv`: GH accuracy on model data by f and n.
- `review/anchor_variance.py`: anchor spread on clean data.
- `review/synthetic_optimum_check.py`, `review/oracle_anchor_check.py`, `review/ORACLE_ANCHOR_CHECK.txt`: attribution.
- `review/metrics_check.py`: macro-F1, endpoints, verdict, and the `evaluate.py` scope test, on synthetic cells.
