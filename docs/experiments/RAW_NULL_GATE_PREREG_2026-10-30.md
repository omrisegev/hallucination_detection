# Pre-registration: label-free clean-answer gate from raw step scores (empirical null, SML/L-SML, answer-level DS)

Date: 2026-10-30, frozen before any real-data run. Branch `claude/raw-null-gate-2026-10-30`, built on
`claude/ds-gate-2026-10-29`.
- **Code:** `spectral_utils/raw_null_gate.py`, `results/raw_null_gate_v1/run_raw_null_gate.py`.
- **Independent verifier:** `results/raw_null_gate_v1/verify_raw_null_gate.py`.
- **Literature for each idea:** `docs/research_notes/RAW_NULL_GATE_LITERATURE_2026-10-30.md`.
- **How to run:** `docs/handoffs/RAW_NULL_GATE_RUNBOOK_2026-10-30.md`.

**Who runs it.** The run is on the other machine, where the raw data are. The file hashes in
`results/raw_null_gate_v1/PROTOCOL.json` freeze the code, and the runner refuses to run if they change.

## 1. Question and why

**The question.** GAP always flags at least one step, so it loses on clean answers. Can a label-free rule decide that
an answer is clean (flag nothing), using:
- **AUTO:** the PRM plus 16 channels, backbone;
- **TELE16:** 16 channels, no PRM, on the backbone and on standalone telemetry;
- **SUB5:** the 6-channel set, no PRM, on the backbone and on standalone telemetry?

**Omri's emphasis.** The tendency should be TELE16 or SUB5, even without the PRM. The SML / L-SML estimates should
enter the answer-level decision, not only the fusion.

**Why earlier attempts failed.**
- **Binary GAP votes and per-answer z-scores remove each answer's own level** (DS-gate E1, E1 fixed, E1b on
  `claude/ds-gate-2026-10-29`). A clean answer then looks like an erroneous one.
- **A two-component mixture on answer scores invents a second group** on heavy-tailed data (Step 484).
- **The share of erroneous answers is not identified from second moments of binary votes** (FRAMING_THEORY P6).

**What is new here.** The rule never estimates that share directly. It learns what a CORRECT step looks like, from
the centre of the raw step-score distribution, where almost every step is correct: Efron's empirical null. It then
asks whether the answer's worst step is still plausible as a correct step.

## 2. Data, cells and inputs (no new data)

**Arms.**
- AUTO, backbone only;
- TELE16, backbone and standalone;
- SUB5, backbone and standalone.

**Benchmarks:** prmbench, processbench, crv_arith and crv_gsm8k.

**PRMs:** qwen_prm_7b, qwen_prm_72b, pure_prm and rlhflow_prm. That is 5 × 4 × 4 = 80 cells. Raw PRMBench backbone
banks exist only on AIRCC, so those cells are reported as `missing`. That is expected and not a failure.

**Loaders, imported read-only from the other machine's code.**
- `bracha_lr_transfer_v1/lr_transfer.load(bench, prm, mode)` loads the cell.
- `E.channels(c, 'N16')` gives the stored answer-z channels in stored orientation.
- `ds_gate_e1b_fuse/votes_e1b.raw_matrix` gives the raw channels. It asserts that answer-standardizing them
  reproduces the stored channels to 1e-4.
- `votes_e1b.clean` gives the raw channels pooled-z-scored with a pooled cubic in relative position removed, which is
  Step 481's procedure. This keeps each answer's level. NaN marks undefined digit steps. It is replaced by 0 for the
  fused readout (3.2), and it defines each group's defined steps for G2/G3 (3.2a).
- The PRM risk is `c.arms['PRM_Q80']` on PRMBench and `c.arms['PRM']` elsewhere.

**Labels.**
- From `run_e1.run_cell`'s own definitions:
  - `keep_gold` on processbench, else `c.keep`;
  - an answer has an error if any kept step is wrong;
  - on PRMBench, answers of class `correct` are not evaluated.
- The metric is `run_e1.metric` (PRMScore, ProcessBench F1, CRV macro-F1).
- **Labels are read only after the label-free stage of each cell has been saved to disk.**

## 3. Algorithm (exact; the verifier re-implements it from this section)

**3.1 The arm's fused score (answer-z), unchanged.**
- **AUTO:** `lsml_continuous_guarded` on [z(PRM risk), N16]. The sign is set by the correlation with column 0.
- **TELE16 / SUB5:** the same on the channels, with the sign set by the correlation with their row mean.
- **Parity, required:** this must equal `stage8_auto_forced_k.auto` / `.fit` to 1e-6. Where the arm exists in
  `run_e1.arm_inputs` (AUTO backbone, TELE16/SUB5 standalone), it must also equal that output to 1e-6.
- **As a linear map:** group g has indices I_g and weights w_g (a singleton group uses the column itself), with cross
  weights c_g and sign σ, so that fused = σ Σ_g c_g (X[:, I_g] · w_g). The rebuilt score must equal the fit to 1e-6.
- **Group orientation:** o_g = σ·sign(c_g). With a single group, o = σ.

**3.2 Raw readout.** The same map is applied to the cleaned raw channels Z (AUTO: [cleaned PRM risk, Z]):
- raw_fused = pooled-z(σ Σ_g c_g (Z[:, I_g] · w_g));
- raw group score u_g = o_g · pooled-z(Z[:, I_g] · w_g).

Here pooled-z means (x − mean)/sd over all steps of the cell, with sd ≤ 1e-12 giving 0.

**3.2a Defined steps.** A step is defined for group g if at least one of the group's channels is defined there, i.e.
not NaN in `votes_e1b.clean` (the digit channels on digit-inactive steps). The PRM column is always defined. For the
per-group gates (G2, G3), u_g is the pooled z over the group's defined steps only, with undefined steps set to 0, and
each group's null is fitted on its defined steps only. Undefined steps get lfdr = 1, P(err) = 0 and LR = 1.

**Why:** a spike of exact zeros at the centre is read by central matching as a narrow null. In a synthetic group with
no errors and 30% zeros, 48% of answers voted "abnormal" (review finding 3).

**3.3 Empirical null (Efron's two-groups model), for any score x over all steps of a cell.**
1. **Range:** lo and hi are the .0005 and .9995 quantiles of x. The histogram has 120 equal bins on [lo, hi], with
   values clipped into [lo, hi].
2. **Density f, by Lindsey's method:** a Poisson regression of the bin counts on a Legendre basis of degree 7.
   - **Basis variable:** t = clip(2(z − lo)/(hi − lo) − 1, −1, 1), at every evaluation point. So log f is held at its
     edge value outside [lo, hi].
   - **Fit:** unpenalized IRLS from β₀ = log(mean count), with at most 100 iterations and tolerance 1e-10 on
     max|Δβ|. The linear predictor is clipped to [−50, 50], and μ is floored at 1e-12 in the working response. **If IRLS has
     not converged within 100 iterations, 3.3 fails** (verifier finding: on degenerate inputs the unconverged β is
     arbitrary).
   - **Density:** log f(z) = basis(z)·β − log(N·bin width).
3. **Central matching:** fit a quadratic, by least squares, to log f at the bin midpoints lying between the .25 and
   .75 quantiles of x. At least 5 such bins are needed.
   - The leading coefficient b₂ must be < 0; otherwise the null is not identified.
   - σ₀ = √(−1/(2b₂)), μ₀ = b₁σ₀², π₀_raw = exp(b₀ + μ₀²/(2σ₀²))·σ₀·√(2π), π₀ = clip(π₀_raw, .5, .995).
4. **lfdr(z)** = clip(π₀·φ(z; μ₀, σ₀)/f(z), 0, 1).
5. **P(err | z)** = 1 − lfdr(z) if z > μ₀, else 0. This is one-sided: an unusually low risk is not an error.

**3.4 Gates.** In every gate, "clean" means the answer gets no flag.

| Gate | Statistic | Clean if |
|---|---|---|
| **G1** (fused) | 3.3 on raw_fused. The worst step w(a) is the argmax of the answer-z fused score within answer a (first on ties), i.e. the step GAP flags first | P(err \| raw_fused[w(a)]) < .5. If 3.3 fails, no answer is clean |
| **G1b** (conformal) | The reference set is the raw_fused of every step not flagged by GAP. p = (1 + #{ref ≥ raw_fused[w(a)]})/(1 + #ref); p_ans = 1 − (1 − p)^{n_a} (Šidák over the answer's n_a steps) | p_ans > .05 |
| **G2** (SML / L-SML product) | For each group with o_g ≠ 0, 3.3 on u_g, computed and fitted on the group's **defined steps** only (3.2a); groups where 3.3 fails, or with fewer than 50 defined steps, are dropped. The same rule applies in G3. π₀ = clip(median of the groups' π₀, .5, .995). Per group, LR_g = clip((π₀/(1 − π₀))(1/max(lfdr_g, 1e-12) − 1), 1e-3, 1e3) where u_g > μ₀_g on a defined step; LR_g = 1 otherwise (**one-sided**, like 3.3.5). Step log-odds = log((1 − π₀)/π₀) + Σ_g log LR_g, and P(err) = logistic(log-odds) | max_a P(err) < .5. If no group fits, no answer is clean |
| **G3** (answer-level Dawid–Skene) | For each group where 3.3 succeeds (defined steps only, as in G2), the vote v_ag = +1 if max over the answer's defined steps of P(err \| u_g) ≥ .5, else −1. Needs at least 3 votes per answer, otherwise the gate is invalid and no answer is clean. Two-coin DS-EM over the answers × groups votes (`prm_anchored_lsml.ds_em_votes`, init 'majority'; anchor = the group containing the PRM for AUTO, else the majority label switch). P(answer has an error) = Bayes' rule with the fitted ψ, η and prevalence. Also reported: the prevalence, ψ, η, and the SML ranking vector (the leading eigenvector of the zero-diagonal vote covariance, sum ≥ 0) | P(answer has an error) < .5 |

**Where SML theory enters:**
- In G2, the step posterior is the product of the groups' likelihood ratios. That is exact under L-SML's model, in
  which groups are conditionally independent given the label (Jaffe et al. 2016).
- In G3, the answer-level gate is itself a Dawid–Skene / SML ensemble of the groups.

**3.5 Flags.** Plain GAP flags of the answer-z fused score (`stage7_hygiene.gap_flags`), set to none in clean
answers.

## 4. Endpoints, per cell and gate (labels only score)

1. **closed_clean:** the share of clean evaluated answers the gate calls clean.
2. **kept_error:** the share of erroneous evaluated answers it does not call clean.
3. **share_called_error** against **true_share_error**.
4. **metric:** the gated metric against GAP_ONLY.
5. **ece_step:** for G1 and G2, the step calibration of P(err) against the step labels on evaluated steps. There are
   10 bins by rank: bin = ⌊rank·10/n⌋ with stable ranking, as in `run_e1`'s ECE, and ECE = Σ_bins share·|mean P(err) −
   error rate|.
6. **Label-free diagnostics:** π₀, μ₀ and σ₀; per-group π₀ (over-identification: under the model they should agree);
   the G3 prevalence, ψ, η and SML ranking.

## 5. Pass rule (fixed now), per (arm, mode, gate)

An (arm, mode, gate) **passes** if:
- **on each of processbench, crv_arith and crv_gsm8k,** in at least 3 of 4 PRMs: closed_clean ≥ .30 and
  kept_error ≥ .85, with the gate valid;
- **and on PRMBench,** where cells exist: the gated metric ≥ GAP_ONLY − 0.5 in at least 3 of the cells.

**How to read the outcome.**
- **The priority** is TELE16 and SUB5 (no PRM). AUTO is reported alongside.
- **Multiplicity:** there are 20 (arm, mode, gate) tests, and VERDICT.json lists all of them. A single pass among 20
  is reported as such, not as "the method works".
- **Secondary readings, not part of the rule:** |share_called_error − true_share_error|, the metric deltas, and ECE.
- **Nothing is tuned on outputs.** Thresholds (.5, .05), bins, degree and quantiles are fixed here.

## 6. Verification (Omri: subagents)

**`check` runs the whole label-free stage on every cell**, so a crash appears before `run`. Missing raw data must be
exactly the 12 expected PRMBench backbone cells; any other missing file is a FAIL, because wrong paths look like
missing files. In VERDICT.json, a benchmark with fewer than 4 cells makes the entry `incomplete`, never a `fail`.
`run` writes RESULTS.csv after every cell.

**Before the run (done on this machine):**
- an independent code review of the module and runner against this section;
- unit tests and a synthetic self-test (`run_raw_null_gate.py selftest`).

**After the run (other machine):** `verify_raw_null_gate.py`, written independently from Section 3 without reading
the module. It recomputes every row of RESULTS.csv from the same loaders and must match:
- the rates exactly;
- π₀ and the metrics to 1e-6.

Any mismatch stops the analysis.

## 6a. Amendments before the freeze (2026-10-30, after the independent code review, no real data run)

The independent code review found these, and they are fixed in this text and in the code:
1. **Line endings:** the Windows checkout converts line endings. Hashes are now computed with CRLF normalised to LF,
   and `.gitattributes` pins LF for the module, the library files and the test file.
2. **Missing data:** an unexpected missing file is a FAIL, not `missing`.
3. **Defined steps per group** (3.2a).
4. **One-sided G2.**
5. **Empirical-null details** stated explicitly (3.3 step 2).
6. **Hash coverage:** the library files `fusion_utils.py`, `prm_anchored_lsml.py` and the test file are hashed, and
   a missing frozen file stops the run.
7. **Robustness:** `check` runs the label-free stage, and `run` writes incrementally and refuses a CHECK.json made
   with different code.
8. **Diagnostics:** FITS.json stores the orientations, cross weights, groups used, per-group μ₀/σ₀/π₀, the share
   of defined steps, and whether the G3 anchor was used.
9. **Infinite values:** infinite raw values stop the run.

**What the review also showed, and what was not changed:**
- **Skewed correct-step scores make the gate over-call abnormal steps.** On a lognormal score with no errors, 80% of
  answers are called abnormal. This is a known property of a Gaussian empirical null; it stays a limit (Section 7).

## 7. Limits stated in advance

- **These benchmarks were inspected before:** ProcessBench and CRV, by the DS-gate experiments.
- **The zero assumption** (the centre of the distribution is correct steps only) is an assumption.
  - Heavy-tailed or right-skewed correct-step scores produce false "abnormal" steps.
  - In review simulations with no errors, a t₃ score called 19% of answers abnormal, and a lognormal score 80%.
  - Surprisal and entropy channels are right-skewed, so G1 on raw TELE16 may close few clean answers for this reason
    alone.
- **At weak separation,** a P(err) < .5 cut keeps only about half of the erroneous answers. This is shown in
  synthetic data: fused-score AUC about .86.
- **The answer-level nuisance** (problem difficulty) stays in the raw scores. Only position is removed, as in
  Step 481.
