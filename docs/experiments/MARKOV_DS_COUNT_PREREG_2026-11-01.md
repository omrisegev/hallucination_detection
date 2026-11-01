# Pre-registration: how many steps are wrong — a multi-voter Dawid-Skene count with a Markov label chain

Date: 2026-11-01. Written before any labelled metric of this study. Branch `claude/raw-null-gate-2026-10-30` (local).
**Pre-label stop (2026-11-01, same day).** The section 5 synthetic recovery test failed for the approximation in
sections 3.3 and 3.4 (one configuration per k). The exact sum over configurations passes it. Nothing is frozen and
no real cell has been fitted. Record: `results/markov_ds_count_v1/prelabel_synthetic/PRELABEL_STOP.md`.
Omri chose the exact model: **Amendment A** at the end of this file replaces sections 3.3–3.4 where they conflict.

Status: **declared, not run.** No code for this study exists yet. Code and tests will be frozen by file hashes in
`results/markov_ds_count_v1/PROTOCOL.json` before any label is read. Anything added after labels are read is marked
POST HOC.

## 1. Question and why

**Omri's question (2026-11-01).** Can the statistics of several weak voters say not only *where* the errors are, but
*how many* wrong steps an answer has? This is part 2 of the two-part (hurdle) design that came out of the adversarial
review of the same day: given that an answer has at least one error, how many does it have? Part 1, whether there is
any error at all, belongs to the raw-null gate and is not tested here.

**Why this study, and not the neighbor readout again.** The neighbor readout study (Step 485,
`results/neighbor_readout_v1/`) fitted a Markov chain on ONE fused score. With one score there is no agreement
between voters to exploit. The model reduced to a one-dimensional Gaussian mixture, found "the upper half of the
scores" (implied share .40–.55 against a true .158 on PRMBench) and, on Socratic, learned position instead of errors.
That did not test the Dawid-Skene idea. Here the voters are the arm's own conditionally independent groups.

**What is already closed, so it is not repeated.**
- **Multi-voter DS with independent steps and binary votes** (`results/label_free_error_count_v1_2026-10-12/`, A–D
  family). It failed. The prevalence followed the vote quota (10/20/30% quotas gave 13.6/22.9/32.6%).
- **The DS gate with a position prior** (`results/ds_gate_v1b_fixed/`, `results/ds_gate_e1b_fuse/`). It failed. The
  answer-level prevalence came out at .85–1.0 everywhere.
- **The single-score Markov count** (H1b in Step 485). It failed in the frozen arms. In the in-scope arms it had a
  partial positive: SUB5 on Socratic Qwen-7B, which is plausibly explained by the higher true error share there.

**What is new here: two ingredients, each with its own control.**
1. **Time.** The error labels form a Markov chain, as in Traganitis and Giannakis (arXiv 1906.09356). Errors come in
   runs: on PRMBench P(error | previous error) = .417, against .123 after a correct step.
2. **An exact likelihood for answer-normalized inputs.** This is a finding from preparing this study. The stored
   telemetry channels are answer-z: mean 0 and SD 1 inside every answer. I checked this on Socratic and PRMBench
   Qwen-7B: the SD of the per-answer means is 0, and the per-answer SD is 1 in all 16 channels.
   - **Why it matters.** After answer-z, an error step's value depends on how many errors the answer has. One error
     in 10 steps sits near +3. Five errors in 10 sit near +1.
   - **Consequence.** A pooled Gaussian emission, as in the neighbor H1 and the earlier DS-HMMs, is wrong in exactly
     the direction that decides the count.
   - **Fix.** Section 3.2 models the answer-z vector as the maximal invariant of an unknown per-answer location and
     scale. That gives an exact likelihood for k.

**Theory link (thesis line).**
- **Voters.** The voters are the arm's L-SML groups. The premise of L-SML (Jaffe et al. 2016) is that groups are
  conditionally independent given the label, with the dependence absorbed inside groups. That is the unit at which
  the DS conditional independence is claimed.
- **Identifiability.** With three or more conditionally independent voters, the class-conditional distributions are
  generically identifiable (Allman, Matias and Rhodes 2009). A single voter is identified only by the Gaussian shape.
- **What this study tests.** The thesis claim from the review is that agreement identifies voter quality but not
  error quantity, and that counting needs an anchor. The question here is whether time structure plus the exact
  invariance supply that anchor. A failure is reported as evidence for the claim, not hidden.

## 2. Data, cells, inputs (no new inference)

**Loader.** `results/prm_boost_collection_v1/evaluate_collection.py`: `E.load`, `E.channels(c, 'N16')`,
`E.answer_z`, `E.gap_counts`, `E.topk_flags`, `E.official` / `E.measure`, `E.boot_weights`. All are used unchanged.

**Benchmarks.**
- **socratic:** 2,995 answers, 26,052 steps. This is the **judging benchmark**: the count varies (true mean 2.98,
  step share .343).
- **hard2verify:** 181–200 answers. Descriptive only.
- **prmbench:** 6,969 answers. Development. The true count is nearly constant (mean 2.12, by construction), so the
  constant K2 is a strong reference there.

**PRMs.** qwen_prm_7b, qwen_prm_72b, pure_prm, rlhflow_prm.

**Arms.** These are the in-scope arms. CHAIN is out of scope and is not run.
- **AUTO = `S8.auto(N16, answer_z(PRM_Q80 | PRM))`** (`results/prm_boost_math_rigor_v1/stage8_auto_forced_k.py`).
  The voters are the K virtual classifiers `meta['virtual_classifiers']` of the same L-SML fit.
- **TELE16 = `S8.fit(N16, anchor=mean)`.** The voters are its virtual classifiers.
  - Checked for AUTO and TELE16: this route reproduces the stored `c.arms['AUTO']` and `c.arms['TELE_N16']` with
    |corr| = 1.0 on Socratic Qwen-7B and PRMBench PURE.
- **SUB5 = `stage2.TeleCell.evaluate(SUB5_IDX)`** (`results/prm_view_selection_v1/stage2/stage2.py`).
  - The voters are its K group components `comps[g] = ans.z(Z[:, idx_g] @ w_g)`. The code returns `groups` and
    `between`; the components are recomputed with the same arithmetic.
  - **Parity gate:** `Σ_g wb_g · comps_g` must equal the stored `fused` to 1e-10.
  - The S8 route is NOT used for SUB5. It gives K = 2 and |corr| = .75 with the stored score on Socratic Qwen-7B.

**Voter orientation.** Each voter is answer-z'd and then oriented so that its correlation with the arm's fused score
is positive. This is label-free; the signs are recorded.

**Validity of the arm.** If an arm has K < 3 voters in a cell, the cell is reported as INVALID (no identifiability
claim) and falls back to GAP.

**Cells.** 3 benchmarks × 4 PRMs × 3 arms = 36.

**Labels** (`c.err`, `c.keep`) are read only by the evaluation stage, after every label-free fit is on disk.

## 3. The model (exact)

**Notation, for one answer.**
- n = the number of steps.
- cap = ceil(n/2). This is GAP's cap; the same cap applies to every rule here.
- z_g = the answer-z values of voter g (g = 1..K).
- u_g = z_g / ||z_g||. This is a unit vector, and Σ_t u_gt = 0.
- f = the arm's fused score. order = the steps sorted by f, descending.
- y_k = the configuration that flags the top k steps of `order`. There is one configuration per k, the same ranking
  that GAP and ORACLE use.

### 3.1 Generative assumptions (stated so they can be attacked)

- **Labels.** The labels y_1..y_n follow a two-state Markov chain:
  - P(y_1 = 1) = e;
  - P(y_t = 1 | y_{t−1} = 0) = e;
  - P(y_t = 1 | y_{t−1} = 1) = ρ.
  - We condition on at least one error.
- **Voters.** Each voter g has a raw score x_gt = a_gi + b_gi · (δ_g · y_t + ε_gt).
  - ε is N(0, 1), independent across steps and across voters given y. This is the DS conditional independence, at
    group level.
  - a_gi and b_gi > 0 are a free location and scale for each answer and voter.
  - δ_g ≥ 0 is the voter's standardized separation, shared across answers. It is the continuous analogue of DS's
    per-voter accuracy.

### 3.2 Invariant likelihood

The answer-z vector is the maximal invariant under (a, b). Its direction u_g has the projected-normal density on the
sphere of the centered subspace, with dimension d = n − 1 and mean vector m = δ_g (y − ȳ). Because Σ_t u_gt = 0:

ℓ_g(y) = −δ_g² · k(n−k) / (2n) + log J_{n−1}( δ_g · Σ_{t: y_t = 1} u_gt ) + const(n),

where:
- J_d(τ) = ∫_0^∞ r^{d−1} exp(−r²/2 + rτ) dr;
- J_1(τ) = √(2π) · e^{τ²/2} · Φ(τ);
- J_2(τ) = 1 + τ · J_1(τ);
- J_{d+1}(τ) = τ · J_d(τ) + (d−1) · J_{d−1}(τ).

**Checked before writing.** The recursion matches numerical quadrature to six decimals for (d, τ) from (1, .3) to
(20, 6).

**The terms.** The first term penalizes splits that would need a large separation. The second rewards alignment of the
voter with the flagged set. Together they carry information about k that answer-z does not destroy.

### 3.3 The count

**Score for each k.** For k = 1..cap:

L(k) = log P(y_k) + Σ_g ℓ_g(y_k),

where P(y_k) is the Markov chain probability of the configuration y_k.

**Posterior over k.** π(k) ∝ exp L(k) over k = 1..cap.

**The count k̂.** The posterior median. The primary metric is the absolute count error, which the posterior median
minimizes.

**Flags.** The top k̂ steps by f. The ranking is the arm's own, so any PRMScore difference from GAP comes from the
count alone.

**Answers with n ≤ 2.** k̂ = 1, because cap = 1; every rule agrees there.

**Declared approximation.** One configuration per k, the top k by f, stands in for the sum over all configurations
with k errors. This is a Viterbi-type approximation. It ignores how many configurations share the same k.

### 3.4 Fitting (label-free, pooled per cell)

**Parameters.** θ = (δ_1..δ_K, e, ρ).

**Objective.** Maximize Σ_i log Σ_k exp L_i(k) over the answers of the cell with n ≥ 3.

**Optimizer.**
- L-BFGS on the unconstrained parameters log δ, logit e and logit ρ.
- Five deterministic starts, with δ_g in {.5, 1, 2} × the voter's pooled top-quartile versus rest standardized gap,
  and (e, ρ) ∈ {(.1, .4), (.2, .2), (.15, .6), (.3, .5), (.05, .3)}.
- The best objective is kept.

**Validity.** A fit is VALID only if all of the following hold. Otherwise the cell falls back to GAP and is counted
as a failure.
- The optimizer converged.
- All δ_g ∈ [.05, 10].
- e and ρ ∈ [.001, .999].
- The implied step share ŝ = Σ_i E[k_i] / Σ_i n_i ∈ [.01, .60].

### 3.5 Candidates and controls (all with the same count rule, flags and cap)

| name | voters | labels | what it isolates |
|---|---|---|---|
| **MDS** (primary) | the K group voters | Markov (e, ρ) | the full model |
| **IND** | the K group voters | independent: ρ fixed to e | the value of the Markov chain |
| **SINGLE** | one voter: the fused f, answer-z | Markov | the value of several voters |
| **SHARE** | none | none | count = clip(round(ŝ_MDS · n), 1, cap): MDS's pooled share only, with no answer-specific information. If MDS is invalid in a cell, SHARE is undefined there, and E1 counts that cell as a failure. |

**References, unchanged:**
- GAP;
- K2 (k = min(2, n));
- ORACLE (the true count, ceiling only).

**Historical reference, read from file, not refitted.** The neighbor study's H1b for the same arm, as a count from
`results/neighbor_readout_v1/fits*/`.

## 4. Endpoints and decision rules (declared)

Every decision is made separately per arm. A success in one arm is reported as a partial success, never averaged
across arms, PRMs or benchmarks.

**Metrics for each cell.**
- The count MAE on answers with at least one kept step, against the true count of kept erroneous steps.
- The official PRMScore (`E.official`).
- The signed count bias.
- Spearman(k̂, true k).

**Paired 95% CIs.** These come from `E.boot_weights` (source-group bootstrap, the evaluator's seeds and draws). The
same weights are applied to the per-answer absolute errors for the MAE contrasts.

**E1 (main: does it count better than GAP?).** For a given arm, all of the following must hold in ≥ 3 of the 4
Socratic PRMs:
- the MDS − GAP count MAE has CI high < 0;
- the MDS − SHARE count MAE has CI high < 0, otherwise the gain is a prevalence artifact;
- the MDS − GAP PRMScore has CI low > −0.5 points.

**E2 (the Markov chain helps).** The MDS − IND count MAE has CI high < 0 in ≥ 3 of 4 Socratic PRMs.

**E3 (several voters help).** The MDS − SINGLE count MAE has CI high < 0 in ≥ 3 of 4 Socratic PRMs.

**E4 (mechanism, descriptive, every cell).**
- ŝ against the true step share (.158 / .343 / ~.41).
- The fitted e and ρ.
- The fitted δ_g against each voter's within-AUC. The within-AUC uses labels, so this comparison is computed after the
  freeze.
- Spearman(k̂, true k) for MDS and for GAP.
- Spearman between the per-step posterior error probability and the relative position, for the model and for the
  truth.

**E5 (descriptive).** The same contrasts on Hard2Verify and PRMBench, with no decision. On PRMBench, also MDS − K2.

**Prediction, declared now.**
- **PRMBench.** MDS overcounts (ŝ > .20 against .158), as every DS-type estimate has so far. It does not beat K2.
- **Socratic.** ŝ lands near the true .343, but that is not evidence by itself. Only E1's SHARE contrast can show
  answer-specific counting.
- **Synthetic expectation.** I simulated answers with known δ, independent labels and a random per-answer affine
  nuisance. The invariant count gave an MAE of 1.38 against GAP's 1.56. Even inside the model the expected gain is
  modest, and on real data with fitted δ it should be smaller.

## 5. Order of work and integrity

1. **Code and tests.** Write `spectral_utils/markov_ds_count.py`, containing log J_d, ℓ_g, L(k), the posterior median,
   the fit and validity. Write `tests/test_markov_ds_count.py`. The tests:
   - J_d against quadrature.
   - The direction density against a Monte Carlo histogram for n = 4.
   - Recovery of δ, e and ρ on synthetic Markov data with a per-answer affine nuisance.
   - Count recovery on synthetic data: MDS beats GAP and SHARE.
   - On Markov data, MDS beats IND.
   - A negative control: a shared position trend added to all voters. This is documented behavior, not pass/fail.
2. **Arm builder.** Write the voter builder with the SUB5 parity gate and the AUTO/TELE16 |corr| = 1 gate.
3. **Freeze.** Freeze the hashes of the module, the tests, `results/markov_ds_count_v1/run.py` and this file in
   `PROTOCOL.json`. The runner refuses to run if a hash differs.
4. **Label-free stage.** For every cell, save the voters' signs, θ, validity, ŝ and k̂ for every candidate to
   `results/markov_ds_count_v1/fits/`.
5. **Evaluation stage.** Write `RESULTS.csv`, `CONTRASTS.csv`, `VERDICT.json`, a per-cell HTML report (in the style of
   the neighbor report, one panel per experiment and per model × benchmark), and `NEGATIVE_RESULT.md` if E1 fails in
   every arm.
6. **Independent check.** Re-implement L(k) and k̂ from this text for one Socratic cell and check exact count parity.

**Runtime.** Expected CPU minutes. No cluster, no GPU, no new inference.

## 6. Known risks, declared before running

- **A shared nuisance breaks conditional independence.** A position trend or "level" drift that moves all groups
  together violates conditional independence across voters. The chain may then pick it up as the error state, as in
  Step 318 and Step 485 on Socratic. Position is deliberately NOT in the model. E4's position Spearman exposes this.
- **The affine nuisance is the only answer effect assumed.** Answer-specific separations (b varies, but δ is shared)
  are assumed away.
- **One configuration per k.** It ignores multiplicity and depends on the arm's ranking being good. A poor ranking
  caps the gain, as it caps ORACLE.
- **The cap binds.** The true count exceeds ceil(n/2) in 9.3% of Socratic answers and 31.5% of Hard2Verify answers
  (Qwen-7B cell; counted during preparation from labels, so this is benchmark structure that has been seen). The cap
  is identical for every rule, and the effect is reported.
- **Clean answers.** Socratic has 0.2% clean answers and Hard2Verify 22.7%. This study is conditional on at least one
  error, so k̂ ≥ 1, as with GAP. Clean answers belong to part 1.
- **Exposure.** Socratic is not untouched. Its scores, the SYM-RECOUNT curves and the H1b counts of the neighbor study
  have all been seen. A Socratic pass is external development evidence, not confirmation.
- **The headroom is small.** On PRMBench Qwen-7B the oracle count is worth about 5 PRMScore points over GAP. A
  realistic gain is a fraction of that.

## Amendment A (2026-11-01, pre-label; approved by Omri: "Amend to exact")

Written after the synthetic stop (`results/markov_ds_count_v1/prelabel_synthetic/PRELABEL_STOP.md`) and before any
freeze, real-data fit or label of this study. It replaces sections 3.3 and 3.4 where they conflict.

- **A1. Exact likelihood.** The answer likelihood sums over all 2^n − 1 configurations with at least one error.
  This is enumeration, with no ranking involved. MDS, IND and SINGLE all use it.
  - **Enumeration limit:** n_max = 16 on Socratic and Hard2Verify. That covers 2,991 of 2,995 and 175 of 181
    answers. On PRMBench n_max = 12, which covers 60% of the answers (3.7M configurations); PRMBench is descriptive.
  - **Longer answers:** they are left out of the fit and their count falls back to GAP. The fallback share is
    reported per cell.
- **A2. Count.** k̂ is the median of the exact posterior of the number of errors, clipped to [1, ceil(n/2)].
  - Answers with n ≤ 2 get k̂ = 1.
  - **Flags:** unchanged, the top k̂ steps by the arm's fused score.
  - **Implied share:** ŝ = Σ E[count] / Σ n over the fitted answers.
- **A3. Secondary arm TOPK.** This is the original single-configuration approximation (sections 3.3 and 3.4 as first
  written), with K voters and the Markov chain. It is reported descriptively. It is not part of E1, E2 or E3, because
  it failed the synthetic recovery test.
- **A4. Starts.** Start j uses δ_g = m_j × gap_g with m = (.5, 1, 2, .5, 1), and (e, ρ) = the j-th declared pair.
  - gap_g has a floor of .1.
  - The start δ is clipped to [.06, 9].
  - The L-BFGS-B limits are maxiter 500 and maxfun 1500.
- **A5. Numerics.**
  - **log J_d:** computed by cubic Hermite tables (step .01, |τ| ≤ 60), checked against direct quadrature to 1.4e-9.
    Values outside the table use direct quadrature.
  - **Degenerate voters:** a voter that is constant inside an answer contributes nothing for that answer.
- **A6. Arm inputs.** SUB5 is the arm as originally defined (stage2 `TeleCell`). Its covariance pool uses `c.keep`:
  - on Socratic, the official include mask, which is all true (checked);
  - on PRMBench, the exclusion of the metadata control rows.
  - Correctness labels do not enter it.
- **A7. Margins and definitions.**
  - **E1 PRMScore margin:** −0.5 points, which on the 0–1 Socratic scale is −0.005.
  - **Count MAE:** the number of flagged kept steps against the number of kept erroneous steps, over answers with at
    least one kept step. This is the same definition as in the neighbor study.
- **A8. Synthetic tests (section 5, step 1) and their pass criteria, declared before running them.**
  The data are Markov (e = .15, ρ = .5, δ = 1.5/1/.8/.6, n = 4..10, 1,500 answers), with a per-answer affine nuisance.
  The exact fit must meet all of the following:
  - **Recovery:** every δ_g within ±.2, e within ±.06, ρ within ±.12.
  - **Counting:** the exact MDS count MAE is below GAP's and below SHARE's (with ŝ_MDS).
  - **Markov versus IND:** the MDS MAE is ≤ the IND MAE. A failure is reported, not hidden.
  - **Negative control:** the position-trend control is documented only.
