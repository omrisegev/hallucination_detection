# Pre-registration: neighbor-aware readout of the fused step score (chain-graph / hidden-Markov readout)

Date: 2026-11-01. Written before any labelled metric of this study. Branch `claude/raw-null-gate-2026-10-30` (local).
Status: **declared, not run.** No code for this study exists yet. The code will be frozen by file hashes in
`results/neighbor_readout_v1/PROTOCOL.json` before any label is read. Anything added after labels are read is marked
POST HOC.

## 1. Question and why

**Today's readout.** Fusion (CHAIN / L-SML) gives one score per step. GAP then sorts the answer's answer-z scores,
sets k to the position of the largest gap (cap ceil(n/2)) and flags the top k. Every step is judged alone. The time
order of the steps is ignored.

**Omri's question (2026-11-01).** A step next to an error that sits slightly below the flag line may still be an
error because it is adjacent. Two things must be settled:
1. **Form.** What is the right model of this neighbor effect?
2. **Weight.** How much should a neighbor count, and can that weight be chosen without labels?

**Evidence that motivated it (development data, already seen; listed so the exposure is explicit).** Source:
PRMBench, 6,211 non-control answers, `results/advisor_completion_2026-10-11/inputs/population.npz`, plus
`results/thinkfix_rescore_v1/prmscore_inputs/bank_N16.npz` S__LSML_CONT. Computed in this session; not a candidate.
- **Errors come in runs.** P(error | previous step error) = .417, against .123 after a correct step.
  - Run lengths are 1/2/3 steps in .596/.254/.110 of runs.
  - A first-order Markov chain with continuation .42 predicts .58/.24/.10.
- **GAP misses the neighbors.** An unflagged step adjacent to a GAP flag is wrong in .190 of cases (N=16,197), against
  .075 for one that is not (N=53,845).
  - The ratio stays about 2x within every relative-position quintile x score tercile.
  - Isolated flagged steps are NOT worse than flagged steps inside a block (.468 vs .430).
- **One fixed smoothing weight raised within-answer AUC.** s + .5·mean(neighbors) moved it .779 → .790. This was a
  single untuned weight, on labels already seen.
- **Label-free feasibility.** This was computed WITHOUT labels, on every cell below.
  - A variogram moment estimate of the Markov eigenvalue λ (section 3.3) is valid in 8/8 PRMBench cells, 4/8 Socratic
    cells and 0/8 Hard2Verify cells.
  - On PRMBench it gives λ = .51–.72 (CHAIN_N16). The labelled run persistence is .42 − .12 = .29, so the moment
    estimate overstates persistence. This is a known risk, declared here.
  - A plain autocovariance version fails, because per-answer z-scoring of short answers biases it by about −1/n.

**What this study is not.**
- It does not change the fusion.
- It does not decide whether an answer is clean (k = 0). That is the separate raw-null-gate study.
- It does not use ProcessBench, whose labels mark only the first error.

## 2. Data, cells, inputs (no new inference)

**Loader.** `results/prm_boost_collection_v1/evaluate_collection.py`: `E.load(bench, prm)`, with the official
metrics `E.official` / `E.measure` and the paired bootstrap `E.contrast` over source groups. All of these are used
unchanged.

**Benchmarks.**
- **prmbench:** development data, primary. PRMScore is on non-controls.
- **socratic:** transfer, 2,995 answers. Official PRMScore, with empty steps handled as in the evaluator.
- **hard2verify:** transfer, 181 answers, small, reported descriptively.

**PRMs.** qwen_prm_7b, qwen_prm_72b, pure_prm and rlhflow_prm. Math-Shepherd is excluded, by the 2026-10-26 decision.

**Fused scores.**
- **Primary:** `c.arms['CHAIN_N16']` (PRM + telemetry).
- **Secondary:** `c.arms['TELE_N16']` (telemetry only).
- Each is answer-z'd with `E.answer_z`.
- This gives 3 benchmarks x 4 PRMs x 2 arms = 24 cells.

**Labels** (`c.err`, `c.keep`) are read only by the evaluation stage, after the label-free stage of a cell is on disk.

## 3. Readouts (exact)

The notation:
- s = answer-z fused score of one answer, with steps t = 1..n in answer order;
- N(t) = the existing neighbors of t among t−1 and t+1;
- k_GAP = `E.gap_counts(s)` on the ORIGINAL s.

### 3.1 References (unchanged)
- **GAP:** `E.topk_flags(s, k_GAP)`. This is the current readout.
- **K2:** top 2 by s. It uses the PRMBench construction prior and is a reference only.
- **ORACLE:** top-k by s with the true count. It uses labels and is a ceiling only.

### 3.2 Stage 1: form, label-using DIAGNOSTIC curves (not candidates)

The count is fixed at k_GAP, so only the ranking changes. The flags are the top k_GAP steps of the transformed score.

| form | transformed score | grid |
|---|---|---|
| SYM(w) | s_t + w · mean_{u∈N(t)} s_u | w ∈ {0, .1, .2, .3, .5, .75, 1.0} |
| LIFT(w) | s_t + w · max(0, max_{u∈N(t)} s_u − s_t) | w ∈ {0, .1, .2, .3, .5, .75, 1.0} |
| LAP(λ) | (I + λ L)^{-1} s, where L is the path-graph Laplacian of the answer | λ ∈ {0, .1, .25, .5, 1, 2} |

**What each form tests.**
- SYM is symmetric smoothing.
- LIFT can raise a step toward a stronger neighbor but never lowers an isolated peak. It tests the asymmetry seen in
  section 1.
- LAP is graph Tikhonov smoothing. It reaches beyond one neighbor.

**Secondary diagnostic.** SYM-RECOUNT(w): GAP recomputed on the transformed score, so both the count and the
ranking change.

**Reported per cell.** Official metric and within-answer AUC at every grid point, the best value, and the argmax.
These curves locate the best label-using weight. They are never reported as methods.

### 3.3 Stage 2: label-free CANDIDATES (frozen before labels)

Each is fit pooled over all answers of one cell, without labels. This is the same fit scope as the existing fusion.
Answer-local fitting is infeasible at about 13 steps per answer, and the cell-pooled scope is disclosed.

#### H1 = HMM-POS: primary candidate

**Model.** A two-state hidden Markov model on s, with states 0 (correct) and 1 (error).
- **Emissions:** s_t | state j ~ Normal(μ_j, σ_j²).
- **Entry:** P(z_t = 1 | z_{t−1} = 0) = sigmoid(a + b·r_t), where r_t = (t − .5)/n is relative position. The first
  step uses the same expression.
- **Stay:** P(z_t = 1 | z_{t−1} = 1) = ρ.

**Fit.** Baum-Welch (EM).
- The (a, b) M-step is a weighted logistic regression on expected transitions, with 10 Newton steps.
- Tolerance is a relative log-likelihood change < 1e-7, with at most 500 iterations.

**Starts.** Five deterministic starts: (μ_0, μ_1) at the pooled quantiles (.5,.90), (.5,.95), (.6,.90), (.4,.85) and
(.5,.80). Each start has σ_j = pooled sd, ρ = .4, a = logit(.1) and b = 0. The fit with the highest log-likelihood is
kept.

**Label switch.** State 1 is the state with the larger μ.

**Validity.** The fit is invalid if:
- the stationary share of state 1 is < .01 or > .60;
- σ_1/σ_0 is outside [.1, 10];
- or μ_1 − μ_0 < .1.

An invalid cell falls back to GAP. Fallbacks are reported, not hidden.

**Readouts.** p_t is the forward-backward posterior P(z_t = 1).
- **H1a (ranking):** top k_GAP steps by p_t. This isolates the neighbor effect on WHICH steps are flagged.
- **H1b (count):** flag the steps with p_t > .5. The count comes from the model, with no cap. This answers HOW MANY.

**Ablation.** H1-NOPOS: the same model with b = 0. It tests whether the position prior matters.

#### V1 = SYM-VARIOGRAM: secondary candidate

**Detrending.** r = s minus the pooled mean of s in its relative-position decile.

**Variogram.** V(L) = ½ mean[(r_t − r_{t+L})²], for L = 1, 2, 3, over answers with n ≥ 4.

**Moment model.** For L ≥ 1, V(L) = σ² + Vx(1 − λ^L). Then:
- d1 = V2 − V1;
- d2 = V3 − V2;
- λ = d2/d1;
- Vx = d1/(λ(1 − λ));
- σ² = V1 − Vx(1 − λ).

**Validity.** d1 > 0, 0 < λ < 1, Vx > 0 and σ² > 0. Otherwise w = 0, which falls back to GAP and is reported.

**Wiener weight.** γ0 = σ² + Vx, γ1 = Vxλ and γ2 = Vxλ². The weights are β = Σ⁻¹ c, where Σ is the 3x3 Toeplitz
matrix (γ0, γ1, γ2) and c = Vx(λ, 1, λ). The weight is w = β_0/β_1.

**Readout.** SYM(w), with count k_GAP.

#### G1 = LAP-GCV: secondary candidate

LAP(λ), with λ chosen per cell by pooled generalized cross-validation over the stage-1 grid:

GCV(λ) = Σ_answers ||s − S_λ s||² / (Σ_answers (n − tr S_λ))²

Count k_GAP.

## 4. Endpoints and decision rules (declared)

**Cell scope.** All primary rules use CHAIN_N16. TELE_N16 is reported with the same rules as a secondary panel.
Deltas and 95% CIs come from `E.contrast` (paired, source-group bootstrap, the evaluator's seeds and draws).

**P1 (which steps).** H1a beats GAP on the official metric with CI low > 0 in ≥ 3 of 4 PRMBench PRMs. In all 4, the
within-AUC delta must not be significantly negative (CI high ≥ 0).

**P2 (how many).**
- H1b official ≥ K2 − 0.5 in ≥ 3 of 4 PRMBench PRMs.
- The mean absolute count error (on non-controls) is below GAP's in ≥ 3 of 4.
- Because K2 uses the construction prior, matching it without that prior counts as success.

**P3 (weight choice).** For each label-free candidate, its gain over w=0 is divided by the gain of the stage-1 peak of
the same form. "Good weight" means this ratio is ≥ .5 in ≥ 3 of 4 PRMBench PRMs. For H1a, the reference curve is
SYM's peak.

**P4 (form).**
- Report which stage-1 form has the highest peak in ≥ 3 of 4 PRMBench PRMs.
- Report whether LIFT ≥ SYM, which is the asymmetry question.
- The peak value is used only as a description. It is not used to pick a method.

**T (transfer).**
- Each candidate that passes P1 or P2 on PRMBench is applied FROZEN, with the same code and the fit redone without
  labels on each cell, to Socratic.
- Transfer holds if the same rule passes in ≥ 3 of 4 Socratic PRMs.
- Hard2Verify is reported descriptively. With 181 answers and wide CIs, it decides nothing.
- A candidate may fail transfer and still be reported as a PRMBench-only result.

**Partial success is reportable.** A rule that passes on some benchmarks, PRMs or metrics and fails on others is
reported with that breakdown, and is not rounded up to a general claim.

## 5. Order of work and integrity

1. Write `spectral_utils/neighbor_readout.py` (SYM, LIFT, LAP, variogram weight, HMM-POS, GCV) and
   `tests/test_neighbor_readout.py`. The tests cover:
   - synthetic Markov data, where H1 must recover ρ and the entry probabilities within tolerance;
   - LAP at λ = 0 and SYM at w = 0, which must reproduce GAP's flags exactly;
   - variogram recovery on simulated s = x + noise.
2. Freeze the hashes of those files and of the runner `results/neighbor_readout_v1/run.py` in `PROTOCOL.json`. The
   runner refuses to run if a hash differs.
3. **Label-free stage.** For every cell, fit and save to `results/neighbor_readout_v1/fits/`: the HMM parameters and
   validity, the variogram (λ, Vx, σ², w) and validity, and the GCV λ.
4. **Evaluation stage.** Read the labels and compute the stage-1 curves, the stage-2 candidates, the references and
   the contrasts. Write `RESULTS.csv`, `CONTRASTS.csv`, `CURVES.csv`, `VERDICT.json` and `README.md`.
5. **Independent re-implementation check.** Recompute H1a/H1b for one PRMBench cell from this text alone, and check
   for exact flag parity.

**Runtime.** Expected CPU minutes. No cluster, no GPU, no new inference.

## 6. Known risks, declared before running

- **The HMM may lock onto a nuisance state.** It could find a persistent nuisance state, such as level or position
  drift, instead of the error state. This is the Step 318 IMM experience. The position-entry term and the
  H1-NOPOS ablation are there to expose it.
- **Gaussian emissions on a skewed score can invent a second class.** This is what happened in Step 484
  (`answer_mixture_gate_v1`). The validity rules catch only the extreme cases.
- **The variogram overstates persistence on PRMBench**, as section 1 shows. V1 may therefore over-smooth, and it falls
  back to GAP on most external cells.
- **PRMBench is already heavily inspected development data.** A PRMBench pass is development evidence. Socratic is the
  stronger check. Its scores have been seen before in other studies, though not with these readouts.
- **The headroom is small.** The oracle count is 75.40 against 70.40 for GAP on the CHAIN_N16 Qwen-7B PRMBench boosting
  line (`results/error_count_label_free_v1/RESULTS.csv`), so a realistic gain is a fraction of 5 points.
