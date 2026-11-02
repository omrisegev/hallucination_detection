# Pre-registration: answer-level hurdle HMM on raw step scores (2026-11-02)

**Status.** Written before any label is read by this experiment. The code and this file are frozen by SHA-256
in `results/hurdle_hmm_v1/PROTOCOL.json` before the real run. Omri approved the plan on 2026-11-02 and asked
for an independent subagent check of the work.

**Development data.** All cells were inspected before (raw-null gate, D1-D3, answer-level DS checks of
2026-11-01). The anchor choice (q = .30) was informed by the label-using "certified clean" check of
2026-11-01 (purity .93–.99 at step level). This is a disclosed dependence, not untouched evidence.

## 1. Question

Can one label-free probabilistic model of each answer, P(answer state, step states | raw step scores),
estimate (a) the share of answers with an error, and (b) per-step error decisions better than GAP?

## 2. Why this model (links to earlier findings)

| Component | Earlier finding it answers |
|---|---|
| Raw pooled scores, no answer-z | T1 (theory note 2026-11-01): answer-z removes the answer level; raw answer max AUC .69–.81 |
| Answer effect b_j ~ N(0, τ²), integrated out | Answer-level DS was pulled to π ≈ .5 by a shared answer factor (class-conditional \|corr\| .20–.39) |
| Answer state A_j (hurdle) | GAP always flags ≥ 1 step; clean answers are 35% (ProcessBench), 79–92% (CRV) |
| Markov chain on steps given A_j = 1 | Errors come in runs (Step 485: .417 vs .123); neighbour effect differs by benchmark (2026-11-01 check) |
| Clean law anchored on the low tail (q = .30) | Step 485 HMM error state took .40–.55 of steps without an anchor; low tail is .93–.99 clean |
| Exact sum over all step configurations | Step 486: the one-configuration approximation failed its synthetic test; the exact sum passed |

## 3. Model (exact definition in `spectral_utils/hurdle_hmm.py`)

Per answer j with n_j steps and raw score x_ij (the arm's `step_raw_fused`, pooled z, higher = more error):
- b_j ~ N(0, τ²); A_j ~ Bernoulli(π).
- A_j = 0: all y_ij = 0. A_j = 1: y follows a 2-state Markov chain (P(y_1 = 1) = e1, P(0→1) = h,
  P(1→1) = ρ), conditioned on at least one error. Implemented exactly with three states.
- x | y = 0, b ~ N(μ0 + b, σ0²); x | y = 1, b ~ N(μ0 + δ + b, σ1²); δ > 0.
- σ0² = (1 − f) s0², τ² = f s0².

**Stage A (anchor).** (μ0, s0) = truncated-normal MLE on the steps with x below the pooled .30 quantile.
**Stage B.** θ = (π, e1, h, ρ, δ, σ1, f) by L-BFGS-B on the exact marginal likelihood (Gauss–Hermite, 20 nodes),
three fixed starts, best final value kept. Fit on at most 2,000 answers (seeded random subset, seed 0); score all.

**Outputs, all label-free.** π̂; P(A_j = 1 | x_j); P(y_ij = 1 | x_j).

## 4. Decision rules (fixed)

- **HH-BAYES:** flag step i if P(y_ij = 1 | x_j) > .5.
- **HH-K (secondary):** if P(A_j = 1 | x_j) > .5, flag the top round(Σ_i P(y_ij = 1 | x_j)) steps by posterior
  (at least 1); else flag none.
- **Baseline GAP:** the same arm's answer-z fused score (`step_z_fused`) with `gap_counts` (copied verbatim).

## 5. Cells

- **Primary (16):** `AUTO_prm` × {ProcessBench, CRV-Arith, CRV-GSM8K} × 4 PRMs, and `TELE16_sa` × PRMBench ×
  4 PRMs (the only PRMBench cells in the bundle). PRMs: Qwen-PRM-7B, Qwen-PRM-72B, PURE, RLHFlow.
- **Data:** `results/raw_null_gate_diag_v1/bundle/` on `claude/raw-null-gate-2026-10-30` @ `f446884e`.
- **Secondary (descriptive only):** the other 52 bundle cells, run after the primary verdict if time allows.

## 6. Endpoints and pass rules

Labels are read only by `evaluate.py`, after all fits are saved.

| ID | Endpoint | Truth / population | Pass in a cell | Pass for a benchmark |
|---|---|---|---|---|
| **P1** | \|π̂ − π\| | π = share of all answers with an error (all answers, because the fit uses all) | ≤ .05 | ≥ 3 of 4 PRMs |
| **P2** | step macro-F1 (mean F1 of error and correct class), HH-BAYES − GAP | `step_mask` steps | > 0 (point estimate; paired answer bootstrap CI reported) | ≥ 3 of 4 PRMs |
| **P3** | answer AUC of P(A=1\|x) minus answer AUC of the raw fused answer max (no-harm) | evaluated answers | ≥ −.02 | ≥ 3 of 4 PRMs |

- **Line verdict:** "works" if P1 and P2 both pass on ≥ 2 of 4 benchmarks. "Partial" if P1 or P2 passes on
  ≥ 2. Otherwise "fails".
- **Secondary (descriptive):** HH-K macro-F1; ProcessBench PRISM-T6-style F1 (error answers: first flagged step
  = first error; clean answers: no flag; harmonic mean); count MAE on erroneous answers (CRV, PRMBench);
  fitted δ/s0 against D3's label-based d.

**Predictions declared now.**
- ProcessBench: P1 likely to fail (D2 contamination: post-error steps are unlabelled but scored).
- PRMBench: P1 may pass by default (π near 1 is easy to hit when the model calls most answers erroneous); P2 is
  the informative test there.
- CRV-Arith: the Gaussian clean law is misspecified (skew 0.8, D2). Over-calling is the main risk.

## 7. Stop rule before labels (synthetic)

`results/hurdle_hmm_v1/synthetic_recovery.py`: data from the model, 1,500 answers, lengths 4–15,
π ∈ {.10, .60, .95}, δ ∈ {1, 2}, 3 seeds. **Required:** |π̂ − π| ≤ .05 in all δ = 2 runs (median over seeds).
δ = 1 is reported, not required. If the requirement fails, stop and report before any real-data fit.

## 8. Independent check

A subagent with no access to my reasoning reviews this file and the code before the real run, re-implements the
likelihood and the metrics, and checks consistency with earlier steps. After the run it recomputes every
endpoint from the saved outputs. Its report is saved in `results/hurdle_hmm_v1/REVIEW.md`.

## 9. Known limits

- Gaussian emissions; position is not modelled (Step 481: position matters on Socratic, which is not here).
- ProcessBench steps after the first error are in the fit but have no labels.
- The fit subsample (2,000 answers) is a compute limit.
- PRMBench uses standalone telemetry only (backbone cells are on the PC).
