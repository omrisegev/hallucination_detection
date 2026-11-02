# Pre-registration v2: calibrated hurdle HMM (2026-11-02)

**Status.** Amendment of `HURDLE_HMM_PREREG_2026-11-02.md` (v1), after the v1 pre-label stop
(`results/hurdle_hmm_v1/PRELABEL_STOP.md`) and the independent review (`REVIEW_STAGE1.md`). Omri chose this
version on 2026-11-02 ("Calibrated: π + clean law from labels"). Written before the v2 synthetic test finished
and before any real label is read by v2. Frozen by SHA-256 in `results/hurdle_hmm_v2/PROTOCOL.json` before the
real run.

## 1. What changes from v1

| Item | v1 | v2 |
|---|---|---|
| π | fitted | **fixed** = share of erroneous answers among m labelled answers (clipped to [1/2m, 1 − 1/2m]) |
| Clean law (μ0, s0) | truncated-normal fit on the bottom 30% | **mean and SD of the labelled clean steps** (keep_eval ∧ ¬error) of the m labelled answers |
| Quadrature | 20 nodes centred on the prior | **adaptive** 20-node Gauss–Hermite (unit test: ≤ 1e-4 vs a dense integral at n = 107) |
| Free parameters | 7 | 6: e1, h, ρ, δ, σ1, f, fitted label-free on all answers (transductive, as AUTO) |
| Access | label-free | **calibrated track (D1)**: m labelled answers |
| Controls | GAP | GAP, SHARE, GATE_GAP, THR (same labelled answers) |
| Evaluation scope | all fit files | **only the 16 primary cells** (review must-fix 5) |

Reason: the v1 oracle check showed that π is recovered once the clean law is known. The option-A synthetic check
showed that, with labels, the plain labelled share estimates π at least as well as the model. So π comes from
labels; the model is used for what labels do not give directly: P(answer has an error) and per-step posteriors
with the Markov structure.

## 2. Labelled set and repetitions

- m = 200 labelled answers per cell.
- R = 5 repetitions (seeds 0–4). Each repetition splits the answers 50/50 at random into a pool and a test half,
  and draws the m labelled answers from the pool.
- All metrics are computed on the test half only. The label-free part of the fit sees all answers without labels.
- Limit: the bundle has no source-question groups, so PRMBench variants of one source question can fall in both
  halves.

## 3. Methods compared (same m labelled answers in every arm)

| Arm | Rule | Labels used |
|---|---|---|
| **HH-BAYES** | flag step i if P(y_i = 1 \| x) > .5 | π, clean law |
| HH-K (secondary) | if P(A = 1 \| x) > .5, flag the top round(Σ_i P(y_i = 1 \| x)) steps (≥ 1); else none | π, clean law |
| GAP | `gap_counts` on the arm's answer-z fused score (`step_z_fused`) | none |
| SHARE | top round(ŝ · n_j) steps by `step_z_fused` (≥ 1), ŝ = labelled step error share | ŝ |
| GATE_GAP | GAP flags only in answers whose raw fused answer max is above the (1 − π̂) quantile of all answers | π̂ |
| THR | global threshold on the raw step score, chosen on the labelled steps to maximize macro-F1 (99 quantiles) | step labels |

## 4. Cells

The 16 primary cells of v1: `AUTO_prm` × {ProcessBench, CRV-Arith, CRV-GSM8K} × 4 PRMs and `TELE16_sa` ×
PRMBench × 4 PRMs. Bundle: `results/raw_null_gate_diag_v1/bundle/` (`claude/raw-null-gate-2026-10-30` @ `f446884e`).

## 5. Endpoints and pass rules

**Per cell**, D = HH-BAYES − BEST, where BEST is the control with the highest mean over repetitions (the hardest bar,
chosen after the run by its own score; disclosed).

| ID | Endpoint | Population | Cell passes if |
|---|---|---|---|
| **P2** (primary) | step macro-F1 | `step_mask` steps of the test half | mean D > 0 and D > 0 in ≥ 4 of 5 repetitions |
| **P3** (primary for ProcessBench only) | PRISM-T6-style F1 (error answers: first flag = first error; clean answers: no flag) | test-half answers | same rule, vs the best of GATE_GAP and THR |
| P1 (secondary) | answer gate balanced accuracy, P(A = 1 \| x) > .5, vs the GATE_GAP gate | evaluated test-half answers | reported |

- **Benchmark passes** P2 (or P3) if ≥ 3 of its 4 PRM cells pass.
- **Line verdict:** "works" if P2 passes on ≥ 2 of the 4 benchmarks. "Partial" if P2 passes on exactly 1 benchmark,
  or P3 passes on ProcessBench. Otherwise "fails". (P1 and P2 do not need to pass on the same benchmarks.)
- Secondary, reported: HH-K; each control separately; fitted δ / s0 vs D3's d; parameters at a bound.

## 6. Predictions declared now

- THR will be a strong control on CRV (global calibration matters there, T2).
- The model's best chance is ProcessBench (P3), where a gate plus first-error structure matters, and PRMBench, where
  errors come in runs.
- At d ≈ 1, gains over SHARE / THR, if any, will be small (< 2 macro-F1 points).

## 7. Synthetic stop rule (before any real-data fit)

`results/hurdle_hmm_v2/synthetic_v2.py`: model data, N = 1,500 answers, lengths 4–15, π ∈ {.10, .60, .95},
δ ∈ {1, 2}, seeds 0–1, m = 200.
- **Required:** median over seeds of HH-BAYES macro-F1 ≥ max(GAP, SHARE, GATE_GAP, THR) − .005 in ≥ 5 of the 6
  (π, δ) settings, and at most 1 of 12 fits with a free parameter at a bound.
- If it fails: stop and report before any real-data fit.

## 8. Independent check

The same reviewer subagent checks this amendment and the code before the real run (stage 1b), then recomputes
every endpoint from the saved outputs after the run (stage 2), including consistency with Steps 484–487, D1–D3
and the 2026-11-01 checks. Reports: `results/hurdle_hmm_v2/REVIEW_*.md`.
