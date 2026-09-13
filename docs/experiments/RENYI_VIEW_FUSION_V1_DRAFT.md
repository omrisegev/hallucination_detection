# Renyi-view fusion v1 — DRAFT (Stage 3 prototype; final design deferred)

**Status: DRAFT.** This document fixes the view definitions, banks, solvers,
diagnostics and the planned comparison table for the Stage 3 prototype of the
2026-09-12 plan. It is implemented and smoke-tested for mechanics only
(27 answers). **The full 13,769-answer run and the final design wait for the
review of the cross-rank experiment.** Nothing here is a performance claim,
nothing is promoted, and no full run may be launched from this draft
(`run_renyi_view_fusion_v1.py` refuses a non-smoke run without `--allow-full`).

Branch/worktree: `claude/varentropy-expansion-fusion-v1`
(`.worktrees/varentropy-expansion-fusion-v1`). New files only:
`spectral_utils/renyi_view_fusion.py`, `scripts/run_renyi_view_fusion_v1.py`,
`scripts/test_renyi_view_fusion.py`, this document, and
`results/renyi_view_fusion_v1/` (smoke artifacts). No tracked module is modified.

## 1. Contract (frozen; identical to every localization experiment)

* 13,769 answers: ProcessBench 6,800 in 8 cells (gsm8k/math/olympiadbench/
  omnimath x Qwen3-4B/8B) + PRMBench 6,969 (Qwen3-8B telemetry).
* Frozen JOINED labels (v3, one-based PRMB conversion) and v2 source-group
  folds (`FOLDS_V2.json`); `localization_full_benchmark_v3/evaluation/JOINED.*`.
* Access: one teacher-forced pass over the provided official answer; saved
  top-50 log-probabilities and selected-token surprisal; gray-box, one pass.
* Readout: per-step top-10 token mean of the token score, earliest argmax.
* No-error decision: external mean-entropy gate, q=0.3 on the held fold
  (`old._gate_contract`, from `fusion_fixed_gate_v1`, arm `dual__iu`).
* PRMScore: q=0.8 per-method calibration on held source-group folds;
  reported conditional when coverage < 100 %, headline null in that case.
* Uncertainty: 10,000-draw paired source-group bootstrap (`base.paired_bootstrap`),
  97.5 % CI for the two primary contrasts, 95 % for all secondary contrasts.
* Fitting: answer-local, unlabeled, from the current answer alone. A failed
  fit is a declared failure: absent from fits, NaN step scores, counted against
  the full denominators; never substituted by another arm.
* Reported: PB all-8 macro F1, Q4, Q8; PRMB within-answer AUC (+n), pooled AUC;
  PRMScore (conditional if needed); coverage per arm; runtime per arm.
* All cached data are development data, not an untouched confirmation.

## 2. Views (renormalized top-15 head, frozen epsilons)

For each token, `q = p / (sum p + 1e-12)` over the saved descending top-15
log-probabilities and `s = -log(q + 1e-12)` (exactly the frozen
`token_feature_views` convention used by entropy15/varentropy15).

| view | definition | note |
|---|---|---|
| `H0.5` | `log(sum q^0.5) / (1 - 0.5)` | |
| `H1` | `sum q s` | Shannon; **equals the frozen entropy15 to 1e-12** (tested on real rows) |
| `H2` | `log(sum q^2) / (1 - 2) = -log(sum q^2)` | collision entropy on the **15-support**; the existing `topk_renyi2_series` is on the 50-support and is a different quantity (their Pearson correlation is reported per answer, they are never treated as identical) |
| `H4` | `log(sum q^4) / (1 - 4)` | |
| `Hinf` | `-log(max q + 1e-12) = min_i s_i = s_1` | min-entropy; identical to the rank-1 renormalized surprisal |
| `H0` (excluded) | `log |{i: q_i > 0}| = log 15` | Hartley; **constant** on the retained support, so it carries no within-answer information and is excluded by construction. It is computed in the diagnostics only, to document that it is constant (`hartley_constant`). |

Epsilon rule: the frozen `+1e-12` enters `H1` and `Hinf` through `s`. For the
generic orders no epsilon is added inside the log because
`sum q^alpha >= 15^(1-alpha) > 0` on the renormalized support (alpha = 4:
>= 2.96e-4). Adding it would dominate the sum for large alpha (checked: alpha = 16
on a flat head breaks monotonicity), which is why the module does not do it.

Properties tested (`scripts/test_renyi_view_fusion.py`, all pass, synthetic +
real ProcessBench gsm8k rows): `H1 == entropy15` (1e-12); `Hinf == s_1`;
`H0` constant `= log 15`; `H_alpha` for alpha = 0.999 / 1.001 brackets and
approaches `H1`; `H_alpha` non-increasing in alpha over a dense grid up to
infinity; uniform head gives `log 15` for every order and a one-hot head gives 0.

Selected-token block `SEL = [a, a^2, a^3]`, `a = -log p(selected)` from the
cached `token_spilled_energies`, validated by `selected_surprisal`.

## 3. Banks, single views and solvers

* `R5` = `[H0.5, H1, H2, H4, Hinf]` (5 columns); `R5_sel` = `R5 + SEL` (8).
* Single-view arms (raw view, natural high-is-risk sign, no data-driven flip,
  no z-scoring): `view__H0.5`, `view__H1` (= plain entropy15), `view__H2`,
  `view__H4`, `view__Hinf`, `view__sel1`. A constant view is a declared failure.
* Fused arms: `R5__equal`, `R5__iu`, `R5_sel__equal`, `R5_sel__iu`, `R5_sel__shrink`.
  * Standardization: `zscore_columns` on the answer's own tokens; columns with
    std <= 1e-10 are dropped (the drop fraction per view is a diagnostic).
    Fewer than 3 tokens or fewer than 3 varying columns = declared failure.
  * Orientation anchor: the answer's own raw K=15 varentropy (sum of the frozen
    contribution matrix; label-free; as in Step 339). Each standardized column
    is sign-oriented by its within-answer Pearson correlation to the anchor
    (recorded as `column_signs` / `column_flips`); the fused score is passed
    through `_orient` once more and the global flip is recorded.
  * `equal`: mean of the oriented standardized columns.
  * `iu`: `upcr_fit(Z.T, **IU_FIT_DEFAULTS)` (features x samples; the
    transposition is tested). Abstention or simple-average fallback = declared failure.
  * `shrink` (R5_sel only): full-level joint-target shrinkage,
    `C_alpha = (1-alpha) C + alpha T_joint`, groups {Renyi views} and {SEL},
    `alpha` from `lw_alpha_memory_bounded`, then `upcr_fit_covariance(C_alpha, **IU_FIT_DEFAULTS)`.
  * Every fit stores `score, weights, state, diagnostics, effective, intercept`
    in the fixed 8-column coordinate system and asserts the affine reconstruction
    `X @ effective + intercept == score` at 1e-8.
* NOT APPLICABLE (registered with reasons, never forced):
  * `R5__joint`, `R5_sel__joint`: Joint L-SML needs >= 3 groups of >= 3
    columns; 5 views (+3 SEL) admit no such partition.
  * `R5__shrink`: with one group the joint target equals C, alpha = 0 and the
    arm is identical to `R5__iu`.

## 4. Redundancy diagnostics (mandatory, per answer, saved in DIAGNOSTICS)

Per answer, over the extended matrix
`[H0.5, H1, H2, H4, Hinf, sel1, sel2, sel3, varentropy15, top1_logprob, renyi2_k50]`:
per-view within-answer std with a near-constant flag (std <= 1e-10);
pairwise Pearson and Spearman matrices; condition number of the correlation
matrix of the varying `R5` and `R5_sel` columns; Hartley value/std/constancy;
correlation of K=15 `H2` with the frozen K=50 `topk_renyi2_series`; per-view
anchor correlation; the fraction of answers in which each view is dropped by
`zscore_columns`. The driver aggregates these over answers (mean/median
matrices, fraction of |r| > 0.95 and > 0.99 per pair, condition-number
quantiles, flip fractions per fused arm). The smoke summary lives in
`results/renyi_view_fusion_v1/DIAGNOSTICS.json` (scope SMOKE, 27 answers,
mechanics only).

### Smoke redundancy table (27 designed answers, 9 cells x {shortest, median, 95th pct}; mechanics only)

Source: `results/renyi_view_fusion_v1/DIAGNOSTICS.json` (36 checkpointed rows;
the 9 extra rows scored by a resume defect are summarized separately under
`all_36_rows` and agree with the 27). 0 fit failures in 11 arms x 36 rows;
frozen K=50 Rényi-2/varentropy replay error 0.0 on every row.

| pair | median Pearson | min Pearson | median Spearman | answers with abs(r) > 0.99 |
|---|---|---|---|---|
| H0.5 – H1 | 0.97 | 0.93 | 1.00 | 0/27 |
| H1 – H2 | 0.99 | 0.96 | 1.00 | 0/27 |
| H2 – H4 | 1.00 | 0.99 | 1.00 | 27/27 |
| H4 – Hinf | 1.00 | 0.99 | 1.00 | 27/27 |
| H2 – Hinf | 0.99 | 0.97 | 1.00 | 1/27 |
| H4 / Hinf – top1_logprob | -1.00 | -1.00 | -1.00 | 27/27 |
| H2 (K=15) – renyi2_k50 (K=50) | 1.00 (min per-answer r 0.9955) | 1.00 | 1.00 | 27/27 |
| H0.5 – varentropy15 (anchor) | 0.88 | 0.60 | 0.98 | 0/27 |
| H1 – varentropy15 | 0.77 | 0.27 | 0.97 | 0/27 |
| Hinf – varentropy15 | 0.57 | -0.17 | 0.96 | 0/27 |
| sel1 – any Rényi view | 0.24–0.29 | -0.30 | 0.94 | 0/27 |
| sel1 – sel2 / sel2 – sel3 | 0.84 / 0.99 | 0.76 / 0.96 | 1.00 | 0/27, 8/27 |

Per view: no near-constant view and no view dropped by `zscore_columns` in any
answer (min within-answer std 0.15–0.36 for the Rényi views); Hartley H0 is
exactly log 15 in all answers (max std 1.8e-15). Condition number of the
correlation matrix: R5 median 4.6e4 (range 1.6e4–2.5e5); R5_sel median
4.8e4 (max 2.5e5). Anchor correlation is positive for every Rényi view in
26/27 answers (one shortest PRMBench answer, 20 tokens, has H2/H4/Hinf and
sel1 slightly negative); sel3 is anchor-negative in 22/27 and sel2 in 9/27,
so the per-column orientation flips a monotone power of sel1 in most answers
(1.3 column flips per R5_sel fit on average; no global `_orient` flip in any
fused arm). IU diagnostics: `R5__iu` returns g2_hat = 0.25 = var_y in all
27 answers (residual at the ceiling; the two-component solve finds no rank-1
signal in the five near-collinear orders); `R5_sel__iu` median 0.25, min 0.048;
shrink alpha median 0.15 (0.04–1.0).

Reading (design input, not a result): H2, H4 and Hinf are one view and are
numerically -log p1; H1 sits between them and H0.5; H0.5 is the only order
with distinct within-answer information and is the closest to varentropy.
Spearman near 1 everywhere means all Rényi orders share the same within-answer
ranking; only the Pearson spread (0.88–0.97 between H0.5 and the rest)
distinguishes them, i.e. a level/shape difference rather than a reordering.

Data-path fidelity: every scored answer replays the frozen K=50
`topk_renyi2_series` and `topk_varentropy_series` from the same saved rows
(1e-12), checks canonical step spans and, for PB, the gate detector value.

## 5. Planned comparisons (frozen before the full run)

No best-single-view selection by labels. Primary (97.5 % CI):

1. `R5_sel__iu` vs `R5_sel__equal` — does learned weighting add anything over equal weights?
2. `R5_sel__iu` vs `view__H1` — does the bank add anything over plain entropy15?

Secondary (95 % CI), all listed and reported, none promoted:
each fused arm vs each single view (`H0.5, H1, H2, H4, Hinf, sel1`);
`R5__iu` vs `R5__equal`; `R5_sel__shrink` vs `R5_sel__equal` and vs `R5_sel__iu`;
`R5_sel__iu` vs `R5__iu`; `R5_sel__equal` vs `R5__equal`;
each single view vs `view__H1`.
Context rows without contrasts: frozen `entropy`, `direct_iu` (temporal v3),
raw varentropy15 and varentropy15-IU (Step 339), plus `historical_references`
and the Mind-the-Gap reference carried from the temporal METRICS.

Per contrast: PB macro delta with CI, PRMB within-answer delta on common
answers with CI, exact PB successes gained/lost (CHANGED_SUCCESSES.json),
per-cell F1 (PB_CELLS.csv), coverage/failure reasons and seconds per arm
(FIT_HEALTH.json).

## 6. Failure accounting

Declared failures per arm: nonfinite head (all arms), fewer than 3 tokens,
fewer than 3 varying columns, IU abstention/fallback, zero/nonfinite weights,
constant single view, shrink without both groups among the varying columns.
Failed arms have NaN step scores, count against the full PB denominators and
reduce PRMB coverage; PRMScore is conditional when coverage < 100 %.
The NOT-APPLICABLE arms are not failures; they are absent by construction.

## 7. Outputs

Smoke (`--smoke --workers 1`): `SMOKE.sqlite`, `SMOKE_MANIFEST.json`,
`SMOKE_STATE.json`, `SMOKE.json` (all failures listed), `DIAGNOSTICS.json`,
`FEASIBILITY.json` (per-arm seconds, projected full runtime =
smoke seconds / 27 x 13,769 / workers).
Full (deferred): `CHECKPOINT.sqlite`, `MANIFEST.json`, `METRICS.json`,
`SCORES.npz`, `COMPARISON.csv/json`, `PB_CELLS.csv`, `CHANGED_SUCCESSES.json`,
`FIT_HEALTH.json`, `DIAGNOSTICS.json`, `SUMMARY.csv`, `RUN_STATE.json`.

## 8. Open design questions for the post-review design

* Single-view orientation: natural sign (chosen here, so `view__H1` is exactly
  entropy15) versus anchor-based flipping as for fused columns.
* Whether the five orders are too collinear to be a bank at all (the smoke
  diagnostics report this); candidates are a smaller order set or a
  difference parameterization (`H0.5 - H1`, `H1 - H2`, `H2 - Hinf`) that
  spreads the head shape rather than repeating its level.
* Whether SEL should enter as `[a, a^2, a^3]` or only `a` given the fixed
  IU two-component solve.
* The role of the K=50 `topk_renyi2_series` (different support) as a
  comparator row.
* Whether `R5__shrink`-style single-group arms should be listed as identical
  duplicates for table completeness rather than as NOT APPLICABLE.
