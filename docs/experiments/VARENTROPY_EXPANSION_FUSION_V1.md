# Varentropy expansion (cross-rank) fusion v1: frozen protocol

Extends `VARENTROPY_CONTRIBUTION_FUSION_V1.md` (Step 339). Worktree/branch
`varentropy-expansion-fusion-v1`; outputs only under
`results/varentropy_expansion_fusion_v1/`. No tracked module is modified; the
historical bank is produced by calling `varentropy_contribution_fusion.fit_all`
unchanged. Implementation and smoke only; the full 13,769-answer run is
scheduled separately after review. Smoke metrics are never performance evidence.

## Scientific contract (frozen)

Benchmark: 13,769 answers = ProcessBench 6,800 (8 cells: gsm8k / math /
olympiadbench / omnimath x Qwen3-4B / Qwen3-8B) + PRMBench 6,969. Labels v3
(`results/localization_prm_label_audit_v1/RELEASE_V3.json`, read through the
frozen `results/localization_full_benchmark_v3/evaluation/JOINED.json|npz`).
Source groups / folds v2 (`results/localization_source_group_audit_v1/FOLDS_V2.json`,
3,483 groups, outer folds 0-4). Readout: top-10 token mean per step, then
earliest argmax. ProcessBench gate: frozen external mean-entropy q=0.3 fold
thresholds (`results/fusion_fixed_gate_v1`, read through `old._gate_contract`).
PRMScore: q=0.8 held-fold calibration. Paired source-group bootstrap, 10,000
draws (97.5% CI for the pre-registered primary, 95% otherwise). Fitting is
answer-local and unlabeled: each answer is fitted alone, tokens are the
observations. Explicit declared failures: a failing arm is ABSENT from `fits`
and yields NaN step scores; no other arm is ever substituted. Reported: PB
macro F1 all-8 / Q4 / Q8, PRMB within-answer AUC (+n), pooled AUC, PRMScore
(conditional if coverage < 100%), coverage, runtime. All cached data are
development data, not an untouched test.

Inputs per answer (from the frozen per-cell pickles via `old._source_row_map`):
`top_k_logprobs['logprobs']` [T,50] (float32 saved), `token_entropies`,
`step_token_spans`, `token_spilled_energies` (= -log p(selected token)). Every
processed answer is checked against the frozen benchmark: step spans equal
`BENCH/scores/<uid>.npz`, PB mean entropy equals the gate detector, and the
recomputed top-50 varentropy equals the frozen `topk_varentropy_series` stream.

## Expansion identity and banks

Frozen top-15 convention (exactly `varentropy_contribution_fusion.contributions`):
`q = p/(sum p + 1e-12)`, `s = -log(q + 1e-12)`, `H = sum q s`, `c_i = q_i (s_i - H)^2`.

    V = sum_i q_i s_i^2 - sum_{i,j} q_i q_j s_i s_j
      = sum_i D_i  -  sum_i P_ii  -  2 sum_{i<j} P_ij

* `D_i = q_i s_i^2` (15 columns, second-moment contributions)
* `P_ii = (q_i s_i)^2` (15 columns, diagonal of the double sum)
* `P_ij = q_i q_j s_i s_j`, i<j (105 columns; each unordered pair once, weight 2 in the identity)
* `SEL = [a, a^2, a^3]`, `a = selected_surprisal(token_spilled_energies)` (not renormalized; as in the retained RBM bank)

Column order of the 138-wide matrix: D(0-14), P_ii(15-29), P_ij(30-134,
lexicographic i<j), SEL(135-137). Identity coefficient vector: +1 on D, -1 on
P_ii, -2 on P_ij, 0 on SEL.

Mandatory identity check, per token, on every processed answer:
`sum D - sum P_ii - 2 sum P_ij` must equal `contributions(lp,15).sum(axis=1)`
within 1e-9 absolute (the residual is `H^2 (sum q - 1)`, of order 1e-11 because
of the frozen 1e-12 denominator epsilon). The max discrepancy is recorded per
answer and in FEASIBILITY.json. The identity-weighted fixed fusion of the RAW
bank must reproduce the `k15__raw` step scores within the same 1e-9 (floating
summation order differs, so bit-exactness is not asserted).

Banks (named separately; none silently extended):

| bank | columns | width | status |
|---|---|---|---|
| `B1_hist` | 15 contributions `c_i` (Step 339 bank, `fit_all` unchanged, k15 results) | 15 | historical reference; must reproduce Step 339 METRICS to 1e-11 |
| `B2d_sel` | [D, P_ii, SEL] | 33 | PRIMARY |
| `B2_sel` | [D, P_ii, P_ij, SEL] | 138 | PRIMARY |
| `B2d` | [D, P_ii] | 30 | secondary (no selected block) |
| `B2` | [D, P_ii, P_ij] | 135 | secondary (no selected block) |

## Solvers (on each of B2d_sel / B2_sel / B2d / B2)

All arms z-score the bank inside the answer (`zscore_columns`, keep = scale >
1e-10; constant columns are dropped and carry weight 0). Fewer than 3 tokens
or fewer than 3 varying columns is a declared failure. Note: P_ij columns for
deep ranks can have scale barely above 1e-10; after standardization those
columns are dominated by float32 rounding of the saved logprobs. The count of
kept columns with scale < 1e-6 is recorded per arm (`tiny_scale_columns`).

* `equal_identity`: fixed, label-free, algebra-derived equal-weight arm.
  Coefficients = identity signs (+1 D, -1 P_ii, -1 P_ij, +1 SEL) / n_kept on the
  standardized columns. It is NOT the identity fusion of the raw bank (that one
  is `B1_hist__raw` up to 1e-9); standardization changes the column scales.
* `equal_oriented`: each column's sign is the sign of its within-answer Pearson
  correlation with the anchor (zero or undefined correlation -> +1, declared);
  weights = sign / n_kept.
* `iu`: `upcr_fit(Z.T, **IU_FIT_DEFAULTS)` (features x samples). Abstention or
  simple-average fallback is a declared failure. For >= 64 kept columns
  `upcr._fit_block` takes the closed-form complete-pair inverse (`m>=64` path):
  B2_sel and B2 use it whenever >= 64 columns vary; B2d_sel, B2d and B1_hist use
  the additive-design path. The two paths are algebraically identical.
* `shrink`: full-level shrinkage IU. `C = Z^T Z / n`; groups = term-type x
  rank-block (below); `T = target_matrix(C, groups, 'joint')`;
  `alpha = lw_alpha_memory_bounded(Z, C, T)` (memory-bounded algebraic LW,
  clipped to [0,1], recorded); `upcr_fit_covariance(shrink(C,T,alpha), **IU_FIT_DEFAULTS)`.
* `joint`: Joint L-SML. `covariance_matrix(Z)` (np.cov), labels = groups,
  anchor_index = kept column with the largest |corr| to the anchor (declared),
  seed = sha256('varentropy-expansion-v1:'+uid)[:4], starts = 5, other
  `fit_joint_lsml` defaults (max_sweeps 5000, tolerance 1e-10). Weights =
  `regularized_joint_map_weights(Z, model_covariance, global_loading, mode='diag', lam=0.0)`,
  i.e. the ungated model-inverse map (lambda zero makes the mode irrelevant and
  builds no graph); it applies `regularized_covariance_weights` with
  target_condition 1e3 (analytic ridge, recorded). The joint arm SCORES whenever
  the map is finite; convergence / multistart status are REPORTED per answer,
  not used to withhold scores (withholding would be a hidden selection).

Grouping (term-type x rank-block): D by rank blocks {1-5},{6-10},{11-15};
P_ii by the same three blocks; P_ij by the six unordered block pairs (sizes
10,25,25,10,25,10); SEL one group of 3. Group counts: B2_sel 13, B2d_sel 7,
B2 12, B2d 6. (The briefing wrote "3 for B2d"; the stated rule gives 3 D blocks
+ 3 P_ii blocks = 6, which is what is implemented and declared here.) If after
standardization any group has < 3 varying columns, or fewer than 3 groups
remain, the joint arm is a declared failure naming the group(s).

Caveat (Omri): P_12 and P_13 share q_1 s_1 whatever the grouping; the grouping
is a candidate, not a validated model. Per answer the joint arm therefore
reports: converged flag, multistart audit status, `hard_lsml_misfit`
(correlation-model fit quality) beside the joint relative off-diagonal misfit,
condition number of the model covariance, ridge and post-ridge condition of the
map, Jacobian rank/condition, sweeps, and the minimum global-loading cosine
across converged starts. Joint is judged against IU and shrinkage, not only
against equal weights.

Orientation: every arm's final token score is oriented with `_orient` (higher =
more risk) using the answer's own raw top-15 varentropy as the label-free
anchor (as Step 339 did); the pre-orientation correlation and the flip flag are
recorded. Method ids are `f'{bank}__{solver}'` (23 methods: 3 historical + 4 x 5).
Standardized weights, raw-coordinate effective weights and intercepts are saved
per arm; weights are padded to width 138 with NaN, never zero. Every fit
asserts `X_bank @ effective + intercept == score` at 1e-8.

## Pre-registered comparisons

PRIMARY (97.5% CI): `B2_sel__iu` vs `B2d_sel__iu` (IU-PCR is the frozen primary
solver; same selected block, pair terms added). All others 95%.

Table A, bank comparison (same solver, different bank):

| a | b | role |
|---|---|---|
| B2_sel__iu | B2d_sel__iu | PRIMARY |
| B2_sel__X | B2d_sel__X, X in equal_identity, equal_oriented, shrink, joint | secondary |
| B2__iu | B2d__iu | secondary (no selected block) |
| B2_sel__iu | B2__iu | secondary, selected-block ablation (added) |
| B2d_sel__iu | B2d__iu | secondary, selected-block ablation (added) |
| B2_sel__iu | B1_hist__iu | secondary |
| B2d_sel__iu | B1_hist__iu | secondary |

Table B, architecture comparison (same bank, different solver):

| a | b | role |
|---|---|---|
| B2_sel__joint | B2_sel__iu | secondary |
| B2_sel__shrink | B2_sel__iu | secondary |
| B2d_sel__joint | B2d_sel__iu | secondary (added) |
| B2d_sel__shrink | B2d_sel__iu | secondary (added) |
| B2_sel__iu | B2_sel__equal_identity | secondary (added) |
| B2d_sel__iu | B2d_sel__equal_identity | secondary (added) |

Table C, references: every arm vs `entropy` (frozen temporal-v3 entropy step
scores) and vs var15 raw (`B1_hist__raw`, which must equal the Step 339
`k15__raw` scores). Frozen references registered exactly as the Step 339 driver
does: entropy / direct_iu / delta_iu from `direct_probability_temporal_v3`
(asserted equal to its METRICS at 1e-12) and `ref__k15__*`, `ref__k50__raw` from
`varentropy_contribution_fusion_v1` (asserted at 1e-12); `historical_references`
and `mind_gap_reference` are copied through. No promotion threshold.

Declared behaviours fixed after the pre-run review (2026-09-12):

* `B1_hist__equal` is NOT anchor-oriented: it is the frozen Step 339 EQUAL arm
  (`mean(zscore(C))`, no learned sign flip), reproduced by calling that module.
* Non-converged Joint fits (multistart BLOCKED, or the 5000-sweep cap) are
  SCORED and flagged per answer; FIT_HEALTH reports `converged_rate`,
  `multistart_pass_rate` and `model_covariance_condition_over_1e12_rate`.
* The shrinkage alpha is expected to clip at 1.0 on most answers (the smoke
  clipped on 27/27 pair-bank and 26/27 diagonal-bank answers); FIT_HEALTH
  reports `alpha_clipped_rate`. With alpha = 1 the `shrink` arm is IU on the
  rank-1-completed cross-group target.
* `equal_identity` on the pair banks (B2, B2_sel) anti-correlates with the raw
  varentropy anchor and is flipped by `_orient` on essentially every answer;
  the scored arm is therefore the ORIENTED identity-sign equal fusion and must
  be named so in reports ("identity-sign equal weights, anchor-oriented").
* FIT_HEALTH also reports `n_lt_p_fits`: answers whose token count is below
  the number of active columns (n < p fits; expected for short answers on the
  135/138-column banks).

## Failure accounting

Declared failures: < 3 tokens / varying columns; IU abstention or simple-average
fallback; joint K < 3 groups or a group with < 3 varying columns (group named);
nonfinite / zero fusion or malformed solver output. Each failure is stored as
`(uid, method, reason)`; PB failures count against full denominators; PRMB
within-AUC reports its eligible count; PRMScore is conditional (headline null)
if coverage < 100%. SMOKE status is PASS only if every failure is declared.
Runtime per arm is recorded from `time.perf_counter()` around each fit.

NOT declared failures: a bank-build or frozen-input fidelity assertion (step
spans vs BENCH/scores, PB gate detector, raw varentropy-50 replay, Step 339
step-score replay, identity residual > 1e-9, identity-weighted fusion vs k15
raw) ABORTS the run for investigation: it signals a frozen-input or code
problem, never an answer-level fitting condition.

Supervised panel: a fold fit whose optimizer never left the origin
(theta_norm < 1e-8) or exits with max gradient > 1e-2 is recorded as STALLED
(declared; scores nothing; never reported as FIT/converged). No perturbed
restart is used.

## Supervised diagnostic panel (separate; smoke only here)

`spectral_utils/varentropy_expansion_supervised.py` +
`scripts/run_varentropy_expansion_supervised_v1.py`, output under
`results/varentropy_expansion_fusion_v1/supervised/`. Per cell, 5 outer
source-group folds (FOLDS_V2 outer). Linear token score `w.z + b` on the
answer-locally standardized B2_sel (and B2d_sel = column subset) columns
(constant columns -> 0). Step score = the SAME top-10 token mean, differentiable
through the selected tokens (stable ties -> earlier tokens; verified equal to
`rbm_matched_top10.top10_value_gradient`). Loss = class-balanced step-level BCE
(each class total mass 0.5 over the training steps; PB: prefix steps 0,
annotated first-error step 1, later steps EXCLUDED, clean answers all 0;
PRMB: step labels >= 0, unknown excluded) + ridge 0.01/2 ||w||^2; L-BFGS-B;
score the held fold; then the same gate / readout / metrics. No token-label
broadcasting. Access label: supervised, other-answer access; a matched
diagnostic, not a ceiling. Bank matrices are cached as float32 (a PRMB training
fold is ~2.6M x 138 tokens; 1.1 GB float32 vs 2.3 GB float64). On the 27-answer
smoke most folds lack a class and are declared fold failures: mechanics only.

## Resource rule (shared machine)

Before loading a cell pickle the drivers wait for free physical memory above a
gate (poll every 5 min, up to 90 min, then BLOCKED): 1.2 GB for ProcessBench
cells, 2.0 GB for the PRMBench cell (main-agent instruction, 2026-09-12; the
initial 3.0 GB guard was replaced before any answer beyond the discarded first
three was scored). One worker for smokes; threadpool limit 1.

## What is NOT claimed

No localization improvement is claimed from this protocol; the smoke gives
feasibility only. Signed fusion outputs are scores, not variances. The
grouping is a candidate structure, not a validated covariance model. Joint
convergence statistics are reported, not gated. The supervised panel uses
other answers' labels and is not a ceiling. No historical24 transfer, no new
graph, no temporal lag, no parameter sweep, no untouched confirmation.
