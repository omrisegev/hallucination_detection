# Renyi-view fusion v2 — Stage 3 (design authorized by Omri, 2026-09-13)

**Question.** Does a *combination* of several Renyi orders of the top-15 head
localize better than one entropy? Omri, 2026-09-13: "we were not looking for an
alternative entropy; we wanted to see whether a combination of several orders,
e.g. 0.1, 0.5, 1, 2, can beat a single entropy." This is a fusion question on the
frozen contract, answered against the single views, the frozen entropy /
varentropy references and simple equal weights.

**Why the v1 grid was replaced.** The v1 prototype (`RENYI_VIEW_FUSION_V1_DRAFT.md`,
alpha in {0.5, 1, 2, 4, inf}, 27-answer smoke) fused five orders but carried only
three distinct directions: on the retained top-15 support `H2`, `H4` and `Hinf`
are pairwise |Pearson| > 0.99 on 100 % of smoke answers (all dominated by `q_1`),
while `H0.5` vs `H1` and `H1` vs `H2` are > 0.95 but never > 0.99. The order
grid where the views actually differ is alpha < 1 (tail-sensitive). v1 stays
frozen as the prototype; nothing in it is modified.

Branch/worktree: `claude/varentropy-expansion-fusion-v1`
(`.worktrees/varentropy-expansion-fusion-v1`). New files only:
`spectral_utils/renyi_view_fusion_v2.py` (imports the v1 view definitions
unchanged), `scripts/run_renyi_view_fusion_v2.py`, `scripts/test_renyi_view_fusion_v2.py`,
`scripts/review_renyi_view_fusion_v2.py`, this document, and `results/renyi_view_fusion_v2/`.

## 1. Contract (frozen; identical to every localization experiment)

Unchanged from v1 §1: 13,769 answers (ProcessBench 6,800 in 8 cells + PRMBench
6,969); JOINED v3 labels; FOLDS_V2 source groups; one teacher-forced pass, saved
top-50 log-probabilities and selected-token surprisal; per-step top-10 token
mean, earliest argmax; external mean-entropy gate q = 0.3 (`fusion_fixed_gate_v1`);
PRMScore q = 0.8 on held folds; 10,000-draw paired source-group bootstrap
(97.5 % for primaries, 95 % otherwise); answer-local unlabeled fitting; declared
failures (NaN, counted against full denominators, never substituted); reported
PB all-8 / Q4 / Q8, PRMB within (+n) and pooled, PRMScore (conditional if
coverage < 1), coverage, runtime. Development data, not untouched confirmation.

## 2. Views

`q = p/(sum p + 1e-12)` on the descending top-15, `s = -log(q + 1e-12)`;
`H_alpha = log(sum q^alpha)/(1 - alpha)`, `H_1 = sum q s` (== frozen entropy15,
tested to 1e-12 on real rows), `H_inf = -log(max q + 1e-12) = s_1`.

| view | alpha | role |
|---|---|---|
| `H0.1`, `H0.25`, `H0.5` | < 1 | tail-sensitive orders (new: 0.1, 0.25) |
| `H1` | 1 | Shannon; the single-entropy reference inside the bank |
| `H2`, `Hinf` | >= 1 | head-concentration orders; alpha = 4 dropped as a duplicate of both (v1 smoke) |
| `H0` | 0 | constant log 15 on the support; diagnostics only |

Selected-token block `SEL = [a, a^2, a^3]`, `a = -log p(selected)`.

## 3. Banks, solvers, groups

* `R6` = six Renyi views; `R6_sel` = R6 + SEL (nine columns).
* Single views: `view__H0.1 … view__Hinf`, `view__sel1` (raw, natural sign).
* Fused: `R6__equal`, `R6__iu`, `R6_sel__equal`, `R6_sel__iu`, `R6_sel__shrink`, `R6_sel__joint`.
  * z-score on the answer's own tokens; constant columns dropped and counted;
    < 3 tokens or < 3 varying columns = declared failure; orientation anchor =
    the answer's own raw K=15 varentropy (label-free; as Step 339 / v1).
  * `iu`: two-component L2 IU-PCR, `IU_FIT_DEFAULTS`; abstention = failure.
  * `shrink`: joint-target Ledoit-Wolf shrinkage, groups {Renyi} vs {SEL}.
  * `joint`: Joint L-SML, lambda 0, model-inverse map, five starts, seed from
    the answer uid, with the **declared** partition tail = {H0.1, H0.25, H0.5},
    head = {H1, H2, Hinf}, sel = {sel1, sel2, sel3}. Justification: three
    generating mechanisms (tail mass of the head distribution, concentration of
    the top mass, realized token). The partition is a candidate model: all six
    Renyi views are functions of the same `q`, so group-conditional independence
    is approximate at best. The arm therefore reports fit quality (relative
    off-diagonal misfit versus the hard-partition fit, multistart audit,
    convergence, Jacobian rank) and conditioning (model covariance condition,
    map ridge), and is judged against IU and shrinkage, not only against equal.
  * `R6__joint` (two groups) NOT APPLICABLE, never forced; `R6__shrink` == `R6__iu`.
* Historical rows in the same table: frozen token entropy, frozen direct IU,
  Step-339 varentropy15 raw and IU (reproduced to 1e-12 by the evaluator);
  `view__H1` must reproduce the frozen entropy metrics exactly (asserted).

## 4. Primary and secondary contrasts

Primary (97.5 %): `R6__iu − view__H1`, `R6__equal − view__H1`,
`R6_sel__iu − view__H1`, `R6__iu − R6__equal` (learned versus simple aggregation).
Secondary (95 %): every fused arm against every single view and against the
varentropy15 references; shrink/joint against IU and equal; `_sel` against the
same bank without SEL; each single view against `H1`.

## 5. Diagnostics reported before ranking

Per-answer |Pearson| and |Spearman| matrices over the nine columns plus
varentropy15, `log p_1` and the K=50 Renyi-2; fractions > 0.95 / 0.99; per-view
std, near-constant and dropped fractions; condition numbers of R6, R6_sel and
the tail block; Hartley constancy; shrink alpha (fraction at 1); IU `g2_hat`;
joint fit quality and conditioning as above. A view that is constant or
duplicates another on most answers is reported as such; the bank is not pruned
post hoc.

## 6. Execution

Tests → smoke (27 designed answers; mechanics/feasibility only, no ranking) →
independent smoke replay (`review_renyi_view_fusion_v2.py --smoke`) → full run
(`--allow-full`, checkpointed) → evaluation → independent replay and metric
re-derivation → figure review. Runtime disclosed with machine contention.

## 7. Reading rules

Small runs establish feasibility only. A point lead without a paired interval
excluding zero on both PB and within-answer AUC is not a winner. The required
framing applies: we have not yet demonstrated a consistent overall advantage
from learned fusion; representation, optimization, normalization and readout
remain partly entangled.
