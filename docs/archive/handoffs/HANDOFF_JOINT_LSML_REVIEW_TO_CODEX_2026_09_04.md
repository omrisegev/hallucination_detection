# Review of Joint L-SML v1 — for Codex (from Claude, 2026-09-04)

**Reviewed**: commit `c5a658a` ("research: publish Joint L-SML structural study") on
`codex/og-sml-agent-b-v1`, at Omri's request. Read in full:
`spectral_utils/joint_lsml.py`, `spectral_utils/og_sml_graph.py`,
`scripts/og_sml_agent_b/run_joint_lsml_v1.py`, both test files,
`results/joint_lsml_v1_r2/{REPORT.md,INDEPENDENT_AUDIT.md,JOINT_STRUCTURAL_LANES.csv}`,
`results/og_sml_agent_b_v1/T0_REPORT.md`,
`docs/experiments/JOINT_LSML_V1.md`, and
`docs/experiments/JOINT_LSML_LOCALIZATION_EVALUATION_V1.md`.

**Independently verified**: all 14 tests in `tests/test_joint_lsml.py` +
`tests/test_og_sml_graph.py` pass on a second machine (Windows 11, Python 3.13,
5.7 s). The math of the estimator was checked by hand (see §2). No scoring, no
labels, no registration — this document is a review only and grants no scoring
authority.

---

## 1. Verdict

Solid, unusually disciplined work. The structural phase is citable as-is:
preregistered, falsification-honoring (T0 stop), parameter-fair, independently
audited to 1e-13..1e-16, tested. The drafted localization evaluation is the
right shape. **Endorsed for the scoring phase, conditional on resolving the
three concerns in §3 — all of which are pre-label changes, so now is the only
time they can be made cheaply.**

## 2. What was verified positively

- **Estimator math is correct.** The Gauss–Seidel coordinate updates are exact
  scalar least-squares minimizers, so the objective is provably monotone; the
  per-sweep NNLS amplitude rescale can only decrease it (amplitude (1,1) is
  always feasible). The monotonicity tolerance correctly absorbs only fp noise.
- **The 16/16 misfit win is parameter-fair.** Joint (`v` + `u`: 2p params) vs.
  the hard two-stage fit in `hard_lsml_misfit` (within + between loadings: 2p
  params). Equal budgets, same partition, same objective. The improvement
  (0.026–0.083 absolute relative-misfit) is a genuine structural finding, not an
  expressivity giveaway — with one attribution caveat, see §4.1.
- **Fail-closed engineering throughout**: frozen-roster and fit-index hash
  checks, C-v2 covariance re-derivation with abort at >1e-12 drift, protected
  entropy anchor that aborts instead of silently restoring, bit-exact alias
  dispatch to `sml_fuse_signed` / `lsml_continuous`, and the string-level
  no-outcome-imports firewall test.
- **T0 hygiene**: the preregistered retrospective prediction failed and the
  overlapping program was stopped, not rationalized. The failed first run
  (`joint_lsml_v1_r1`) → continuation (`joint_lsml_v1_r2`) change is confined to
  failure handling. Both are the right calls and worth keeping citable.
- **Prior-free alignment (Step 199)**: orientation, roster, K, and partition are
  all derived from data structure. The independent removal of
  `entropy_pe_series` (weak + sign-unstable) reproduces the old "pe_mean
  adaptively suppressed" finding from the continuous L-SML canonical-state work
  — a nice convergence.

## 3. Main concerns (ordered), with suggested solutions

### 3.1 Every fitted lane selected K=3 or K=4 → the frozen candidate's SML stage runs at (or near) zero degrees of freedom

From `JOINT_STRUCTURAL_LANES.csv`: 14 lanes chose K=3, 2 chose K=4. The frozen
candidate map `hierarchical_joint` runs signed SML across only K virtual
classifiers. At K=3 the rank-1 SML fit has 3 off-diagonal covariance entries
and 3 unknowns: exact interpolation, no noise averaging — any covariance noise
maps directly into the cross-group weights `a_g`. Step 205 already documented
fragility of the L-SML family at sizes 3–4 in a related setting.

Compounding this: **median LOAO ARI is 1.0 in every fitted lane**, so the
primary selection criterion saturates and the "smaller K" tie-break is doing
real work. K=3's dominance may partly be an artifact of the tie-break rather
than the data preferring 3 groups.

Suggested solutions (all label-free, pre-registration):

- **(practical)** Add a LOAO stability diagnostic on the cross-group SML
  weights `a_g` themselves — exactly analogous to Task 1's ridge diagnostic:
  recompute `a_g` per held-answer fold, report min/median pairwise Spearman of
  the resulting donor scores. If K=3 lanes are unstable here, that is decisive
  pre-label evidence to prefer larger K or to gate those lanes.
- **(practical)** When median ARI saturates (ties at 1.0 across multiple K),
  break ties by **mean ARI, then minimum ARI, then smaller K** — minimum ARI is
  already computed and discriminates (it spans 0.27–1.0 across lanes).
- **(general)** Report per-K weight-map agreement as a sensitivity row in the
  structural table, so the K choice's practical consequence is visible before
  labels.

### 3.2 The four weight maps disagree most on exactly the lane the candidate uses

Minimum pairwise donor-score Spearman among the four maps: **0.708–0.879 on
`v2_active28` lanes** vs. 0.917–0.976 on `h2_24` lanes. The fused score is
genuinely sensitive to the map choice on the candidate's own lane.

The protocol already does the right thing — `hierarchical_joint` is frozen
before labels with a stated label-free rationale (closest extension of
continuous L-SML; independent of the frequently-clipped fitted diagonal). Keep
that freeze absolute.

Suggested solutions:

- **(practical)** Pre-register the freeze irrevocably in the preregistry (it is
  currently in a DRAFT doc): after scores are visible, no map substitution under
  any wording. The two inverse maps stay structural diagnostics only.
- **(practical)** Add one pre-label agreement diagnostic on the *new scorer
  population's donor rows* (Spearman of `hierarchical_joint` vs. the other
  three, no labels needed). If agreement collapses on the new population
  (<~0.5), that is a legitimate label-free abort/flag criterion — decide the
  threshold now, not after.
- **(general)** In any advisor-facing write-up, present the map disagreement
  range as an explicit uncertainty statement about the candidate, not a
  footnote.

### 3.3 The all-folds LOAO admissibility rule gets harsher as cells grow, and both blocked lanes are the candidate's own lane type

Admissibility requires the consensus AND **every** held-answer fold to keep all
K groups at size ≥3 — an all-folds intersection whose blocking probability
rises with the number of answers even under a genuinely stable partition. Two
of nine active-23 lanes blocked (`processbench_math_qwen3_4b`,
`processbench_omnimath_qwen3_8b`); minimum LOAO ARI drops to 0.27–0.32 in some
passing lanes despite perfect medians. The draft protocol's consequence is
severe: one blocked cell ends its entire benchmark panel as
`STRUCTURAL_NO_SCORE`, no fallback.

With a ~2/9 observed base rate, the Phi-4 run has a real chance of losing the
ProcessBench panel structurally. That may be acceptable — but decide it now.

Suggested solutions:

- **(practical)** Before registering, estimate the blocking probability on the
  donor cells by subsampling answers to the expected new-population sizes and
  re-running admissibility. If blocking probability is high, either accept it
  in writing or change the rule *pre-registration*.
- **(general)** If the rule changes, the principled softening is a quantile
  rule (e.g. "≥95% of held-answer folds admissible") rather than the minimum —
  it keeps the stability semantics while removing the n-scaling pathology.
  Changing it after label access is not an option, so this is now-or-never.
- **(practical)** Alternatively, pre-register the panel-death policy explicitly:
  "if any ProcessBench subset cell blocks, the panel reports
  STRUCTURAL_NO_SCORE and the experiment is still considered fully executed" —
  so a blocked outcome cannot be reframed later.

## 4. Smaller items

### 4.1 Attribution of the misfit win

Joint's within-blocks are rank-2 (`v` + `u`) while hard's are rank-1; the model
classes don't nest, so part of the 16/16 win could be "within-blocks want
rank 2" rather than "the shared factor is real". Cheap ablation: free rank-1
between + two free per-block loadings (3p params, upper bound). If that 3p
model barely beats joint's 2p, the shared-factor story is strongly supported.
Not required for the scoring phase; strengthens the thesis chapter.

### 4.2 Pruning has no efficacy readout by design

All four arms of the next experiment consume the same pruned 23-column matrix,
so the run carries no information on whether pruning helped or hurt. Step 206's
pool-composition lesson (removing views hurt U-PCR, −0.50pp) was a different
channel, but the category of risk is the same. Option: add an *unpruned-28*
structural diagnostic lane (no scoring arm, no label access) so at least the
structural cost of pruning is on record.

### 4.3 Code nits (none blocking)

- `_fit_one_start`: an NNLS amplitude of exactly 0 permanently zeroes a factor
  (subsequent coordinate updates cannot revive it — `denominator <= EPS` →
  `new_value = 0`). Multistart agreement would likely catch a degenerate
  outcome, but the absorbing state is undocumented. Add a comment or a
  diagnostic counter.
- Bare `except Exception` fallbacks around `_rank1_masked` (in
  `residual_affinity`, `_complete_initializer`, `_initial_loadings`) could mask
  real bugs; consider catching the specific expected failure or logging which
  path was taken.
- The 9-donor-cell contract is hardcoded in `consensus_orientation_and_roster`
  and `global_degree_roster`; fine for this protocol, worth a docstring note
  that it is a protocol constant, not an algorithmic requirement.
- The fixed-family continuous-L-SML control contains singleton and 2-stream
  groups (1,3,8,2,3,6). The protocol correctly disclaims theorem validity;
  given Step 205, treat that arm's numbers as an algorithmic control only —
  never quotable as "hard L-SML" in any advisor-facing table.
- `results/joint_lsml_v1_r2/REPORT.md` contains absolute macOS paths
  (`/Users/osegev/...`) in the Agent-A handoff section; the registries are
  committed in-repo, so consider repo-relative paths for portability.

### 4.4 Honest-attribution bundling

The protocol itself notes a Joint win cannot be attributed between automatic
grouping and joint fitting — they are bundled in the single candidate. Fine
for v1; expect the advisors to ask. A post-hoc decomposition (joint fit on the
frozen provenance families; continuous L-SML on the learned partition — the
latter already exists as the same-partition diagnostic) can be pre-registered
as secondary analyses now at near-zero cost.

## 5. Suggested pre-registration checklist (condensed)

1. Add LOAO cross-group-weight stability diagnostic (§3.1).
2. Change K tie-break to mean → minimum ARI when medians saturate (§3.1) — or
   explicitly re-affirm the current rule in the preregistry.
3. Make the `hierarchical_joint` freeze irrevocable in the preregistry, with a
   numeric pre-label agreement abort threshold on the new population (§3.2).
4. Estimate blocking probability at the new population sizes; either soften the
   admissibility rule to a quantile now, or pre-register the panel-death policy
   in writing (§3.3).
5. Optional: unpruned-28 structural diagnostic lane (§4.2); 3p attribution
   ablation (§4.1); secondary attribution analyses (§4.4).

— Claude (review requested by Omri, 2026-09-04). Tests re-run and passing on a
second machine; no result namespace was edited; no scoring authority is opened
by this document.
