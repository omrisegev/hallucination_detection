# DEEM vs IU-PCR 24-cell v1 — execution record, results, and gate critique

Companion to the frozen protocol `DEEM_VS_IUPCR_24CELL_V1.md`.  Three purposes:

1. the complete execution record of the run that produced the registered
   decision (chronology, failures, amendments, job IDs, commit trail);
2. the final results in one place;
3. a post-hoc **critique of the conditional max-null gate**, written after the
   registered decision was emitted, with concrete numbers from the frozen
   evaluation artifacts — plus a menu of v2 variants for Omri + Codex + Claude
   to compare before any new run is authorized.

Authored by Claude (2026-08-23) at Omri's request, after the registered
decision was final and pushed.  Nothing in this document reopens the v1
decision: `CONTINUOUS_DEEM_NONINFERIOR_TO_IUPCR`, `eligible_for_B999=false`,
graph closure unchanged.

---

## 1. Execution record

### 1.1 Predecessor: residual-graph phase0 (archived)

- **Jobs 217597–217627** (2026-08-22): phase0 crashed with a NaN inside the
  nuisance-encoder ZCA whitening — `torch.linalg.eigh` backward is undefined at
  (near-)degenerate spectra, and the diverging nuisance head produced exactly
  that.  Root cause and evidence: `PHASE0_FAILURE_REPORT.md`.
- Fix (Codex-designed, Claude-implemented under Omri's direction): ridge-
  Cholesky whitening (`cholesky` + `solve_triangular`), equivalent to ZCA up to
  an orthogonal rotation under which every penalty in the objective is
  invariant; plus finite checks on gradients and parameters
  (commit `645a405`).
- **Job 218512**: phase0 re-ran numerically healthy on all ten synthetic
  worlds and stopped on its own frozen criterion — target-graph effect without
  specificity.  A legitimate scientific stop, not a defect.  Codex archived
  the graph experiment and authored this successor benchmark.

### 1.2 Benchmark chain attempts

| Attempt | Lead job | Stopped at | Root cause | Response |
|---|---|---|---|---|
| 1 | 219646 | adapter boundary | `CUBLAS_WORKSPACE_CONFIG` unset → every deterministic CUDA fit raised | export in sbatch (`8457893`) |
| 2 | 219682 | adapter boundary | B2 (repaired soft/rank `deem==0.2.0`) collapses on the 30-feature fixture on every seed (`score_sd` 1.1e-6–1.3e-4 vs the 1e-3 floor), deterministic | **Amendment A1** (`11641fc`): B2 health recorded, not blocking; B0/B1/B3 still hard-gated |
| 3 | 220081 | Stage A, 4 cells | same B2 degeneracy, different exit: collapsed posterior → "risk-consensus alignment is ambiguous" raise, so A1 had nothing to record | **Amendment A1.1** (`2c61b37`): identity fallback on ambiguous alignment, worker-only opt-in, default `raise` preserved |
| 4 | 220153 | evaluate-199 | the shared evaluator's `load_seed_score` was a **fifth** enforcement site of the pre-A1 hard health gate (after worker exit, preflight, cell checkpoint, FIT_COMPLETE); crashed loading the recorded-degenerate B2 fits — found post-label-open | **Amendment A1.2** (`43e87f8`): loader policy flag + constant-score Spearman → 0.0; disclosed; full restart from the pre-label boundary under a fresh identity |
| 5 | **220244–220274** | — | clean end-to-end | registered decision + rebuild verification **pass** |

Also restored en route (data, not code): the Drive source for
`internalstates_gsm8k_qwen25_7b` was the pre-Z_n-backfill file (75 MB vs the
registered 146 MB).  Restored byte-exact from Git LFS (pointer oid == registry
sha256), stale copies archived under `archive_pre_registry_freeze/`, all 24
canonical manifests deployed to the cluster's `code/dataset_cache/repgrid/`.

Preflight narrative: `DEEM_VS_IUPCR_PREFLIGHT_REPORT.md`.  Commit trail on
this branch: `c95bca6 → 645a405 → 4a7d2fa → 8457893 → 7d0e26f → 11641fc →
2c61b37 → 43e87f8 → 9e99698` (results).  Large artifacts:
`gdrive:hallucination_detection/cluster_results/deem_vs_iupcr_24cell_v1/`.

---

## 2. Final results (registered, B=199, chain 220244–220274 on `43e87f8`)

**Decision: `CONTINUOUS_DEEM_NONINFERIOR_TO_IUPCR`. `eligible_for_B999=false`
(final — the promotion lane was never submitted).**

Primary contrast B3−B0, equal-family AUROC: **+0.005691**, paired
family-blocked bootstrap 95% CI **[+0.001258, +0.009752]**, Holm p **0.0061**,
**17W/1T/6L**, worst cell −0.0095, QA +0.0050 / math +0.0082 (both within
limits), `b3_stability_pass=true`.  Every bootstrap gate passed; the sole
blocker was the conditional max-null family (§3).

| Equal-family macro | B0 (IU-PCR) | B1 (hard) | B2 (soft/rank) | B3 (continuous) |
|---|---|---|---|---|
| AUROC | 0.7428 | 0.7082 | 0.6780 | **0.7485** |
| AUPRC | 0.7749 | 0.7195 | 0.7169 | **0.7780** |

B3−B1: +0.0403, CI [+0.0267, +0.0544], 24/0/0.  B3−B2: +0.0705, CI
[+0.0201, +0.1301], 21/1/2 — must be read with the recorded B2 health tables
(protocol interpretation rule): B2 is degenerate on the wide inventories and
sits at exactly 0.5 on two whole families below.

Per-family AUROC:

| family | B0 | B1 | B2 | B3 | B3−B0 |
|---|---|---|---|---|---|
| gsm8k | 0.7694 | 0.7119 | 0.7529 | 0.7781 | +0.0087 |
| hotpotqa | 0.5696 | 0.5664 | 0.5768 | 0.5836 | +0.0140 |
| math500 | 0.8142 | 0.8062 | 0.7576 | 0.8212 | +0.0071 |
| nq_open | 0.7469 | 0.7202 | 0.5000 | 0.7409 | −0.0060 |
| sciq | 0.7448 | 0.7075 | 0.7184 | 0.7464 | +0.0016 |
| squad_v2 | 0.8103 | 0.7643 | 0.8066 | 0.8191 | +0.0089 |
| triviaqa | 0.8243 | 0.7863 | 0.8115 | 0.8292 | +0.0049 |
| truthfulqa | 0.6631 | 0.6031 | 0.5000 | 0.6695 | +0.0064 |

B3 wins 7/8 families; the one loss (nq_open, −0.60pp) is smaller than five of
the seven wins.  Rebuild verification: **pass** — a genuinely fresh Stage A
(all 480 fits) reproduced every evaluation output byte-identically,
DECISION.json included.

**What B3 is** (for readers landing here first): the graph-free continuous
additive DEEM — per registered feature family g,
`c_g = w_g·x_g + (2/|g|)·tanh(V_g·tanh(W_g·x_g+d_g)+e_g)` (family width 8),
`ℓ = b + Σ_g 1ᵀ c_g`, trained label-free by free-energy contrast with
persistent MALA, float64 CPU, 100 epochs, 5-seed ensemble, risk-consensus
orientation.  It shares the equal-family/equal-feature prior with B0 (`w_g`
initialized to `2/(G·|g|)`) and contains **no graph term** (`lambda_=0`, no
Laplacian).  Its only structural advantage over the linear IU-PCR inventory
combination is the bounded within-family nonlinearity.

---

## 3. Critique of the conditional max-null gate (post-hoc, does not reopen v1)

### 3.1 What the gate actually computes

`whole_search_null` forms **18 statistics**: 3 contrasts (B3−B0, B3−B1,
B3−B2) × 2 metrics (AUROC, AUPRC) × 3 scopes (equal-family, QA, math).  For
each null family (exact-length permutation, cross-fitted propensity CRT,
family/group-blocked) it draws B=199 label resamples, recomputes all 18
**raw, unstandardized** deltas per draw, takes the **max over all 18**, and
compares each observed statistic against that max distribution
(`p_by_statistic`).  The superiority gate consumed
`p_by_statistic["B0|B3|auroc|equal_family"]`.

This is a single-step max test **without studentization**.  Its known failure
mode: when the family mixes statistics of very different noise scales, the max
is owned by the noisiest member, and every low-variance member is compared
against noise that is not its own.

### 3.2 The frozen numbers show exactly that failure mode

From `evaluation/B199/WHOLE_SEARCH_NULL.json`:

| Quantity | exact | crt | family_group |
|---|---|---|---|
| global p (observed_max = +0.1241, from B1\|B3 auprc\|math) | 0.005 | 0.005 | 0.005 |
| null_max_mean | 0.0868 | 0.0575 | **0.0072** |
| null_max_q95 | 0.0915 | 0.0637 | **0.0124** |
| p_by_statistic, B0\|B3 auroc\|equal_family (observed +0.0057) | 1.0 | 1.0 | 0.67 |

Readings:

1. **The global test passed at the empirical floor (p = 1/200) in all three
   nulls.**  The labels demonstrably carry signal aligned with the arm
   differences; "no advantage anywhere" is rejected as strongly as B=199 can
   reject anything.
2. **The exact/crt null-max scale (0.06–0.09) is AUPRC/math noise, not a
   distribution of realistic AUROC gains.**  Under full label permutation the
   math-macro AUPRC delta swings by ~±0.09; the max inherits that.  The
   claim "the null generates 5.75pp, ten times the observed gain" therefore
   does **not** mean "a competing method typically gains 5.75pp by luck" —
   it means the loudest statistic in an unstandardized family is loud.
3. **The most structure-preserving null (family_group) has a max scale of
   0.0072** — the same order as the observed +0.0057 — and even there the
   primary statistic was charged against the max of 18, not against its own
   marginal.  Its own marginal p under family_group was never computed
   (only max-based `p_by_statistic` exists in v1).
4. **Under family_group, B1|B3 and B2|B3 pass all six of their statistics at
   p=0.005 each.**  The gate can be passed; the primary contrast fails it
   specifically because it is the quietest statistic measured against the
   loudest one's noise.
5. **The right empirical reference scale for "how big is +0.57pp?"** is this
   project's own history of matched comparisons on these caches: removing
   inventory views moves the deployed arm by ~−0.50pp (Step 206), adding the
   two strongest unused views is negative, the whole published-selector
   family sits at or below the floor (Steps 221–222), and the DEEM package
   adapters lose by 4.0–7.1pp here.  Against that distribution a +0.57pp
   equal-family gain with a CI excluding zero, 17W/1T/6L, 7/8 families
   positive, full seed stability, and byte-identical rebuild is a rare and
   genuine observation — Omri's reading, which this analysis confirms.

### 3.3 What stands regardless

The registered decision stands as registered — modifying a gate after seeing
the data is the forking path the freeze exists to prevent.  The correct public
sentence is:

> Continuous additive DEEM shows strong evidence of a small real advantage
> over IU-PCR (+0.57pp equal-family AUROC, 95% CI [+0.13, +0.98], Holm
> p=0.006, 17W/1T/6L), which did not clear a preregistered multiplicity gate
> that post-hoc analysis shows to be dominated by the noise of an unrelated
> statistic (math-scope AUPRC).  The registered claim is noninferiority.

Not "no advantage", and not "superiority" either.

---

## 4. v2 variants to compare (Codex + Claude + Omri — discussion first, no runs)

Per the project rule (`feedback_tailor_not_transplant`): **one variant, one
discussion, then build** — nothing below is authorized to run, and any v2 is a
new frozen protocol with a fresh run identity.  Two families of variants:

### 4.1 Gate variants (fix the multiplicity scheme)

| ID | Variant | Mechanism | Cost / risk |
|---|---|---|---|
| G-A | Studentized max (maxT) | divide each statistic by its own null SD before taking the max | needs per-statistic null SDs (cheap; same draws); standard, defensible |
| G-B | Westfall–Young step-down | max on the p-value scale with step-down | strictly more powerful than G-A's single-step; slightly more machinery |
| G-C | Split the family | primary max over the 3 equal-family **AUROC** contrasts only; AUPRC + QA/math scopes demoted to descriptive secondaries | smallest change; must be justified as design (AUROC was always the named primary metric) not as data-peeking |
| G-D | Marginal blocked null | per-statistic marginal p under family_group, Holm across the 3 contrasts | abandons the "whole-search" framing entirely; simplest to interpret |

Validation options, in decreasing order of strength:
(i) preregister the corrected gate and evaluate on **new cells** (fresh
models/datasets — the only option that yields a clean superiority claim);
(ii) apply the corrected gate to the frozen v1 scores as a clearly-labelled
post-hoc sensitivity analysis (informative, never headline).

### 4.2 B3 architecture variants (isolate where the gain comes from)

| ID | Variant | Question it answers |
|---|---|---|
| M-A | Linear-only ablation (`w_g` terms only, no tanh net) | is the within-family nonlinearity the source of the +0.57, or the free-energy training itself? |
| M-B | No family partition (one global net, same parameter count) | how much of B3's edge is the family prior it shares with B0? |
| M-C | Family width sweep (4 / 8 / 16) | is width-8 sitting on a ridge? |
| M-D | Cross-family interaction term (bounded, rank-1) | does the additive-across-families restriction cost anything? |
| M-E | Ensemble size (1 / 3 / 5 seeds) | how much of the edge is ensembling, which B0 does not get? |

M-A and M-E are the honest-comparison priorities: if either explains the gain,
the "continuous DEEM" framing changes.  **Omri has additional specific
variants in mind — this table is an opening menu, and his list takes
precedence; record it here before any protocol is drafted.**

### 4.3 Hard constraints carried into any v2

- `eligible_for_B999=false` is final for v1; no promotion run ever.
- The graph closure is not reconsidered by anything in this document.
- B2's wide-inventory degeneracy is a finding about `deem==0.2.0`'s repaired
  adapter, frozen in the v1 manifests; v2 should decide up front whether B2
  is retained, replaced, or dropped rather than re-imposing A1 mid-run.
- Label firewall, score freeze, rebuild verification, and the linear
  `afterok` chain pattern are non-negotiable inheritances.

---

## 5. Exploratory addendum: the graph arms on the real 24 cells (2026-08-24)

At Omri's request, the archived graph arms G2/G3/G4 were run on the real 24
cells (AIRCC job 225501, commit `c84c852`): 3 arms x 5 lambdas x 24 cells x
5 seeds = 1,800 fits, zero failures, reusing the frozen v1 bundles/sidecars
and compared against the frozen B0/B3 reference (recomputed reference matched
v1 exactly: B0 0.7428 / B3 0.7485).  EXPLORATORY: open labels, no registered
lambda, full grid reported; the graph closure and the v1 decision are
unchanged (section header of this document).

Equal-family AUROC by arm and lambda (delta vs B3, wins/ties/losses vs B3 at
the 0.0005 tie threshold):

| arm | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|---|
| G2 (residual uniform, target) | +0.0000 (0/24/0) | +0.0000 (0/24/0) | +0.0000 (0/24/0) | +0.0001 (5/16/3) | +0.0004 (10/10/4) |
| G3 (residual DUFS, target) | +0.0000 (0/24/0) | +0.0000 (0/24/0) | +0.0000 (0/24/0) | +0.0001 (6/15/3) | +0.0005 (10/10/4) |
| G4 (residual DUFS, nuisance) | -0.0000 (0/24/0) | +0.0000 (0/24/0) | +0.0000 (0/24/0) | +0.0000 (0/24/0) | +0.0000 (2/20/2) |

Per-cell, across all 1,800 fits: median |delta vs B3| = 0.00005, maximum
gain anywhere +0.0046 (one cell, lambda=1.0), maximum loss -0.0026.

**Reading.**  The graph term does nothing on real data.  Below lambda=0.3 the
graph arms are B3 to within the tie threshold in every cell; at lambda=1.0 --
the setting with the worst phase0 specificity violations -- the best arm gains
+0.05pp macro, a tenth of B3's own +0.57pp edge over B0 and far inside the
noise floor.  Three consequences:

1. **B3's edge over IU-PCR owes nothing to graphs.**  The continuous additive
   part is the entire story, which sharpens the M-A/M-E ablation questions.
2. **Phase0's stop cost nothing.**  The chain it blocked would have burned the
   full 24-cell budget to measure zeros.  The synthetic gate called it
   correctly: "effect without specificity" on synthetic worlds, no effect at
   all on real cells at any admissible-magnitude lambda.
3. **The graph closure stands on two independent legs now** -- mechanism
   (phase0) and performance (this addendum).

Evidence: `results/graph_arms_exploratory_v1/` (SUMMARY.json, PER_CELL.csv,
FIT_SUMMARY.json); large fit artifacts remain on AIRCC under
`results/graph_arms_exploratory_v1/`.
