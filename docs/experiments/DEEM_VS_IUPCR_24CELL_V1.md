# DEEM vs IU-PCR v1 — frozen 24-cell benchmark

**Experiment ID:** `deem_vs_iupcr_24cell_v1`

**Status:** frozen before natural-label access
**Scope:** Global completed-trace hallucination detection only

## 1. Why this experiment exists

Residual-Graph DEEM Phase 0 completed 3,050 synthetic fits after its numerical
repair.  The repair succeeded—50/50 checkpoints were healthy, including G4 at
`lambda=1.0`—but no target-graph lambda survived the frozen specificity gate.
At every nonzero lambda, G3 also improved worlds in which a target-graph gain
was forbidden.  The registered outcome is therefore:

`CLOSE_RESIDUAL_GRAPH_EXTENSION_SPECIFICITY_FAILURE`

This closes G0–G5 as an active extension in this research cycle.  It does not
falsify B3: B3 is the graph-free continuous additive DEEM baseline from which
the failed graph arms were derived.  No natural label was opened during the
graph Phase 0, so a separate graph-free benchmark can be frozen without
outcome-dependent revision.

The historical hard-DEEM comparison is not a substitute.  It reported macro
AUROC 0.7543620 versus IU-PCR 0.7541776, a difference of only +0.0001844 with
high seed variability.  It used a different variable-inventory contract and
predated both repaired soft/rank DEEM and continuous additive B3.

## 2. Frozen question and contrasts

The primary question is whether continuous additive DEEM is a better
target-free detector than matched-inventory IU-PCR.

| ID | frozen name | role |
|---|---|---|
| B0 | `iu_pcr_inventory` | matched current IU-PCR comparator |
| B1 | `deem_inventory_hard_adapter020` | binary/original-style packaged-DEEM control |
| B2 | `deem_inventory_soft_rank_adapter020_repaired` | repaired soft/rank packaged-DEEM control |
| B3 | `deem_inventory_continuous_additive` | continuous additive DEEM candidate |

The three preregistered contrasts are `B3-B0` (primary), `B3-B1`, and `B3-B2`
(mechanism controls).  B1/B2 are `deem==0.2.0` adapters, not paper-exact
claims.  B3 is a continuous-visible adaptation and carries no categorical
DEEM theorem claim.

G0–G5, graph construction, DUFS, residualization, graph lambda search,
nuisance encoders, graph controls, Localization, Early Detection, and Phase
2/3 are absent from this experiment.  Their code and the earlier protocol stay
archived for auditability.

## 3. Population and target-free input contract

Reuse, without modification, the registry in
`configs/residual_graph_deem_24cell_v1_registry.json`:

- 24 cells and 48,607 rows;
- the exact historical present inventory in each cell (19–30 features, seven
  schemas);
- the same feature order, fixed confidence signs, row IDs, group IDs, source
  hashes, admission hashes, dataset families, and QA/math task types;
- no imputation, zero filling, population change, fingerprint matching, or
  positional fallback.

Bundles remain physically label-free and load with `allow_pickle=False`.
They contain `X_raw`, feature names/signs, `row_id`, `group_id`, raw length,
family/task identifiers, and provenance hashes.  Natural labels are created in
a separate `row_id -> y_H` sidecar only after a complete immutable score
freeze.  The Stage-A runner cannot import the label module or accept a sidecar
path.

For each cell, fit the frozen transform on the complete label-free cell:

`z=(X_raw-mu)/sigma`, followed exactly once by `x_risk=-sign*z`.

For an exact constant coordinate use `sigma=1` and record the constant mask.
B0–B3 receive the identical transformed matrix and row order.

## 4. Frozen models

### B0 — matched IU-PCR

Rerun IU-PCR on the current risk-oriented inventory.  Do not read historical
`F`, `rho_polarity`, scores, or feature availability.  Orient the output to the
external equal-present-family risk consensus; an ambiguous orientation fails.

### B1/B2 — packaged adapter controls

Run each fit in an isolated process with seeds `0..4`, explicit Torch
determinism, `deem==0.2.0`, one Sparsemax preprocessing layer, identity
initialization, hidden dimension one, five sampler steps, batch 1024, momentum
0.9, weighted MV, and `mv_rand`.

- B1: empirical median split, learning rate `1e-3`, 100 epochs.
- B2: average-rank pseudo-probabilities, clip `1e-3`, learning rate `1e-4`,
  100 epochs.

Final class orientation uses the external equal-family risk consensus.  The
package majority-vote map is diagnostic only.  Package version drift,
orientation ambiguity, non-finite scores, or posterior SD below `1e-3` fails
the fit.  Partial package history is retained on failure.

### B3 — continuous additive DEEM

Reuse `fit_continuous_deem` unchanged and without a graph configuration:

`c_g = w_g*x_g + (2/|g|) tanh(V_g tanh(W_g x_g+d_g)+e_g)`

`ell = b + sum_g 1^T c_g`, `q=sigmoid(ell)`

`E(x,h)=0.5||x-a||^2-h*ell`.

Use family width 8; `a=b=0`; the existing equal-family/equal-feature anchor;
all other frozen initialization, free-energy loss, correctly adjusted
persistent MALA, float64 CPU, full batch, SGD `lr=1e-3`, momentum zero, and
100 epochs.  Atomic contributions must reconstruct the aligned logit within
`1e-8`.  Risk-orientation ambiguity, non-finite state/history, posterior SD
below `1e-3`, or a failed MALA health check fails the fit.

## 5. Label-free preflight

Before Stage A, run all seven registered inventory schemas on deterministic
target-free synthetic fixtures.  Full preflight uses all five seeds and the
frozen 100-epoch B3 configuration; it also executes pinned B1/B2 package fits
on the 19- and 30-feature boundary schemas.  It checks:

- exact schema/order/sign contracts;
- finite healthy B0/B3 output and B3 contribution reconstruction;
- exact deterministic replay for a repeated B3 fit;
- pinned B1/B2 version/config and healthy noncollapsed scores;
- no target-like field or label-module import in Stage A.

`--smoke` is a three-epoch local software test and is permanently barred from
Stage A.  Full preflight chooses no method or hyperparameter and opens no
natural target.

## 6. Stage A, artifacts, and score freeze

Produce exactly 20 fit stems per cell: four arms times five seeds.  B0 is
deterministic but is serialized under every seed so the artifact and ensemble
contracts remain uniform.  Total expected fit artifacts: 480 JSON/NPZ pairs.

Every artifact records score, posterior where available, B3 logit/atomic and
family contributions, objective/package history, health, runtime, source/code/
config/environment hashes, transform, and row/inventory provenance.  Writes
are atomic.  A failure retains partial history and last finite state.

A cell-complete marker requires all 20 stems healthy.  The immutable
`SCORE_FREEZE_MANIFEST.json` requires all 24 cells, all five seeds for every
arm, no debug/incomplete/unhealthy/missing record, and verified hashes for
every JSON/NPZ artifact plus the run definition.  Only then may the separate
sidecar builder open correctness.

## 7. Evaluation and multiplicity

There is no pooled row-level AUROC.  Report AUROC and AUPRC per cell; equal-cell
means within each of the eight dataset families; equal-family macro; QA and
math macros; worst cell/family; and wins/ties/losses (tie `0.0005`).

For all three B3 contrasts, run 10,000 paired family-blocked bootstrap draws
with seed `20260821`, including leave-one-family-out effects.  Superiority
p-values are Holm-corrected across the three preregistered AUROC contrasts.
Run the same frozen-score exact-length, cross-fitted propensity CRT, and
family/group-blocked target null with `B=199`, taking the maximum over the
three contrasts, AUROC/AUPRC, and equal-family/QA/math summaries.  Stage-A
scores are never refit under a label null.  Run `B=999` only if every other
primary superiority gate passes at B=199.

## 8. Frozen gates and decisions

Mechanical validity requires every artifact/health/orientation/firewall/hash
and rebuild gate to pass.  B3 stability additionally requires every cell to
have five healthy seeds and median within-cell absolute seed Spearman at least
0.90.

Primary superiority over B0 requires all of:

- equal-family AUROC delta at least `+0.005`;
- paired-bootstrap lower 95% bound above zero;
- Holm-adjusted p-value at most 0.05;
- QA and math each no worse than B0 by more than 0.005;
- at least 14/24 wins or ties;
- worst cell delta at least `-0.02`;
- all three conditional max-null p-values at most 0.05.

Noninferiority to B0 requires a bootstrap lower bound above `-0.0025` and the
same QA/math degradation limits.  B3 superiority over B1 or B2 uses a minimum
equal-family AUROC delta `+0.0025`, lower bound above zero, and the matching
Holm/max-null gates.

Emit exactly one primary decision:

- `DEEM_BENCHMARK_MECHANICAL_FAILURE`
- `CONTINUOUS_DEEM_UNSTABLE`
- `CONTINUOUS_DEEM_INFERIOR_TO_IUPCR`
- `CONTINUOUS_DEEM_NO_ADVANTAGE`
- `CONTINUOUS_DEEM_NONINFERIOR_TO_IUPCR`
- `CONTINUOUS_DEEM_SUPERIOR_TO_IUPCR`
- `CONTINUOUS_DEEM_SUPERIOR_TO_IUPCR_AND_ADAPTERS`

The graph closure is not reconsidered by any outcome here.

## 9. Rebuild and execution

After B=199 (or promoted B=999), re-evaluate from the original immutable
checkpoints, then recompute Stage A into a fresh directory.  Compact summaries,
decision, and semantic NPZ hashes must match.  Any mismatch changes the final
decision to `REBUILD_VERIFICATION_FAILURE`.

Large bundles/checkpoints/scores move only between AIRCC and
`gdrive:hallucination_detection/cluster_results/deem_vs_iupcr_24cell_v1/`.
Git receives only source, frozen configs/protocol, compact manifests,
evaluation/report, and rebuild evidence.

Claude launch command after clean synchronization:

```bash
bash cluster/submit_deem_vs_iupcr_chain_v1.sh
```

Conditional B=999 command:

```bash
bash cluster/submit_deem_vs_iupcr_promotion_v1.sh
```

---

## Amendment A1 — B2 health is recorded, not blocking (pre-label, 2026-08-23)

**Trigger.** Preflight job 219682 (the first run in which the adapter boundary
executed at all -- job 219646 died earlier on a cuBLAS environment defect)
stopped Stage A on `adapter_boundary_pass`.  All 20 boundary fits completed
under pinned `deem==0.2.0` with exact deterministic replay, and B1 was healthy
on both fixtures, but B2 collapsed on the 30-feature fixture on every seed:
`score_sd` between 1.1e-6 and 1.3e-4 against the 1e-3 health floor, finite
throughout.  `scripts/deem_soft_collapse_probe.py` documents the same
degeneracy for the soft path (27/30 constant-score failures in the registered
sweep).  This is a property of the packaged comparator, not of this benchmark's
infrastructure.

**Change.** B2's health becomes recorded diagnostic data instead of a
mechanical blocker, at every enforcement site:

- the adapter worker exits nonzero only on a failed fit or non-finite scores,
  and additionally records `score_finite`;
- the preflight adapter gate requires all 20 fits complete/finite/pinned,
  full health for every B1 fit, and full health for B2 on the **narrow**
  (19-feature) fixture -- a sanity anchor that the arm still works where it is
  known to work.  B2 health on the wide fixture is recorded in
  `PREFLIGHT_COMPLETE.json` under `adapter_unhealthy_recorded`;
- Stage A cell checkpoints, `FIT_COMPLETE.json`, and the score freeze accept a
  B2 fit that is complete with finite scores; B0/B1/B3 still require full
  health.  Collapsed B2 fits are listed in `FIT_COMPLETE.json` under
  `b2_unhealthy_recorded`.

**Unchanged.** Arms, seeds, inventories, the 1e-3 health definition itself,
every B0/B1/B3 gate, the B3 stability gate, all evaluation statistics,
multiplicity, and the decision list.  The primary contrast `B3-B0` does not
involve B2 at all.

**Interpretation rule.** Any reading of the `B3-B2` contrast MUST be
accompanied by the recorded B2 health tables.  On cells where B2 is collapsed,
`B3-B2` measures "continuous additive DEEM versus a degenerate comparator" and
must be described as such; it carries no mechanism-control force there.  The
narrow-inventory cells where B2 is healthy carry that force.

**Why not the alternatives.**  Lowering the 1e-3 floor would hide collapse for
every arm everywhere.  Re-tuning the B2 configuration on the very fixtures
that exposed the collapse would be outcome-dependent tuning of a control.
Dropping the arm would discard the cells where the soft/rank adapter does
function and change the frozen run shape.  Recording the collapse keeps the
comparison honest and the evidence reviewable.

**Legitimacy.**  No natural label has ever been opened in this experiment
(`natural_targets_opened = False` in both preflight attempts); the only data
observed before this amendment were label-free synthetic fixtures, which is
precisely what a preflight exists to expose.  Decision taken by Omri
(delegated explicitly after review of the job-219682 stop); implementation on
a reviewed commit with a fresh `code_sha256` and run identity.

### A1.1 — deterministic identity fallback on ambiguous risk-consensus (pre-label, 2026-08-23)

Stage A (job 220081) exposed a second manifestation of the same B2 degeneracy:
on four cells (`se_nq_open_llama8b`, `semenergy_triviaqa_qwen3_8b`,
`trace_math500_qwenmath15b_k10`, `truthfulqa_llama8b`) 17 B2 fits crashed with
`risk-consensus alignment is ambiguous` instead of producing a collapsed score.
The mechanism is identical: a degenerate posterior makes the risk-consensus
orientation difference fall inside `anchor_tolerance`, and the aligner raised.
Amendment A1 could not record what the worker never emitted; 20/24 cells
completed (36 collapsed B2 fits correctly recorded) and 4 could not.

Change: `risk_consensus_align` accepts an `ambiguous` policy.  The default
`"raise"` preserves the historical fail-closed behavior for every existing
caller (including `deem_soft_collapse_probe`).  The deem-vs-iupcr adapter
worker alone opts into `"identity"`: the identity class map is adopted
deterministically, the run result records
`alignment = "risk_consensus_identity_fallback"`, and the produced
near-constant score then flows through the ordinary Amendment A1 policy --
recorded for B2, still blocking for B0/B1/B3 (a degenerate B1 remains a
mechanical stop through its health gate, unchanged).

Orientation of a zero-signal posterior carries no information in either
direction, so this changes no measurable quantity; it converts an adapter
crash into the same recorded degeneracy that A1 already governs.  No natural
label had been opened (`stage_a` stopped before any score freeze).
