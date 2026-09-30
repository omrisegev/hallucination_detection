# Short cycle 1: can Joint fit a useful localizer inside one long answer?

September 7, 2026. Status: completed and reviewed. Results are in
`results/localization_short_cycle01/REPORT.md`.
The broader continuation document is a backlog. Stop after this pilot's report
and use its findings to choose the next stage with Omri.

## Why this first

The newly available v2 report gives PRMB step AUROC 0.673414 for
`internal_joint_modelinv_lam0`, 0.672394 for `internal_joint_liu010`, and
0.673177 for the shuffled-graph control. These descriptive values motivate
trying the Joint model-inverse mechanism without an added graph penalty.
They do not establish significant superiority or validate answer-only fitting.
The current report still has missing contrast fields; do not duplicate
Claude's continuing report computation.

The main unresolved premise is whether the current answer supplies enough
useful information to learn its localizer. A new lambda sweep or the 24-cell
transfer would not answer that question.

### Completed decision

The pilot found IU better than Joint on the common strict subset (0.72072 vs
0.69148; grouped bootstrap Joint−IU −0.02875, 95% CI [−0.04837, −0.00768]).
Joint had strict coverage 24/30: four partitions were blocked and two fits
were finite but unconverged. Equal averaging was 0.70134 on the same subset,
with an inconclusive Joint−equal interval. Do not start lambda or transfer
experiments yet. The next short question is a non-clustering Joint reference
under the same 30-answer protocol, or a decision to prioritize IU/equal.

## Fixed small scope

| Item | Pilot choice |
|---|---|
| Population | One existing cell: PRMBench / Qwen3-8B |
| Sample | At most 30 answers from distinct source questions, length 1,024–2,048 tokens; choose by a fixed hash of IDs, without reading labels |
| Availability | Metadata-only check found 290 eligible answers from 253 source groups among 6,969 answers; no labels or raw feature array were read |
| Representation | Width 32, one fixed fitting/scoring-grid policy, original 30 window-feature definitions with explicit constant/unavailable columns; 32–64 full fitting windows per selected answer |
| Main candidate | Joint model-inverse with lambda zero; grouping and fitting use this answer only |
| References | Canonical IU and equal averaging on the identical windows/features, also fitted independently per answer |
| Trajectory readout | Fixed mapping back to token risks and the existing PRMB span-max step readout; no learned trajectory-fuser search in this cycle |
| Outputs | Fit coverage, degeneracy/conditioning, within-answer perturbation stability, official-step ranking results on the exact shared pilot IDs, and runtime |

**Implementation clarification:** the executed capsule derives feature
orientation from each answer's own window covariance, anchored to its entropy
feature. The feature definitions and entropy anchor semantics are shared
engineering conventions; no cross-answer fitted sign vector, group, scale,
fusion weight or correctness threshold enters the score phase. This is the
strict answer-local implementation that was actually run.

The current LOAO
grouping API requires several answer owners: adapt and identify a within-answer
blocked stability procedure explicitly. Blocks are parts of one answer, not
new independent answers. Do not borrow provenance groups or silently replace
Joint by another estimator to make the pilot run.

## Execution and stopping boundary

1. Implement the minimal window adapter and verify answer isolation and the
   canonical lambda-zero weight rule. Reuse the existing extractor and mapping
   utilities. This needs code adaptation; it is not just rerunning a command.
2. Freeze IDs, source/config identity, width, fit/stability settings and the
   readout before scoring. Check a single answer first for runtime and failures.
3. Cap the numerical pilot at 30 answers and 20 minutes of wall time after
   setup. Checkpoint per answer and record any timeout; do not launch a queue
   of continuation jobs. This is an execution cap, not a measured runtime
   estimate. Avoid competing with Claude's local report job; use an isolated
   output/code namespace and a small CPU allocation if local contention matters.
4. Freeze scores before joining official step labels. Reuse the PRMB evaluator
   definitions, source groups and relevant existing fold assignments on the
   exact pilot IDs. Mark single-class or unavailable metrics explicitly.
   Compare all three recipes on these same IDs; do not compare a 30-answer
   pilot number with the published full-cell AUROC.
5. Return a small table and a few trace plots, then stop. There is no automatic
   expansion to more widths, methods, cells or the separate 24-cell experiment.

Answer isolation and stability are required mechanical questions, while
accuracy differences on 30 long answers are only directional evidence.
If computation or stability fails, the next discussion concerns grouping,
regularization or the declared pooled fallback. If fits are stable and local
ranking is promising, the next discussion can extend coverage or study the
trajectory axis. If stable scores lack useful ranking, inspect the window
representation and readout before adding graph variants.

This pilot does not settle short-answer coverage, a strict answer-only
all-correct decision, ProcessBench first-error F1, superiority over the full
comparator registry, or final-answer transfer. Those remain separate later
questions. CPU computation can use existing traces; no new inference/GPU run
is part of this stage.

Evidence: Claude worktree `results/joint_lsml_optimization_v2/REPORT.md` and
`evaluation/report_tables.json`; label-free pilot availability from members
`token_offsets`, `row_ids`, `group_ids` of
`cells/prmbench_qwen3_8b.npz` (344,387,719 bytes). The availability check did
not select a final cohort or fit a model.

## Concrete pre-run implementation choices

- Four chronological blocks of nonoverlapping windows replace answer owners
  in the existing consensus engine. Keep K in {3,4,6,8}, minimum group size 3,
  and held-admissibility fraction 0.95 (all four block omissions must qualify).
  No provenance fallback; failure remains visible.
- Canonical Joint uses five starts and 5,000 sweeps, with its existing
  convergence diagnostics; lambda zero retains analytic ridge conditioning
  at 1,000. Finite nonconverged outputs are marked descriptive separately.
- Width/stride 32; the final full window may overlap for scoring but is not
  an extra fitting observation. Stability refits omit the first or last
  quarter of fit windows, including refitting scaling and groups.
- Shared v2 output convention: orient to the signed-feature mean (entropy
  fallback) and set within-fit score SD to one; negate confidence to risk.
  Average overlapping window risks at tokens, then use official-span maximum.
- CPU-only source snapshot uses exact canonical functions with minimal package
  initialization. SHA-256 binds its files, the adapter, runner, inputs and IDs
  before scoring. The run has a 1,200-second total cap and a 90-second cap per
  answer including its stability refits; timeouts are reported, not retried.
- Preserve v2's actual pooled PRMB step AUROC as an explicitly named
  compatibility statistic on shared pilot IDs. Also report fold-wise AUROC
  and within-answer diagnostics; do not interpret uncalibrated across-answer
  ranking as proof of local error identification. Use 2,000 paired source-group
  resamples, with unavailable draws/metrics disclosed. Official labels are
  joined only after the score freeze. No full-cell number is a pilot baseline.
