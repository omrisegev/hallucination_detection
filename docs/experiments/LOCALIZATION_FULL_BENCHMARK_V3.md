# Full localization benchmark on the corrected cached release

Frozen scope, 2026-09-07. This answers Omri's request to stop relying on small
development cohorts and restore comparison with historical leaders.

## Population and meaning

Use every row in `localization_prm_label_audit_v1/RELEASE_V3.json`:
6,969 PRMBench answers, 3,400 ProcessBench answers scored with each of
Qwen3-4B and Qwen3-8B: 13,769 model-answer rows in nine cells.
PRMBench's three existing alignment exclusions stay declared. Do not create
a new length-selected cohort. All 21 rows shorter than 64 tokens remain in
the denominator; the existing recipe can report insufficient fitting support.
The 11 PRMBench rows longer than 2,048 tokens are included.

This is full **cached development evaluation**, not an untouched test and
not the historical 24-cell final-answer task. The latter remains separate.

## Execution order

1. Run the unchanged 19 `fusion_replication.ARMS` over the whole release.
   These are inexpensive additional outputs of the same two bank fits, not
   19 separate model fits. Retain moment/context IU, equal, native Joint,
   graph and permuted-graph outputs and the declared single/dual routing.
   Reuse the 110 frozen original outputs after exact input/identity checks.
   This first pass establishes full-population anchors; it does NOT complete
   the historical-leader comparison or include every newer contender.
2. Evaluate the fixed condition100 Joint+graph and its lambda-zero/permuted
   controls, equal+graph/permuted controls, entropy-risk fitting-row selection
   (IU, Joint and equal controls), and static IU/Joint trajectory mean/GLS.
   Preserve the original 110 predictions as a numerical replay requirement.
   This is a fixed shortlist, not a fresh parameter sweep. IMM is a negative
   temporal control, not a promoted candidate.
3. Refit historical multi-answer methods on corrected source-group folds:
   canonical IU/U-PCR, LIU/DUFS-LIU, CONT/L-SML, Joint model-inverse lambda0
   and LIU graph, with corresponding controls. Restore dedicated localization
   incumbents (family6/top5, GL-LIU, token-IU29, Unified28, entropy/top5).
   Every exact implementation/access/selection contract must be registered.
4. Add CIW/DEEM adapters and separately identified PRM/critic comparators.
   Reuse immutable predictions only when that preserves their stated scope.
   Old leaked-fold fits need refitting, not merely corrected-label rescoring.

The machine-readable method registry records these completion states. An
unimplemented or unfitted comparator is pending, never silently omitted or
represented by an unmatched historical headline.

## Fixed score and metric contracts

Pass 1 uses the exact existing answer-only recipe: width8 moments/context,
minimum8 fitting windows, original grouping/optimizer/seed, entropy sign
anchor, local normalization, explicit dual-bank routing and IU fallback.
All fitting is within the current answer. No error label enters scoring.
Step score is maximum token risk inside the official step. The answer-only
GMM chooses whether to open the gate; an open gate selects the highest-risk
step. Retain gate/peak distinctions and native-fit versus fallback coverage.

PRMBench: corrected v3 step labels, pooled step AUROC for continuity,
within-answer mean AUROC over mixed-label answers, both denominators, and
common-valid comparisons. Pooled AUROC alone does not establish improved
within-answer localization. Multi-answer OOF methods additionally require
fold-averaged AUROC; pooled old-fold scores are a historical bridge only.
No post-label sign reversal or rescaling to maximize a metric.

ProcessBench: exact first-error/no-error harmonic score per cell; an invalid
decision is a failure, never a silently dropped row. Report all eight cells,
Qwen3-4B and Qwen3-8B four-cell macros separately, and the eight-cell macro.
The Qwen3-8B macro is the model-matched extension of the 110 pilot. Report
clean-answer accuracy and exact-error accuracy separately.

Intervals must resample canonical source groups, keeping duplicate answers
and repeated scoring models together. Do not apply the pilot's per-cell
independent group bootstrap blindly to the repeated-model full population.
No winner claim from unreviewed point estimates. The first scoring pass can
run while the comparison/interval adapter is completed, with status visible.

## Operational safeguards and review

Read Claude's worktree only. Reuse local telemetry; no new model inference,
remote transfer, cloud mutation or deletion. Expand only the seven required
NPY members of each frozen telemetry archive into this run's own input cache.
Workers memory-map these arrays. Verify source and extracted-input hashes.
At most three CPU workers, one BLAS thread each. Save per-answer atomically,
resume by manifest and output hashes, and keep a current run-state file.
An eight-hour submission cap checkpoints rather than discards work.

Before the full run: verify all source joins/spans/counts, replay representative
existing scores and decisions, exercise the short-answer failure path, and
check source hashes. At completion freeze all score files before evaluation;
review target joins, coverage, metric arithmetic and matched pilot replay.
No claim that running a driver alone constitutes the completed benchmark.
