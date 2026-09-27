# Codex review and research-plan handoff — 2026-09-23

## Checkpoint update — 2026-09-24

The current decision supersedes the original reading order and CT7 recommendation
below: bank11 STEP-level L-SML is the research starting point, PRMBench is primary,
and CT7/equal fusion are controls. See [month audit](LSML_PRMBENCH_MONTH_REVIEW_20260923_HE.md)
and [runtime protocols](../experiments/PRMBENCH_RUNTIME_FUSION_PLAN_HE.md).
The saved-score PRMScore audit uses retrospective calibration, not nested refitting.
Its donor-fitted model does not demonstrate answer-only learning.

Checkpoint inventory and large-input locations are recorded in
`results/codex_checkpoint_20260924/MANIFEST.json`. Large local-only bank11 arrays
remain dependencies, not files backed up by this commit. The removed
`review-token-axis-20260921` worktree was a disposable clean checkout; reconstruct
its recorded commit in a separate checkout when replaying that historical audit.
Do not recreate it merely to run the new external benchmark pipeline.

Omri authorized implementation on `codex/lsml-external-generalization-v1` after
this checkpoint: Hard2Verify and Socratic-PRMBench, frozen and local L-SML,
matched published comparators, and an AIRCC timing estimate before the full-run
budget decision. MedPRMBench is deferred. No external inference is claimed here.

This package contains Codex's independent reviews, their compact evidence, and
the proposed follow-up protocols. Per Omri's latest instruction, Claude handles
publication of his own experiment branches and data manifests separately.
The publication branch for this package is
`codex/token-local-fusion-optimization-v1`.

## Reading order

1. [SSL, pseudo-label and residual localization plan, v1.1](../experiments/SSL_PSEUDOLABEL_RESIDUAL_LOCALIZATION_PLAN_HE.md):
   implementable protocols, controls, source splits, statistical analysis,
   required artifacts and report questions. It incorporates completed Step432 A1
   results; begin with S0, including calibration on existing nested scores.
2. [Independent readout review](CLAUDE_READOUT_REVIEW_20260922_HE.md):
   plan adherence, metric replay, the causal-readout defect, limits of the
   multiplicity procedure, and which scientific conclusions remain open.
3. [Earlier token-axis cross-branch review](TOKEN_AXIS_FUSION_CROSS_BRANCH_REVIEW_20260921.md):
   binary/soft fusion inconsistencies and historical source-group leakage.
   Its findings describe the inspected historical implementation, not a claim
   that every later branch retains those defects.

The independent audit supports the reported metrics and the decision to retain
CT7. It does not establish an information ceiling or rule out other pseudo-label,
self-supervised, residual or learned-pooling formulations. Step432's position
evidence improves PRMB within-answer ranking while weakening thresholded
PRMScore; those are different objectives. The proposed experiments remain plans.

## Source versions and evidence

| Review | Inspected source | Published evidence |
|---|---|---|
| Token-axis, September 21 | `2b321fa3aaf3da20ee5ba938077631d02d580e83` | [EVIDENCE](../../scratch/token_axis_review_20260921/EVIDENCE.json), [source hashes](../../scratch/token_axis_review_20260921/SOURCE_HASHES.json), [prediction comparison](../../scratch/token_axis_review_20260921/REPLAY_COMPARISON.json), [replay report](../../scratch/token_axis_review_20260921/replay/REPORT_ALL.json) |
| Readout, Steps428–429 | `e572fbdc1fc20523e6537b838e717a1f60ae0afb` | [AUDIT](../../scratch/claude_readout_review_20260922/AUDIT.json) |
| Step432 A1 follow-up | `claude/readout-quickest-detection-v1`, inspected at `cf70933d1f9947577886129cac67a0e90308ede1`; the audit records the actual report and manifest hashes | [AUDIT](../../scratch/step_evidence_plan_review_20260923/AUDIT.json) |

The token-axis replay directory includes both pooled and per-subset prediction
CSVs and reports. `REPLAY_COMPARISON.json` records unequal-cell counts for each
column against the original saved predictions; every count is zero over 3,400
rows per scope. `SOURCE_HASHES.json` records the original source/input hashes.

The later audits replay 157 PB / eight PRMB methods and 57 PB / ten PRMB methods,
respectively. Numerical differences are at floating-point precision. Coverage,
source overlaps, selected normalization checks and manifest exceptions are
recorded explicitly in their audit JSONs and interpreted in the reviews/plan.
The Step432 completion-log append and source-snapshot basename collision remain
documented limitations; this publication does not repair Claude's frozen files.

## Reproduction requirements

Run the scripts from a Python environment with NumPy, pandas and SciPy:

```powershell
python scratch/review_token_axis_20260921.py
python scratch/review_claude_readout_20260922.py
python scratch/review_step_evidence_for_plan_20260923.py
```

These commands overwrite only their corresponding audit JSONs. The token-axis
script also refits the five historical binary models in memory to inspect their
fit/readout algebra; it does not regenerate the archived full replay directory.
That directory came from the inspected `cumulative_vote_fusion_v1.py`, with its
localization lane pointed at the root repository's matching lane and `--out`
pointed at `scratch/token_axis_review_20260921/replay`. The later two scripts
perform saved-score replay and diagnostics without model fitting or inference.

A fresh clone alone is insufficient to rerun these audits. The scripts and some
document links expect these local source checkouts:

- `.worktrees/review-token-axis-20260921`, at the token-axis commit above.
- `.worktrees/readout-quickest-detection-v1`, containing the frozen
  `readout_family_v1`, `quickest_detection_diagnostics_v1` and `step_evidence_v1`
  result directories, job records and source snapshots.

They also require the root localization lane, source-question metadata, joined
labels/offsets, and the frozen score/profile arrays referenced by the scripts.
The readout audit reads the `paths.joined` setting in the historical
`configs/readout_family_v1.json`; that setting is an absolute path on the original
machine and needs a local path mapping in a separate reproduction checkout.
Preserve frozen inputs and verify their recorded hashes before interpreting a
rerun. Worktree links are local navigation links, not files included by this
publication commit.

Large caches and original experiment arrays are intentionally not duplicated in
this review package. Consult the experiment's manifests and the configured
`gdrive:hallucination_detection/` archive; inspect exact paths, sizes and manifests
before restoring data. Claude's branch publication is a separate dependency.

## Repository record

Publication checks passed: all three audit scripts parse and compile, all eight
evidence/report JSONs parse, all 30 local Markdown links resolve in this
workspace, and both archived replay CSVs match their original predictions in
all 3,400 rows per scope. These checks validate the publication package; the
source-score audits and their scientific limitations are documented above.

`HISTORY.md` records the earlier cross-branch audit and the tagged Codex
Steps430–431; `PROGRESS.md` points to the current plan and reviews. Step numbers
from parallel branches may overlap: use the author tag, date and document path
to distinguish them. Compact replay outputs are included with this package;
temporary publication inventories, console logs and unrelated workspace files
are outside its scope.
