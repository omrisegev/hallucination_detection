# Run the reviewed conditional-IU experiments on another machine

Branch: `codex/conditional-iu-followups-v1`. Use the existing project Python environment
(NumPy, SciPy, scikit-learn, PyTorch and the existing project dependencies).
No container or GPU is needed locally. The AIRCC container-build memory requirement
is separate from the algorithm's local memory requirement.

Run these PowerShell commands from the **main repository containing the frozen data**:

```powershell
git fetch origin
git worktree add ..\conditional-iu-followups-v1 origin/codex/conditional-iu-followups-v1
$experimentSource = (Get-Location).Path
python ..\conditional-iu-followups-v1\scripts\check_conditional_iu_inputs.py --source-root $experimentSource
```

Proceed when the check reports PASS. Git transfers code and small review records, not
the nine large feature caches and other untracked results. The check names every missing
or different input. All 22 data/provenance files must match the registered hashes.
Do not regenerate a cache with different normalization or annotations under these names.

Run one family at a time to keep partial results available without memory competition:

```powershell
python ..\conditional-iu-followups-v1\scripts\complete_conditional_iu_fusion.py --source-root $experimentSource --family position
python ..\conditional-iu-followups-v1\scripts\complete_conditional_iu_fusion.py --source-root $experimentSource --family graph_local
python ..\conditional-iu-followups-v1\scripts\complete_conditional_iu_fusion.py --source-root $experimentSource --family graph_tv
```

Inspect each command's result before starting the next. Each supervisor first checks
the actual-data smoke on 27 answers, then runs the full 13,769-answer benchmark if its
review passes. Normal eight-hour invocation caps resume automatically. Numerical or
integrity errors stop the supervisor; do not remove its locks while it is running.
You may run different families in separate terminals when the machine has enough RAM.
Never start two processes for the same output directory.

Results are written under the **new code worktree**:
`results/conditional_iu_fusion_v1/{position,graph_local,graph_tv}/`.
Read `PROGRAM_STATE.json`, `smoke/RUN_STATE.json` or the full `RUN_STATE.json` for
progress. Full outcomes are in `COMPARISON.csv`, `PER_CELL.csv`, `METRICS.json`,
`CONTRASTS.json` and `RESULT_REVIEW.json`. Checkpoints preserve completed answers.

Use a fresh output directory on the second machine. Do not mix Linux/Windows partial
fits: the manifest records runtime versions and exact source/code hashes. Shared science
is unchanged: frozen RBM12 features, answer-local native IU baseline, Top10, original
entropy gate, source-group folds, and nested PRMScore calibration. Full protocol:
`docs/experiments/CONDITIONAL_IU_FUSION_V1.md`.


## Frozen gate and folds now included in Git (2026-09-14)

The branch now carries these exact input files, matching the registered hashes:
- results/localization_source_group_audit_v1/FOLDS_V2.json
- results/fusion_fixed_gate_v1/DETECTORS.npz
- results/fusion_fixed_gate_v1/METRICS.json

Verification record: results/frozen_gate_folds_bundle_v1/REVIEW.json.
These supply the fixed source-group split and saved gate data/thresholds.
The remaining cache/provenance requirements still apply: run the full input checker.
If --source-root points to a separate data checkout, these three files must also
be present under the same relative paths there. Copy the committed versions only
when missing; investigate any hash mismatch instead of overwriting it.


## Frozen v3 annotation bundle also included (2026-09-14)

The branch also contains results/localization_full_benchmark_v3/evaluation/JOINED.json,
JOINED.npz in the same directory, and results/localization_prm_label_audit_v1/RELEASE_V3.json.
JOINED.npz contains labels and concatenated-step offsets plus historical scores.
Original label sentinel values are preserved. Token span boundaries still come
from the feature caches; those cache inputs remain required.
RELEASE_V3.json is byte-exact provenance (including original-machine paths); it matches
the joined answer roster and tracked fold hash, but does not bundle every file it references.
Audit: results/frozen_gate_folds_bundle_v1/LABELS_REVIEW.json.
If using a separate --source-root, these runtime inputs must exist at their relative paths there.
