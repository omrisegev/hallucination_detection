# Joint L-SML v2: late integrity repair and diagnostic correction

Date: September 6, 2026. Status: PREPARED IN A SEPARATE WORKTREE; NOT APPLIED.

## Why this amendment exists

The original recursive manifest used basenames. All inner folds reuse
`scores_inner.npz` and `meta_inner.json`, so its dictionary kept only one
hash per name. The launch record/config requested by the protocol was also
absent. The evaluator had no mandatory integrity check before reading labels.
These are reproducibility defects; they do not establish that scores changed.

The correction binds every exact relative path, including all five inner
folds, R1 artifacts, label-sidecar bytes, and the automated audit receipt.
Hashing a label file does not decode its contents. The receipt is required
before a late record can be created; any drift or missing artifact blocks
the patched evaluator before it creates its evaluation directory.

This is an honestly dated **late snapshot**, not proof of a launch-time
freeze. It records the currently observed producer source separately from
the new audit source. It cannot recover in-memory launch state or retroactively
prove that a named launch configuration existed. Retain this limitation in
the results report and retain the original protocol and R1 amendment.

## The centering correction

The promised within-answer diagnostic subtracted one column mean over the
training step matrix. The corrected calculation subtracts each answer's own
column means, using its full-length outer-training steps. This changes an
unscored diagnostic, not the registered B1 scores, detector, locator, or R1 grid.

For a completed existing run, `repair_centered_diagnostic.py` writes a new
`CENTERED_DIAGNOSTIC_REPAIR.json` beside each fold. It preserves `moduleb.npz`,
`moduleb_meta.json`, old manifests and old profiles. It uses the saved float32
order statistics, promoted to float64; it does not claim bit-exact recovery
of the producer's unsaved float64 intermediate. Degenerate fits are recorded
as unavailable. The repaired future producer uses its original float64 values.

## Handoff after Claude finishes

Keep Claude's current structure/R1 work running. Do not replace its files or
reload modules mid-run. The patched programs live in
`C:/Users/omris/TAU/hd_per_answer_wt`. Before applying them, verify that all
structure and R1 processes have ended, all 45 folds are complete, and no
evaluation directory exists. Do not run `structure` again from the repair branch.

The following commands are the prepared sequence, not a record of execution:

```powershell
$v2Root = 'C:/Users/omris/TAU/hd_jlsml_v2_wt/results/joint_lsml_optimization_v2'
$producerRoot = 'C:/Users/omris/TAU/hd_jlsml_v2_wt'
./.venv/Scripts/python.exe scripts/joint_lsml_optimization_v2/repair_centered_diagnostic.py --results-root $v2Root
./.venv/Scripts/python.exe scripts/joint_lsml_optimization_v2/audit_prelabel.py --results-root $v2Root
./.venv/Scripts/python.exe scripts/joint_lsml_optimization_v2/freeze_integrity.py --results-root $v2Root --producer-root $producerRoot --amendment docs/experiments/JOINT_LSML_V2_LATE_INTEGRITY_AMENDMENT_20260906.md
```

The protocol's independent scientific review still needs to assess admission,
fallback rates, stability, source/config fidelity and the late-freeze limitation.
A successful file hash check does not establish those scientific conditions.
Use the patched evaluator's `--results-root` only after those conditions pass.
Do not invoke the old evaluator in the producer checkout, which lacks the gate.

This repair is for the already authorized v2 study. It does not choose or
evaluate the new per-answer window study's fusion methods.
