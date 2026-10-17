# Reproduction and artifact access

Use the hashed temporal_context_data_v1 bundle and the45 original
energy_context_stability_v2 fit directories, including DIAGNOSTICS.npz,
FIT.json, COMPLETE.json and global LANDMARKS.npz/PREPARED.json/PROVENANCE.json.
The scorer verifies old source hashes and excludes held source groups.
These large local/Drive artifacts are required; Git summaries alone are not
sufficient to reproduce token scores. No original outputs are modified.

```powershell
python scripts/run_context_weighted_levels.py
python scripts/evaluate_context_weighted_levels.py
python scripts/diagnose_context_weight_age.py
python scripts/report_context_weighted_levels.py
```

The first command reuses45 outer fits and creates10 nested PRMB pair-exclusion
fits. It computes all19 policies on every token, using16 past-only weight updates
and unchanged original-feature Top10 evidence. The evaluator loads correctness
labels only after all scores are frozen. The age diagnostic is explicitly
post-ranking and does not add or tune a scoring arm.

ARTIFACTS.json records sizes and hashes of local archives; fit JSON records
reference groups and preprocessing. New ANCHORS.npz files preserve weights
but not repeated C matrices; original outer C matrices remain in Step384.
Small summary/provenance/report files are in Git; large NPZ scores remain local.
The paused FM/DiFlo/TCN models and their source artifacts are unchanged.
