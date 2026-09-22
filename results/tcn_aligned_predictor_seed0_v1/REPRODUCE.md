This is a completed archived run. Read REPORT.html and AUDIT.json for the preserved evidence. Do not rerun the training runner in place: it rewrites RUN_STATE.json and REUSE.json, including the initial reuse record. For a fresh replication use an isolated copy, retain the source files matching the manifest, and keep the original result directory separately. From that temporal research worktree with the existing frozen context data:

```powershell
python -B -X utf8 scripts/run_tcn_aligned_study.py
python -B -X utf8 scripts/analyze_tcn_aligned_predictions.py
python -B -X utf8 scripts/evaluate_tcn_aligned_study.py
python -B -X utf8 scripts/report_tcn_aligned_study.py
```

The original90-job flow queue must remain paused. Training and scoring use its shared job directory but a separate TCN-only roster. Do not overwrite old fold0 artifacts. Large checkpoints and SQLite/token arrays stay local with hashes.
