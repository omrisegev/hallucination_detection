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
