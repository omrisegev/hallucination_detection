From the temporal research worktree, with the existing hash-verified data and frozen Ridge artifacts:

```powershell
python -B -X utf8 scripts/run_aligned_context_predictors.py
python -B -X utf8 scripts/evaluate_aligned_context_predictors.py
python -B -X utf8 scripts/report_aligned_context_predictors.py
```

The scoring manifest rejects changed source code or inputs. Each answer is checkpointed in ANSWERS.sqlite. The runner never loads correctness labels; evaluation is separate. Large arrays/SQLite/diagnostics remain local; their hashes are published in ARTIFACTS.json. The same source/data bundle is required for replay.
