---
description: Audit a Colab notebook for the 8 most common bugs in this project. Spawns an Explore sub-agent to read the notebook, then reports findings with cell index and fix. Use before committing any notebook with new or modified cells.
---

Spawn an Explore sub-agent to audit the specified notebook. If no notebook path is given, ask which notebook to audit.

## Sub-agent brief

Spawn `subagent_type=Explore` with this exact prompt (filling in `<NOTEBOOK_PATH>`):

---

Read the Jupyter notebook at `<NOTEBOOK_PATH>`. For each cell (0-indexed), check for the following 8 bugs. Report all findings as a numbered list with: cell index, the problematic code snippet (≤ 2 lines), and the fix.

**Bug checklist:**

1. **Wrong git branch** — Look for `git clone` commands. Flag any that use `-b master` instead of `-b feature/meta-agentic-integration`. The `baselines.py` module only exists on `feature/meta-agentic-integration`.

2. **Missing None guard after `extract_all_features()`** — Look for any line matching `f = extract_all_features(...)` or similar. Check if the NEXT non-empty line is `if f is None: continue`. If not, flag it. Missing this guard causes `TypeError` on short traces.

3. **Stale pkl not detected** — Look for three-branch pkl reload patterns. Flag any that do NOT call `_valid_res()` or equivalent check (e.g., `any(v for v in res.values() if v)`). Without this, all-None results from previous broken runs are loaded silently.

4. **Missing incremental Drive save** — Look for loops that iterate over 5+ items and compute results. If the `pickle.dump` save only appears AFTER the loop (not inside it), flag it. Long loops must save after each iteration to survive Colab disconnects.

5. **Invalid `load_lciteeval` kwargs** — Look for calls to `load_lciteeval(...)`. Flag any that pass `split=`, `n=`, or any kwarg other than `task=` and `n_samples=`. Correct signature: `load_lciteeval(task: str, n_samples: int)`.

6. **Wrong `lciteeval_grounding_label` signature** — Look for calls to `lciteeval_grounding_label(...)`. Flag any that pass only one argument (e.g., `lciteeval_grounding_label(row)`). Correct signature: `lciteeval_grounding_label(citation_ids: list, row: dict)`.

7. **Invalid `best_nadler_on` kwarg** — Look for calls to `best_nadler_on(...)`. Flag any that pass `normalize=True` or `normalize=False`. This kwarg does not exist — `best_nadler_on` applies z-score internally. Passing it causes `TypeError`.

8. **`pip install git+https://`** — Look for any pip install commands using `git+https://`. This ignores the branch, conflicts with Colab's module cache, and has failed repeatedly. Use `git clone -b {BRANCH} https://...` instead.

For each finding, output:
```
[CELL <N>] <Bug name>
  Code: <problematic snippet>
  Fix:  <one-sentence fix>
```

If no bugs are found, output: `AUDIT CLEAN — no issues found in <N> cells`

---

## After the sub-agent returns

Synthesize the findings:
- If AUDIT CLEAN: confirm the notebook is safe to commit
- If findings exist: group them by severity (HARD FAIL vs WARNING) and offer to fix each one using `NotebookEdit`

**Severity:**
- HARD FAIL (will cause runtime errors): bugs #1, #2, #5, #6, #7, #8
- WARNING (will lose progress on disconnect but won't crash): bugs #3, #4
