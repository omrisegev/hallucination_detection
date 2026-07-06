---
description: Fetch finished AIRCC results to the local machine and validate them — scp the raw pkls, check the rich-save schema (7 keys per candidate, K candidates, labels), and run extract_all_features as an offline sanity check. Accepts the remote results dir name (e.g. edis_aime24).
---

Fetch and validate cluster results. `$SHARED` = `/shared/cycle2_tau_averbuch_prj/omrisegev1`.

## Step 1 — Connectivity pre-check

`ssh -o ConnectTimeout=5 aircc 'echo ok'`. On failure: **check TAU VPN**, stop.

## Step 2 — List and fetch

```bash
ssh aircc 'ls -la $SHARED/results/<RUN_DIR>/'
mkdir -p cache/<RUN_DIR>
scp "aircc:$SHARED/results/<RUN_DIR>/raw_*.pkl" cache/<RUN_DIR>/
```

Ignore any `.pkl.tmp` files (in-flight checkpoint scratch — never fetch them).

## Step 3 — Validate each pkl

Run a Python check (Bash tool, from repo root) over every fetched file:

1. Loads without error; top-level is `{idx: entry}`.
2. Each entry has `question`, `gold_row`, `candidates`.
3. Each candidate has all 7 rich keys: `full_text`, `token_entropies`,
   `token_spilled_energies`, `token_offsets`, `top_k_logprobs`, `gen_token_ids`, `label`.
4. `len(candidates) == K` for every problem (else PARTIAL — job unfinished or preempted
   pre-completion; report which idx are short).
5. `top_k_logprobs` is `{'ids': int32 [T,50], 'logprobs': float32 [T,50]}` with T ==
   `len(gen_token_ids)`.
6. Label distribution: print accuracy per temp. All-zero or all-one labels on a math
   dataset is a red flag (grading bug regression) — flag it, don't silently accept.

## Step 4 — Offline feature sanity

For ~5 random candidates: `extract_all_features(token_entropies)` from
`spectral_utils.feature_utils` → assert 16 finite features (`FEAT_NAMES`). This proves the
traces are analysis-ready without any notebook.

## Step 5 — Report

Print a table: file | problems | candidates/K | accuracy | mean trace len | verdict
(VALID / PARTIAL / STALE). Remind: raw pkls are gitignored-scale artifacts — keep them in
`cache/`, do not commit; back up to Drive if the run is expensive to redo.
