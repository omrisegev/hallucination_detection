# AIRCC handoff — Token-local temporal innovation B3 v1

This is the execution task for branch
`codex/token-local-fusion-optimization-v1`.  The objective is to produce two
independent, byte-identical target-free Phase-2 score freezes on AIRCC, audit
them before any target import, then evaluate and plot the five registered
variants.  Do not tune, remove arms, lower epochs, lower MALA steps, lower the
60,000-token donor cap, or open labels to shorten the run.

## Hard boundaries

- Work from the exact pushed commit in a new clean, detached local worktree.
- AIRCC code must be synced with `cluster/sync_code.sh`; do not use GitHub on
  compute nodes and do not sync code while any of these jobs is active.
- The prepared input is the signed release whose build-A fit manifest SHA-256
  is `a0785ca055346c1148f9b5ba12a32803a86b761cfd0d8381893a4b71ddac0eac`.
- First inspect whether that release already exists under `$SHARED`.  If it is
  absent, stop and report the exact missing path.  Do not upload to Google
  Drive, copy 2.9 GB through the local machine, or substitute another release
  without explicit user authorization.
- No ProcessBench/PRMBench target file, `response_scores`, evaluator, or old
  evaluation table may be opened by the Phase-2 fit/audit processes.
- Do not evaluate until both Phase-1 and Phase-2 A/B freezes pass independent
  pre-label audits.  An A/B `diff` is necessary but not sufficient.
- Leave final artifacts under `$SHARED`; do not mutate Drive or push result
  files unless separately authorized.

## 1. Clean source sync

On the local machine, fetch the pushed branch, resolve its exact commit, and
create a clean detached worktree.  Run `git status --short` there and require
no tracked or untracked files before sync.  Then:

```bash
bash cluster/sync_code.sh aircc
```

The sync must produce `$SHARED/code/SYNC_COMMIT.json` with the exact commit and
`"dirty": false`.  If `sync_code.sh` refuses because another AIRCC job is
running or pending, do not override it; wait for a clean boundary or coordinate
with the owner.

## 2. Resolve the prepared release, fail closed if absent

Use read-only SSH discovery.  The intended shared layout is:

```text
/shared/cycle2_tau_averbuch_prj/omrisegev1/inputs/reconstruction_benchmark_v1/releases/2026-08-24_localization_v1
```

Set:

```text
SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
RELEASE=<verified shared release root>
OUT=$SHARED/results/token_local_temporal_innovation_b3_v1/<exact-commit>
COMMIT=<exact pushed commit>
```

Require this command to match before submission:

```bash
sha256sum "$RELEASE/build_A/localization/inputs/MANIFEST.json"
```

Expected digest: `a0785ca055346c1148f9b5ba12a32803a86b761cfd0d8381893a4b71ddac0eac`.
The output root must not already exist.

All `sbatch` commands below run on the AIRCC login node from `$SHARED/code`
(either inside one SSH shell or via an equivalent quoted `ssh aircc` command).

## 3. AIRCC smoke gate

Submit on the currently valid partition/QoS reported by `sdata` (historically
`power-gpu` / `owner_880`):

```bash
sbatch --parsable -p power-gpu --qos=owner_880 \
  --export=ALL,PHASE2_EXPECTED_COMMIT="$COMMIT" \
  cluster/smoke_token_temporal_innovation_b3_v1.sbatch
```

Wait for `COMPLETED` and inspect the log.  Both focused suites must pass:
11 Phase-2 tests and 6 Phase-1 regression tests.  Do not submit real fits after
a smoke failure.

## 4. Submit the immutable fits

Run the compatible Phase-1 baseline refreeze and the 18-task Phase-2 array.
The Phase-1 refreeze is required because the Phase-2 evaluator verifies the
current source snapshot and cannot consume the older local certificate after
the shared I/O source changed.

```bash
P1_JOB=$(sbatch --parsable -p power-gpu --qos=owner_880 \
  --export=ALL,PHASE2_EXPECTED_COMMIT="$COMMIT",PHASE2_LOCALIZATION_RELEASE="$RELEASE",PHASE2_OUTPUT_ROOT="$OUT" \
  cluster/run_token_local_fusion_phase1_v1.sbatch)

P2_JOB=$(sbatch --parsable -p power-gpu --qos=owner_880 --array=0-17%8 \
  --export=ALL,PHASE2_EXPECTED_COMMIT="$COMMIT",PHASE2_LOCALIZATION_RELEASE="$RELEASE",PHASE2_OUTPUT_ROOT="$OUT",PHASE2_FIT_WORKERS=24 \
  cluster/submit_token_temporal_innovation_b3_v1.sbatch)

ASSEMBLE_JOB=$(sbatch --parsable -p power-gpu --qos=owner_880 \
  --dependency=afterok:"$P2_JOB" \
  --export=ALL,PHASE2_EXPECTED_COMMIT="$COMMIT",PHASE2_LOCALIZATION_RELEASE="$RELEASE",PHASE2_OUTPUT_ROOT="$OUT" \
  cluster/assemble_token_temporal_innovation_b3_v1.sbatch)
```

Record all three job IDs.  The Phase-2 array mapping is fixed: indices 0–8 are
fit A and 9–17 are the independent fit B, in the runner's registered cell
order.  Each task fits all 5 folds × 5 arms × 5 seeds for one cell.  The array
must be exactly `0-17`; do not submit only a favorable subset.

Monitor with `squeue`, `sacct`, and bounded log tails.  A preempted or failed
cell must be rerun with the same commit/environment.  Never overwrite an
existing cell directory: first verify whether it is complete; move an invalid
partial directory aside under a clearly superseded name before resubmission.

## 5. Freeze checks before labels

Require all of the following:

1. Phase-1 A/B job succeeds and `phase1_fit_A` equals `phase1_fit_B` byte for
   byte.
2. All 18 Phase-2 array tasks succeed.
3. The assembly job succeeds, writes both manifests, and reports a clean
   recursive A/B diff.
4. Both Phase-2 manifest hashes are identical; all nine cell records and score
   artifacts are present.
5. No source sync occurred between smoke, fit, and assembly.

Report job IDs, elapsed time, max RSS, exact output paths, manifest hashes,
environment hash, and any retries.  Do not report a model winner yet.

## 6. Independent pre-label audits

Assign an independent agent that did not implement or run the fits.  Its task
must be explicitly target-free and fail closed on ambiguity.  It must inspect
both A and B and, at minimum:

- verify the full manifest/record/score/source/environment/input hash chain;
- verify A/B byte identity independently;
- prove the token-only fit path never indexes/materializes `response_scores`;
- reconstruct every frozen token and step score from the serialized model
  state, innovation map, prepared token members, fold roster, and `max`
  reducer, within the registered exact tolerances;
- verify all five folds, five seeds, five arms, donor-only preprocessing,
  60k cap, first-token mask, future/row boundaries, B3 health/Spearman, PSTG
  bootstrap/exact-subset gates, and nonempty/non-dense support counts;
- run the bound evaluator preflight with targets unavailable and prove that a
  missing/mismatched certificate fails closed.

Only a genuine PASS may create:

```text
$OUT/PRELABEL_AUDIT_PHASE1.json
$OUT/PRELABEL_AUDIT_PHASE2.json
```

The certificate schemas and fields must satisfy the evaluator's exact
`_phase1_preflight` and `_phase2_preflight` checks.  In particular they bind
the appropriate score manifest, protocol, evaluator source, independent agent
ID, `labels_opened_during_audit=false`, and their own canonical payload hash.
Do not hand-author a PASS certificate merely to unblock evaluation.

## 7. Post-audit evaluation and plots

Only after both audits PASS, run in a separate process:

```bash
python scripts/reconstruction_benchmark/evaluate_token_local_temporal_innovation_b3.py \
  --score-freeze "$OUT/fit_A" \
  --audit-certificate "$OUT/PRELABEL_AUDIT_PHASE2.json" \
  --phase1-score-freeze "$OUT/phase1_fit_A" \
  --phase1-audit-certificate "$OUT/PRELABEL_AUDIT_PHASE1.json" \
  --localization-release "$RELEASE" \
  --out-dir "$OUT/evaluation"

python scripts/reconstruction_benchmark/plot_token_local_temporal_innovation_b3.py \
  --freeze-a "$OUT/fit_A" \
  --freeze-b "$OUT/fit_B" \
  --evaluation-dir "$OUT/evaluation" \
  --out-dir "$OUT/plots"
```

Return the macro/cell metrics, paired CIs, promotion decisions, PRMBench guard,
PSTG support plots, variant correlations/deltas, and A/B-difference plot.  The
scientific conclusion must follow the preregistered gates:

- if only `LOCAL_TOKEN_B3` passes, advance B3 and report temporal innovation
  negative;
- if a rook innovation arm passes B3, self-only, non-rook, IU29, and all
  consistency/CI gates, advance it; prefer PSTG over all-rook only for at least
  `+0.001` F1, otherwise prefer the sparser stable arm;
- if no arm passes, retain `LOCAL_IU29`.

Do not begin the fresh Qwen3-14B/Gemma confirmation in this task.  Stop after
the retrospective 8-cell development result and plots, and request review.
