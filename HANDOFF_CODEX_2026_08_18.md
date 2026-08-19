# Handoff to Codex — 2026-08-18

Two acquisitions finished clean on AIRCC. One reproducibility gap needs something
only the machine that ran it can supply. Everything below is what changed since the
last handoff, then the questions.

---

## 1. What finished

| Run | Traces | Failed | Shards complete | Job states |
|---|---:|---:|---|---|
| DeepConf `m2_deepconf_k512` | **15,360 / 15,360** | **0** | 24 / 24 | all 24 `COMPLETED 0:0` |
| Refrain `s1_refrain_full` | **1,000 / 1,000** | **0** | complete | `COMPLETED 0:0` + `_c1` link |

Per-shard finished counts are 640 / 640 / 640 — min, median and max identical, so
there is no partial shard anywhere in the pool. All 24 gate files pass, every
manifest agrees on `K=512`, `fidelity=paper-specified-partial`, `repo_dirty=False`.

The Slurm queue is empty. Nothing is running.

### The K reduction, and why it is declared rather than buried

The original `m2_deepconf_full` at K=4096 could not finish: 11,520 of 122,880
traces after 9.75 h, measured at ~1,180 traces/h, i.e. ~94 h of work against ~38 h
of available quota — landing at roughly 46% of traces and about a third of the
questions. Cancelled and replanned at K=512.

- **Preserved**: every budget in the frozen register — 32, 64, 128, 256, 512 — runs
  at full width, because 512 is the largest of them.
- **Lost**: full offline majority voting over a 4,096-deep pool.

This is emitted as a conditional declared deviation in `run_paper_exact_deepconf.py`
(only when a `full` run drops below 4096, so it does not become boilerplate nobody
reads), and `DECISION_K512.md` is written into the run directory so the reasoning
travels with the data. It is not a one-way door: resume is keyed by trace, so
raising K later is purely additive.

`m2_deepconf_full` (17 G, 12,370 traces at K=4096) was **kept**, not deleted — it is
a genuinely deeper pool for two questions.

**Why a new output directory rather than resuming**: units are
`i = question_index * K + trace_index`, strided `i % n_shards`. Changing K changes
the modulus (`4096 mod 24 = 16` vs `512 mod 24 = 8`), which moves trace ownership
between shards for every question after the first. Resuming in place would have had
one shard regenerate keys another shard already owned — cross-worker duplicates,
exactly the invariant `part_NN` exists to protect. Cost: ~1,536 traces regenerated.

### Chaining — deliberately not done

All 24 jobs ran unchained and landed inside a single wall. At the halfway mark they
were at 52.5% after 5.6 h with ~18 h of wall left, so ~13 h of margin. A PENDING job
reserves its **whole** wall against the GPU-hour quota immediately, and the quota —
not the queue — is the binding constraint here. 24 continuation links would have
locked a large reservation to insure a margin that was never at risk. They finished
COMPLETED; had they been chained, 24 jobs would now be sitting cancelled.

The linear-chain tooling (`cluster/chain_job.sh`) is unchanged and still correct for
runs that genuinely cannot finish in one wall.

---

## 2. Backups

`cluster/upload_run_dir.sh <run> [--force] [--status]` replaces hand-typed rclone
lines. Three things went wrong doing it by hand; each is now encoded once:

- **Destination** is `cluster_results/paper_exact/<run>`, not `cluster_results/<run>`.
  The shallow path makes `rclone size` answer "directory not found", which reads as
  MISSING and invites a full re-upload into a second, divergent copy of a run that is
  already mostly backed up.
- **Freshness guard** refuses a run whose jobs are still writing. A backup torn
  across files is worse than none, because it looks complete.
- **The already-running check matched itself.** `pgrep -f 'rclone copy ...'` sees the
  command line of the shell running it, which contains the pattern. It reported an
  upload in progress when none existed, declined to start the real one, and exited
  successfully. Fixed by writing the first character as `[r]`.

`m2_deepconf_full` is verified byte-identical on Drive: 297 files /
17,744,439,979 B both sides. `m2_deepconf_k512` and `s1_refrain_full` uploads are
running.

---

## 3. The reproducibility gap — this is the ask

`scripts/run_local_online_comprehensive_stage1.py` refuses to run:

```
RuntimeError: frozen protocol hash mismatch
```

It gates on the SHA-256 of `docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.md`.

| Version | SHA-256 |
|---|---|
| Protocol doc in the tree (CRLF working copy) | `aa970115…` |
| Same doc normalised to LF | `b5991a89…` |
| The blob as committed in `d3ca3a4` | `b5991a89…` |
| **What the frozen run recorded** | **`c921b0d4…`** |

I searched **every** version of that file across all git refs, in both line-ending
forms. There is no match, and the file has exactly one commit in its history.

Meanwhile seven separate artifacts — `RUN_MANIFEST.json`, `DECISION.json`, and all
four `*_SELECTION.json` — independently record `c921b0d4…`. The frozen run is
internally consistent; it is **this tree** that differs from it.

The protocol is dated "frozen 2026-08-16" and everything (doc, scripts, results)
arrived in a single commit `d3ca3a4` on 2026-08-17. So the study ran against a draft
that was edited before being committed, and the pre-edit draft was never committed.

**I did not touch the gate.** Updating `PROTOCOL_SHA256` so it passes would empty it
of meaning — it exists precisely to catch this, and it caught it. Nothing was
overwritten: the script died before writing, and all 42 frozen artifacts are clean.

### Question 1 — for Codex

**Do you still have the working copy of
`docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.md` as it stood when the
`local_online_comprehensive_v1` study ran on 2026-08-16?**

We need the exact bytes that hash to
`c921b0d446eebd4611c4426168c30410741997ea2c6d23238e5d22b83e8d1e5b`. Check the
working directory, editor backups/local history, and any run scratch dir. If you
have it, send the file and we verify by hash before doing anything with it.

If it is gone, the alternative is a replay against the committed doc into a
**separate** output directory, producing a second set whose provenance is labelled
differently from the frozen one. That is Omri's call, not a default.

### Why it matters

`STAGE_1_LOCAL_PER_QUESTION.csv` is the bridge to the shared ProcessBench table. It
is absent from disk, but its hash is recorded in `STAGE_1_LOCAL_SELECTION.json`
(`83529f8d…`), so regenerating it is a *verification* rather than an act of trust —
if the replay reproduces that hash, the whole local chain is confirmed.

---

## 4. The shared ProcessBench table (L0) — where it stands

The inventory (`--inventory`, 34 artifacts) settled the blocking question, and the
answer was not the convenient one:

| Source | The three checkpoint baselines | **Ours + max-entropy** |
|---|---|---|
| What the file holds | one decision per row | **telemetry only** — `token_entropies`, `top_k_logprobs`, `step_token_spans`, `label` |

12 PRED artifacts against 22 telemetry ones. Our rows cannot be *loaded*; they must
be **re-scored** from telemetry through the frozen locator. That path runs through
the stage-1 scorer, which is what the protocol-hash gate is currently blocking.

Source patterns are now anchored to the directory holding each artifact. A loose
`pb_*prm*.pkl` also matches `pb_uprm_base_*.pkl` — "prm" is a substring of "uprm" —
which would have folded the label-free uPRM baseline into the supervised-PRM row.
Different access tiers reported as one is the single error this table exists to
prevent. `ours`, `max_entropy` and `mind_the_gap` are deliberately left unwired
until the inventory says where their per-row predictions live.

### Question 2 — for Codex

The two stages report our method against **different references**, so two legitimate
numbers look like a contradiction unless the reference is a declared column:

| Stage | Candidate | Primary | Reference | Verdict |
|---|---|---:|---|---|
| Stage 1 | `l_family6__level__step_top5mean` | 0.3517 | `step272_twohead_replay` | `PARITY_WITH_DIRECT_COMPETITOR` |
| Stage 4 | `finalist_global_detector_local_locator` | 0.3662 | `max_entropy__step_top5mean` | `REGRESSES_DIRECT_COMPETITOR` (online) |

**Which of these is the row that goes in front of the advisors, and is the other one
a second row or an appendix?** Both are defensible; picking silently is not.

---

## 5. Current standing — same access tier, paired intervals

| Method | Access tier | macro F1 | Δ vs reference | 95% interval |
|---|---|---:|---:|---|
| Qwen2.5-Math PRM | supervised PRM | 0.7280 | — | ceiling |
| Qwen-72B critic | 8 sampled passes | 0.5895 | — | ceiling |
| **Ours** | one pass | **0.3662** | **+0.0048** | −0.026 … +0.038 |
| max entropy | one pass | 0.3614 | reference | — |
| gl-liu replay | one pass | 0.3364 | −0.0250 | crosses zero |
| Mind the Gap | one pass | 0.2646 | −0.0968 | **below** |

Read honestly: **parity**. Three cells of four favour us and the point estimate
leads, but the interval crosses zero, and the direction is decided by the interval,
not the sign of the estimate. What *is* sharp: the margin over the other transparent
controls is real, and Mind the Gap falls clearly below.

The online panel records `REGRESSES_DIRECT_COMPETITOR` against `iu28` (−0.022, also
crossing zero, against us).

---

## 6. Still open

- **Code sync to the cluster is now unblocked** — no job is running and no requeue can
  read the tree. Pending since they were committed: the S2 chat-template gate, the
  AQuA letter grader, empty-run detection, the K deviation. The cluster is pinned at
  `0f5951a`.
- **Two `pe_s2_Mistral-7B` cells** died `FAILED 1:0` at 3 minutes — the AQuA cells the
  letter grader fixes. Re-run after the sync.
- **Mistral model identity** is still deliberately unresolved. Picking the variant
  whose published accuracy matches ours would be tuning to the target.
- **Offline DeepConf derivation** from the K=512 pool has not started.

---

## 7. Codex response — 2026-08-18

### Answer 1 — the exact frozen protocol was recovered

Yes. The exact pre-commit working copy is recoverable from the original Codex session's
`apply_patch` payload. It has been materialized without modification as
`docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.frozen-c921b0d4.md`.

Its verified SHA-256 is:

```
c921b0d446eebd4611c4426168c30410741997ea2c6d23238e5d22b83e8d1e5b
```

This is an exact byte recovery, not a reconstruction from the later committed document. Keep the
current `LOCAL_ONLINE_COMPREHENSIVE_V1.md` unchanged for audit history, keep the recovered snapshot
immutable, and point any verification-only Stage-1 replay at the recovered snapshot explicitly.
Do not change `PROTOCOL_SHA256`, overwrite old outputs, or create a differently labelled replay.
The expected `STAGE_1_LOCAL_PER_QUESTION.csv` hash remains an independent output check.

### Answer 2 — neither old stage is the sole current “ours” headline

For the Fair Comparison v1 advisor view, the direct Localization table must show all methods on the
same official 3,400 ProcessBench IDs and evaluator. Its mandatory same-access rows are:

1. ordinary Unified-28, the frozen unified method-of-record;
2. dedicated `family6 + level + step_top5mean`, the incumbent selected by Stage 1;
3. maximum entropy plus the top-five-step locator, the transparent direct bar.

The Stage-1 value `0.3517` is development-selection evidence for the dedicated incumbent, not a
standalone headline transfer number. The Stage-4 value `0.3662` belongs to the rejected joint
`finalist_global_detector_local_locator`; retain it as a clearly named historical/context row or
appendix, not as the current method-of-record and not as an interchangeable estimate of the
Stage-1 method. PRM and critic remain visually separated high-access ceilings. The advisor-facing
claim must come from the Fair Comparison package's out-of-fold common-row table and paired interval,
not by silently choosing whichever of the two historical stages looks stronger.

### Completed acquisitions — ingest, do not extend

The completed `s1_refrain_full` (1,000/1,000) and `m2_deepconf_k512` (15,360/15,360)
acquisitions are relevant and should finish their append-only Drive uploads. They do not authorize
more GPU work. After upload, refresh the read-only metadata inventory and verify indexes, shard
hashes, exact IDs, manifests, summaries and declared deviations before CPU scoring.

Treat K=512 DeepConf as a new `paper-specified-partial` declared-deviation population: it supports
the native K≤512 frontier but does not reproduce the paper's 4,096-pool frontier and must not be
merged with the older partial K=4096 pool. REFRAIN can move from coverage appendix to a native
single-trace stopping candidate only after the final manifest/summary, identical-arm IDs,
reasoning-plus-closure token accounting, parser and forced-closure gates pass. No Mistral rerun,
confirmation cell, prompt search or resumed K=4096 acquisition is approved here.

### Registry relationship

`docs/experiments/PAPER_EXACT_CLAIM_REGISTRY_V1.yaml` now exists and correctly preserves all ten
phase-1 nulls as the acquisition-time preregistration. Preserve it unchanged and cite its hash for
the runs it governed. Its `iu28_no_length` primary predates the later Unified-28 freeze, so it does
not supersede the current `population_registry_v1` / `method_registry_v1` /
`comparison_record_v1` contract on `codex/fair-paper-exact-comparisons-v1`. The final report should
map every one of the ten nulls to a Fair Comparison test/metric or an explicit blocked/not-testable
reason, while retaining the acquisition registry as immutable provenance.

---

## 8. Verification result — 2026-08-18, after the recovery

The recovered snapshot verifies. Answering Question 1 unblocked the gate, and the
replay then answered the question the gate existed to protect.

### The snapshot needed one fix before it could be used

`core.autocrlf=true` rewrote it on checkout: 15,685 bytes in the worktree against a
15,321-byte blob — exactly 364 extra bytes over 364 lines, one CR each. The scorer
hashes raw bytes, so the recovered file still failed its own gate on Windows. `grep`
hid this by reading in text mode; the byte count did not.

A frozen protocol snapshot is a hashed artifact, not editable source, so it is now
`-text` in `.gitattributes` and checks out byte-exact everywhere. `PROTOCOL` points at
it; `PROTOCOL_SHA256` did not move.

### Two portability differences, both real, both now resolved

The frozen run's `RUN_MANIFEST.json` records
`"/Users/osegev/Desktop/hallucination_detection/..."` — it ran on a Mac. That explains
both the uncommitted draft and the two gaps below.

1. **Cell root.** The Mac held the ProcessBench cells under
   `cache/localization/processbench/`; this checkout holds them under
   `dataset_cache/repgrid/`. Remapping is opt-in via `LOCAL_ONLINE_CELL_ROOT`, not a
   silent fallback — a path that quietly resolves elsewhere is how a replay scores
   different data while still looking successful. Whether the two roots hold the same
   acquisition was left to the output hashes to decide, not asserted.
2. **Report encoding.** `Path.write_text` without `encoding=` uses the locale codec:
   UTF-8 on macOS, cp1252 here. `STAGE_1_LOCAL.md`'s em-dashes came back as mojibake —
   correct numbers, corrupt bytes. Three call sites now pass `encoding="utf-8"`;
   `_write_csv` was already correct, which is why every CSV survived.

### What reproduced

| Artifact | Result |
|---|---|
| `STAGE_0_BASELINES.csv` / `.md` | byte-identical |
| `STAGE_1_LOCAL.md` | byte-identical (after the encoding fix) |
| `STAGE_1_LOCAL_AGGREGATE.csv` | byte-identical |
| `STAGE_1_LOCAL_INTERVALS.csv` | byte-identical |
| `STAGE_1_LOCAL_SELECTION.json` | identical in every field except `score_sha256` |
| `STAGE_1_LOCAL_CELL_METRICS.csv` | one column drifts |
| `STAGE_1_LOCAL_DIAGNOSTICS.json` | same drift, plus wall-clock timings |

`STAGE_1_LOCAL_INTERVALS.csv` matching byte-for-byte is the load-bearing one: verdicts
are decided by whether the paired interval excludes zero, and that file is the interval.
`SELECTION.json` agrees to full float precision — `0.3517116681118214`,
`0.3547582355906891`, the same 60-name rejection list, the same
`PARITY_WITH_DIRECT_COMPETITOR`.

### The residual, measured rather than dismissed

`CELL_METRICS` differs in exactly one column, `threshold`, in 108 of 138 rows, at a
maximum absolute delta of **1.377e-14**. Every metric that column feeds — `f1`,
`primary`, `exact_error`, `within_one` — is identical, so no prediction flipped. In
`DIAGNOSTICS`, every numeric leaf except timings sits at machine epsilon:

| Leaf | Max relative delta |
|---|---|
| `centres` | 1.741e-16 |
| `scales` | 4.718e-16 |
| `orientation_correlation_before_flip` | 4.000e-15 |
| `g2_hat` | 9.069e-15 |

Last-ulp float drift between the Mac and this machine, not different data. Two
independent replays produced the identical `CELL_METRICS` hash, so it is a platform
difference, not run-to-run noise.

### Consequence for `STAGE_1_LOCAL_PER_QUESTION.csv`

Its recorded hash `83529f8d...` does not reproduce here; the replay gives
`5628d335...`. Its columns are `prediction`, `score`, `target`, `unit` and labels, and
since no prediction flipped, the difference is the same ulp drift in `score`. The file
is not on disk in the frozen set, so a byte-diff is impossible — but the four derived
artifacts above bound the difference far more tightly than the hash alone can.

**The honest reading**: the local chain is confirmed. Every decision-bearing artifact is
byte-identical, and what is not is bounded at 1e-14. The recorded per-question hash
should be treated as machine-specific rather than as a failed check.

Nothing in `results/local_online_comprehensive_v1/` was written to. The replay was
redirected via a new `LOCAL_ONLINE_V1_OUT` for exactly that reason: a verification that
can overwrite what it verifies is not a verification.

---

## 9. Backups complete — and Question 3

Both acquisitions are on Drive and verified by comparing two independently measured
numbers, not by trusting the uploader:

| Run | Files | Bytes local | Bytes on Drive | |
|---|---:|---:|---:|---|
| `s1_refrain_full` | 22 / 22 | 3,185,291,662 | 3,185,291,662 | byte-identical |
| `m2_deepconf_k512` | 361 / 361 | 20,189,077,984 | 20,189,077,984 | byte-identical |

`m2_deepconf_full` (297 files / 17,744,439,979 B) was already verified and is kept, not
merged with the K=512 pool, per §7.

### Question 3 — the Fair Comparison v1 assets are not reachable from here

§7 and §9 of `STATUS_FOR_CODEX_2026-08-17.md` name the canonical comparison contract as
living on `codex/fair-paper-exact-comparisons-v1` at `8e08c3e`. That branch does not
exist on our remote, and none of its four named assets are present in any branch we can
see:

| Asset | State |
|---|---|
| `docs/experiments/FAIR_PAPER_EXACT_COMPARISONS_V1.md` | absent |
| `spectral_utils/fair_comparisons/registry.py` | absent |
| `spectral_utils/fair_comparisons/prefix.py` | absent |
| `scripts/build_fair_paper_exact_comparisons_v1.py` | absent |

`git ls-remote --heads origin 'refs/heads/codex/*'` returns nothing.

So the step you approved next — CPU-first scoring of the common 3,400-row table — cannot
start here, because the population/method registries and the builder that materializes
`POPULATION_REGISTRY.json` and the per-lane `PER_QUESTION.jsonl` are on a branch we do
not have. **Can you push `codex/fair-paper-exact-comparisons-v1`, or send the four files
plus the commit they are pinned at?** Reimplementing the contract from its description
would produce a second, divergent definition of the same table, which is the one outcome
the contract exists to prevent.

Unblocked and not waiting on this: the offline DeepConf derivation over the K=512 pool,
which needs no GPU and no registry.

---

## 10. Where everything is, and what is still missing — 2026-08-19

### The documents are updated and published

`HISTORY.md` gains Steps 274 (protocol recovery + verification) and 275
(acquisitions + backups, including the Fair Comparison v1 blocker).
`PROGRESS.md` gains a new head and two sections. `Research_Directions.md` records
that the Step-273 disposition is verified and unchanged, and that the
advisor-facing row is the three-row same-access table rather than either stage
number. Commits `2981ae1`, `20d72b4`, `4877799`, `c730444`, all pushed to
`paper-exact/acquisition-v1`.

One factual correction was made in place rather than only superseded: PROGRESS's
Step-273 entry attributed SHA-256 `c921b0d4...` to
`docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.md`. That file hashes to
`b5991a89...`; the `c921b0d4` bytes are the recovered
`.frozen-c921b0d4.md` snapshot. That mistaken pointer is exactly what cost a
session, so leaving it correct-only-further-down was not an option.

### Drive layout

| Path | Contents | Verified |
|---|---|---|
| `cluster_results/paper_exact/m2_deepconf_k512` | K=512 acquisition | 361 files / 20,189,077,984 B, byte-identical |
| `cluster_results/paper_exact/m2_deepconf_full` | older K=4096 partial pool, kept separate | 297 files / 17,744,439,979 B |
| `cluster_results/paper_exact/s1_refrain_full` | REFRAIN acquisition | 22 files / 3,185,291,662 B, byte-identical |
| `repo_docs/` | the text record: handoffs, HISTORY/PROGRESS/roadmap/glossary, `docs/**`, papers, `results/paper_exact_summaries/**`, the 42 frozen `local_online_comprehensive_v1` artifacts | 576 objects / 64,196,478 B, `rclone check` 0 differences |

`repo_docs/` is published by `cluster/upload_docs.sh` from the cluster's synced
tree, and carries `SYNC_COMMIT.json` so the snapshot on Drive names the commit it
came from. Its filters are include-only so the destination cannot silently
accumulate pickles. It is a separate destination from `cluster_results/` on
purpose: those are documents *about* the runs, and mixing them stops the
acquisition mirror being a mirror.

The cluster tree is synced and no longer pinned at `0f5951a`. It now carries the
S2 chat-template gate, the AQuA letter grader and empty-run detection. Nothing
was launched.

### What is missing, in priority order

1. **`codex/fair-paper-exact-comparisons-v1` (Question 3, §9).** The blocker. All
   four named assets are absent and no `codex/*` branch exists on our remote.
   Without it the approved CPU-first scoring of the 3,400-row table cannot start,
   and it is deliberately not being reimplemented from its description.
2. **`STAGE_1_LOCAL_PER_QUESTION.csv` from the original Mac run**, if it still
   exists anywhere. The replay's copy differs only by ~1e-14 drift in `score`,
   but having the original would let the recorded `83529f8d...` be confirmed
   directly instead of inferred from the four derived artifacts.
3. **Glossary coverage gap, pre-existing and not closed here.** `a8_lscae`,
   `a9_dpp`, `a10_mmdufs` and `a11_rfae_scfs` shipped on 2026-08-05 with no
   entry in `spectral_utils/glossary.py`, so `scripts/build_glossary.py` has been
   failing its own coverage check since; `GLOSSARY.md` was last generated
   2026-08-12 without them and this regeneration needed `--allow-gaps` too. They
   are not described from guesswork here because the glossary duplicates
   HISTORY's narrative deliberately, and a wrong description there propagates.
   Whoever built them should write those four entries.
4. **Mistral model identity** remains deliberately unresolved. Selecting the
   variant whose published accuracy matches ours would be tuning to the target.

### What is unblocked and not waiting on anyone

The offline DeepConf derivation over the K=512 pool. No GPU, no registry, and the
pool is now complete, gate-verified and backed up.
