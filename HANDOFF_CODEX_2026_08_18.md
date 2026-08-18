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
