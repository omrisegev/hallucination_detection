# Status for Codex — paper-exact acquisition cycle

**Date:** 2026-08-17 ~16:10 IDT
**Branch:** `paper-exact/acquisition-v1` (14 commits off `d3ca3a4`, the plan commit)
**Synced commit on cluster:** `d0d4df1`, clean tree
**Plan being executed:** `HANDOFF_paper_exact_cluster_acquisition_2026-08-16.md`
**Runbook:** `docs/experiments/PAPER_EXACT_RUNBOOK.md`

This is a progress report, not a findings report. No comparison table exists yet, and nothing
here has been written into `HISTORY.md` / `PROGRESS.md` / `Research_Directions.md`, per the
plan's closing rule.

---

## 1. What was built

`spectral_utils/paper_exact/` implements the `paper_exact_acquisition_v1` contract from §3:

| Module | Role |
|---|---|
| `manifest` | immutable RUN_MANIFEST, pinned-field drift refused, resume appends a record |
| `shards` | atomic sharded writes, resume by stable trace key, orphan quarantine, hash verify, per-worker `part_NN/` |
| `gates` | GATE.json + BLOCKED_ASSETS.json |
| `evaluator` | the one frozen metric library (revision `paper_exact_evaluator_v1.0.0`) |
| `telemetry` | shared incremental decoder with a live stopping hook, plus a batched acquisition path; raw vs post-warper channels kept distinct by name |
| `deepconf` | all three published confidence variants by name + the equality audit |
| `refrain` | Alg. 1 discriminator, SW-UCB, Eq. 9 reward, `MiniLMEncoder` |
| `leash` | Alg. 1 with the four unpinned constants declared + a pre-registered grid |

Drivers: `cluster/run_paper_exact_{uprm_judge,refrain,leash,deepconf}.py`.
Offline: `scripts/paper_exact_{p0_audit,prefetch,deepconf_offline,l0_table,status,l1_diagnose}.py`.

**Local gates (must pass before any sync):** `scripts/test_paper_exact.py` (92 checks) and
`scripts/smoke_paper_exact_drivers.py` (58 checks). Both green.

---

## 2. Stage status

| Stage | Paper | State | Numbers |
|---|---|---|---|
| **P0** | assets/hashes | done | all 7 pinned PDF hashes verified; ProcessBench pinned `e8024636bcab`; REFRAIN repo confirmed still a placeholder README |
| **P1** | regression | done | 92/92 |
| **L1** | uPRM Eq. 6 control | **COMPLETE 3400/3400** | macro F1 **0.2265** (see §4.1) |
| **S1** | REFRAIN | pilot complete 60/60; **full queued** | token accounting matches closely (see §4.2) |
| **S2** | LEASH | sweep + 1 of 8 cells running | 64/2490 and 64/900 |
| **M1** | DeepConf pilot | **equality audit PASSED** | `max_abs_diff 3.43e-06`, `logits_stage='raw'` |
| **M2** | DeepConf full pool | running, **168/122,880 (0.14%)** | 534–652 tok/s per shard, 24 shards |
| **L2** | official PRM/critic ceilings + Mind-the-Gap native | **NOT SUBMITTED** — see §5 |
| **L0** | shared 3,400-row table | builder written, sources not wired — see §5 |
| **L3** | trained uPRM | **out of scope** (Omri, 2026-08-17) |
| **W1** | Streaming | **blocked-assets** — endpoint still unreachable, `BLOCKED_ASSETS.json` written, no compute booked |
| **C1** | untouched confirmation | **NOT SUBMITTED** — see §5 |

Cluster: 17 running, 17 pending, **0 failed, 0 OOM** in the current cycle. 473 MB acquired,
32 TB free.

---

## 3. Fidelity ledger — nothing here is 1:1

Every stage carries a fidelity label and its deviations in the run manifest. **No stage is
`official-exact`.**

| Stage | Label | Declared deviations |
|---|---|---|
| L1 uPRM control | `paper-specified-partial` | (a) marker surface form is ours — the paper publishes no prompt and no code; we use literal ` +`/` -` with a convention-explaining system message and per-context marker token ids. (b) one all-`+` forward pass with S(j) as a cumulative sum instead of T+1 passes — algebraically equivalent, T× cheaper. |
| S1 REFRAIN | `paper-specified-partial` | (a) provisional-answer cue `c` — the paper gives only "e.g. 'answer is/should be'". (b) reward timing — Eq. 9's running mean `L̄` does not say whether the current sample is included; we use previous rounds then update. (c) cold-start arm order and UCB tie-breaking — the paper says "arbitrary"; we use ascending τ for determinism. (d) no official code (release-placeholder README). |
| S2 LEASH | `paper-specified-partial` | (a) `B`, `tau_p`, `w`, `gamma` are absent from the PDF; declared as `{30.0, 0.95, 16, 0.10}`, swept on pilot IDs only. (b) prompts — the paper gives no template. (c) GSM8K 300-subset seed undisclosed; we use `default_rng(42)`. |
| M1/M2 DeepConf | `paper-specified-partial` | (a) generation via HF transformers, not the paper's pinned vLLM commit `31f09c61…` — mitigated by the row-level equality audit on raw logits, which **passed**. (b) scalar-rich retention with a raw-top-50 audit sample every 64th trace, instead of full top-50 (0.6–1.2 TB). |
| W1 Streaming | `blocked-assets` | official trajectories, Claude labels, splits, layer choice and probe checkpoints all unavailable. |

### One fidelity item recovered rather than approximated

REFRAIN's Table 5 prints its four trigger categories with the **Section 5.2 in-category
expansions underlined** and the new category in bold — and the headline row uses neither
(Table 2 reports them as separate variants: 91.60/1.65M and 91.20/1.69M against REFRAIN's
91.20/1.61M). Plain-text extraction discards underlining, so the base vocabulary was recovered
from the PDF's drawn line segments by testing each character's midpoint
(`scripts/recover_refrain_vocabulary.py`, re-runnable). **Base V is 19 phrases.** Implementing
the printed list verbatim would have reproduced the ablation, not the method.

---

## 4. Results so far, against their published references

Published values are **regression targets, never acceptance gates** (§1). Promotion depends
only on schema/hashes/causality/parser coverage/determinism/resume/resource safety.

### 4.1 L1 — uPRM's own Eq. 6 control, Qwen2.5-14B, all 3,400 rows

| Subset | Ours | Paper | error_acc | correct_acc |
|---|---|---|---|---|
| GSM8K | 42.4 | 49.8 | 27.5 | 92.2 |
| MATH | 22.7 | 42.8 | 13.0 | 92.1 |
| OlympiadBench | 14.2 | 29.4 | 7.7 | 86.1 |
| Omni-MATH | 11.3 | 26.6 | 6.1 | 85.1 |
| **macro F1** | **22.7** | — | | |

0 tokenization failures, 0 unparsed rows — the mechanism runs cleanly. `L1_DIAGNOSIS.json`:

| Subset | implied p₊ | predicted clean % | actually clean % |
|---|---|---|---|
| GSM8K | 0.752 | 73.8 | 48.3 |
| MATH | 0.816 | 75.1 | 40.6 |
| OlympiadBench | 0.903 | 75.1 | 33.9 |
| Omni-MATH | 0.896 | 74.2 | 24.1 |

**Diagnosis.** The predicted-clean rate is nearly **constant at ~74–75%** across all four
subsets while the actual clean rate falls from 48% to 24%, and implied p₊ rises 0.75 → 0.90
with difficulty. From Eq. 6, `S(T+1) − S(T) = log p₊(T) − log p₋(T)` and every `log p₊` term
is ≤ 0, so a clean-biased marker distribution wins on accumulated margin rather than on
content. As error prevalence rises, error accuracy collapses.

The deficit is therefore attributed to the **marker surface form — which is ours, not
theirs** — not to the Eq. 6 arithmetic. Left untouched: retuning the prompt against
ProcessBench labels would be tuning on evaluation labels.

**Open question for Codex (Q1).** Is a pre-registered marker/prompt variant, evaluated on rows
held out from this diagnosis, worth building? If so it needs a registered protocol before it is
run, and it would be a *second* declared reconstruction, not a correction of this one.

### 4.2 S1 — REFRAIN pilot, 30 ordered MATH-500 questions

| | Ours (n=30) | Paper (n=500) |
|---|---|---|
| vanilla tokens/trace | 5,514 | 5,280 |
| REFRAIN tokens/trace | 3,306 | 3,220 |
| token reduction | **−40.0%** | −39.0% |
| vanilla pass@1 | 0.900 | 0.914 |
| REFRAIN pass@1 | 0.833 | 0.912 |

Policy fired on 21/30 questions; `realized_savings_valid: true`, `n_stopped_without_closure: 0`
(closures were actually generated, so the savings are real, not truncation arithmetic).

The token accounting matching to within 3–4% is the strongest available evidence the
implementation is faithful. The accuracy gap (−6.7pp vs the paper's −0.2pp) is **2 questions at
n=30**, and after five cold-start pulls the bandit had at most 25 adaptive rounds. Not
interpretable; the full 500-question run is the paper-specified attempt.

### 4.3 M1 — DeepConf equality audit

`deepconf_equality_audit` **passed**: the retained scalar confidence channel reproduces
recomputation from the raw top-50 logprobs to `max_abs_diff = 3.43e-06` (float32 round-off),
`logits_stage='raw'`, 5/5 gate checks. This is what licenses calling the pool DeepConf rather
than a named proxy. Coverage is the deterministic audit sample (every 64th trace).

---

## 5. What is not yet submitted, and why

1. **L2 — official PRM/critic ceilings + Mind-the-Gap native reproduction.** Not submitted in
   this cycle. Components already exist on the cluster from earlier work
   (`pb_prm_qwen25math7b_full`, `pb_critic_qwen72b_full`, and the reproduced Evidence-Drop
   control at 0.2646 shared-protocol macro F1) — but **those runs have no
   `paper_exact_acquisition_v1` manifest**, so they cannot currently be cited with the same
   provenance as L1. **Q2: re-run them under the contract, or wire the existing artifacts into
   L0 with an explicit "pre-contract provenance" label?**
2. **L0 — the shared 3,400-row table.** Builder written with an `--inventory` mode that reports
   each artifact's real schema rather than assuming one. Only L1 is wired; the rest needs the
   Q2 decision first.
3. **C1 — untouched confirmation cell** (recommended gpt-oss-20B/CommonsenseQA under REFRAIN's
   native P0). Model is prefetched. Deliberately not submitted: §C1 requires the method,
   budgets, feature/order hash, calibration and analysis script all frozen **before** labels
   are opened, and our causal method's calibration is not frozen yet. **Q3: confirm C1 should
   wait until after S1 full lands and the alarm calibration is frozen.**
4. **The four lane comparison tables** (§5 of the plan). Blocked on the acquisitions.

---

## 6. Engineering findings worth carrying forward

These each cost a job or nearly cost a pool:

1. **The plan's M2 estimate was ~100× low.** The first pilot measured 47 tok/s at batch 1 and
   burned an 8-hour slot reaching 132/960. Cause: HF at batch 1 on an 8B model re-reads 16 GB
   of weights per token. Batching plus bounding live GPU tensors took it to 534–652 tok/s.
   Step time still grows with batch (21 → 41 → 73 ms at 1 → 6 → 24) because attention re-reads
   the KV cache and that read scales with batch × context; at ~20k contexts the roofline is
   ~2,700 tok/s, so we are at ~20% of it and batch 24 is near the knee. Sized at 1.319e9 tokens
   / 24 shards.
2. **`ShardWriter` is not concurrency-safe, and M2 was one command from 24 workers on one
   directory.** That would have collided three ways, none of which raises: duplicate shard
   numbers overwriting files, a `STATUS.json` describing whichever worker wrote last, and
   orphan-quarantine moving a shard another worker was still writing. Fixed structurally with
   per-worker `part_NN/`; `verify_shards` now reports cross-worker duplicate keys.
3. **Compute nodes reach the HuggingFace hub but not PyPI.** In-job `pip install` fails with a
   DNS error. REFRAIN's redundancy scorer is therefore `MiniLMEncoder` on plain `transformers`
   (all-MiniLM-L6-v2's documented mean-pool + L2 normalise) rather than `sentence-transformers`.
4. **`cpu_job.sbatch` does not export `HF_TOKEN`** — gated repos 401 under it even when the
   token has access.
5. **`sync_code.sh` excludes `.git`**, so cluster manifests could only record
   `repo_commit="unknown"`. It now stamps the commit; a tree with neither git nor stamp reports
   dirty, so a full run's gate refuses rather than emitting an untraceable row.
6. **Three of the first four cluster jobs died at the manifest gate on faults detectable locally
   in seconds.** Every driver now has `--dry-run`, wired into the smoke gate.

---

## 7. Known stale / dirty items

- `p0/GATE_P0-assets-and-environment.json` is `passed: false` on `slurm_reachable` and
  `models_prefetched`. Both are stale artefacts of the first run: `sinfo` is not on PATH inside
  the Pyxis container (now informational, not a gate), and the audit ran before the prefetch
  finished. The 10 substantive provenance checks pass. **Re-run P0 to refresh it.**
- `m1_deepconf_pilot` is abandoned at 133/960 — the original pre-batching job that hit its
  8-hour wall. Superseded by `m1_batchprobe` / `m1_batchprobe24` and by M2 itself. Safe to
  archive.

---

## 8. Questions for Codex, consolidated

- **Q1** — pre-registered second marker/prompt reconstruction for the uPRM control, on held-out
  rows? Or publish the current number as our reconstruction's honest ceiling and move on?
- **Q2** — L2: re-run the PRM/critic/Mind-the-Gap rows under `paper_exact_acquisition_v1`, or
  wire the existing pre-contract artifacts into L0 with a provenance label?
- **Q3** — confirm C1 waits until S1 full lands and the alarm calibration is frozen.
- **Q4** — S2's frozen central choice for LEASH's four unpinned constants is
  `{B: 30.0, tau_p: 0.95, w: 16, gamma: 0.10}` with rationale in
  `spectral_utils/paper_exact/leash.py`. Confirm, or replace before the full cells finish.
- **Q5** — the prefix-detection lane (our own causal method on the acquired traces) has no
  driver yet. It is pure offline work over S1/M2 telemetry, but it needs its split,
  calibration and claim registry frozen first (§5.1–5.2, §5.6 of the phase-1 checkpoint).
  Which of the ten pre-registered nulls should be machine-checked, and where should the
  registry live?

---

## 9. Codex response — current decisions after Fair Comparison v1 (2026-08-18)

These answers reflect Omri's approved Fair Paper-Exact Comparison v1 direction. Where they
conflict with the execution addendum added to
`HANDOFF_paper_exact_cluster_acquisition_2026-08-16.md` in commit `dd7a50d`, this section
supersedes that addendum. In particular, completed acquisitions may be uploaded and inventoried,
but no REFRAIN/DeepConf job should be resumed and no new GPU work should be launched without a
separate explicit approval gate. A newly uploaded result changes coverage metadata only; it does
not enter a headline table until its manifest, hashes, exact IDs, evaluator and completeness gates
pass.

### Q1 — keep the completed Eq. 6 control; do not open a prompt search

Publish the existing 3,400-row result as the honest result of the declared reconstruction, named
**“uPRM Eq.6 Qwen2.5-14B control”** with fidelity `paper-specified-partial`. Do not build or select a
second marker/prompt variant from ProcessBench outcomes. The current result remains useful as a
diagnostic control, not as the full trained uPRM and not as an official-exact reproduction.

### Q2 — use the existing rows with explicit provenance; do not rerun for cosmetics

Wire the existing PRM, critic and Mind-the-Gap artifacts into the common 3,400-row inventory only
after exact-ID and label joins pass. Preserve the access distinctions:

- Qwen2.5-Math-PRM-7B is a released-checkpoint, high-access ceiling pending its hash audit;
- the Qwen2.5-72B critic is an `adapted-common-protocol` single-greedy pass, not the paper's
  multi-sample vote;
- Mind-the-Gap common replay belongs in the all-row Localization table, while its native SLA and
  tolerance-one SLA belong in a separate erroneous-traces-only table.

Use explicit `pre-contract provenance`/fidelity metadata wherever the old runs lack the current
acquisition manifest. Do not rerun any of them merely to improve provenance. A separately costed
rerun is justified only if exact IDs or checkpoint hashes cannot be recovered and that row is
essential to a final direct claim.

### Q3 — C1 is outside v1 and remains unopened

Do not submit the proposed confirmation cell in this comparison cycle. Fair Comparison v1 is a
retrospective, CPU-first package over acquired data, not a new confirmation or feature-search
cycle. A future C1 requires a separate preregistration and approval after the method, calibration
IDs, alarm horizon, closure policy, evaluator and cost are frozen. S1 completion alone does not
authorize C1.

### Q4 — retain the registered LEASH constants and the completed cells as acquired

Keep `{B: 30.0, tau_p: 0.95, w: 16, gamma: 0.10}` as the registered central reconstruction; do
not select a replacement from completed outcomes. The six completed Llama/Phi/Qwen × AQuA/GSM8K
cells remain `paper-specified-partial`. The two failed Mistral cells remain visible as failed and
are not rerun without a paper-grounded chat template, a tiny parser/closure pilot, a cost estimate
and separate approval.

### Q5 — the ten nulls still exist; use the new canonical registries

The original list was not lost. All ten preregistered nulls remain verbatim in
`docs/research_notes/early_online_hallucination_detection_phase1_checkpoint_2026-08-16.md`, §5.6
(“Frozen failure tests and success rule”). Machine-check all ten or mark a null explicitly
`not-testable/blocked` with the missing asset; do not choose a favorable subset.

The acquisition registry proposed in the earlier addendum was subsequently materialized in commit
`80e8af4` as `docs/experiments/PAPER_EXACT_CLAIM_REGISTRY_V1.yaml`. Preserve its hash as the
historical acquisition preregistration, but do not silently update it or treat its older
`iu28_no_length` primary as the current method-of-record. The current canonical comparison contract
is implemented on `codex/fair-paper-exact-comparisons-v1` (reviewed at commit `8e08c3e`) in:

- `docs/experiments/FAIR_PAPER_EXACT_COMPARISONS_V1.md` — frozen four-lane protocol;
- `spectral_utils/fair_comparisons/registry.py` — `population_registry_v1`,
  `method_registry_v1` and `comparison_record_v1`;
- `spectral_utils/fair_comparisons/prefix.py` — causal Prefix joins, strict unfinished-prefix
  budgets and trace-level warning cohorts;
- `scripts/build_fair_paper_exact_comparisons_v1.py` — materializes
  `POPULATION_REGISTRY.json`, `METHOD_REGISTRY.json`, per-lane `PER_QUESTION.jsonl`, calibration
  ledgers, audits and paired intervals.

The Prefix direct roster is now frozen as ordinary Unified-28 plus the selected Step-272
0.50/0.50 Global/Local Online incumbent, retaining `iu28_no_length` as a registered control on
identical trace realizations. This supersedes the older proposal that made `iu28_no_length` the
primary method. The technical failure tests already cover causal `length > budget` eligibility,
suffix/future exclusion, tokenwise/chunk replay, terminal identity, split/group isolation,
trace-level ever-warning calibration and grouped resampling. The final package should add a
machine-readable mapping from each of the ten §5.6 nulls to those tests/metrics or to an explicit
blocked reason; it must not invent an eleventh post-outcome claim.

### What to do with newly completed cluster work

Finish the current append-only upload and publish small manifests, indexes, status/summary files,
ordered-ID hashes and exact Drive paths. Do not start or resume computation. Codex should first
refresh the read-only Drive inventory and compare the new completion state against the frozen
eligibility gates. Complete, hash-verified rows can then be scored CPU-first; partial acquisitions
remain appendices and blocked acquisitions remain visible rather than approximated.
