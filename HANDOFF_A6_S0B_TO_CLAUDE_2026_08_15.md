# Handoff: A6-S0b from Codex to Claude

> **CLOSED 2026-08-20 — DO NOT EXECUTE.**
> The PTNI/A6 direction was rejected by Omri on scope grounds (the thesis is
> not pursuing a self-supervised identification mechanism). Registered verdict
> `CLOSE_A6_S0B_DIRECTION_REJECTED`. The gate never ran: Slurm job `196764`
> was cancelled while PENDING, `Elapsed 00:00:00`, and no artifact exists on
> the cluster, on Drive, or on disk. This document is retained as a record of
> the frozen design only. See HISTORY.md Step 282 [A6/PTNI] and the amendment
> at the top of
> `docs/research_notes/research_status_consolidated_2026-08-19.md`.

**Date:** 2026-08-15

**Repository:** `/Users/osegev/Desktop/hallucination_detection`

**Starting HEAD for this session:** `f6db0373c70c9ccf81a72bbf9675f0119e120c85`

**Scope:** continue the mandatory A6-S0b shortcut/matching audit, then S1 only if S0b passes.
**Important:** this handoff describes development code that has not yet been frozen or executed as a sealed S0b boundary.

## 1. User objective and working constraints

The project objective remains a material, verified improvement over IU-PCR for hallucination detection under these constraints:

- not supervised by human or benchmark correctness labels;
- gray-box;
- one model pass at deployment;
- no hand-picked prior feature groups or target knowledge;
- explicitly addresses failure modes found in A0--A5;
- must ultimately improve held natural hallucination detection, not merely a simulator or mechanical contrast.

The method may be fully unsupervised (preferred) or mechanically self-supervised. Human/benchmark labels may not fit, orient, tune, or select the method.

The user explicitly requested a faster workflow: automated tests during development and **one independent review only immediately before freeze/run**, not repeated reviewer loops. Only Claude on the other computer should operate the AIRCC cluster. Codex did not access the cluster in this session.

## 2. Authoritative project state before this handoff

- A0--A4 are closed.
- A5 is closed at S1a. On 98 usable sealed repetitions, candidate-minus-IU AUROC was `-0.0384836068`, with exact 20,000-draw CI `[-0.0474950732, -0.0296585954]`. Final target preference was 62/98 and correction target preference was 25/98.
- A6-S0a completed and independently verified `PASS_S0A`:
  - 1,800 reciprocal quartet groups;
  - 6,000 prompt-only natural-manifest rows;
  - 7,200 inner-fold assignments;
  - 36 null cells;
  - 7,800 immutable checkpoints;
  - boundary SHA-256 `698261d467a3f0a394ef244dafcac67d1cf8a69a9cf2de8888f0ff54678c545e`;
  - aggregate SHA-256 `2a11b37c4fd649490675e8da4d826084c137a2a072c77ab2fdd5efcad8e8685a`.
- No A6 response telemetry, natural response, correctness sidecar, PopQA content/target, or sealed S1 seed has been opened.
- The current mandatory stage is S0b. It uses only the public S0a mechanical quartets and prompt-only Pythia NLL. It is not a detector-performance experiment.

Canonical documents:

- `docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A6_V1.md`
- `docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A6_S0_S1_EXECUTION_V1.md`
- `PROGRESS.md` Step 268

## 3. New S0b implementation created in this session

These files were created and were untracked before the handoff commit:

- `spectral_utils/a6_s0b.py`
- `spectral_utils/a6_s0b_input.py`
- `scripts/automatic_group_free_phase_a6_s0b.py`
- `scripts/download_a6_s0b_pythia.py`
- `scripts/test_a6_s0b.py`
- `scripts/test_a6_s0b_input.py`
- `scripts/test_automatic_group_free_phase_a6_s0b.py`

### 3.1 `spectral_utils/a6_s0b.py`

Implemented:

- the exact 43,200-row crossed shortcut table over the 1,800 S0a groups;
- frozen 21 continuous and 8 categorical columns;
- Qwen-only vocabulary fitting and Llama reuse;
- sparse CSR classifier designs;
- exact weighted logistic objective/gradient, zero initialization, frozen L-BFGS-B settings, and fail-closed `gradient_inf <= 1e-8` usability gate;
- fold-local OOF bundles for four frozen ridges;
- exact AUROC with half credit for ties;
- 19 shortcut gate names and five-fold macro statistics;
- grouped 20,000-draw bootstrap with one multiplicity per reciprocal group;
- prompt/response marginal 50/50 audits;
- matching vectors, q75 caliper, frozen group metadata and 60 partitions;
- deterministic Control-2 Fisher-Yates derangements;
- deterministic exact Hungarian Control-3 matching with component decomposition preserving the global integer objective;
- prompt-only Pythia mean next-token NLL using CPU model output and float64 `logsumexp` reduction.

### 3.2 `spectral_utils/a6_s0b_input.py`

Implemented the immutable Pythia input contract:

- repo `EleutherAI/pythia-410m-deduped`;
- exact revision `c4fc8d586d62df497f1f9b69d66d3ca419992d3e`;
- complete official eight-path tree validation;
- selected files only:
  - `config.json` (570 bytes);
  - `model.safetensors` (911,373,632 bytes; LFS SHA-256 `e7ae132489f63d5d86009a8178a75c7d5d195410d067fca01a3160623e370fae`);
  - `special_tokens_map.json` (99 bytes);
  - `tokenizer.json` (2,113,710 bytes);
  - `tokenizer_config.json` (396 bytes);
- exact Git-blob or LFS-pointer/payload verification before Torch/Transformers imports.

The duplicate `pytorch_model.bin` is deliberately excluded, so the required download is about 913.5 MB rather than about 1.8 GB.

### 3.3 `scripts/automatic_group_free_phase_a6_s0b.py`

Implemented so far:

- stdlib-first source/runtime/prior/input boundary;
- exact source closure and runtime/thread environment recording;
- authenticated Pythia materialization before model import;
- CPU/float32, one-thread, deterministic offline Pythia loading;
- 14,400 append-only prompt-NLL checkpoints and completion manifest;
- shortcut table, OOF, bootstrap, matching graph, 400 control schedules, closures, aggregate and completion;
- fail-closed output allowlist and symlink rejection;
- numerical non-convergence is emitted as `CLOSE_S0B_NUMERICAL_NONCONVERGENCE`, not an uncaught crash;
- a new no-write `verify` path that recomputes every Pythia NLL and replays analysis artifacts;
- replay-mode canonical byte comparison through `_emit_json`;
- ordinary resume loaders for completed bootstrap and control checkpoints, while authoritative replay still regenerates them.

The latest verifier/resume additions were written at the end of the session. They compiled and the seven boundary/input tests passed, but they have **not** yet received dedicated adversarial tests for every resume/replay branch. Claude should treat them as development code, not reviewed/frozen code.

## 4. Tools and commands used

All work was local and read/write only inside the repository or ignored `local_cache/`.

- repository inspection: `git status`, `git log`, `rg`, `sed`, `wc`, `find`, `stat`, `du`;
- source edits: Codex `apply_patch` only;
- execution: `.venv/bin/python`, `py_compile`, `unittest`;
- dependency/runtime: existing `.venv` (`huggingface_hub 0.36.2`);
- network attempt: the approved local command `.venv/bin/python scripts/download_a6_s0b_pythia.py`;
- no raw SSH, no AIRCC command, no cluster job, no Drive mutation, no telemetry generation.

Commands that were green before handoff:

```bash
.venv/bin/python -m py_compile \
  spectral_utils/a6_s0b.py spectral_utils/a6_s0b_input.py \
  scripts/automatic_group_free_phase_a6_s0b.py \
  scripts/download_a6_s0b_pythia.py \
  scripts/test_a6_s0b.py scripts/test_a6_s0b_input.py \
  scripts/test_automatic_group_free_phase_a6_s0b.py

.venv/bin/python -m unittest \
  scripts.test_a6_s0b_input \
  scripts.test_automatic_group_free_phase_a6_s0b -v
```

Result: 7/7 PASS.

After the final verifier/resume patch, the boundary/input suite passed 7/7 and the full core suite passed 14/14 in 97.581 seconds: 21/21 total. These tests cover the core equations and current boundary basics, but not yet every new replay/resume branch listed in Section 7.

## 5. Development preflight and what it means

A full real-S0a shortcut-table preflight was performed with fake prompt NLL values; it was not a sealed experimental result.

One Qwen outer fold, all four frozen ridges, produced:

```text
built (23040, 2286) 599040
CLOSE 0.01 gradient=1.672774314026397e-07
CLOSE 0.1  gradient=2.1804521413546532e-07
CLOSE 1.0  gradient=8.500750706607815e-08
CLOSE 10.0 gradient=7.931774066100567e-08
```

All fail the preregistered `gradient_inf <= 1e-8` condition. A prior fake-NLL run similarly stopped at gradient `2.8905953484406523e-07`.

Interpretation:

- this suggests the sealed S0b run may close quickly as numerical non-convergence;
- it must **not** be reported as the S0b verdict because the Pythia NLL was fake and only one fold was exercised;
- do not relax the optimizer tolerance, change solver, increase iterations, or add a rescue path after seeing this preflight;
- run the exact frozen input/boundary and accept the registered close if it reproduces.

## 6. Pythia acquisition attempted and failed

The exact Pythia snapshot is not present under the inspected Google Drive HF cache paths. A local downloader was added to obtain only the selected immutable files and authenticate them.

Two approved attempts failed before any payload was downloaded:

1. `urllib.request.urlopen` failed with:

   ```text
   ssl.SSLCertVerificationError: unable to get local issuer certificate
   ```

2. The downloader was changed to `ssl.create_default_context(cafile=certifi.where())`; the same TLS failure remained.

The destination `local_cache/a6_s0b_pythia_c4fc8d5/` is currently empty (`0B`). No model bytes were partially accepted.

Recommended Claude action:

1. On the other machine, use the exact repo/revision and selected allowlist above.
2. Prefer the machine's working Hugging Face or `curl` TLS stack; do not disable TLS verification.
3. Save the exact official API response bytes from the contract URL.
4. Run `spectral_utils.a6_s0b_input.validate_official_tree` and `verify_selected_bytes` over every selected file.
5. If local CPU runtime is unreasonable, prepare the exact frozen boundary locally, sync the committed code and authenticated snapshot to AIRCC through the established Claude workflow, and run there. Codex did not claim or attempt cluster access.

## 7. Required continuation sequence for Claude

Do not redesign S0b. Continue in this order:

1. Read `CLAUDE.md`, current `PROGRESS.md`, this handoff, the parent A6 protocol, and the S0/S1 execution contract.
2. Inspect the handoff commit and verify that only the intended S0b files, this handoff, and the PTNI-Guided note update are included.
3. Finish tests for the latest runner changes:
   - replay mismatch never repairs or creates a file;
   - authoritative `verify` cannot be downgraded to hash-only PASS;
   - bootstrap resume validates all stored draws/summary and skips recomputation;
   - control resume validates canonical partitions, bijection, strata/eligible edges, seeds and hashes;
   - terminal artifact exclusivity and exact stage-specific layout;
   - unexpected exceptions remain implementation-invalid, not scientific closure.
4. Run the full S0b suite and relevant A6 regressions.
5. Acquire and authenticate exact Pythia bytes without disabling TLS.
6. Benchmark a small exact-Pythia local sample. If 14,400 CPU prompts are too slow, use Claude's cluster workflow; keep checkpoint/resume and immutable boundary semantics.
7. Obtain exactly one independent no-edit review of the stable source boundary.
8. Commit/freeze the reviewed source. Prepare the canonical S0b boundary from that exact commit.
9. Run `run-pythia`, `run-analysis`, and authoritative `verify`.
10. If S0b closes, record the registered verdict and proceed to A7/next preregistered route. If S0b passes, implement/freeze S1 exactly as preregistered before opening any sealed S1 seed.

## 8. Requested parallel subagent task

While the main Claude agent continues the exact S0b execution path, spawn one separate read-only subagent with this task:

> Review the complete A0--A5 negative results, the frozen A6/PTNI design, and the queued PTNI-Guided Neutral Residual Mode proposal. Build a concise mechanism-level comparison of every materially distinct unsupervised or mechanically self-supervised alternative already tried. Identify which failure each route exposed (identifiability, length, redundancy, instability, nuisance capture, non-convergence, transfer, or deployment incompatibility), which ideas are genuinely closed, and which one or two next routes remain scientifically non-redundant. Recommend the next route after A6 without changing, rescuing, or delaying the frozen S0b/S1 execution. Do not inspect natural labels, PopQA targets, or sealed A6 results; do not edit files. Return an evidence-linked recommendation to the main agent.

The main agent should decide; the subagent only surveys evidence and alternatives.

## 9. PTNI-Guided NRM proposal included in this handoff

Read and preserve:

`docs/research_notes/ptni_guided_nrm_research_proposal_2026-08-14.md`

It is queued only after the frozen A6/PTNI program reaches its registered outcome. It may run only if PTNI establishes a valid target direction and leaves a registered stability, redundancy, or nuisance-transfer limitation. It cannot rescue, modify, or delay S0b/S1. It also must respect the corrected goal wording: unsupervised is preferred; mechanically self-supervised is allowed; human/benchmark-supervised selection is forbidden.

## 10. Files that must remain excluded

The following unrelated/unreviewed paths were present and must not be staged by this handoff commit:

- `.beagle-retrieval-diag/`
- `dataset_cache/four_localization/`
- `dataset_cache/repgrid/pb_llama31_8b/`
- `docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A6_TOKENIZER_RESTORE_V1.md`
- `docs/meetings/Advisor_Update_Aug2026_NRM_PTNI_short.md`
- `docs/reports/`
- `docs/research_notes/nrm_harp_ptni_detailed_tutorial_2026-08-15.md`
- `papers/2026.03.23.713692v2.full.pdf`
- `papers/Learning_From_Crowdsourced_Noisy_Labels_A_signal_processing_perspective.pdf`
- `results/automatic_group_free_phase_a6_tokenizers_v1/`
- `results/automatic_group_free_phase_a6_tokenizers_v2/`

## 11. Honest handoff status

No A6 detector performance number exists yet. S0a proved only the mechanical construction. S0b has not been frozen or run. The current code is meaningful progress but unfinished: the exact Pythia input remains inaccessible on this machine because of TLS trust, and the newest verifier/resume paths need targeted tests plus the one final review. Claude should continue from this commit rather than restarting the S0b implementation or reopening the scientific design.
