# Four-Localization-Benchmark Cluster Campaign — Run Record

**Date:** 2026-08-11 · **HISTORY:** Step 242 · **Design:** `docs/experiments/FOUR_LOCALIZATION_BENCHMARKS_CLUSTER_HANDOFF.md`
**Raw data:** AIRCC `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/` and repo `dataset_cache/four_localization/` (2.3 GB)

Standing instructions for this campaign (Omri, 2026-08-10): skip the N=30 GPU pilots and submit at full size; very large generation limits; **no QwQ-32B-Preview**; include RefChecker with the strongest fully-open configuration and report a blocked cell rather than substitute a number.

> **These are four different prediction problems.** They do not share a label space or an official metric. Their scores must never be averaged into one leaderboard number.

---

## 1. Token / character-span localization — RAGTruth

**Competitor:** LettuceDetect-large (`KRLabsOrg/lettucedect-large-modernbert-en-v1`), supervised token classifier trained on RAGTruth's own train split. Fidelity level 1.

| Metric | Value |
|---|---|
| Example-level F1 | **0.792899** |
| Published (model card) | 0.7922 |
| **Gate delta** | **+0.0007 — PASSED** |
| Precision / Recall | 0.8046 / 0.7815 |
| TP / FP / FN / TN | 737 / 179 / 206 / 1578 |
| Overlap hits | 702 / 943 gold-hallucinated |
| Truncated rows | **0** |
| Runtime | 176 s (vs 9,218 s for the old CPU run) |

The `--max-length 8192` arm returned byte-identical output, so no RAGTruth test row exceeds 4096 tokens. Truncation retired by measurement.

**Why the number moved from 0.7590.** The previous run called `predict(context=[prompt], question="")`, which re-wraps an already-complete RAGTruth prompt in the library's own `"passage N: ..."` template. Official preprocessing (`preprocess_ragtruth.py::create_sample`) uses the whole prompt as one string, so `predict_prompt` is the matching call. Predicted character spans, confidences and per-token probabilities are now persisted — the old run saved only a span count and a boolean.

---

## 2a. Sentence localization — GASP on RAGTruth

**Competitor:** GASP-threshold (arXiv:2607.04223). **Fidelity level 2** — arXiv-only, no code release or response-ID list located, so our own seed and our own sentence splitter.

| Item | Value |
|---|---|
| Responses | 400 (exactly 200 hallucinated / 200 clean) |
| Items | 2,508 |
| Protocol | K=5 sentence-grouped chunks, 700-tok context cap, 200-tok answer cap, Summary + Data2txt only |
| Mean full-vocabulary JSD | 0.0698 (ln 2 = 0.6931 ceiling) |
| Max | 0.3392 |
| Hit context cap / answer cap | 1,119 / 705 items (the paper's protocol as specified) |
| Target to check against | 0.713 response AUC / 0.673 span AUC (Qwen2.5-1.5B specifically) |

**The JSD is now exact.** The existing arm approximates Eqs. (9)/(11) from top-50 log-probs plus one tail bucket. This driver computes the full-vocabulary divergence online in the forward pass and keeps only the per-token scalar (a dense `[T, V]` tensor is 122 MB per response per condition). Both are kept, so the cost of the approximation is now measurable on identical rows.

---

## 2b. Claim localization — RefChecker  ⚠️ 2 of 3 settings

**Competitor:** RefChecker's own `NLIChecker` (`ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`). Fidelity level 2.

| Setting | 3-way accuracy | Macro F1 | n |
|---|---|---|---|
| **Overall** | **0.6751** | **0.4440** | 7,414 |
| noisy_context (MS MARCO) | 0.7620 | 0.4616 | 3,420 |
| accurate_context (Dolly) | 0.6007 | 0.4336 | 3,994 |
| zero_context (NQ) | **BLOCKED** | — | — |

Per class (overall): Entailment F1 0.8091 (n=6129) · Neutral 0.2771 (n=804) · Contradiction 0.2457 (n=481). **The open checker is strong only on the majority supported class.**

- **BLOCKED reason:** `HTTP 403 Forbidden` on `https://storage.googleapis.com/natural_questions/v1.0/dev/...`. The bucket is not anonymously readable over plain HTTPS. Fix = an HF-based NQ source.
- **Scope limit:** this panel measures the **checking stage only**. Human labels are attached to Claude-2-extracted triplets, so a different extractor yields claims the gold does not cover. Claim extraction is out of scope by construction — this is *not* an end-to-end RefChecker reproduction.
- Our arm (teacher-forced evidence contrast over the identical claims) completed for all 7,414 claims. Fidelity level 3, adaptation.

---

## 3. Every-step correctness — PRMBench

**Competitor:** Qwen2.5-Math-PRM-7B, supervised. Fidelity level 1. 6,969 rows, **0 reward-count mismatches**, 134 s.

| Metric | Value |
|---|---|
| Pooled F1 | 0.9156 |
| correct_step_acc | **0.9543** |
| wrong_step_acc | **0.3047** |
| negative F1 | 0.3935 |
| first_error_acc | 0.3768 |

By category: sensitivity 0.9345 · soundness 0.9267 · **simplicity 0.8831** (weakest, matching the paper). **The supervised PRM massively over-accepts** — that asymmetry, not the 0.9156, is the real finding.

**Corpus facts pinned by known-answer test:** the paper's 83,456 step labels reproduce exactly, but the official loader drops 5 duplicate `multi_solutions` rows (85 steps), so **83,371** are evaluated. 100 rows annotate an error step past the end of their own trace; upstream treats those as inert, so rows are kept and the count reported.

**Our telemetry (Qwen3-8B):** 6,969 traces, 94,203 step spans, **0 unmapped steps**, 3 misaligned rows.

> **Hard constraint measured before scoring: 71.0% of PRMBench steps are shorter than 32 tokens** (median 24; 3.5% under 8). `compute_stft_features` needs 32 and `compute_spectral_features` needs 8, so most of the trace-level feature pool is structurally unavailable at PRMBench step granularity.

---

## 4. First-error localization — ProcessBench (full N = 3,400)

**Qwen2.5-Math-PRM-7B** (supervised ceiling), official F1 per subset:

| Subset | Error acc | Correct acc | F1 | (N=30 pilot) |
|---|---|---|---|---|
| gsm8k | 71.01 | 96.37 | **81.77** | 81.4 |
| math | 67.00 | 91.13 | **77.23** | 73.3 |
| olympiadbench | 54.77 | 85.25 | **66.69** | 61.8 |
| omnimath | 54.68 | 83.40 | **66.05** | 73.0 |

The pilot was noisy: omnimath moved −6.9 points and olympiadbench +4.9 at full N.

**Our no-training LLM-as-a-Judge control** (Qwen3-8B — **NOT uPRM**, which needs RL training): F1 15.27 / 10.38 / 5.13 / 7.79. Also much lower than its pilot (26.2 / 18.2 / 0.0 / 8.8). *This job was scaled against the handoff's own rule that it must not be; the number is legitimate but must never be labelled uPRM.*

**Qwen2.5-72B critic:** still running, 3 of 4 subsets, through a linear 6-wall resume chain.

---

## Background: HLE full run (not a localization panel)

2,158 items, Qwen2.5-72B-Instruct, job 176045. Accuracy **0.076** (164/2158) under the **placeholder ROUGE-L grader** — gate FAILED (`acc outside [0.2, 0.85]`). This is a floor, not the real accuracy. **Do not compute AUROC or fit any method on this cell until a real judge regrade lands.**

---

## Infrastructure bugs fixed this campaign

1. The 72B critic had no resume chain and would have died at its 8 h wall (Slurm does not auto-requeue a clean exit 85). It since did hit the wall — the chain caught it automatically.
2. Two chains fanned out onto the same output file (`afterany:177759` twice) and would have raced. Linearized.
3. `cluster/sync_code.sh` was uploading **6.2 GB** per sync — `*.pkl` does not match the `*.pkl.part-NN` LFS chunk files. Now **39 MB**.
4. RefChecker corpus build hit the NQ 403 above.

## Three official-protocol details read from source, not guessed (PRMBench `mr_eval`)

- The evaluated question is `modified_question`, **not** the dataset's own `question` field.
- **`labels[i] == 1` means the step is VALID** — the positive class of the official F1 is *correct* steps, so a risk score must be inverted.
- The all-steps-correct control class is **constructed** by the loader from `redundency` rows, not shipped, and is scored but not pooled.

---

## File inventory (2.3 GB, in `dataset_cache/four_localization/`)

| Directory | Files | Size |
|---|---|---|
| `prmbench_qwen3_8b_telemetry_full` | `prmbench_telemetry.pkl`, manifest | 1,069 MB |
| `hle_full` | `raw_hle_T0.0.pkl`, manifest | 911 MB |
| `gasp_ragtruth_exact_qwen15b_full` | `gasp_exact.pkl`, manifest | 162 MB |
| `refchecker_knowhalbench_open_full` | telemetry pkl, `nli_checker_predictions.json`, manifest | 79 MB |
| `ragtruth_lettuce_large_span_full` | `lettuce_spans_test.pkl`, manifest | 10.3 MB |
| `ragtruth_lettuce_large_span_ml8192` | `lettuce_spans_test.pkl`, manifest | 10.3 MB |
| `pb_critic_qwen72b_full` | 3 of 4 subset pkls (still running) | 14 MB |
| `pb_uprm_baseline_qwen3_8b_full` | 4 subset pkls, manifest | 7.5 MB |
| `pb_prm_qwen25math7b_full` | 4 subset pkls, manifest | 7.4 MB |
| `prmbench_qwen25math7b_full` | `prmbench_prm.pkl`, manifest | 2.0 MB |

**The .pkl files are not in this Drive folder.** They cannot be uploaded through the Drive API path available to Claude (file bytes must pass through the conversation). They are on the cluster and fetched into the repo at the path above; use Google Drive for Desktop, `rclone`, or a manual upload of `dataset_cache/four_localization/` to place them under `cluster_results/`.

---

## Open items

- **zero_context (NQ) blocked** — needs an HF-based Natural Questions source.
- **No scorer consumes any of this yet.** There is no consumer of the ProcessBench competitor pkls and no span / sentence / PRMBench metric harness. Until those exist the campaign has four competitor ceilings and zero numbers of our own beside them. This is now priority 0 in `Research_Directions.md`.
- A resume wall overwrites its predecessor's manifest timing (`gasp_exact` reports 4 s; the real runtime is job 177894's 2 m 39 s).
- `scripts/build_glossary.py` fails its coverage gate on four pre-existing selector families; GLOSSARY.md regenerated with `--allow-gaps`.
- Nothing committed — per the handoff's git rule, no commit or push until the final benchmark report is reviewed.
