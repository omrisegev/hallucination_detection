# Full external telemetry collection - 2026-09-24

All three full collections, raw-data audits and Drive backups are complete.
The approved two GPU-hour budget used **22 minutes 56 seconds of GPU allocation**.
No external benchmark quality was evaluated. The data are ready for CPU feature
extraction after the remaining source calibration and method-freeze work.

| Dataset/backbone | Answers | Steps | Answer tokens | GPU elapsed | GPU job | Audit/archive job |
|---|---:|---:|---:|---:|---:|---:|
| Hard2Verify / Qwen3-8B | 200/200 | 1,860 | 389,770 | 2m33s | 266081 | 266094 |
| Socratic / Qwen3-8B | 2,995/2,995 | 26,055 | 2,505,065 | 8m42s | 266082 | 266095 |
| Socratic / QwQ-32B | 2,995/2,995 | 26,055 | 2,505,065 | 11m41s | 266083 | 266096 |

Every job exited successfully. The table counts dataset/backbone records: 6,190
records represent 3,195 distinct benchmark answers across the two datasets.
There are 53,970 step observations and 5,399,900 scored answer tokens across models.
No missing or extra answer IDs and no truncation were found. The three empty
Socratic steps remain present for each backbone; later scoring must account for them.

## Verification

The AIRCC auditor read every raw record and checked aligned lengths, valid spans,
finite probabilities and entropy, sorted/unique top50 IDs, probability mass,
actual-token log-probabilities and replay of historical top15 entropy. Actual
log-probabilities are retained for tokens outside top50: 1,920 Hard2Verify/Qwen3,
36,374 Socratic/Qwen3 and 39,224 Socratic/QwQ observations.

A separate compact-manifest check verified hashes, pinned models and answer-file
identities, complete input-ID coverage, unique measurements, step/token totals,
empty-step counts and the absence of quality evaluation. Evidence:
[FULL_COLLECTION_VERIFICATION.json](../../results/lsml_external_generalization_v1/FULL_COLLECTION_VERIFICATION.json).
This does not claim a second independent raw-array replay; the raw audit ran on AIRCC.

The collection protocol, pinned checkpoints, source answers and bf16 precision
are unchanged from the validated smoke runs. Gate-v2 checks causal invariance and
target offsets. QwQ reproduces the exact, archived numerical exception documented
in the [smoke report](LSML_EXTERNAL_COLLECTION_STATUS_20260924.md); its original
prefix-only check is not retroactively declared passed. Historical generated-cache
Gate B and cross-benchmark scientific evaluation remain separate fidelity items.

## Cost and provenance

The three one-GPU jobs had 30/30/60-minute limits and no automatic requeue.
Observed full-collection GPU allocation was 1,376 seconds (0.38222 GPU-hours),
including startup and serialization/I/O time within each allocation. Scorer-only
time summed to 417.20 seconds; model-load time summed to 49.40 seconds. These
components are different from end-to-end job time. All three ran concurrently.

CPU audit/archive jobs used two cores each, 8GB RAM and no GPUs. Their elapsed
times were 55/204/205 seconds, totaling 0.25778 allocated CPU core-hours. Peak
GPU allocations were 20.19/17.64/66.79 GB respectively. No full-run retries were
needed. Earlier smoke/diagnostic failures remain in the separate cost ledger.

[Budget approval](../../results/lsml_external_generalization_v1/FULL_BUDGET_DECISION.json),
[preflight](../../results/lsml_external_generalization_v1/FULL_PREFLIGHT.json),
[job paths and costs](../../results/lsml_external_generalization_v1/FULL_JOBS.json),
[exact submission script](../../results/lsml_external_generalization_v1/FULL_SUBMISSION.sh).
Inference snapshot: `411316927`; archive script: `ad796cfc8`.
The collector and transitive imports matched the extracted package that passed
five-example CPU collection, serialization, interruption/resume and the deliberately
shifted-target rejection check. Eight external and six runtime tests also passed.

## Storage and restoration

Three compressed telemetry archives total **1,918,336,109 bytes (1.92 GB)**.
Each Drive archive passed rclone checksum verification with zero differences
and six matching files. Restore paths, sizes and SHA256 values:
[FULL_ARCHIVES.json](../../results/lsml_external_generalization_v1/FULL_ARCHIVES.json).

Private Drive prefix:
`gdrive:hallucination_detection/cluster_results/lsml_external_generalization_v1/full_archives/<run>/`.
Download `telemetry.tar.gz` to private storage and extract it. It contains the
original per-answer records plus timing, tokenization, alignment and raw audit.
Verify its SHA256 against the manifest before using it. Only compact manifests
were downloaded locally; raw arrays and decrypted Hard2Verify text stay out of Git.

Exact normalized answer-only input files are also backed up and checksum-verified
under the sibling `inputs_20260924/{hard2verify,socratic}/answers.json` paths.
They preserve original questions, ordered steps and source-group metadata. Evaluator
annotations remain separate and can be regenerated from the pinned source releases.
Do not mix annotation fields into feature extraction or fitting inputs.

## Scientific status and next work

This completes the approved telemetry collection stage. It does not establish
L-SML generalization or comparative benchmark performance. Source fit/calibration
separation and method freeze, development/external source overlap auditing,
CPU feature replay, comparator inference, official evaluator parity, prediction
sealing and registered paired analysis remain subsequent work. No extra GPU
training, method sweep or comparator inference was launched here.

## ????? ??? ??????

?????? ???? ????? ????? ??? ????? ????????. ?? 6,190 ??????? ??????, ??? ?????
??????, ?????? ???? ?????? ?-Drive ?? ????? checksum. ????? 22 ???? ?-56 ?????
GPU ???? ???? ?? ??????. ??????? ?????? ?????? ???'??? ?-CPU; ????? ?? ?????
????? ????? ?? ????'????? ??????.
