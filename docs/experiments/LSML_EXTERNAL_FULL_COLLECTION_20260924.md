# Full external telemetry collection - 2026-09-24

The user approved the measured two GPU-hour budget. All three full collections
were submitted from committed snapshot411316927 after current-session preflight
PASS. No features, fusion weights or benchmark quality are evaluated in this stage.

| Dataset/backbone | Answers | GPU job | GPU time cap | CPU audit/archive job |
|---|---:|---:|---:|---:|
| Hard2Verify / Qwen3-8B | 200 | 266081 | 30min | 266094 |
| Socratic / Qwen3-8B | 2995 | 266082 | 30min | 266095 |
| Socratic / QwQ-32B | 2995 | 266083 | 60min | 266096 |

Each inference job uses one GPU, with no automatic requeue. CPU audit/archive jobs
use two cores,8GB RAM and no GPU, with a60min cap each and afterok dependencies.
Retries must remain within the approved total GPU budget. Initial state: running.

## Protocol and provenance

[Budget approval](../../results/lsml_external_generalization_v1/FULL_BUDGET_DECISION.json),
[current preflight](../../results/lsml_external_generalization_v1/FULL_PREFLIGHT.json),
[job paths and state](../../results/lsml_external_generalization_v1/FULL_JOBS.json).
The collection protocol, pinned checkpoints, source answers and production bf16
precision are unchanged from the validated smoke runs. Gate-v2 retains the exact
QwQ numerical exception documented in the [smoke report](LSML_EXTERNAL_COLLECTION_STATUS_20260924.md).
The archive SHA256 is recorded in FULL_JOBS.json; its collector and transitive
imports match the extracted package that passed the five-example CPU smoke.

## Storage and verification

The label-blind auditor validates every record, checks exact expected answer IDs,
finite probabilities/entropy, top50 integrity, actual-token probability even
outside top50, step spans and historical top15 entropy replay. An audit failure
prevents archival success. Full-set step and token totals must match the input
and tokenization manifests before completion is reported. Empty steps are retained.

The CPU archive script is committed atad796cfc8. It saves a deterministic gzip
tar of each completed run, compact manifests and SHA256 values under private
AIRCC storage, copies directly to
`gdrive:hallucination_detection/cluster_results/lsml_external_generalization_v1/full_archives/<run>/`,
then requires rclone checksum verification. Only compact manifests return locally.
Hard2Verify decrypted text and token arrays remain outside Git/public artifacts.
Restore by downloading telemetry.tar.gz to private storage and extracting it.
The packed archive contains original JSON records, tokenization, timing and audit.

## Acceptance and limits

Collection completion requires all200+2995+2995 dataset/backbone records, passing
raw audits and verified archives. This is reusable telemetry, not an evaluation
of L-SML generalization. Source calibration, method freeze, external overlap,
comparators, official metrics and paired uncertainty remain subsequent work.

## ????? ??? ??????

????? ????? ????? ????? ?????? ????????. ??????? ?????? ?-AIRCC ???????? ??????
?-Drive ???? ????? ??????. ???? ??? ?????? ???? ????; ??? ????? ????? ????? ?????.
