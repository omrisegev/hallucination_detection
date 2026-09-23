# External telemetry collection: 2026-09-24

The telemetry pipeline runs on AIRCC. Two timing cells are complete and verified;
QwQ collection is pending a timing retry after a completed numerical diagnostic. These are engineering
results, not benchmark quality results. No external PRMScore or Balanced F1 has
been calculated, and no fusion variant has been selected using these datasets.

## Verified collection

| Cell | Complete timing answers | Captured answer tokens | Steps | GPU job elapsed | Job |
|---|---:|---:|---:|---:|---:|
| Hard2Verify / Qwen3-8B | 12 | 29,277 | 122 | 100 s | 265843 |
| Socratic / Qwen3-8B | 12 | 11,180 | 103 | 97 s | 265851 |

All 200 Hard2Verify and 2,995 Socratic answers were tokenized, without truncation,
before the length-stratified timing samples were chosen. Longest full inputs are
11,723 and 3,339 tokens, respectively. The full releases have 1,860 and 26,055
steps. Three empty Socratic steps retain explicit zero-length spans; later scoring
must account for them. Eighty Socratic rows contain inert out-of-range annotation
indices; no labels were repaired or used by the collector.

The raw-data audit checks token/step lengths, sorted and unique top50 entries,
probability mass, actual-token probabilities, finite full entropy/logsumexp and
exact reconstruction of the historical top15 entropy. It found 152 and 182
provided tokens outside the top50, respectively; their actual log-probabilities
are present. This demonstrates why top50 alone would have been insufficient.
The fixed-prefix GPU alignment check passed for both Qwen3 jobs.

Raw data remain on AIRCC and private Google Drive. Only compact metadata were
downloaded. Restore locations and checksum-check outcomes are in
[ARCHIVES.json](../../results/lsml_external_generalization_v1/ARCHIVES.json).
The [timing evidence](../../results/lsml_external_generalization_v1/timing/)
contains no decrypted Hard2Verify text or per-token arrays.

## Costs and budget decision

Measured timing projects roughly 34 seconds of GPU computation/model load for
full Hard2Verify and 219 seconds for full Socratic on Qwen3, including a 20%
retry factor on scoring. This is a **compute-only projection**, not end-to-end
wall time: container setup, JSON serialization, filesystem/Drive writes and
queue delays are excluded. Projected raw JSON storage is approximately 0.56 GB
and 3.38 GB. Feature extraction, fusion, bootstrap and comparator generation
are also excluded. Evidence:
[QWEN3_COMPUTE_ESTIMATE.json](../../results/lsml_external_generalization_v1/QWEN3_COMPUTE_ESTIMATE.json).

The requested full-collection budget is capped at one GPU-hour total, with two
30-minute allocations, to allow substantial operational margin. This decision
has been requested from the user and is not yet recorded as approved. It covers
Qwen3 telemetry only; QwQ and critic/PRM inference remain separate.
The [cost ledger](../../results/lsml_external_generalization_v1/COST_LEDGER.json)
includes failed startup attempts, rather than hiding their allocation cost.

## QwQ and operational repairs

The pinned QwQ-32B checkpoint was prepared on a CPU node in job265848 (372 seconds,
four allocated CPU cores). Timing job265853 stopped before collecting telemetry:
the first actual-token log-probability differed by0.374979 nat between full-trace
and serial-prefix bf16 layouts. A mismatch is not proof of an indexing bug, but
it is not a passed validation either. Diagnostic job265860 completed in103 seconds:
at positions0/8/15 the fixed-shape causal errors were zero, and the same weights
under fp32 had zero prefix discrepancies. This evidence supports a numerical
explanation for this example, not validation of the whole corpus.

Collector956a1aa19 adds direct full-vocabulary target-offset and future-token
checks at fixed shape (maximum1e-5). Its .05 prefix limit remains, with one
explicit exception bound to the exact diagnostic SHA256, checkpoint, answer ID
and reproduced differences. No global bf16 tolerance was increased. Production
precision is unchanged. Job265865 then failed because the archive omitted an indirect helper import.
The corrected full archive passed the five-example CPU test after extraction;
replacement timing job265869 is submitted, not yet complete.
The CPU smoke passed five forwards/save/resume checks and rejected an injected
off-by-one target alignment error.

Other startup repairs: use current cycle3/owner_940 account settings; install
small pinned offline wheels because compute-container package DNS failed;
preserve NGC PyTorch and NumPy; avoid host ensurepip; invoke remote submissions
without a trailing Windows carriage return. Each stopped attempt is recorded.

Before collection, 9,971,019,380 bytes of unneeded local cache copies were reclaimed
after matching Drive size and checksum. Tracked files were reduced to exact HEAD
LFS pointers; five untracked copies were deleted. Scientific source caches needed
for later bank11 work were retained. The cleanup ledger records archive paths.

## Remaining scientific work

Finish QwQ validation/timing, obtain budget decisions, collect full telemetry and
verify archives. Then complete source fit/calibration separation and freeze both
L-SML variants before external quality evaluation. Complete overlap auditing,
CT7/comparator implementation and official evaluator parity, seal predictions,
and run registered paired analysis. Existing helper code is not a completed
external evaluation. MedPRMBench remains deferred.

## פירוש קצר בעברית

האיסוף ב־AIRCC עובד: הושלמו ונבדקו 12 תשובות מכל benchmark עם Qwen3, והמידע
גובה ישירות ל־Drive. נשמרו כל נתוני ההסתברויות הדרושים לחילוץ הפיצ'רים בהמשך,
כולל הסתברות הטוקן האמיתי גם כשהוא מחוץ ל־Top50. עדיין לא נמדדה איכות השיטה.
הריצה המלאה ממתינה להחלטת התקציב שביקשנו; QwQ נבדק בנפרד בעקבות אי־התאמה
נומרית שנמצאה בבדיקת התקינות. לא הסרנו את הבדיקה כדי להעביר את הריצה.
