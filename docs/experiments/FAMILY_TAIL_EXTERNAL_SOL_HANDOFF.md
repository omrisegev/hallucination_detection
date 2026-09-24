# Astra to SOL execution handoff — 2026-09-24

The user explicitly requested continuing this active task with **SOL**, transferring
the necessary Astra memory. This handoff is operational state, not a result report.

## User objective (finish it; do not restart)

Evaluate the frozen `F15_tailtie_lsml` handoff from branch/worktree
`codex/family15-tail20-transfer-v1` on Hard2Verify and both Socratic backbones.
Compare all10 registered alternatives (including equal controls, bank11 and CT7),
produce organized standalone PNG/PDF plots, show both F1 components of PRMScore and
class precision/recall, compare against published literature with access caveats,
and obtain independent subagent verification before conclusions. Add reporting
without contradicting the source handoff. Clarify substantive recipe conflicts
before running; none was found. No new GPU/LLM inference is needed or authorized
by this CPU follow-up. Prior authorization to commit/push our own work persists.

## Live execution — resume this handle

- Workspace `C:/Users/omris/TAU/hallucination_detection`.
- Root branch `codex/lsml-external-generalization-v1`, HEAD `4687660f8`.
- **External prediction run ACTIVE: tool exec session `34772`.**
- Command: `C:/Users/DELL/AppData/Local/Programs/Python/Python313/python.exe -B C:/Users/omris/TAU/hallucination_detection/scripts/run_family_external.py --workers 4`.
- Last observed output: `external predictions 51 / 6190 hard2verify_qwen3_8b seconds 70`.
- Poll with `tools.write_stdin({session_id:34772,chars:'',yield_time_ms:50000,...})`.
  Do not launch another writer or rerun source parity. If session access does not
  transfer, inspect record counts/WRITER.lock and process liveness before action.
- No new external aggregate quality has been evaluated yet.

The full source raw-telemetry extraction gate has **already PASSED** all13,769
answers/145,597steps/all48channels, max error `6.661338147750939e-14` versus1e-6
tolerance; wall3557.808s. Evidence `results/family_tail_external_v1/source_full_v1/GATE.json`.
Source extraction code hashes/mtimes remained unchanged from the explicit mid-run
observation to completion; `SOURCE_CODE_COMPLETION.json` PASS. Do not call that
observation prelaunch. Both model+analysis freeze and source gate are enforced by
the external runner. Four feature tests, six metric tests (340 exhaustive official
synthetic fixtures) and two portable API tests passed.

## Environment and repository boundaries

- PowerShell tools ignore cwd/workdir: use absolute paths and `git -C` everywhere,
  `login:false`. Python is the exact executable above; `-B` avoids pycache writes.
- `functions.exec` can yield an orchestration cell; use `functions.wait` only after
  that yields. The numeric session34772 belongs to `write_stdin`, not `wait`.
- Long tool waits <=50seconds permit Hebrew progress updates about once/minute.
- Filesystem writes only within root. Source worktree is read-only and clean;
  source commit `df8371f3c56574b69f69db14299a460ebc225bc1`.
- Canonical CLAUDE.md, PROGRESS.md, operating contract and LESSONS standing rules
  were read. Reorient from this handoff rather than the older PROGRESS headline.
- HISTORY.md, PROGRESS.md, LESSONS.md contain OTHER agent's uncommitted group-confidence
  changes. Preserve them. Stage explicit paths; never git-add-all or commit their
  unrelated work. Root LESSONS also has two OUR entries at its tail (positive test
  count; actual precision/mid-run hash timing). Stage only our additions if using
  shared files. No destructive git operations.
- Use `require_escalated` for git index/commit/push/network when needed. Push our
  dedicated branch, never master. Source proof files were copied and verified
  byte-for-byte into root; no source branch mutation.
- Hebrew via PowerShell here-strings can corrupt encoding. Use apply_patch or
  ASCII JSON Unicode escapes decoded by Python for UTF-8 writes.
- Roughly23GB free before this run. No cleanup needed.
- New continuation/review subagents should use `model:"gpt-6-sol"` per user.

## Frozen recipe and interpretation

Read `docs/experiments/FAMILY15_TAIL20_AGENT_HANDOFF_HE.md` (exact source copy),
`docs/experiments/FAMILY_TAIL_EXTERNAL_V1.md`, EXECUTION_PLAN and IMPLEMENTATION_FREEZE.
The byte-exact portable scorer is `spectral_utils/family_tail_transfer.py`; source
lock is `results/family_tail_transfer_v1/TRANSFER_LOCK_V1.json`, SHA
`65b35336fcc2f66b7843ec040d3bdafbbaa03bb44ae5f61f1d00335abfaea5cf`.

28 source-oriented step channels -> within-answer z ->15 equal family means ->
family answer-z. Source fit on within-answer centered top-ceil20% fractional-tie
marks, no extra pooled z. Frozen weights apply to CONTINUOUS family features,
then final answer-z (sd guard1e-8). Frozen source folds0-3 fit/fold4 q80 calibration;
candidate threshold `0.878464199608311`. No target fitting or threshold tuning.
Empty steps excluded before fusion and restored null score/pred-valid0.

Actual historical CT7/hist streams are float64; bank11 tokens float32. Initial
review repeated a stale float32 comment; it was corrected to actual bytes before
full gate. Chosen CT7 feature is ordinary Top10, no first-step replacement; entropy
exceedance is robust-normalized fraction>=1, not Top10. All48 source channels now
fully replay from raw telemetry. No handoff recipe change.

K=2: two CUSUM families vs13 others. Outer absolute spectral coefficients equal;
final L1 mass is approximately29.9246% CUSUM /70.0754% others due within-group weights
and normalization. Do not claim learned relative group reliability. Source feature
selection/signs used development labels. This is source-fitted offline transfer,
not answer-local target fitting or an entirely label-free design. The external
benchmarks were already evaluated previously: **exploratory follow-up, not untouched
confirmation**. Equal fusion is control-only, not a proposed final method.

Ten arms: B11_lsml,F15_tailtie_lsml,F15_cov_lsml,K28_cov_lsml,K28_equal,F15_equal,
B11_equal,B11_partition_equal,A48o_equal,ct7. Three B11 arms must replay old scores
within1e-6 and exact decisions; CT7 reuses previous sealed same-telemetry predictions.

Six contrasts ×3cells =18,100,000 paired source-question bootstrap draws,
seed20260924,Bonferroni across18: tail-B11,tail-F15equal,tail-F15continuous,
F15continuous-K28continuous,F15equal-K28equal,K28equal-A48equal. No posthoc selection.

## Data and existing results

Current result root `results/family_tail_external_v1/`.
Cells: Hard2Verify/Qwen3-8B200answers1860steps79groups;
Socratic/Qwen3-8B2995answers26055steps1765groups;
Socratic/QwQ32B same2995answers. Total6190/53970steps/5,399,900tokens.
Three empty Socratic steps per backbone, kept in official evaluation.80 Soc rows
have102 out-of-range error indices; preserve author evaluator behavior.

Raw external shards: `scratch/external_generalization_private/evaluation_archives/<cell>/records/*.record.json`.
Gold separate: `scratch/external_generalization_private/inputs/evaluator_only/{hard2verify,socratic}.json`.
Answer-only inputs `<inputs>/<bench>/answers.json`.
Pinned official code `<private>/sources/hard2verify/utils.py` and
`<private>/sources/prmeval_classified_task.py`. Hard2 endpoint is harmonic mean of
class recalls; Socratic PRMScore mean of class F1. Socratic exact degenerate -1
sentinel semantics are implemented and exhaustively tested. Never average the
benchmarks' different metrics. Source-overlap exact-hash/component closure removes
442 Socanswers, leaving2553/1514groups/22179steps. Not semantic/pretraining exclusion.

Prior external root `results/lsml_external_generalization_v1/evaluation/`.
Previous frozenB11 primary scores (%)43.66954/63.22130/64.23823;
equal40.88224/60.79124/61.50338; partition39.75761/61.26967/62.10976;
CT737.75100/58.75289/60.16558. These must replay. Prior B11 flags41/42 fully-correct
Hard2answers. Prior answer-local fusion lost to matched equal on Socratic. Keep
this context; do not silently change the registered10-arm study to add a sweep.

Source pool (needed by independent coverage audit and archive):
`C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/2d14a8c9-8b3f-489b-b26f-812b4b84a8b3/scratchpad`.
`pool_z.npy` SHA `d9abccfb561f459d2f3a6c5b4346a1f6766df7fdbaddb230338aa40349077a16`;
`pool_names.json` alongside. Source JOINED roster in
`results/localization_full_benchmark_v3/evaluation/JOINED.json`.
All raw source PKLs already locally present; full gate records their SHA hashes.

## Implemented scripts — do not reimplement

- `run_family_external.py`: active4workers, atomic resumable RecordStores. Writes
  nine computed arms+CT7, all48 raw step features, masks, telemetry/reference hashes
  and per-record CPU timings. CPU_EXECUTION.json only written upon completion.
- `evaluate_family_external.py`: verify frozen files/full populations; seal ALL
  three cells BEFORE reading annotations; compute all30 rows, components/categories,
  disjoint panel, within-answer AUC, answer-length/position strata, fully-correct
  false flags,18 corrected contrasts and independent official evaluator replay.
  Supports `--seal-only`. Full command no args uses correct root/private paths.
- `diagnose_family_external.py`: after all-cell seals, label-free constant-channel
  counts and CUSUM/other signed+absolute score contributions; saves
  REPRESENTATION_DIAGNOSTICS.json. No new method or quality selection.
- `render_family_external.py`: after RED_TEAM.md exists, creates15 PNG/PDF pairs
  if diagnostics exist, English REPORT.md and REPORT.html gallery. Reads optional
  INTERPRETATION_HE.md into MD and an RTL HTML intro. Main figures01_all_methods,
  02_prmscore_components,03_precision_recall,05_paired_contrasts,09_literature.
  Synthetic rendering/layout checks passed. Inspect actual output images too.
- `archive_family_external.py --pool-dir <above> --output <new scratch.tar.gz>`:
  requires final reports/seals; packages full results and original source pool,
  verifies each member hash, writes ARCHIVE.json/ARCHIVE_CONTENTS.json. No upload,
  deletion or raw benchmark text. Archive creation code is predeclared/frozen.
- Core extractors `spectral_utils/family_{external,hist}_features.py`; metrics helper
  `family_external_metrics.py`. All code identities pinned; do not edit scored code.

## Independent review — prepared, not yet executed on external quality

Prior independent agents prepared scripts without external labels/summaries. They
are idle now; user asked SOL continuation, so use fresh SOL reviewer subagents to
execute/review these paths. Root+SOL lead leaves two active reviewer slots; run two
then the third as a slot frees. Three independent verdicts before headline claims.
Give them raw-only access, no main METRICS/CONTRASTS/REPORT or other reviewer outputs.

1. `independent_metrics/recompute.py`: all6190 raw shards, all30 primary scores,
   sixcomponents+categories,18 point differences, pinned official replay, hashes.
2. `independent_coverage/audit_population.py`: full source saved-feature replay vs
   pool and all6190 external raw/reference hashes,10arms,masks,source weights,
   thresholds, independent reconstruction of nine arms, CT7 reuse and seals. Requires
   SOURCE_CODE_COMPLETION schema already written. No label/quality decoding.
3. `independent_null/audit_null_math.py --mode full --authorization root-authorized-after-all-cells-sealed`:
   200 global and200 within-answer class-count-preserving label shuffles, shared Soc
   draws; all10arms. Also full6190 deterministic within-answer permutation of the
   standardized level_entropy family for tail candidate/F15equal, fixed weights,
   no refit. Independently reconstructs scores/metrics. Synthetic100tail/tie cases
   and6K=2 eigen cases passed. PREDECLARED.json exists. Exact authorization literal
   should be checked in script before running. Script author also wrote hist
   extraction, so original audit appropriately limits extraction independence;
   fresh SOL reviewer should preserve that caveat. Do not modify frozen script.

After null execution root/lead can create compact FEATURE_NULL_SUMMARY.json omitting
`rows` and recording full FEATURE_NULL.json SHA. Full file is gitignored/archived.
Reconcile all independent metrics/counts with main output yourself, then write
RED_TEAM.md as CLAIM | VERDICT | EVIDENCE with actual N. Do not promise superiority.
Negative result requires NEGATIVE_RESULT.md per `.claude/commands/negative-result.md`.

## Literature already verified

LITERATURE_CONTEXT.json and LITERATURE_AUDIT.md are complete and committed. Primary
sources: https://arxiv.org/html/2510.13744v1 (Hard2 tables2/5) and
https://arxiv.org/html/2505.23474v1 (Socratic tables3/4).
29Hard2 rows,11Socrows,220category values, available class recalls. Unpublished
class F1/precision are null, never infer them. Paper results are **context**, not
same-run reproductions, and exact published masks/thresholds are partly undocumented.
Hard2 published PRMs target-tune thresholds on100 responses, unlike our source-frozen
threshold. Hard2 Qwen3critic53.51,QwenMathPRM7B42.37,Universal60.27;
Soc QwenMathPRM7B68.0,QwQcritic73.8. No SOTA claim or matched significance vs papers.

## Remaining steps in order

1. Monitor active session34772 to full6190; on failure inspect and fail loudly,
   preserve shards. Correct only justified implementation defects; no silent recipe
   or threshold change. Do not start a duplicate writer.
2. Run evaluator `--seal-only`; notify SOL reviewers after ALL_CELLS_SEALED exists.
3. Run full evaluator and representation diagnostics; reviewers independently run
   on raw data. Main bootstrap may take several minutes. Frozen analysis code stays
   unchanged; report any defect transparently before fixing a necessary new version.
4. Reconcile all outputs, write RED_TEAM, negative evidence if applicable. Write
   concise Hebrew interpretation explaining whether tail adds value over B11 and
   matched equal, and which PRMScore components explain the change. No averaging
   promotion or unexposed-test claim.
5. Render15 plot pairs, visually inspect main/F1/literature/contrast panels. Write
   English findings supplement and `docs/experiments/FAMILY_TAIL_EXTERNAL_RESULTS_20260924.md`
   or equivalent discoverable report linking gallery/artifacts. Include coverage,
   source gate, limitations and CPU cost ledger (source validation vs external
   feature/scoring costs; zero new GPU). No fake peak-memory measurement.
6. Update HISTORY, PROGRESS and Research_Directions with own scoped sections,
   preserving others. Stage only own shared-file additions via index patch if needed.
   Append relevant LESSONS if reviewers find errors; existing own lesson already
   records precision/late hash timing/test-count issue.
7. Create verified archive, then upload using configured rclone if authorized.
   User broadly authorized Drive storage; auto-review previously required explicit
   authorization for a specific private source archive. If new upload rejected,
   do not bypass: explain action/reason and ask exact approval while finishing
   unaffected report work. Core evaluation can finish with truthful pending backup.
8. Run scoped diff/link checks; commit/push own compact evidence, code/docs/plots.
   Final Hebrew response self-contained with key results and clickable gallery/
   plots/report, independent-review status and material caveats. Do not stop at
   another progress-only response while the authorized task can continue.

## Backup / git details

Checkpoint commits: f0cdace65 (implementation),231890093 (analysis/plots/audits),
18136fd76 (source proof copy),4687660f8 (full source gate). Remote last pushed231890093;
later own commits still need push. Existing unrelated dirty files remain untouched.
Source23-file copy was verified against both df837 and committed root bytes.

Prior source dependency264,238,660-byte archive and prior external result44,876,664-byte
archive are SHA-verified on Drive, under
`gdrive:hallucination_detection/cluster_results/lsml_external_generalization_v1/evaluation_archives/`.
See old SOURCE_DEPENDENCY_ARCHIVE.json and EVALUATION_ARCHIVE.json for exact names.
Raw external telemetry1.918GB compressed is already Drive archived; do not upload
it again or publish decrypted Hard2 text. New feature/result archive can use same
project prefix with a new unambiguous SHA-derived filename.

rclone shared client occasionally403quota; retry transparently after about a minute.
Use copyto --immutable --checksum, bounded timeouts, then verify remote SHA via
streaming `rclone cat` or verified download stream. Never delete/move/sync Drive.
RESTORE.md describes dependency manifests and separate raw/private input requirements.
