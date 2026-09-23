# L-SML external generalization v1

Approved 2026-09-24. Branch: `codex/lsml-external-generalization-v1`.
Prior Codex checkpoint: `efe85052f`. MedPRMBench is deferred.

User execution priority update: collect complete reusable telemetry on AIRCC
first. Further feature extraction, source calibration and fusion work follow on
CPU. Collection is label-blind and does not inspect external quality; freeze both
methods before opening any external prediction/evaluation result. Retain all
raw quantities below so feature choices do not force another model pass.

## Scientific contract

Evaluate Hard2Verify and Socratic-PRMBench independently, using two locked methods:
frozen bank11 step-level continuous L-SML and CPU answer-local token-level L-SML.
PRMBench/ProcessBench remain previously exposed development data. No GPU training,
new encoder, answer generation, external calibration or external method selection.
Teacher-forced telemetry and separate comparator inference are authorized.

| Dataset | Telemetry | Comparator inference | Primary metric |
|---|---|---|---|
| Hard2Verify | Qwen/Qwen3-8B | same Qwen3 critic; Qwen/Qwen2.5-Math-PRM-7B; universalprm/Universal-PRM | official step Balanced F1 |
| Socratic-PRMBench | Qwen/Qwen3-8B; Qwen/QwQ-32B | same QwQ critic; Qwen PRM7 | official PRMScore |

Sources: [Hard2Verify release](https://github.com/SalesforceAIResearch/Hard2Verify),
[protocol](https://arxiv.org/html/2510.13744v1),
[Socratic release](https://github.com/Xiang-Li-oss/Socratic-PRMBench),
[protocol](https://arxiv.org/html/2505.23474v1).
Hard2Verify's harmonic mean of class recalls is NOT macro-F1. Never average
the two official metrics. Closed models appear only as published context.

## Method and source-validation lock

Preserve the audited 11 names/signs, guarded estimator and entropy orientation.
Frozen: per-channel Top10 within step, column z-score within answer, learned
weights, final within-answer z-score. Validate each existing source fold as test,
the next fold cyclically as calibration, and the other three as fit. Then deploy
one fit on four lowest-ID folds and calibrate only on the highest-ID fold.
Calibration is NumPy linear q80 of calibration step scores, valid iff risk < threshold.
Freeze groups, weights and each arm's threshold. No target recalibration,
including QwQ backbone transfer. Record all fit/calibration/test group identities.

Local: normalize token columns within the answer; use deterministic rows [::8];
require >=3 active channels and >=3*p fitting rows; guarded continuous L-SML;
orient against the fixed entropy column; score every token; Top10 each step;
final answer z-score. Calibrate on the same reserved source fold. This changes
both fitting scope and observation axis. Fit failure uses chosen-token surprisal,
never a hidden average. Save reason, active channels, weights, partition, timing.
Each variant has ordinary equal and learned-partition equal controls with
identical preprocessing, readout, fallback and native masks. CT7 is fixed.

## Data and leakage contracts

Pin dataset, code, model and tokenizer revisions plus prompt, dtype, thinking,
decoding, parsing and threshold provenance. Adapters provide stable answer IDs,
source-question groups, original questions, ordered steps, generator where known,
exact character/token spans. Labels/categories live only in evaluator files.
Preserve official masks and indexing. Never put decrypted Hard2Verify text in Git.
Audit exact whitespace-normalized questions and released source IDs against
PRMBench, ProcessBench and each external dataset. Do not claim paraphrase detection.
Report full official and disjoint-transfer panels; bootstrap source groups together.

## Execution gates

1. Validate source fits, adapters and evaluator parity; pin inputs and tokenize all
   examples without inspecting external prediction quality.
2. Resolve AIRCC SSH and disk prerequisites. Require >=10 GB local/TMPDIR and
   >=50 GB shared free, CPU smoke and preflight PASS in the submission session.
3. Timing only: <=12 deterministic examples per dataset/backbone, across length
   quartiles and including the longest, <=1 allocated GPU-hour per job.
4. Measure GPU/CPU hours, memory, storage, wall time, generation and retry costs.
5. Obtain the user's full-run budget decision BEFORE any full inference.
6. Run both datasets under the same locked recipe; first results cannot alter the
   second. Use commit-specific snapshots, Pyxis and NGC pytorch:25.01-py3.

Reuse the teacher-forced forward/quantity code and strict alignment checks.
Retain raw top50 IDs/logprobs, actual-token logprob even outside top50, full-vocab
logsumexp, entropy, IDs, offsets and step spans. No truncation. One writer per
shard, atomic records, explicit linear resume, exit85 on interruption. Large
results go directly AIRCC -> `gdrive:hallucination_detection/cluster_results/`.
Keep compact hashes/manifests locally. No automatic resume submission.

## Preregistered analysis

Seal all predictions before aggregate external quality evaluation. For each of
the three dataset/backbone cells, compare each L-SML variant to both matched
controls (12 primary fusion contrasts total). Register comparator and CT7
contrasts in the configuration before evaluation; apply Bonferroni across the
entire registered primary family, not separate convenient subsets. Paired source
question bootstrap: 100000 draws, seed20260924. Report native and complete-policy
metrics, class errors, within-answer ranking, coverage, failures and runtime.
Published target-calibrated PRM protocols must be separate from source-only
calibration; undocumented settings and parser failures are explicit deviations.

## Validation and acceptance

Test counts, duplicates, source grouping, masks/indexing, exact short/long-step
alignment, no truncation, telemetry/feature replay, label isolation, matched
controls, deterministic fallback and interruption/resume integrity. Check official
evaluators on fixed fixtures and independent confusion-count calculations.
Success is complete accounting for every registered arm on both available
benchmarks, with locked configs, overlap/provenance manifests, bundles, telemetry,
scores/decisions, diagnostics, paired intervals, costs and English report with
concise Hebrew interpretation. A negative result is valid completion. Partial
implementation or blocked inference must be labelled incomplete. Falsification:
no positive corrected fusion contrast means no demonstrated fusion advantage;
better source evidence alone does not establish external generalization.

Implementation status and executable commands are maintained in
[the runbook](LSML_EXTERNAL_GENERALIZATION_V1_RUNBOOK.md).
