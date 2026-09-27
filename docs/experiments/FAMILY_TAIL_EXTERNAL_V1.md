# Frozen family-tail transfer to Hard2Verify and Socratic-PRMBench

Execution date: 2026-09-24. Branch: `codex/lsml-external-generalization-v1`.
This is the authorized external execution of the handoff on
`codex/family15-tail20-transfer-v1`, commit
`df8371f3c56574b69f69db14299a460ebc225bc1`. Its worktree remains read-only.
The lock and portable scorer are copied byte-for-byte. No method sweep, GPU
training, target fitting, target calibration or new language-model pass is added.

## Contract and decision

Assumption: existing teacher-forced telemetry is sufficient to reproduce all
48 original step features. Success means complete, numerically faithful transfer
and an audited comparison, whether or not the candidate improves the score.
An unexplained source parity failure falsifies implementation readiness and stops
external scoring. A substantive change of definition requires clarification first.

The candidate is `F15_tailtie_lsml`: source-fitted L-SML on centered, fractional-tie
top-ceil(20%) step marks, with weights applied to continuous family features.
There is no extra pooled standardization of the tail marks. Source fit folds0-3
and calibration fold4 remain fixed; its threshold is `0.878464199608311`.
Within-answer normalization at deployment is offline full-answer processing.
Feature selection and signs used source development evidence; this is not an
entirely label-free design or answer-local fitting at deployment.

All ten locked arms are retained: bank11 L-SML, family15 tail L-SML, family15
continuous L-SML, bank28 continuous L-SML, bank28 equal, family15 equal,
bank11 equal, bank11 partition equal, original48 equal and CT7. Averaging is a
control, not a proposed final method. CT7 predictions are reused only after
checking identical telemetry and masks against the previous sealed evaluation.
Bank11 scores and decisions must also replay the previous three matching arms.

The source extraction gate checks all13,769 answers/145,597 steps, all48 channels,
against the source pool after answer-z, at maximum absolute tolerance1e-6.
The earlier two-channel reducer audit and the18-answer extraction smoke are
limited checks, not substitutes for this full gate. Actual source storage is
float64 for CT7/historical streams and float32 for bank11 tokens; preserve each
rounding point, including the temporal intermediate and evidence-drop stream.
CT7 chosen-surprisal features use ordinary Top10 with no first-step replacement.
The robust entropy exceedance feature is a fraction, not Top10.

## Population and sequencing

| Cell | Answers | Official endpoint |
|---|---:|---|
| Hard2Verify / Qwen3-8B | 200 | Balanced F1: harmonic mean of correct/error recalls |
| Socratic-PRMBench / Qwen3-8B | 2,995 | PRMScore: mean of correct/error F1 |
| Socratic-PRMBench / QwQ-32B | 2,995 | PRMScore: mean of correct/error F1 |

Preserve original inclusion masks and all step positions. Remove empty steps
before normalization/fusion; restore them with null score and invalid prediction.
Seal every cell before the evaluator opens external annotations. Replay pinned
official evaluators, including Socratic's degenerate-value conventions and its
out-of-range error annotations. Never average the different benchmark metrics.

The complete population is primary. Also report the observed source-disjoint
sensitivity panel with the existing exact-question/source-group closure. This
does not rule out semantic or pretraining overlap. These datasets were already
evaluated in the earlier study: this run is exploratory follow-up, not untouched
confirmation, even though its new recipe and analysis are locked before scoring.

The six locked paired contrasts in each of three cells form one family of18:
tail vs bank11, tail vs family equal, tail vs continuous family, continuous
family vs continuous28, family equal vs equal28, and equal28 vs original48 equal.
Use100,000 source-question bootstrap draws, seed20260924, Bonferroni correction
across18. Keep all variants of a source question together. Per-method95% intervals
are descriptive; use the paired corrected intervals for comparative claims.

## Added reporting, without changing the method

Produce standalone PNG and PDF figures plus an English technical report and a
concise Hebrew interpretation/gallery:

- All ten alternatives in each cell; fixed order and consistent colors.
- Both class F1 components of Socratic PRMScore, plus both precisions/recalls.
- Category panels, answer-length and relative-step-position diagnostics.
- Correct/error recalls and wholly-correct-answer false alarms for Hard2Verify.
- All18 corrected paired contrasts and the source-disjoint sensitivity panel.
- Published comparator scores and reported components in separately marked
  literature panels. Missing components remain NA; no invented decomposition.
- Coverage, constant channels, empty steps, runtime and frozen-score provenance.

Published scores are context, not same-run reproductions. Hard2Verify's published
PRM results use target threshold tuning, unlike our frozen-source threshold.
Separate those access conditions, model sizes and inference protocols. Literature
sources and extracted tables are in `LITERATURE_CONTEXT.json` and
`LITERATURE_AUDIT.md`; comparator inference remains outside this CPU follow-up.

Before reporting headline findings, three independent audits recompute raw-record
metrics, verify full-population coverage/frozen algebra, and perform null/math
checks. Reconcile discrepancies before drawing conclusions. Perturbation checks
are diagnostics and cannot silently replace the registered candidate.

## Reproduction and artifacts

Root artifact directory: `results/family_tail_external_v1/`.

```text
python scripts/verify_family_external_source.py --pool-dir <source-pool-directory> --output results/family_tail_external_v1/source_full_v1 --workers 4
python -m unittest discover -s tests -p test_family_external_features.py -v
python scripts/run_family_external.py --workers 4
python scripts/evaluate_family_external.py
python scripts/render_family_external.py
```

The source gate output directory must be new. The scorer is resumable with a
single writer per shard and rejects changed input/code identities. The evaluator
requires complete populations and seals before labels. Rendering requires the
independent-review artifact. Use absolute paths when the shell ignores its cwd.

`IMPLEMENTATION_FREEZE.json` pins extraction, scoring, analysis, source lock and
input bytes before external predictions. `SOURCE_CODE_OBSERVATION.json` records
an explicit mid-source-run hash/mtime observation: it is not falsely described
as a prelaunch attestation. The completion observation rechecks those identities.
`EXECUTION_PLAN.json` is an immutable initial specification; its initial status
does not describe later run completion. Completion belongs in separate artifacts.

Raw telemetry, decrypted Hard2Verify text, per-answer feature records and large
bootstrap arrays stay out of Git. Retain compact hashes, gates, reports and plots;
archive the reproducibility dependencies and outputs with a restore manifest.
MedPRMBench remains deferred.
