# Specification: frozen NRM-CS-IU confirmation on PRMBench v1

**Date:** 2026-08-13

**Status:** pre-registered confirmation of the already-frozen
`neutral-residual-mode-cs-iu-v1-2026-08-13` candidate.

## Question

Does the label-free neutral residual mode improve response-level correctness
ranking on a new model, dataset, and error taxonomy with enough positives to
resolve the inconclusive HLE interval?

No NRM direction, eigenmode rule, feature contract, IU configuration, trust
scale, or exclusion is changed after the HLE result.  The immutable calibration
is `results/neutral_residual_mode_cs_iu_v1/FROZEN_CALIBRATION.json`.

## Data and fixed exclusions

Use the Qwen3-8B teacher-forced telemetry for all 6,969 rows of
`hitsmy/PRMBench_Preview`.  Exclude exactly the three rows identified by the
independent data-readiness audit before this experiment:

```text
confidence_confidence_prm_train_p1_303
deception_deception_prm_test_p1_87
step_contradiction_step_contradiction_prm_test_p2_991
```

The resulting cohort has 6,966 rows.  The prediction unit is the complete
reasoning response.  `classification == "correct"` is correctness-positive;
every shipped error class is negative.  This is a response-level adaptation of
a step-localization benchmark, not the paper's official step-level metric.

Rows sharing `source_idx` are one bootstrap group.  This preserves the 758
paired control/error constructions and prevents them from being treated as
independent observations.

## Frozen score phase

The score phase:

1. filters only by the exact predeclared IDs;
2. passes only `token_entropies`, `token_spilled_energies`, `token_logsumexp`,
   and `top_k_logprobs` to the mixed-v2 constructor;
3. applies the frozen NRM calibration unchanged;
4. writes row IDs, source IDs, and correctness-oriented IU, CB-CS-IU, and
   NRM-CS-IU scores;
5. records hashes of the script, this specification, contribution module,
   frozen calibration, raw telemetry, and score artifact.

Neither `classification`, `category`, `error_steps`, nor any target field may
enter fusion.  Hashes must verify in a separate command before the report phase
reads `classification`.

## Evaluation

Primary metric: response-level correctness AUROC.  Report AUROC and AUPRC for
IU, NRM, and CB.  Estimate a paired 95% interval for NRM-minus-IU with 5,000
multinomial bootstrap draws over unique `source_idx` groups.

As diagnostics, compare the correct controls separately against every one of
the nine error classifications.  These subgroup contrasts are not tuning
targets and do not change the primary decision.

Pre-registered gates:

1. all frozen score/source hashes verify before targets are read;
2. fit payload is telemetry-only and declares no target fields;
3. all scores are finite, contribution reconstruction passes, the correction
   is IU-orthogonal, and effective weights reconstruct the score;
4. overall NRM-minus-IU AUROC point delta is positive;
5. the paired source-group bootstrap 95% lower bound is positive.

All five gates must pass for this confirmation to pass.  A failure is recorded
without changing v1.

## Interpretation boundary

NRM remains a trans-environment unsupervised fusion method: its six-dimensional
direction was calibrated from unlabelled original cells, not PRMBench.  The
PRMBench response label is unusually clean but constructed by benchmark error
class rather than natural answer correctness prevalence.  A pass demonstrates
cross-model, cross-example, and cross-error-taxonomy transfer of the fusion
correction; it does not establish the benchmark's official step-level score.
