# External evaluation execution lock - 2026-09-24

This execution supplements LSML_EXTERNAL_GENERALIZATION_V1.md before external quality evaluation.
Assumption: the historical bank11/CT7 recipes can be replayed from saved telemetry.
Success: complete seven-arm predictions for all 6,190 answer/backbone records,
source-separated calibration, official metrics, paired uncertainty and explicit failures.
A replay discrepancy falsifies implementation fidelity and stops quality evaluation.

## Methods and access

Seven arms: frozen_lsml, frozen_equal, frozen_partition_equal, local_lsml,
local_equal, local_partition_equal, ct7. No architecture or target-threshold sweep.
Frozen fitting preserves the original PB+PRMB 13,769-answer donor population.
Fit development folds 0-3; reserve fold4 for each arm's q80 linear-quantile
threshold, using all its unlabeled step scores. The same threshold transfers to
both datasets and both backbones. Source validation rotates evaluation fold k,
calibration (k+1)%5 and the remaining three fitting folds. No labels enter fitting
or q80 calibration. PRMB non-control source metrics are reported separately.

Empty external steps retain their IDs and official inclusion. All seven arms
predict incorrect (0), without consulting labels; their score is null. Answer-z
uses only nonempty steps. Native-score coverage excludes empty steps; primary
complete-policy metrics retain them. Local fit failure uses chosen surprisal for
all three local arms with matched routing. No silent equal-weight fallback.

## Contrasts and evidence

Primary family18: in each of three dataset/backbone cells compare each of the two
L-SML methods to its ordinary-equal, partition-equal and CT7 control. Paired source
question bootstrap100000, seed20260924; Bonferroni18 intervals. Native and disjoint
panels are descriptive; never choose a preferred method from these panels.
All three prediction sets must be sealed before aggregate external metrics.
Hard2Verify uses harmonic correct/error recall; Socratic uses pooled binary macro-F1.
Category panels, within-answer AUC, class error rates, failures and CPU timing are
secondary. Published comparator results are contextual, not reproduced inference;
no new comparator GPU run is part of this CPU completion request.

Exact normalized original/evaluated question overlap and available source IDs
are audited against development data and between external datasets. This cannot
exclude paraphrases or model-pretraining contamination. Keep full-set official
metrics distinct from disjoint-transfer diagnostics.

CT7 is the original seven-step-view recipe, with historical storage precision,
BOCPD temporal residual and pooled chosen-token step statistic. A Top10 fusion of
seven token streams is not an equivalent reference. Replay on source inputs is a gate.

No GPU training, new model pass or benchmark answer generation. Final results
must distinguish ranking transfer, threshold transfer and incremental fusion value.
