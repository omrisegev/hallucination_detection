# Reasoning Localization 0.3662 — Historical-Regime H3 Head-to-Head V1

Status: `FROZEN_BEFORE_RUN`; retrospective matched-regime comparison. This
experiment must complete before any additional Phase-3 fusion or Phase-5 work.

## Primary question

On the exact historical Stage-4 ProcessBench audit population, does the frozen
H3-equal complete system outperform the historical `0.3662328342` finalist?

The primary directional hypothesis is `H3 > historical finalist`. Its null
boundary is zero. The older `+0.003` development promotion margin is reported
as practical context only and is not the null hypothesis here.

## Population and immutable anchors

- eight cells: Qwen3-8B and Llama-3.1-8B × GSM8K, MATH,
  OlympiadBench and OmniMath;
- historical deterministic `40%` calibration and `20%` audit roles from
  `_stage_partition`;
- 635 source-question audit groups and 1,270 scorer rows;
- exact historical ProcessBench harmonic macro-F1 evaluator;
- equal-cell aggregate and source-question grouping shared across scorer
  copies;
- historical 2,000-draw grouped bootstrap and seed `20260816`.

Before candidate labels open, the executable must verify the existing
checksum-equivalent P0-S0 replay and its exact anchors:

- entropy/top-five: `0.3614213583669282`;
- historical finalist: `0.3662328341717007`.

Any hash, population, alias or count mismatch is a hard failure before a new
scientific result.

## Frozen candidate application

H0, H2 and H3 are imported from their already frozen Qwen and Llama score
artifacts. Their token transforms, family membership, reducer, C7/C8 scores,
orientation and H3 `0.5/0.5` within-response rank fusion are not refit.

- H0: frozen five-family/top-ten localizer and `equal_feature_mean` response
  detector score.
- H2: remove sampled-token energy, remove partition `energy_series`, insert
  frozen C7 inside entropy dynamics; H0 detector retained.
- H3: equal rank fusion of H2 with frozen C8; H0 detector retained.

The only regime-specific fitted value is one H0 detector threshold per cell,
using the historical 40% calibration labels and H0's frozen detector/localizer
scores. H2 and H3 copy the resulting H0 clean/error decision exactly. Audit
labels are forbidden until all candidate scores/locators and score hashes are
frozen. This is the exact current-system application to the historical rows;
it does not refit H3 on the historical audit.

The current score-side transformer is label-free and was originally fit on the
full registered cell, whereas the historical heads were fit on their 40%
calibration rows. This access difference is part of the end-to-end system
comparison and must remain explicit; the detector/localizer cross below
separates its observable decision contribution.

## Frozen comparisons

Primary end-to-end contrast: `H3 − HISTORICAL_FINALIST`.

Required end-to-end context:

- H2−historical, H0−historical;
- H3−H0 and H3−H2;
- H3−entropy and historical−entropy.

Required shared-historical-detector diagnostic:

- historical localizer;
- H0 localizer;
- H2 localizer;
- H3 localizer.

Required 2×2 detector/localizer cross:

- historical detector + historical localizer;
- historical detector + H3 localizer;
- H0 detector + historical localizer;
- H0 detector + H3 localizer.

The end-to-end comparison is primary. The shared-detector and 2×2 results are
mechanism diagnostics and cannot override it.

## Metrics, inference and verdict

Report macro F1, exact first-error accuracy, clean abstention, within-one,
per-cell metrics, paired delta, grouped-bootstrap CI, cell W/T/L and worst-cell
delta. The primary conclusion is:

- `SUPPORTED_IMPROVEMENT` if the H3−historical F1 interval is wholly above
  zero;
- `NUMERICALLY_BETTER_UNRESOLVED` if its point estimate is positive and the
  interval crosses zero;
- `NO_EVIDENCE_OR_WORSE` for a non-positive point estimate, with the interval
  reported rather than translated into equality.

Because H3 was developed after earlier ProcessBench results opened, every
result remains `RETROSPECTIVE` and requires fresh-question confirmation.

If H3 is supported above the historical system and retains the already frozen
PRMBench advantage, its program status becomes
`PHASE4_COMPLETE / DUAL_TASK_DEVELOPMENT_WINNER /
FRESH_CONFIRMATION_REQUIRED`. If H2 wins instead, H2 and H3 remain distinct
ProcessBench and PRMBench candidates. No method is altered after audit labels
open.
