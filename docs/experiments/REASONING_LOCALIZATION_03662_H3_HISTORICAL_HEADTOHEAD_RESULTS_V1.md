# Reasoning Localization 0.3662 — Historical-Regime H3 Head-to-Head Results V1

Status: `COMPLETE / NUMERICALLY_BETTER_UNRESOLVED / RETROSPECTIVE`.

The historical evaluator was reproduced before any candidate audit result was
opened: entropy/top-five is exactly `0.3614213584` and the Stage-4 finalist is
exactly `0.3662328342`. Candidate locators were then frozen for the same eight
cells, fixed historical 40% calibration and 20% audit roles, 1,270 scorer rows
and 635 source-question groups. The score-freeze manifest records
`labels_selected=false`.

## End-to-end result

| System | Macro F1 | Exact error | Clean abstention | Within one |
|---|---:|---:|---:|---:|
| Historical entropy/top-five | 0.361421 | 0.288464 | 0.506719 | 0.498569 |
| Historical finalist | 0.366233 | 0.277686 | 0.580929 | 0.481794 |
| Current H0 | 0.374099 | 0.270195 | 0.614725 | 0.463622 |
| H2 cleanup+C7 | **0.374793** | 0.270569 | 0.614725 | **0.476993** |
| H3 equal+C8 | 0.372663 | 0.268241 | 0.614725 | 0.475836 |

Paired macro-F1 contrasts against the historical finalist:

| Contrast | Delta [95% CI] | Cell W/T/L | Worst cell |
|---|---:|---:|---:|
| H0 − historical | +0.007866 [−0.023593,+0.040751] | 6/0/2 | −0.025092 |
| H2 − historical | +0.008560 [−0.024610,+0.040869] | 6/0/2 | −0.037604 |
| H3 − historical | +0.006431 [−0.026891,+0.039473] | 5/0/3 | −0.034609 |

H2 is the raw-best arm, but none of the three current systems has a paired
interval wholly above zero. H3 is therefore numerically better than the
historical finalist on this matched regime, but the improvement claim remains
unresolved. This is not rejection and it is not evidence of equality.

H3's decomposition relative to the historical finalist is exact-error
`−0.009446 [−0.042440,+0.022608]`, clean abstention
`+0.033796 [−0.000121,+0.068607]`, and within-one
`−0.005958 [−0.038383,+0.024250]`. H2 has the same clean/error decisions and
its clean-abstention delta is `+0.033796 [+0.002114,+0.070698]`. The current
systems' favorable F1 point differences are therefore associated more with
the response detector's abstention behavior than with a demonstrated
first-error-localizer gain.

## Detector × localizer diagnostic

| Combination | Macro F1 |
|---|---:|
| Historical detector + historical localizer | 0.366233 |
| Historical detector + H3 localizer | 0.363163 |
| H0 detector + historical localizer | **0.379352** |
| H0 detector + H3 localizer | 0.372663 |

Under the historical detector, H3-localizer minus historical-localizer is
`−0.003070 [−0.028453,+0.020618]`. With the historical localizer held fixed,
H0-detector minus historical-detector is
`+0.013119 [−0.005697,+0.034240]`. The detector-by-localizer interaction is
`−0.003619 [−0.012159,+0.004363]`. Every interval crosses zero, so this is a
descriptive attribution pattern rather than a supported isolated effect.

## Program consequence

H3 retains its separate supported PRMBench advantage, but the preregistered
ProcessBench promotion condition required the H3-minus-historical interval to
be wholly above zero. That condition did not pass. H3 therefore remains a
`PRMBENCH_SPECIALIST / DUAL_TASK_DEVELOPMENT_CANDIDATE`, not
`DUAL_TASK_DEVELOPMENT_WINNER`. H2 must remain the separate ProcessBench
raw-best candidate in any independent confirmation because it outscored H3 in
this bridge as well as in the Llama scorer-family transfer.

Canonical artifacts are under
`results/reasoning_localization_03662_v1/phase_4/h3_historical_headtohead_v1/`.
The living report includes the absolute forest, paired-delta forest,
exact-versus-clean plot, and 2×2 detector/localizer heatmap.
