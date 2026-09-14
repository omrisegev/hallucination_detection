# Fusion input-normalization ablation v1

Status: **COMPLETE / REVIEW PASS — DEVELOPMENT ONLY**

The frozen q15 four-view bank was evaluated under answer-z, scale-only and
literal raw inputs. Every ProcessBench result below uses the already frozen
tail15 Top10 q=.33 gate; no q was changed. Joint L-SML was not run because this
bank is structurally inadmissible and its expensive multistart fit was deferred.

| Solver | Input | PB all-8 | PB raw exact | PRMB within | PRMB fold AUC | PRMB pooled OOF | PRMScore |
|---|---|---:|---:|---:|---:|---:|---:|
| equal | answer_z | 37.1057% | 32.9356% | 0.746602 | 0.675240 | 0.675153 | 0.589883 |
| equal | scale_only | 37.1057% | 32.9356% | 0.746602 | 0.716026 | 0.715708 | 0.630093 |
| equal | raw | 36.8759% | 32.6655% | 0.751115 | 0.721815 | 0.721407 | 0.634805 |
| local_iu | answer_z | 37.2406% | 33.0932% | 0.745703 | 0.671593 | 0.671526 | 0.587001 |
| local_iu | scale_only | 36.3883% | 31.6974% | 0.751131 | 0.650926 | 0.650361 | 0.559994 |
| local_iu | raw | 32.8165% | 27.2850% | 0.681698 | 0.519638 | 0.519723 | 0.506705 |
| external_iu_static | answer_z | 37.1865% | 33.0257% | 0.746279 | 0.675035 | 0.674944 | 0.589507 |
| external_iu_static | scale_only | 37.2066% | 33.0257% | 0.746353 | 0.715664 | 0.715304 | 0.630132 |
| external_iu_static | raw | 37.2079% | 33.1157% | 0.746155 | 0.717314 | 0.716816 | 0.629219 |
| external_iu_position | answer_z | 35.0811% | 31.2697% | 0.739485 | 0.663664 | 0.663602 | 0.583683 |
| external_iu_position | scale_only | 35.7377% | 31.7875% | 0.732939 | 0.697048 | 0.696759 | 0.616186 |
| external_iu_position | raw | 36.0467% | 31.6974% | 0.729173 | 0.697127 | 0.696693 | 0.616460 |
| local_shrink_pooled | answer_z | 37.2606% | 33.0932% | 0.745667 | 0.670549 | 0.670486 | 0.587014 |
| local_shrink_pooled | scale_only | 36.6461% | 31.8325% | 0.749488 | 0.633967 | 0.633418 | 0.550590 |
| local_shrink_pooled | raw | 32.0697% | 26.3845% | 0.664522 | 0.517677 | 0.517764 | 0.506926 |
| local_shrink_position | answer_z | 36.4428% | 32.7330% | 0.747381 | 0.671646 | 0.671582 | 0.587410 |
| local_shrink_position | scale_only | 36.0895% | 31.3147% | 0.748161 | 0.633403 | 0.632851 | 0.550177 |
| local_shrink_position | raw | 31.0212% | 25.3940% | 0.675666 | 0.522050 | 0.522097 | 0.509854 |
| local_shrink_position_scale_only | answer_z | 36.1387% | 32.3503% | 0.746325 | 0.672621 | 0.672558 | 0.588215 |
| local_shrink_position_scale_only | scale_only | 36.5457% | 31.7875% | 0.751606 | 0.647370 | 0.646808 | 0.556445 |
| local_shrink_position_scale_only | raw | 30.8668% | 25.3489% | 0.695851 | 0.534314 | 0.534308 | 0.513834 |

Best tested fusion arm by PB is `answer_z__local_shrink_pooled` at 37.2606%. The current
development-frozen per-view-Top10 locator remains 37.4749%
PB, 0.753436 PRMB within and
0.634412 PRMScore.

## Interpretation

- For equal fusion, removing centering alone is localization-invariant but
  restores pooled OOF AUROC from 0.675153
  to 0.715708 and PRMScore
  from 0.589883 to
  0.630093. This is calibration
  recovery, not better step selection.
- Natural-unit equal raises PRMB within by
  +0.004513
  but changes PB by
  -0.230pp.
- Literal raw local IU and local shrinkage collapse. The median input
  second-moment condition number rises from
  65.8
  under answer-z to
  23334.6
  in natural units, violating the scale assumptions used by their spectral
  solve.
- Raw/scale-only external stationary IU recovers cross-answer calibration but
  produces essentially no localization gain over its answer-z version.

Decision: `RETAIN_CURRENT_LOCATOR; KEEP_SCALE_ONLY_AS_CALIBRATION_CONTROL`. Non-z arms passing the within-solver development screen:
`scale_only__local_shrink_position_scale_only`.
Arms also reaching the current frozen locator's PB/within noninferiority region:
none.

Literal noncentered inputs are documented deviations from IU-PCR's expected
z-scored contract. Condition diagnostics and all paired contrasts are retained
in `METRICS.json`; this is not external confirmation.
