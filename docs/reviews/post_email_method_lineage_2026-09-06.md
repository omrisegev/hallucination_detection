# August 28 evidence and the new September method

The August 28 email is the motivation and historical baseline. The current
method is defined by the subsequent repository work, most recently Joint
L-SML optimization v2 with R1/R2 at `7803cd55`. Do not restart the method
selection from the thirteen answer-level methods in the older email.

## Verified development since the email

| Date | Commit | What changed |
|---|---|---|
| August 29-31 | `e2123168`, `ff8cd0d8`, `35a7a3a3` | Localization reconstruction, feature/dynamics studies, H3 transfer and the historical 0.3662 contract bridge. |
| September 1 | `250e092e` | Phase 3 froze eleven completed experiments without promoting a successor. Unrun variants remained untested, not failures. |
| September 4 | `c5a658a6` | New Joint L-SML estimator, stable partition selection, orientation/pruning, numerical diagnostics and comparisons of weight calculations. |
| September 4 | `18f9d270`, `88866daf`, `0ce48968` | Existing-data evaluation, separate PB coverage amendment, and diagnosis of scale and fitted-model/readout mismatch. |
| September 5-6 | `601db6c6` through `7803cd55` | Implemented current v2: common score scale, grouping/readout comparisons, DUFS integrations, alternative Joint model-inverse weights, trajectory fusion and coverage-aware selection. |
| September 6 | `6dea2f79` | Separate window-feature implementation and corrected label-free feasibility audit; no window fusion model evaluated yet. |

## What is new in Joint L-SML

The model fits the shared and group-specific factors together:

`Sigma_off = v v^T + blockdiag_g(u_g u_g^T)`.

The first term captures covariance shared across groups. The second captures
additional dependence within each group. The implemented fit uses deterministic
multi-start coordinate updates and numerical/identifiability diagnostics.
The partition is selected by leave-one-answer-out agreement, rather than
choosing whichever partition minimizes covariance residual on the same data.
This is a new project estimator, not merely a new fixed feature subset or a
renaming of ordinary continuous L-SML. Literature-level novelty still requires
its own precise comparison; the repo establishes implementation, not priority.

The structural report at `c5a658a6:results/joint_lsml_v1_r2/REPORT.md` records
16 admissible fits out of 18 lanes and better misfit in all 16 fitted lanes.
The two blocked lanes have no model. Those are structural observations, not
18 independent localization trials. Existing-data scoring then showed that
the original hierarchical readout did not deliver a localization gain.

That motivated a second substantive step: the current v2 separates estimating
the model from deriving useful weights. Its regularized model-inverse rows
use the model covariance containing both the shared and group-specific fitted
factors. Other rows test the original hierarchical map, ordinary continuous
L-SML on matching groups, soft gates and gate-informed grouping. Common
fused-score SD=1 addresses scale transfer. R1 learns across trajectory order
statistics; R2 bars incomplete configurations from tuned selection.

## Consequence for the next experiment

The next question is: does the current Joint model and its score construction
improve localization when observations are token windows, and does fitting
inside one answer help relative to pooled training windows?

Use the current Joint/L-SML family as the candidate lineage, its registered
IU family as the matched comparator, and averages/unchanged L-SML as controls.
Keep the original fixed `internal_joint` and `internal_cont` rows distinct from
inner-selected families. Tune only within training/development boundaries.
Do not label the v1 hierarchical map, v2 Joint model-inverse and future window
adapters as one interchangeable algorithm.

The comparison must separate three changes: estimator/readout, token versus
window representation, and single-answer versus pooled fit scope. Keep the
response/no-error mechanism and primary reducer matched while isolating each
change. The 23-token-stream contract cannot simply be pasted onto 30 window
feature definitions. Adapt provenance/eligibility and check brief-signal and
boundary preservation explicitly. These checks extend the current method;
they are not a return to global-detection method searching.

Current-source entry points:

- `hd_jlsml_v2_wt/spectral_utils/joint_lsml.py`: `fit_joint_lsml`,
  `hierarchical_joint_weights`, `regularized_joint_map_weights`.
- `hd_jlsml_v2_wt/spectral_utils/joint_lsml_v2_localization.py`: `fit_v2_arms`
  and the explicit 16-vs-16 roster.
- `JOINT_LSML_OPTIMIZATION_PLAN_V2.md` and amendments R1/R2 define the current
  development contract. The older Claude handoff and v1 draft are historical
  context where superseded.

No new outcome scores were read or generated for this lineage review.
