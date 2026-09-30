## Findings and next action

No recipe establishes a consistent improvement on both benchmarks. Keep the
full-grid IU/equal references and the Joint controls. Do not extend this
particular sensitivity objective into a larger search or promote the most
favorable point estimate from the pilot.

The larger-lambda suggestion was tested directly. Joint graph lambda 0.1
gives PRMB 0.66255 and PB 8.33%; lambda 1 gives 0.61426 and 8.33%; lambda 10
gives 0.51713 and 9.17%. The graph changes the score: its mean correlation
with the lambda-zero trajectory is 0.963 at lambda 1 and 0.923 at lambda 10.
It is not an inactive implementation. Larger influence does not provide a
consistent gain. Lambda-10 minus lambda-0.1 PRMB is -0.14542, CI
[-0.36596,+0.01892]; the small pilot does not establish a precise population
effect. PB difference is +0.83 points [-12.50,+14.40].

The automatic Joint graph rule gives PRMB 0.66760 versus lambda-zero 0.66171
on the same seven answers, difference +0.00590 [-0.04071,+0.04134]. PB falls
from 12.50% to 5.88%, difference -6.62 points [-19.79,+9.52]. Choosing lambda
without labels is feasible, but this objective has not chosen a winning
localizer. Joint feature-bank and grouping development remain open; neither
changed in this experiment.

Sensitivity did decrease on the eight additional, unused perturbations:
the full-matrix correction has 0.463 times the parent's sensitivity for IU,
and 0.542 for Joint. Yet IU PB falls from 17.71% to 5.77%, and Joint PRMB
falls from 0.66171 to 0.60331. The latter exploratory paired difference is
-0.05839 [-0.09582,-0.00805]. This separates stability under the imposed
perturbation from usefulness for error detection. Those perturbations may
disturb useful reasoning changes; the data does not establish them as noise.
The extra perturbation audit was added after freezing, did not use labels
or change any selection, and is not a new-answer confirmation test.

The error gate remains an important unresolved interface. IU with the selected
graph correction scores 14.45% PB under the unchanged GMM rule, but 24.64%
if we preserve its parent's binary error decisions and change only location.
This is a diagnostic, not a registered new winner; PRMB is still lower than
the parent (0.61141 versus 0.62261). It shows why a single F1 number should
not merge fusion ranking and clean/error gating into one explanation.

Next audit the answer-only normalization and no-error interface. Establish
which absolute uncertainty information is removed by per-answer centering,
and whether the current mixture states have evidence of a correctness
interpretation. Then freeze one bounded gate experiment with the fusion
scores and equal/entropy controls held fixed. If a pooled unlabeled gate is
needed, report it as a separate hybrid fit scope; do not call it answer-only
or calibrate it with test correctness labels. Preserve the feature-bank,
grouping and task-aware sampling tracks as open supporting-fusion work.

The complete comparator replay, untouched two-benchmark confirmation and
historical 24-cell transfer still remain. These are measured steps in
developing our fusion method, not completion of the full research objective.
