# Evidence-domain independence diagnostic v1 (Step 414 [Claude], 2026-09-17)

Diagnostic, development only. **Labels are used for measurement and never for fitting**: the
conditional correlations below need the step label by definition. No arm here is a candidate method.
Script: scratchpad (`family_independence.py`, `four_domain_test.py`); numbers reproducible from
`results/digitfree_broad50_v1/extracted` + `results/joint_feature_selection_bocpd_v1/INPUTS.npz`.

## Question

Omri, after the Step 413 result that no fusion rule beats averaging: L-SML needs more than three
conditionally independent evidence sources (below that the small-m guard makes it an average by
construction). Do we have such sources? Specifically: are tail and entropy distinct; is there more
temporal evidence; does answer structure add an independent channel?

## Method

Thirteen candidate evidence families on the 50-stream digit-free bank plus BOCPD residual, plus two
structural channels (relative step position, answer length in steps). Each family virtual is the
equal mean of its z-scored, oriented members. On the 94,203 labelled PRMBench steps (14.0% error):
marginal correlation among family virtuals, then correlation after centering **within each label
class** (the conditional independence L-SML assumes), and the participation ratio
`(sum lambda)^2 / sum lambda^2` of each correlation matrix as the effective number of independent
signals.

## Result

| family | m | member AUC | family AUC | within-family mean abs corr |
|---|---|---|---|---|
| rank_risk | 15 | .684 | .699 | .80 |
| provided_token | 4 | .668 | .675 | .84 |
| tail | 2 | .678 | .680 | .96 |
| entropy_shape | 7 | .690 | .693 | .95 |
| varentropy | 6 | .682 | .710 | .70 |
| top2_ratio | 1 | .662 | .662 | - |
| shape_innov | 7 | .694 | .698 | .93 |
| ve_innov | 6 | .685 | .713 | .69 |
| dynamics (turnover, JS) | 2 | .569 | .571 | .50 |
| bocpd | 1 | .647 | .647 | - |
| noreset | 1 | .709 | .709 | - |
| position | 1 | .646 | .646 | - |
| n_steps | 1 | .667 | .667 | - |

Effective independent signals (participation ratio over the 13 family virtuals):

| | value |
|---|---|
| marginal | 2.75 |
| **conditional, given the step label** | **2.83** |

Selected conditional correlations: tail-entropy_shape .64, tail-rank_risk .80,
entropy_shape-shape_innov .96, varentropy-ve_innov .95, ve_innov-noreset .95,
dynamics-everything-distributional .07 to .24, position-everything .02 to .22,
n_steps-everything -.05 to -.01.

## Answers to the three questions

1. **Tail and entropy are not distinct.** Conditional correlation .64 with each other and .80
   between tail and the rank-risk family. They are the same evidence under different transforms.
2. **The temporal channel is mostly captive.** Prefix innovations stay at .87-.96 with the family
   they are differenced from: an innovation is the same signal minus its own running mean, not a new
   source. Only BOCPD (.44 with dynamics, .42-.69 with the distributional families) and the
   turnover/JS dynamics pair (.07-.24 with everything distributional) behave as separate channels.
3. **Answer structure is the most independent channel measured, and partly unusable.** Relative step
   position is nearly orthogonal to all telemetry (.02-.22). Answer length in steps is orthogonal to
   all thirteen families (|corr| <= .05), **but it is constant within an answer**, so after
   answer-standardization it is exactly zero and carries no within-answer information at all. Its
   .667 pooled AUC is entirely the between-answer length prior (consistent with the Step 354 length
   control). It can only inform the gate, never the step locator.

## Reading

The effective number of independent signals in one forward pass of telemetry is **under three**, and
it does not move when families are added: every subset tried (3, 4, 5, 6, 7 families) has
participation ratio 2.7-2.9. This is the structural reason the whole Step 413 ladder collapsed to
averaging. It is not a defect of L-SML, Joint or the K rule: below three conditionally independent
classifiers the L-SML eigen-stage is undetermined (Step 205) and the registered guard returns equal
weights, so an average is the correct output of the method on this bank.

The independent channels that do exist are the weak ones (dynamics .571, position .646) and the
strong ones are all one signal (.69-.71, mutually .8-.96). An exploratory fusion of the
independent-only domains under a label-free orientation scored far below the redundant blob
(equal weight: 29.5 PB on position+dynamics+ve_innov versus 39.2 on the plain 20-stream mean), and
the L-SML/equal contrasts on those small sets were unstable in sign, so no ranking is claimed from
them. Adding the independent-but-weak channels to the strong blob did not reach the 6-component
leader either (equal weight: 36.1 / 37.6 / 38.0 for 5 / 6 / 7 families versus 40.27).

Consequence: a bank with more than three independent sources cannot be built by adding transforms of
the output distribution of one greedy pass. It requires a measurement channel we do not currently
extract - internal layer states, multiple samples of the same answer, or a second model - each of
which changes the access contract and must be declared as such.
