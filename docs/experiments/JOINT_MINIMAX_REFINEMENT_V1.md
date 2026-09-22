# Step411: rank deletion proposals by worst initial-factor information loss

One fixed intervention relative to Step410, frozen before fits or evaluation.
For each eligible feature deletion j, rank ascending by
max_f[1 - I_f(active without j)/I_f(initial)], over initial factors with
I_f(initial)>1e-8. Use the existing conditional-information deletion identity
for ranking and direct proposal calculation for the unchanged95% constraint.
The factor matrix and denominator stay INITIAL through every regrouping.
Stable ordering breaks floating-point ties as before. This changes ranking
from mean incremental relative loss of the current refitted factors to worst
cumulative relative loss of the original factors. It introduces no labels.

Preserve staged sparse membership, exact-alias treatment, K3/K4 source-fold
group discovery, same min2 eligibility and min8 limit, top3 proposal budget,
fit/discovery seeds, checked Joint identification, group reliability readout,
H1 sign orientation, and explicit recorded fallback. Every accepted state must
retain95% of every relevant initial factor; do not reset the reference, alter
thresholds, protect BOCPD, or choose a feature count from held labels.

25 fresh fits on all five banks: base50+BOCPD,15 exact copies,15 iid noise,
15 near copies,15 correlated nuisance streams.13769 answers/145597 steps,
five corrected source folds, hybrid training from other answers, same fixed
non-digit Tail15 Top10 q=.33 gate. This is studied development data, not an
untouched confirmation. No new inference/downloads or digit channels.

Keep all prior comparisons, including Step408 quality, Step409 regrouping,
Step410 feasible proposals, same-bank Continuous/equal, innovation5 and its
historical BOCPD correction. Six primary contrasts: new near vs Step410 near;
new base vs Step408 base; all four new additions vs new base.10000 paired
source-group bootstrap draws; two endpoints and Bonferroni confidence1-.05/12.
Each preservation criterion requires native13769, PB lower bound>-.01 and
within lower bound>-.002. Prior-base quality must pass independently of all
addition tests. No retrospective confidence/margin changes and no over40 claim
from the old comparator.

Tests verify budget-before-discovery, original-reference retention, native
group rejection and minimax ordering against direct reduced-system solves.
Audit membership, every accepted state, all proposed ranks via independent
direct solves, group rediscovery/guards, scores,37 metric bundles and all12
intervals. Numerical disagreement in an auditor must be inspected, not silently
waived. The selected state is last accepted. Fit cap25; at mostP-8 accepted
deletions per fit, at most3 attempts per deletion. Smaller counts are not proof
of useful selection. Stop after this bounded full run and review its evidence.
