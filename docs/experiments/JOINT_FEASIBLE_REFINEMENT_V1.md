# Step410: information feasibility before terminating refinement

One fixed change to Step409. For each of the same top3 eligible deletion
proposals, compute retention of the original factor information BEFORE group
rediscovery/fitting. If any relevant initial factor falls below.95, reject that
proposal and try the next of the same top3. Otherwise run unchanged K3/K4 source-
fold group rediscovery and checked Joint. Stop when none of the three passes
both information and native-fit constraints, or at the same min8 limit.
Select the last accepted state; there is no accepted sub95% crossing state.

Preserve the original information matrix/reference, current-factor importance
ranking, current-group min2 eligibility, seeds, group stability rule, checked
fit guards, staged sparse membership, exact-alias coordinates, group-reliability
readout and H1 orientation. No reference resets, wider proposal search, threshold
retuning, manually protected feature or label-based choice. Log information
rejections separately from group/fit rejections; a rejected information proposal
has no group discovery call. This is a new stopping rule, not a bug fix to the
correctly implemented first-crossing rule in prior frozen experiments.

25 fresh fits on the same five full banks,13769 held answers,145597 steps.
Five source folds, hybrid training on other answers, full development population.
No digit inputs, same Top10/BOCPD and fixed Tail15 gate, explicit H1 fallback only
if a final model fails; all failures counted separately. Existing Step408 quality
and Step409 robustness scores, same-bank Continuous/equal and historical
innovation5/BOCPD remain in the table. No new inference or data downloads.

Six primary contrasts: new near vs Step409 near; new base vs Step408 base;
four new additions vs new base. Two endpoints,10000 paired source-group draws,
Bonferroni confidence1-.05/12. Every preservation condition needs native13769,
PB lower bound>-.01 and within lower bound>-.002. Preserve the PRIOR Step408
base as well as the new-base addition performance. Do not lower that bar if the
new baseline falls. Over40 remains a hoped-for result, not permission to tune N.

Independent audits: sparse membership; fixed initial reference and all accepted
retentions; exact top3 ranking; each budget rejection via direct linear solve;
each admitted proposal's grouping replay/fit guard; final selected readout and
held scores; metrics, native coverage and all12 confidence intervals. Synthetic
tests check budget-before-discovery and preserved constraints only. Cap25 fits,
P-8 accepted deletions and at most3 proposals each; no additional retries.
