# Step409: rediscover groups during information refinement

Frozen one-variant test motivated by Step408's all-row stage diagnostic.
Keep staged sparse membership, aliases, seeds, source-fold access and gate.
During the95% refinement only, rediscover groups for EVERY proposed deletion
using the existing training-source-fold stability rule K={3,4}. Use its highest
ranked admissible partition; reject the deletion if that partition fails checked
Joint. No fallback to old labels and no extra K/threshold search.

Keep the three lowest-importance eligible deletion proposals, the minimum-two
current-group eligibility rule, min8 remaining, the per-step checked-fit seeds,
and first95% retention crossing. Eligibility uses CURRENT groups; admissibility
is checked again after regrouping. Discovery seed remains outer_seed+100 for
all proposals; checked-fit seed is outer_seed+200+deletion_step.

Crucially, retention always refers to the INITIAL global and group factor matrix
and INITIAL information, even when the currently fitted groups change. Do not
reset the reference after a deletion. Importance for the next proposal uses the
current fitted global/group factors, as before. Group-reliability final readout
and H1 orientation are unchanged. Store partitions, discovery records, accepted
and rejected proposals, factor information and all fit guards along the path.

Full13769 matched development benchmark and five banks from Step408: base51 with
BOCPD, exact copies, iid noise, near copies and structured noise.25 fresh fits,
bounded by P-8 deletion steps and at most3 proposal attempts each. No additional
solver retries. Source-fold hybrid fitting, no digit inputs or protected feature.
Unchanged guard failures are explicit; failed final models use declared H1 and
are counted separately. Preserve the Step408 references, original Joint,
same-bank Continuous/equal and historical innovation5/BOCPD.

Six primary contrasts: new near vs Step408 near; new base vs Step408 base;
all four new additions vs new base.10000 paired source-group bootstrap draws,
two endpoints each, confidence1-.05/12. Preservation requires native13769 and
lower CI bounds >-.01 PB, >-.002 within for each addition. Also report whether
new base preserves old base under these margins: moving both base and near
downward is not a useful solution. No score-exact baseline requirement here,
because refinement intentionally changes on the base too. No claim of >40
unless actually achieved by this frozen method, not a historical comparator.

Independent audit of staged membership, sourcefold grouping replay, checked-fit
guards, initial-reference retention throughout the path, selected final weights,
held scores, full metrics and intervals. Synthetic checks address reference
retention and explicit rejection, not real-world quality. No causal claim that
all near-copy regression comes from fixed grouping; the experiment tests that
specific intervention. Do not open another variant based on intermediate folds.
