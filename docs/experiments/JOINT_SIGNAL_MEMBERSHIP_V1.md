# Step407: Global-signal membership with nuisance groups

Frozen before the new fits; one mechanistic variant, no parameter grid.
Step406 showed every structured-noise group had exactly zero sparse global
loadings across all converged starts, yet survived through its local loadings.
The new rule retains local-only rows when their group has a globally connected
member in ANY converged start, and removes a whole group from active signal
membership only when ALL its global loadings are exactly zero in ALL converged
starts. This is a model-support decision, not a correctness certificate.

Apply the rule inside each existing sparse membership round, before rediscovery.
Keep the old feature-wise (v OR u) rule within surviving groups. Relearn groups
on the retained signal coordinates; at most three monotone membership rounds.
If no admissible identified structure remains, declare failure. Keep K3/K4,
profiled Jacobian, multistart, diagonal and conditioning guards unchanged.
Do not simply preserve the two provisional non-noise groups or waive K>=3.
The dropped groups' local nuisance fits remain recorded in each round audit.

Everything else stays frozen: source-fold seeds, 31 sign draws and .95 null
calibration, original sparse objective, final unpenalized Joint, 95% information
refinement, group-reliability readout, H1 sign orientation, fixed Tail15 gate,
Top10 feature bank, corrected labels/source groups, explicit H1 fallback only
for a failed final model. No digit inputs or protected BOCPD coordinate.

Full matched benchmark: 13769 answers, 145597 steps, PB6800/PRMB6969, within
AUC6030. Hybrid training on other source folds; whole-answer standardization.
Five banks each get five new fits: base51 including BOCPD; 15 exact copies;
15 iid noise; 15 near copies; 15 persistent correlated noise. Reuse the exact
deterministic addition generators from Steps401/406. No new inference.

Retain old refined Joint, same-bank Continuous/equal and historical innovation5
and BOCPD scores. Primary paired contrasts: structured new vs structured old;
new base vs old base; all four new additions vs new base. Six contrasts, two
endpoints, 10000 source-group bootstrap draws and Bonferroni confidence
1-.05/12=99.583333%. Preservation requires full native coverage and CI lower
bounds >-.01 PB, >-.002 within for each addition. Require baseline score replay
within1e-11 to call the base preserved. Report approximate-copy failures openly.
This is a development intervention, not untouched confirmation or a promised
solution to near-copy instability. No best-N or after-result threshold choice.

Execution cap:25 fits; no retries beyond declared solver starts. Save each fold
and scientific hashes; resume only unchanged checkpoints. Independent audit of
new support decisions, sparse objective/null covariance, full refinement path,
final weights/held scores, metrics/coverage and bootstrap contract. Synthetic
fixtures test membership logic only, never establish real-data quality.
