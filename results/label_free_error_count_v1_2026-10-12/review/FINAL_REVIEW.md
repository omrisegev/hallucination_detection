# Final independent review — PASS

Reviewer: `independent_review`, which implemented no study arm. Review work was
limited to static inspection, independent arithmetic scripts/evidence, and the
root-requested synthetic replay-wrapper guard tests. No original advisor file,
canonical helper, registered model, prediction, or protocol was changed by this
reviewer. **No unresolved material finding remains.**

## Protected primary result

Independent reconstruction agrees with the delivered evaluator: **none of the
seven N16 candidates passes the combined criterion**. The multiplicity remains
seven, the recall allowance remains one percentage point, and failed candidates
do not reduce the family or become zero-count predictions. N56 remains a
prespecified sensitivity check.

N16 IU-PCR Q80 has MAE `1.425696`, PRMScore `0.653859`, and error recall
`0.487566`. LOCAL gives MAE `1.411045`, PRMScore `0.649044`, and recall
`0.475473`. Its MAE change versus Q80 is not established, while its PRMScore
decreases. B and D are complete but do not improve the protected result. N16 A
has only `1,269/6,211` noncontrols because four folds have infeasible emissions;
its partial points cannot qualify as a full-population result.

## Independent verification

- Authenticated the main 64-file prediction seal and expected population hash
  before opening labels. Population joins reproduce 6,969 answers, 94,203 steps,
  707 whole source groups, 6,211 noncontrols, 758 synthetic clean controls, and
  934 valid-step true-zero answers. Source groups remain confined to one fold.
- Reconstructed stable within-answer ranks and exact top-k placements shared
  by every count arm. Candidate quotas are identical under IU-PCR and L-SML;
  method-specific Q80/entropy controls retain their own quotas. No global OOF
  ranking metric is used.
- Independently reconstructed 44,662 saved count distributions, posterior
  means/medians, selected-triple posteriors, D active/marginal posteriors and
  structural-zero mixtures. Maximum PMF error is `7.99e-15`; maximum posterior
  error is `6.89e-15`; D training log-likelihood error is `7.11e-15`.
- Recomputed all 440 candidate group triples over their 200 whole-source
  bootstrap draws. Draw identities, eligibility masks, covariance bounds,
  normalized prevalence IQRs and selected triples agree; prevalence error is
  at most `1.12e-16`. Stability does not establish conditional independence.
- Rebuilt 988 point-metric rows and compared 19,812 evaluator values, including
  all nine types, MS valid-step F1, source coverage and length masks. Maximum
  error is `9.37e-17`. Reproduced the exact 20,000 source multiplicities and all
  28 protected contrast intervals/gates exactly.
- Verified compensated coordinates against an independent Decimal80 oracle.
  The severe-cancellation fixture reconstructs V within `4.0e-15`, retaining
  the original `1e-9` bar. The original failed numerical attempt remains a
  separate archive; no count-model conclusion is inferred from that failure.
- Authenticated 83 immutable quota-diagnostic files. Rebuilt 550 quota/shuffle
  result rows, exact per-answer cardinalities and analytic quota/length null
  moments. Absolute-Q80 counts match primary A exactly; zero-center votes use
  the frozen donor threshold zero. Historical transductive reference fits and
  exploratory held-out diagnostics are distinguished from the primary family.
- Reconstructed 756 emission-parameter rows, 8,964 conditional correlations,
  48 exposure-weighted aggregates and 30 posterior calibration rows. Full
  answers are centered before class selection. Brier/ECE error is below
  `8.62e-15`; failed fits receive no repaired probabilities. Legacy parameter
  diagnostics use raw MoM emissions, while legacy posteriors use their recorded
  clipped emissions.
- Reconstructed component-oracle flags, all type/category point results,
  4,528 point values and 36 paired intervals exactly. These are evaluation-label
  component diagnostics, not deployable models, additive gains or general
  pooled-PRMScore upper bounds; the all-zero control has zero error recall.
- Independently rebuilt all 672 component contrasts, 112 quantity-change rows,
  identical-posterior mean/median checks and their common-coverage denominators.
  Point and paired-interval errors are zero. Verified 152 saved CPU timing rows
  and 32 timing summaries; failed-stage timing is not imputed as zero.

Receipts and reproducible independent scripts are in this directory:
`INDEPENDENT_SEALED_AUDIT.json`, `INDEPENDENT_COMPENSATED_BRIDGE_AUDIT.json`,
`INDEPENDENT_QUOTA_AUDIT.json`, `INDEPENDENT_CALIBRATION_AUDIT.json`,
`INDEPENDENT_ORACLE_AUDIT.json`, and `INDEPENDENT_COMPONENT_AUDIT.json`.

## Findings resolved before delivery

Baseline coverage/eligibility is required by every protected comparison;
label-source bytes and group/fold identities are authenticated; canonical
prediction exceptions leave missing dependencies/counts; B is described as
one selected triple; live logs are excluded from the quota seal; clean/zero
exploratory strata suppress the misleading half-macro PRMScore; and fresh
replay refuses source, sealed identity or runtime drift before importing the
driver or creating a destination. All seven new synthetic replay guard tests
pass, including rehashed-code-freeze drift and existing-directory preservation.
The root separately reports 55 existing error-count tests passing.

The final HTML and short findings accurately describe the negative protected
result, partial cohorts, quota sensitivity, permutation instability, calibration
and dependence limits. Partial points are explicitly annotated; local links and
assets resolve. Reproduction distinguishes the core A–D replay from the separate
supplementary historical lineage and integrated report renderer.

## Claim limits

All evidence is internal development validation on previously inspected data.
Intervals condition on saved OOF fits and do not include the entire learning
procedure or historical search. Latent orientation, conditional voter/step
independence and correspondence to semantic error remain assumptions. The
bounded family is closed without claiming that all label-free count estimation
is impossible or selecting a new quota using truth.
