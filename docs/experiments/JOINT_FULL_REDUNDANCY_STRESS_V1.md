# Controlled full-population redundancy/noise test, 2026-09-17

Previous goal turn made verified progress (Step400). Remaining question: does
the alternating Joint inclusion mechanism preserve localization when additional
features are irrelevant or duplicates? Synthetic copy recovery is insufficient.

Base: frozen B51 = broad50 Top10 plus pure cached BOCPD residual. Controls use
the exact Step400 source folds, source-excluded fits, non-digit H1 orientation,
whole-answer Tail15 Top10-mean gate / cell midrank>=.33, ungated PRMB within AUC.
All13769 answers/145597 steps; no correctness label enters perturbation or fit.

Two separate stresses, fixed before fitting/evaluation:
1. Append exact copies of all15 probability-rank step columns, producing66.
2. Append15 independent Gaussian step columns, producing66. Random generator
   seed is SHA256(401170:answer_uid) first8 bytes little-endian; each column is
   centered/scaled inside the answer, single-step answers zero. The noise cannot
   depend on outcomes or on the ordering/chunking of answers.

Rediscover groups for each stress from training sources using the unchanged
four-training-fold deletion consensus K={3,4}; do not hand-assign copies to
groups. Alternating checked Joint fit/deletion uses the frozen95% information
retention stop, permits at most P-8 removals (same minimum8 endpoint as B51),
stops after the first threshold crossing. Same five initial starts/seeds/guards.
If discovery or the initial fit fails, report native failure and use explicit H1
fallback for end-to-end metrics; this does NOT count as successful filtering.

Arms per stress: Joint full, Joint auto, Continuous L-SML, equal-full and
equal-selected. Base replay controls are loaded frozen. Measure complete metrics,
per-cell results, selected new features/retained duplicate pairs, group changes,
score/peak stability, coverage and runtime. Reuse historical innovation5 and
BOCPD correction references as context, not as results of this new mechanism.

Four primary paired contrasts x2 endpoints: each stress versus the original
bank for Joint-auto and Joint-full.10000 source-group bootstrap draws,99.375%CI.
Predeclared practical preservation margins: PB loss no greater than1 percentage
point and within-AUC loss no greater than.002; require the lower CI endpoints
above those margins for BOTH outcomes and full native coverage. This is an
engineering noninferiority check, not universal robustness or exact invariance.
Separately report exact prediction invariance for the copy condition.
No threshold, feature set, noise seed or gate tuning after results.

Next action depends on this evidence; do not quietly add a noise filter or
deduplicate beforehand and present it as the same Joint selector.
