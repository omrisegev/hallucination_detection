# Step408: settle zero-row membership before global-group exclusion

One frozen ordering change, motivated by the verified Step407 iid fold2 failure.
In each original sparse fitting round compute the union of coordinates with
v OR u nonzero over converged starts. If any coordinates leave, remove only those
zero rows and rediscover groups in the next round. Apply the Step407 global-group
exclusion ONLY when this ordinary row support is unchanged in that round.
Keep local-only rows in groups connected globally in any converged start.
At most three monotone rounds, same seeds/calibration/solver, followed by the
unchanged checked Joint and95% information refinement. No minimum-group or
identification guard changes. No protected feature or label input.

Use the same five full banks and all13769 held answers as Step407. Fit scope is
hybrid source-fold training, not answer-only. Fixed non-digit gate, H1 orientation,
Top10, BOCPD, labels/source IDs, native versus explicit H1 failure accounting.
The approximate-copy case remains a required test, not excluded to make this pass.

Reuse is allowed only by a complete training-decision equivalence proof: a valid
Step407 model has the same input/seed and EVERY stored sparse round produces
identical active coordinates under the staged rule. Then subsequent discovery,
fit, refinement and weights are unchanged, so copy the audited fit with revised
decision telemetry and a hash of its parent. If any round differs, or the parent
failed, execute the full new fit for that fold. No reuse based on metric quality.
Independently audit the proof, factor/refinement algebra and held scores for all
25 bank/fold combinations. Report fresh fits separately from equivalent replays.
Cap25 fresh fits; no additional solver retries. Keep old files/code frozen.

References: Step407 signal, Steps405/406 old Joint, each same-bank Continuous and
equal, historical innovation5/BOCPD. Primary six paired comparisons are new iid
vs Step407 iid, new base vs Step407 base, and four additions vs new base.
Two endpoints,10000 source-group draws, confidence1-.05/12=99.583333%.
Preservation requires full native13769 and interval lower bounds >-.01 PB and
>-.002 within. Baseline preservation also requires max score delta<=1e-11.
Report changes against all saved references; no after-result selection of N,
threshold, group K, rank summary or new variant. Existing development data only.

The claim being tested is whether ordering fixes the diagnosed iid failure while
preserving the structured-noise repair. Success does not solve approximate
duplication, prove robustness across random seeds, or establish an over40 result.
