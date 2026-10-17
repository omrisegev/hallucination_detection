# Full corrected historical Joint and graph comparison

This extends the first historical comparator panel with the actual Joint
variants previously tested by Claude. It is a full-development comparison,
not a new tuning sweep. Small runs check implementation fidelity and cost only.

The population, v3 labels, canonical source groups, outer/inner folds, active23
telemetry, deterministic 60,000 fit-token cap, orientation, B0 readout, nested
PB threshold calibration, metrics and uncertainty contract are exactly those
in `HISTORICAL_FUSION_REFIT_V3.md`. All 13,769 model-answer rows remain included.
These are explicitly pooled-training controls, with label-calibrated PB
decisions, alongside the primary answer-only anchors. No claim of an
end-to-end unsupervised historical PB decision rule is made.

| Arm | Historical mechanism |
|---|---|
| internal_cont | Internal feature groups, continuous L-SML |
| internal_joint | Joint covariance fit, hierarchical weight map |
| internal_joint_gate050 / gate100 | Soft DUFS gates, strength .5 / 1.0 |
| internal_joint_liu010 / liu050 | Joint model-inverse map plus feature-graph Laplacian, lambda .1 / .5 |
| internal_joint_diag010 / diag050 | Joint model-inverse map plus diagonal penalty, lambda .1 / .5 |
| internal_joint_modelinv_lam0 | Exact unregularized model-inverse reference |
| permctl_graph_internal_joint_liu010 | Same .1 penalty with permuted graph nodes |

The diagonal variants are penalty controls, not an additional graph geometry.
`internal_joint` and `internal_joint_modelinv_lam0` use different weight maps;
the former is not the lambda-zero ablation of the LIU map.

Call the original `fit_v2_arms` with these nine requested rows and retain its
automatically computed graph-permutation control. Preserve the historical
random seed schedule, including the gate-permutation draw that precedes the
graph-node permutation. Do not replace graph permutation with feature-column
permutation, or substitute the current answer-only Joint implementation.
Other returned equal/family/gate controls may be used for integrity checks;
they are not silently added as newly selected candidates.

The grouping learner tries K=(3,4,6,8), requires groups of at least three
features and at least .95 admissible leave-one-answer-out partitions. Preserve
its exact historical fallback to provenance groups when blocked and report
those fallbacks. Preserve historical admission of model fits; report exposed
convergence flags. This reproduction does not add a new Jacobian guard or
claim that historical admissibility alone proves statistical identifiability.
Any future stricter variant must be separately named and compared.

## Execution and evidence

First replay the saved GSM8K-Q8 outer0 weights, top-10 step means, step maxima
and full-answer detectors, including the separate lambda-zero amendment file.
Bind the preflight reference hashes and runtime. No correctness labels or
pilot performance ranking enter this preflight.

An execution-only acceleration replaces repeated sklearn sparse-contingency
ARI calls on tiny integer feature partitions with dense integer counts and
the same pair-confusion formula. It is installed only on the private imported
historical module and restored after each fit. The historical files and
sklearn are unchanged. Exhaustive binary partitions through length five,
random 23-feature partitions and fallback cases match sklearn exactly; the
complete historical fold replay matches all 40 arrays and grouping/gate
diagnostics exactly. The local full-replay timing was 431.87 seconds before
and 180.89 seconds with the acceleration; machine contention was not held
fixed, so this is an observed runtime comparison, not a controlled speedup
guarantee. `PREFLIGHT_FAST.json` binds the arithmetic/replay evidence and code.

Then execute the same 245 corrected fits as the first panel. Recompute the
training preparation and require exact agreement with the saved first-panel
fit indices, means, standard deviations and medians. Reuse that panel's five
controls without refitting or recalibrating their scientific recipes. Save
combined per-fold arrays with both control and Joint arms, preserving the
source control hashes. Inner and outer training groups remain disjoint from
the current outer test groups throughout.

Each job is checkpointed atomically and resumes by manifest/hash verification.
One CPU worker is the initial local execution setting; an eight-hour invocation
cap stops after a complete fit and requires resuming that same frozen run.
Do not restart a live fit because a progress file has not changed. Grouping
can take much longer than IU fitting. CPU/cluster expansion requires source,
dependency and score replay checks before accepting cross-platform artifacts.

The complete report joins 19 original answer-only anchors, the five corrected
historical controls and these ten Joint arms. Retain full coverage/failure
tables, common-valid PRMB comparisons and every PB row in its denominator.
Add direct LIU .1/.5 contrasts against model-inverse lambda0 and .1 against
the node-permuted graph, so a model-inverse improvement cannot be attributed
to graph geometry. The .5 arm has no matched .5 permutation here; .1 permutation
does not establish a .5 geometry effect. Record the distinction explicitly.
Also register diagonal .1/.5 versus model-inverse lambda0, gate .5/1.0 versus
hierarchical Joint, and hierarchical Joint versus internal CONT. These eight
mechanism contrasts are exploratory, not multiplicity-adjusted confirmation.

The complete benchmark still requires other historical feature contracts,
DUFS-LIU and dedicated localizers, full answer-only sampling and the newer
answer-only shortlist. This panel neither closes those families nor chooses
a publication winner. Cached data remain exposed development data; untouched
confirmation after method lock is still required.
