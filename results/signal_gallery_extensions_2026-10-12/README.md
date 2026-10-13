# Gallery score and dependence extensions

Run with `.venv/bin/python scripts/build_signal_gallery_extensions.py` from the repository root.
The destination refuses reuse to preserve its locked protocol and prediction seal. To reproduce
in a fresh destination, change only the script OUT path in an isolated copy and retain its hash.

`extension.json` is the display payload. `FEATURE_PRMSCORE.csv` and `FAMILY_PRMSCORE.csv`
contain the score summaries. `PAIRWISE_CONDITIONAL.npz` stores empirical pair matrices,
class sample counts and per-feature error rates; columns match the original gallery order.
`PREDICTIONS.npz` retains all fixed source-OOF decisions and donor Q80 thresholds.
The official evaluator was replayed for every feature, overall and for all nine subtypes.

Higher feature values retain the previously frozen label-free error orientation. No evaluated
labels are used to choose signs, thresholds, features, models, or peer sets. The diagnostics
condition on valid/error truth separately, pool steps within each displayed context and do
not adjust for within-answer/source dependence, answer heterogeneity, position, or length.
No confidence interval, independence test, or external validation is claimed. Small pairwise
dependence does not establish joint conditional independence. Constant error margins and the
MS absent erroneous class remain undefined, rather than being represented as zero.

Family lines are equal mean member-feature scores, not the performance of a new fused family.
Overall PRMScore is pooled macro-F1 over noncontrols; eight error-type scores use pooled macro-F1
and MS uses valid-step F1. No AUC is defined for MS. NR/NCL use the declared validity-head
fallback, preserving the previous evaluator adaptation. This is an internal development
diagnostic. No GPU, package installation, Git publication, or frozen-package mutation.
