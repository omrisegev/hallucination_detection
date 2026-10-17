# Linear chronological context reference — registered before evaluation

All 13,769 answers. Original4 and innovation5 banks. Fit a regularized linear
predictor of the current standardized feature vector from the preceding 16
token vectors, observation masks and relative position. Ridge penalty 1;
16,384 observations drawn uniformly by source group, then answer, then token.
Training uses the same source-excluded, separate unlabeled validation-group
split as neural predictors. No correctness labels enter fitting. The fixed
ridge strength does not use validation labels or task results.

Evaluate actual histories and a deterministic within-answer permutation of
the 16 historical vector slots, preserving feature vectors, masks and position.
This preserves the historical set while breaking lag identity. A second control
replaces the historical vectors by zero, keeping mask and position. These are
inference ablations of the same fitted model; they are not separately refitted
null models. Report that limitation, especially under distribution shift.

Readouts: mean signed standardized innovation and mean squared standardized
innovation, each token curve summarized with step Top10. Report standalone and
base + gamma * std(base) * z(auxiliary), gamma .25 and 1, where normalization
is within answer and the base comes from the frozen float64 replay. Identity
gamma 0 is checked. The PB gate stays frozen. Primary contrasts are gamma .25
squared residual vs the corresponding bank base, one per bank (97.5% CIs).
Other readouts are development diagnostics, not independent confirmations.

Fit five outer excluded-fold models and ten pair-excluded models. Pair models
provide source-excluded calibration scores for PRMScore thresholds; PB and
within-answer ranking use outer predictions. Every reported quality metric
uses the complete eligible population. Save group manifests, coefficients,
predictions, fit/validation loss, runtime, score coverage and failure status.
