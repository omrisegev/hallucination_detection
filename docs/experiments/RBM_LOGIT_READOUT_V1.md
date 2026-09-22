# Saved RBM score/readout isolation

Base: 7575cb237. Branch: codex/rbm-logit-readout-v1.

Full corrected v3 population: 13,769 answers. No fitting. Replay trained and
initial six/twelve-feature states. Preserve representation, normalization,
orientation, source groups, folds, labels and step spans.

Compare posterior versus oriented raw logit, each followed by top-10 token
mean and either original argmax or the frozen first_near_max (.25 population
standard deviations; epsilon 1e-6 times max(sd,1e-12)). Short steps use all
tokens. Eight new arms accompany the 24 saved diagnostic references.

Keep the saved entropy q=.3 gate. Recompute PRMScore q=.8 thresholds using
other folds only. Report full denominators, missing scores and failures.
Nonfinite input/logit aborts as an integrity error; absent saved models remain
explicit missing candidate scores, never a baseline fallback.

Primary contrasts: trained RBM6 and RBM12 logit/near minus posterior/max.
10,000 canonical-source-group bootstrap draws, 97.5% intervals for these two;
95% descriptive intervals otherwise. Compute the score/readout interaction
inside every draw. Diagnose near-set changes, early/late errors, recoveries,
gate suppression, numerical ties, and training versus initialization.

Smoke cases are shortest, median and 95th-length answers in each cell, only
for mechanics/runtime. Performance conclusions require the full population.
Verify all saved posterior scores exactly, scalar manual top10 cases,
independent metrics, fold separation and bootstrap interaction identities.

Interpret jointly with the previous variance, residual and context-reliability
diagnostics. No acceptance threshold, conditional model, CD, refit, added
units/features or HTML. Stop for discussion after full results. All findings
remain development evidence; prior research selection uncertainty is excluded.
