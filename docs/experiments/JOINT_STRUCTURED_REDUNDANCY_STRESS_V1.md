# Next verification: approximate replicas and structured nuisance

Steps404-405 preserve the complete benchmark under exact aliases and independent
Gaussian noise while restoring the previous automatic Joint quality. Do not
generalize from these easier additions to arbitrary extra feature families.
Freeze the resulting sparse-membership + information-refinement algorithm.

Next full matched verification, with no penalty/retention/readout tuning:
1. Add15 approximate copies: original standardized rank-probability columns
   plus0.05 times independent answer-standardized Gaussian noise, then center
   and scale each new column within its answer. One-step answers zero.
2. Add15 structured nuisance channels independent of original data/outcomes:
   .8 times a shared AR(1) signal (coefficient.8, stationary Gaussian start)
   plus.6 times independent Gaussian innovations per channel; standardize the
   shared trajectory, private channels and final mixtures within each answer.
   One-step answers zero. This produces a correlated family with persistence.

Noise seeds derive from SHA256(kind:414170:uid), first8 bytes little-endian.
Source-fold fitting, complete13769 answers, same non-digit gate/labels, same
BOCPD51 base. Reuse base scores exactly. Report all native fit failures, selected
new/original channels, BOCPD retention, group structure, source-fold runtime and
full PB/PRMB within metrics. No feature-name-dependent exclusion rule.

Two primary comparisons (each new bank vs frozen sparse-refined base) x2
endpoints,10000 source-group bootstrap,98.75% intervals. Same practical margins
PB-1pp / within-.002 AND native13769. Keep historical strong references and
simple equal/Continuous controls for context where fitted. Exact invariance is
not expected for approximate copies; empirical quality preservation is tested.

This file freezes the next verification question and perturbations. It does not
claim a run has started. Failure should lead to a new declared model change,
not retrospective changes to perturbation strength or its seed.
