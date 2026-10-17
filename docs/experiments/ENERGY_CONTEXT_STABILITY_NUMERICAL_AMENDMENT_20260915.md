# Numerical amendment, before any diagnostic outcome review

Initial run v1 stopped at fit34/45 (pb_omnimath_q4, excluded fold3).
The exact-face solver's absolute objective improvement deadband1e-13 retained
a boundary solution with KKT residual1.44822632e-7, correctly triggering its
1e-7 acceptance check. Removing that deadband yields residual2.29e-16, with
maximum weight change9.77e-7 and objective improvement7.07e-14. Independent
SLSQP agrees to8.1e-8 in weights. The saved Q is positive definite.

The v2 real-bank module uses the same objective, face enumeration, constraints
and feasibility/KKT acceptance thresholds, with strict objective improvement
instead of the absolute1e-13 deadband. No equal-weight fallback, tolerance
relaxation or hyperparameter change. The original synthetic module is untouched.

Preserve v1's33 completed fits, provenance and numerical failure fixture/report.
Recompute ALL45 fits into energy_context_stability_v2 under new hashes. No v1
fit checkpoints are reused. This amendment changes numerical implementation,
not the frozen scientific protocol. No correctness labels or quality scores
have been opened. Numerical diagnosis scripts retain access to the v1 fixture.
