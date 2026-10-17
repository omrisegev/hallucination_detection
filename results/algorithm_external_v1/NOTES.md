# algorithm_external_v1: notes kept outside the frozen protocol (so its hash stays fixed)

- 2026-09-29, pre-seal review: correction of amendment A1's reason text. The recorded maximum difference over non-constant
  answer-columns is 1.4e-10 (spread computed as E[x^2]-E[x]^2), within the declared 1e-9 bound; the 4.9e-12 figure in the
  amendment came from a diagnostic that masked whole columns rather than answer-columns.
- Pre-seal review (no blocker): the evaluator's seal no longer includes the evaluator's own hash (recorded in
  EVALUATION_PROVENANCE instead); the seal now records the per-cell feature and manifest hashes and the fit / scoring script
  hashes, and the evaluator asserts those scripts are unchanged; non-finite scores are checked before sealing.
- Label-free observation before sealing: the stopping-rule statistic S1 is 0.182-0.278 on every external bank-cell, above
  tau* = 0.1734, so the diagnostic switch never turns grouping on externally (F_SW = F_BASE everywhere); prediction P3 is
  therefore already false. The F_SW - F_BASE contrast is reported as "never switched on", not "no effect".
- The banks include the three digit features (Omri's approval 2026-09-29, beyond the 2026-09-17 digit exclusion); results are
  labelled digit-inclusive.
