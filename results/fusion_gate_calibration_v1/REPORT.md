# Fixed scalar gate calibration: completed simulation

Neither readout passed the frozen advancement screen. No real-answer gate change is promoted. Full matched benchmarking is now the priority.

384 evaluation trials, 39 calibration draws for each of fitted-rho and known-rho generators. The source is a declared stationary Gaussian AR process, plus registered +3SD midpoint-jump controls. The true-rho reference is simulation-only. These are simulation gate-open counts, not semantic-error rates.

| N | rho | Jump | Readout | Native /32 | Fitted /32 | Known /32 | Same native detections retained | Screen |
|---:|---:|:---:|---|---:|---:|---:|---:|---|
| 16 | 0.0 | False | raw | 6 | 2 | 3 | 2 | True |
| 16 | 0.0 | False | imm | 6 | 1 | 1 | 1 | True |
| 16 | 0.6 | False | raw | 5 | 3 | 2 | 3 | True |
| 16 | 0.6 | False | imm | 11 | 1 | 1 | 1 | True |
| 16 | 0.9 | False | raw | 6 | 2 | 2 | 2 | True |
| 16 | 0.9 | False | imm | 16 | 4 | 3 | 4 | True |
| 16 | 0.0 | True | raw | 5 | 4 | 2 | 4 | True |
| 16 | 0.0 | True | imm | 25 | 9 | 11 | 9 | False |
| 64 | 0.0 | False | raw | 1 | 2 | 2 | 1 | True |
| 64 | 0.0 | False | imm | 8 | 2 | 2 | 2 | True |
| 64 | 0.6 | False | raw | 1 | 1 | 2 | 1 | True |
| 64 | 0.6 | False | imm | 8 | 2 | 2 | 2 | True |
| 64 | 0.9 | False | raw | 11 | 6 | 4 | 6 | False |
| 64 | 0.9 | False | imm | 20 | 2 | 5 | 2 | True |
| 64 | 0.0 | True | raw | 11 | 16 | 21 | 10 | True |
| 64 | 0.0 | True | imm | 32 | 32 | 32 | 32 | True |
| 256 | 0.0 | False | raw | 0 | 3 | 3 | 0 | True |
| 256 | 0.0 | False | imm | 14 | 4 | 4 | 4 | True |
| 256 | 0.6 | False | raw | 1 | 5 | 5 | 1 | True |
| 256 | 0.6 | False | imm | 9 | 4 | 4 | 4 | True |
| 256 | 0.9 | False | raw | 8 | 1 | 4 | 1 | True |
| 256 | 0.9 | False | imm | 12 | 2 | 2 | 2 | True |
| 256 | 0.0 | True | raw | 32 | 32 | 32 | 32 | True |
| 256 | 0.0 | True | imm | 32 | 32 | 32 | 32 | True |

Raw fails the N64/rho.9 stationary cell: native11 openings, fitted6; the fixed screen required at most half (5.5). IMM fails short-jump retention: 9/25 original detections retained, below75%. The screen is a fixed engineering decision, not statistical confirmation.

The replacement gate can both close and reopen compared with native BIC. This experiment covers scalar readout processing, not the complete feature-fusion fitting procedure. The fitted source model may absorb a true jump as dependence; the positive controls expose a sensitivity loss.

Review PASS: {'sources': 30336, 'lag_regressions': 384, 'mixture_likelihoods': 60672, 'pvalue_decisions': 1536, 'direct_vector_imm': 36, 'actual_mixture_refits': 72}. All sources, likelihood/rank algebra checked; representative direct-vector IMM and shared-kernel GMM refits. No missing mixture fits. Same-session review, not external review.

[Frozen protocol](../../docs/experiments/FUSION_GATE_CALIBRATION_V1.md) | [Results](RESULTS.json) | [Review](REVIEW.json) | [Frozen outputs](FROZEN.json)
