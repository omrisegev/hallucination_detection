# Capacity suite — interpretation (Claude, 2026-09-12; read-only over the reviewed run)

Sources: `capacity/METRICS.json`, `capacity/FIT_HEALTH.json`, `capacity/CHECKPOINT.sqlite`,
`capacity/CAPACITY_CONVERGENCE.{csv,json}`, `capacity/CAPACITY_MAXITER_PROBE.json`
(all from `scripts/analyze_rbm_capacity_convergence.py`), `capacity/MODEL_MECHANISMS.csv` and
`MODEL_MECHANISM_SUMMARY.csv` (from the pre-existing `analyze_rbm_completion_mechanisms.py --suite capacity`).
No score, fit or metric of the reviewed run was changed.

## 1. Reviewed task result (unchanged)

| Configuration | PB all-8 % | PRMB within | PRMScore |
|---|---:|---:|---:|
| exact H1, bank6, posterior | 36.2017 | 0.735982 | 0.630749 |
| exact H4, bank6, posterior | 25.3205 | 0.710100 | 0.616757 |
| CD-10 H1, bank6, posterior | 35.9925 | 0.742042 | 0.594137 |
| CD-10 H4, bank6, posterior | 34.4456 | 0.734925 | 0.580452 |
| exact H1, bank12, logit | 36.2712 | 0.745204 | 0.622215 |
| exact H4, bank12, logit | 28.6553 | 0.721693 | 0.601725 |
| CD-10 H1, bank12, logit | 30.5433 | 0.719634 | 0.510910 |
| CD-10 H4, bank12, logit | 30.5041 | 0.718182 | 0.507889 |

Primaries: exact4 − exact1 PB −10.88 pp [−13.10, −8.74] (bank6 posterior), −7.62 pp [−9.49, −5.76]
(bank12 logit); within-answer AUC also falls in both.

## 2. What "nonconverged" means here — an optimization-budget limitation

| | bank6 | bank12 |
|---|---:|---:|
| exact H4 fits stopped by `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT` (maxiter 100) | 13,768 / 13,769 | 13,769 / 13,769 |
| exact H1 nonconverged | 1 | 0 |
| exact H4 final gradient max: p05 / p50 / p95 | 0.0015 / 0.031 / 0.110 | 0.021 / 0.072 / 0.195 |
| NLL gain of exact H4 over exact H1 (nats/token): p05 / p50 / p95 | 0.61 / 0.98 / 1.19 | 0.89 / 1.78 / 2.57 |
| exact H4 fits with worse NLL than exact H1 | 0 | 0 |

Every exact-H4 fit ended at the registered iteration cap with a gradient two to five orders of
magnitude above the gtol of 1e-6. H4 nevertheless fits the density better than H1 on every answer.
So the registered H4 states are **valid finite-budget fits, not converged maximum-likelihood fits**.
CD arms run a fixed epoch budget and carry no convergence flag; they are never called converged.

**Feasibility probe (not a benchmark result):** refitting exact H4 from the same seeded start with
maxiter 1000 on the 27 capacity smoke answers (54 fits, ≤ 1.4 s each): 48 / 54 converge (median
≈ 400 iterations), median further NLL decrease 0.47 nats/token, and the top-10 peak moves in 7 / 54
(logit) and 8 / 54 (posterior) fits. A full-population refit at a larger budget is therefore cheap
(order of hours), but it is a new registered experiment, not something inferred here.

## 3. What the saved H4 representation looks like — three named conditions

Per exact-H4 unit: (a) **saturated** = logit varies (std > 1e-8) but posterior constant (std ≤ 1e-10);
(b) **duplicate** = |corr| ≥ 0.999 with an earlier live unit; (c) **dead** = logit std ≤ 1e-8.

| | bank6 | bank12 |
|---|---:|---:|
| mean saturated units per fit | 0.120 | 0.217 |
| mean duplicate units per fit | 0.004 | 0.004 |
| mean dead units per fit | 0 | 0 |
| answers with 4 / 3 / 2 varying posterior views | 12,262 / 1,367 / 140 | 11,135 / 2,281 / 353 |

Saturation is common (one unit in 9.9% of bank6 fits and 16.6% of bank12 fits, two units in 1.0% /
2.6%); duplicated or dead units are rare. This is representation behaviour of the finite-budget
fits; it is also the exact cause of the depth-suite smoke failures (see the depth amendment).

## 4. Task link (associations only)

PB exact successes, exact H4 versus exact H1, retained readouts:

| stratum | bank6 gained / lost | bank12 gained / lost |
|---|---:|---:|
| all | 285 / 802 | 299 / 701 |
| gradient quartile q1 (best optimized) | 24 / 48 | 35 / 91 |
| q2 | 57 / 144 | 81 / 171 |
| q3 | 90 / 277 | 86 / 189 |
| q4 (least optimized) | 114 / 333 | 97 / 250 |
| 0 saturated units | 256 / 721 | 246 / 578 |
| 1 saturated unit | 29 / 80 | 48 / 118 |
| 2 saturated units | 0 / 1 | 5 / 5 |

Losses are heavier in the least-optimized quartile, but the best-optimized quartile still loses
about two successes for each gain, and fits with no saturated unit account for most losses. So the
H4 deficit at the registered budget is not explained by the worst-optimized or collapsed fits alone.

## 5. Classification for the Stage-1 account

- **Implementation failure:** none found in the capacity suite (review PASS; H1 provenance
  bit-identical; the depth failure is a representation property of the saved H4 states).
- **Optimization limitation:** the registered maxiter-100 budget leaves every exact-H4 fit
  unconverged; the probe shows convergence is reachable cheaply, so the H4 result is conditional on
  that budget. Any larger-budget or restart-based H4 evaluation is a new registered experiment.
- **Negative scientific result (at the registered budget):** four exact hidden units score
  substantially below one on both benchmarks; CD-10 H4 is closer to H1 but not better. None of this
  isolates representational capacity from optimization, and nothing here supports a claim that
  more hidden units are inherently worse or better.
