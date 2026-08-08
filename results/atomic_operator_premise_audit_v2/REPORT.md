# Frozen atomic-operator premise audit

**Decision: STOP AOG FOR THIS PROXY.**

## Terms

- **Atomic operator:** the Laplacian penalty created by one feature.
- **Proxy:** the registered score computed without correctness labels.
- **Usefulness:** AUROC change when that operator is added to IU-PCR.
- **Spearman association:** whether higher proxy ranks correspond to higher usefulness ranks; 1 is perfect, 0 is no monotone relation, and -1 is reversed.
- **pp:** AUROC percentage points; +0.5pp means AUROC increases by 0.005.

## Primary result

The median within-cell proxy/usefulness association is **-0.312**.
The equal-family mean is **-0.032**, with bootstrap interval [-0.319, +0.249].
The within-cell feature-identity permutation p-value is **0.6905** and the exact eight-family sign-flip p-value is **0.5820**.
Top-proxy operators beat bottom-proxy operators in **3/8 families**.
After nuisance adjustment, the family mean partial association is **-0.077**, interval [-0.290, +0.110].
The selected top-proxy atom changes equal-family AUROC by **-1.178pp**, interval [-2.110, -0.372].
The label-only atomic oracle changes equal-family AUROC by **+0.483pp**, interval [+0.263, +0.709].
Order-invariant unique-value quotient graphs marked **0** cell-feature operators invalid because they had fewer than three unique values.

## Continuation gates

| gate | observed | pass |
|---|---:|:---:|
| all primary cell associations and quartile contrasts are defined | 0 | yes |
| median within-cell Spearman > 0 | -0.312 | no |
| family-bootstrap association lower > 0 | -0.319 | no |
| feature-identity permutation p <= 0.05 | 0.690 | no |
| eight-family association sign-flip p <= 0.05 | 0.582 | no |
| positive top-minus-bottom in at least 6 of 8 families | 3 | no |
| partial-association family-bootstrap lower > 0 | -0.290 | no |
| partial Freedman-Lane p <= 0.05 | 0.995 | no |
| eight-family partial sign-flip p <= 0.05 | 0.730 | no |
| median abs(proxy, ridge-distance Spearman) < 0.8 | 0.386 | yes |
| top-proxy atomic family-bootstrap AUROC lower > 0 | -2.110 | no |
| top-proxy atomic family sign-flip p <= 0.05 | 0.984 | no |
| top-proxy atomic improves at least 14 of 24 cells | 7 | no |
| top-proxy atomic worst loss no worse than -2pp | -3.658 | no |
| oracle atomic family-bootstrap AUROC lower > 0 | 0.263 | yes |

## Practical headroom

| method | cell-macro AUROC | change vs IU-PCR | wins/losses | worst |
|---|---:|---:|---:|---:|
| iu_pcr | 0.7741 | +0.000pp | 0/0 | +0.000pp |
| projected_ridge_lambda1 | 0.7737 | -0.034pp | 9/15 | -0.539pp |
| uniform_atomic_k15_lambda1 | 0.7745 | +0.041pp | 14/10 | -0.462pp |
| top_proxy_atomic | 0.7657 | -0.838pp | 7/17 | -3.658pp |
| oracle_atomic | 0.7785 | +0.447pp | 23/0 | +0.000pp |

The oracle is a label-only headroom diagnostic. It is not a usable method.

## Family diagnosis

| family | cells | proxy association | top-bottom usefulness | partial association |
|---|---:|---:|---:|---:|
| triviaqa | 4 | +0.052 | -0.210pp | +0.120 |
| hotpotqa | 1 | +0.319 | +0.061pp | +0.068 |
| sciq | 1 | +0.533 | +0.425pp | +0.189 |
| nq_open | 1 | +0.407 | +1.007pp | +0.284 |
| squad_v2 | 1 | -0.730 | -1.946pp | -0.667 |
| truthfulqa | 1 | -0.262 | -0.964pp | -0.142 |
| gsm8k | 10 | -0.367 | -0.818pp | -0.220 |
| math500 | 5 | -0.207 | -0.056pp | -0.251 |

## What the proxy components say

| label-free quantity | median cell association | family interval |
|---|---:|---:|
| primary_proxy | -0.312 | [-0.318, +0.240] |
| full_alignment | -0.112 | [-0.162, +0.248] |
| bootstrap_alignment | -0.115 | [-0.154, +0.250] |
| operator_reproducibility | +0.300 | [-0.385, +0.332] |
| rank_change_reproducibility | -0.377 | [-0.398, +0.144] |
| stability_actuation_proxy | -0.226 | [-0.350, -0.094] |
| full_actuation | -0.441 | [-0.483, +0.137] |
| anisotropy | -0.342 | [-0.375, +0.047] |

### Duplicate-threshold diagnostic

This table changes only the full cross-fitted alignment component. It is not a rerun of the complete proxy and cannot replace the registered 0.95 threshold.

| threshold | median cell association | family interval |
|---:|---:|---:|
| 0.90 | -0.021 | [-0.145, +0.284] |
| 0.95 | -0.112 | [-0.164, +0.245] |
| 0.99 | -0.202 | [-0.183, +0.275] |

## Parameter sensitivity

The largest descriptive median association on the frozen grid is -0.108 at k=30, lambda=0.3. This is not a selected replacement for the primary setting.
At 40 subsamples the convergence reference is, by definition, 1.0; earlier checkpoints are shown in `figures/proxy_convergence.png`.

Parameters that can change the mechanism are graph neighbourhood size `k`, Laplacian strength `lambda`, duplicate threshold, and the stability sampling budget. They may be changed only in a newly registered run. If the primary association is absent across the sensitivity grid, more tuning cannot solve the missing target-identification problem.

## Conclusion

The registered premise did not pass. Do not build or tune AOG-IU-PCR from this proxy. The failure means that reproducibility, cross-fitted smoothness, and actuation do not jointly identify correctness-relevant atomic graphs well enough on the existing cells. Continue only if a new source of self-supervision supplies a different, theoretically justified target; do not rescue the result by selecting the best observed k or lambda.

## Audit

The fit completed 24 cells in 5.4 minutes. Source and artifact hashes were verified before labels were read. The 24 cells are retrospective development data, not external confirmation.

Figures: `figures/proxy_vs_utility.png`, `cell_associations.png`, `family_top_bottom.png`, `proxy_convergence.png`, `k_lambda_sensitivity.png`, and `atomic_headroom.png`.
