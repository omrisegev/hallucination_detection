# Renyi locator feature-bank experiment v1 — results

Status: **COMPLETE / REVIEW PASS**
Population: 13,769 answers, 145,597 steps. Development comparison; not external confirmation.

## Questions and answers

1. **Does native H1 help the locator?** No. Across 12 matched pairs it changes PB by -0.225 percentage points on average and PRMB-within by -0.001053; only 1/12 PB and 1/12 PRMB comparisons improve.
2. **Does q15 Hinf help?** Not uniformly. It creates the single best PB and PRMB arms in different solvers, but its mean effects are -0.116 PB points and -0.001262 PRMB. Neither isolated winner is a joint improvement.
3. **Should VE1 q15 be replaced by VE1 q50?** No for the uniform locator. Its mean effect is -0.290 PB points and +0.000130 PRMB. PRMScore improves in all 12 matched pairs, but that does not offset the PB loss under the frozen joint objective.
4. **Does a deployable fusion capture the complementary errors?** No tested fusion does. The label-using union localizes 554 additional error answers beyond the baseline, but no label-free arm improves both primary benchmarks. This is evidence for future conditional selection, not a deployable result.
5. **Does the full candidate help with the frozen gate?** No. The independent replay gives PB delta +0.133 percentage points and PRMB-within delta -0.003139; the latter exceeds the frozen .002 loss margin.

## Headline comparison

| Method | PB all-8 | PB raw exact | PRMB within | PRMB fold | pooled OOF | PRMScore | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `VE1-q15/raw` | 37.4749% | 33.1607% | 0.753436 | 0.722708 | 0.722305 | 0.634412 | retain |
| `VE1-q15+Hinf/scale` | 37.8901% | 33.6110% | 0.748489 | 0.716352 | 0.716060 | 0.630791 | best PB only |
| `VE1-q15+Hinf/raw` | 37.1320% | 32.8681% | 0.753681 | 0.722356 | 0.721955 | 0.634496 | best PRMB only |
| `VE1-q50/scale` | 37.6077% | 33.5885% | 0.750297 | 0.720920 | 0.720623 | 0.634340 | min-regret; rejected by replay |

## All 24 arms

| Feature bank / solver | PB all-8 | PRMB within | PRMScore | regret |
|---|---:|---:|---:|---:|
| `VE1-q15/IU` | 37.2406% | 0.745703 | 0.587001 | 3.989 |
| `VE1-q15/raw` | 37.4749% | 0.753436 | 0.634412 | 2.076 |
| `VE1-q15/scale` | 37.7829% | 0.749893 | 0.633273 | 1.894 |
| `VE1-q15+Hinf/IU` | 37.3275% | 0.744080 | 0.587344 | 4.801 |
| `VE1-q15+Hinf/raw` | 37.1320% | 0.753681 | 0.634496 | 3.790 |
| `VE1-q15+Hinf/scale` | 37.8901% | 0.748489 | 0.630791 | 2.596 |
| `VE1-q15+H1/IU` | 36.9627% | 0.745992 | 0.589177 | 4.637 |
| `VE1-q15+H1/raw` | 37.2081% | 0.752985 | 0.634633 | 3.410 |
| `VE1-q15+H1/scale` | 37.6818% | 0.748844 | 0.631578 | 2.418 |
| `VE1-q15+H1+Hinf/IU` | 36.7269% | 0.744002 | 0.589581 | 5.816 |
| `VE1-q15+H1+Hinf/raw` | 37.0545% | 0.752530 | 0.634561 | 4.178 |
| `VE1-q15+H1+Hinf/scale` | 37.5474% | 0.746385 | 0.628356 | 3.648 |
| `VE1-q50/IU` | 36.6306% | 0.746411 | 0.590752 | 6.297 |
| `VE1-q50/raw` | 37.1851% | 0.753116 | 0.634795 | 3.525 |
| `VE1-q50/scale` | 37.6077% | 0.750297 | 0.634340 | 1.692 |
| `VE1-q50+Hinf/IU` | 36.7300% | 0.745744 | 0.593230 | 5.801 |
| `VE1-q50+Hinf/raw` | 37.1297% | 0.753151 | 0.634581 | 3.802 |
| `VE1-q50+Hinf/scale` | 37.5050% | 0.749115 | 0.631551 | 2.283 |
| `VE1-q50+H1/IU` | 36.6582% | 0.746283 | 0.595585 | 6.159 |
| `VE1-q50+H1/raw` | 37.0916% | 0.752621 | 0.634774 | 3.993 |
| `VE1-q50+H1/scale` | 37.4616% | 0.748790 | 0.632498 | 2.446 |
| `VE1-q50+H1+Hinf/IU` | 36.0853% | 0.743650 | 0.597191 | 9.024 |
| `VE1-q50+H1+Hinf/raw` | 37.0731% | 0.752020 | 0.634701 | 4.085 |
| `VE1-q50+H1+Hinf/scale` | 37.3877% | 0.746381 | 0.629096 | 3.650 |

## Final integrated recommendation versus the original start

The recommendation is unchanged: q15 H0lim/VE0/VE0.75/VE1, natural-unit per-view Top10 equal locator, plus tail15 whole-answer Top10 gate at q=.33.

| Metric | Original start | Recommended integration | Delta |
|---|---:|---:|---:|
| PB all-8 | 36.1674% | 37.4749% | +1.3075 pp |
| PRMB within | 0.751115 | 0.753436 | +0.002321 |
| PRMB fold AUROC | 0.721815 | 0.722708 | +0.000893 |
| PRMB pooled OOF | 0.721407 | 0.722305 | +0.000898 |
| PRMScore | 0.634805 | 0.634412 | -0.000392 |

The PB comparison includes both accepted changes (locator and gate). The PRMB comparison isolates the locator because PRMB has no answer-level gate. The slight PRMScore decrease prevents claiming universal improvement; the gains are PB and AUROC gains.

## Replay evidence

- Both independently recomputed score streams are bitwise identical to their frozen archives (`max_abs=0`).
- Candidate PB delta 95% group-bootstrap CI: [-0.006749, +0.009370].
- Candidate PRMB-within delta 95% group-bootstrap CI: [-0.004875, -0.001398].
- Joint L-SML was deliberately deferred; the baseline four-view bank is structurally inadmissible and the expensive extension is only warranted if a cheaper arm first succeeds.

![Experiment 3 Pareto plot](PARETO.png)
