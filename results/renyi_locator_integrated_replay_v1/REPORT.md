# Renyi locator integrated replay v1 — results

Status: **COMPLETE / REVIEW PASS**

The independently reconstructed minimum-regret candidate is not promoted. Both its score stream and the current reference are bitwise identical to the Experiment-3 archives (`max_abs=0`) over 13,769 answers and 145,597 steps.

| Method | PB all-8 | PB raw exact | PRMB within | pooled OOF | PRMScore |
|---|---:|---:|---:|---:|---:|
| `VE1-q15/raw` | 37.4749% | 33.1607% | 0.753436 | 0.722305 | 0.634412 |
| `VE1-q50/scale` | 37.6077% | 33.5885% | 0.750297 | 0.720623 | 0.634340 |

Candidate minus current: PB +0.1328 pp (95% CI [-0.6749, +0.9370]); PRMB-within -0.003139 (95% CI [-0.004875, -0.001398]).

The PRMB loss exceeds the frozen .002 noninferiority margin. Final integrated recommendation: retain q15 H0lim/VE0/VE0.75/VE1 in natural units, Top10 per view, equal step mean, and tail15 whole-answer Top10 gate q=.33.

This is a development replay; external/new-model confirmation is still required.
