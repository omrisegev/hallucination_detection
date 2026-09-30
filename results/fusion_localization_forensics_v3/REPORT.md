# Where our fusion localizers fail — corrected-label forensics

Step314. Frozen scores; no new detector or claimed algorithm gain. Review PASS.

All 110 answers / 71,385 tokens / 1,112 steps pass exact token-ID and span replay. Three original scalar streams match the extracted input exactly. There are 16 legitimate separator tokens outside official spans. All1,760 method/answer score projections replay within2.67e-15.

## The two bottlenecks

| Fusion | Clean decisions correct /33 | Raw peak exact /53 errors | Exact after gate /53 | Exact peaks hidden by gate | False alarms /33 |
|---|---:|---:|---:|---:|---:|
| dual__iu | 18 | 20 | 12 | 8 | 15 |
| dual__cond100_graph010 | 17 | 18 | 13 | 5 | 16 |
| dual__equal_graph_perm | 19 | 18 | 11 | 7 | 14 |
| ar1__iu | 13 | 16 | 10 | 6 | 20 |
| ar1__graph010 | 12 | 16 | 10 | 6 | 21 |

## Boundary ties are real, but a limited part of this failure

IU and graph100 each have16/86 PB answers with tied highest-scoring steps; those ties have shared window support. IU hits20/53 first errors with its actual peak; the correct step occurs somewhere in the tied top set in22/53. Joint gives18/53 and21/53. A perfect label-using tie choice with the existing gate raises IU PB from30.16% to31.77%. This is an oracle diagnostic, not a deployable gain.

Both original IU and Joint miss31/53 first errors using their actual peaks. Even their combined tied top sets miss28/53. This restricts choosing among those existing peaks, not the potential of full-trajectory fusion or new measurements.

## What changed with prediction residuals

AR+Joint changes16 binary gate decisions and13 peak locations. Only2/13 changed peaks had an old top-two gap <=0.1 score SD. It loses10 previously correct complete decisions and gains2. Four previously correct raw peaks are lost and two gained. Its weaker PB result is therefore not explained solely by almost-tied maxima or solver failure. This is descriptive attribution to changed outputs, not a proven cause inside the weights.

## Component-conditional oracles, not new methods

| Fusion | Actual PB | Perfect binary gate; same peak | Perfect locator; same gate | Perfect top-tie choice; same gate |
|---|---:|---:|---:|---:|
| dual__iu | 30.16% | 56.74% | 47.23% | 31.77% |
| dual__cond100_graph010 | 30.22% | 51.67% | 46.21% | 32.32% |
| dual__equal_graph_perm | 31.32% | 53.69% | 50.39% | 31.32% |
| ar1__iu | 23.50% | 47.86% | 43.86% | 25.35% |
| ar1__graph010 | 18.01% | 46.11% | 44.44% | 20.90% |

These columns use the answer labels deliberately. A perfect gate cannot fix a wrong peak; a perfect locator cannot fix false alarms or a closed gate. They are not achievable-performance forecasts.

## Decision for the next bounded experiment

Keep IU-PCR, Joint graph100 and the simple permuted-graph control. Do not make boundary tie-breaking or a finer lambda grid the main next experiment: the measured recoverable set is small and both gate and risk localization fail. Inspect whether existing single-pass token confidence carries correctness evidence that our uncertainty-oriented fused trajectory suppresses, then freeze one supporting feature/readout change with matched controls. Audit earlier implementations before calling it new. No new candidate is selected here.

## Limits and review

The review re-read raw pickle metadata and used the independent existing alignment API, incidence-matrix score projection, pairwise AUC, PB counts and all48 oracle endpoint reconstructions. All98 corrected metric bundles,16 native summaries and6 transition bundles pass. Shared tokenizer, raw metadata reader and saved fusion scores are disclosed. This validates downstream alignment for these110 answers, not the original model logit-position slice or every top-K-derived feature. No new inference or multi-answer fitting.

The broader corrected historical bridges/refits, full comparators, IMM/LOCA/Flows/KalmanNet adaptations, untouched confirmation and historical24 transfer remain open.
