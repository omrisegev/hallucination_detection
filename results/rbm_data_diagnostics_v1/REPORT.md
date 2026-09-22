# RBM saved-data diagnostics

Full cached development data. No model retraining. First-near-max is a research readout, not an independently confirmed improvement.

| Method | PB old | PB near | PRMB within old | PRMB within near | PRMScore near |
|---|---:|---:|---:|---:|---:|
| RBM with shared diagonal variance | 35.807% | 28.655% | 0.73304 | 0.72452 | 0.60620 |
| Entropy | 35.444% | 36.190% | 0.73011 | 0.73061 | 0.62574 |
| RBM before learning, twelve features | 36.295% | 34.641% | 0.74632 | 0.74949 | 0.60768 |
| RBM before learning, six features | 35.966% | 33.651% | 0.74407 | 0.74642 | 0.61469 |
| Longest step control | 33.694% | 33.694% | 0.61814 | 0.61814 | 0.52788 |
| Random step control | 20.275% | 20.275% | 0.49848 | 0.49848 | 0.49794 |
| RBM, twelve features | 36.375% | 27.516% | 0.73870 | 0.71186 | 0.57308 |
| RBM, six features | 36.202% | 31.095% | 0.73598 | 0.73943 | 0.63081 |
| RBM with weight shrinkage | 35.871% | 31.340% | 0.73863 | 0.74207 | 0.62984 |
| Varentropy, top 15 | 35.961% | 36.946% | 0.73779 | 0.73836 | 0.62700 |
| Varentropy contributions, equal weights | 35.598% | 36.571% | 0.74698 | 0.74887 | 0.61382 |
| Varentropy contributions, IU-PCR | 35.350% | 36.655% | 0.74682 | 0.74898 | 0.62324 |
| Varentropy, top 50 | 35.676% | 36.437% | 0.74246 | 0.74383 | 0.63290 |

Primary readout changes:
- RBM6: PB change -5.107 percentage points; 97.5% CI [-7.11297687 -3.15431445]. Within-answer AUC change 0.00344; CI [0.000882521017370259, 0.006002624373852161].
  Readout gains/losses: 456/732; learning gains/losses under near readout: 228/390.
- RBM12: PB change -8.859 percentage points; 97.5% CI [-11.21090586  -6.5220978 ]. Within-answer AUC change -0.02684; CI [-0.03240932395411469, -0.021482371856090858].
  Readout gains/losses: 448/882; learning gains/losses under near readout: 212/573.

Evidence: EVIDENCE.csv/JSON; full per-answer arrays BANK6/12_DIAGNOSTICS.npz; ERROR_CASES.csv; original/new per-cell tables PB_CELLS.csv.
No new architecture is authorized by a model mismatch alone. Consult the reviewed task-linked priority assessment before new training.


The first-near-max rule is not a safe default for the current RBM scores.
Claude's entropy/Varentropy result replays; transfer of that improvement to RBM
fails. Preserve the original RBM readout and all negative results. Keep near-max
as a candidate for Varentropy and contribution IU, not a universal replacement.
Every lost RBM success moved earlier; no gate decision changed. Broad near-max
sets cover about44%/53% of PB steps for RBM6/12 versus21% for Varentropy50.
The sigmoid may compress useful score separation, but that mechanism needs a
saved-weight logit/readout ablation; it was NOT run in this diagnosis.

The most task-relevant new modeling evidence is a reversal of feature reliability.
On the SAME1672 eligible PRMB answers, selected surprisal trails entropy in the
early half by0.1061 AUC, then exceeds it in the late half by0.04236. Both features
use step means; no aggregation mismatch. The paired reversal is0.14846,
95%CI[0.12459,0.17245]. This motivates conditional fusion, not a claim that such
a label-free learner already exists or improves the benchmark.

Class variance is different on4030 eligible PRMB answers, often LOWER on error
steps. Residual correlation and serial dependence are also real model mismatches.
However, residual dependence is present in successes too. Do not equate fitting
these statistics better with better hallucination localization. The observed
near-zero training residual mean is expected from fitted location parameters;
its difference from non-refitted synthetic means is not evidence for more units.

| Priority | Smallest useful direction | Current decision |
|---|---|---|
| 1 | Same weights, inspect pre-sigmoid scoring/readout | Next proposal; no new fit |
| 2 | A single regime-dependent fusion mechanism | Supported by matched reliability reversal; discuss after1 |
| 3 | Separate conditional variances | Statistical evidence; task link and label-free recovery open |
| 4 | One sequential fusion component | Serial structure exists; task value not isolated |
| 5 | Four hidden units/shared noise | Residual mismatch alone is insufficient |
| 6 | Restarts/CD, then deeper layers | Current fits mostly converge; no restart evidence yet |

NEXT_STEPS.json/CSV records effects, uncertainty, coverage and alternative
explanations. EVIDENCE.json/CSV contains232 descriptive diagnostic contrasts;
RELIABILITY_REGIMES.json contains the same-aggregation matched follow-up.
All results remain development evidence. No new model was trained, no HTML
was generated, and the separate DUFS run was left unchanged.
