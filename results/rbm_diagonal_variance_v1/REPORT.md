# RBM conditional diagonal variance: full development result

13,769 answers; original six-moment bank; same normalization, mean6 orientation, frozen entropy q0.3 gate, Top10 readout, v3 labels/source folds. Code freeze57f0f7edf. Scoring271.16s. Full metric arithmetic reviewPASS; saved-state reviewPASS (55,076 records,108 independent mixture/step replays on27 real answers).

The original assumption audit found predicted feature variance median1.36154 while observed variance is1 after normalization. Median relative covariance error.60196. This establishes model mismatch, not its causal effect on localization.

Candidate: exact Gaussian RBM, shared learned diagonal conditional variance D, floor0.05 on VARIANCE, penalty0.1*sum(log D)^2. Both candidate and fixed-D control start from the same fitted original RBM and receive100 additional optimizer iterations. No coefficient/floor search.

| Method | PB macro % | Within AUC | PRMScore |
|---|---:|---:|---:|
| Original RBM | 36.2017 | 0.735982 | 0.630749 |
| Fixed variance + matched extra training | 36.2017 | 0.735982 | 0.630749 |
| Learned diagonal conditional variance | 35.8069 | 0.733045 | 0.625301 |
| Before learning | 35.9662 | 0.744068 | 0.613085 |
| Varentropy15 | 35.9610 | 0.737786 | 0.625781 |
| Varentropy50 | 35.6755 | 0.742465 | 0.632777 |
| Order6 RBM saved reference | 36.3750 | 0.738702 | 0.629276 |

Primary diagonal minus continued control:10000 canonical-source paired draws;97.5% intervals.
PB delta -0.3947pp, CI [-1.0278,+0.2255].
Within-AUC delta -0.002937, CI [-0.004973,-0.000890].

No demonstrated PB benefit; PRMB within ranking regresses.84 PB successes gained,105 lost. Additional training alone gives unchanged headline metrics. No candidate promotion or further sweep.

| Diagnostic median | Original | Diagonal candidate |
|---|---:|---:|
| Variance RMSE | 0.441614 | 0.100681 |
| Relative covariance error | 0.601956 | 0.538323 |
| Relative offdiagonal error | 0.686650 | 0.687037 |

Conditional variance median.7310, minimum.09064; no floor hits.25/13769 diagonal fits lack optimizer convergence; all finite outputs retained, no failed/collapsed output. Original fit has one nonconverged case; continued fixed-D fits all converge. Comparisons describe this fixed100-additional-iteration procedure, not a guaranteed global maximum-likelihood solution.

Interpretation: fixing conditional variance improves marginal variance agreement but leaves substantial dependence mismatch. Better generative fit did not improve hallucination localization. This does not reject all Gaussian/diagonal or RBM models. No claim that the floor caused failure; it was never reached.

## Paper requested by Omri

Shaham et al., ICML2016, A Deep Learning Approach to Unsupervised Ensemble Learning. https://proceedings.mlr.press/v48/shaham16.html ; Lemma4.1 on thirdPDFpage: binary Dawid-Skene conditional-independence distribution and single-hidden-unit binary RBM have a bijective parameter map. This does not establish equivalence of our Gaussian-moment implementation or of its optimizer to L-SML/IU. Historical Step141 statement was too broad. Lemma4.2 consistency requires its stated assumptions and MLE, not just a converged arbitrary fit.

Per-cell results SUMMARY.csv; all contrasts and diagnostics METRICS.json; prior audit summary ASSUMPTIONS_SUMMARY.json. Full raw assumption audit stays in results/rbm_covariance_assumptions_v1/AUDIT.json. No global24 experiment or new HTML.
