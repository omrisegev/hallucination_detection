# RBM logit/readout isolation: complete

13,769 answers; 32 configurations; both saved banks and initial/trained states. No refit. Code freeze 5d18172de, base 7575cb237. Gate, labels, groups and folds unchanged. All 24 original score/metric references replayed; all 32 arms have full coverage. PRMB within-answer AUC uses 6,030 eligible mixed-label answers.

| Method | Token score / readout | PB macro % | PRMB within AUC | PRMB pooled AUC | PRMScore |
|---|---|---:|---:|---:|---:|
| RBM with shared diagonal variance | Original score / First near maximum | 28.6549 | 0.724524 | 0.688809 | 0.606195 |
| RBM with shared diagonal variance | Original score / Original maximum | 35.8069 | 0.733045 | 0.704727 | 0.625301 |
| Entropy | Original score / First near maximum | 36.1900 | 0.730610 | 0.702850 | 0.625743 |
| Entropy | Original score / Original maximum | 35.4444 | 0.730111 | 0.702664 | 0.625426 |
| Initial RBM, 12 features | Logit / First near maximum | 19.4917 | 0.685853 | 0.651257 | 0.568460 |
| Initial RBM, 12 features | Logit / Original maximum | 20.1812 | 0.689104 | 0.650662 | 0.567796 |
| Initial RBM, 12 features | Posterior / First near maximum | 34.6411 | 0.749493 | 0.691035 | 0.607681 |
| Initial RBM, 12 features | Posterior / Original maximum | 36.2946 | 0.746325 | 0.690110 | 0.605313 |
| Initial RBM, 6 features | Logit / First near maximum | 20.5556 | 0.703844 | 0.664658 | 0.582502 |
| Initial RBM, 6 features | Logit / Original maximum | 21.7316 | 0.706096 | 0.664152 | 0.581753 |
| Initial RBM, 6 features | Posterior / First near maximum | 33.6510 | 0.746425 | 0.695323 | 0.614689 |
| Initial RBM, 6 features | Posterior / Original maximum | 35.9662 | 0.744068 | 0.695046 | 0.613085 |
| Step length control | Original score / Original maximum | 33.6944 | 0.618136 | 0.586199 | 0.527878 |
| Random control | Original score / Original maximum | 20.2750 | 0.498480 | 0.501535 | 0.497941 |
| Trained RBM, 12 features | Logit / First near maximum | 36.3998 | 0.748781 | 0.706328 | 0.624156 |
| Trained RBM, 12 features | Logit / Original maximum | 36.2712 | 0.745204 | 0.705849 | 0.622215 |
| Trained RBM, 12 features | Posterior / First near maximum | 27.5159 | 0.711865 | 0.678211 | 0.573077 |
| Trained RBM, 12 features | Posterior / Original maximum | 36.3750 | 0.738702 | 0.710417 | 0.629276 |
| Trained RBM, 6 features | Logit / First near maximum | 34.7246 | 0.737987 | 0.700110 | 0.622670 |
| Trained RBM, 6 features | Logit / Original maximum | 34.9708 | 0.735636 | 0.700004 | 0.622233 |
| Trained RBM, 6 features | Posterior / First near maximum | 31.0950 | 0.739427 | 0.701811 | 0.630811 |
| Trained RBM, 6 features | Posterior / Original maximum | 36.2017 | 0.735982 | 0.708110 | 0.630749 |
| RBM with shrinkage | Original score / First near maximum | 31.3405 | 0.742075 | 0.702877 | 0.629840 |
| RBM with shrinkage | Original score / Original maximum | 35.8711 | 0.738630 | 0.707997 | 0.629544 |
| Varentropy, top 15 | Original score / First near maximum | 36.9464 | 0.738363 | 0.710396 | 0.627000 |
| Varentropy, top 15 | Original score / Original maximum | 35.9610 | 0.737786 | 0.710138 | 0.625781 |
| Varentropy contributions + equal weights | Original score / First near maximum | 36.5705 | 0.748872 | 0.700354 | 0.613818 |
| Varentropy contributions + equal weights | Original score / Original maximum | 35.5980 | 0.746980 | 0.699988 | 0.612812 |
| Varentropy contributions + IU-PCR | Original score / First near maximum | 36.6546 | 0.748978 | 0.710555 | 0.623241 |
| Varentropy contributions + IU-PCR | Original score / Original maximum | 35.3498 | 0.746824 | 0.710261 | 0.622689 |
| Varentropy, top 50 | Original score / First near maximum | 36.4366 | 0.743826 | 0.715920 | 0.632903 |
| Varentropy, top 50 | Original score / Original maximum | 35.6755 | 0.742465 | 0.715783 | 0.632777 |

Primary paired contrasts (logit/near minus posterior/max):
- rbm6_primary: PB -1.4771 pp, 97.5% CI [-2.7656, -0.2263]; within AUC +0.002005, CI [-0.00030372397362131827, 0.004403351576107275].
- rbm12_primary: PB +0.0247 pp, 97.5% CI [-1.2677, 1.3213]; within AUC +0.010079, CI [0.007569001932379963, 0.01268326135889674].

10,000 paired canonical-source-group draws. Remaining contrasts use descriptive95% intervals; PRMScore and pooled AUC are descriptive point estimates. PRMScore q=.8 is recomputed using other folds. Complete cell results, denominators, raw peaks, gate suppression and thresholds are in METRICS.json and PB_CELLS.csv.

Mechanism: bank6/bank12 PB near-set fraction falls43.96%/53.25% to22.43%/24.36%. Logit/near recovers425/732 and577/882 prior posterior/near lost successes. In404/548 recovered cases the original maximum is unchanged and the near-set shrinks. None of the425 bank6 recoveries has collapsed token pairs; only55 of577 bank12 recoveries do. This supports score compression and aggregation as mechanisms beyond finite-precision ties. It does not prove sigmoid is the only bottleneck.

Against posterior/max, trained bank6 gains261 and loses334 gated successes; bank12 gains342 and loses322. Bank12 raw exact peaks decline1456 to1429, while suppressed correct peaks decline244 to197. Thus gated hits rise1212 to1232; the macro F1 change is only+.0247pp. Do not describe this as a clear improvement in first-error localization. Q4 improves37.1185 to37.5285; Q8 falls35.6315 to35.2711.

Training matters under logit/max: initial/trained PB21.7316/34.9708 (bank6),20.1812/36.2712 (bank12). This is a matched readout result, not evidence that training beats the strongest initial posterior. Bank12 logit/near versus Var15/IU/near: PB delta-.2548pp,95%CI[-1.1260,.6154]; within AUC delta-.000197,CI[-.002296,.001930]. No general fusion winner was established.

Review PASS: separate arithmetic for all32 metric bundles and group-separated thresholds;432 manually sorted top10/score vector checks on27 fixed cases;110152 full near-max vector checks;80000 interaction identities across bootstrap draws. This is separate code review in the same session, not an external scientific replication.

Next: discuss the single position-conditioned candidate described in NEXT_STEPS.json. The prior feature-reliability reversal motivates it; residual correlations alone do not. Stop here before defining or training a new model. All results are development evidence. No HTML.
