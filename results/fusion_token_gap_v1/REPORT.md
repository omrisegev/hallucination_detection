# Provided-token gap inside our fusion — Step315

Reviewed development result: no consistent improvement. Retain original IU and Joint graph references.
Same110 answers, v3 labels, v2 groups, exact original graph seed namespace. P27, width8, original81/29 bank route.

| Arm | PRMB AUC | Within-answer AUC | PB F1 |
|---|---:|---:|---:|
| dual__iu | 0.68130595 | 0.76881338 | 30.16% |
| gap__iu | 0.68018702 | 0.76881338 | 28.82% |
| dual__cond100 | 0.65381234 | 0.75465314 | 25.47% |
| gap__joint0 | 0.64917679 | 0.76080845 | 27.61% |
| dual__cond100_graph010 | 0.65545077 | 0.75348127 | 30.22% |
| gap__graph010 | 0.64993606 | 0.74078004 | 30.22% |
| dual__equal_graph_perm | 0.69225543 | 0.78360256 | 31.32% |
| gap__equal_graph_perm | 0.69089674 | 0.77659037 | 29.48% |
| gap_scalar | 0.65756873 | 0.66158970 | 0.00% |
| surprisal_scalar | 0.65900735 | 0.67763446 | 0.00% |

Native Joint 100/110 versus107; ten explicit fit-failure fallbacks. All9 final outputs valid.
Gap-IU loses one correct PB decision; gap-Joint graph changes one incorrect prediction and preserves all exact-success indicators.
Scalar gap and surprisal controls both flag all33 clean PB answers and have0% PB F1 under this fixed GMM readout.
Raw top50 audit: 71057/71385 provided tokens retained; 328 outside. Omitted probabilities are not independently recoverable from top50.
Review PASS; complete107 rows,25 comparisons, failure/coverage counts and five explicit bootstraps in REPORT.html.
No new inference, untouched confirmation or overall goal completion.
Next: bridge earlier unique readout/sampling/regularization methods to v3 labels and v2 source groups before reusing their conclusions. Multi-answer refits and the full research mandate remain open.
