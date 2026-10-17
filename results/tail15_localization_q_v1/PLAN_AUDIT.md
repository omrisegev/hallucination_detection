# Gate experiment plan audit

This audit separates the original proposal from the experiments actually run.

| Planned component | Status | Evidence / outcome |
|---|---|---|
| Single-feature screening | Complete, expanded | 11 token signals x 11 readouts = 121 candidates on all 15 math cells. |
| Whole-answer mean | Complete | Included for every signal. |
| High quantile | Complete | q75, q90 and q95 were included. |
| Top10 mean | Complete | Included; selected for the final tail15 gate. |
| Maximum rolling mean, window 8 | Complete | Included. |
| Four answer-time regions | Complete | Quarter means were included. |
| Start-to-end slope | Complete | Included. |
| One threshold q | Complete | Math selected q per candidate; the final fixed tail15 Top10 gate then searched one uniform q=.01-.99 on ProcessBench exact localization and froze q=.33. |
| Near-duplicate removal | Complete | 21 candidates were removed at absolute Spearman correlation >= .995. |
| Forward selection | Complete | Forward equal-mean selection produced a three-feature numerical winner; it was not promoted because its gain did not justify the added layer. |
| Equal/simplex/logistic fusion | Complete | All three arms ran; learned simplex/logistic arms used leave-one-cell-out cross-fitting. None was promoted. |
| Fully nested grouped CV of the entire feature/readout/fusion selection pipeline | Not run | Per the development-search decision, all available math cells were used for method development. The final q used all PB development labels. External/new-model data must provide the unbiased confirmation. |
| Two-locator robustness contract from the first proposal | Partially replaced | Integration compared the original static locator and the selected q15 locator. The earlier proposed VE0.75/local-IU pair was not the selection contract. |
| Cumulative integration of accepted decisions | Complete | Start 36.1674%; locator-only 36.6201%; gate-only 36.8759%; complete frozen development method 37.4749%. |
| External generalization | Pending | No independent new-model/data confirmation has run yet. |

The complete development specification is recorded in `FROZEN_METHOD.json`.
The numerical results and uncertainty intervals are in `METRICS.json` and
`REPORT.md`.
