# Joint feature inclusion plus BOCPD, 2026-09-17

Authorized continuation: first attribute gate/locator errors, then test feature
selection inside Joint, permitting changed feature count/readout; add BOCPD.
Frozen-score audit found Joint1193 final hits,293 gate-only misses,2428
locator-only misses,528 both,935 clean false alarms. Its 75-hit deficit against
innovation5 is entirely locator changes under the shared gate (200 gained,
275 lost). PRMB is ungated. Prioritize locator fitting; gate remains frozen.

One mechanism: alternate checked Joint fit and backward inclusion updates.
Initial source-fold consensus groups stay fixed. Fit exposes loadings A with
one global column and one group-local column per group. Conditional contribution
is the loss of diagonal entries of A' C^-1 A from deleting a feature, normalized
per active factor then averaged. C is observed covariance with fixed1e-4
diagonal shrinkage for this calculation only. Drop the lowest contributor with
at least two remaining group members; refit Joint, preserving all fit guards.
Try at most three deletion candidates if fit validity fails. Stop at min-size or
43 removals. No protected feature; H1 is only an external orientation reference.

Automatic selection: the last state before any initial factor's retained
information falls below95%, evaluated against the initial A and retained C.
This is a label-free reconstruction surrogate, not a correctness guarantee.
Also report predeclared retained counts40/30/20/12/8 for broad50 diagnostically,
using the nearest available count >= requested if path stops. No label-selected
winner is presented as automatic selection. Exact/near duplicates and irrelevant
noise receive explicit synthetic checks of the criterion.

Banks: B50, B50+BOCPD residual, B50+noreset residual. Historical predictor
archive contains innovation5 + .25 standardized signed residual. Subtract the
identical innovation5 baseline then answer-standardize to obtain a pure residual
channel without repeating the baseline. Fixed Top10, hazard1/32, reset-before-
observation Gaussian predictor; noreset is its matched supporting control.
This is one scalar residual view per predictor, not five separately re-extracted
primitive residuals. Both use whole-answer normalization (offline).

Each bank: full Joint, automatic sparse Joint, full equal, equal on selected
features, Continuous L-SML. Fixed five source folds; train on other sources only.
Readout for remaining bank features is still per-stream Top10. Fitting is hybrid
pooled, not answer-only. Same frozen Tail15 whole-answer Top10-mean PB gate,
cell-midrank>=.33; PRMB within AUC ungated. Historical innovation5 and BOCPD
correction scores are separate controls. No gate change or new predictor training.

Four primary contrasts x2 endpoints: sparse-full within B50 and B50+BOCPD;
sparse BOCPD minus sparse B50; sparse BOCPD minus sparse noreset. 10,000 paired
source-group bootstrap draws,99.375% intervals. Other comparisons descriptive.
Full13769-answer development evidence, failures/fallback/missing counts reported.
Synthetic duplication tests assess the selector separately from full real-bank
quality. Do not claim all redundancy problems solved by model-fit validity.
