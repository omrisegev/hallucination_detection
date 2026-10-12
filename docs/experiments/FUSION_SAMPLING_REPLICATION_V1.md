# Observation selection on the fixed current110 - Step317

Decision question: does changing the fitting observations improve the existing
answer-only IU/Joint fusion on BOTH corrected localization benchmarks?
This follows the weak original58 sampling signal, not a search for a winner
among new label-chosen budgets. Freeze this document, code, tests and roster
before scoring. Same previously evaluated development110:24 PRMB/86 PB.
Use v3 raw-verified labels, v2 source-question groups, and retain the original
v1 graph scoring namespace. This is not untouched confirmation.

## Fixed design

Keep original width8 nonoverlapping fitting windows plus the scoring-only
end-anchored window; P27 moment/context bank and original 81/29 route per
answer. Never switch banks in response to a new selector's fit outcome.
Use original moment/context definitions and existing selected-row fit_bank
from fusion_prediction_quality: equal/IU/Joint0/Joint graph.1/permuted graph,
plus equal graph/permuted graph. Condition100 is fixed. Global score sign
uses the declared negative-entropy anchor. Group discovery retains K3/4/6/8,
minimum3, four chronological blocks of selected-row ranks and existing
admissibility/multistart/Jacobian checks. No labels enter any fitted quantity.

Selectors reuse fusion_window_sampling.choose_all exactly: full, uniform,
entropy-risk, transposed DUFS, its probability-permutation control, and direct
window-graph diffusion. For N original rows retain min(N,max(32,ceil(N/2))).
Target-free feasibility:72/110 eligible,38 copy the full references exactly.
Eligible counts:16 PRMB,8 GSM8K,16 MATH,16 Olympiad,16 Omni. All selectors use
the same budget and shared chosen rows for all seven fusion cores. Choices
are positions within original fit_indices, sorted before fitting. The
scoring-only final window must never become an extra independent fitting row.

DUFS transposition gates N windows, with P feature coordinates as graph nodes.
Direct window diffusion graphs N observations. The graph penalty in Joint is
a third, separate operation. Unsupervised selector construction may see all
current-answer fitting rows. Fusion normalization, orientation, covariance,
groups and weights see selected rows only. Keep original graph seed identity
for all selectors; different selected N changes permutation length, not seed.

Every original window remains scored. GMM fits risk on ALL ORIGINAL fitting
windows, matching the first sampling pilot; do not fit it only to selected
high-risk rows. Same overlap-mean token projection, maximum per official step,
and peak conditional on the unchanged no-error rule. Keep the original
moment-IU gate as a diagnostic. Joint MODEL-fit failure alone explicitly
falls back to selected-row IU in the SAME fixed bank. Selector failures,
graph/numeric failures after a valid Joint fit, and readout failures remain
failures. Save native failed-fit evidence and all fallback sources.

Full-grid and all ineligible selector/core outputs copy the seven verified
graph-conditioning anchors exactly.42 displayed selector/core entries include
seven full aliases;35 recipes can create new fits. Keep all107 v3 external
anchors in the same result registry. Entries are not independent methods.

## Diagnostics and evidence

Save selected indices, probabilities, seed Jaccard, time-quartile occupancy,
largest gap, and two fixed within-window two-token-block perturbations.
For context recompute EMA from the perturbed raw answer before selection;
this affects later EMA windows and is disclosed. Stability is descriptive,
not a tuning objective or correctness proxy. Dense features remain computed;
no end-to-end inference or feature-extraction speedup is claimed.

After scoring is frozen, measure error-step token overlap with the selected
FIT windows, especially first errors <=32 tokens; report actual subgroup size
including zero. These are fit-support diagnostics, not sparse prediction
recall, since unselected windows still get scores. Show eligible-only results
beside full110, native coverage/fallback/failure counts, within-answer AUC,
and exact correct decisions gained/lost by PB subset.

Primary endpoints unchanged: pooled PRMB step AUROC with valid-answer counts;
PB mean of four harmonic means of clean/exact-first-error accuracies, all
failures penalized. For comparisons PRMB uses common valid IDs; PB keeps all
selected IDs. 1000 fixed source-group bootstrap draws, exploratory unadjusted
intervals. Original58 corrected results remain a separate contextual panel.
The manifest registers each reduced selector vs full for all seven cores,
matched fusion controls, graph-selector controls, eligible and native scopes,
and comparisons against the strong original permuted equal-graph control.

## Execution and review

Three CPU workers,1200-second scoring submission cap, answer checkpoints,
bounded PermissionError retry for output atomic replace. Terminal failures
are diagnosed; observation timeouts do not authorize restarting live work.
No new inference, no Claude-worktree mutation, no old artifact modification.

Before freezing test actual selectors' axes/budgets/determinism, rank ties,
original vs scoring-only indices, fit-only dependence, explicit fallbacks,
dense GMM support, support projection and exact anchor replay. Afterwards
review raw labels independently of derived labels, original banks/routes,
selected-row normalization/orientation/IU, covariance/weights/graphs, native
versus fallback outputs, dense projection/GMM and all149 metrics/registered
contrasts. Replay representative expensive grouping/Joint/DUFS operations
and explicit bootstraps; disclose any shared kernels. Render an English HTML
report with exported scientific plot, historical context and limitations.
This bounded stage cannot by itself finish the broader research mandate.
