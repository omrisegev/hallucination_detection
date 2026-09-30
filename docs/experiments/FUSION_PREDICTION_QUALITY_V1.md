# Prediction views inside IU/Joint: bounded quality test v1

Step 312, 2026-09-07. Freeze before new quality evaluation. Development only;
the 110 answers were already exposed. No publication winner can be confirmed
on this set. Fusion remains the core; AR(1) is an added measurement component.

## Question and fixed scope

Do the nine prediction-error columns audited in Step311 improve our current
IU/Joint localization method on both PRMBench and ProcessBench, beyond matched
simple fusion and simple residual controls? Use all 110 unchanged Step306
input identities and corrected source groups. Reuse the exact frozen Step311
moment/context augmented matrices. No new inference or parameter search.

Keep all 77 Step310 quality arms as exact external frozen score references,
not copies of all their window arrays in each new artifact. Their complete
metrics are replayed. Older58 and cross-answer experiments remain separate
history context, not paired algorithm comparisons.

For each answer choose the bank from its ORIGINAL dual route: context only
for `context_joint`, otherwise moment. That same bank is used by every new
core and every residual control. New fit validity never switches banks.
This is the existing same-answer routing algorithm, not a label-selected
bank. It still incurs original Joint eligibility fitting in a fresh deploy.

## Exactly 21 new arms

Three fixed residual families: AR(1), last observation, EMA32. Seven cores
per family: equal, IU, Joint zero graph, Joint graph0.1, Joint permuted graph,
equal graph0.1, equal permuted graph. Total98 arms including77 references.

All use the exact original normalization/orientation and two-component IU
recipe. Joint retains minimum-three groups, K in3/4/6/8, four chronological
blocks, held admissibility .95, ARI tiebreak, seed2026090601, five starts and
5000 sweeps. Enforce convergence, multistart PASS, full global Jacobian rank
and condition<=1e8. Save C/v/u and groups, including failed-fit diagnostics.
No pair-group relaxation or data/label-based feature pruning.

Use condition100 for all native maps, matching the existing frozen cap100
anchor rather than comparing changes in both features and inverse cap.
Graph DUFS seeds0/1/2, epochs120, k7, lambda0.1, exactly as before. The graph
is computed even if Joint fails, to supply equal-graph controls. Original
per-answer graph permutation seed/namespace is reused across residual kinds.
Equal-graph uses identity C and uniform v with the same trace-matched inverse.
An equal zero-graph map equals ordinary equal fusion. These are adaptations.

## Failures, coverage and readout

Only failure of the Joint MODEL fit invokes augmented-IU fallback in that
same bank/family. The three Joint-family arms then have identical IU maps.
A numerical failure of an otherwise valid Joint inverse or graph construction
stays a failed head. A GMM/readout failure also stays failed and never triggers
fallback. No clean decision or low score changes routing.

Keep original width8 grid, nonoverlap fitting rows, end-anchored score window,
token overlap averaging, official-step maximum and native answer-only GMM
no-error decision followed by argmax when open. Keep the original moment-IU
gate diagnostic. Full-answer fusion fitting is offline despite causal AR
predictions. Score failure counts as a miss for full-population PB; PRMB
uses common valid answers in paired comparisons and reports coverage.

Record native Joint eligibility separately from deployed fallback coverage.
Register native-fit subgroup comparisons to distinguish native Joint effects
from fallback. These conditional subsets are diagnostic, not full-population
performance. Comparing with original Joint additionally requires its same-bank
fit valid. Report subset sizes and undefined bootstrap counts.

## Registered comparisons and evidence

74 comparisons are frozen in MANIFEST: 21 new-versus-corresponding unchanged
cores; 27 within-family fusion/graph/control contrasts; 14 AR-versus-last/EMA
contrasts (all seven cores); 12 native-fit subgroup contrasts (four per family).
Each includes pooled PRMB AUROC, mean within-answer AUROC, native PB macro
F1 and fixed-original-IU-gate PB. Use the existing exact source-group stratified
1000-draw bootstrap, seed2026090706. Intervals are exploratory and unadjusted
for74 comparisons. Require both primary endpoints before any favorable
development recommendation; a best per-benchmark row is not one winner.

Source, protocol, tests, exact augmented arrays, original/Step310 scores,
label-byte hashes and evaluator are bound before scoring. Separate evaluator
joins targets only after the complete score freeze. All worker input is
target-free. Existing stages are immutable. Three CPU workers;1200-second
scoring cap with checkpoints. No cluster/GPU work is needed for this stage.

Review must check original-column/routing replay, active features/orientation,
group validity, representative exact Joint/gate refits, every native covariance
construction and inverse, source graph with independent Laplacian/trace algebra,
all step/GMM/fallback decisions, all98 metric bundles, all74 paired point
bundles and representative explicit bootstrap draws. Tests specifically
separate model-fit fallback from readout/numerical failures. Report independent
and shared review components, runtime and limitations honestly.
