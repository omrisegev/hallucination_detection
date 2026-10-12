# Graph structure after native Joint conditioning

2026-09-07. Bounded development interaction experiment, not confirmation.

## Question and fixed scope

Does the existing Joint graph lambda .1 add useful localization information
after native conditioning is strengthened, relative to the same conditioned
Joint without a graph and to a node-permuted graph? Keep ALL three previous
caps 30/100/300; do not select the best observed cap. Keep original feature
groups, fit validity, C/v/u and bank routing. Reuse the Step-309 saved fits:
no Joint refitting, no new pair admission, no wider K or graph-lambda search.

Use the same 110 development answers: 24 PRMB, 86 PB over four subsets,
Qwen3-8b, corrected v2 source groups. The full cache was already evaluated.
Access remains one teacher-forced gray-box model pass on fixed official
answers and within-answer fitting of all quantities. The declared negative-
entropy anchor, width-eight features, normalization and temporal mapping
remain. Freeze predictions before this new evaluator opens labels.

## Methods and control for fusion contribution

Retain all 45 preceding arms exactly. Add 24 original-fit native heads:
moment/context/single/dual x condition 30/100/300 x real/permuted graph .1.
At condition1000, reproduce the existing real/permuted graph weights,
window/step scores and native decisions before emitting a new native head.
Reuse original same-answer DUFS gates and graph k=7; use the original v1
scoring identity namespace and exact permutation seeds. Do not share fits
or gate values across answers.

Add eight simple controls: each bank/route with real/permuted graph-smoothed
equal fusion. Define this adaptation by substituting C=I and v=ones/P in
the SAME trace-matched native graph map. At lambda zero it is exactly equal
fusion after the unchanged orientation/scale normalization. With lambda .1,
the graph term is PSD with trace P, so the condition number is bounded by
1+.1P <=3.7. All tested caps therefore produce the same equal-graph control;
report it once, verifying this property. It is not a paper-exact method.
This tests the Joint covariance/loading contribution beyond the graph
mechanism applied to simple aggregation. Retain ordinary IU/equal anchors.

Equal-graph controls cover banks even when their original Joint was invalid.
Where original DUFS gates were not computed, compute the same answer-local
recipe (seeds 0/1/2, 120 epochs) for these controls only. The bank route is
still original. Native single/dual arms use moment IU on the same old
fallback answers. Equal-graph routes use the corresponding bank's equal-
graph score, including moment on those fallback answers, matching the old
routed-equal convention. Readout failures and no-error decisions never route
to another method. No hidden fallback.

There are 77 total arms. For each of four families and three caps, compare
the real native graph with same-cap zero, same-cap permutation, original
condition1000 real graph, same-bank/routed IU and equal, and the real equal-
graph control; compare the native permuted graph with permuted equal-graph
(84 contrasts). Compare each equal-graph real control with plain equal and
its own permutation (8). At each cap compare dual native real graph with
context equal, corresponding single native graph, and context native graph
with moment native graph (9). Total: 101 preregistered exploratory contrasts.

## Evaluation, execution and review

Keep the existing step maximum and native GMM no-error/peak readout. Use the
fixed-original-IU gate only as a diagnostic. Report common-valid PRMB AUC,
within-answer AUC, all-population PB native/fixed-IU harmonic macro-F1,
coverage, failures, subset counts and runtime. Retain all historical anchors
and the older 58-answer panel separately. Use 1000 source-group bootstrap
draws, seed 2026090706, unadjusted exploratory intervals and undefined counts.
No consistent winner claim without matched evidence on both primary tasks.

Before freezing, test synthetic Laplacian/inverse algebra and the equal-
control condition-cap invariance; real historical graph replay on all three
original route cases without Joint refitting; and failed-head/readout route
preservation including the distinct equal-control fallback convention.
Maximum three CPU workers, 600-second scoring cap per invocation, checkpoint
completed answers and finish in-flight work. Store parent sources, scores,
original gates, raw-input and opaque label hashes. Independent review must
reconstruct graphs/Laplacians, inverse/step/native-GMM maps, simple controls,
route inheritance, all metrics/point contrasts and explicit bootstraps.
Document reused graph-builder/DUFS/GMM kernels. Produce HTML/Markdown reports.

The fusion core remains IU-PCR/Joint L-SML. Supporting geometry, temporal,
IMM/KalmanNet/flow and sampling tracks, corrected-fold multi-answer refits,
full comparator coverage, untouched confirmation and historical24 transfer
remain open regardless of this one graph/conditioning interaction result.
