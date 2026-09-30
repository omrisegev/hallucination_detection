# Fused trajectory readout pilot v1

2026-09-07. Registered before running this version. Retrospective development;
the parent labels and diagnostic peak results have already been inspected.

## Question and immutable parent

Can a chronological interpretation improve localization while keeping our
IU-PCR / Joint L-SML fusion and its answer-level error decision fixed?
Reuse all 58 IDs, official spans and saved scores from
`results/answer_localization_representation_pilot_v1`. Verify its release,
preparation and score hashes. No feature extraction, normalization, group,
gate, weight or graph refitting. No inference or other-answer training.

Core arms: moments27_local8 equal, IU, Joint lambda zero, Joint graph 0.1,
Joint permuted graph; entropy_mean_w8 is a diagnostic control. Keep the
historical representation pilot as an immutable bridge. A result on these
already exposed answers is not confirmation.

## Fixed error gate; six location/readout arms

Every arm retains the parent's valid/invalid fit and its frozen binary
error decision (`prediction != -1`). A readout cannot change a clean decision
to error or the reverse. An unavailable readout counts as a miss in both
classes, including when the parent gate says clean. This experiment isolates
localization given that decision. It cannot establish better clean detection.

1. `parent_first`: exact original step scores and prediction replay.
2. `parent_peak`: original step scores; if the gate says error, choose their
   maximum (earliest tie). This directly tests first crossing versus peak.
3. `hold_peak`: use only nonoverlapping fit-window scores, fill each window
   with its score and extend the last score over the remaining tail. Take
   official-step maxima and their argmax. This controls the temporal methods'
   removal of the extra overlapping tail window.
4. `hmm_entry`: reuse the existing reversible shared-variance HMM from
   `latent_state_localizer.py`, fitted to this one nonoverlapping window
   sequence. Seeds 11/23/37, max_iter 120, tolerance 1e-5, variance floor 1e-3,
   occupancy floor .02, separation .25, transition floor 1e-6, seed locator
   agreement .80. Preserve its guards; require the selected candidate to stop
   before the iteration cap. No fallback for an invalid HMM. PRMB ranking is
   high-risk-state occupancy; PB locator is high-risk-state entry posterior
   (including initial state occupancy at window zero). These are distinct,
   predefined outputs of the same model, not label-selected alternatives.
5. `imm_level`: interacting bank of two scalar random-walk Kalman filters,
   F=H=1, process variance Q=(.01 R, R), row-stochastic transition matrix with
   self probability .95, initial mode probabilities (.5,.5), initial state
   zero and variance one. Mix state means AND covariances before prediction,
   then update filter and mode probabilities from likelihood. R is
   clip((median(abs(diff(x)))/(.67448975 sqrt(2)))^2, .05, 1), a within-answer
   heuristic. The posterior fused level is both ranking and locator curve;
   a single slow Kalman filter is included as an additional matched control
   `kalman_level` (making seven arms in total).
6. `bocpd_rise`: a reset-before-observation Gaussian product-partition
   adaptation, known observation variance one, segment mean prior N(0,1),
   hazard 1/32 windows, no run-length truncation (at most 256 windows here).
   Each reset branch consumes its current observation. PRMB ranking is the
   posterior expected segment mean. PB locator is reset probability times the
   positive part of the signed standardized continuation innovation. At the
   initial window use hazard times max(x[0]/sqrt(2),0); it is an explicit
   boundary heuristic, not a correctness probability or an assumed clean
   first step. Record reset probabilities and predictive quantities.

Map all temporal curves using the same hold policy as `hold_peak`, then
official-step max. No chronological model is claimed to make the complete
pipeline causal: fusion, HMM smoothing and R estimation use the whole answer.

## History and fidelity

HISTORY Step 246 already tested a pooled, token-level IU-initialized HMM:
PB F1 .3003 vs ordinary IU .3167, with worse absorbing control. The new use
changes fit scope and sampling support; do not claim HMM-after-fusion is new.
The existing ordinary Kalman final-answer precursor also lost to its controls.

Primary references: [Adams and MacKay](https://arxiv.org/html/0710.3742v1),
[FilterPy IMM implementation](https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/IMM.html).
The original BOCPD paper's after-observation boundary convention can give
P(r=0)=constant hazard. That alone is not an implementation bug. Our old
`temporal_models.bocpd_gaussian` uses a prior predictive for a reset yet leaves
its sufficient statistics unupdated by the current observation; it must not
be treated as the verified reset-before-observation implementation here.
Keep its historical outputs unchanged. The new recursion needs an exact
short-sequence partition-enumeration test.

## Evaluation, attribution and stopping

Freeze this protocol, all new code/imported kernels and every new output
before this version's evaluation phase. Scores never load label files or
the parent's EVALUATION.json. Evaluation rejoins frozen source labels by ID.
Retain PRMB pooled step AUROC plus within-answer AUC and PB equal-subset
macro F1 with the parent's failure policy. Paired uncertainty uses source
groups, 1,000 draws and seed 2026090706; all intervals are exploratory.

Compare each temporal method with hold_peak on the same fusion core; compare
each learned fusion with equal under that same readout; graph versus lambda
zero/permutation; IMM versus single Kalman. Report the parent-peak change,
unchanged binary gate, strict availability, first-step bias and gate-imposed
PB ceiling. Do not select a winner from different available PRMB populations.

Compute cap: 58 answers, seven readouts, six cores; at most three CPU workers.
Checkpoint per answer and resume verified missing outputs only. Review HMM
probabilities/guards, IMM covariance mixing, BOCPD exact enumeration, span
mapping, no-error invariance and independent endpoints before reporting.
Stop this bounded experiment at a factual report; full benchmark replay,
gate improvement, feature/hyperparameters, sampling, LOCA/KalmanNet/flows,
untouched confirmation and 24-cell transfer remain in the active program.
