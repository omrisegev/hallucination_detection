# Fusion context-bank pilot v1

Date: 2026-09-07. Adaptive development on the same frozen 58 answers.
Question: can a context-aware feature bank improve IU/Joint localization and
Joint's admissible grouping, and does the restricted K roster hide useful
partitions? Fusion remains the method; no other-answer or correctness fitting.

## Source audit and representation

Reuse the linear smoothing idea already implemented in
`spectral_utils/unified_causal_iu.py` (`_OneChannelDSP.update` and
`CausalFeatureBankState.update`). That historical full pipeline and
`unified_causal_subset_search.py` explicitly permit supervised development of
rosters, signs and references. None of their fitted artifacts or selected
seven-stream roster enters this experiment. This is an answer-local adaptation,
not a replay of Unified-28 or a claim that EMA is new.

Keep all nine primitive streams of the current moment bank. At token t,
compute the raw value, EMA with span 8, and EMA with span 32. Use alpha=2/(s+1)
and initialize each EMA to that answer's first observed value, rather than the
historical zero initial state. This preserves constant inputs and avoids an
artificial zero-origin startup ramp. Average each of these three token streams
over each existing eight-token window: 27 feature definitions, base-major.
The level column is exactly the parent's window mean. Names retain the
`entropy_series__level` anchor. There is no learned horizon or selected subset.

Fit on the original non-overlapping windows only. Retain the extra tail window
for dense scoring and the same overlap-mean token / span-maximum step map.
Normalizing/signing/grouping/weight fitting use only this answer, exactly as in
representation v1. Remove nonfinite, constant and exact affine duplicate
coordinates. Retain the declared negative-entropy global sign anchor.
Smoothing adds temporal context, not observations or independent evidence.
Feature extraction is prefix-based; full-answer fitting and step scoring are
still offline. This is not a separately learned second trajectory-fusion stage.

## Two factors, fixed budget

Compare the existing `moment` bank and new `context` bank. Joint uses either
the original K roster {3,4,6,8}, or all integers from 3 through floor(P_active/3).
The expanded roster has at most seven candidates here, versus four originally.
Do not interpret larger K as automatically better. Keep four chronological
leave-block-out fits, consensus/held minimum group size 3, held admissible
fraction >=0.95, and the same ARI selection/tie rule. No correctness labels,
artificial feature duplicates or forced partitions. Record every candidate.

Keep Joint's native model-inverse lambda-zero, graph lambda 0.1, and permuted
graph control; five starts, 5,000 sweeps, seed 2026090601. Require convergence,
multistart PASS, profiled global Jacobian full rank/condition <=1e8. The inverse
uses the same PSD projection and condition-1000 ridge. No hidden IU fallback.
Graph DUFS gates remain three seeds (0,1,2), 120 epochs, k=7. Share the graph
within a feature bank across grouping rosters. Reuse the parent's within-answer
gates on the unchanged moment bank when available. The graph-node permutation
is exactly the parent's SHA of `release_id/cell/row_id/moments27_local8`, also used on
the context bank to retain a common permutation. When the two K rosters select
identical groups, cache/reuse the same fitted Joint model within this run.

17 arms: five moment parent controls, five context/legacy-K cores (equal, IU,
Joint zero, graph, permuted graph), three moment/expanded-K Joint cores, three
context/expanded-K Joint cores, and the entropy parent. Moment and entropy
parents replay frozen window/step scores exactly. Equal/IU do not depend on K.

## Readout, evaluation and attribution

Primary: original one/two-component GMM BIC rule on non-overlapping scores,
then the maximum-risk official step if its gate opens; otherwise -1. Same
rule, but a changed feature score can change the gate. Keep fit validity
separate from native readout validity. Invalid fits always remain failures.

Diagnostic: use the **same parent IU binary gate** for every core and bank,
while keeping each method's own step ranking and fit validity. This common
within-answer gate has full parent coverage and avoids silently assigning a
clean decision where the old Joint fit failed. It is distinct from each
method's own old gate and is not a newly selected primary candidate.

Keep PRMB pooled AUROC, mean within-answer AUROC, PB four-subset macro harmonic
clean/exact-error accuracy, coverage, errors and runtime. Paired PRMB contrasts
use common valid IDs; PB uses all fixed IDs with failures. Include exact-peak
hits among erroneous PB answers independently of gating. Report the K/active-P,
group-size, admissibility, rank, condition and off-diagonal fit distributions.
Do not infer better localization from an answer-specific scale change alone.

31 registered native contrasts: context vs moment (five cores); expanded vs
legacy K within each bank (three Joint cores each); context vs moment under
expanded K (three); IU/equal, Joint0/IU, graph/Joint0, graph/permutation and
graph/IU in context legacy (five); Joint0/IU, graph/Joint0, graph/permutation
and graph/IU in each expanded-K bank (eight); and four unchanged moment
controls (IU/equal, graph/IU, graph/Joint0, graph/permutation). Total 31.
Use 1,000 source-group bootstrap draws, stratified by cell, seed 2026090706.
Also show paired within-answer and common-IU-gate diagnostics under these same
draws. Intervals are exploratory, unadjusted, and undefined class-missing draws
retain explicit counts. No label-selected winner or confirmation claim.

The bootstrap can use sufficient pair-count matrices for speed only after
matching explicit source-group resampling and the parent metric computations.
Freeze code/protocol/parents before scoring; freeze scores before evaluation.
Five scientific contracts cover EMA closed form/initialization/causality,
constant/affine behavior, exact parent level columns, grouping budget, and
source-group bootstrap equivalence. Additional cached Joint/parent score
replays and independent raw-feature/weight/gate/metric/label/hash reviews are
required. At most three CPU workers, resumable answer checkpoints; no inference
or change to Claude's worktree. Preserve all remaining mandate stages.
