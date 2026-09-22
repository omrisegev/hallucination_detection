# Fusion Independence Atlas v1

## Status and scope

This is a development-only experiment on the frozen 13,769-answer localization
population.  It starts from `f7e62c9aeb7f0bff992cd6b76385f0177259568f` and
does not constitute external confirmation.  Correctness annotations may define
operational errors and may select a roster inside nested source-group folds.
They are forbidden from signal extraction, predictor fitting, orientation, and
fusion-weight fitting.

The first cycle covers aligned, replayable signals.  FM, DiFlo, DOT, and any
method without a complete aligned score artifact remain `PENDING_EXPANSION`.
Expansion is not automatic.

## Immutable data contract

The canonical token bundle has 6,968,779 rows and exactly four prediction
targets: `H0lim`, `VE0`, `VE0.75`, and `VE1`.  A predictor must observe only
past values of those primitive levels.  It must never predict `H0lim`
innovation, digit innovation, or another derived stream as a target.
Innovations are computed once, after prediction, as observed minus causal
background.  Token zero is masked because no prior observation exists.

The answer roster has 13,769 rows and 145,597 steps.  The source-group split is
the corrected five-fold map in
`results/localization_source_group_audit_v1/FOLDS_V2.json`.  Learned predictors
must exclude every source group in the held fold.  Pair-excluded fits are used
where an inner calibration fold must also be unseen.

Every output array is bound to a manifest by shape, dtype, semantic identity,
source hashes, code hashes, and SHA-256.  Long stages are answer-resumable and
record a completion bitmap.  Existing artifacts are verified before reuse and
are never overwritten on contract drift.

## Registry

The registry contains three immutable records:

- `SignalSpec`: provenance family, insertion point, resolution, transform,
  access scope, orientation, inactive-mask semantics, and implementation hash.
- `ReadoutSpec`: a label-free reduction with its input/output resolutions,
  chronological semantics, short-trace rule, and implementation hash.
- `FusionSetSpec`: members from one legal insertion point and compatible
  resolution, a positive fusion head, maximum size six, and fitted-weight hash.

The registry API has no label argument.  A candidate that cannot be reproduced
and explained is retained as `REPORT_ONLY`; incomplete FM/DiFlo/DOT candidates
are `PENDING_EXPANSION`.

The aligned atomic roster contains the four q15 primitives, the 31
Renyi/escort streams, VE1-q50/H1/Hinf, the nine Step-395 alternatives, the
17-coordinate direct-probability family and its registered temporal controls,
and digit disagreement/opportunity/rate controls.  Exact aliases, including
H1 versus native top-15 entropy, are de-duplicated by score hash.

## Causal transforms and readouts

Every primitive receives a level view and a direct causal innovation.  The
registered backgrounds are prefix mean, trailing mean-16, no-reset Bayesian
mean, BOCPD with hazard 1/32, source-excluded Ridge, and a new four-target TCN.
Digit has both token-clock and opportunity-clock innovations.  An inactive
view is masked rather than interpreted as evidence zero.

The unique step readouts are Top-k for k in `{1,2,3,5,8,10}`, mean, median,
top 25%, top 50%, q75, q90, first4, last4, and best-contiguous10.  Each readout
is an expert with its own error vector.  Decoders are tested separately:
argmax, first-near-max 0.25, persistent-q90-3, step-Top5, and earlier-VE peak.

Tail15 has four distinct registered roles:

1. level with per-step mean;
2. level with per-step Top10;
3. causal prefix-mean and prefix-Top10 innovations;
4. answer-level `Top10 - mean` gate prominence.

The fourth quantity is constant across steps and cannot alter a locator peak.

## Legal fusion points

Fusion is evaluated independently at five points: backgrounds of one
primitive, token signals before readout, step scores after readout, decoder
decisions, and answer gates.  Flat fusion across targets, resolutions, or
insertion points is invalid.  The primary order is per-view readout followed by
fusion.  Only finalists receive the token-fusion-then-readout sensitivity.

Positive heads are equal-rank, family-equal, and nonnegative shrunk-simplex.
IU is evaluated only for `INDEPENDENCE_COMPATIBLE` sets of at least three.
Negative or sign-unstable IU weights make IU diagnostic-only.

## Error targets and dependence contract

Five matrices remain separate:

1. continuous predictor residuals for alternative backgrounds of the same
   primitive;
2. ProcessBench raw-locator miss before a gate;
3. PRMBench positive-negative pair misordering, ties worth one half;
4. gate false-open on clean answers and false-close on erroneous answers;
5. final ProcessBench error after both locator and gate are frozen.

For PRMBench, the frozen `JOINED` contract uses label `1` as the positive
high-risk ranking class.  A pair is misordered when its label-1 score is below
its label-0 score; a tie contributes one half.  Baseline replay includes an
asymmetric orientation fixture because reversing this bit returns exactly the
complementary within-answer AUC.

Conditional diagnostics use outer source-group cross-fitting with cell,
`log1p(tokens)`, `log1p(steps)`, relative first-error position, and digit
opportunities where applicable.  Per pair the atlas records conditional phi,
conditional odds ratio, joint-failure ratio, mutual information, unique wins
and losses, score Spearman, Top-k Jaccard, and peak agreement.

`INDEPENDENCE_COMPATIBLE` requires simultaneous 95% intervals wholly inside
`abs(phi) <= 0.20` and odds ratio `[0.67, 1.50]`, with at least 50 successes,
50 failures, and 20 source groups.  A group of size k at most six additionally
requires the upper interval of maximal conditional residual correlation at
most 0.25 and the lower interval of effective rank at least `0.70*k`.
Failure to pass is not proof of dependence; insufficient support is
`UNRESOLVED`.  A failed independence screen with held-fold unique successes
and improved OOF fusion is `DEPENDENT_COMPLEMENTARY`.

## Selection, uncertainty, and promotion

Within each provenance family, retain at most the utility leader and one
diversity representative within one ProcessBench percentage point and 0.005
PRMB-within of that leader.  Build a compatibility graph per insertion point
and enumerate all passing cliques of sizes two through six.  Selection is
nested in five source folds; a roster must recur in at least four folds.

After one finalist is frozen at each legal point, run the full 2^5 incumbent
versus finalist factorial to estimate main effects and interactions.  Report
10,000 paired source-group bootstrap draws with simultaneous intervals.  Gate
draws recompute within-cell midranks, rank fusion, and the 0.33 decision.

The factorial has a strict typed-composition precondition.  Each finalist must
provide an executable recipe whose output type is the input type of the next
stage; independently selected end-to-end locator experts may not be flat-fused
or treated as interchangeable stage transforms.  If the five finalist recipes
do not establish that contract, the run writes all 32 declared arms with zero
evaluated arms, `fabricated_arms=false`, and status
`BLOCKED_COMPOSITION_CONTRACT`.  Upstream independence, nested-selection,
uncertainty, Pareto, and leave-one-signal-out results remain reportable as
`DEVELOPMENT_PARTIAL`, but the factorial is not inferred from them.

Required exact baseline replay:

- original4: 0.374749 ProcessBench;
- innovation5: 0.398314 ProcessBench;
- digit025: 0.413300 ProcessBench and 0.776036 PRMB-within;
- current: 0.432546 ProcessBench and 0.776036 PRMB-within.

A performance successor needs either +1 ProcessBench point with a positive
lower interval and PRMB-within noninferiority at -0.002, or +0.005 within with
ProcessBench noninferiority at -1 point.  A simplification successor must meet
the same noninferiority margins and remove at least 30% of active families or
one complex stage.  Positive estimates whose interval crosses zero are
`RESEARCH_CANDIDATE`, not winners.

## Expansion gate

FM/DiFlo/DOT or new predictors may be proposed only if the aligned cycle finds
neither (a) an independence-compatible group spanning at least three
provenance families nor (b) a performance or simplification successor on the
Pareto frontier.  The proposal must be reviewed before execution.
