# Frozen supervised manifold external-to-discovery audit v1

## Claim boundary

This is one frozen **retrospective external-to-discovery audit** of the
representation nominated by
`SUPERVISED_CONDITIONAL_MANIFOLD_DISCOVERY_V1.md`.  None of the validation
dataset families contributed to the feature weights, support, linear
comparator, graph rule, or thresholds.  However, their outcomes already exist
elsewhere in the project, so this is not prospective confirmation and cannot
establish a universal hallucination manifold.

The primary estimand remains completed-answer incorrectness.  ProcessBench and
RAG grounding are not pooled with it.

## Frozen validation population

The audit requires at least three independent dataset families and at least one
cell in which both the dataset and exact model family were absent from internal
discovery.  The registered local population is:

- AQuA: the paper-exact `cot|central` response from Qwen2.5-7B,
  Llama-3.1-8B, and Phi-3-mini; one response per question.  Its normalized
  top-15 entropy is deterministically reconstructed from retained raw top-50
  log-probabilities because the historical greedy live `sampled_entropy`
  channel contains `NaN` from a `0 * -inf` telemetry operation.  Correctness is
  replayed with the already frozen `fair_aqua_option_parser_v1.0.0`, replacing
  the known invalid numeric parser stored in the acquisition rows;
- HLE: one Qwen2.5-72B response per text-only question, using the completed
  official-prompt interim judge file rather than the invalid provisional
  ROUGE-L label;
- CoQA: the first stored Llama-7B response per source question.  Selecting
  candidate zero is target-blind and prevents the ten repeated generations
  from being treated as independent rows; questions whose first response lacks
  the complete frozen feature contract are then dropped without labels.

Every feature matrix is constructed before its label vector is attached.  Raw
features are oriented with `CONFIDENCE_FEATURE_SIGNS_V1` and z-scored within
the unlabeled validation cell, matching the transductive discovery contract.
The six top-k distribution summaries are harmonized on retained raw top-50
log-probabilities in all external cells; this target-blind compatibility rule
avoids the `-inf` padding present in some post-warper sampling caches, but it is
a telemetry variant relative to parts of the internal discovery pool and is an
additional reason the audit is not prospective confirmation.
Missing features, repeated row IDs, a single-class target, or non-finite values
fail closed.

## Four fixed graph views

For every cell and target-blind tie seed `(101, 211, 307)`, evaluate:

1. `metric_graph`: the frozen supervised diagonal metric;
2. `linear_score_graph`: a one-dimensional graph on the frozen balanced-logit
   score;
3. `equal_weight_graph`: the same 16 oriented features without supervised
   weights;
4. `linear_residual_graph`: the metric features after an unlabeled least-squares
   projection removes the frozen linear score.

The fourth view asks the decisive visual/mechanistic question: is any local
geometry left after the shared linear confidence direction has been removed?

Graph construction uses the frozen self-safe local-scale union-kNN rule and the
smallest label-free `k` in `{3,5,7,10,15,25}` passing graph health.  Exactly 999
exact-length and 999 cross-fitted CRT draws are used.  The six primary tests
(metric exact/CRT, residual exact/CRT, metric-vs-linear exact/CRT) receive Holm
correction within each cell and tie seed.  Tie seeds are robustness dimensions,
not independent replicates.

## Frozen gates

A cell passes conditional geometry only when every tie seed has a healthy and
eligible graph, exact and CRT effects at least `0.02`, and Holm-adjusted
`p <= 0.05`.  It passes residual geometry under the same criteria after the
linear score is removed.  It passes distinct geometry only when the metric
also beats the linear-score graph by at least `0.02` under both nulls with both
adjusted p-values at most `0.05`.

Within a multi-model dataset family, at least two thirds of cells must pass;
within the complete panel, at least two thirds of independent dataset families
must pass.  The equal-family mean metric-vs-linear advantage must be at least
`0.02` and its family-bootstrap lower bound must exceed zero.  LIU utility is a
separate gate: mean `LIU - IU >= 0.005` AUROC with a positive family-bootstrap
lower bound.

Possible decisions are:

- `INSUFFICIENT_EXTERNAL_COVERAGE`;
- `CONDITIONAL_NULL_INELIGIBILITY_INVALIDATES_EXTERNAL_AUDIT`;
- `RETROSPECTIVE_EXTERNAL_TRANSFER_FAILURE`;
- `RETROSPECTIVE_EXTERNAL_SHARED_DIRECTION_ONLY`;
- `RETROSPECTIVE_EXTERNAL_DISTINCT_GEOMETRY_CANDIDATE`.

Even the last decision remains a candidate awaiting a genuinely prospective
dataset/model grid.  If the metric does not beat the linear graph, residual
geometry collapses, or transfer fails, further weight/subset/topology search on
the same 16-feature space is closed.  Reopening requires a new representation
or intervention, not more tuning of this matrix.

Ineligibility is not a negative effect result.  If fewer than two thirds of
independent dataset families support both registered conditional nulls, the
audit is invalidated and cannot be called a transfer failure.  A future
protocol may pre-register a calibrated local length-matched null, but it may
not be substituted after observing this run.

## One-shot and provenance rules

The validator requires exactly 999 conditional draws and refuses to overwrite
a non-empty output directory.  `RUN_DEFINITION.json` seals the candidate,
manifest, and all derived matrix hashes.  The manifest records every raw source
hash and the target-blind row-reduction rule.  Raw caches and derived matrices
remain ignored local assets; only their hashes and compact audit outputs belong
in Git.

## Required outputs

- `RUN_DEFINITION.json`;
- `CELL_GRAPH_METRICS.csv`;
- `FAMILY_SUMMARY.csv`;
- `DECISION.json`;
- `REPORT.md`;
- static figures and a deterministic verification record.
