# Joint L-SML existing-data localization v1

Status before registration: `DRAFT_RETROSPECTIVE_OPENED_DEVELOPMENT`

## Question

Does the already frozen active-23 Joint L-SML fusion head improve localization
on the existing Qwen ProcessBench and PRMBench development populations enough
to justify a separately registered fresh-population generalization experiment?

This run can answer only that development question. The populations and their
outcomes have been opened in earlier project work.

## Frozen populations

- ProcessBench: Qwen3-4B and Qwen3-8B, each on GSM8K, MATH,
  OlympiadBench, and Omni-MATH (3,400 source questions; two model copies).
- PRMBench: Qwen3-8B, 6,966 admitted responses and 94,112 official step spans.

Raw telemetry is joined to the canonical opaque, outcome-free release by token
count and exact step-span signature, with mixed-v2 confidence correlation used
only to disambiguate signature collisions. The trusted sanitizer writes only
`raw`, `token_offsets`, `row_ids`, `segment_offsets`, `segment_starts`, and
`segment_ends`.

## Frozen feature and fusion contract

The absolute raw-domain 29-sign registry and globally pruned roster are copied
unchanged from Joint L-SML R2. The active roster has 23 streams. It contains
five provenance families in the maintained repository mapping; their observed
active counts are recorded by the runner rather than restated manually.

Every arm receives the same deterministic 60,000-token fit sample, imputation,
population z-score, absolute confidence orientation, and active-23 columns.

The efficacy roster is exactly:

1. `joint_lsml23_hierarchical_v1_1`: INTERNAL groups selected from rank-one
   residual affinity by exact leave-one-answer-out consensus over
   `K in {3,4,6,8}`; five-start Joint fit; hierarchical Joint weight map.
2. `iu_pcr_active23`: incumbent two-component IU-PCR on the matched matrix.
3. `equal_family_active23`: equal mass across the five present provenance
   families and equal mass within each family.
4. `fixed_family_continuous_lsml_active23`: maintained continuous L-SML with
   the fixed provenance-family partition.

Same-INTERNAL-partition continuous L-SML is a structural diagnostic only. It is
not an efficacy arm.

The fixed-family control includes small provenance groups and is an algorithmic
control only; it must not be described as theorem-valid hard L-SML. Because all
four efficacy arms share active-23, this run also cannot estimate the efficacy
effect of pruning itself.

## Structural gate

For each cell, before any score artifact is written:

- an admissible `K >= 3` partition must exist, with every consensus group and
  at least 95% of LOAO folds keeping every group at three streams or more;
- at least four of five Joint starts must converge, objective traces must be
  monotone, the frozen multistart agreement audit must pass, the profiled
  global-loading Jacobian must be full rank, and all four efficacy weights must
  be finite;
- Joint off-diagonal misfit and diagonal clipping are reported but do not
  select a variant.
- the donor-score Spearman among the frozen hierarchical map and the three R2
  diagnostic maps must remain at least 0.50. The hierarchical map is
  irrevocably primary; a disagreement cannot substitute another map.

LOAO itself is exact over every admitted answer. Pairwise held-fold ARI is a
diagnostic only: when the quadratic pair roster exceeds 32,768 pairs, its
summary uses a deterministic seed-bound uniform pair sample. This does not
enter K selection or any pass gate.

Claude's pre-registration review identified that median ARI saturated at 1.0
and that an all-fold intersection becomes harsher as the number of answers
grows. Before registration and without outcomes, v1.1 therefore freezes two
engineering corrections: tied K values are ordered by median ARI, mean ARI,
minimum ARI, then smaller K; and held-fold admissibility is a 95% quantile rule.
These are part of the single candidate, not scoring variants. The original R2
rule remains preserved in its historical namespace.

The 0.50 map-agreement threshold was chosen after the opened, label-free R2
structural ledger showed minima of roughly 0.708--0.879 on the active lane. It
is therefore a conservative engineering catastrophe guard informed by prior
structural data, not prospective evidence for v1.1 and not an efficacy claim.

All eight ProcessBench cells form one panel: if any cell fails, ProcessBench is
`STRUCTURAL_NO_SCORE`. PRMBench is a separate panel and may proceed or fail
independently. No fallback is substituted inside this namespace.

## Fixed localization adapters

ProcessBench uses the same local curve for detection and localization:

- detector score: maximum token risk over the response;
- step score: mean of the largest `min(10, step_length)` token risks;
- locator: step with maximum step score;
- five deterministic source-question folds shared by the two model copies;
- threshold fit separately for each model from its four subsets, inside every
  bootstrap replicate;
- report the equal mean of the Qwen3-4B and Qwen3-8B official subset-macro
  values.

PRMBench uses maximum token risk within each official step span and reports
error-positive step AUROC/AUPRC. It never pools independently fitted cells.

Both panels use 2,000 paired source-group bootstrap draws. ProcessBench
resamples within subset x frozen-fold strata so every replicate retains all
five calibration folds; both model copies and all arms remain in one paired
source-question payload. Candidate-minus-
control intervals are descriptive 95% percentile intervals. There is no
confirmatory alpha family, SESOI, promotion gate, or post-label tuning.

The complete multiplicity, fold, seed, pairwise-diagnostic, and bootstrap
contract is frozen in `configs/joint_lsml_existing_localization_v1.json`.

## Firewall and artifacts

The run has four irreversible phases:

1. trusted target-free sanitization;
2. registry freeze binding protocol, code, source telemetry, safe spans, signs,
   roster, and test hashes;
3. structural fit and minimal score freeze with no label import;
4. independent score-freeze audit, followed by a separate evaluator that alone
   may load the canonical PB/PRM outcomes.

The evaluator cannot mutate the score freeze. It produces separate PB and PRM
JSON/CSV reports and plots. Every result is labeled
`RETROSPECTIVE_OPENED_DEVELOPMENT`.

## Decision vocabulary

- `DEVELOPMENT_SUPPORTED`: positive point delta versus IU-PCR and no wholly
  negative paired interval versus either L-SML/equal-family diagnostic control.
- `INCONCLUSIVE`: no clear harm, but the development support rule is not met.
- `HARM`: Joint-minus-IU-PCR interval is wholly below zero.
- `STRUCTURAL_NO_SCORE`: a pre-label structural or coverage gate fails.

These states decide only whether a new-population protocol is worth preparing.
They cannot establish generalization.
