# Fair Paper-Exact Comparisons v1

Status: **frozen comparison protocol**
Date: 2026-08-18
Method of record: **ordinary Unified-28**

## Scope freeze

Unified-28 is exactly the seven registered base streams crossed with
`{level, ewma16, positive_area, persistence}`, ordinary two-component L2
IU-PCR, and the Identity accumulator. Its roster, signs, task weights,
accumulator, and ordinary (non-DUFS) setting are immutable. Evaluation outcomes
may not select or modify a method. This package is a retrospective comparison
and provenance exercise; it does not reopen feature, DUFS, architecture, or
confirmation search.

The package has four independent lanes. There is no cross-lane leaderboard.

1. Global final-answer detection: Unified-28 and the dedicated
   `classic_mixed_v2_no_length` incumbent are mandatory in every eligible direct
   table. The Llama ProcessBench direct roster is exactly Unified-28,
   `classic_mixed_v2_no_length`, the frozen registered
   `mixed_v2_dufs_liu_l0p1_no_length` control, and maximum entropy. The Unified
   DUFS lambda sweep, the task-reweighted candidate, and ordinary-36 remain
   visible in a historical context table but are ineligible for direct or
   primary claims.
2. ProcessBench first-error Localization: Unified-28 and the dedicated frozen
   `family6 + level + step_top5mean` scorer/locator are mandatory. The
   max-entropy plus top-five-step locator is the transparent direct bar.
3. Causal Prefix detection: Unified-28, `iu28_no_length`, and the selected
   Step-272 0.50/0.50 Global/Local Online architecture are mandatory whenever
   all three join the same causal population.
4. Stopping/adaptive compute: Unified-28 is ineligible in v1 because no frozen
   policy has real forced-closure outputs. Single-trace LEASH and multi-trace
   DeepConf remain separate tables.

## Canonical interfaces and identity

The machine-readable schemas are `population_registry_v1`,
`method_registry_v1`, and `comparison_record_v1`. ProcessBench IDs are
`processbench@e8024636bcab::<subset>::<official_id>`. The official 3,400-row
numeric order has SHA-256
`7a7edc7a6e4a67ac16968c915900805620fc86314368105afe05a4d1ffd20e10`.
Twenty-four-cell trace IDs add the candidate ordinal so repeated samples cannot
collide; their group ID omits the ordinal. S2 trace IDs include arm, while its
paired comparison unit is the source-question group. DeepConf groups every
sample of one AIME24 question.

There is no positional ID fallback. Duplicate IDs, label disagreement, an
unknown source-artifact hash, or less than 100% coverage fails a headline join.
Unparsed external predictions stay in the population and count wrong.

## Evaluation, calibration, and uncertainty

Error/risk is always the positive class. No pooled AUROC is allowed across
separately fitted folds or heterogeneous cells. Causal rows exist only when
final trace length is strictly greater than budget. ProcessBench clean is `-1`.
Realized token savings count generated reasoning plus generated closure tokens;
potential remaining tokens are diagnostic only.

Five deterministic folds are assigned by sorting SHA-256 hashes within each
`(family, binary clean/error)` source-question stratum and round-robin assigning
folds. Every scorer/model copy remains in its source-question group. Score and
method parameters are frozen from the originally registered fit IDs;
cross-fitting changes decision thresholds only.

- Localization thresholds maximize equal-subset macro ProcessBench F1 on four
  folds; ties prefer higher clean accuracy and then the higher threshold.
- Global 5%/10% thresholds use correct calibration rows only and never split a
  tied score block.
- Prefix 5%/10% thresholds use each correct trace's maximum over the complete
  six-budget monitoring path, controlling trace-level ever-warning.
- Fixed PRM/critic/judge predictions receive no fit.

Uncertainty uses 2,000 paired source-question bootstrap draws within family,
seed `20260818`. All methods, budgets, arms, and copies travel together.
Permitted thresholds are recomputed in every replicate. Primary contrasts are
predeclared as Unified-28 versus its dedicated incumbent and Unified-28 versus
the strongest same-access published competitor.

## Fidelity and access

Every method has exactly one fidelity label from:

`official-exact`, `paper-specified`, `paper-specified-partial`,
`adapted-common-protocol`, `published-context-only`, `blocked-assets`.

Access is orthogonal: input type, supervision, model passes per question, and
traces per question are always recorded. The registry additionally states
whether “model passes” means an extra scorer invocation on an already generated
trace or generation performed by the method itself. High-access PRM/critic ceilings are
visually separated and cross-tier differences are not described as wins or
losses. The Qwen2.5-14B acquisition is named “uPRM Eq.6 Qwen2.5-14B control” and
is `paper-specified-partial`; the older Qwen3-8B reconstruction is never called
uPRM.

Every file asset hashes its actual bytes. Derived and composite assets instead
hash an explicit canonical JSON projection containing their member ledger; any
upstream package fingerprint accepted by a row is declared inside that same
projection. Method provenance records the package-build commit separately from
the artifact-generation commit (or explicitly records that a pre-contract
source did not preserve one). The evaluator identity hashes the complete
comparison closure: all lane evaluators/adapters/parsers, the paper-exact
primitive evaluator, fold logic, Unified causal evaluation, and this builder.

The frozen Drive observation binds the exact byte size and SHA-256 of L0, the
six L1 metadata/gate objects, REFRAIN S1, the 30 metadata objects for the six
complete S2 cells, and all 24 live M2 status objects inspected read-only.
Acquired L1 and S2 metadata are required to match those remote identities
byte-for-byte; their local shards are independently bound from the verified
cache. Hard-coded status prose is never treated as a file hash.

## Compute gates

The v1 package is CPU-first. REFRAIN, failed Mistral LEASH cells, DeepConf M2,
a Unified stopping policy, full trained uPRM, Streaming probes, and any new
confirmation cell require separate approval. Before any GPU work: all local
hashes and joins must pass; evaluator, folds, parser, closure, and claim
registries must be frozen; an offline reconstruction must be proven
insufficient; and a bounded parser/equality/throughput pilot must pass. No GPU
run is authorized merely to improve provenance.

Approval estimates are frozen as: REFRAIN remainder approximately 6–8 B200
GPU-hours; two Mistral LEASH cells approximately 2–3 B200 GPU-hours; DeepConf
M2 roughly 1,000 remaining B200 GPU-hours; full trained uPRM approximately 44
H200 GPU-hours if an exact official path appears; and a Unified stopping pilot
below one GPU-hour before any separately costed full run. Their scientific value
and prerequisite gates are emitted as structured fields in `GPU_GATES.json`.

## Exact-access gates observed during implementation

The CPU implementation fails closed when a filename, row count, or question ID
does not prove the same generated trace. The following are package facts, not
new method-selection decisions:

- The registered Llama ProcessBench population joins all 3,400 official IDs for
  Global and Localization. The Prefix direct panel joins four registered cells
  (1,717 traces) for Unified-28, `iu28_no_length`, Step-272, mean/max entropy,
  and the explicitly named historical DeepConf entropy proxy. The original
  IU/DeepConf calibration and evaluation anchors reproduce with zero score
  difference on this telemetry. The older eleven-cell score package remains
  context because its trace realization differs.
- The Global direct panel also contains a frozen no-length mixed-v2 DUFS-LIU
  secondary control. It fits on exactly the same 32 registered Qwen3-8B IDs per
  family as `classic_mixed_v2_no_length`, with seeds `{11,23,37}`, 80 epochs,
  graph `k=7`, and lambda `0.1`; none is selectable. Before the 3,400 Llama
  scores are materialized, the implementation must reproduce the registered
  length-enabled Qwen3-8B/GSM8K pre-label score hash
  `c75d27be8492278ced261f93c3d809ca16c5e95103ef6288be3212ad2c659be5`
  and match both frozen Qwen and Llama telemetry hash ledgers. This is an
  `adapted-common-protocol` project control, not a published competitor.
- The six complete S2 COT cells pass manifest, index, shard-hash, and question-ID
  checks but cannot run the frozen Global or Prefix methods. They retain raw
  pre-warper entropy, sampled-token energy, raw top-50 log probabilities, and
  raw log-sum-exp; the frozen methods require legacy post-warper entropy,
  sampled-token energy, and top-k streams. Substituting raw streams would create
  an adapted new method and is forbidden.
- The 24-cell local source-size audit admits 22 authorized registered cells. Full
  raw-hash and label-free identity replay proves 6 independent cells / 3,238 rows:
  `epr_triviaqa_mistral24b` (621), `lapeigvals_gsm8k_mistral24b` (1,319),
  `losnet_hotpotqa_mistral7b` (500), `math500_dsmath7b` (300),
  `math500_r1distill8b_mn4096` (300), and `sciq_llama8b` (198). Each proven
  cell is a complete direct population with its own ordered-ID hash, evaluator,
  paired source-question bootstrap, and table containing Unified-28, ordinary
  full-pool mixed-v2 IU-PCR, full-pool mixed-v2 DUFS-LIU at lambda 0.1, and max
  entropy. The incumbent rows come only from the hash-frozen
  `hard_filter_dufs_liu_24cell` score files and exact `mixed_v2__full__*` keys;
  the ProcessBench anchor must reproduce before Unified-28 transfer scoring.
  The ordinary and DUFS incumbents are explicitly labeled as unsupervised,
  transductive full-target-cell fits; they are not presented as the same access
  tier as frozen per-trace Unified-28 or max entropy.
  Multi-candidate questions are assigned as one fold using the preregistered
  question-level stratum `any_candidate_error`; the row-level outcome remains the
  metric label, and no candidate sibling may move to another fold or bootstrap draw.
  Equal-cell and equal-family Math/QA/overall summaries over these six cells
  include AUROC, error AUPRC, and the 5%/10% TPR, precision, and observed-FPR
  operating points. They are coverage-only context, never
  a 23-cell headline or win/tie/loss claim. Ordinary-36 has no auditable per-row
  records and remains visibly unavailable as historical context.
- Sixteen authorized sources fail exact feature identity. The
  `internalstates_gsm8k_qwen25_7b` payload remains outside the approved
  minimal-movement wave and is not opened or scored even if a local copy is
  present. `spilled_triviaqa_llama8b` still lacks the ordered identity/admission
  ledger needed to map its 256 bundle rows. These 17 registered-source cells and
  the additional unregistered-identity cell remain blocked from direct scoring.
- REFRAIN (512/1,000), DeepConf M2 (12,370/122,880), and both failed Mistral
  LEASH cells remain partial appendices. Unified-28 stopping remains ineligible.

There is no exact same-access published Global or Prefix competitor that joins
the direct populations. Those primary contrasts are recorded as unavailable;
historical proxies are never promoted to paper-exact rows.

## Acceptance

A direct claim is eligible only when its table carries one population/order
hash, one evaluator, identical registered comparison units, and a paired
interval. Unified-28 and the lane's dedicated incumbent must appear wherever
telemetry makes both eligible. Incomplete acquisitions never enter headline
aggregates. Prefix claims carry separate registered hashes for budget 64,
budget 128, and the complete six-budget warning cohort; LEASH claims carry the
paired source-question group-order hash rather than an arm-specific trace hash.
Join expectations are derived from those population registries, never from the
rows a method happened to emit.

Publication builds require exactly 2,000 bootstrap draws, clean runtime code,
the Unified worktree hash, and the 24-cell identity audit. Any deviation is
available only behind an explicit `--testing-only` flag and produces a stamped,
non-publication package. Acceptance requires two independently written builds
from the same clean commit to have byte-identical complete-tree manifests and
canonical JSON/CSV bytes before research-history documents are updated. A clean
build therefore records only build-mode eligibility and a pending independent-
rebuild status; it never self-certifies final publication acceptance.
