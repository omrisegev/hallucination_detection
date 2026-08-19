# Research Directions — Thesis Roadmap
*Omri Segev | Supervised by Bracha Laufer-Goldshtein & Ofir Lindenbaum*

---

## Step-273 verification and the advisor-facing comparison — Steps 274-275

Step 273's disposition below is now independently verified rather than merely
recorded. Its frozen protocol had become unrunnable — the gated document was
revised after the run and before it was committed, so the study's own hash
matched nothing in the repository. Codex recovered the exact pre-commit bytes as
`docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.frozen-c921b0d4.md`, and the
replay reproduces every decision-bearing artifact byte-for-byte, including
`STAGE_1_LOCAL_INTERVALS.csv`. Since verdicts here are decided by whether a
paired interval excludes zero, that file matching is what makes the verification
load-bearing. The residue is one column drifting at 1e-14 with no prediction
changed. **Nothing in the Step-273 roadmap disposition changes.**

What does change is how the result is reported. Neither number previously
floated is the advisor-facing row: Stage 1's 0.3517 is development-selection
evidence, and Stage 4's 0.3662 belongs to the joint finalist that Step 273
rejected, so it stays a clearly named historical row rather than an
interchangeable estimate. The direct Localization table puts every method on the
same official 3,400 ProcessBench IDs and evaluator, with three mandatory
same-access rows:

1. ordinary Unified-28, the frozen unified method of record;
2. dedicated `family6 + level + step_top5mean`, the Stage-1 incumbent;
3. maximum entropy plus the top-five-step locator, the transparent direct bar.

Qwen2.5-Math-PRM-7B and the Qwen-72B critic stay visually separated high-access
ceilings, never inline competitors. The advisor claim comes from that
out-of-fold common-row table and its paired interval, not from whichever
historical stage reads strongest.

**Blocking dependency**: the contract implementing this — the four-lane protocol,
the population/method registries, the causal prefix joins and the builder — lives
on `codex/fair-paper-exact-comparisons-v1`, which is not on our remote, and none
of its four named files exist in any branch we can see. It is deliberately not
being reimplemented from its description, since that would produce a second,
divergent definition of the same table. Requested as Question 3 in
`HANDOFF_CODEX_2026_08_18.md`.

**Data status**: both paper-exact acquisitions are complete, gate-verified and
backed up byte-identically to Drive — DeepConf at K=512 (15,360 traces, a
declared deviation preserving every registered budget 32-512 at full width) and
REFRAIN at 1,000. The older K=4096 partial pool is kept separately and must not
be merged. No new GPU work is approved.

**Only unblocked line of work**: the offline DeepConf derivation over the K=512
pool. It needs no GPU and no registry.

## Comprehensive Local/Online transfer result — Step 273

The broadened existing-cache cycle closes without promoting a new joint
architecture. Provenance-balanced family-six features are useful development
mechanisms: level plus a step-top-five locator reaches 0.3517 Local F1 on the
S1 cells, and fast/slow reaches 0.6020 Online AUROC@64/128 on S2. Ordinary IU
remains preferable to equal, historical U-PCR compatibility, uniform/DUFS/
temporal Laplacians, and hierarchical fusion under grouped uncertainty.

The S3 two-head finalist uses the registered Global signal for error detection
and Online scoring and retains a family-six Local head for the step locator.
It does not survive scorer transfer as a co-primary improvement. Across
Qwen3-8B and Llama-3.1-8B audit cells, Local is 0.3662 versus 0.3614 for the
maximum-entropy/top-five direct bar, delta +0.0048
[-0.0264,+0.0375]. Online is 0.5882 versus 0.6104 for IU28, delta -0.0222
[-0.0502,+0.0042], with family wins/losses 1/3. The Online loss breaches the
frozen 0.015 margin, so the joint finalist is rejected.

Forward disposition: keep IU28 as the strongest direct S4 Online bar. Keep the
family-six Local/top-five mechanism only as a retrospective candidate; maximum
entropy remains the transparent Local reference until fresh evidence supports
replacement. Do not reopen graph fusion, event-coordinate combinations,
raw-seven post-hoc pruning, or Global-only Online scoring on these opened
cells. A new GPU/inference run is not justified. A future cycle must add a
materially new signal or fresh unopened evidence under a new protocol. Full
report: `results/local_online_comprehensive_v1/REPORT.md`.

## Three-output architecture result — Step 272

The broader CPU-only search resolves the intended architecture question on
existing telemetry: retain **two ordinary-IU heads**, not one universal head
and not three independent heads.

- Global keeps the historical full-trace mixed-v2 feature system.
- Local uses a new nine-channel raw token-level risk head; onset transforms are
  worse in the frozen development screen.
- Online is derived from a 0.50/0.50 standardized causal prefix Global score and
  running-maximum Local evidence. A separate sustained-state Online head wins
  its isolated head screen but is redundant in the architecture cross.
- The locator is the Local peak. The previously assumed 0.75/0.25 blend is not
  retained.
- Ordinary IU-PCR stays. Same-matrix DUFS/uniform/temporal gains are tiny,
  uncertain, and costlier.

Across twelve ProcessBench cells, two heads versus one improve Global AUROC by
+0.0271 [0.0085,0.0449] and Local F1 by +0.0740
[0.0458,0.1013]. Online changes by +0.0067 [-0.0121,0.0260]. Three heads versus
two change only Online, by -0.0067 [-0.0248,0.0126], so the third head does not
earn its 27 features and 36 state scalars.

The warning policy remains a limitation: at a calibration target of 10%
trace-level false warnings it detects 25.0% of wrong traces, with 8.1% observed
false warnings. Phase-15 early transfer is weak (0.5142/0.5555 AUROC at
64/128), despite strong final discrimination. The selected architecture is
therefore a retrospective candidate, not a deployment or fresh-generalization
claim.

The next useful method cycle is narrower and cheaper: preregister Local subsets
suggested by the drop-one diagnostics and replace repeated mixed-v2 prefix
recomputation with an exact or validated incremental implementation. Only then
consider an explicitly approved fresh-data run. Do not spend GPU time on DUFS,
a Laplacian, or the third Online head based on this evidence. Canonical report:
`results/global_local_online_architecture_v2/REPORT.md`.

## Scope correction and reopened architecture search — Step 271

Step 270 did not complete the intended architecture optimization. It validly
closed only current/running-maximum, persistence/area, and slope/recovery
transforms of two already globally aggregated signals on the saved coarse
monitor grid. The Global and Local heads, head allocation, 0.75/0.25 blend,
IU-PCR configuration, and number of heads were frozen. Identical localization
hashes therefore show non-interference, not a newly optimized localizer.

The active protocol is now
`docs/experiments/GLOBAL_LOCAL_ONLINE_ARCHITECTURE_V2.md`. It treats completed
answer detection, first-error localization, and causal early prediction as
three separate co-primary outputs. It begins from raw token telemetry and
tests head-specific reducers: full-trace mean/tail/extreme summaries for
Global, level/onset curves for Local, and level/EWMA/onset/persistence states
for Online. It then compares one, two, and three-head harnesses and retests the
fixed Global/Local mixture instead of assuming 0.75/0.25.

The Laplacian is deliberately low priority. Ordinary IU-PCR is the simplicity
baseline; uniform, DUFS, and temporal variants are admitted only as exact
same-matrix controls after the feature and architecture choices freeze. They
remain only if a grouped benefit pays for their runtime and memory.

The cycle uses twelve complete existing ProcessBench telemetry cells (three
scorer models by four families), with shared question identities treated as
repeated measurements. Every cell is historically opened, so any selected
architecture is retrospective development evidence and still needs a future
explicitly approved fresh-confirmation/GPU gate. The Step-270 “no GPU” decision
continues to apply only to its three coarse-grid mechanisms.

## Application result override — retain IU28 and close coarse-grid dynamics (Step 270)

The frozen existing-cache Global-Local-Online IU cycle is complete. The
deployable application architecture remains the frozen Global and Local heads
with `iu28_no_length` as the Online head. The three preregistered dynamic
alternatives over saved CUSUM/`sw_var` monitor trajectories do not pass the
early-panel promotion gate:

| Online candidate | equal-family 64/128 AUROC delta vs IU28 | 95% grouped CI | family W/L |
|---|---:|---:|---:|
| current + running maximum | -0.0051 | [-0.0553,+0.0519] | 2/3 |
| positive area + run persistence | -0.0079 | [-0.0663,+0.0639] | 2/3 |
| slope + recovery | -0.0270 | [-0.0979,+0.0561] | 1/4 |

The first two arms are also 0.993 and 0.949 Spearman-correlated with the equal
CUSUM/`sw_var` magnitude control on the frozen endpoint. Removing `sw_var`
costs 0.026--0.046 equal-family AUROC depending on the arm, while removing
CUSUM is neutral or slightly positive on average. The tested temporal
summaries therefore do not expose a new independent signal at the existing
coarse absolute monitor grid.

The localization panel is unchanged: ProcessBench and PRMBench score hashes
are bit-identical for all Online-only candidates. The fixed trajectory-first
anchors reproduce at 0.3070 ProcessBench macro F1 and 0.6711 PRMBench step
AUROC; historical GL-LIU v1 remains 0.3136 versus 0.2571 for Mind the Gap.
Same-matrix graph controls again show only tiny ordinary-to-DUFS changes
(global +0.002193; local +0.000578), while the temporal local detector loses.
Graph regularization is not promoted.

**Research disposition**: retain the frozen Global/Local heads and IU28 Online
head. Close current/running-maximum, persistence/area, and slope/recovery
transforms of the saved coarse CUSUM/`sw_var` trajectories. Do not request GPU
inference on this evidence. Reopen the application-method search only for a
token-native causal recurrence or genuinely new telemetry/data, with a new
frozen protocol, independent question/family grouping, and explicit approval.
The canonical report is `results/global_local_online_iu_v1/REPORT.md` and the
machine decision is `results/global_local_online_iu_v1/DECISION.json`. This
result leaves the separate frozen A6/PTNI program untouched.

## Historical application charter — joint reasoning localization and early detection (Step 269; completed by Step 270)

The user-directed application focus is now the joint optimization of the two
reasoning tasks on which the label-free method has shown its strongest useful
structure:

1. **reasoning hallucination localization** — decide whether a trace contains
   an error and locate the first erroneous token/step; and
2. **early/online final-error detection** — while the trace is still being
   generated, estimate whether its completed answer will be wrong and decide
   when that estimate is stable enough to declare.

These are co-primary tasks. They must remain separate evaluation panels and
must never be averaged into one headline number. They should nevertheless use
one shared causal token-feature backbone wherever the evidence permits. Every
proposed change must therefore run a regression check on both panels. A gain
on one task does not justify a material loss on the other.

The optimization target is the **smallest label-free causal architecture on
the joint performance/compute Pareto frontier**, not the most elaborate
fusion. The working decomposition is:

```text
shared confidence-aligned causal token streams
    -> global head: is any reasoning error present?
    -> local head: where did the first error begin?
    -> online head: has the final error decision converged enough to declare?
```

The orientation contract is frozen. `CONFIDENCE_FEATURE_SIGNS_V1` aligns
registered features so that larger values mean greater confidence; online risk
uses the opposite direction. This does **not** mean every raw feature is
intrinsically monotone. The four recurrently non-monotone raw views are either
quarantined or replaced by the frozen mixed-v2 transforms, and raw plus
transformed copies must not coexist. No benchmark label may re-estimate a sign
or choose between those representations.

### Evidence that motivates the joint program

The first existing-cache online screen is broad enough to justify further
method work but not a superiority claim. It covers 11 materialized
dataset/model/generator cells and five dataset families with causal prefix
replay and no new inference. IU28 reaches macro AUROC 0.648 at 64 tokens and
0.694 at 128, versus 0.616 and 0.671 for the same-access DeepConf entropy-w64
proxy. Equal-family IU-minus-DeepConf deltas are +0.024
[-0.005,+0.056] and +0.014 [-0.031,+0.058], respectively. The intervals cross
zero: this is promising parity, not established leadership and not a reason to
close the comparison.

The score also measurably converges toward its own completed-trace value. For
IU28, prefix/final Spearman correlation rises from 0.417 at 64 tokens to 0.659
at 128 and 0.817 at 512; final-decision agreement rises from 0.640 to 0.739 and
0.880. Early declaration remains the main weakness: a calibration-constrained
IU28 policy has 0.366 macro coverage and 0.137 held-out ever-wrong rate, with
only 5/11 cells meeting the 10% target.

A causal Global-Local follow-up improves late/completed scoring but does not
yet create an early jump. At 64 tokens, the global, fused, `sw_var_peak`, IU28,
and DeepConf-w64 AUROCs are 0.638, 0.635, 0.643, 0.648, and 0.616. At 128 they
are 0.679, 0.678, 0.679, 0.694, and 0.671. At the completed trace, the simple
fixed CUSUM/`sw_var` combination is best in that screen at 0.798, while the
global head reaches 0.788 and IU28 0.764. At 512 tokens, fused Global-Local
beats IU28 by an equal-family +0.066 [+0.043,+0.089]. The next online scorer
should therefore model the causal evolution of CUSUM and `sw_var`—magnitude,
persistence, slope, onset/change point, and stability—rather than reuse a
frozen maximum or locator output.

The existing localization evidence remains the second anchor. GL-LIU v1
reaches 31.36% ProcessBench F1 versus 25.71% for the reproduced Mind the Gap
control; the later fixed trajectory-first IU package reaches 30.70% across
eight cells and 30.35% versus 24.96% on the matched Qwen3-8B four-subset
population. PRMBench step AUROC improves from 0.6136 for the old step-first
adapter to 0.6711 for trajectory-first IU. These are competitive label-free
results, not a claim of leadership over supervised PRMs or large critic models.

### Current architecture and ablation decision

Ordinary Global-Local IU-PCR is the primary simplicity baseline. The
Laplacian is a robustness/control arm, not the presumed contribution. On the
existing localization component table, global mixed ordinary IU reaches
0.791369 AUROC and global DUFS-LIU 0.793561, a gain of only 0.002193. For the
local head, ordinary top-five IU reaches 0.723303 detection AUROC and DUFS
0.723881, only +0.000578; the temporal Laplacian falls to 0.691528 and its
development localization gain fails confirmation. The main reusable value is
the Global-Local decomposition and the dynamic token features, especially
CUSUM and `sw_var`, not Laplacian complexity.

The next clean ablation must hold the feature matrix, preprocessing, IU
subspace, reducer, split, and calibration fixed while comparing `lambda=0`, a
uniform graph, a DUFS feature graph, and—only where temporally meaningful—a
temporal graph. Existing IU28-versus-GL-LIU comparisons are architecturally
confounded and cannot answer this question. A graph remains only if it provides
a material paired gain or a documented robustness benefit that pays for its
runtime and memory.

### Permanent evaluation contract for this direction

- **Localization panel:** answer-error AUROC, exact first-error localization,
  tolerance-one localization, clean-trace abstention, and ProcessBench F1.
- **Early panel:** equal-family AUROC/AUPRC on at-risk unfinished traces at
  16/32/64/128/256/512 tokens; correlation with the method's own final score;
  final-decision agreement; declaration coverage; held-out ever-wrong rate;
  and selective error.
- **Inference unit:** questions/response identities, not tokens. Splits,
  bootstrap intervals, and comparisons must preserve shared-question and
  repeated-model structure.
- **Label boundary:** scores, orientations, components, and feature selection
  are label-free. Labels may be used only in an explicitly declared
  development selection, threshold calibration, and final evaluation split.
  Consequently the current system is an unsupervised scorer with calibrated
  decision policies, not a wholly label-free policy.
- **Efficiency gate:** report feature count, fit cost, per-token update cost,
  wall time, peak memory, and incremental component ablations. Within
  statistical uncertainty, the cheaper architecture wins. No correlated
  duplicate or graph is retained without measurable incremental value.
- **Promotion gate:** preregister non-inferiority margins from development
  variability before model selection; require grouped evidence of improvement
  on at least one co-primary task, no material regression on the other, and no
  hidden family whose collapse is masked by the macro average.

The execution prompt used by Step 270 was
`docs/research_notes/reasoning_localization_early_detection_optimization_prompt_2026-08-16.md`.
Canonical result artifacts are
`results/early_online_existing_data_v1/REPORT.md`,
`results/early_online_localization_models_v1/REPORT.md`,
`results/ours_only_localization_v1/REPORT.md`, and
`results/fixed_application_pipelines_v1/REPORT.html`.

This application charter did not edit, rescue, delay, or reinterpret the
frozen A6/PTNI program below. If A6 work continues, it must follow its existing
stage boundaries exactly. The joint application program began from existing
caches; Step 270 completed that retrospective screen and did not
justify a new GPU collection.

## Active program — automatic group-free IU successor (Step 268)

The next core-method program is now explicitly reopened under a new-evidence
standard rather than as another static covariance sweep. The goal is an
automatic successor to IU-PCR/NRM that removes the manual provenance quotient
while preserving label-free fitting, one-pass inference, and an affine score.

The broader optimization goal does not require the winning method to be
self-supervised. A fully unsupervised method is preferable when it clears the
same target-identification, robustness, transfer, and performance gates;
self-supervision from mechanically generated interventions is also admissible.
Human or benchmark supervision is forbidden in fitting and selection. Natural
labels remain restricted to the explicitly frozen veto and confirmation
stages and may not adaptively choose a method. The deployable method remains
gray-box and one-pass: it may use the registered internal telemetry, but not a
second model call, an external judge, handwritten feature families, or a
semantic/provenance grouping supplied as prior knowledge. Success requires a
material held-data improvement over IU-PCR, not merely a structural or
simulation PASS.

The program separates measurement-structure recovery, target-component
identification, and orientation/trust. Its initial soft-factorial,
multi-environment, paired-view, and label-free continuous-structure routes were
audited in A1--A5 and closed. The deployable ladder now ends at mechanically
self-supervised interventions. Minimal-label orientation is outside the active
goal: it may remain a supervised diagnostic ceiling, but it cannot fit, select,
orient, or promote the deployable method.

All alternatives receive a premise test, but only frozen finalists may touch
new confirmation labels. Existing original, ProcessBench, SemGrad, PRMBench,
HLE, and RAGTruth evaluations are retrospective development surfaces. The
current Family-NRM stays frozen as comparator and may not select or orient the
automatic method.

Canonical contract and experiment registry:
`docs/experiments/AUTOMATIC_GROUP_FREE_IU_RESEARCH_PROGRAM_V1.md`.

Phase A0 passes with no new labels beyond the frozen mixed-v2 IU input contract.
The audit contains 30 code-registered features and 23 source environments, including exact
missingness and feature-pair coverage. Seventeen features are universal. Six
cells have fewer valid bundle rows than manifest attempts (minimum 19.8%), so
the live structural route uses the bundle population with equal-environment
weighting. Most importantly, 3,400 fixed ProcessBench responses have exact
content-and-ID matches across Qwen3-4B, Qwen3-8B, and Llama3.1-8B telemetry.
That three-view surface gives A4 a genuine response-preserving nuisance
intervention on the scorer model. It does not change the response target and
therefore cannot by itself justify a self-supervised hallucination claim.

Phase A1 has been executed and closed as a detector basis. Its rank-6 soft
hybrid reduced equal-environment audit covariance MSE from 0.034704 to 0.032009 at the point
estimate and decisively beat random partitions, which supports the weaker
claim that mechanical channel/operator structure can regularize a learned
subspace. The seven-environment grouped interval still crossed zero, however,
and a rho=0.999 duplicate received 3.009 times the original feature's combined
mass. The method therefore fails the frozen robustness premise and may not be
promoted into A3. Hard channel, operator, and factorial bases were substantially
worse, so no manual-like hard quotient is being retained through the back door.

Phase A2 is also closed as a detector basis after a missing-aware 30-atom
nested audit. JBD reached MSE 0.028700 versus 0.032864 for a pooled-PCA control
with the same block sizes, mechanism count, and ridge, but the paired interval
[-0.012164, 0.000838] crossed zero and the LOEO mechanism-rank ratio 0.618
missed its 0.70 gate. Fold structure was unstable, including one all-singleton
fold, and the advantage disappeared under a PSD stationary null. A3 closes by
premise because both inputs failed.

Phase A4 is now closed. Its frozen CorrCA component reached 0.997881 Qwen
repeatability and 0.955465 held-Llama correlation, but the nested-selected
`single:1 = trace_length` baseline reached 0.966908. The paired CorrCA-minus-
baseline delta was -0.011444 [-0.016036,-0.009034], failing the mandatory
material-improvement gate. Post-held adversarial diagnosis showed that the
CorrCA loading on trace length was 0.997897--0.999279 across folds. Because the
nuisance design predicted standardized linear token count only from log-count
terms and their squares under ridge shrinkage, it left a deterministic length
residual; the two Qwen count views are exactly identical. Trace-only reproduced
the baseline, while deleting the frozen trace term reduced held-Llama
correlation to 0.866653. Thus neither formal confound-gate success nor coarse
length-decile null success establishes a non-length shared mechanism.

The correct A4 conclusions are
`CLOSE_SHARED_REPEATABLE_COMPONENT_PREMISE` and
`CLOSE_NO_TARGET_CONTRAST`. The remaining post-held ablated correlation is not
a registered candidate and cannot be promoted. A5 continuous weak supervision
has now also closed at its sealed nuisance hard stop. A0--A5 artifacts
are frozen under
`results/automatic_group_free_phase_a0_v1/` and
`results/automatic_group_free_phase_a1_v1/`, and
`results/automatic_group_free_phase_a2_v1/`, and
`results/automatic_group_free_phase_a4_v1/`, and
`results/automatic_group_free_phase_a5_v1/`.

The frozen A4 protocol remains
`docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A4_V1.md`; it is not edited
retroactively. Its post-held interpretation correction is preserved in the
result report and `POST_HELD_TRACE_LENGTH_DIAGNOSTIC.json`.

Phase A5 is restricted to one new route: an IU-anchored equal-covariance
continuous latent mixture with a sparse within-component precision and an
affine discriminant. It uses the automatically defined 17-feature complete
core, excludes trace length and every A4 component from fitting, and contains
an exact IU fallback. A required observational-equivalence audit makes the
semantic boundary explicit: the mixture can recover density structure only;
its target interpretation is conditional on the inherited IU anchor.

The protocol was corrected through three independent adversarial passes. It
now purges globally repeated prompt content across outer and inner environment
folds, uses one deterministic response per prompt, rebuilds a target-free raw
tensor, compares against capacity-identical `alpha=0`, diagonal, one-Gaussian,
and training-selected matched-random controls, and refits three full-pipeline
null families. A sealed nuisance-dominant synthetic is an early stop before
large real-cache transfer. Retrospective labels may only veto one frozen score
bundle and cannot tune or select it. Canonical protocol:
`docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A5_V1.md`.

The A5 implementation completed its first sealed decision stage, and A5 is
now closed. All 100 nuisance-dominant seeds ran against the committed boundary;
98 were usable and two produced registered numerical nonconvergence, giving
the frozen formal verdict `CLOSE_NUMERICAL_NONCONVERGENCE`. The usable-only
adversarial analysis independently failed every semantic gate: final and
correction target preference were only 62/98 and 25/98, while candidate minus
IU averaged -0.038484 with 95% interval [-0.047495,-0.029659]. Alpha 1 was
selected 46 times and lost 0.080974 AUROC on average, demonstrating that the
likelihood selector repeatedly trusted the stronger planted nuisance.

No real cache or retrospective label was accessed. S1b and the real A5 stage
must not open; numerical repair would not address the decisive semantic
failure. The program therefore advances to A6. Its key new resource is
interventional asymmetry: separately verified target-changing pairs and
nuisance-only pairs must teach an affine IU-anchored direction that transfers
to natural hallucinations without interventions at deployment. Existing RAG
evidence-ablation results are premise evidence, not a new confirmation result.

A6 is now preregistered under one target ontology and one candidate. Ordinary
task-answer correctness is the designated parsed answer assertion; contextual
support and RAG faithfulness are adjacent tasks. The candidate, reciprocal
PTNI-IU, scores a complete 2x2 crossover of two verified task worlds and their
two deterministic answers. This balances every prompt/response marginal while
holding answer bytes fixed across the target contrast. Three verified AST
mutation families and three semantics-preserving rendering families provide
factorial target, nuisance, and interaction effects without feature groups.

The learned atomic error direction is nuisance-whitened, projected exactly
IU-orthogonal in each target's unlabeled covariance, and deployed as one affine
mixed-v2 score. The trust grid contains exact IU fallback. The protocol adds a
sealed simulator, complete-block admission, a conditional sign-permutation
test, two split-local placebos, a nuisance-as-target control that must first
prove nonvacuous nuisance recovery, LO target/nuisance families, a frozen Llama
quartet audit, and untouched greedy Llama errors before any retrospective label
or PopQA confirmation may open.

Eight adversarial review passes resolved prompt-marginal confounding,
teacher-forced-to-natural transport, closed answer parsing, target-local
normalization, unique final selection, missing-roster controls, alpha-zero
fallback, null exchangeability, bootstrap grouping, and matched-control
capacity. The final verdict is `NO BLOCKERS`; no A6 telemetry or result has
opened. Canonical protocol:
`docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A6_V1.md`.

Step 265 implements an explicitly **unsealed development base**, not an A6
result boundary. The mechanical layer now constructs and independently verifies
the reciprocal arithmetic, relational aggregation/lookup, and finite
set/counting tasks; preserves complete rejection ledgers; and rejects semantic
AST or raw-prompt reuse across folds and the Qwen/Llama populations. The
numeric layer implements the exact mixed-v2 target-local transform, factorial
PTNI moments, leave-one-nuisance moment fits, nominal-roster transport,
duplicate quotient IU, orthogonal correction, and one fixed affine fallback.
All 55 A6 development tests pass. A full 1,800-group proxy build also passes
with the available Qwen2.5 tokenizer, including the compact 40--80-token
certificate, but the pinned Qwen3/Llama tokenizers have not been frozen or run.

Do not open telemetry or sealed simulator seeds from this base. The next
boundary must jointly bind the three 2,000-prompt natural cohorts and PopQA
identity reservation, derived-answer quotas, exact tokenizers, shortcut audit,
manifest-bound duplicate evaluation preflight, feature-permutation
canonicalization, nested LO selection/controls/nulls, all eight simulator
worlds, and append-only execution. It requires a fresh independent no-edit
review. The small pre-telemetry protocol clarification in Step 265 is explicit:
“affine” means one head over the frozen mixed-v2 transformed coordinates, not
over raw telemetry values that undergo nonlinear frozen transforms.

Step 266 freezes the missing execution layer before boundary implementation.
The adversarially reviewed S0a/S0b/S1 contract binds exact model/tokenizer
revisions, the required future content-addressed snapshot/hash procedure,
the contextual chat-prefix/response-span tokenization rule, all
quartet/natural/PopQA identity schedules, folds and null strata, the S0b
shortcut and matching
procedures, complete nested PTNI/control/LO/null selection, eight explicit
simulator worlds, separated robustness arms, deterministic RNG bytes, and
append-only stage provenance. Its exact pre-freeze body SHA-256
`5c869db42633d04bf4c46110d95de83891c6ca6b10fdf381653b8a618a750615`
received the independent verdict `NO BLOCKERS`.

This is still a preregistration rather than an experimental result. The pinned
Qwen3/Llama tokenizer artifacts have not been loaded, no S0 boundary has been
prepared, and no telemetry, correctness sidecar, PopQA field, or sealed S1 seed
has opened.

Step 267 implements S0a exactly from that contract. The S0a implementation now
provides the complete 1,800+6,000 schedules, contextual response-span tokenizer
audit,
global disjointness replay, folds/null strata, recursive target firewall,
content-addressed offline tokenizer/config manifest machinery, effective
EOS/pad resolution rules, append-only checkpoints, missing-only resume, and
full-replay verification. Five stable-hash adversarial rounds closed ledger forgery,
numeric JSON type substitution, unmanifested payloads, symlink escape, and
interruption recovery; 89/89 relevant tests pass and the final verdict is
`NO BLOCKERS`.

This remains implementation evidence, not an S0a result. The immediate action
is a read-only availability check for all three exact tokenizer revisions. If
they are present, prepare a new empty source/runtime/input boundary and subject
that concrete artifact to independent verification before executing S0a. If
any revision is unavailable, record `BLOCKED_TOKENIZER_ACCESS` without
substitution or download-driven drift.

Step 268 completes that action. All three tokenizer/config snapshots were
authenticated against the frozen official revisions and Git/LFS identities,
the self-contained boundary was independently cleared, and authoritative replay
returned `PASS_S0A`. The run contains 1,800 reciprocal quartets, 6,000
prompt-only natural rows, 7,200 inner-fold assignments, 36 null cells, and
7,800 contiguous checkpoints. No response telemetry, natural response,
correctness sidecar, PopQA content, or sealed S1 seed was opened. The active
next stage is A6-S0b, not detector evaluation.

### Conditional successor after A6 — PTNI-guided Neutral Residual Mode (queued)

The prospective note
`docs/research_notes/ptni_guided_nrm_research_proposal_2026-08-14.md` is now an
explicit roadmap task after the frozen A6/PTNI outcome. It does not modify or
delay A6 and is not a rescue route if PTNI fails to identify or transfer a
target direction.

The preferred candidate is the note's Option C: project the nested PTNI
steering direction into a permutation-calibrated atomic neutral-residual
subspace, remove its target-local IU component, and retain exact `alpha=0`
IU-PCR fallback. A new protocol must compare the hybrid directly with frozen
PTNI-IU, Family-NRM, Atomic-NRM, cardinality-matched random projectors, and
norm-matched PTNI shrinkage.

The proposal receives a mandatory trigger assessment after the frozen A6
outcome. Experimental execution opens only if A6 establishes a valid
intervention-derived target direction and exposes a preregistered stability,
redundancy, or nuisance-transfer limitation; otherwise the assessment records
a principled closure rather than using NRM as a rescue. Its decisive question
is incremental: whether the NRM projector adds reproducible held-family/held-
scorer value beyond PTNI alone, not merely whether the combined method beats
IU-PCR. Projector definition, projection order, trust path, nulls, and all
decision thresholds require a separate preregistration and independent no-edit
review before any result is opened.

The untouched confirmation boundary is now PopQA with Gemma-3-4B-it, with a
pre-sealed Qwen3-4B fallback if gated checkpoint access fails before any
collection. The primary correctness rule uses normalized token-boundary alias
matching; official substring matching is a secondary diagnostic only.

---

## Fixed application packages — Step 253

The immediate application decision is now concrete: use one shared
token-resolved mixed-v2 feature basis, but preserve the natural structure of
each task before IU-PCR fusion.

- **RAG:** represent the fixed answer under full context, no context, and each
  available leave-one-chunk-out condition as `X[i,t,c,f]`. Build fixed evidence-
  drop blocks from the same base feature streams, then fuse them with two-
  component IU-PCR. Aggregate the resulting token curve only after fusion.
- **Reasoning:** represent the uninterrupted trace as `X[i,t,f]`, fuse the
  complete token trajectory first, and only then take the maximum risk inside
  each reasoning step. A registered global/local gate handles ProcessBench's
  no-error case.

The shared contract contains 29 token streams and covers every one of the 30
original mixed-v2 global features. CUSUM magnitude and location share one
stream. Eighteen streams have an exact whole-trace reduction; eleven use
documented causal rolling analogues. This is the agreed feature basis for both
packages, not a replacement Evidence-Contrast feature set.

The final application heads use **IU-PCR**, not DUFS-LIU-PCR. This is an
evidence-based simplification: DUFS/Laplacian remains a useful control, but its
increment beyond IU-PCR was approximately zero in the completed Original-30
LOO, PRMBench, and broad-localization controls. The task axes themselves carry
the useful structure here: evidence condition for RAG and token order for
reasoning.

Current evidence is competitive but bounded. The RAG score slightly exceeds
the local GASP reproduction on the same 400 responses (sentence AUROC 0.6598
versus 0.6556) and preserves the earlier Original-30 LOO response macro. The
reasoning score beats Mind the Gap on all eight ProcessBench cells and on the
matched Qwen3-8B four-subset macro (F1 0.3035 versus 0.2496; paired delta
+0.0539 [0.0316,0.0773]), while remaining
close to frozen GL-LIU (0.3125). Trajectory-first fusion improves PRMBench IU
AUROC from 0.6136 to 0.6711, but supervised PRM remains 0.7983.

**Next research action:** review and freeze this package with the advisors,
then confirm it on a benchmark whose labels were not used in prior development.
Do not tune another RAGTruth or ProcessBench fusion variant. Any future DUFS
claim must show an incremental gain over the corresponding IU head under the
same feature matrix and protocol.

Canonical artifact: `results/fixed_application_pipelines_v1/REPORT.html`.

---

## Confirmed bounded exception — Neutral Residual Mode (Steps 247--250)

The HARP-inspired contribution-space investigation found the first justified
exception to the generic fusion freeze below.  It does not copy HARP's hidden-
state projection or supervised classifier.  Instead, it decomposes the existing
IU-PCR score into six exact provenance-family contributions, removes their
linear IU component, and uses the unit-variance null of standardized residuals
to distinguish a neutral target-correction mode from redundant and strongly
shared nuisance modes.

The resulting **NRM-CS-IU** calibration is label-free and frozen.  It selects
the residual-covariance eigenvector closest to eigenvalue one, orients the
remaining global sign toward an equal-family confidence anchor, and adds the
target residual at fixed scale `1/G`.  It requires no new model inference or
feature and remains one affine weight vector over the mixed-v2 matrix.

The frozen PRMBench/Qwen3-8B response-level confirmation passed all five
registered gates: IU AUROC 0.720602 to NRM 0.725206, **+0.460pp**, with paired
`source_idx` bootstrap interval **[+0.068,+0.841]pp**.  Retrospective transfer
was also positive on original LOFO, both ProcessBench scorer families, and
SemGrad.  HLE was directionally positive but underpowered, so it supplies no
separate confirmation claim.

**Research decision:** retain NRM-CS-IU as a confirmed, frozen response-level
addition to IU-PCR.  Do not generalize this result to PRMBench's official
step-level metric, claim that NRM universally replaces DUFS-LIU, or reopen
static graph/selector sweeps.  The next useful test is native task validation:
carry the unchanged calibration into a preregistered step/localization protocol
or a new naturally distributed response benchmark with enough positives, and
compare IU, DUFS-LIU, and NRM under the same label boundary.

Canonical artifacts: `SPEC_NEUTRAL_RESIDUAL_MODE_CS_IU_V1.md`,
`SPEC_NEUTRAL_RESIDUAL_MODE_PRMBENCH_CONFIRMATION_V1.md`,
`results/neutral_residual_mode_cs_iu_v1/REPORT.md`, and
`results/neutral_residual_mode_prmbench_v1/REPORT.md`.

## Provenance-family audit — Atomic NRM rejected (Step 251)

The provenance partition is no longer an unnamed implementation detail. It is
an explicit empirical inductive prior required by the current NRM evidence.

The group-free audit fixed the obvious high-dimensional defect before testing:
instead of selecting one arbitrary eigenvector closest to one, Atomic NRM used
a 1,000-permutation simultaneous null band, retained the full two-dimensional
neutral subspace, and projected a symmetric inverse-dependence anchor into it.
The candidate, feature exclusions, scale `1/sqrt(17)`, direction, and hashes
were frozen before labels. It is label-free, permutation-equivariant to feature
order, affine in the existing mixed-v2 matrix, and adds no inference.

It nevertheless lost consistently. Equal-group AUROC changes versus IU were
-0.667pp on original LOFO, -1.106pp on Llama ProcessBench, -1.305pp on Qwen
ProcessBench, and -4.216pp on SemGrad. Frozen family NRM was positive on the
same four domains (+0.277, +1.580, +0.557, +1.310pp). Direct contrasts favor
family NRM with intervals excluding zero throughout.

This is not merely a dimensionality or information-loss result. Fifty random
partitions matched to family sizes, deterministic refinements/coarsenings, and
dependence-learned groups do not reproduce the family rule. Conversely, a
class-balanced supervised atomic head has *more* headroom than the family head:
at prior 0.3, +1.298pp versus +0.721pp over IU, direct +0.577pp
[+0.102,+0.910]. The missing ingredient is label-free target orientation, not
atomic target signal.

**Research decision:** keep NRM-CS-IU v1 frozen and state its exact assumption:
aggregate feature contributions by the fixed measurement-provenance registry
`FEATURE_TO_VIEW` / `VIEW_ORDER` before neutral-residual calibration. Do not
replace it with Atomic NRM, learned dependence clusters, or arbitrary matched
partitions. Do not consume a new held-out target for a formula already rejected
at the retrospective gate. De-grouping may reopen only with a new label-free
analogue of a target/steering model; eigenvalue-null geometry alone is closed.

Canonical audit:
`docs/research_notes/atomic_nrm_grouping_audit_2026-08-13.md` and
`docs/research_notes/atomic_nrm_null_spectrum_literature_2026-08-13.md`.

## Current research decision — August 2026 session close

### Freeze the fusion core and pivot to applications

The current algorithm-development cycle is complete. The bounded conclusion is:

> Further fusion changes are not a justified priority for the current
> single-pass, static answer-feature pool. Use **DUFS-LIU mixed-v2** as the
> common scoring core and move the contribution to how that core is applied to
> structured hallucination tasks.

This is not a theorem that U-PCR can never be improved. It is a decision based
on the completed evidence. Sparse-error recovery, condition-controlled inverse
weights, SpecRaGE and cross-view fusion, learned micro-views, atomic operators,
sample-local family gates, alternating diffusion, per-feature transformations,
repeated-measurement reliability, and deployed-style hard prefiltering either
lost to IU-PCR/DUFS-LIU or produced changes far below the uncertainty of the
comparison. Stable graphs, factors, groups, and bootstrap covariance were
repeatedly found. None supplied the missing label-free link from structure to
hallucination correctness.

The final 24-cell filtering check strengthens this decision. Full-pool
mixed-v2 DUFS-LIU remains best at 0.776562 macro AUROC. Applying deployed
U-PCR's `rho_max/3` hard filter lowers it to 0.774249, and the strictest tested
filter lowers it to 0.764153. DUFS and estimated rho already agree strongly on
feature importance (median Spearman 0.794), so deletion mainly removes
features that DUFS already downweights softly while also discarding their
covariance information. Previous IU-PCR, DUFS-LIU, and deployed-U-PCR scores
were reproduced exactly in all 24 cells.

The implementation standard going forward is the frozen `mixed-v2` feature
contract. Historical stable-only results remain in their original reports for
audit; they are not a reason to create more feature-contract versions.

### Application priority 1: hallucination localization

**Step-269 scope update:** localization is now co-primary with early/online
final-error detection under the joint program at the top of this file. Frozen
GL-LIU remains a comparator; it is no longer an instruction to assume a
Laplacian head in the optimized architecture.

The strongest result of the cycle came from changing the task decomposition,
not the covariance solver. Global full-trace fusion decides whether an error is
present; a token-resolved head decides where it begins.

Frozen GL-LIU v1 uses global mixed-v2 DUFS-LIU and a temporal LIU locator. It
raises ProcessBench F1 from 25.71% for the reproduced Mind the Gap control to
31.36% across eight model/dataset cells, and from 24.74% to 30.76% across the
six cells excluded from component selection. The eight cells reuse four
dataset families across two model sizes; they are not eight independent
datasets.

The factorial follow-up gives a simpler application candidate: use DUFS-LIU in
both heads, with the frozen five token curves in the local head. It reaches
31.72% F1 and 31.41% on the six non-selection cells. The +0.37-point change
over GL-LIU v1 is descriptive and mixed, not a confirmed replacement. The
broad 28-curve local pool falls to 29.03% and is rejected.

The next localization work should optimize the **application**, not invent a
new fusion family. Start with the simpler ordinary-IU global/local heads,
preserve frozen GL-LIU and the core-five DUFS head as controls, develop
token/window/span onset outputs, and evaluate every change simultaneously on
the causal early-detection panel. Keep the broad-28 local pool closed unless a
new token-resolved feature has an explicit localization hypothesis and earns
its incremental cost.

Canonical artifacts: `docs/methods/gl_liu_v1.md`,
`results/ours_only_localization_v1/REPORT.md`, and
`results/gl_liu_factorial_v2/REPORT.md`.

The published-method and benchmark map is now explicit. Mind the Gap remains
the only external method measured in the frozen shared-protocol artifact, but
it is not the only relevant method in the field. The first missing label-free
peer to audit is **Unsupervised Process Reward Models (uPRM)**. Supervised PRMs,
automatically supervised PRMs, and critic models should be reported in separate
access categories. See
`docs/research_notes/reasoning_localization_methods_and_benchmarks_2026.md`.

**Step 237 (2026-08-09): external-family confirmation launched.** Llama-3.1-8B-Instruct
teacher-forced ProcessBench scoring (Gate-B generation cell submitted, job 173188) tests
frozen GL-LIU v1 against the unified core-five DUFS-LIU candidate with zero selection on
the new family's labels. `scripts/gl_liu_external_v1/run.py` is built and dry-run validated.
See HISTORY.md Step 237 and `cluster/manifests/pb_llama31_8b_external_v1.json`.

**Step 238 (2026-08-09): first real result, mixed.** Full 4-subset run completed
(3,400 rows), scored with labels opened after hash-freeze. gl_liu_v1_frozen
31.71% macro F1 clearly beats the Mind the Gap reproduction (25.45%, +5–10pp
every subset — genuine transfer) but is a statistical wash against the
simplest transparent baseline, max token entropy (31.50%, sign flips per
subset, macro gap 0.21pp — noise at ~850 rows/subset). unified_core_five_dufs
(31.62%) is likewise a wash against frozen v1. Read this as "beats the
published competitor, does not clearly beat doing nothing clever" rather than
a confirmed win — see HISTORY.md Step 238 for the per-subset table.

### Queued application: hallucination in RAG citations

#### Current exploratory direction: original-30 LOO IU-PCR

A follow-up experiment now answers the intended question without replacing the
base method. It extracts the exact 30 mixed-v2 features from `full`, `noctx`,
and every `loo_j` condition, fits the mixed-v2 coordinate system on unlabeled
full rows, and summarizes the within-response evidence changes.

The original full-only detector is not a general RAG score: test AUROC is
0.7698 on QA but 0.4345 on Data-to-Text. Evidence perturbation repairs this
failure. Original-30 LOO IU-PCR reaches 0.7178 on QA and 0.7150 on
Data-to-Text, raising task-macro AUROC from 0.6002 to 0.7164. The paired macro
gain over full-only IU-PCR is **+0.1163 [0.0795,0.1544]**. GASP-top50 remains
slightly higher at 0.7225 task-macro; the difference is not resolved.

DUFS is not the main source of this repair. It adds **+0.0065
[0.0047,0.0085]** task-macro in the compact no-context matrix, but essentially
zero in the 175-column LOO matrix and Hybrid. Evidence-block permutation
causes a large loss, while graph regularization barely changes the high-
dimensional IU solution. The useful mechanism is paired evidence change in the
original features; the large DUFS/Laplacian expansion is redundant.

RAGTruth labels were already open, so these values are exploratory. Do not
promote the pooled no-context score of 0.8013: it hides a Data-to-Text AUROC of
0.5851. The next confirmation should freeze **Original-30 LOO IU-PCR** and
compare it with GASP-top50, EC-IU-PCR, and compact no-context DUFS-LIU on a new
benchmark or scorer. Canonical report:
`results/ragtruth_mixed_v2_evidence_aware_v1/REPORT.html`.

#### Earlier registered Evidence-Contrast result

The first RAGTruth Evidence-Contrast experiment is complete. It kept every
published answer fixed and rescored it under full context, no context, and
leave-one-chunk-out context. Evidence sensitivity was treated as the desired
signal rather than nuisance variation. Score fitting was label-free and
transductive; dev and test scores were hashed before labels were opened, and
all uncertainty grouped complete `source_id` units.

The main result separates feature construction from graph fusion. On 12,958
LOO test sentences, EC-DUFS-LIU reaches **0.7026 AUROC** versus **0.6721** for
GASP-top50, a paired **+0.0305 [0.0237,0.0378]**. The sign is positive in both
QA and Data-to-Text. However, ordinary EC-IU-PCR reaches **0.7031** and beats
EC-DUFS-LIU by 0.00048 with a non-zero paired interval. Ungated and permuted
graphs also tie IU-PCR. The Evidence-Contrast contract worked; DUFS and the
Laplacian did not add the claimed mechanism.

The leading RAG method is therefore **EC-IU-PCR**, not EC-DUFS-LIU. EC-U-PCR,
GASP-LL and GASP-top50 remain required controls. This does not change the
frozen DUFS-LIU mixed-v2 standard for the separate 24-cell intrinsic-detection
benchmark. It means that RAG evidence did not rescue the graph mechanism.

A separately hashed post-hoc response audit reconstructed the old intrinsic
mixed-v2 detector. Its pooled AUROC is 0.7629, but the task slices are 0.7698 on
QA and 0.4345 on Data-to-Text. EC-DUFS-LIU reaches 0.7056 on Data-to-Text. The
old pooled value is therefore a task-composition artifact, not evidence that
intrinsic mixed-v2 transfers as a general RAG detector. Because this audit was
added after labels opened, it cannot change the registered decision.

The current application risks determine the next work:

1. Freeze the EC contracts and IU solver. Do not tune another graph or feature
   contract on the opened RAGTruth test labels.
2. Test transfer to a new benchmark or scorer. RAGTruth++, TofuEval and
   TRIVIA+ are candidates; short-answer RAGBench is a deliberate failure test.
3. Target **conflict** hallucinations, where sentence AUROC is only 0.6256,
   versus 0.7438 for baseless information.
4. Repair or narrow the response-level claim. Residualizing length, chunk count
   and context length changes pooled response AUROC from 0.7484 to 0.6481, and
   Data-to-Text response AUROC is below GASP-top50 despite the stronger pooled
   result.
5. Treat localization as sentence ranking in this version. A future citation
   or exact-span claim requires token/claim-level outputs and the corresponding
   benchmark metric.

The old Phase-10 RAG cache is engineering evidence only. Its labels use weak
substring/citation fallbacks, only 19.2% of answers contain citations, and only
10.4% have the gold answer in the first 15 prompted documents. Its AUROC values
must not be presented as a publishable grounding benchmark.

Canonical result: `results/ragtruth_evidence_contrast_v1/REPORT.html`.
Mathematical definition: `results/ragtruth_evidence_contrast_v1/METHODS.md`.
Original plan: `docs/research_notes/evidence_contrast_upcr_rag_direction.md`.

**Step 237 (2026-08-09): implementation launched.** RAGTruth vendored
(`data/ragtruth_protocol/`), the preregistration frozen
(`docs/research_notes/ragtruth_ec_preregistration_v1.md`), and the Qwen2.5-1.5B
Gate-B generation cell submitted (job 173189). The evidence graph is a NEW
exogenous construction — chunk-text TF-IDF similarity, not score covariance —
built in `spectral_utils/evidence_contrast.py`. Measured finding: RAGTruth's
Summary task has zero natural paragraph structure in 100% of test rows, so
the leave-one-chunk-out condition only has real coverage on QA and Data2txt
this round. See HISTORY.md Step 237.

**Step 239 (2026-08-09): first real result — promising, not confirmed.** Full
test split scored (N=2,700, hashes frozen before labels). Point estimates
favor the new evidence-graph fusion (arm 5b, response AUROC 0.7536, best of
all 7 arms), but the preregistered novelty test (grouped bootstrap by
`source_id`, arm vs the fusion-isolation naive-average ablation) puts it at
+2.51pp with 95% CI [−0.58pp, +5.72pp], P(Δ≤0)=0.066 — just above
conventional significance. The temporal-chain arm (the preregistration's own
"default") and EC-U-PCR are both statistically indistinguishable from naive
averaging. The evidence-contrast intervention design itself IS confirmed
useful (full-context-only and raw likelihood-drop are both significantly
worse than naive averaging), so this is not a null campaign — it's a
specific "does the Laplacian fusion earn its complexity over naive
averaging" claim that hasn't cleared the bar yet, closest on the newest,
least-validated graph construction. GASP-threshold reproduction (0.7137)
essentially matches the paper's own Qwen2.5-1.5B number (0.713), confirming
the reproduction's fidelity. A real sign-convention bug was found and fixed
in the same pass (`scripts/rag_ec_v1/run.py`, `anchor_sign`) — three arms
initially looked like they scored far below chance; they were inverted, not
weak. See HISTORY.md Step 239 for the full table and the caught-bug account.

The consolidated RAG field map is
`docs/research_notes/rag_localization_methods_and_benchmarks_2026.md`. It
identifies GASP as the closest direct perturbation competitor, RAGTruth as the
primary span benchmark, TRIVIA+ as a long-context and label-noise confirmation
candidate, RAGBench as a planned short-answer failure test, and L-CiteEval as a
later citation-specific benchmark. Supervised token classifiers, external
claim verifiers, and mechanistic white-box methods must remain separate
comparison categories.

### What is paused

- no broader `lambda`, `k`, factor-count, family, or feature-transformation
  sweep on the existing 24 cells;
- no additional hard-filter or gating sweep: the ordinary `rho_max/3` filter
  reduces mixed-v2 DUFS-LIU from 0.776562 to 0.774249 and changes DUFS's
  incremental contribution over IU-PCR from +0.048 to -0.025 AUROC points;
- no new static graph, selector, local gate, or covariance decomposition built
  from the same answer-level feature matrix;
- no promotion of repeated-measurement Wiener filtering: it changed DUFS-LIU
  by only +0.0006/+0.0013 AUROC on GSM8K/MATH, with both paired intervals
  crossing zero;
- no claim that the Laplacian is the main answer-detection gain: its global
  effect remains small, and the RAGTruth application now also shows a
  statistically negative EC-DUFS-LIU minus EC-IU-PCR difference;
- no tuning of EC features, `k`, or `lambda` on the opened RAGTruth test set.

The advisor decision is now about application scope and validation data, not
which additional U-PCR variant to implement.

### Earlier 24-cell fusion decision

**Baseline update:** DUFS-LIU had still been running on `fixed_stable_v1`, so
the four non-monotone views were removed. A complete 256-contract development
search selected a mixed next-run contract: `pe_mean=squared`,
`stft_spectral_entropy=mode`, `cusum_shift_idx=raw`, and `rpdi=raw`. The
retrospective score is 0.776562 versus 0.774139 stable-only (+0.242pp). LOFO is
only +0.123pp and falls to about +0.022pp without one MATH-500/Qwen cell.
Therefore mixed-v2 is frozen for external confirmation but does not replace the
historical headline or reopen static-graph fusion as the leading contribution.
See `docs/research_notes/dufs_liu_mixed_feature_contract_conclusion.md`.

The broader research junction below is unchanged.

Repeated Cross-View Alternating-Diffusion IU-PCR (RCV-AD-IU-PCR) is complete
and closes static repartitioning of the current feature matrix as the leading
direction. The registered dependency-blocked method tied IU-PCR at +0.004pp,
10 wins and 14 losses, with equal-family interval [-0.052,+0.029]pp. Random
and provenance-family partitions were also control-level ties.

This was a valid mechanism test, not an execution failure. Sixteen repeated
partitions converged: median graph CKA was 0.536 and the T=8/T=16 output
Spearman was 1.000. Increasing k largely repaired graph disconnection without
improving AUROC. Stronger lambda made the result worse. Stable cross-partition
geometry is therefore another static proxy that does not identify correctness.

**Current decision:** retain the Step-229 finding that family expertise changes
across IU-PCR score regimes, but do not build another graph from partitions,
families, operators, or stability of the same matrix. A next method requires a
genuinely independent view such as a separate generation, an evidence view, or
a controlled perturbation. If single-pass inference is a final requirement,
first demonstrate the mechanism with the independent view and only then test
whether it can be distilled.

Full conclusion:
`docs/research_notes/repeated_cross_view_diffusion_conclusion.md`. Frozen
report: `results/repeated_cross_view_diffusion_v1/REPORT.md`.

### Previous junction: conditional family relevance

The graph-coupled family relevance diagnostic gives a more precise research
junction than the earlier static-geometry failures. The main scientific premise
is now **partly supported**: different feature families are best in different
IU-PCR score regimes. A label-only diagnostic family expert has +2.833pp
equal-family headroom across frozen IU-PCR-rank quartiles, with Holm
`p=0.006`. This is evidence of conditional specialization, not a deployable
algorithm.

GCFR-U-PCR, the attempted label-free router, is rejected. It used
within-family oriented-rank agreement and a fixed semantic family Laplacian to
make sample-local gates. The registered path lost 0.135pp to IU-PCR and
0.243pp to the same gate without graph smoothing. Every positive graph
strength was negative on average. The gates were active, and the registered
semantic graph did not beat a permuted graph. Measurement relatedness is not
the same as shared reliability for hallucination correctness.

**Current decision:** do not fit a learned mixture to the 24 development-cell
labels. Retain IU-PCR rank as a regime coordinate and require an independent
interventional self-supervised signal before another router is built. Repeated
generations, benign prompt or decoding perturbations, evidence-conditioned
answers, and semantic answer consistency are candidate observations. The next
frozen premise test must ask whether such an observation predicts which family
expert helps inside held-out cells and feature families. Coherent repeatable
hallucinations are the explicit failure case.

Full conclusion:
`docs/research_notes/family_relevance_diagnostic_conclusion.md`. Frozen report:
`results/family_relevance_real_v1/REPORT.md`.

### Previous junction: static atomic geometry

The Phase-0 atomic-operator premise audit closes **AOG-IU-PCR for its registered
proxy**. Median within-cell association between the label-free proxy and atomic
usefulness was -0.312. The top-proxy atom lost -0.838pp cell-macro, with a
-3.658pp worst cell. Only 3 of 15 continuation gates passed. Every registered
`k` and `lambda` setting remained negatively associated with usefulness.

The proxy was numerically stable, the graphs were valid, and a label-only
oracle showed +0.447pp optimistic atomic headroom. The missing component is
therefore not graph construction or gate optimization. It is label-free target
identifiability: stable, self-consistent geometry in the static feature matrix
does not identify hallucination correctness.

The incumbent remains confidence-oriented U-PCR/IU-PCR, based on Tenzer et
al.'s continuous spectral regression model. DUFS-LIU, uniform atomic fusion,
and the atomic operator code remain required controls and diagnostics. DUFS is
not the next learner: differentiable gates cannot repair a wrong objective.
CA-SpecRaGE, micro-view learning, and AOG from the current proxy are closed as
leading extensions for this feature bundle.

The next research junction is to find an **independent interventional
self-supervised target**. Repeated generations, benign prompt or decoding
perturbations, evidence-conditioned answers, or semantic answer-consistency
views may add information outside the static covariance. The first task is a
literature-and-data design and a frozen premise test, not another fusion model.
Systematically consistent hallucinations are the main falsification case.

Full conclusion:
`docs/research_notes/atomic_operator_premise_audit_conclusion.md`. Frozen
report: `results/atomic_operator_premise_audit_v2/REPORT.md`. Historical
SpecRaGE and AOG plans remain for audit and must not be read as live methods.

---

## Earlier thesis framing (historical; not the current method definition)

The section below records the earlier L-SML framing. It is retained for
history, but it must not be presented as the current leading algorithm. The
current method definition is GL-LIU v1 above.

**Claim**: Spectral features of the per-token entropy trajectory H(n) — fused via the Spectral Meta-Learner (L-SML; Jaffé–Fetaya–Nadler 2016) — detect LLM hallucinations at state-of-the-art AUROC in a single forward pass, with no ground-truth labels at inference time.

**Operating regime**: The method works on reasoning-heavy generation (MATH-500, GSM8K, multi-hop QA) where the entropy trace is long enough (≥100 tokens) to carry discriminative spectral structure. Performance is reduced on short factual QA traces (<60 tokens) and MCQ formats where entropy dynamics are structurally suppressed.

**Why it's novel**:
1. Spectral features of H(n) — not hidden states, not verbalized confidence, not sampling ensembles
2. L-SML fusion is unsupervised: no labels used at inference time; feature directions calibrated once offline from aggregated empirical evidence
3. Single-pass (K=1): cost is fixed and independent of question difficulty

---

## Supervisor Connections

| Supervisor | Core connection |
|-----------|----------------|
| **Ofir Lindenbaum** | Spectral decomposition of uncertainty signals maps onto his core methodology (diffusion maps, multi-view spectral methods, VSDE). L-SML is a spectral fusion method applied to entropy signals. |
| **Bracha Laufer-Goldshtein** | The L-SML score is a continuous input to LTT/COIN calibration, turning AUROC (a ranking result) into a deployable detector with a formal false-negative-rate guarantee — the conformal chapter. |

---

## Core Method (what is settled as of Jun 2026)

### Step 1 — Feature extraction

From a single greedy forward pass, extract per-token entropy H(n). From H(n), compute spectral and time-domain features:

| Feature set | Features |
|-------------|----------|
| **GOOD_5** (primary candidate) | `epr`, `low_band_power`, `sw_var_peak`, `cusum_max`, `spectral_entropy` |
| **STABLE_H9** | GOOD_5 + `spectral_centroid`, `dominant_freq`, `rpdi`, `cusum_mean` |
| **All-16** | Full `FEAT_NAMES` list in `spectral_utils/feature_utils.py` |

All 16 features are implemented in `spectral_utils/feature_utils.py`. Feature count is **open** — the logistic oracle (Item 2) will bound the headroom from more features.

### Step 2 — Fusion via L-SML (CONT configuration)

`lsml_continuous_pipeline(feats, subset, FEATURE_SIGNS)`:
1. Pre-orient each feature: `x_i_oriented = sign_i · x_i` (offline consensus direction)
2. Z-score normalize
3. Binarize → L-SML rank-1 eigenvector → continuous cross-cluster score
4. Returns one real number per sample (higher = more likely correct)

**FEATURE_SIGNS** = orientation vector derived offline from majority vote across 29 cells (AUROC-weighted). Unsupervised at inference time — no per-sample labels used.

**Why CONT over binary**: continuous encoding beats the old `np.sign()` pipeline by **+4.9pp macro** (65.2→70.1%) and +7.2pp on the reasoning regime. The binarization was the largest single source of lost signal.

### What is NOT settled (pending meeting experiments)

- **Final feature set** (5 / 9 / 16) — logistic oracle (Item 2) will bound the headroom
- **Whether sampling fusion adds lift** — Item 5 tests SE K=10 + spectral
- **Whether temperature diversity matters** — Item 6 ablates same-T vs mixed-T multi-pass
- **Scope on factual QA** — Item 3 extends to CoQA, SQuAD v2, TruthfulQA (priority corrected Step 155 — published SE/SC baselines exist)

The final method choice has not been made. CONT + GOOD_5 is the current strongest result, not a decided thesis configuration.

---

## Current Best Results

*(Step 135, honest numbers — do not cite old Step-117 supervised numbers: 96.7/71.3/88.1)*

| Domain | Model | CONT AUROC | Competitor (K=10 sampling) |
|--------|-------|-----------|---------------------------|
| MATH-500 | Qwen2.5-Math-7B | **94.4%** | SE NLI 87.7%, SC 87.2% |
| GSM8K | Llama-3.1-8B | **75.6%** | SC 78.5%, SE 77.4% |
| Macro avg (29 cells) | multiple | **70.1%** | simple avg5 68.1%, oracle best-single 68.3% |
| Reasoning regime only (5 cells) | multiple | **78.3%** | — |

GPQA Diamond (MCQ science) is structurally out-of-regime: entropy dynamics are suppressed by the fixed-choice format. Phase 14 will compare against K=2 VC/SC/SCVC baselines (arXiv:2603.19118) on DeepSeek-R1-0528-Qwen3-8B.

---

## Completed Experiments

| Phase / Step | Description | Key result |
|-------------|-------------|-----------|
| Phase 1–3 (Steps 46–51) | Spectral features on GSM8K, 3 models | Best fusion 75.9% (Qwen-1.5B); sw_var_peak most robust |
| Phase 4 (Step 54) | MATH-500 + GPQA, 8 configs, T=1.5 | Honest best: 88.3% (Qwen-1.5B), 90.0% Qwen-7B T=1.0 |
| Phase 5 (Steps 56–58) | Temperature ablation T=1.0 vs T=1.5 | T=1.0 better for capable models; sw_var_peak temperature-stable |
| Phase 8 (Step 80) | GPQA / Qwen2.5-72B-AWQ | ~65% GPQA accuracy; spectral AUC modest — MCQ structurally limited |
| Phase 10 (Steps 85–91) | RAG — 4 models × 4 datasets (16 cells) | llama8b/hotpotqa **87.7%**, beats LOS-Net 72.92% unsupervised |
| Meta-analysis (Steps 89–91) | Cross-domain feature stability, 29-cell diagnostics | sw_var_peak most stable; epr dominant on math; rpdi/spectral_entropy on RAG |
| Steps 105–111 | Paper alignment: correct L-SML vs Step-100 leakage | Honest 65–91% math, 41–62% GPQA, 64–82% RAG (29 cells) |
| Steps 133–134 | 12-variant grid (CONT/PROD × encoding variants) | CONT best overall; encoding is the dominant lever; cross-weights always K=2 equal |
| Step 135 | Benchmarking vs SE/SC competitors | MATH-500: 94.4% vs 87.7%; GSM8K: 75.6% vs 78.5% |
| Step 137 | Theorem validation (branch `analysis/theorem-validation`) | HTML report + flowchart generated; **pending commit** |

### What we explored and moved on from

**Early EPR ensemble (Steps 27–45)**: 6-view Nadler ensemble (T=0.3/1.0/1.5/2.0 + Verify + Skeptic) on TriviaQA/WebQ with Falcon-3-10B — reached 81.5%/76.0%. This was the initial approach. We pivoted to spectral features because: (a) it requires 6 forward passes vs our 1, (b) the spectral framing is a cleaner contribution that connects to Ofir's methods.

**Supervised Step-100 numbers (96.6% MATH-500)**: had four methodological errors — (a) label-based sign orientation, (b) in-sample subset selection bias, (c) continuous features violating Lemma 1's binary contract, (d) M-matrix instead of rank-1 eigenvector. All corrected by Steps 105–110. The honest best is 90.0% (Qwen-7B, T=1.0).

**EDIS (Zhu et al. 2026, arXiv:2602.01288)**: Formula validated (Steps 41–42, spike ratio 4.02×). Not adopted as a core L-SML view — ρ(EDIS, spectral) too high on some cells. Useful as a comparison baseline only.

**Phase 13 (AMC23/AIME24)**: has `\boxed{}` grading bug — results invalid. Do not cite until fixed.

---

## Meeting Action Items — Jun 17, 2026 (Ofir, Bracha, Amir)

*Confirmed by email (Omri → Ofir/Bracha/Amir Jun 17 2026). These 6 items are the current experimental priority.*

| # | Action | Status | GPU? |
|---|--------|--------|------|
| 1 | L-SML follow-up literature search (Nadler post-2016) | ✅ Completed (Steps 139–141) | No |
| 2 | Logistic regression oracle (5/9/16 features, 5-fold CV) | ✅ Completed (Steps 142–143, 147) | No |
| 3 | Extend QA evaluation (CoQA > SQuAD v2 > TruthfulQA — priority corrected Step 155) | Nearly complete (Steps 160–171): CoQA/INSIDE full-N scored 68.4 vs 80.4 (floor caveat, Step 171); SQuAD v2 / NQ-Open / TruthfulQA / SciQ scored | Yes (AIRCC) |
| 4 | Benchmarking completion (Phase 12 Corrected run done — Step 152; 4 open issues before citable; QA + Phase 14 remaining) | In progress (Steps 160–171): one cell left in flight (ars_math500_qwen3 wall 3/4); A2 qwen3-GSM8K documented REJECT (ceiling + truncation leakage); report now carries CSV-driven figures (report_figs.py) + LOS-Net Table-1 baseline family incl. p(True) | Partial |
| 5 | Experiment 1 — sampling fusion: SE (K=10) + spectral features | ✅ Completed — verdict REVISED (Step 174): answer-agreement SC K=5 fuses with 1-pass L-SML to 95.2 [91.8, 98.0] (ρ +0.23) — gate PASSES; the Step-152 FAIL was the NLI-based arm | No |
| 6 | Experiment 2 — temperature variation: T effect + diversity ablation | ✅ Completed (Step 158) — diversity hurts, same-T sampling helps | Yes |

---

### Item 1 — L-SML Literature Search ✅ COMPLETED (Step 139)

**Result, corrected Step 226**: U-PCR (Tenzer, Dror, Nadler, Bilal, Kluger;
AISTATS 2022, [PMLR](https://proceedings.mlr.press/v151/tenzer22a.html)) is Nadler's
continuous-input extension of L-SML. Under the uncorrelated-error variant, covariance
off-diagonal `C_ij = rho_i + rho_j - g²` recovers expert-response covariances without labels.
The actual paper also contains **SU-PCR**, `C=L+S` with sparse correlated errors; the local file
named `Tenzer2022_...pdf` is the older Dror et al. 2017 paper and caused this extension to be
missed. Our CONT pipeline approximates **IU-PCR only**. Cite the PMLR paper and distinguish IU-PCR,
SU-PCR, and our tailored dependency-weighted extension.

Also found and deeply read (Step 141):

**FUSE** (Lee et al., arXiv:2604.18547, 2026) — applies Jaffe-Nadler moment structure to LLM verifiers for Best-of-N response selection with zero labels. Same theoretical base as our work (Jaffe et al. 2015). Different task: multi-response selection vs single-generation hallucination detection. Strong related-work citation. Critical finding for us: **our closed-form eigenvector weights (`w = (v₁ᵀρ̂/λ₁)·v₁`, then `score = w@F`) underperform naive equal-weight averaging in 7/10 FUSE benchmark settings** (Figure 3). FUSE's fix: pseudo-label logistic regression trained on MoM-estimated triplet posteriors `p̂(r_i)` — fully unsupervised (`p̂` never uses true labels). This is the single biggest available architectural upgrade to our pipeline. **Next experiment**: implement FUSE-style pseudo-label LR as replacement for `w@F` in `lsml_continuous_pipeline`.

**Positioning against FUSE (Step 147, both Ofir and Bracha flagged it).** Three concrete differentiators, in decreasing order of importance: (1) **Signal** — we fuse spectral views of *one* model's own entropy/probability trace (internal, single-pass, no extra compute); FUSE fuses scores from *many external verifier models*. (2) **Task** — per-answer hallucination detection (absolute, across queries) vs within-query Best-of-N selection. (3) **Dependence handling** — FUSE detects dependent verifier pairs (triplet-conditional-independence violation) and *transforms* the scores so a single spectral fusion is well-conditioned; ours runs *K-group spectral clustering* then hierarchical within-/across-group fusion. Net: FUSE innovates on the fusion; our contribution is the **signal**, so the two are complementary. The thesis must foreground the entropy-trace signal, not "unsupervised spectral fusion," to avoid overlap. (Memory: `project-fuse-positioning`.)

**Deep L-SML** (Shaham et al., ICML 2016, arXiv:1602.02285) — Lemma 4.1 proves our L-SML IS already an RBM: Dawid-Skene model = single-hidden-node RBM (bijective parameter map). Our covariance+eigenvector step = closed-form MoM training of that RBM. Stacked RBM (Deep L-SML) handles correlated features without exclusion — each hidden layer decorrelates the representation. Relevant if 16-feature expansion triggers heavy ρ > 0.75 filter exclusions (band-power pairs ρ 0.77–0.88). Still fully unsupervised (objective = `log P(features)`, no labels).

**STDR** (Aizenbud et al., arXiv:2102.13276, 2021) — hierarchical tree-structured dependency recovery via Fiedler vector, O(m² log m). Not relevant at 5–16 features; revisit if feature set expands to 50+.

**Empirical confirmation (Step 140)**: U-PCR ≈ L-SML continuous on 5/9 features (low correlation regime, assumption holds); L-SML wins on 16 features (band-power block violates U-PCR's uncorrelated-error assumption; clustering compensates).

Implementation: `upcr_fuse()` + `upcr_pipeline()` added to `spectral_utils/fusion_utils.py`. Comparison script: `scripts/run_upcr_comparison.py`.

---

### Item 2 — Logistic Regression Oracle ✅ COMPLETED (Steps 142–143, 147)

Fit supervised logistic regression on our feature sets to upper-bound what any fusion method can extract from the same features.

**Setup**: 28 common LR-valid cells; `sklearn.LogisticRegression(class_weight='balanced')`, 5-fold stratified CV with **per-fold AUROC averaging** (not concatenated OOF — see `SUPERVISED_ORACLE_CORRECTION.md`).

**Result (Step 147, common-cell macro AUROC)** — supervised LR beats unsupervised L-SML everywhere once corrected:

| Feat set | L-SML (CONT) | LR bal-CV | gap | in-sample ceiling |
| :-- | :-: | :-: | :-: | :-: |
| GOOD_5 | 64.2% | 68.9% | +4.7pp | 70.5% |
| STABLE_H9 | 62.9% | 66.8% | +3.8pp | 73.7% |
| ALL_H16 | 64.1% | 67.8% | +3.6pp | 79.3% |

- **Per-domain**: gap ~0 on reasoning (both near the ~84% ceiling), +4.9pp GPQA (ceiling 60.9%), +5.8pp RAG+QA (ceiling 69.5%). The gap is largest exactly where the feature ceiling itself is low → **features are the bottleneck, not the fusion**. This lands in the "< 5pp on reasoning / moderate elsewhere" interpretation band: L-SML is near-optimal where the signal exists.
- **"5 features best" explained** (`scripts/lr_convergence.py`): the named sets are non-nested (STABLE_H9 drops `spectral_entropy`, a top-3 feature), and in a proper nested ranked sweep the CV is flat from k=5 to k=16 (~68–69.5%) while the in-sample ceiling climbs to 79.3% — the extra features overfit rather than generalise. The same 9-feat dip appears in the unsupervised L-SML, so it is a feature-composition effect, not a supervision artifact.
- **LR vs L-SML weights** (`scripts/lr_weight_analysis.py`, answers Bracha Q4): correlate only weakly (Spearman ≈ 0.1–0.2, ~0.32 on GPQA). Both lean on epr/spectral_entropy/cusum_max but weight them differently — the features are correlated/redundant, so the weighting is underdetermined and both reach similar AUROC through different routes.

**Scripts**: `scripts/logistic_oracle.py` (oracle + `logistic_oracle.png`), `scripts/oracle_report.py` (common-cell tables + `oracle_feature_count.png`), `scripts/lr_convergence.py` (`lr_convergence.png`), `scripts/lr_weight_analysis.py` (`lr_weight_agreement.png`). No GPU needed.

---

### Item 3 — Extend QA Evaluation

**Priority corrected (Step 155)** — pick datasets with published SE/SC baselines so results are directly comparable (AmbigQA/PopQA have none):
1. CoQA (SE-ICLR primary dataset — 8K dev, published SE numbers; INSIDE 80.4 EigenScore reference)
2. SQuAD v2 (includes unanswerable questions — tests specificity; INSIDE reference 81.5)
3. TruthfulQA (hallucination-specific benchmark; LapEigvals + HSAD references)

**Setup**: folded into the Step-155 replication grid — AIRCC inference-only presets per competitor protocol (K, T, prompt, labeling locked per paper), all scoring local CPU. Loaders (CoQA, SQuAD v2, NQ-Open, TruthfulQA, SciQ) are the implementation follow-up.

**Decision gate**: ≥3 of 4 datasets show CONT AUROC ≥ 65% → method extends credibly to factual QA domain.

---

### Item 4 — Benchmarking Completion

**Done (Step 135, old caches)**:
- MATH-500 / Qwen-Math-7B: CONT **94.4%** vs SE NLI 87.7% / SC 87.2% (K=10) ✅
- GSM8K / Llama-8B: CONT **75.6%** vs SC 78.5% / SE 77.4% (K=10) ✅

**Done (Step 152, Phase 12 Corrected — fresh shared caches, paper-accurate baselines)**:
- GSM8K / Llama-8B: **L-SML 1-pass 0.754 beats every multi-pass baseline** (SCGPT-official 0.701; D-SE/LW-SE/SC K=10 all ≈0.61). Third run at 75.4–76.0.
- MATH-500 / Qwen-Math-7B: L-SML 0.230 = global sign flip (no `anchor_orient`; flipped ≡ 0.770 — still far below the 94.4 old-cache number, unresolved). SC K=10 wins at 0.863.
- GPQA / Qwen2.5-7B: all sampling baselines at chance (0.50); VC 0.428; L-SML 0.553 best.
- RAG×4: SelfCheckGPT below chance everywhere (official 0.24–0.44 < hard 0.32–0.48) — orientation/grading investigation needed.
- ⚠ Fresh-cache SE/SC baselines collapse vs old Phase 12 (GSM8K SC 78.5→60.8, SE 77.4→61.4; GPQA SE 70.6→50.1; MATH SE 87.7→63.0 with SC stable). NLI truncation on long traces is the prime suspect. **Neither table is citable until reconciled** — see PROGRESS.md Priority 1.

**Still needed**:
- **QA datasets**: SelfCheckGPT / Semantic Entropy comparison on same model + dataset (WebQ, TriviaQA)
- **GPQA Phase 14**: re-run Cell 9 with `DeepSeek-R1-0528-Qwen3-8B` at T=0.6. **Fix `boot_auc(n_boot=1000)` kwarg bug first.** Compare L-SML@K=1 vs VC/SC/SCVC@K=2 from arXiv:2603.19118:

| Method | K=2 AUROC |
|--------|----------|
| VC | 77.0 ± 2.0 |
| SC | 64.8 ± 3.0 |
| SCVC | 80.3 ± 1.5 |

Notebook: `Spectral_Analysis_Phase14_GPQA_Comparison.ipynb`.

---

### Item 5 — Experiment 1: Sampling Fusion

Fuse Semantic Entropy (K=10 generations) with single-pass spectral features.

**Primary question**: does SE K=10 (10× compute) add meaningful lift on top of CONT K=1?
**Secondary question**: does spectral (K=1) + SE (K=10) beat SE alone? — tests whether single-pass spectral adds orthogonal signal beyond the sampling budget.

**Dataset**: MATH-500 / Qwen2.5-Math-7B (T=1.0 cache exists) or GSM8K / Llama-8B.

**Decision gate**: ρ(SE score, CONT score) < 0.75 AND fused AUROC > max(CONT, SE) + 1pp → complementary signals; claim: "single-pass spectral provides cheap orthogonal signal to sampling-based methods."

**✅ COMPLETED (Step 152) — gate NOT passed.** Fusion = L-SML GOOD_5 + LW-SE as 6th view in `lsml_continuous_pipeline`, run inside Phase 12 Corrected:

| Cell | ρ(L-SML, LW-SE) | L-SML alone | LW-SE alone | Fused | Gain vs max |
|------|-----------------|-------------|-------------|-------|-------------|
| GSM8K / Llama-8B | 0.263 | 0.754 | 0.613 | 0.758 | +0.4pp — FAIL |
| MATH-500 / Qwen-Math-7B | −0.251 | 0.230 (sign flip) | 0.625 | 0.232 | invalid (flip) |
| GPQA / Qwen2.5-7B | −0.188 | 0.553 | 0.501 | 0.573 | +2.0pp — passes numerically, but LW-SE is at chance |

- **Primary answer**: SE K=10 (10× compute) adds ≈nothing on top of 1-pass spectral on reasoning.
- **Secondary answer**: the orthogonality runs the other way — spectral adds **+14.5pp** on top of LW-SE (GSM8K). Supports the "cheap single-pass signal" framing, but as spectral rescuing SE rather than SE lifting spectral.
- MATH-500 fusion must be re-run with `anchor_orient` before the row is usable (PROGRESS.md Priority 1).

---

### Item 6 — Experiment 2: Temperature Variation

**Questions from the meeting**:
1. Does higher temperature improve detectability? (Plot CONT AUROC vs T)
2. Does multi-temperature fusion gain from diversity or just from more passes?
   - **Condition A**: K=5 at T=1.0 (same T, more passes)
   - **Condition B**: K=5 at T∈{0.3, 0.6, 1.0, 1.5, 2.0} (different T, same K)
   - If B >> A: temperature diversity is the source of lift
   - If A ≈ B: multiple passes alone explain the gain; T doesn't matter

**Setup**: Qwen2.5-Math-7B / MATH-500. ~~Existing caches: T=1.0 and T=1.5~~ — **claim corrected (Step 157)**: no reusable raw cache exists for this cell. The T=1.5 88.3% cell is Qwen-**1.5B**; Step 148 established MATH-500/Qwen-7B has no raw entropy-trace cache anywhere; Phase 12 Corrected `p2` predates the Step-149/150 grading fixes and has no top-k logprobs. → **All 9 runs fresh** (5 temps + 4 extra T=1.0), each saving the full raw-data schema. T=1.0 run0 doubles as the canonical raw-trace cache for this cell, repaying the Extension E data debt.

**Status (Step 158) — ✅ RAN on Colab A100. Both pre-registered gates FAIL; the negative result is clean and interpretable.**

Results (9 runs × 200 MATH-500 / Qwen2.5-Math-7B; full narrative in HISTORY Step 158; consolidated `cache/phase15_temperature/results/phase15_results.pkl`):

- **Q1 — AUROC vs T (single-pass L-SML-continuous, GOOD_5)**: inverted-U — 0.545 / 0.644 / 0.851 / 0.878 / 0.629 at T = 0.3 / 0.6 / 1.0 / 1.5 / 2.0 — **confounded by accuracy collapsing 80% → 4%** across the curve, so the "peak" partly reflects the shifting class mix, not detectability alone. T=2.0 is underpowered (8 correct). **G-T1 FAIL** (T=1.5's higher 0.878 has overlapping CIs and sits at 27.5% acc).
- **Q2 (primary) — diversity vs more passes**, paired on the 200 common samples (labels = T=1.0 run0):
  - **AUC(A: K=5 same-T=1.0) = 0.912**, **AUC(B: K=5 multi-T) = 0.859**, single-pass base 0.851.
  - paired **AUC(B) − AUC(A) = −0.053 [−0.103, −0.011]** → **G-T2 FAIL, sign negative** — temperature diversity *hurts*.
  - paired **AUC(A) − AUC(base) = +0.061 [+0.004, +0.128]** → more same-T passes *help*.
  - Mechanism: A off-diagonal Spearman ρ +0.45 (same signal + independent noise → averaging denoises); B off-diagonal ρ +0.01, but that decorrelation is the off-temperature passes being *near-random* (T=0.3/0.6 weak, T=2.0 degenerate), not independent true signal.
  - **Answer to the meeting question**: A ≈ B is refuted in the *unfavourable* direction — the multi-pass lift is **variance reduction from repeated sampling at a single good temperature (T≈1.0)**, and mixing temperatures dilutes it. Temperature is not the lever; repeated sampling is.
- **Two method flags surfaced (not fatal, → follow-up)**: (1) `spectral_entropy` sign is temperature-dependent — AUROC 0.261 @ T=1.0 / 0.140 @ T=1.5 with the fixed −1 sign (i.e. informative if flipped); (2) the label-free L-SML fusion **underperforms the best single feature at every T** (fused 0.851 vs `cusum_max` 0.927 @ T=1.0; fused 0.545 vs `cusum_max` 0.811 @ T=0.3) because the `epr` anchor is weak at low T (0.681 @ T=0.3) → fragile global-sign orientation. The low-T "poor detectability" in Q1 is plausibly a fusion/anchor artifact, not a signal property.

**Data-debt repaid**: T=1.0 run0 is now the **canonical MATH-500/Qwen-7B raw-trace cache** (entropies + spilled energies + top-50 logprobs + token ids, N=200, 70.5% acc) — closes the Extension E gap.

**Follow-up experiments on this data — all CPU once the 9 caches are downloaded** (prioritised):

1. **Self-consistency / semantic-entropy baseline** (highest value; also closes Item 5). ✅ **DONE (Step 174)** — answer-agreement SC K=5 fused with 1-pass L-SML → 95.2 [91.8,98.0], gate PASS (ρ +0.23, +10.1pp over best single arm).
2. **K-sweep for Condition A**: AUROC(A) at K = 1..5 — does the same-T lift saturate at K=3? ✅ **DONE (Step 181)** — 0.851/0.869/0.863/0.905/0.912 for K=1..5; no early saturation, a dip at K=3, most of the lift arrives at K=4-5. Repeated sampling needs close to the full K=5 budget on this cell.
3. **Anchor / sign robustness across T**: re-fuse with (a) a stronger, more T-stable anchor (`cusum_max`), (b) per-feature label-free sign via each feature's own anchor, (c) leave-`spectral_entropy`-out. **BLOCKED (Step 181)** — needs raw per-sample GOOD_5 feature values at T≠1.0; `phase15_results.pkl` only stores scalar per-(feature,temp) AUROCs. Needs `math500_qwen7b_T{0.3,0.6,1.5,2.0}_run0.pkl` copied from Drive to `local_cache/`.
4. **New feature families from saved-but-unused data**: (a) spilled-energy suite; (b) top-50 logprob features (margin, varentropy, Rényi entropy, tail mass). ✅ **DONE (Step 181)** — `cusum_max_spilled` (AUROC 0.909) fused as a 6th view clears the Item-5-style gate (+1.13pp over GOOD_5, CI excludes 0) — a second genuine complementary signal. New `topk_tail_mass`/`varentropy`/`renyi_entropy_2` logprob features added (`repgrid_scoring.logprob_features_extended`); `topk_tail_mass` (AUROC 0.902) fusion gain +0.72pp is CI-significant but below the 1pp gate (near-miss).
5. **Fairer diversity set**: re-run B dropping the degenerate T=2.0 (and maybe T=0.3), e.g. B′ = {0.6, 1.0, 1.5}. ✅ **DONE (Step 181)** — B′ simple-avg 0.881 / L-SML 0.856, statistically indistinguishable from a matched K=3 same-T arm (0.863; CI spans 0). Confirms the negative Q2 result isn't an artifact of the degenerate passes.
6. **Cross-temperature probing**: does a hot pass's entropy trace predict the *cold* (T=1.0) answer's correctness? ✅ **DONE (Step 181)** — every hot T predicts its OWN label far better than the COLD label (e.g. T=1.5 own 0.878 vs cold 0.626; T=0.3 own 0.545 vs cold 0.388, anti-predictive) — mechanistic reason mixing temperatures hurts fusion: each pass's signal is entangled with its own generation, not a stable per-question difficulty read.
7. **Length-controlled AUROC per T**: hot traces are longer/degenerate — partial out trace length to confirm the spectral signal isn't just length. **BLOCKED (Step 181)** — same data gap as #3 (needs per-sample trace length at T≠1.0).
8. **Streaming earliest-prefix replication (Extension E)** — now unblocked by the fresh raw cache; run absolute-budget prefixes on the T=1.0 run0 traces. Still open — not attempted.

Items 2/4/5/6 results: `results/repgrid/phase15_followups.json` (Step 181). A couple (K-sweep beyond K=5, more temperatures for the pooling curve) would need a small extra GPU run; everything else is local CPU.

---

## Meeting Action Items — Jul 2026 (Ofir, Bracha)

*Ofir and Bracha were pleased with the results shown; FUSE is not considered blocking. The one
concrete action item from this meeting: add a new contribution to the algorithm, and the chosen
candidate is a principled, label-free, in-pipeline **feature-subset selection step** — see Extension G
below. Bracha also raised conformal calibration; **explicitly parked** (still Extension A, unchanged
priority).*

| # | Action | Status |
|---|--------|--------|
| 1 | Feature-subset selection step: literature survey (Lindenbaum FS line, Nadler portfolio, tabular foundation-model frontier, assumption diagnostics) + assumptions audit + candidate designs | ✅ Research memo complete (Step 185) — `docs/research_notes/feature_subset_selection_landscape.md`; see Extension G |
| 2 | Avoid the fixed subset — add a label-free selection step ahead of fusion | ✅ **Done, and it is a tie rather than a win.** Two arms reach the hand-picked bar carrying only the anchor bit: `upcr.rho_polarities` **0.7551** and `a2.dufs_pf` **0.7507**, against GOOD_6 0.7594 / GOOD_5 0.7519. No gap is significant (Steps 186–206, standing page Step 207) |
| 3 | Review Ofir's earlier FS work (DUFS, GroupFS) | ✅ Both benchmarked on the same 25 cells: `a2.dufs` 0.7502, `a2.dufs_pf` 0.7507 (Eq. 7, no lambda to tune), `a2.select` (GroupFS) 0.7481 |
| 4 | Assumptions audit of the L-SML / U-PCR line, in the spirit of FUSE on verifier dependence | ✅ Seven real deviations found; fixing all of them **hurts** (69.1% faithful vs 73.9%). Independence assumption tested head-on and refuted (−4.5pp). The one thing that helped came out of the same audit: polarity from `sign(rho)`, **+1.5pp, 20W/5L, p < 0.001** (Step 204) |
| 5 | Bracha's conformal calibration suggestion | ⬜ **Not started, deliberately** — parked until the selection track closed. Still Extension A, unchanged priority. ITCR (arXiv:2606.08831, ICML 2026) is the paper to read first; **split conformal with *marginal* coverage**, never "conformal risk control", and never claim a per-instance guarantee |

**Advisor-facing deliverables for this block**: `results/upcr_study/comparison.html` (all 196 variants,
sortable, with a provenance column) and `results/action_items/labelfree_standing.html` (the two
label-free arms scored per-cell against the published roster, Step 207). Both self-contained.

**Reporting rule established in Step 207, do not regress**: the "+8.7pp on 11 cells, p = 0.042"
external result is **Bar B** (unsupervised, one forward pass, *any* access — it includes white-box
competitors). It is **not** our own cost class; that is Bar A, grey-box only, where the numbers are
+6.17pp / +6.56pp over 5 cells at p = 0.312. Quote Bar B as what it is.

---

## Meeting Action Items — Jul 30, 2026 (Ofir, Bracha, Amir)

*The **feature-selection line is closed**. L-SML over the full ~30-view pool, L-SML after DUFS
selection, and U-PCR's own built-in exclusion all tie, on essentially every cell — Step 207's
`upcr` 0.7551 / `dufs_pf` 0.7507 / GOOD_6 0.7594, no pairwise contrast significant. Three action
items replace it. Recorded in HISTORY.md Step 209.*

| # | Action | Status |
|---|--------|--------|
| 1 | **Understand why we fail where we fail** — per-cell deep dive, not another aggregate | 🔵 **ACTIVE — diagnosis only.** Nine cells pinned as "failing" (below). Repairs are pre-registered and tested in a *later* step so the diagnosis cannot be tuned to make a fix look good |
| 2 | Consider a clustering mechanism inside U-PCR | ❌ **Already answered — do not rebuild.** Step 204 §D built it (`spectral_utils/upcr_clustered.py`): failed both pre-registered gates, **−4.46pp (9W/16L, p = 0.030)**, and the premise was a confound (2.03× same-vs-cross fit gap → 0.97–1.00× matched on \|C_ij\| decile; a random partition reproduces it). One untried variant — K-means on the (v₁,v₂) coordinates — rated **low**: `lambda2_threshold` is inert and one-component U-PCR is exactly PC1 of the survivors, so the second component has nothing to cluster on |
| 3 | Consider adjacent applications — localization, and detection early in generation | ✅ **Superseded by Step 269.** Steps 232--235 established localization first; Step 269 now makes localization and early detection co-primary. GL-LIU v1 reaches **31.36% ProcessBench F1** versus **25.71%** for Mind the Gap, while the new 11-cell causal prefix screen shows IU/DeepConf parity at 64--128 tokens and motivates joint optimization rather than deferral. |

> ### ⚠ STEP 216 INVALIDATES TWO ROWS OF THE TABLE BELOW — read this first
>
> **Two of the nine "failing" cells were not failing; they were generating malformed traces**, and
> they are exactly the two *base* checkpoints in the grid (23 of 25 run instruct-tuned models).
>
> * **`seiclr_triviaqa_opt30b`** (`facebook/opt-30b`) — no learned EOS for the few-shot format, so
>   99.7% of generations sit at `max_new=64` and run on past a 3-token median answer. The grader
>   already cropped (`first_answer_line`) while the features did not, so label and features were
>   computed over different spans. **REPAIRED by cropping**: `GOOD_6 0.5884 → 0.8311`, `DUFS+L-SML
>   0.5614 → 0.7726`, `U-PCR 0.5751 → 0.8119`, best single view ~0.62 → **0.8258**. Against
>   SE-ICLR'23's published **83.0** it is now level rather than ~27pp below. **It was never a method
>   failure.**
> * **`inside_coqa_llama7b`** (`huggyllama/llama-7b`) — a chat template applied to a base
>   checkpoint; 45.1% of answer spans are `[/INST]` echoes or fabricated turns, `pos_rate` 0.002 on
>   the broken rows vs 0.239 on the usable ones. **REJECTED** (`scripts/inscope_cells.REJECTED_CELLS`)
>   until re-generated with `raw_prompt=True`.
>
> **The roster is 24, and `GOOD6_EXPECTED` is 0.7733, not 0.7594.** Both leading arms moved:
> **U-PCR + sign(rho) 0.7551 → 0.7741**, **DUFS+L-SML 0.7507 → 0.7687**, GOOD_6 0.7594 → 0.7733 —
> and `GOOD_6 − U-PCR` is now **−0.08pp (p=0.819)**, the first time the label-free arm sits
> nominally above the hand-picked subset. **The QA deficit was these two cells**: GOOD_6's QA lead
> was 1.49pp over 10 cells and is 0.25pp over 9.
>
> **Consequences for the diagnosis**: seven of the nine remain genuinely hard and the mechanism work
> in Steps 210–215 stands for them. But the per-cell account of `seiclr_triviaqa_opt30b` in Step 215
> (selection miss + L-SML sign disagreement on 2 of 12 views) is **retracted** — it was measuring
> the run-on artifact. Two rows below are kept for the record with their pre-repair numbers.

**The nine "failing" cells** (eight named by Omri off `results/action_items/labelfree_standing.html`,
plus TruthfulQA, which ranks 4th of 25 on every weakness measure and is interleaved with the named
cells) — **numbers below are PRE-Step-216**:

| Cell | Page name | GOOD_6 | DUFS+L-SML | U-PCR |
|---|---|---:|---:|---:|
| `losnet_hotpotqa_mistral7b` | HotpotQA / Mistral-7B-v0.2 | 0.5810 | 0.5684 | 0.5696 |
| `inside_coqa_llama7b` | CoQA / LLaMA-7B (base) | 0.6674 | 0.5320 | 0.5355 |
| `seiclr_triviaqa_opt30b` | TriviaQA / OPT-30B (base) | 0.5884 | 0.5614 | 0.5751 |
| `truthfulqa_llama8b` | TruthfulQA (gen.) / Llama-3.1-8B | 0.6572 | 0.6606 | 0.6634 |
| `internalstates_gsm8k_qwen25_7b` | GSM8K (T=0.8) / Qwen2.5-7B | 0.7036 | 0.6911 | 0.7082 |
| `noise_gsm8k_phi3mini` | GSM8K / Phi-3-mini | 0.6801 | 0.6764 | 0.6831 |
| `trace_math500_qwenmath15b_k10` | MATH-500 (K=10) / Qwen2.5-Math-1.5B | 0.6760 | 0.6901 | 0.6861 |
| `ars_gsm8k_r1distill8b` | GSM8K / R1-Distill-Llama-8B | 0.7623 | 0.7142 | 0.7385 |
| `lapeigvals_gsm8k_llama3b` | GSM8K / Llama-3.2-3B | 0.7025 | 0.7087 | 0.6992 |

`losnet_hotpotqa_mistral7b` is multi-hop RAG, which Step 191 declared out of scope; it is still one
of the 25 and was named, so it stays in the diagnosis with that caveat attached.

**The confound this diagnosis must not fall into**: the nine cells are exactly the nine lowest
`anchor_auc` in the grid, and Spearman(anchor_auc, U-PCR AUROC) = **+0.967, p = 3.8e-15** — which
looks like an orientation story. It is not. **Spearman(anchor_auc, best_single_feature) = +0.975**:
`epr` is itself a pooled feature, so a weak anchor just means every view is weak on that cell. And
`h1_orientation_summary.csv` rules it out independently — the `allsigns` / `z2` / `raw` / `oracle`
anchor conditions all return **identically 0.7594** with `cells_below_0.5 = 0`.

---

### Extension K — Reshaping non-monotone views (Step 217, 2026-08-02) — 🔵 OPEN, next session

**Status**: the phenomenon is **confirmed**; one transform family was tested and **failed**; Omri's
call is that the line stays open because a non-monotone feature needs reshaping and the right
reshaping has not been tried yet.

**What is established** (`scripts/nonmono_shape_test.py`, `results/nonmono_transform/`):
- Under a fair test — isotonic (best *monotone* function) vs an unconstrained 10-bin map, both
  cross-fitted on the same folds, against each pair's own **label-permutation null** — **32 of 682
  cell × view pairs are genuinely non-monotone**, several at 5–7× their null on cells with thousands
  of rows (`semenergy_triviaqa_qwen3_8b`/`rpdi` **+0.1227**, n=4392, across-seed sd 0.0019).
- The effect is **cell-specific**, which is why a per-feature-mean gate missed it entirely. Recurrent
  views: `rpdi` (7 of 24 cells), `pe_mean` (6), `cusum_shift_idx` (6), `hurst_exponent` (3).
- **No label-free handle via marginal bimodality**: `Spearman(shape_gain, KDE peak count) = +0.014,
  p = 0.72`. P(y|x) can bend without the *marginal* density of x having two humps.
- **A correction to our own code**: `nonmono_gain` in `results/advisor_inscope/ladder_featdiag.csv`
  is **inflated** — `gap_ladder.py:64,220` folds `max(p, 1−p)` onto each fold's binned score, a
  one-sided noise floor (`Spearman(corrected gain, inflation) = −0.171, p=7e-06`). Do not quote it.

**What failed, and why that is not the end of it**: `|x − median|`, `x²`, `|Φ⁻¹(rank%)|` in
Replace/Add, chosen leave-one-cell-out, gave **G1 FAIL on both arms** (−0.07pp / −0.04pp) with G2
and G3 passing (the LOCO choice is stable on 92–100% of folds, so the test had power). But all
three are **symmetric and centred on the middle of the distribution**, and the actual curves are
not: `semenergy/epr_energy` is an inverted-U peaking at **decile 6–7**, `semenergy/rpdi` is
**W-shaped**, `se_nq_open/rpdi` peaks at the **edge** with an interior dip. Two further gaps: arm A
**held the DUFS selection fixed** (only 5 of 24 cells moved at all), and **"add a view carrying the
same information monotonically" was never tested**.

**Pre-registered next tests**, priority order:
1. **Fit the centre, do not assume it** — `|x − c|` with `c` chosen **leave-one-cell-out**, or
   centred on the **KDE mode** (`nonmono_shape_test.py` computes it label-free).
2. **Use the winning curve itself as the view** — the cross-fitted bin-mean map *is* the +12pp
   function; test whether a LOCO-fitted version transfers across cells. Strongest form of the idea.
3. **Re-run selection with the reshaped view in the pool.**

**The competing explanation to test against, not assume**: a single view's shape may be largely
**redundant** once 15–20 other views are fused. Test 2 above is what separates that from "wrong
transform family". Gate stays G1/G2/G3 on both arms.

---

## Future Extensions

Not the current priority. Ordered by proximity to the main thesis.

### Extension A — Conformal Calibration (Bracha chapter)

Convert the AUROC result into a deployable detector with formal guarantees.

**A1 — Frozen-weights scorer + detection metrics under class imbalance** (engineering prerequisite for A2/A3)

Our cells are heavily imbalanced (GSM8K 79% majority, RAG/hotpotqa 91% majority) — raw accuracy is meaningless. Build:
- `fit_lsml(calibration_batch)` → freeze cluster assignment, group weights, cross-weights, per-feature μ/σ/sign. Unsupervised, fit once on a representative batch.
- `score_one(features)` → true single-sample inference (current experiments are transductive: fit+evaluate same batch, valid for AUROC but not streaming deployment).
- `decision_report(scores, labels, τ)`: recall (detection rate / TPR), precision, F1, balanced accuracy, TPR@FPR(1/5/10%), AUPRC.

**A2 — LTT calibration**: split calibration (100) + test (100); find threshold τ with P(FNR ≤ α) ≥ 1−δ.

**A3 — Label-free calibration via PPI**: use model-generated pseudo-labels (Verify > 0.9 → pseudo-correct) + PPI correction for pseudo-label noise.

### Extension B — Agentic Flow (Ofir alignment)

3-step HotpotQA agent chain; fuse per-step EPR with AUQ verbalized confidence (Zhang et al. 2026, arXiv:2601.15703).

Key check: ρ(EPR_step, verbalized_conf) < 0.5 → fusion is viable.
Target: Φmin AUROC > 0.791 (AUQ paper best on ALFWorld).
Model: Qwen3-7B. No new infrastructure for spectral features — same `generate_full()` per step.

### Extension C — Hidden State Variance (VSDE connection, Ofir alignment)

Register a forward hook on a transformer layer; compute variance of hidden states across K=5 temperature-varied generations as an additional L-SML view alongside spectral features.
- Low effort: one hook, existing fusion infrastructure
- Direct connection to Ofir's VSDE (high-variance regions ≈ hallucination) and PRAE

### Extension D — VLM Hallucination Detection

Apply spectral features to visual language models; split visual-description tokens vs factual-claim tokens. Not started. Only if committee wants a multimodal chapter.

### Extension E — Streaming / Online Detection (pivot candidate — pilot ✅ COMPLETED, Step 148)

**Status**: Pilot run 2026-07-02, local CPU, pre-registered gates. **Verdict: pivot NOT supported in its original framing (G2 FAIL); one significant surviving thread.** Full narrative: HISTORY.md Step 148; explainer: `results/Streaming_Pilot_Explainer.html`.

**Hypothesis**: the spectral suite computed on growing prefixes of H(n) detects a failing CoT *while it is generated* — unsupervised, logprob-only — and beats a naive windowed statistic in that streaming regime.

**Competitor** (closest prior work): *Streaming Hallucination Detection in Long CoT Reasoning*, arXiv:2601.02170 (BUPT/NTU/SWJTU/RUC, **arXiv preprint Jan 2026**, no venue as of Jul 2026). SUPERVISED probes over intermediate **hidden states** (anchor + synchronization losses), step labels annotated by Claude-4.5; custom MuSiQue-derived long-CoT set (10k+ trajectories / 200k+ steps). Prefix-level AUC: LLaMA-3.1-8B 72.69 / Qwen2.5-7B 81.05 / R1-Distill-8B 92.18. Their own limitations: "not directly applicable to black-box or API-only settings" — exactly our setting. **Reproducible baseline**: DeepConf (arXiv:2508.15260, Meta, Aug 2025) lowest-group-confidence — black-box, computable on our cached traces, hence the primary bar (G2).

**Pilot results** (2 clean cells: GSM8K/Llama-8B n=200, MATH-500/Qwen-1.5B n=400 non-canonical; 2 R1/GPQA cells excluded — 99–100% truncated at 1024-token cap):
- **G1 PASS** — AUROC@50%-of-trace ≥ 95% of full-trace on both clean cells; 32 tokens ≈ 91% of full signal on GSM8K. Early signal is real.
- **G2 FAIL** — fused L-SML does not clear +2pp over the best DeepConf window at ≥2 absolute budgets on ≥2 clean cells. Over most of the trace, the fusion ≈ windowed entropy mean.
- **Surviving thread** — the only *significant* spectral edge (paired bootstrap) is in the **earliest 10% of the trace, on both clean cells**: +9.8pp GSM8K, +4.6pp MATH-500. Fusion helps exactly where windows starve.
- **G3 context** — our unsupervised GSM8K/Llama-8B 75.4 (L-SML-5) vs their supervised hidden-state 72.69 on the same model family (different benchmark + label protocol; context only).
- **E3/E4** — best causal monitor flags 38% of wrong GSM8K traces @10% FA, saving 28% of wasted tokens.

**Data debt exposed**: MATH-500/Qwen-7B (our ~90% cell) has NO raw-trace cache anywhere (Phase-12 K10 files are texts-only); no clean R1 cell exists (all capped at 1024 mid-`<think>`).

**Next steps (in order)**:
1. Colab re-inference: MATH-500/Qwen-7B + one R1 cell with ≥4096-token cap, saving `token_entropies` + top-50 logprobs (raw-data rule).
2. Replicate the earliest-prefix edge there — absolute budgets only (fractions need oracle length), n large enough for the paired test.
3. If replicated → reframe as **hybrid early-warning monitor** (spectral early / windowed late), not "fusion wins everywhere" (G2 refutes that).
4. Method: per-budget refusion is sign-unstable (anchor_orient mitigates; 16-feat still erratic) → fit fusion weights once at a reference budget offline, reuse across budgets.
5. Advisor decision: pursue hybrid framing vs fold streaming in as a thesis section.

### Extension G — Automatic Feature-Subset Selection (meeting priority, Jul 2026)

**Status**: Memo (Step 185) → **full multi-algorithm bench EXECUTED (Step 186, 2026-07-17/18)** →
**punch-list follow-ups + split-half honest oracle (Step 189, 2026-07-18) — see below, the motivating
+7.6pp prize itself is now known to be mostly winner's-curse** —
six label-free selector families implemented + benched on both pools (H16 51 cells, 46-view 19
repgrid cells) through one select→same-L-SML→AUROC harness with labels structurally unreachable
during selection. All results in `results/selector_bench/comparison.csv` + the dashboard
(`results/selector_bench/dashboard.html`); research note
`docs/research_notes/selector_bench_results.md`; no pass/fail gatekeeping — the researcher reads
the full leaderboard.

**UPDATE (Steps 194-195, 2026-07-22) — two supersessions of the Step-186 verdict below:**
1. **Best learned selector is now `a6.pl_dufs`** (pseudo-label-supervised gates, Omri's idea):
   macro 0.7524 on the 25 in-scope cells, significantly better than `a2.dufs` (+0.22pp,
   p=0.0273) and the FIRST label-free selector to nominally edge GOOD_5 (0.7519, p=0.17 n.s.).
   Both pre-registered gates FAILED (mechanism rho 0.207 vs 0.30 bar; effect below +1.0pp), so
   the claim stays "GOOD_5 parity, GOOD_6 gap not closed" — but it is the selector of record.
2. **A fixed subset that honestly beats GOOD_6 exists.** The sizes-3-5 exhaustive sweep over
   the 30-view pool (Step 194/195, `results/subset_sweep_c46/`) yields a LOCO consensus stable
   in 22/25 folds: `{cusum_max, logprob_margin, min_energy, spectral_entropy, topk_tail_mass}`
   — LOCO-honest +1.59pp vs GOOD_5 (19W/2L); vs GOOD_6 on the same 24 cells 0.7705 vs 0.7632
   (+0.73pp, p=0.029), sign label-free. Coverage 24/25 (`inside_coqa_llama7b` lacks the energy
   views). This REVERSES the Step-154 "LOCO cannot beat GOOD_5" verdict — the enlarged pool
   changed the answer. Pruning stays negative (LOCO drop list empty in all folds).

**UPDATE (Step 198, 2026-07-24) — the selection line is now measured out, and the bottleneck is renamed:**

3. **`GOOD_6` is unbeaten by every label-free selector, and it is a local optimum.** Post-fix
   seven-arm bench on 25 in-scope cells (`results/advisor_inscope/seven_arm_summary.csv`, one run,
   canonical `eval_subset_flex`): GOOD_6 0.7594 > D1_D2 0.7580 > D2 (PL-mRMR) 0.7573 >
   `a6.pruned_dufs` 0.7537 (**⚠ CORRECTED in Step 206 to 0.7514** — the old rows carried a stale
   `mu3` NameError on 11/25 cells; re-benched clean, which moves it BELOW `a6.pl_dufs` and GOOD_5
   in this ordering, see HISTORY Step 206 §D) > `a6.pl_dufs` 0.7527 > GOOD_5 0.7519 > D1 0.7506.
   D2 beats GOOD_5
   significantly (p=0.037) and beats every prior DUFS variant, but under LOCO-CV budget selection
   lands 0.7572, below GOOD_6, and its math edge is p=0.2114 (9W/6L). The best D2 configuration is
   GOOD_6 **plus one** selected feature at 0.7590, i.e. adding any selected feature to GOOD_6 hurts
   macro at every budget K=7..20 even with the budget chosen on test data.
4. **Adaptive-K (D1) is refuted.** Five label-free size rules tested against oracle K
   (`results/advisor_inscope/adaptive_k_validation_rules.csv`): best rule r_s = +0.007, p = 0.975.
   The residual correlating with AUROC (r=0.65) does not transfer to predicting the optimal size.
   `D1_alone` is the worst of seven arms. Closed.
5. **The one real win of the step is the pseudo-label seed rule**: `ANCHOR_PRIORITY`x4 -> `GOOD_6`
   (`A6_SEED_RULE` env, default `good6`) takes the pseudo-label from 0.7249/0.6821 QA to
   0.7594/0.7274 QA and removes 2 sign-inverted cells. A weak consensus target points the gates at
   the wrong features, it does not merely add noise.
6. **The bottleneck is estimation, not model capacity, and the QA deficit is two cells.** The
   per-cell supervised oracle is logistic regression, i.e. a stationary global linear model with
   fixed per-feature signs, and it reaches 0.7810 macro / 0.7524 QA on the same 30 features
   (`lr_oracle_audit.csv`, `fset=30`). So the linear class already contains a solution above us.
   The QA gap concentrates in `inside_coqa_llama7b` (0.667 vs oracle 0.826, INSIDE publishes 0.804:
   **estimation failure**) and `seiclr_triviaqa_opt30b` (0.588 vs oracle 0.720 while SE publishes
   0.830: **feature-coverage failure that no fusion change can fix**). The other 8 QA cells average
   ~0 gap to the supervised oracle. This kills the "stationary sign bottleneck" framing and the
   three methods proposed on top of it (regime-conditional signs, SNF, GMM density ratio).
7. **Next**: `SPEC_gap_ladder.md` (repo root) specifies a 7-rung gap-decomposition ladder at two
   feature sets with pre-registered kill-gates: `R3->R4` (supervised nonlinear vs supervised linear)
   kills the nonlinear directions if flat, `R3->R5` (oracle regime signs) kills the non-stationary
   sign direction. Both run with labels, so a negative is conclusive. Gemini implements
   `scripts/gap_ladder.py`, Claude reviews and analyses. Candidate follow-ons if sign recovery
   dominates: Z2 synchronisation on the pairwise-sign matrix (`sign(cov_ij)` estimates `s_i*s_j`)
   and a robust (Spearman / Tyler) covariance in place of Pearson.

**Step-186 outcome (headline numbers, superseded as above)**:
- **No learned selector beats the curated subsets.** c46/repgrid-19 macro: GOOD_6 0.7440 >
  top_macro_5 0.7364 > GOOD_5 0.7328; best learned = **GroupFS `a2.select` 0.7323 — a
  label-free TIE with GOOD_5** (first learned selector to reach it); everything else trails by
  1-6pp. On H16/51-cell every learned family lands 0.56-0.63 vs GOOD_5 0.671.
- **Pre-registered admissibility (A1.0)**: no label-free objective is globally admissible as a
  selection criterion; the relative Eq-14 residual is weakly admissible on repgrid/qa only
  (median Spearman −0.109/−0.17); the lsml-vs-upcr structural-residual router is NOT-USEFUL in
  every domain (worse than best-constant by 3-6pp). The ρ-filter refutation (Step 153)
  replicates as a family-wide pattern.
- **Clustering swap** (theorem-validation follow-up): GroupFS's discovered groups replacing
  L-SML's spectral clustering ≈ tie on GOOD_5 (0.717-0.728 vs 0.733) — clustering is not the
  bottleneck on the repgrid pool.
- The **+7.6pp RAG/GPQA oracle prize remains uncaptured** by every label-free method tried.

**Step-189 correction — the prize itself was mostly winner's-curse.** A split-half honest oracle
(`scripts/selector_splithalf_oracle.py`: bounded greedy search on held-out half A, refit + scored on
half B, R=10 splits × 51 cells) found the 0.7472-macro exhaustive-sweep oracle **collapses to 0.668
macro when fully honest — a statistical TIE with GOOD_5 (0.6692) on the identical splits.**
Per-domain, this lands exactly on the two domains the "+7.6pp prize" framing above was built on:
**RAG's claimed +14.1pp shrinks to ~+1.6pp honestly; GPQA's claimed +10.2pp shrinks to ~+1.6pp
honestly.** This retroactively explains the uniform Step-186–189 negative results (six selector
families, A1–A5, all failing to beat GOOD_5): the 65,536-subset exhaustive search (Step 153)
guarantees a large multiple-comparisons overfit at n≈100–500 per cell, so "no selector captures the
prize" was never really a selector-design failure. **Also this session**: an autopsy of `a2.select`'s
one catastrophic miss (`inside_coqa_llama7b`, −14pp) found GroupFS's gates saturate open (selects
100% of a 23-feature pool containing 7 anti-oriented features) on a severely imbalanced, small-n-style
cell — connects directly to the still-open Step-187 feature-sign-fix item. An mRMR hybrid (A5,
`spectral_utils/selectors/a5_mrmr.py`) salvages part of A4's "picks epr's clones" pathology on the
46-view pool (+0.57pp over bare epr) but not on H16, and still doesn't clear GOOD_5. Full detail:
HISTORY Step 189; `docs/research_notes/selector_bench_results.md` (split-half section).

**Next steps (revised)**: the selection direction's motivating premise needs re-scoping with
Ofir/Bracha — realistic honest headroom looks like ~1–2pp, not ~7–8pp, which changes whether further
selector-design investment is worthwhile at all. Before any further design work: (0) the Step-187
feature-sign fix (13/30 anti-oriented features) is the one still-open, concrete, likely-cheap win,
independent of the selection-prize question — do this first, it may already close some of the small
residual gap on its own. If selection work continues: (i) GroupFS on the 46-view pool remains the
best label-free tie-with-GOOD_5 result and needs no further justification to ship as a deployable
default; (ii) the D5-(ii) cross-cell signature router is the one design from the original memo never
attempted — lower priority now given (2) above suggests little headroom exists to route toward.

**Motivation**: 46 registered fusion features (`CANONICAL_POOL`); no fixed macro wins consistently —
GOOD_5, the documented main configuration, wins only 3/40 per-cell picks in the repgrid headline
comparison. In-cell oracle subset selection is worth **+7.6pp macro AUROC** over fixed GOOD_5 (0.747
vs 0.671, 51-cell sweep), concentrated in RAG (+14.1pp) and GPQA (+10.2pp) — but LOCO (leave-one-cell-
out) subset transfer is flat (0.664 vs 0.674), so **the prize is only reachable by an in-cell,
label-free selection mechanism**, not a domain lookup table.

**Approach**: follow the FUSE precedent (Candès et al., arXiv:2604.18547) — turn a label-free
assumption-violation statistic into a selection objective, the same move FUSE makes for verifier
binarization thresholds. Full assumptions audit (SML/L-SML/U-PCR/FUSE), 4-thread literature survey,
and 5 candidate pipeline-step designs (D1 assumption-violation-minimizing subset search — lowest
risk/highest priority; D2 unsupervised gated FS pre-fusion step; D3 rank/eigengap-guided grouping; D4
FUSE-style transformation search; D5 Omri's dual-use data-signature router, two access-tier flavors)
are in the memo. Key finding: **U-PCR and continuous L-SML are not the same algorithm** — different
structural covariance models (multiplicative rank-1 vs. additive) — and which one fits a given cell
better is itself a candidate label-free diagnostic (domain-dependent: L-SML dominant on GSM8K 90% win
rate, near coin-flip with U-PCR on GPQA/RAG 53%).

**Full memo**: `docs/research_notes/feature_subset_selection_landscape.md` — problem statement +
evidence, per-method assumptions audit with primary-source quotes, annotated bibliographies (Thread A:
Lindenbaum's Gated-Laplacian trace criterion identified as the "sub-matrix trace" method; Thread B:
Nadler portfolio incl. Parisi 2014 PNAS lineage-root citation gap closed, Kritchman-Nadler rank
estimation; Thread C: tabular foundation-model concepts, Concrete Autoencoders flagged as the most
directly adoptable primitive; Thread D: FUSE's Ŝ statistic, vanishing-tetrad tests, MetaOD as the
closest per-instance-router precedent), candidate designs, open questions for Ofir/Bracha.

**Next steps**: resolve the open questions in the memo (§5) with Ofir/Bracha, then pilot D1 (lowest
implementation risk, reuses existing L-SML/U-PCR residual code) on the 19-cell replication grid.

### Extension F — Step-Level Error Localization (ProcessBench) — ACTIVE APPLICATION

This extension is no longer deferred. Steps 232--233 implemented the grading
harness, token-to-step mapping, global/local decomposition, repeated threshold
protocol, and two controlled ProcessBench studies.

Scoring uses teacher-forcing (`cluster/run_teacher_forced.py`, reusing
`backfill_views.forward_batch` / `candidate_quantities` — one forward pass
per row over the *given* solution, no generation), which measures **our
model's surprise at another model's text**, not the same quantity as our
own-generation cells — relevant for every new scorer family added to this
benchmark.

Current evidence:

- GL-LIU v1: 31.36% ProcessBench F1 versus 25.71% for the reproduced Mind the
  Gap control;
- unified global/local DUFS-LIU with the core five token curves: 31.72% F1;
- naive broad-28 local curves: 29.03%, rejected;
- temporal LIU is the frozen v1 locator, but core-five local DUFS-LIU transfers
  slightly better descriptively and is the simpler next candidate.

The next evidence must come from a new dataset/model family and additional
localization baselines. Do not tune another locator on the current labels.
Develop span/onset outputs and threshold transfer only under a new registered
application protocol. See the current-decision section at the top of this file
and `docs/research_notes/localization_research_handoff_2026-08-08.md`.

**Competitor ceilings added (Step 241, 2026-08-10, N=30/subset pilots — not
yet scored against our own method's numbers)**: ProcessBench's own critic-
model baseline (Qwen2.5-72B, F1 70.4/50.0/47.1/65.9 on gsm8k/math/
olympiadbench/omnimath), the published Qwen2.5-Math-PRM-7B supervised
ceiling (F1 81.4/73.3/61.8/73.0), and a reconstruction of uPRM's own cheap
"LLM-as-a-Judge" no-training control (Qwen3-8B, F1 26.2/18.2/0.0/8.8) —
**not uPRM itself**, which requires training a new model via RL (~44
GPU-hours, no public code); see `papers/digests/unsupervised-process-reward-
models.md`. All three completed items 3 and part of item 1 of
`docs/research_notes/reasoning_localization_methods_and_benchmarks_2026.md`'s
"what should be added to the benchmark" ordered list. Full N=3400 runs are
not yet submitted — pilot health review needed first. Full account: HISTORY
Step 241.

**Scaled to full N (Step 242, 2026-08-11).** All three ProcessBench
competitors were promoted to the full 3,400 rows: `pb_prm_qwen25math7b_full`
and `pb_uprm_baseline_qwen3_8b_full` are complete, `pb_critic_qwen72b_full`
is running through a linear 6-wall resume chain (3 of 4 subsets done). Note
the uPRM reconstruction was scaled against this project's own rule that it
must not be — it is cheap and the number is legitimate, but it is **our
no-training LLM-as-a-Judge control and must never be called uPRM**. Still
outstanding: no local scorer consumes these competitor pkls yet, so none of
the three has been placed beside our own numbers.

---

### Extension L — Four hallucination-localization benchmarks (Step 242, 2026-08-11) — ACTIVE

Extension F establishes first-error localization on ProcessBench. This
extension widens the application claim to the **four different localization
tasks** the field actually distinguishes, each with its own published
competitor on identical rows. Design:
`docs/experiments/FOUR_LOCALIZATION_BENCHMARKS_CLUSTER_HANDOFF.md`.

**These are four different prediction problems.** They do not share a label
space or an official metric, and their scores must never be averaged into one
leaderboard number. The deliverable is four separate panels; its "macro
summary" is a status table of each task's own primary metric.

| # | Task | Benchmark | Competitor | Competitor status |
|---|---|---|---|---|
| 1 | which CHARACTERS are unsupported | RAGTruth (2,700 test) | LettuceDetect-large (supervised, trained on RAGTruth's own train split) | ✅ example F1 **0.792899** vs published 0.7922 — fidelity gate passed |
| 2a | which SENTENCES are unsupported | RAGTruth, GASP protocol | GASP-threshold (arXiv:2607.04223) | ✅ exact protocol, full-vocabulary JSD, 400 balanced responses |
| 2b | which CLAIMS are unsupported | RefChecker benchmark | RefChecker's own NLIChecker | ⚠️ 3-way acc 0.6751 / macro F1 0.4440 on 7,414 claims — **2 of 3 settings**; zero_context BLOCKED |
| 3 | is EVERY step correct | PRMBench (6,216 problems) | Qwen2.5-Math-PRM-7B (supervised) | ✅ F1 **0.9156**, 0 reward-count mismatches |
| 4 | WHICH step is first wrong | ProcessBench (3,400) | Qwen2.5-72B critic + PRM ceiling | 🔵 running (see Extension F above) |

**Why panel 3 is not a duplicate of panel 4.** ProcessBench labels only the
first wrong step and certifies nothing after it, so it structurally cannot
measure an every-step classifier. PRMBench supplies the missing per-step
ground truth (83,456 shipped step labels; 83,371 after the official loader's
own dedup).

**Two competitor findings that set the bar lower than their headlines
suggest.** The supervised PRM over-accepts badly — `correct_step_acc` 0.954
against `wrong_step_acc` 0.305 — and the open NLI checker is strong only on
the majority supported class (Entailment F1 0.809, Neutral 0.277,
Contradiction 0.246). On both panels the published competitor is weak
precisely on the class that matters.

**A measured constraint on panel 3 before any scoring.** 71.0% of PRMBench
steps are shorter than 32 tokens (median 24; 3.5% under 8).
`compute_stft_features` needs 32 tokens and `compute_spectral_features` needs
8, so most of the trace-level feature pool is structurally unavailable at
PRMBench step granularity. State this in the report; do not let it surface as
an unexplained weak number.

**Fidelity levels are mandatory on every competitor row**: 1 exact
reproduction, 2 protocol reproduction, 3 adaptation, 4 published context only.
Levels 2–4 must never be described as an exact reproduction. GASP is level 2
(arXiv-only, no code release or response-ID list located, so our own seed and
our own sentence splitter). The RefChecker panel is **checking-stage only** —
its human labels are attached to Claude-2-extracted triplets, so claim
extraction is out of scope by construction.

**Decision gate for this extension**: our own IU-PCR and DUFS-LIU are not yet
scored on any of the four panels. No claim about localization breadth can be
made until Phase 2 (the scoring modules) exists. Full account: HISTORY
Step 242.

---

### Extension H — Prior-Free L-SML: derive orientation, size, and selection from structure alone (NEW top priority, Step 199, 2026-07-25)

**Omri's decision (2026-07-25).** Stop optimizing the prior-dependent selector. Every piece of the
current pipeline is bootstrapped from hand-picked prior knowledge, and Step 199 proved that caps it:

- **Seeds = GOOD_6** (a hand-picked subset) build the pseudo-label. The GOOD_6-seeded pseudo-label
  is **byte-identical to the GOOD_6 fused score on 25/25 cells** (`pseudolabel_quality_audit.csv`),
  so the selector is guided by GOOD_6 and mathematically cannot beat it.
- **Anchor = `epr` / `logprob_margin`** (a hand-picked feature) sets the orientation sign.
- **K = 15** is a fixed hyperparameter.

A full week of variants over this scaffolding moved macro AUROC ~1pp and stayed 0.2pp under GOOD_6.
The goal now: a selector with **zero hand-picked features or subsets**, deriving all three decisions
from the data's own structure. Three sub-problems.

**H1 — Orientation without an anchor feature.** *Current*: `anchor_orient` against `epr`.
*Target*: recover the fused score's sign from structure alone. *Candidate*: Z2 synchronization —
`sign(cov_ij)` is a noisy observation of `s_i * s_j`, so recover the relative sign vector
`s in {+/-1}^p` spectrally / by SDP from the pairwise-sign matrix (no anchor). The single remaining
global +/-1 ambiguity is broken by a **distributional** prior, not a feature: e.g. the class-imbalance
mode (hallucination is the minority) or the skew of the consensus score. *Honest caveat (Step 199)*:
at small subsets orientation costs ~0 (R2-R0 = +0.0002 at GOOD_6) but at the full pool it costs
~2pp — so prior-free orientation matters most precisely in the large-pool regime a prior-free
selector operates in.

**H2 — Feature-set size without a fixed K.** *Current*: fixed K=15; the residual-elbow rule already
**FAILED** (Step 198, r_s=+0.007 vs oracle-K). *Target*: a label-free size from the covariance
spectrum. *Candidates*: effective rank / participation ratio of `cov(V)`; count of eigenvalues above
a Marchenko-Pastur noise floor (the "signal dimension" under L-SML's low-rank model); bootstrap
stability selection. *Must* be validated against oracle-K the same honest way D1 was — correlating
with AUROC is not predicting the optimal size.

**H3 — Feature selection without seed priors.** *Current*: mRMR against a GOOD_6-seeded pseudo-label
(proven ≡ GOOD_6). *Target*: a seed-free consensus. *Candidate*: build the pseudo-label as the
**L-SML consensus over ALL features** (Nadler/Jaffe: the ensemble's self-consistent agreement, which
down-weights uninformative views through the covariance structure), then select features by
agreement with that consensus and iterate. *Risk*: garbage features polluting the consensus — bound
it with the never-run **R6** (perfect-target ceiling, `SPEC_gap_ladder.md`) and a low-signal-cell
guard. This is the natural escape from the GOOD_6 cap because the full-pool consensus is not tied to
any hand-picked subset.

**Grounding**: all three sit in the project's spectral-meta-learning lineage (Nadler 2012, Jaffe
2014, Parisi) — the right place to look for structure-only estimators of orientation (Z2 sync),
dimension (spectrum), and consensus (L-SML over all views).

**Decision gate**: a prior-free pipeline is worth adopting only if it reaches **GOOD_6 (0.7594)**
with zero hand-picked input. Matching GOOD_6 prior-free is itself a real contribution (it removes the
hand-tuning); beating it is the headline. First concrete step (CPU): H3's full-pool L-SML consensus
pseudo-label + H1's Z2-sync orientation, benched vs GOOD_6 on the 25 cells, before touching H2.

> **STATUS: ✅ CLOSED AS BOUNDED — gate NOT met (Steps 200–202, 2026-07-25).**
> Built (Step 200), audited (Step 201 — 9 defects, the "GroupFS sweep" never ran GroupFS), fixed and
> re-measured on a canonical scoring path with GOOD_6 = 0.7594 asserted on every bench (Step 202).
> All four components are bounded and **GOOD_6 remains unbeaten**:
>
> | Component | Verdict | Evidence |
> |---|---|---|
> | **H1** orientation | no headroom | L-SML is **gauge-invariant** to feature signs — 1150/1150 sign vectors bit-identical, so sign is worth exactly **0.0pp**; the prior-free skew tiebreaker costs **−10.7pp** and its premise (hallucination = minority) is false here (9/25 cells have pos_rate > 0.5) |
> | **H2** label-free K | **REFUTED** | no rule met the pre-registered oracle-K bar; `eff_rank` Spearman **−0.0995** (p=0.64); **fixed K=15 beats every adaptive rule**. Oracle-K median is **14** while the spectrum rules predict 4–6 — they count independent directions (~4.5), but L-SML exploits **correlated** views, so effective rank is the wrong quantity |
> | **H3** selection | bounded | **R6 = 0.7676 DEAD** — +0.82pp, under the +1.0pp gate. Note the corrected contrast gives **22W/3L, p = 0.00014**: a perfect label-derived target buys a *reliable but sub-1pp* gain, not nothing. The fixed `a7.iter_consensus` lands −2.16pp (8W/17L, p=0.011) |
> | **Phase 4b** GroupFS grouping | bounded | now genuinely swept (λ1 guard PASS: 71/700 configs vs 0/350 for the stand-in); best 0.7508, and its **label-peeking ceiling 0.7585 only ties GOOD_6** (−0.09pp, p=0.33) — bounded, not mis-tuned |
>
> **Durable gain (subtractive)**: `ALL_SIGNS` — 42 hand-derived per-feature polarities — is provably a
> no-op in the fusion path and can be deleted at zero cost. The only orientation prior that remains is
> a **single ±1 bit**, and the `epr` anchor already spends it optimally (an oracle bit ties it).
>
> **Where the gap actually is**: the ladder's clean rungs put the deploy-point gap in **weight
> estimation**, not sign, target quality, or K. Any successor direction should attack that.

---

### Extension I — Inverted-fit selection: the criterion is right, the sign is wrong (Step 203, 2026-07-26)

**Status**: ❌ **CLOSED AS REFUTED (Step 204, 2026-07-27). Do not build I1.** The whole premise — that
the fit criterion's sign is inverted — was an artifact of the L-SML loading scale.
`_estimate_von_voff` returned the unit-norm eigenvector where Lemma 1 requires the loadings to
reproduce the covariance, so misfit was inflated by group size × coupling strength — largest exactly
where the clustering succeeded. With a masked rank-one completion estimator:

| loading scale | Spearman(misfit, AUROC) | positive cells | **re-measured Step 205** |
|---|---|---|---|
| `unit` (what Step 203 measured) | **+0.223** | 24/25 | **+0.222**, 23/25 |
| `eigen` (the SPEC's literal fix) | +0.183 | 25/25 | +0.188, 25/25 |
| `complete` (exact on the unit checks) | **−0.006** | 12/25 | **−0.022**, 10/25 |

Shift **−0.228, Wilcoxon p = 0.0015**; the `unit` arm reproduces Step 203 exactly, so the harness is
sound. **Re-run on Step-205's fixed code** (this study's size grid starts at 3, so it was the one
most exposed to the small-m degeneracy): shift **−0.243, p = 0.0006** — the conclusion holds and
strengthens slightly. **The criterion never needed inverting — it needed scaling.** I1 (sign-flip the selectors)
would have been curing a symptom. I2–I4 and theorems T1–T4 rest on the same artifact and are void as
stated; T1 in particular ("redundancy inflates misfit monotonically") is *true of the broken
estimator* and is precisely the bug, not a property of the data.
Evidence: `results/pruning_study/06_scale_vs_criterion/`, `results/residual_scaling/`, HISTORY Step 204 §B.

<details><summary>Original Step 203 framing, kept for the record</summary>

**The finding.** The L-SML residual — the quantity every trimming rule in this project steers by — is
*informative about subset quality but anti-correlated with the direction we optimise*:

| Evidence | Number | Consistency |
|---|---|---|
| Within-size Spearman(residual, AUROC), live 30-view pool | **+0.223** mean / +0.185 median | **24/25 cells positive** |
| Repair worst-fitting group vs repair **random** group | **−2.22pp** | W/L 7/18, p = 0.032 |

Residual is *misfit* (lower = better rank-one fit), so a positive correlation means **worse-fitting
subsets score higher**. Minimising misfit is descending the wrong gradient.

**Why (mechanism, measured).** The worst-fitting group is reliably the near-duplicate confidence
cluster — `epr`, `epr_spilled`, `epr_energy`, `mean_top1_logprob`, `logprob_margin` — i.e. the
*strongest* individual views. They break the rank-one model **because** they are several readings of
one underlying quantity, and that duplication is precisely the extra shared structure a single-factor
model cannot absorb. In this data **redundancy and informativeness travel together**, so poor fit
marks where the signal is concentrated.

**This reframes three earlier closed results rather than contradicting them.** Extension G/H closed
because every label-free selector *minimising* something rank-one-flavoured landed at ≈0.75. If the
sign is inverted, they were all optimising away from the answer, and "bounded" may be "bounded in the
direction tested".

> **⚠ RUN `SPEC_residual_scaling_fix.md` FIRST (raised 2026-07-26, after this block was written).**
> Omri asked why the clustering — which is *supposed* to group dependent features together — ends up
> flagging those groups as the worst fit. Answer: `_estimate_von_voff` returns the **unit-norm**
> eigenvector, but Lemma 1 requires `v_i·v_j = r_ij`, i.e. `a = √λ₁·v`. A perfect `m`-duplicate block
> is therefore scored with misfit/pair rising 0.25 → 0.83 as `m` goes 2 → 11, so misfit is inflated by
> **group size × coupling strength** — largest exactly where the clustering *succeeded*. That makes
> "repair the worst group" mean "dismantle the biggest tight cluster", i.e. **the selection step
> optimises against the clustering step**. It also sits in the deployed detector, since K is chosen by
> minimising this residual (15/25 cells pinned at K ≥ 7). **If the scaling fix flips the sign, I1 below
> is the wrong remedy** — the criterion needed scaling, not inverting. Treat I1 as the fallback.

#### Proposed experiments (in dependency order)

- **I0 — Fix the eigenvalue scale first.** See `SPEC_residual_scaling_fix.md` (checks U1-U2, predictions P1-P3, anchors R1-R3). Gates everything below.
- **I1 — Sign-flip the existing selectors (only if I0 does not flip the sign).** Re-run `a1.residual`,
  `a6.pl_dufs`, and the Step-203 cluster-localized arm with the criterion **maximised** instead of
  minimised. Pre-register: label-free macro ≥ 0.7524 (`a6.pl_dufs`, the automatic-picker bar) with
  W/L and Wilcoxon reported; report effect sizes, do not gate on <1pp.
- **I2 — Redundancy-preserving trimming.** Trim to *increase* misfit, i.e. keep at least one member
  of each near-duplicate family and cut from the best-fitting (most "explained") groups. This is
  Yen's Q3 read backwards and is the natural constructive form of the finding.
- **I3 — Is the inversion a property of the data or of the estimator?** Recompute the correlation
  after removing one member of every |ρ|>0.9 pair. If the correlation collapses toward 0, the
  inversion is *caused by* duplication (and disappears in a de-duplicated pool); if it survives, it is
  a property of the fusion. This decides whether I2 is a fix or a workaround.
- **I4 — Per-cell sign check.** The 1/25 cell with negative correlation is a natural held-out probe:
  is it structurally different (fewer duplicate pairs? lower second-factor share?), or noise?

#### Candidate theorems / analytical targets

These are conjectures suggested by the measurements, stated so they can be proved or refuted rather
than assumed:

- **T1 (redundancy inflates misfit monotonically).** For `x_j = a_j y + ε_j`, adding a duplicate view
  `x_{m+1} = x_k + δ` with `Var(δ) → 0` increases the Eq.(14) rank-one residual by a quantity
  monotone in `a_k²`. *If true, misfit is partly a measure of how much signal a subset carries, which
  explains the observed positive correlation directly and predicts the effect scales with the
  duplicated view's loading.*
- **T2 (the localizer is a signal detector).** Under T1, `argmax_g` (within-group misfit per pair)
  selects the group maximising `Σ_{i,j∈g} a_i²a_j²` rather than the group containing the least
  informative view. *This would make "repair the worst group" provably a signal-removal operator, and
  the −2.22pp vs random a predicted consequence rather than an empirical accident.*
- **T3 (no interior optimum in expectation).** For subsets drawn uniformly at size k from a pool with
  bounded loadings, `E[AUROC(k)]` is non-decreasing in k. *Consistent with 25/25 cells here and with
  D1/H2's refutations; if provable it closes "find the right K by following a curve" analytically
  rather than one grid at a time.*
- **T4 (gauge invariance extends to the diagonal).** Step 201 proved L-SML is invariant to input
  feature signs (`X→XD`). Does the same hold for the precision-weighted variant `w_j ∝ a_j/(R_jj −
  a_j²)`, which *reads* the diagonal? Step 203 measured precision weighting at −0.13pp; a proof that
  it is also gauge-invariant would explain why it cannot move.

---

</details>

### Extension J — Weight estimation (the R3 − R2 = +1.45pp term) — first evidence in, still open

**Status**: **Experiment implemented and pre-registered in Step 226; awaiting the data-machine run.**

> **Step 226 corrects the literature baseline and turns the live sparse-Delta idea into a
> controlled experiment.** The real Tenzer et al. 2022 paper already contains sparse-error
> SU-PCR; the repo's file with that name is Dror et al. 2017. Therefore "add sparse Delta" is not
> the contribution. `SPEC_DEPENDENCY_FUSION_EXPERIMENT.md` registers a 2x2 factorial separating
> independent vs sparse reliability estimation from PCR vs structured regularized weights, plus
> iRBM/deep-hard/deep-soft DEEM baselines. H1 prices published SU-PCR; H2 (`SDSF - SU-PCR`) is the
> contribution test. Full-pool is primary, the incumbent keep set is secondary, and labels enter
> only after scores are frozen. Implementation: `spectral_utils/dependency_fusion.py`,
> `spectral_utils/deem_adapter.py`, `scripts/run_dependency_fusion_experiment.py`, and the
> dataset-free planted-world gates in `scripts/test_dependency_fusion.py`.

> **Step 205 closes the last U-PCR lead and corrects how one of its numbers should be read.**
> The `lambda2_threshold` hypothesis — that we are pinned at 2 eigenvectors and the redundant second
> factor is what hurts — is **refuted**. `lambda2_frac` is tightly clustered just above the hardcoded
> 0.1 (median 0.1435, range 0.0942–0.2328), so the threshold flips 24/25 cells at once; sweeping it
> to remove the second component everywhere buys **+0.43pp (9W/15L, p = 0.16)** (`exp07`).
> **And Step 204's "−3.67pp for the 2-eigenvector rule" is a factorial MAIN EFFECT**, averaged over
> the 32 combinations of the other five factors. At the deployed configuration the same switch is
> **−0.43pp mean / +0.07pp median, 15W/9L, p = 0.16** — a wash. The sign reversal of Step 142 stands;
> the magnitude does not transfer, and should not be quoted as a deployed cost.
> Separately, Step 204's B1 ("the g2 range never binds") holds only for the **pre-exclusion** fit: the
> g2 the pipeline returns comes from the post-exclusion refit and is at the grid ceiling in 24/25
> cells. Un-pinning it is −0.28pp (12W/13L), so the conclusion survives and the mechanism does not.
> **Nothing in the g2 / component-count apparatus is actionable. Do not re-open it.**

The **previously implemented independent-error U-PCR flag sweep** is closed as inert: every
paper-faithful
flag hurts or does nothing, one-component U-PCR is *exactly* PC1 of the surviving features (cosine
deviation 7e-12) so its ρ/g²/Eq.-20 machinery enters only through the exclusion mask, and no
configuration of 64 reaches GOOD_6. The dependent-features extension (fit the additive system on
cross-cluster pairs only) is **refuted**: it fails both pre-registered mechanism gates and loses
−4.46pp (9W/16L, p = 0.030), and its premise was a confound — fit error is essentially pair
correlation (Spearman 0.870), the raw 2.03× same-vs-cross gap collapses to 0.97–1.00 when matched on
|C_ij| decile, and magnitude-only clustering separates it *better* than L-SML.
**What is left of J**: the disagreement diagnostic itself (rank agreement +0.186, top-5 overlap 1/5)
is unexplained, and the one lever that moved anything this session was **orientation**, not weighting.

Step 203 supplies the first systematic measurement; no repair found.

**What was tested (Step 203, Exp 5).** A 2×4×2 factorial over the three slots the five proposed
paradigms actually occupy — conditioning (none / RMT eigenvalue clipping) × loading estimator
(eigenvector / triplets / low-rank+sparse / robust-IRLS) × weighting (signal / precision). **Span:
0.7434–0.7555.** Main effects: triplets +0.21pp over the eigenvector, low-rank+sparse +0.11pp, robust
IRLS −0.33pp, RMT cleaning +0.14pp, precision weighting **−0.13pp**. Nothing separates from noise on
25 cells.

**Diagnostic (Exp 4), which should aim any next attempt.** Second factor at median **0.312** of the
first (one factor explains ~81% of `R_off`'s squared spectral mass) — the rank-one premise is
approximate, not exact. Guessed vs supervised trust levels: rank agreement **+0.186**, sign agreement
**0.55** (after resolving the single global flip), **top-5 overlap 1/5**. The label-free estimator and
the supervised model largely disagree about *which* views matter — this is not a calibration problem.

**Ruled out analytically or empirically**: Ledoit–Wolf linear shrinkage (provably inert — scales
`R_off` by a positive constant, identical eigenvector); non-linear shrinkage *as usually specified*
(modifies eigenvalues, keeps eigenvectors — inert unless the diagonal is re-zeroed and re-decomposed,
which is the form tested here); bagging (identical to 4 d.p.).

**Next**: run the registered Step-226 dependency-fusion experiment on the data machine. Its SU-PCR
arm is explicitly rank two; its SDSF arm changes only the final weights. Do not tune sparse support,
ridge, DEEM architecture, or seeds after reading AUROC.

---

## Recommended Priority Order

*(Authoritative current order — updated 2026-08-16, Step 270)*

0. **Boundary, not an application task: preserve frozen A6.** Do not execute A6
   as part of this joint program. If the separate core-method program is later
   advanced by explicit instruction, run A6-S0b and every later stage exactly
   as preregistered; the application work may not rescue, modify, reinterpret,
   or delay it.
1. **Retain the frozen application architecture.** The completed Step-270
   retrospective keeps ordinary frozen Global/Local heads plus
   `iu28_no_length`. Do not reopen running-maximum, persistence/area,
   slope/recovery, elapsed-length, or graph variants on the same saved monitor
   grid; none earned promotion under grouped uncertainty.
2. **Require a genuinely new identifiable signal before more method work.** A
   token-native causal recurrence or new telemetry/data may justify a new
   protocol. Another transformation of the same coarse CUSUM/`sw_var`
   trajectory does not. Any reopened candidate must preserve separate
   localization and early panels, question/family grouping, the label boundary,
   and the cross-task non-inferiority gate.
3. **Treat native competitor and fresh-data work as a new approval gate.** Only
   if a new mechanism has a credible retrospective premise should the project
   verify exact DeepConf raw-logit conditions or collect a genuinely new
   dataset/model family. No GPU inference follows from Step 270.
4. **Keep RAG citation grounding queued and generic fusion-core sweeps paused.**
   Reopen them only after this joint reasoning program reaches a stable
   architecture or supplies a new identifiable signal. The current reasoning
   architecture is stable but does not itself create evidence for another
   generic fusion sweep.

### Historical July-30 priority record

The list below is retained as an audit trail of the decisions that led to the
current pivot. It is no longer the active execution order. In particular, its
statements that localization is deferred and that dependency fusion is next
were superseded by Steps 226--235.

**Now — no GPU needed**

0-TOP. **← FAILURE DIAGNOSIS (Jul-30 action item 1), DIAGNOSIS ONLY.** Nine cells, all 25 measured
   as the comparison group. ✅ **DONE (Step 210)** — `results/failure_deepdive/index.html`.
   **The mechanism is label-free relative-sign recovery.** L-SML's job is to give back what a
   simple average loses by not knowing the per-view signs; it recovers **0.919–1.247 of it on
   every healthy cell** (median 1.025) and **below 0.90 on 4 of 8 weak cells, 0 of 14 healthy**
   (Fisher p = 0.0096 — **significance withdrawn in Step 212**: requiring a stable denominator gives
   p = 0.0735 / 0.2500, and U-PCR's version never reaches it. The *pattern* holds and both arms
   show it — Spearman(L-SML, U-PCR recovery) = +0.707, p = 0.0002 — but the repair, not the table,
   is the confirmation). Three secondary mechanisms: the selector drops the pool's strongest view
   on 4 cells (−4.8pp worst), CoQA's views are non-monotone (z = +3.19 vs every other cell), and
   3 of the 9 trip **nothing** — they are simply hard. **Cleared as suspects: orientation**
   (global bit costs 0.00pp on 25/25) and **K-selection** (0/25 degenerate; eigengap helps 5/25,
   mean −1.39pp). **No fixes were run**; three are pre-registered with gates on the page.
   **⚠ Repair 1 (Z₂ as the label-free sign estimator) was then built and REFUTED (Step 213)** —
   1/4 on gate (i), 13/16 and 15/16 on gate (ii), full regression a dead wash (+0.04pp, p = 0.89).

0-NEXT. **← REPAIR 3: keep the pool's strongest view unconditionally.** Promoted by **Step 214**
   (features vs algorithm) and kept there by **Step 215**, which was an adversarial review that
   **withdrew three of Step 214's four findings**. Site:
   `results/action_items_jul2026/item1b_feature_comparison/`. Read the withdrawals before quoting
   anything from this line.
   - **SURVIVES**: some of the gap is genuinely in the features on all three pairs, and on none is
     it all of it (ceiling-gap / label-free-gap = 51% / 52% / 84% on the rebuilt pairs). **Quote
     the sign, not the percentage** — bootstrap CIs [25,74] / [14,111] / [59,104] overlap, two
     include 100%, and the TriviaQA figure ranges −2% to 60% across defensible partner cells.
   - **SURVIVES, and is the reason for Repair 3**: on `seiclr_triviaqa_opt30b` the loss is
     **selection + sign, and dilution explains 0.0pp**. Honest best single over the pool 0.6200
     (split-half ±0.0067) → best inside the 12 the selector chose 0.5791 (**the selector discarded
     the pool's two strongest views**) → L-SML 0.5614. ≈4.1pp selection + ≈1.8pp sign. L-SML's
     effective per-view signs are wrong on 2/12 views here vs 0/12 on the other five cells.
   - **⚠ WITHDRAWN — the pair's strong cell was disqualified.** `spilled_triviaqa_llama8b` has
     n_pos = 6/256, `trace_length` alone scores 0.925 on it, and `scripts/advisor_report.py:783`
     already said do not headline it. Replaced with `semenergy_triviaqa_qwen3_8b` (n=4392).
   - **⚠ WITHDRAWN — "label-driven correlation ~3× smaller".** Algebraically implied by the
     per-view Cohen's d already reported (entrywise corr ≥ +0.99997, 6/6 cells); the consistency
     was an uncontrolled class prior (κ-adjusted: 28.9× / 3.8× / 3.4×).
   - **⚠ WITHDRAWN — "the estimation-noise hypothesis is retired", and the do-not-run instruction
     is RESCINDED.** The statistic was normalised by 1/√n, the SE of a single correlation, when C
     and W share rows; wrong by ~√n. Against a permutation null the excess is 46×–639× its null.
     `Spearman(label-free − ceiling, n) = −0.462, p = 0.020` over 25 cells. **The
     subsample-to-matched-signal test is back on the table.**
   - **⚠ WITHDRAWN — "the method matches supervision on strong cells".** Label-free exceeds the
     ceiling on 4/25 cells, **all with n ≤ 700** (Mann-Whitney p = 0.035). A small-sample effect.
   - **⚠ WITHDRAWN — "16.2pp of reachable headroom".** The gap is real (0.7223 vs 0.5614) but
     splits into ≈1.8pp label-free-reachable, ≈4.1pp needing labels, ≈10.2pp supervised-only.
     **Honest label-free target: single digits.**

0-THEN. **Adjacent applications (Jul-30 action item 3)** — Extension E first, since it already has
   a replicated effect (+5.6pp [+0.9, +10.6] over the best DeepConf window at the earliest 10% of
   the trace) and an adopted formulation (SPRT stopping rule; report **(AUROC at budget, tokens
   consumed)**). Extension F (localization) stays deferred.

0-CLOSED. ~~**Feature-subset selection**~~ ❌ **CLOSED BY THE JUL-30 MEETING.** The three arms tie
   and nothing separates them. Superseded as a *direction*; the results stand.

0. ~~**Extension H — Prior-Free L-SML** (Step 199 pivot): strip every hand-picked prior (`epr`
   anchor, `GOOD_6` seeds, fixed K); derive orientation (H1), size (H2), selection (H3) from
   structure alone~~ ✅ **CLOSED AS BOUNDED (Steps 200–202) — decision gate NOT met.** H1 has no
   headroom (sign is a gauge, worth 0.0pp), H2 is refuted (fixed K=15 beats every label-free rule;
   `eff_rank` r_s = −0.0995), H3 is capped (R6 perfect-target = DEAD at +0.82pp), and GroupFS
   grouping's own label-peeking ceiling only ties GOOD_6. **GOOD_6 (0.7594) still unbeaten.** The one
   durable gain is subtractive: `ALL_SIGNS` (42 priors) is a provable no-op and can be deleted.
   See the Extension H status block above.

0-NEXT. ~~**Extension I — inverted-fit selection (Step 203)**~~ ❌ **CLOSED AS REFUTED
   (Step 204), re-verified on fixed code (Step 205).** The +0.223 correlation was an artifact of
   the L-SML loading scale. Re-measured after the Step-205 small-m fix: unit **+0.222** (23/25
   positive) → complete **−0.022** (10/25), shift **−0.243, p = 0.0006** (Step 204 published
   −0.228, p = 0.0015). **P2 holds, slightly strengthened. Do not build I1.** See the Extension I
   block above.

0-NOW. **← Orientation without hand-picked signs (Step 204, Phase E).** The one lever that moved
   anything: deriving per-feature polarity from `sign(rho)` beats the 42 hand signs by **+1.46pp
   (20W/5L, p < 0.001)**, and **15 of 30 pool features carry the wrong hand sign**. Two caveats
   that shape the experiment: correcting the signs is a **0.0000pp no-op on the L-SML path**
   (sign-gauge invariance), so the value is only for sign-SENSITIVE consumers; and the global
   ±1 is **provably not recoverable** from the covariance (a global flip leaves rho
   bit-identical), so the `epr` anchor bit cannot be removed this way. Pre-register against the
   0.7524 automatic-picker bar, not 0.7594.

0-READY. **← RUN DEPENDENCY-AWARE WEIGHT ESTIMATION (Extension J, Step 226).** The experiment is
   pre-registered and fully implemented in `SPEC_DEPENDENCY_FUSION_EXPERIMENT.md`. It separates the
   already-published SU-PCR reliability correction from our SDSF weighting change, includes the
   ridge-only ablation needed to prevent a false novelty claim, and benchmarks modern DEEM hard/soft
   against its iRBM endpoint. The repo-only machine cannot produce the number; the data machine only
   needs to install the pinned extra, run the planted gates, and execute the runner.
0b. ~~Feature-subset selection: memo (Step 185), full bench (Step 186), a6 pseudo-label gates +
   30-view LOCO sweep (Steps 194-195), D1/D2 build + honest refutation (Steps 197-198)~~ ✅ **CLOSED
   as bounded** — no label-free selector beats GOOD_6; D1 (adaptive-K) refuted, D2 (PL-mRMR) beats
   GOOD_5 (p=0.037) but not GOOD_6. The one durable win is the seed rule (→GOOD_6, +3.5pp macro).
   `LOCO_5` (sweep consensus, 77.1% on 24 cells) is the strongest fixed subset found and still
   warrants naming + `REFERENCE_SUBSETS` entry independent of Extension H.
1. ~~L-SML literature search (Item 1)~~ ✅ done (Step 139)
2. ~~Logistic regression oracle `scripts/logistic_oracle.py` (Item 2)~~ ✅ done (Steps 142–143, 147)
3. ~~Streaming pivot pilot (Extension E)~~ ✅ done (Step 148 — G1 PASS / G2 FAIL; earliest-prefix edge is the surviving thread)
4. Present streaming pilot verdict to advisors → decide hybrid framing vs thesis section (Extension E step 5)

**Next Colab session**
5. Benchmarking: fix `boot_auc` kwarg → Phase 14 Cell 9 re-run (Item 4)
6. **Raw-trace regeneration** (Extension E step 1): MATH-500/Qwen-7B + one R1 cell with ≥4096-token cap, saving `token_entropies` + top-50 logprobs — unblocks the earliest-prefix replication AND repays the raw-data debt
7. Sampling fusion: SE K=10 + CONT spectral (Item 5)
8. Temperature variation: T=0.3/0.6/2.0 inference + A/B ablation (Item 6)

**Subsequent Colab sessions**
9. Streaming earliest-prefix replication on the regenerated cells (Extension E steps 2–3; local CPU once traces exist)
10. Extend QA evaluation: CoQA > SQuAD v2 > TruthfulQA (Item 3, priority corrected Step 155 — runs on AIRCC as part of the replication grid)
11. Extension A (Conformal): A1 frozen scorer + imbalance metrics first, then A2 LTT

**Later**
12. Extension B (Agentic): Qwen3-7B, HotpotQA multi-hop
13. Extension C (Hidden states): one forward hook on Falcon
14. Extension D (VLM): only if committee wants multimodal chapter

**CLOSED — do not re-open without new evidence (Step 206, 2026-07-28)**
- **Pool composition is not a lever, in either direction.** *Removal*: null on L-SML
  (WS3 LOCO, −0.22…+0.04pp, n.s.) and **significantly harmful on U-PCR** (`exp08`, −0.50pp,
  7W/18L, p = 0.0096); at any drop threshold ≥ 0.2pp **no view qualifies at all**. Corroborated by
  the pool-size experiment (16→30 views within 0.11pp) and `feature_inclusion_audit_c46` (every
  view has non-zero LOVO cost somewhere). *Addition*: all six pre-registered ADD variants land
  below GOOD_6 (`exp09`; best `ref.GOOD_6+topk` 0.7587, worst `ref.ENTROPY_6` 0.7462).
- **The mechanism, which is why both directions fail together**: U-PCR's Algorithm-1 exclusion is
  **data-dependent** — removing a zero-weight view still perturbs C → ρ̂ → *which other views
  survive* (29% of the time). And on the add side, high individual informativeness does not imply
  additive value: `topk_tail_mass` ranks #1 of 30 and is a `ref.LOCO_5` member, yet adds nothing to
  a subset already covering that direction. **What governs is redundancy and estimator coupling,
  not view quality.**
- ⇒ **Orientation is the only remaining lever.** Priority item unchanged.

**De-prioritized (valid but not blocking)**
- Step 132: MATH-500 SpilledEnergy GPU run — run opportunistically when Colab is free
- Merge decision (continuous L-SML → master): contingent on Step 132
- Phase 13 (AMC23/AIME24): fix `\boxed{}` grading bug before any re-run

---

## Thesis Narrative Thread

> *The per-token entropy trajectory H(n) is a signal, not a scalar. Collapsing it to its mean (EPR) discards temporal structure that predicts hallucination. Spectral features of H(n) recover that structure. L-SML fuses those features without labels, in a single forward pass. This gives a detector that is cheap (K=1), interpretable (spectral signal processing on an information-theoretic signal), and formally calibratable (the L-SML score is a continuous input to LTT). The thesis validates this on math reasoning, extends it to RAG and QA, and closes with a conformal chapter that turns the AUROC result into a deployment-ready detector with a formal false-negative-rate guarantee.*
