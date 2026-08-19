# Early/Online Hallucination Detection — Phase-1 Checkpoint

**Date:** 2026-08-16

**Status:** Phase-1 research and protocol audit only. **No expensive inference has been launched.**

**Decision requested:** approve, revise, or reject the staged execution plan in §8.

## 1. Executive decision

The literature does not offer one clean benchmark that simultaneously tests (a) causal
single-trace final-correctness detection, (b) within-trace error localization, (c) early
stopping, and (d) multi-sample adaptive compute. These are different tasks and must remain
separate evaluation lanes.

The most defensible minimal cycle is:

1. **Zero-GPU reanalysis:** use the two existing rich Drive caches to test the frozen
   29-stream, trajectory-first IU-PCR task adapter on two dataset/model families, with a
   problem-grouped split and truly causal prefix replay. These heavily reused cells are a
   retrospective leakage-controlled audit, not fresh confirmation.
2. **One paper-aligned reproduction:** reproduce REFRAIN on Qwen3-8B/MATH500 using the
   paper's native prompt/decoding/stopping protocol while capturing the four telemetry
   channels required by the frozen feature map. This is the best accepted, cheap,
   single-trace stopping comparison, although its public repository is still a README-only
   release placeholder; the result must therefore be called a **paper-specified
   reproduction**, not a code-exact reproduction.
3. **A small official-code DeepConf pilot:** validate the native multi-sample protocol on
   one 30-problem cell. Do not approve the full 4,096-trace-per-problem pool yet.
4. Treat Streaming Hallucination Detection as an **asset-gated supervised ceiling**. Its
   linked code site was unreachable from this environment, so no compute should be booked
   until the annotated trajectories/checkpoints can be inventoried.
5. Reserve one genuinely untouched confirmation cell—recommended gpt-oss-20B/CommonsenseQA
   under the REFRAIN paper protocol—and hash the frozen implementation before opening its
   results. The existing caches cannot support the requested generalization claim.

The first approval gate (H0+A0+L1+A1+B1) costs roughly **1.5–7 B200-equivalent GPU-hours**
and **5–21 GB** of new retained artifacts. Conditional L2/A2/A3 add roughly **3.5–13 hours**;
the untouched A4 confirmation adds roughly **2–6 hours**. A full native DeepConf cell is a
separate decision costing roughly **75–150 GPU-hours** and **20–60 GB** with a native-minimal trace/
confidence record, or **0.6–1.2 TB** if full top-50 telemetry is retained.

### 1.1 Competitive position after the completed Local/Online cycle

The completed Step-273 transfer audit narrows the claim substantially:

- **Localization:** the family-six/step-top-five candidate reaches 0.3662 ProcessBench macro
  F1 versus 0.2646 for the reproduced Mind-the-Gap control on the same evaluation rows and
  scorer copies. This is a real shared-protocol advantage over that published method. It is
  only parity with maximum entropy plus the same locator (0.3614; paired delta +0.0048,
  95% CI [-0.0264,+0.0375]), so the evidence does not yet establish that spectral fusion adds
  value beyond the strongest transparent token statistic.
- **Early detection:** retain IU28, not the joint finalist. IU28 reaches 0.6104 equal-family
  AUROC at 64/128 tokens versus 0.5926 for mean entropy and 0.5922 for the existing
  entropy-window DeepConf proxy. Earlier grouped intervals against that proxy crossed zero,
  and the proxy is not the paper's exact confidence. This supports competitiveness in the
  one-trace log-probability tier, not a paper-level DeepConf win.
- **Higher-access ceilings:** the Qwen2.5-Math-PRM-7B and Qwen2.5-72B critic reach 0.7280 and
  0.5895 localization F1, respectively. The current method is not score-competitive with these
  supervised/additional-model tiers.
- **uPRM:** the paper reports per-subset ProcessBench F1 of 58.3/52.6/42.7/39.8. Its trained
  model and exact scoring prompt are not publicly available. The existing Qwen3-8B
  reconstruction is only the paper's independent LLM-as-a-Judge control and scores poorly;
  it cannot stand in for uPRM. A new Qwen2.5-14B same-backbone run is the minimal honest
  localization comparison.

Accordingly, the publishable strength to test is not "state of the art." It is: **a cheap,
single-pass, label-free token-statistics method that beats Mind the Gap, is currently tied with
maximum entropy for first-error localization, and may provide an earlier warning than simple
window confidence under a one-trace access contract.**

## 2. Concise field map

| Work | Unit / label | Access and learning | Intervention | Role here |
|---|---|---|---|---|
| **Historical `lsml16`** (this project, Step 182) | One trace; final answer correctness at prefix fractions | Entropy-derived H16; label-free but transductively refitted/oriented per cohort and budget | Fixed-budget retrospective detection | Historical reconstruction only; preserve separately |
| **Causal adapter to frozen trajectory-first IU-PCR** (this project) | One trace; final answer correctness, with token risk as the latent online signal | White-box token uncertainty; label-free development fit; calibration labels only for alarms | Detect/abstain; optional common forced-closure policy | Primary method; elapsed-length substitution disclosed |
| [Streaming Hallucination Detection](https://arxiv.org/abs/2601.02170) | Sentence/step and prefix hallucination labels plus final correctness | Hidden states; supervised Claude-labeled probe | Detection/localization, not a stopping policy | Supervised ceiling, conditional on assets |
| [DeepConf](https://arxiv.org/abs/2508.15260) | Many complete traces per problem; answer-vote correctness | Top-k log probabilities; training-free | Stop sampling, filter traces, weighted vote | Native adaptive-compute baseline; per-trace score is a declared adaptation |
| [REFRAIN / Stop When Enough](https://aclanthology.org/2026.acl-long.1256/) | One reasoning trace; task-answer accuracy | Text steps, reflection cues, SBERT similarity; training-free | Stop reasoning and force answer closure | Primary paper-aligned stopping reproduction |
| [LEASH](https://arxiv.org/abs/2511.04654) | One trace; task-answer accuracy | Token entropy slope and logit-margin plateau | Stop rationale and force answer | Cheap sensitivity baseline; under-specified and accuracy-losing |
| [uPRM](https://arxiv.org/abs/2605.10158) | First erroneous reasoning step on ProcessBench | Multiple traces plus trained Qwen teacher/model | Step localization | ProcessBench ceiling/control, not final-correctness detector |
| [HALT](https://arxiv.org/abs/2602.02888) | Full-response answer-level hallucination label | 25 log-probability features plus bidirectional GRU | Post-hoc classification | Non-causal learned ceiling only |
| [ZIP-RC](https://arxiv.org/abs/2512.01457) | Joint terminal reward and remaining length | Trained reserved logits and external reward signal | Multi-trajectory meta-MDP | Learned adaptive-compute field marker; different access contract |
| [Adaptive Inference-Time Compute](https://arxiv.org/abs/2410.02725) | Multi-trace answer quality | Trained generative self-evaluation | Restart/prune traces | Adjacent adaptive-compute work |
| [JET](https://arxiv.org/abs/2509.23392) | Reasoning sufficiency | Model training | Teach model to stop | Field marker, not a detector |
| [Online Auditing of Information Flow](https://arxiv.org/abs/2310.14595) | Social-event stream | Supervised SVM and transition model | Bayesian/optimal stopping | Formulation inspiration only; not an LLM baseline |

The key taxonomy is:

- **Detection lane:** “Given only prefix tokens, will this trace's final answer be wrong?”
- **Localization lane:** “Has a particular reasoning step become erroneous?”
- **Single-trace stopping lane:** “Should this trace stop reasoning and emit an answer now?”
- **Multi-trace compute lane:** “Should sampling continue, and which traces should vote?”

No headline table should rank methods from different lanes as though they solve the same
problem.

## 3. Reproduction cards and feasibility

### 3.1 Historical `lsml16` — known-answer reconstruction

The Step-182 result was produced in the code era represented by commit `f13e8bc`. Its method
is H16 prefix features, continuous L-SML refitted separately on the entire cell at each
budget, label-free EPR-anchor orientation on that same cohort, and the entropy-based
DeepConf approximations with windows 32/64/128. The reported +5.6-point paired difference
used the earliest **10% of eventual trace length**, so it is a retrospective historical
diagnostic, not a deployable result.

The raw input is the Phase-15 Qwen2.5-Math-7B/MATH500 T=1.0 run-0 cache identified in
`HANDOFF_punchlist_and_reruns.md`; it is present on Drive. Re-run the frozen `f13e8bc` code in
an isolated temporary worktree, with the raw artifact and manifest hashes recorded, and
target the historical `+5.6pp [+0.9,+10.6]`. However, the original
`results/streaming_replication.pkl` and the one-off paired-bootstrap driver are missing.
Therefore:

- matching the number establishes a reconstructed known answer;
- a mismatch must trigger provenance recovery, not code adjustment toward the target;
- even a match cannot be relabeled as causal or as exact DeepConf.

This historical reconstruction is CPU-only and scientifically separate from the new frozen
IU-PCR protocol.

### 3.2 REFRAIN — recommended first native reproduction

**Paper protocol to pin:** Qwen3-8B thinking mode; MATH500 (500 questions); prompt P0;
temperature 0.6, top-p 0.95, top-k 20; maximum 16,384 new tokens; seed 42; blank-line
reasoning-step segmentation; the appendix reflection-trigger list; SBERT
`all-MiniLM-L6-v2`; similarity candidates `{0.60, 0.65, 0.70, 0.75, 0.80}`; sliding-window
UCB; length-normalized answer likelihood minus a length penalty; force an answer after a
stop. The exact P0 is:

> `{question}` Please answer step by step. End your response with: `Final Answer:
> \boxed{your final answer here}`. Make sure to wrap your final answer in `\boxed{}`.

Run both vanilla and REFRAIN.

REFRAIN is one trace per question, but its SW-UCB state adapts **across questions**. Freeze
the exact MATH500 row order, dataset hash, reset scope, five-arm cold-start order,
tie-breaking, running-mean-length initialization/update, reward timing, model/tokenizer
revision, and chat template. The paper specifies a first-round `0.0001 * L` cold-start
penalty. Preserve its native closure for the native reproduction; a shared closure used to
compare our alarm is a separate adapted-policy lane.

The paper reports Qwen3-8B/MATH500 changing from 91.40% accuracy and 2.64M tokens to 91.20%
and 1.61M tokens. Those are the reproduction targets, not acceptance gates. The public
[Adaptive-Reasoning repository](https://github.com/RLSNLP/Adaptive-Reasoning) currently
contains a release-placeholder README rather than executable code. Freeze our local
implementation and exact trigger list before seeing outcomes; record any paper ambiguity,
especially UCB state/order and forced-answer formatting. The audited paper text defines
`W`, exploration constant `C`, and length weight `lambda` symbolically but does not report
their numerical settings; with executable code still unavailable, those are blocking
reproducibility gaps. A1 must freeze a declared default plus a small sensitivity grid without
calling any setting an exact reproduction.

The 30-question A1 job is only an implementation/protocol pilot: after one cold-start pull
per threshold arm, at most 25 rounds exercise the adaptive controller. Only the full
500-question A2 job can be called the paper-specified MATH500 reproduction.

**Why this cell:** accepted work, native single-trace stopping, modest compute, MATH500
already central to this project, and Qwen3 gives a second Qwen family distinct from the
existing Qwen2.5-Math cache.

### 3.3 DeepConf — official-code protocol pilot, full reproduction deferred

Use the official [DeepConf repository](https://github.com/facebookresearch/deepconf) at a
pinned commit and the paper-pinned vLLM commit
`31f09c615f4f067dba765ce5fe7d00d880212a6d`. Its native result is not a one-trace
detector: it builds a large complete trace pool, uses token confidence derived from top-k
log probabilities, applies 2,048-token group confidence, filters/terminates sampling, and
aggregates answers.

The first job should be a small protocol-validation pilot on the paper's Qwen3-8B/AIME24
cell, preserving native decoding (temperature 0.6, top-p 0.95, top-k 20, 32k maximum), the
paper prompt, and native online warm-up (`N_init=16`). The pilot may use K=32/64
and is **not** a reproduction of the paper's table. A table-level reproduction requires a
4,096-complete-trace pool for each of 30 questions and 64 resampled runs at each working K.

Published known answers for this cell include majority@512 80.0% at 2.32e8 tokens and online
DeepConf-low 86.5% at 0.90e8 tokens (61.1% fewer); the appendix's offline top-25% filter reports
86.9%. Reproduce the paper's exact table metric rather than choosing whichever variant looks
best after the pilot.

Pin the repository commit before defining the confidence statistic. The paper appendix's
sampled-token-exclusion snippet and the current repository's `processors.py` are not the
same implementation: current `main` softmaxes the logits, takes `conf_topk` probabilities,
and averages their negative log probabilities without a sampled-token special case. Record
the exact function, `conf_topk`, whether logits are raw or post-warper, window warm-up,
percentile threshold, answer normalization, and vote rule (`beta >= 0.95` or K budget). No
stored-cache score is “exact DeepConf” until equality with that pinned function is tested.

### 3.4 Streaming Hallucination Detection — conditional reproduction

The paper's linked [code URL](https://anonymous.4open.science/r/Streaming-Hallucination-Detection-D186/)
exists, but the site repeatedly returned connection resets from the web client, curl, and
the in-app browser on 2026-08-16. Therefore the annotated data, split files, representation
layer, probe checkpoints, and scripts have not been audited.

Its native card is BBH plus MuSiQue; more than 10,000 generated trajectories / 200,000 steps;
Llama-3.1-8B, Qwen2.5-7B, and DeepSeek-R1-Distill-8B (roughly 2.5k/2.9k/2.8k usable traces);
sentence-level prediction from exponentially weighted within-step hidden representations;
Claude Sonnet 4.5 step/prefix supervision, semantic answer judging, logical filtering, and a
5% human audit. It trains a supervised probe and does not define an online compute-stopping
intervention. Published step AUCs are 87.83/86.70/93.27. For prefix evaluation, distinguish
the final-prefix AUCs 72.69/81.05/92.18 from local-average prefix AUCs 87.30/88.02/87.98.
Exact prompt, decoding, layer, split, and checkpoint pins remain **unknown until the linked
assets are readable**; do not fill them from inference or silently substitute our prompts.

Do only an asset audit first. If the repository exposes the paper's BBH/MuSiQue trajectories,
Claude prefix/step labels, logical-filter decisions, and checkpoints, reproduce its final
prefix AUC and local-average prefix AUC separately. Do not repeat the reported 10k+ trace
generation and Claude annotation campaign merely to create a baseline. The paper's
step-level AUC and two prefix summaries are not interchangeable.

### 3.5 Do not schedule as core reproductions

- **uPRM:** useful first-error-step ProcessBench ceiling, but it needs a trained 14B model
  and multiple-trace distillation (reported training about 44 H200 GPU-hours).
- **HALT:** bidirectional/full-response classifier; not online-causal and no audited public
  assets.
- **LEASH:** easy to implement, but the published HTML leaves critical trigger constants
  insufficiently pinned and its reported GSM8K savings accompany roughly ten-point accuracy
  losses. Use only as a predeclared sensitivity baseline.
- **ZIP-RC, Adaptive Inference-Time Compute, JET:** require model training or solve a
  materially different adaptive-compute problem.
- **Online Auditing:** reproduce neither its Weibo SVM nor its event-transition model. A
  calibrated stopping rule inspired by it must be labeled a new adaptation.

## 4. Existing telemetry sufficiency

### 4.1 Rich replicated caches: sufficient for all 29 streams

The local `.pkl` files are Git-LFS pointer stubs, but the actual artifacts are present on
Drive and their local manifests record all required channels:

| Cell | Drive artifact | Shape / capture | Size | Sufficiency |
|---|---|---|---:|---|
| GSM8K / Llama-3.1-8B-Instruct | `gdrive:hallucination_detection/cluster_results/regen/trace_gsm8k_llama8b_k10/raw_gsm8k_T1.0.pkl` | 500 problems × 10 traces; top-50; entropy, spilled energy, logsumexp | 1,119,977,097 B | **29/29** project streams; DeepConf exactness needs raw-vs-warper row audit |
| MATH500 / Qwen2.5-Math-1.5B | `gdrive:hallucination_detection/cluster_results/regen/trace_math500_qwenmath15b_k10/raw_math500_T1.0.pkl` | 300 problems × 10 traces; same four channels | 1,283,184,154 B | **29/29**, but only 17.63% accuracy, below the historical 20% quality gate |

Both use temperature 1.0, maximum 2,048 new tokens, and top-50 capture. They are useful for a
no-inference retrospective audit, but they are not native REFRAIN or DeepConf cells and have
already influenced extensive method development. A new split does not erase that corpus-level
selection leakage.

### 4.2 Phase-15 temperature cache: only 25/29 streams

The Drive Phase-15 collection has 12 objects totaling 568,285,797 B. A read-only schema
audit of `math500_qwen7b_T1.0_run0.pkl` (200 rows; 53,155,006 B) found:

`question`, `full_text`, `token_entropies`, `token_spilled_energies`, `top_k_logprobs`
(top-50 ids/logprobs), `gen_token_ids`, `label`, `done`.

It lacks `token_logsumexp`, so the four energy/logsumexp-derived streams cannot be rebuilt.
It supports 25 of the frozen 29 streams, but **not** the declared primary method without
changing its feature contract. Its saved top-k values are post-generation-warper values and
must be labeled a DeepConf proxy unless equality to a pinned official function is shown. It
is nevertheless fully sufficient for the historical entropy-only `lsml16` reconstruction.

### 4.3 Historical replication artifact is missing

`results/streaming_replication.pkl`, cited by the historical log for a +5.6 percentage-point
earliest-prefix result, is absent locally, absent from Git history, and was not found in the
read-only Drive inventory. That historical number is not currently machine-verifiable and
must not be used as a new baseline result.

### 4.4 Frozen-feature causality warning

The 29 token streams cover 30 mixed-v2 global features because entropy CUSUM maps to both
magnitude and location. However, constructing the token matrix once from a completed answer
is **not a valid online evaluation**:

- `trace_length` reveals final length unless it is replaced by observed prefix length;
- CUSUM centers by the complete input's mean;
- STFT uses complete-input centering, padded frames, and interpolation;
- short-prefix PE/window features are backfilled from the first available window.

There is no literal causal version of the frozen 29-stream contract: its 29th stream is
constant **final response length**. Replacing it by elapsed prefix length is a deliberate
semantic change and makes it a deterministic time coordinate. In addition, 11 streams are
documented rolling analogues rather than exact identities to their historical global
features. The deployable primary must therefore be named the **causal 29-stream task
adapter**. Report a causal 28-stream no-length ablation, while full-final-length IU-PCR stays
only as a non-causal ceiling.

The contract can remain frozen only if each decision point rebuilds the complete feature
matrix from `row[:t]` and derives its alarm from that observed-prefix matrix. Then a row for
an earlier token may be retrospectively revised as more of the prefix arrives, but the alarm
at time `t` is still a function only of information available at `t`. This is causal
answer-level detection, **not** a claim that historical token-localization scores are stable
online. Consuming only the terminal row is a stricter alternative, but it makes a
prefix-centered CUSUM's terminal value identically zero and would silently disable three
CUSUM streams covering four global features.

Build the development training matrix by the same prefix-replay rule (with
`trace_length=t`), fit the IU-PCR transform and weights there, and freeze them before any
test-prefix replay. Freeze explicit equal-problem/equal-trace/equal-budget weighting; never
stack all nested prefix rows unweighted, which would let long traces and late budgets
dominate. Report streams that are constant or degenerate at 16/32 tokens. Building the
completed-test token matrix and slicing it later is prohibited. Add synthetic tests proving
that an alarm at `t` is unchanged when arbitrary suffix tokens are appended and that
tokenwise and chunked replay give identical decision-time scores.

## 5. Preregistered apples-to-apples plan

### 5.1 Splits and leakage barriers

- Split by `problem_id`, never by trace: 40% development / 20% calibration / 40% evaluation.
  All K traces for a question remain together; for balance, stratify by a preregistered bin
  of the number correct among K rather than an ambiguous trace-level label. On A0, call the
  last partition **evaluation**, not locked test: these cells have already shaped the method.
  Only the untouched A4 cell is confirmation, with its manifest and score hash frozen first.
- Fit preprocessing, imputation, IU-PCR components, confidence orientation, and any DUFS-LIU
  control on **development only**, with its labels firewalled from label-free arms.
  Orientation must remain label-free. A supervised logistic oracle may use development
  labels, must be marked a different access tier, and never enters the primary same-access
  comparison.
- Use calibration only to freeze alarm boundaries and stopping hyperparameters. Because the
  monitor is repeatedly inspected, calibrate on the **maximum over the complete monitoring
  horizon** under the identical budget grid and censoring rule. Define trace-level FPR as
  “ever alarmed before natural completion,” not fixed-time FPR. Report exact false-alarm
  counts and intervals; 5% is coarse with only about 60 MATH calibration problems.
- Do not sweep an evaluation/test threshold or orient a score using evaluation/test labels.
- Primary prefix grid is absolute observed tokens: `{16, 32, 64, 128, 256, 512, 1024}`
  clipped to model maximum. Oracle fractions of eventual response length may be reported
  only as an explicitly non-deployable secondary diagnostic.
- Bootstrap and paired tests at the `problem_id` level, not the trace level.

### 5.2 Detection lane

**Common target:** binary final-answer incorrectness for a single generated trace.

**Primary score:** at each decision `t`, rebuild the causal 29-stream task-adapter matrix on
`row[:t]`, apply the frozen trajectory-first IU-PCR transform to every observed-prefix row,
and take the maximum risk in that matrix. The alarm process is the running maximum of those
decision scores. This matches the localization pipeline's maximum-risk semantics but is a
new, preregistered answer-level task adapter; retrospective row scores must not be presented
as causal token localization. Report terminal-row risk and top-20%-mean risk as secondary,
not tuning candidates.

**Baselines on the identical traces:**

1. pinned-reference DeepConf per-token confidence and lowest-group confidence when a raw-
   logits equality test passes; otherwise an explicitly named post-warper proxy; either is
   a single-trace adaptation;
2. causal entropy running mean, maximum, slope, and current-token entropy;
3. trailing entropy-window statistics at 16/32/64/128 tokens, with the one primary window
   selected on development only;
4. top-1 log probability and log-probability margin;
5. a development-label-trained supervised logistic oracle using the same frozen inputs,
   isolated as a higher-access ceiling;
6. full-trace IU-PCR as an explicitly non-causal ceiling;
7. DUFS-LIU only as a mechanism/control analysis, never as the primary result.

**Metrics:** AUROC and AUPRC at every prefix (always show prevalence and normalized AP);
recovered above-chance discrimination
`(AUROC_t - 0.5) / (AUROC_full - 0.5)`; earliest absolute budget reaching 95% of that
above-chance signal; sensitivity and precision at calibration-frozen 5%/10% trace-level
ever-alarm FPR; median detection token among detected wrong traces; potential post-alarm
wrong-token exposure; false-abort cost on correct traces; selective risk/coverage; 95%
problem-bootstrap intervals and paired deltas. Only actual branch generations may be called
realized token savings. Report each metric by dataset, model, and preregistered response-
length band, plus score/weight stability across development seeds and adjacent budgets.

### 5.3 Single-trace stopping lane

Use identical model, prompt, decoding, maximum length, answer extraction, and forced-closure
mechanism for all intervention policies. Compare:

- vanilla generation;
- fixed token budgets;
- native REFRAIN;
- LEASH only if its missing constants are frozen from a documented independent rule;
- IU-PCR alarm plus the same forced-answer closure;
- adapted DeepConf alarm plus the same forced-answer closure.

Stopping changes the continuation, so an offline prefix cut cannot determine the emitted
answer. Each policy requiring forced closure needs an actual branched generation after its
calibration-frozen stop. Report task accuracy/pass@1, generated tokens, wall latency,
accuracy-token Pareto dominance, paired per-question accuracy change, and closure failure
rate. Do not treat REFRAIN's similarity trigger as a hallucination probability or place it
in the detection-AUROC table.

### 5.4 Native multi-trace DeepConf lane

Keep the paper's pool/resampling/voting unit and report answer accuracy versus sampled
tokens at K. Do not compare its K-trace answer accuracy directly with one-trace detector
AUROC or one-trace stopping accuracy. The official-code K=32/64 pilot is a protocol check;
only the 4,096-pool/64-resample run can claim table reproduction.

### 5.5 Transfer and confirmation

The no-GPU audit covers two distinct dataset/model families—GSM8K/Llama-3.1-8B and
MATH500/Qwen2.5-Math-1.5B—but both are historically opened development assets. The Qwen
cell's low accuracy also makes it a stress test. Directional consistency there is useful
retrospective evidence, not generalization.

Reserve gpt-oss-20B/CommonsenseQA validation (1,221 questions) under REFRAIN's native P0
protocol as A4. It is a new dataset/model family, and the paper reports both vanilla and
REFRAIN targets. Freeze the causal adapter, baselines, split/order, thresholds, environment,
and score hash before opening outcomes. A generalization claim then requires a positive
paired effect with an interval excluding zero on A4 and the same sign in at least one
retrospective family; otherwise report heterogeneity. MATH500/Qwen3-8B is not independent
dataset transfer.

### 5.6 Frozen failure tests and success rule

Before test labels are opened, write a machine-readable claim registry covering these nulls:

1. the historical earliest-fraction edge disappears at absolute budgets;
2. the effect has the wrong sign on either dataset/model family;
3. residualizing or matching on observed prefix length removes the effect;
4. censoring/cap-pinning or already-finished short traces explain the result;
5. IU-PCR fails to beat the development-selected single entropy/window control;
6. weights, orientation, or selected components are unstable across development resamples or
   budgets;
7. pinned official DeepConf confidence (or the declared proxy if raw equality fails) removes
   the historical approximate-baseline advantage;
8. no stopping policy lies on a better accuracy-versus-token frontier than vanilla/fixed
   budgets;
9. performance collapses in the shortest preregistered response-length stratum;
10. one dataset or one error subtype accounts for the pooled effect.

The primary success claim passes only if the paired, problem-bootstrap improvement over the
strongest same-access baseline is positive with a 95% interval excluding zero on the primary
protocol, has the same sign in at least two dataset/model families, provides a useful earlier
warning or token saving, uses no final length, and the selected intervention has no material
accuracy loss. “Material” must be frozen numerically before A2; recommended non-inferiority
margin is **1 absolute accuracy point**, with both the paired confidence bound and the
accuracy-token frontier reported. If only the early regime wins, a spectral-then-window
hybrid is a new registered method and cannot rescue the primary claim post hoc.

Report access and cost strata separately in every summary:

| Tier | Methods/examples | Model passes / labels |
|---|---|---|
| One trace, logit/log-probability only | IU-PCR, entropy/window controls, adapted DeepConf, LEASH | One generation; no task labels for fitting IU-PCR |
| Multiple traces | Native DeepConf | 16 warm-up traces and then up to K; paper pool uses 4,096/question |
| Hidden-state white-box | Streaming Hallucination Detection | One trace but intermediate representations |
| Supervised | Streaming probe, logistic ceiling, uPRM, adapted online-auditing rule | Development annotations/labels and training passes |
| External judge | Streaming dataset construction | Claude prefix/step labels and semantic answer judgments |

## 6. Required new jobs, cost, and storage

All estimates are order-of-magnitude B200-equivalent inference costs, excluding queue time.
Throughput and response length dominate uncertainty; measure them in the pilot and revise
before scaling. The latest explicit cluster-budget figure in the project log is stale, so
re-query accounting before submission.

| ID | Gate / job | New generation | GPU-hours | Retained storage | Decision |
|---|---|---:|---:|---:|---|
| **H0** | Historical `lsml16` reconstruction from Phase-15 T1 run0 under commit `f13e8bc` | none | 0 | ~0.1 GB input/output | Do first; known-answer diagnostic only |
| **A0** | Materialize the two rich Drive caches and run retrospective leakage-controlled causal replay | none | 0 | ~2.4 GB local working copy; derived scores <1 GB | Feasibility/audit only |
| **L0** | Recompute pinned current DeepConf confidence and Mind-the-Gap paper metrics on existing ProcessBench telemetry | none | 0 | derived scores <0.5 GB | Required before new inference; exact shared-row baseline audit |
| **L1** | Qwen2.5-14B ProcessBench pilot, 30 rows/subset: rich teacher-forced telemetry plus the uPRM paper's independent LLM-as-a-Judge control | none (fixed-chain scoring) | 0.25–1 | <1 GB | Same rows and scorer backbone; paper-aligned reconstruction, not uPRM |
| **L2** | Scale L1 to all 3,400 ProcessBench rows after alignment/marker gates pass | none (fixed-chain scoring) | 0.5–2 | ~0.5–1.5 GB | Recommended new scorer-family transfer; still an opened benchmark |
| **A1** | REFRAIN implementation/protocol pilot, Qwen3-8B/MATH500, first 30 ordered questions, vanilla + REFRAIN, full telemetry | roughly 0.2–0.5M tokens | 0.25–1 | 0.2–0.5 GB | Not a reproduction; approve before A2 |
| **A2** | REFRAIN full 500-question cell, vanilla + REFRAIN | paper reports ~4.25M tokens combined | 2–8 | 2–4 GB | Recommended if A1 passes |
| **A3** | Common forced-closure branches for frozen IU-PCR / adapted DeepConf policies | depends on alarms; likely <2M tokens | 1–3 | <1–2 GB | Only after offline calibration is frozen |
| **A4** | Untouched confirmation: gpt-oss-20B/CommonsenseQA val, native REFRAIN P0, vanilla + REFRAIN plus frozen telemetry scoring | paper reports ~1.25M tokens combined | 2–6 | ~1–3 GB | Required before a transfer/generalization claim |
| **B1** | Official DeepConf protocol pilot, Qwen3-8B/AIME24, K=32/64 | roughly 15–30M tokens | 1–5 | 5–15 GB with top-k telemetry | Recommended protocol check |
| **B2** | Full native DeepConf pool, 30 × 4,096 complete traces | roughly 1.8B generated tokens at paper-like mean lengths | 75–150 | ~20–60 GB native-minimal; 0.6–1.2 TB with full top-50 telemetry | **Defer; separate approval** |
| **C0** | Streaming Hallucination Detection asset audit | none | 0 | negligible | Do when repository becomes reachable |
| **C1** | Probe train/eval if annotations + cached representations exist | no/low inference | ~1–10 | asset-dependent | Conditional only |
| **C2** | Regenerate/extract 3-model hidden-state corpus if assets absent | 10k+ long traces | ~50–150 | ~2–10 GB for selected-layer step summaries; ~0.1–0.5 TB for token-level selected-layer states; potentially multi-TB for all layers | Do not schedule for minimal cycle |

For A1/A2, capture per generated token: entropy, spilled probability/energy, logsumexp, sampled
token id, top-50 token ids/log probabilities, timestamp, stop reason, provisional/forced
answer, and the exact textual step boundaries. Persist manifests, environment, model revision,
dataset hash, prompt, seed, and decoding parameters. Do not retain full hidden states.

## 7. Protocol incompatibilities and failure modes

1. **Different statistical units:** DeepConf's native decision is over a population of
   traces; our primary target is one trace. A per-trace DeepConf score is useful but adapted.
2. **Different labels:** Streaming Hallucination Detection/uPRM use semantic step or prefix
   labels; final-answer correctness can disagree with them. Keep their metrics separate.
3. **Different action semantics:** flagging/abstaining, forcing a final answer, terminating a
   sample, restarting, and filtering a vote have different utility functions.
4. **Non-causal feature construction:** full-trace centering, final trace length, centered
   STFT frames, and prefix backfill leak future information unless features are recomputed
   on the observed prefix. Replacing final length by elapsed length creates a causal adapter,
   not a literal frozen-method reproduction.
5. **Transductive fitting:** the historical streaming helper fitted/fused scores on the
   evaluation cohort and oriented via cohort correlation. It cannot support a deployable
   claim. Development fitting and held-out application are mandatory within a new run.
6. **Opened-corpus selection leakage:** the two A0 cells have already shaped feature and
   method choices. No new split makes them confirmatory; A4 is the untouched test.
7. **Oracle prefix fractions:** “10% of the trace” requires knowing final length. Use absolute
   token/time budgets for primary results.
8. **Repeated-monitor calibration:** a fixed-time 5% threshold does not imply a 5%
   ever-alarm probability. Calibrate the maximum over the full monitoring horizon or use a
   time-uniform boundary.
9. **Threshold leakage:** sweeping an alarm threshold on evaluation/test labels inflates
   results. Freeze boundaries on calibration problems.
10. **Correlated repeats:** K generations of one question are not K independent examples.
   Split and bootstrap by question.
11. **Prompt/decoder mismatch:** entropy and log-probability dynamics move with model,
   temperature, top-p/top-k, thinking mode, and maximum length. A paper result can only be
   claimed under its native protocol.
12. **DeepConf distribution mismatch:** the paper snippet, current repository logits
    processor, raw logits, and post-warper stored log probabilities are not interchangeable.
    Pin and equality-test one implementation.
13. **REFRAIN state coupling:** dataset order, reset scope, cold-start/tie rules, and running
    reward/length state couple questions; shuffling changes the algorithm.
14. **Stopping counterfactual:** truncating a saved trace does not reveal the answer a forced
    closure would generate. Run the branch.
15. **Telemetry truncation:** Phase-15 lacks logsumexp and cannot silently stand in for the
    29-stream method.
16. **Asset uncertainty:** REFRAIN code is not released beyond a placeholder; the Streaming
    code host was unreachable; uPRM/HALT/LEASH did not yield audited official runnable assets.
17. **Hardware and stochasticity:** a B200 reproduction of A100/HF sampling is protocol-
    aligned but not bitwise identical. Report revisions and uncertainty rather than imposing
    exact numerical equality.
18. **Dataset saturation/degeneracy:** GSM8K may be near saturation for Qwen3; the existing
    Qwen2.5-Math-1.5B cell is near the opposite extreme. Report class balance and AUPRC.

## 8. Approval checkpoint

The proposed first authorization is intentionally bounded:

1. copy the verified Phase-15 run-0 and two rich Drive artifacts to a controlled scratch path;
2. run H0 under commit `f13e8bc` and preserve the known-answer mismatch if any;
3. implement and test the causal 29-stream adapter, no-length ablation, grouped splits,
   horizon-level alarm calibration, and a pinned DeepConf raw/post-warper equality audit;
4. run A0 (CPU/no inference) as a retrospective feasibility audit;
5. run L0, then run only the L1 Qwen2.5-14B ProcessBench pilot; do not scale to L2 yet;
6. implement REFRAIN from the frozen paper card and run only A1 (30 ordered questions);
7. run only B1's small official-code DeepConf protocol pilot;
8. return the observed throughput, storage, known-answer checks, and revised L2/A2/A4/B2 budgets
   before requesting any full run.

No Drive mutation, full REFRAIN run, full DeepConf pool, supervised streaming-corpus
regeneration, or canonical project-log update is authorized by this checkpoint.
