# External data collection plan for answer-level hallucination detection

Date: 2026-08-07

## Research scope

The main benchmark must match our actual research task:

- text and reasoning outputs;
- one generated answer at a time;
- a continuous correctness or hallucination score;
- one model generation, without external verifiers;
- label-free fitting;
- correctness labels used only after scores are frozen.

The main methods are deployed U-PCR, IU-PCR, DUFS-LIU, and at most one current
candidate selected before external labels are opened.

This scope excludes Best-of-N response selection, RAG-specific token labeling,
vision-language hallucination, and supervised deferral from the main comparison.
Those tasks can be studied later in separate tracks.

## Corrections to the earlier plan

### FUSE is not a core baseline

FUSE is an unsupervised spectral method, but its task is different. It receives
many candidate answers and scores from several external verifiers, then selects
one answer. Our method scores a single generated answer using token statistics
from the generating model.

FUSE remains useful related work because it studies dependence between imperfect
signals. It may later suggest diagnostics or a Best-of-N extension. It is not a
primary performance baseline and its IMO/HLE protocols are removed from the
immediate data collection plan.

### HLE is an external correctness test, not a hallucination benchmark

Humanity's Last Exam can provide difficult correct/incorrect answers for testing
our detector. It does not provide our uncertainty traces and was not designed as
an answer-level hallucination detection benchmark. It is therefore a secondary
transfer test.

### HUB is relevant, but HALT is a supervised ceiling

HUB is directly relevant because it covers factual and logical hallucinations
across ten capabilities. However, HALT trains a GRU using labels. HALT must be
reported as a supervised baseline, not as a label-free competitor.

The HALT paper also computes log-probability sequences by teacher-forcing
existing annotated responses through Llama 3.1-8B and Qwen 2.5-7B. We are not
using teacher forcing. HUB can enter our on-policy experiment only if its prompts,
provenance, model settings, and grading rules are sufficient to regenerate new
responses on the cluster. Otherwise, it remains a literature benchmark until we
obtain the missing protocol information from the authors.

### White-box methods do not define what we collect

D-Score, SemGrad/HybridGrad, and REDE remain relevant method references. We do
not need to collect their hidden states, gradients, or attention maps for our
method. To test U-PCR under their data protocol, we collect only our own raw
token telemetry during fresh cluster generation.

This gives a protocol-level comparison. It is not a direct paired comparison
unless both methods score exactly the same generated answers.

## What remains valid from the first plan

The following conclusions remain important:

1. Match the other paper's dataset, model checkpoint, prompt, decoding settings,
   sample count, answer parser, and grading rules as closely as possible.
2. Use real on-policy generation on the cluster. Do not use teacher forcing in
   this phase.
3. Reuse a public cache only when it contains the exact outputs and sufficient
   token telemetry for our registered features.
4. If questions and labels are public but token telemetry is missing, rerun the
   inference protocol on the cluster.
5. Save compact raw token statistics during generation instead of trying to
   reconstruct them later.
6. Run a schema smoke test and a scientific pilot before a full cluster run.
7. Fit and hash every unsupervised score before the evaluation code can open the
   label file.
8. Separate label-free methods, supervised methods, external-verifier methods,
   and multi-sample methods in every table.
9. Do not compare published and local metrics as if they were paired when the
   generated outputs differ.
10. A new cluster run is allowed only when a named competing method has already
    been evaluated under the same dataset protocol, or when we will run that
    competitor on the same new outputs.

## Mandatory competitor gate

Before any cluster job is approved, its manifest must name:

- the competing method and paper;
- the exact dataset subset used by that paper;
- the model, prompt, decoding, and grading protocol;
- the competitor's supervision, access, and sampling regime;
- whether the comparison is on the same outputs, the same full protocol, or
  only the same dataset.

Different access regimes are allowed. A supervised method, a hidden-state
method, an external verifier, or a multi-sample method can be a competitor. The
target must still be comparable: a confidence or hallucination score for the
same answer-level correctness label.

Current decisions:

| Proposed data | Named competitors in published evaluation | Decision |
|---|---|---|
| Completed Mind the Gap data: GSM8K, MATH, ProcessBench | Evidence Drop; LN-S; LogTokU; Shannon uncertainty | Audit existing artifacts; do not rerun |
| RRB perturbations over AIME 2024/2025 | None found for hallucination or correctness-confidence detection on the perturbed RRB cells | Block new generation |
| HUB | HALT-L, HALT-Q, PPL, overall entropy, alternative-token entropy, entropy change, Lettuce; FAVA/RAGTruth subsets also include LLM-Check and FAVA Model | Block until an exact response plus log-probability cache is available, or a valid on-policy protocol is released |
| Official HLE | Model-stated confidence and RMS calibration in the HLE paper; HTC in Agentic Confidence Calibration uses HLE under a different agent protocol | Conditional: use the official HLE protocol and compare against stated confidence on the same outputs |
| HLE-Verified Gold alone | No answer-level UQ detector was found that reports on this exact verified subset | Do not run as a standalone benchmark; use it only as a label-quality slice of an official HLE run |
| SemGrad: SciQ, TriviaQA, TruthfulQA | SemGrad, HybridGrad, ParaGrad, LN-PE, P(True), Self-Consistency, Deg, INSIDE, Semantic Entropy, Semantic Density, mutual information, G-NLL, SAR, ExGrad | Approved for a protocol audit; strongest candidate for the next new cluster collection |
| ReDe: TruthfulQA, Math, CodeElo, MultiHopQA | ReDe with CCS or supervised probing; Perplexity; Verbalized Certainty; TSV; HaloScope; CED; HaMI; SelfCheckGPT; Semantic Entropy; EigenScore; P(True); RACE; RHD; HalluGuard; ARS | Approved in principle, but wait for official code and exact data construction before generation |
| D-Score: FAVA-Annotation, RAGTruth | D-Score; LLM-Check Hidden Score; perplexity; window entropy; logit entropy | Block for our method unless a cache contains the original outputs and token telemetry; these are fixed-response corpora and otherwise require forced scoring or new, non-paired generations |

## Comparison levels

There are three levels of evidence:

1. **Same exact outputs:** both algorithms score the same answers. This is the
   strongest direct comparison.
2. **Same inference protocol, new sampled outputs:** dataset, model, prompt,
   decoding, and sample count match, but the answers are newly sampled. This is
   a protocol reproduction, not a paired algorithm comparison.
3. **Same dataset only:** model or prompt differs. This is external validation,
   not reproduction of the other paper.

Every reported result must state its level.

## Revised priority

### 0. Audit the completed Mind the Gap run

Mind the Gap has already been generated on the other computer. Do not rerun it.
After the artifacts arrive:

1. Verify the exact model, model revision, dataset split, prompt, chat template,
   decoding settings, number of samples, and seeds.
2. Check that each answer contains complete token telemetry and a valid stop
   reason.
3. Reproduce the paper's response counts, correctness prevalence, and basic
   accuracy before evaluating hallucination detection.
4. Confirm whether ProcessBench, GSM8K, and MATH were included and whether their
   prompts match the paper exactly.
5. Fit deployed U-PCR, IU-PCR, DUFS-LIU, and one frozen candidate without labels.
6. Hash the scores and only then compute AUROC, AUPRC, and selective-risk
   metrics.

This is the first result to analyze because its data collection is already done
and Mind the Gap is close to our answer-level, single-generation setting.

### 1. Do not run RRB yet

RRB evaluates model accuracy, efficiency, refusal, and robustness under prompt
perturbations. It does not evaluate a hallucination detector or a correctness
confidence method on the perturbed cells. ReDe includes AIME 2024 and 2025 in a
larger Math set, but it uses greedy decoding and different Qwen/DeepSeek models;
this is not the RRB perturbation protocol.

Therefore the proposed 13,440-response RRB collection is cancelled for now. It
can be reopened only if one of these conditions becomes true:

1. a published detector evaluates on the RRB perturbation protocol;
2. we pre-register and run a named competitor on the same RRB outputs;
3. the experiment is explicitly reclassified as a robustness diagnostic and is
   funded separately from the benchmarking data collection.

### 2. Perform a HUB feasibility audit before scheduling generation

HUB is a valuable target because its test set covers:

- Algorithmic Reasoning;
- Commonsense Reasoning;
- Mathematical Reasoning;
- Symbolic Reasoning;
- Code Generation;
- Chat;
- Data-to-Text;
- Question Answering;
- Summarization;
- World Knowledge.

The paper reports 60,008 training, 7,342 validation, and 8,114 test responses.
The reasoning capabilities are held out from training and are especially useful
for testing task transfer.

Before any cluster job, determine whether the official release provides:

- the original prompt for every response;
- the source dataset and stable row ID;
- the original generating model, exact checkpoint, and chat template;
- decoding parameters and seed policy;
- response-level correctness or hallucination labels;
- redistribution and evaluation licenses.

Then choose one of three outcomes:

1. If prompts and full generation metadata exist, regenerate selected reasoning
   subsets on-policy and collect our token telemetry.
2. If only prompts and labels exist, run a clearly named new HUB-style
   generation with a registered open model. This is external validation, not an
   exact HALT reproduction.
3. If only annotated responses exist, contact the authors and defer HUB. Do not
   teacher-force those responses under the current protocol.

HALT may later be reproduced as a supervised ceiling using the official
train/validation/test split. Its score must never be presented as label-free or
used to choose U-PCR features and hyperparameters.

### 3. Use official HLE only as a conditional transfer test

HLE is useful for difficult, broad-domain correctness. It is secondary because
it is not a hallucination benchmark and extreme difficulty may leave too few
correct responses for stable AUROC.

Use on-policy cluster generation:

- pin the exact dataset and model revisions;
- begin with text-only questions;
- use the official output format asking for Explanation, Answer, and Confidence;
- use temperature 0;
- allow at least 8,192 completion tokens for a reasoning model;
- use the official parser or judge;
- keep the model's stated confidence only as an evaluation baseline.

The direct baseline is the confidence value requested by the official HLE
prompt. Compute its AUROC, AURC, Brier score, and RMS calibration error on the
same outputs used by U-PCR. This gives a clear same-output competitor even
though it is a verbalized-confidence method.

Agentic Confidence Calibration also evaluates HTC on HLE, but its outputs come
from a smolagents CodeAct trajectory. Its published number is contextual only
unless we reproduce that agent protocol.

HLE-Verified divides 2,500 items into 668 Gold, 1,143 Revision, and 689
Uncertain items. It is a label audit, not a separate detector benchmark. Use:

- text-only Gold as the primary quality-controlled subset;
- text-only Revision as a separate secondary subset, with revised question and
  answer used together;
- no Uncertain items in the headline result.

First run a stratified 200-item pilot from the official HLE IDs, while retaining
the HLE-Verified status for later slice analysis. Continue only if generation
quality is healthy and both correctness classes are present. A final AUROC is eligible only
when the evaluated cell contains at least 30 correct and 30 incorrect answers.
Otherwise, report HLE only as a descriptive stress test.

### 4. Make SemGrad the next new protocol candidate

SemGrad is the cleanest next benchmarking target because the official paper
provides code and data, generates answers on-policy with greedy decoding, uses
three open models, and evaluates eleven established UQ baselines.

The exact published cells are:

- SciQ validation: 1,000 questions;
- TriviaQA open-domain test: 11,313 questions;
- TruthfulQA: all 817 questions;
- Qwen3-Instruct-4B, Mistral-Nemo-Instruct-12B, and
  Llama-3.1-Instruct-8B;
- greedy decoding;
- BEM correctness labels;
- AUROC as the primary metric and AURC as a secondary metric.

Start with one exact model, preferably Qwen3-Instruct-4B, and the SciQ and
TruthfulQA cells. Before cluster generation, reproduce the official prompt,
chat template, model revision, answer length, BEM threshold, and code commit.
Collect our raw token telemetry during the same greedy generation. Run at least
the sampling-free competitors LN-PE and G-NLL on the same outputs. If resources
allow, run official SemGrad/HybridGrad as white-box competitors and SAR or
Semantic Entropy as a multi-sample competitor.

### 5. Keep ReDe as the reasoning-specific follow-up

ReDe is more directly about reasoning traces. It evaluates TruthfulQA, a Math
mixture containing MATH500 and AIME 2024/2025, CodeElo, and MultiHopQA with
Qwen3-8B/32B and DeepSeek-R1-Distill 8B/32B. The default generation is greedy.
Its competitor set is broad and includes supervised, unsupervised, white-box,
verbalized, and multi-sample methods.

Do not start generation until the exact dataset construction, prompt templates,
official model revisions, judge configuration, and implementation are available.
The current arXiv version does not link an official code repository.

### 6. Keep D-Score in a separate fixed-response track

D-Score evaluates FAVA-Annotation and RAGTruth by feeding existing annotated
responses through Llama-2-7B, Llama-3-8B, and Vicuna-7B and reading hidden
states from a single forward pass. It compares with LLM-Check Hidden Score and
single-response probability/entropy baselines.

This is a clear competitive benchmark for hidden-state methods, but it is not a
valid new generation target for U-PCR unless the exact response-level token
telemetry is released. Fresh generation would create different answers and
labels. Keep it blocked under the current no-forced-scoring decision.

### 7. General audit rule for any later method

For each later method:

1. Pin the paper version, official code, dataset revision, model checkpoint,
   prompt, decoding settings, and sample count.
2. Decide whether its task is answer-level, single-generation hallucination
   detection.
3. Name at least one competitor that will be reported on the same outputs or the
   same full protocol before approving any cluster work.
4. If it matches our scope, reproduce the inference protocol on the cluster and
   collect our token telemetry.
5. Treat the paper's published score as contextual unless the methods score the
   same exact outputs.
6. Do not collect hidden states, gradients, or attention maps unless we later
   explicitly decide to reproduce that competing method itself.

Other categories remain separate:

- HALT is a supervised ceiling.
- FlowScore uses external decomposition and verification.
- Frequency-Aware Attention and TRIVIA+ belong to a contextual/RAG track.
- GAUSS and GRAPHEVAL are multi-generation methods.
- LHD belongs to long-form factual consistency, not CoT reasoning.
- Vision-language methods do not belong in this text benchmark.

## Raw data contract for every new cluster run

Save telemetry during generation. For each answer, save:

- dataset name, split, exact revision, source row, and item ID;
- model ID and exact model revision;
- tokenizer revision and chat-template hash;
- system prompt, user prompt, and fully rendered prompt;
- input token IDs;
- generated text and generated token IDs;
- generated-token character offsets;
- post-warp top-50 token IDs and log-probabilities;
- raw pre-warp top-50 token IDs and log-probabilities;
- raw full-vocabulary `logsumexp` at each generated token;
- negative log-probability of the sampled token;
- the top-15 entropy trace used by the current feature implementation;
- all decoding parameters and the random seed;
- stop reason, truncation flag, runtime, and peak memory;
- answer extraction and grading diagnostics.

Full vocabulary logits do not need to be stored. Top-k values alone are not
enough for the registered partition-energy features; the full-vocabulary
`logsumexp` must also be saved.

Correctness labels must be stored separately. The score-fitting program must
write and hash all scores before the evaluator can open labels.

## Cluster execution ladder

### Stage A: lock the protocol

Create an immutable manifest containing:

- source paper and version;
- dataset, model, and tokenizer revisions;
- exact prompts and chat template;
- decoding parameters, sample count, and seeds;
- grading implementation and version;
- telemetry schema version;
- frozen methods and hyperparameters.

Any later change creates a new experiment version.

### Stage B: schema smoke test

Run two questions with two responses each. Verify:

- token arrays have compatible lengths;
- generated token IDs and sampled-token probabilities agree;
- entropy, sampled-token surprise, and `logsumexp` are finite;
- saved text can be regraded;
- checkpoint/resume creates no duplicates;
- fitting inputs contain no labels.

### Stage C: scientific pilot

- SemGrad protocol: 200 SciQ and 200 TruthfulQA questions with one exact model.
- Official HLE: 200 stratified text-only questions, with HLE-Verified status
  retained only for slice analysis.
- HUB: define a pilot only after the feasibility audit and cache gate pass.
- RRB and D-Score: no pilot is approved.

The pilot checks accuracy prevalence, truncation, feature variance, numerical
rank, runtime, and storage. It must not be used to select a method by AUROC.

### Stage D: full generation

Use immutable output paths, resume-safe sample IDs, periodic checkpoints, and a
completion manifest containing counts and hashes.

### Stage E: freeze and evaluate

Fit without labels:

- deployed U-PCR;
- IU-PCR;
- DUFS-LIU;
- one pre-registered current candidate;
- registered single-feature controls.

After score hashes are fixed, evaluate AUROC, AUPRC, selective risk/AURC,
accuracy at fixed coverage, runtime, feature diagnostics, and fallback rates.
Use cell-macro and dataset-family-macro summaries. Confidence intervals must
respect question-level and dataset-level dependence.

## Immediate next actions

1. Audit the incoming Mind the Gap artifacts and compare against Evidence Drop,
   LN-S, LogTokU, and Shannon uncertainty.
2. Freeze the exact method roster and feature contract for external tests.
3. Pin the SemGrad repository commit and reproduce its Qwen3-Instruct-4B prompt,
   model, BEM grading, and sampling-free baselines locally on a tiny fixture.
4. Prepare a 200+200 SciQ/TruthfulQA SemGrad-protocol pilot with full telemetry;
   do not submit it until the manifest names the exact competitors to report.
5. Audit HUB release availability and contact the authors if generation
   provenance is missing.
6. Keep RRB and D-Score blocked.
7. Audit ReDe code and data release as the next reasoning-specific candidate.
8. Run official HLE only after SemGrad, and only with same-output stated
   confidence as a registered competitor.

## Primary sources

- [HALT and HUB](https://arxiv.org/html/2602.02888v1)
- [Robust Reasoning Benchmark, version 3](https://arxiv.org/html/2604.08571v3)
- [Official Humanity's Last Exam repository](https://github.com/centerforaisafety/hle)
- [Official HLE dataset](https://huggingface.co/datasets/cais/hle)
- [HLE-Verified repository](https://github.com/SKYLENAGE-AI/HLE-Verified)
- [HLE-Verified paper](https://arxiv.org/abs/2602.13964)
- [FUSE, retained as related work only](https://arxiv.org/html/2604.18547)
- [SemGrad and HybridGrad](https://arxiv.org/html/2605.04638v2)
- [Reasoning Denoiser (ReDe)](https://arxiv.org/html/2607.22098)
- [D-Score](https://arxiv.org/html/2607.24586)
- [Agentic Confidence Calibration](https://arxiv.org/html/2601.15778)
- [Mind the Gap](https://openreview.net/pdf?id=gllCfOG1Gt)
