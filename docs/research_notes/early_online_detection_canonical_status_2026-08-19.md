# Early and Online Hallucination Detection: Canonical Status and Benchmarking Contract

**Date:** 2026-08-19

**Status:** historical evidence consolidated; current method development frozen; fair paper comparison pending
**Scope:** predicting final-answer failure from an unfinished reasoning trace

## 1. Purpose

This document is the canonical entry point for the project's early and online
hallucination-detection work. It consolidates the original streaming pilot, its
clean-cache replication, the later causal-prefix studies, the current method
position, competitor fidelity, and the next apples-to-apples benchmarking
protocol.

The central question is:

> Can a label-free score predict that the completed reasoning answer will be
> wrong while the answer is still being generated, early enough to support a
> useful warning or stopping decision?

This is not the same task as first-error localization. Early detection predicts
the final answer label from a prefix. Localization identifies where the first
reasoning error begins. The two tasks may share causal token features, but they
must have separate metrics, calibration, and claims.

## 2. Claim boundary

The evidence supports an early signal, but not a completed online system or a
leadership claim.

- The historical spectral method has one replicated positive result at the
  earliest 10% of a trace.
- The original preregistered superiority gate failed at absolute token budgets.
- The historical DeepConf comparison used an approximation, not the paper's
  exact confidence signal or online multi-trace protocol.
- Later, broader causal-prefix experiments show parity or regression against
  the strongest matched simple baselines. They do not confirm that spectral
  fusion is the best online detector.
- Fixed-budget AUROC is a detector metric. It is not evidence of realized token
  savings. A stopping claim requires forced-closure inference and an
  accuracy-versus-total-tokens evaluation.
- Current ProcessBench studies are retrospective because their labels and
  populations were used in earlier development.

The correct present description is:

> Early hallucination evidence exists in causal confidence traces. Our methods
> are competitive with a same-access DeepConf-style proxy, but exact competitor
> reproduction and a validated stopping policy remain open.

## 3. Historical result: Extension E

### 3.1 Step 148 pilot

The original pilot computed the spectral feature suite on growing prefixes and
compared L-SML fusion with causal entropy summaries and a DeepConf-style window
baseline. It used final-answer correctness only for evaluation.

Two clean cells were usable:

- GSM8K / Llama-3.1-8B, `n=200`;
- MATH-500 / Qwen2.5-Math-1.5B, `n=400`, non-canonical.

Two R1/GPQA cells were excluded from scientific interpretation because 99--100%
of the traces were truncated at a 1,024-token cap.

The preregistered results were:

| Gate or diagnostic | Result | Interpretation |
|---|---:|---|
| AUROC at 50% of the trace at least 95% of full-trace AUROC | PASS on both clean cells | Useful signal appears before completion |
| Fusion beats the best DeepConf-style window by at least 2 points at two absolute budgets on two clean cells | **FAIL** | No general streaming superiority |
| Earliest 10%: GSM8K | **+9.8 AUROC points**, 95% CI `[+2.3,+17.1]` | Significant retrospective fractional-budget edge |
| Earliest 10%: MATH-500 | **+4.6 points**, 95% CI `[+0.6,+8.7]` | Significant retrospective fractional-budget edge |
| GSM8K causal monitor at 10% false alarms | detects 38% of wrong traces | Partial warning coverage |
| GSM8K potential wrong-trace token saving | 28% | Counterfactual abort-at-warning estimate, not realized system saving |

The fractional 10% budget uses the completed response length and is therefore
not deployable. All publishable online comparisons must use absolute budgets.

### 3.2 Clean-cache replication

On the canonical fresh MATH-500/Qwen-7B raw-trace cache, `lsml16` again beat the
best historical DeepConf-style window at the earliest 10%:

```text
delta = +5.6 AUROC points
paired 95% CI = [+0.9,+10.6]
```

This confirms that the early-prefix effect was not caused only by the original
non-canonical MATH cell. It does not repair the failed absolute-budget gate or
make the 10% budget causal.

Canonical historical artifacts:

- `results/Streaming_Pilot_Explainer.html`;
- `HISTORY.md`, Step 182, and `Research_Directions.md`, Extension E, record the
  clean-cache replication. The historical `results/streaming_replication.pkl`
  payload is not present in this checkout.
- `spectral_utils/streaming_utils.py`;
- `scripts/streaming_pilot.py`;
- `scripts/streaming_pilot_report.py`;
- `HISTORY.md`, Step 148 and the Step-182 replication;
- `Research_Directions.md`, Extension E.

## 4. Competitor fidelity

### 4.1 Streaming Hallucination Detection in Long Chain-of-Thought Reasoning

The paper at arXiv:2601.02170 defines the closest direct prefix-detection task.
It uses supervised probes over intermediate hidden states, with anchor and
synchronization losses and externally produced step annotations. Its custom
BBH/MuSiQue-derived assets, official rows, training split details, and complete
reproduction boundary are not currently available in a form that supports an
exact local reproduction.

Therefore:

- its published scores are `published-context-only`;
- it must not be shown as an exact head-to-head result;
- an adapted common-protocol hidden-state probe is a separate white-box and
  supervised access tier.

### 4.2 Deep Think with Confidence (DeepConf)

DeepConf, arXiv:2508.15260, is primarily an adaptive multi-trace reasoning and
stopping method. Its local token confidence `C_i` is calculated from top-k
probabilities and excludes the sampled token in the official implementation.
Its published local/tail summaries use a 2,048-token window and operate inside
a multi-trace selection and consensus process.

The project's historical helper
`streaming_utils.deepconf_lowest_group_conf` instead uses `-H(t)` and a
64-token window. Every Step-148 "DeepConf" comparison is therefore a
**DeepConf-style entropy-window proxy**, not a faithful reproduction.

The exact distinction is mandatory in every report:

| Row | Fidelity label |
|---|---|
| Historical `-H(t)`, window 64 | `adapted-common-protocol` / DeepConf-style proxy |
| Exact saved `C_i`, verified against pinned official code | `paper-specified` or `official-exact`, depending on the remaining protocol |
| Published adaptive multi-trace accuracy/token results | `published-context-only` until the full protocol is reproduced |

The new DeepConf M2 acquisition was only 3.75% complete at the Step-274 audit.
It cannot enter a headline table until completion, row alignment, confidence
verification, and manifest checks pass.

### 4.3 Online Auditing of Information Flow

Oren-Loberman, Azar, and Huleihel provide the missing decision-theory concept:
a sequential risk that prices both error and delay, with a two-sided posterior
threshold related to an SPRT. The paper is not a hallucination benchmark and
its offline estimator is supervised. It is method inspiration for a stopping
rule, not a reproduced competitor result.

### 4.4 Other online or adaptive-compute methods

REFRAIN and LEASH belong in the stopping/adaptive-compute lane when their exact
assets and protocols are available. They must not be compared numerically with
prefix AUROC. The fair-comparison inventory must continue to search the current
2025--2026 primary literature for new prefix detectors, online PRMs, selective
generation, and adaptive inference methods before freezing the final competitor
set.

Primary paper records:

- `papers/index.md`;
- `papers/digests/deep-think-with-confidence.md`;
- `papers/digests/online-auditing-of-information-flow.md`;
- `papers/PAPER_EXACT_SOURCES.md`;
- `docs/research_notes/reasoning_localization_methods_and_benchmarks_2026.md`.

## 5. Later causal-prefix development

### 5.1 Step 269 broad existing-cache screen

The screen expanded the evidence to 11 materialized cells and five dataset
families without new inference. At absolute budgets:

| Method | AUROC at 64 | AUROC at 128 |
|---|---:|---:|
| IU28 | 0.648 | 0.694 |
| same-access DeepConf entropy-w64 proxy | 0.616 | 0.671 |
| equal-family delta | +0.024 `[-0.005,+0.056]` | +0.014 `[-0.031,+0.058]` |

Both intervals cross zero. The correct verdict is promising parity, not
leadership.

IU28 also converged toward its own completed-trace decision: prefix/final
Spearman correlation rose from 0.417 at 64 tokens to 0.659 at 128 and 0.817 at
512. However, the calibration-constrained declaration rule reached only 0.366
macro coverage and 0.137 held-out ever-wrong rate; only 5/11 cells met the 10%
target.

### 5.2 Step 270 dynamic monitor test

Three label-blind heads based on current/running maximum, positive
area/persistence, and slope/recovery did not beat frozen IU28. Their paired
deltas versus IU28 at the equal-family 64/128 endpoint were:

- `-0.0051 [-0.0553,+0.0519]`;
- `-0.0079 [-0.0663,+0.0639]`;
- `-0.0270 [-0.0979,+0.0561]`.

The added dynamics mainly repackaged the existing CUSUM/`sw_var` magnitude
signal. This closed those specific coarse-monitor transformations, not all
token-native online modeling.

### 5.3 Steps 271--272 token-native architecture

The next frozen experiment created separate Global, Local, and Online feature
semantics from raw causal token channels. The architecture comparison selected
two physical heads, Global and Local. A third independent Online head did not
help:

| Architecture | Global AUROC | Local F1 | Online AUROC at 64/128 |
|---|---:|---:|---:|
| one shared head | 0.6892 | 0.2397 | 0.6009 |
| two Global+Local heads | 0.7164 | 0.3136 | 0.6075 |
| three independent heads | 0.7164 | 0.3136 | 0.6009 |

At 5%/10% trace-level false-warning targets, the selected policy warned on
14.7%/25.0% of wrong traces and produced 3.6%/8.1% observed false warnings.
These are warning statistics, not realized token savings. Phase-15 transfer was
weak at 0.5142/0.5555 AUROC at 64/128 despite 0.8368 final AUROC.

### 5.4 Step 273 transfer audit

The comprehensive Local/Online cycle selected richer development candidates,
but transfer rejected promotion:

- Local: 0.3662 versus 0.3614 for maximum entropy plus the matched locator,
  delta `+0.0048 [-0.0264,+0.0375]`;
- Online: 0.5882 versus 0.6104 for IU28,
  delta `-0.0222 [-0.0502,+0.0042]`.

Local was parity and Online breached the frozen non-inferiority margin.

### 5.5 Step 274 Unified-28 decision

Unified-28 is the frozen unified method-of-record for the next comparison
replay. It uses seven causal streams crossed with `level`, `ewma16`,
`positive_area`, and `persistence`. On the frozen Llama transfer panel:

| Task | Unified-28 | matched task-specific incumbent | Delta |
|---|---:|---:|---:|
| Global AUROC | 0.6629 | 0.6870 | -0.0241 |
| Localization F1 | 0.2880 | 0.2419 | **+0.0461** |
| Early AUROC | 0.5587 | 0.5777 | -0.0189 |

Localization improves significantly, while Global and Early breach their
regression margins. Unified-28 must be included in the next benchmark suite,
but it does not replace the dedicated task heads.

Later contextual-router and geometry studies do not change this early-detection
decision. They found no reliable target-routing key and are diagnostic-only.

## 6. Current method roster for benchmarking

Every eligible common-protocol early-detection table should contain:

1. **Unified-28 ordinary IU-PCR** — frozen unified method;
2. **the dedicated frozen Online incumbent** — required because Unified-28
   regresses on Early;
3. **IU28 without final length** — historical causal reference;
4. **historical `lsml16`** — only for exact reproduction of the original
   earliest-prefix result;
5. **maximum, mean, and windowed entropy controls**;
6. **DeepConf-style entropy-w64 proxy**, labelled as an adaptation;
7. **exact DeepConf-derived confidence**, only after official-code and telemetry
   verification;
8. **full-trace score**, clearly labelled as a non-causal ceiling.

Do not silently call `lsml16`, IU28, Unified-28, and the dedicated Online head
the same method. They answer different historical or current questions.

## 7. Permanent causal and statistical rules

- A score at token `t` may use only tokens `1..t`.
- Do not use final response length, future steps, suffix statistics, final
  correctness, or test-selected orientation.
- Use absolute budgets `16/32/64/128/256/512`; fractional budgets are
  retrospective diagnostics only.
- Fit standardization, weights, signs, and thresholds on the declared
  development/calibration population and freeze them before evaluation.
- Labels may calibrate a decision threshold only when the access tier and split
  say so. The resulting system is an unsupervised scorer with a calibrated
  policy, not a wholly label-free policy.
- Split and bootstrap by source question. Keep repeated scorer/model copies of
  the same question in one resampling unit.
- Report AUROC and AUPRC at every budget, time to warning, ever-warning FPR,
  wrong-trace coverage, selective error, and prefix/final agreement.
- Report score computation, model passes, hidden-state/logit access, wall time,
  memory, and total generated tokens.
- Never convert a potential remaining-token estimate into a token-saving claim
  without forced-closure inference.

## 8. Apples-to-apples comparison lanes

Early work must be reported in two separate lanes.

### Lane C: causal prefix detection

Use identical generated traces and absolute budgets. Compare scores using
AUROC/AUPRC, grouped intervals, time to reliable warning, and ever-warning
behavior. A hidden-state probe is a separate white-box/supervised access tier.

### Lane D: stopping and adaptive compute

Compare accuracy/pass@1 against total generated tokens, latency, forced
closures, parser failures, and full accuracy-compute frontiers. Keep
single-trace and multi-trace methods separate. Detector AUROC is not a token
savings metric.

Every result row must use one fidelity label:

- `official-exact`;
- `paper-specified`;
- `paper-specified-partial`;
- `adapted-common-protocol`;
- `published-context-only`;
- `blocked-assets`.

The complete four-lane integration contract, including Global and
Localization, is `HANDOFF_fair_paper_exact_comparisons_2026-08-18.md`.

## 9. Next approved research checkpoint

Before new inference, produce a read-only protocol and asset audit:

1. update the 2025--2026 primary-literature map;
2. create protocol cards for every serious prefix/stopping competitor;
3. inventory local and `gdrive:` assets, row IDs, telemetry, completion, and
   manifests;
4. identify which exact comparisons are CPU-scoreable now;
5. separate exact reproduction, adapted common protocol, published context,
   and blocked assets;
6. estimate GPU hours and storage for each missing high-value reproduction;
7. obtain user approval before downloading large artifacts or launching jobs.

The first execution priority is the smallest CPU-first replay on already paid
for data. New generation is a last resort.

## 10. Documentation map

| Purpose | Canonical source |
|---|---|
| Original experiment narrative and exact pilot numbers | `HISTORY.md`, Step 148 |
| Clean-cache replication | `HISTORY.md`, Step 182, and `Research_Directions.md`, Extension E. The historical pickle payload is not present in this checkout. |
| Historical thesis direction and failed gate | `Research_Directions.md`, Extension E |
| Current handoff status | `PROGRESS.md`, Steps 269--274 |
| Joint Local/Online protocol and results | `docs/experiments/GLOBAL_LOCAL_ONLINE_IU_V1.md` and later v2/comprehensive protocols |
| Current unified method decision | `docs/experiments/UNIFIED_CAUSAL_IU_V1.md`, `docs/experiments/UNIFIED_CAUSAL_SUBSET_SEARCH_V1.md`, `docs/reports/UNIFIED_CAUSAL_IU_V1_REPORT.md`, and Step 274 records |
| DeepConf formula and fidelity warning | `papers/digests/deep-think-with-confidence.md` |
| Sequential stopping inspiration | `papers/digests/online-auditing-of-information-flow.md` |
| Fair paper comparison program | `HANDOFF_fair_paper_exact_comparisons_2026-08-18.md` |

This document supersedes informal summaries that describe the project as
already having a complete online method or an exact DeepConf reproduction.
