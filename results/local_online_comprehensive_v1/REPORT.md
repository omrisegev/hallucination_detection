# Comprehensive Local and Online hallucination-detection cycle

**Final decision: do not promote the joint finalist.** Local transfer is numerically positive but uncertain; Online transfer breaches the frozen regression margin and loses to IU28 in three of four families. No new GPU/inference run is justified by this retrospective cycle.

## Executive result

| stage | frozen evidence | candidate | direct bar | delta | verdict |
|---|---|---:|---|---:|---|
| S1 Local feature/locator | Qwen3-4B GSM8K+MATH development | 0.3517 | Step-272 0.3503 | +0.0014 | PARITY |
| S2 causal Online | Qwen3-4B GSM8K+MATH development | 0.6020 | Step-272 0.5899 | +0.0121 | PARITY |
| S3 Local architecture | Qwen3-4B Olympiad+Omni architecture | 0.3515 | Max entropy 0.3407 | +0.0108 | PARITY |
| S3 Online architecture | Qwen3-4B Olympiad+Omni architecture | 0.5769 | Step-272 0.5746 | +0.0023 | PARITY |
| S4 Local transfer | Qwen3-8B+Llama, four-family audit | 0.3662 | Max entropy 0.3614 | +0.0048 | PARITY |
| S4 Online transfer | Qwen3-8B+Llama, four-family audit | 0.5882 | IU28 0.6104 | −0.0222 | REGRESSES |

The apparent gains in S1-S3 did not become a stable two-task improvement. On the scorer-transfer audit, Local reaches 0.3662 ProcessBench F1 versus 0.3614 for maximum entropy plus the top-five step locator, delta +0.0048 with grouped 95% CI [−0.0264,+0.0375]. Online reaches 0.5882 AUROC@64/128 versus 0.6104 for IU28, delta −0.0222 [−0.0502,+0.0042]. The Online result triggers the preregistered regression verdict even though the interval still includes zero.

## What was tested

- Local representations: raw nine token risks; opened raw-seven drop; all 28 broad token views; provenance-balanced six-family compression; historical core-five.
- Local dynamics: level, innovation, short/long contrast, and their frozen combinations. Locators: peak, first persistent calibration-q90 run, and step top-five mean.
- Online dynamics: level/slow, fast/slow, slow/positive-area/persistence, short-long/innovation/recovery, and the five-state combination. Every score was recomputed from explicitly truncated telemetry at 16/32/64/128/256/512 tokens.
- Same-matrix fusion: equal average, ordinary IU-PCR, historical U-PCR compatibility, uniform Laplacian, DUFS-gated Laplacian, temporal Laplacian for Local, and hierarchical U-PCR where identifiable.
- Architecture: shared Local, independent Local/Online, Global+Local, Global+Online, and all quarter-grid three-signal simplex weights.

## Feature and architecture findings

The useful development mechanism was provenance balancing. `family6 + level + step_top5mean` was the S1 Local selection; `family6 + fast/slow` was the S2 Online selection. The raw-seven opened drop and uncompressed broad-28 candidates did not satisfy the family-stability guard. Innovation and short-long event coordinates were inconsistent for localization.

No same-matrix fusion alternative had a wholly positive interval over ordinary IU. Local hierarchical fusion was +0.0098 numerically but uncertain; Online DUFS/uniform/compatibility changes were below +0.001. The S3 simplicity rule selected the registered Global signal for completed-trace detection and Online scoring, retaining the family-six Local head only for the step locator: two physical heads, 36 fitted coordinates, and six persistent Local state scalars.

Transfer exposed the weakness of that choice: the Global-only prefix score generalized worse than IU28. The finalist beat the Local direct bar in three of four families but beat the Online direct bar in only one of four.

## Direct and compute-heavy competitors

On S4, same-access Tier-A Local methods rank: finalist 0.3662, max entropy/top-five locator 0.3614, GL-LIU v1 0.3364, Step-272 0.3078, and Mind the Gap 0.2646. Tier-B Qwen2.5-Math-PRM-7B reaches 0.7280 and the Qwen2.5-72B critic protocol 0.5895. The critic has 1262/1270 valid scorer-row predictions (8 abstentions, all in OmniMath); its score is therefore a partial-coverage ceiling, never a same-access reference.

On S4 Online, IU28 is 0.6104, Step-272 0.6082, mean entropy 0.5926, DeepConf-w64 0.5922, max entropy 0.5921, finalist 0.5882, and DeepConf-w32 0.5853.

Cross-protocol papers remain context only. uPRM uses next-token probabilities and reports gains over an LLM judge on ProcessBench, but it was not reproduced here. The supervised Streaming Hallucination Detection probe uses hidden states and a different annotated dataset. ProcessBench evaluates the first erroneous step or no-error outcome, while DeepConf motivates the black-box group-confidence baseline. [uPRM](https://arxiv.org/abs/2605.10158), [Streaming Hallucination Detection](https://arxiv.org/abs/2601.02170), [ProcessBench](https://arxiv.org/abs/2412.06559), [DeepConf](https://arxiv.org/abs/2508.15260).

## Non-withdrawable warning behavior

| calibration target | method | audit false warning | wrong-trace coverage | precision | mean first budget |
|---:|---|---:|---:|---:|---:|
| 5% | finalist | 3.7% | 11.8% | 72.5% | 131.4 |
| 5% | Step-272 | 3.2% | 13.0% | 75.4% | 157.9 |
| 5% | IU28 | 3.5% | 8.3% | 56.6% | 381.8 |
| 10% | finalist | 10.1% | 22.3% | 61.8% | 151.6 |
| 10% | Step-272 | 7.6% | 21.2% | 66.5% | 207.5 |
| 10% | IU28 | 8.2% | 16.3% | 61.5% | 334.8 |

Warnings are one-sided and never withdrawn. Potential remaining tokens are diagnostic only; no forced-stop inference was run, so they are not realized savings. At the 5% target, the finalist covers fewer wrong traces than Step-272 (11.8% versus 13.0%). At the 10% target it covers slightly more (22.3% versus 21.2%) but exceeds the transfer false-warning target (10.1% versus 7.6% for Step-272).

## Length, ablations, and failure strata

| method | raw AUROC@64 | residualized | raw AUROC@128 | residualized |
|---|---:|---:|---:|---:|
| finalist | 0.5916 | 0.5825 | 0.5848 | 0.5690 |
| Step-272 | 0.6182 | 0.6096 | 0.5981 | 0.5789 |
| IU28 | 0.6150 | 0.6120 | 0.6057 | 0.5907 |

Length residualization is a non-deployable diagnostic because it uses completed trace length. It does not reverse the finalist/IU28 ordering.

The Local missing-family audit changes only the locator; the Global detector and its threshold stay fixed. Mean deltas after removal are:

| removed source | mean delta F1 | minimum | maximum |
|---|---:|---:|---:|
| family::entropy_dynamics | -0.0127 | -0.0463 | +0.0089 |
| family::entropy_level | +0.0049 | -0.0426 | +0.0418 |
| family::partition_energy | +0.0033 | -0.0277 | +0.0203 |
| family::sampled_energy | -0.0056 | -0.0394 | +0.0418 |
| family::structural | +0.0079 | +0.0000 | +0.0178 |
| family::topk_distribution | +0.0056 | -0.0216 | +0.0401 |
| primitive::entropy | -0.0125 | -0.0486 | +0.0185 |
| primitive::partition_energy | +0.0033 | -0.0277 | +0.0203 |
| primitive::sampled_energy | -0.0056 | -0.0394 | +0.0418 |
| primitive::topk_distribution | +0.0056 | -0.0216 | +0.0401 |

Entropy dynamics and the combined entropy primitive are the only removals with a material mean loss (about −0.0127/−0.0125). Removing structural or top-k families is mildly positive on average but heterogeneous; these are outcome-opened diagnostics and do not authorize post-hoc pruning. Error-position quartiles, calibration-defined short/medium/long strata, and the answer-correct/process-error versus answer-wrong/process-clean cases are retained in `STAGE_4_STRATA.csv` rather than pooled into one misleading score.

## Robustness and cost

All eight cells pass repeated-fit identity, label-permutation identity, feature-order score equivalence, suffix replacement, and chunk-endpoint identity. The largest feature-order remapping discrepancy is 3.47e-16.

Median per-cell feature/head fit time is 109.9s; median complete six-budget Online scoring time is 32.6s; median measured Python peak memory is 148.1 MiB. All work was CPU-only over existing caches: zero GPU hours, no new inference, and no Drive mutation.

## Decision

- Do not replace the Online incumbent with the S3 joint finalist. Retain IU28 as the strongest S4 direct Online bar; Step-272 remains statistically tied and has better 5% warning coverage than the finalist.
- Treat the Local family-six/top-five mechanism as a promising retrospective candidate, not a confirmed replacement. Its transfer delta is positive but uncertain, and simple maximum entropy remains the strongest transparent direct bar.
- Do not reopen graph fusion, event operators, raw-seven pruning, or a Global-only Online path on these opened cells.
- Do not request a GPU run merely to rescue this result. A future cycle needs a materially new signal or fresh unopened evidence: for example, explicit process supervision/hidden-state access under a new authorization, or a genuinely external model/dataset transfer protocol.

## Evidence boundary

All twelve ProcessBench telemetry cells and competitor artifacts were historically opened before this cycle. S1/S2 selected on Qwen3-4B GSM8K/MATH; S3 selected architecture on Qwen3-4B OlympiadBench/OmniMath; S4 audited Qwen3-8B and Llama-3.1-8B scorer telemetry. These scorer copies repeat the same source questions and were resampled together. The result is rigorous retrospective development evidence, not independent confirmation or a SOTA claim.

Frozen protocol SHA-256: `c921b0d446eebd4611c4426168c30410741997ea2c6d23238e5d22b83e8d1e5b`.
