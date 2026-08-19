# Causal GL-LIU for early/online final-error detection — v1

Status: frozen CPU-only retrospective follow-up, 2026-08-16. This campaign
continues the comparison after the IU28 maximum-risk screen showed parity at
64–128 tokens but did not establish superiority.

## Question

Does the two-head architecture that succeeded at ProcessBench localization
improve early prediction of final-answer error when it is replayed causally on
the existing telemetry caches?

This is not a new-inference campaign. It does not authorize GPU jobs, Drive
mutation, raw-data mutation, or an exact competitor reproduction.

## Frozen models

All fits use only the group-disjoint calibration half and never accept labels.
Every evaluation prefix is rebuilt after truncating every aligned telemetry
channel. Completed-trace features or token curves are never sliced.

1. `global_gl_liu_no_length` is the selected localization system's mixed-v2
   answer-level DUFS-LIU head (`lambda=0.1`, `k=7`, seeds 11/23/37), with final
   trace length excluded.
2. `global_gl_liu_elapsed_length` is the same causal task adapter with the
   currently observed prefix length. It must not be called the literal frozen
   final-length model before the trace ends.
3. `local_temporal_gl_liu_max` is the selected localization head,
   temporal-Laplacian IU (`lambda=0.3`), reduced by its historical maximum.
4. `local_dufs_gl_liu_top5` is the DUFS feature-graph local head at
   `lambda=0.3`, reduced by the top 5% mean that was stronger for whole-trace
   error detection in the frozen localization diagnostics.
5. `fused_gl_liu` is the fixed equal-weight mean of standardized
   `global_gl_liu_no_length` and `local_dufs_gl_liu_top5`. Component means and
   scales come from completed calibration traces without labels.
6. `cusum_max`, `sw_var_peak`, and their fixed standardized equal-weight mean
   are mechanism ablations.
7. Controls are the previous `iu28_no_length`, DeepConf entropy windows,
   maximum entropy, and mean entropy.

The local heads use exactly five native token series: entropy, sliding-window
entropy variance, absolute entropy CUSUM, sliding-window spilled-energy
variance, and absolute spilled-energy CUSUM. Step spans are not used to build
or score any method. The target is `final_answer_correct`, never the
ProcessBench first-wrong-step label.

## Evaluation

- Same deterministic group-disjoint 50/50 split and absolute budgets
  16/32/64/128/256/512 as the previous screen.
- At a budget, evaluate only traces whose completed length is strictly larger
  than that budget.
- Report AUROC/AUPRC, convergence to each model's own completed score, final
  decision agreement, grouped paired deltas against DeepConf-w64 and IU28, and
  unchanged held-out early-declaration transfer.
- Equal-family aggregation treats MATH-500 and each ProcessBench dataset as
  five independent dataset families; generator cells do not multiply a
  family's weight.

## Interpretation

The previous superiority gate is not a stop rule. Three conclusions are kept
separate:

1. **Promising parity:** a localization-derived model is close to or above
   DeepConf at 64–128 tokens across more than one family. This justifies
   continuing the comparison and diagnosing the best architecture.
2. **Retrospective superiority:** an equal-family paired delta whose 95%
   interval is above zero, with consistent direction and useful held-out
   declarations. This supports moving to an exact-paper pilot.
3. **Fresh confirmation:** requires a dataset/model family not used to select
   GL-LIU. ProcessBench selection cells remain development evidence; reusing
   the same examples under another detector/model size is not an independent
   dataset-family confirmation.
