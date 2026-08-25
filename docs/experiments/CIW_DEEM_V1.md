# CIW-DEEM v1

**CIW-DEEM** means **Cross-fitted Innovation-Weighted DEEM**. It is the
official structured-input challenger built on continuous B3.

## Method

The original feature inventory and historical B3 family boundaries are kept.
The highly duplicated H15/H-saved entropy pair is represented as common and
support-difference coordinates. On the universal 3-by-3 source-by-operator
core, every coordinate is ridge-predicted from the four coordinates sharing
its source or operator. Five group/length folds estimate out-of-fold R-squared.
The frozen coordinate gate is

```text
alpha_j = 0.5 * clip(R2_oof_j, 0, 1)
innovation_j = (x_j - xhat_j) / sd(x_j - xhat_j)
x'_j = (1 - alpha_j) * x_j + alpha_j * innovation_j
```

No correctness labels enter the input model or B3 fit. The sample-conditional
residual gate and matched non-rook support are frozen negative ablations; the
static R-squared rook gate is the registered v1 method.

## Registered 24-cell result

- Equal-dataset-family AUROC: `0.7492330051057238`.
- Cell-macro AUROC: `0.7820255514493354`.
- Equal-dataset-family AUPRC: `0.7791317276773182`.
- Cell-macro AUPRC: `0.7517170841581265`.
- Equal-family AUROC delta versus frozen B3: `+0.0007316506044068283`.
- Exact eight-family one-sided sign-flip p-value: `0.13671875`.
- Promotion threshold: `+0.0025`; not met.

The method is therefore an **official challenger**, not a promoted champion.
It has the highest equal-family point estimate among the directly comparable,
five-seed B3 input variants in this development line.

## Task boundary

The registered score is completed-response hallucination detection. It is not
a first-error localization or causal early-detection score. Those tasks require
separate adapters:

- localization: construct the same source/operator core over causal moving
  windows, fit a window-level CIW-DEEM score, then freeze a token locator;
- early detection: recompute the same feature contract from each causal prefix,
  fit only on calibration prefixes, and evaluate at absolute token budgets.

Neither result may be inferred from the completed-response experiment. Existing
GL-LIU localization and IU28 early-detection numbers belong to different
methods and cannot be relabeled as CIW-DEEM.

## Canonical files

- `spectral_utils/ciw_deem.py`
- `spectral_utils/deem_b3_unsupervised_input_gate.py`
- `configs/ciw_deem_v1.json`
- `scripts/run_ciw_deem_v1.py`
- `scripts/evaluate_ciw_deem_v1.py`
- `local_cache/deem_b3_moe_v1/unsupervised_input_gate_full/`
- `local_cache/deem_b3_moe_v1/unsupervised_input_gate_full_eval/`
