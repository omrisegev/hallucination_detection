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

### Stable reporting convention

The two macro scores are different summaries of the same 24 cells and must
always be named explicitly:

- **cell-macro AUROC `0.7820255514493354`** gives every cell equal weight;
- **equal-dataset-family AUROC `0.7492330051057238`** first averages cells
  within each of the eight dataset families and then gives every family equal
  weight.

Neither value supersedes the other. The equal-family metric is the registered
primary; the cell-macro metric is the easiest direct summary of all 24 cells.

## Supervised linear diagnostic

A separate descriptive diagnostic fitted L2 logistic regression (`C=1`) on
the exact CIW input. Predictions were five-fold out-of-fold with
`StratifiedGroupKFold`; groups never crossed train/test folds and each fold's
standardization was fitted on its donor rows only. This uses correctness labels
and is therefore not a CIW-DEEM arm.

- LR on CIW input: equal-family AUROC `0.7427084969104820`, cell-macro AUROC
  `0.7827757140615349`.
- LR on the pre-CIW D1 input: equal-family AUROC `0.7433574384486479`,
  cell-macro AUROC `0.7834087245664737`.
- CIW minus D1 for the same LR: `-0.0006489415381658592` equal-family and
  `-0.0006330105049388024` cell-macro AUROC.
- Balanced class weights produced essentially the same result
  (`0.7425784754165117` equal-family AUROC).

The strict grouped-OOF LR and transductive B3/CIW scores are not identical
evaluation contracts. The diagnostic supports only the bounded conclusion
that the CIW transform does not generally make the current features more
linearly separable; its small benefit is specific to the nonlinear B3 energy
model.

## IU-PCR and DUFS-LIU transfer diagnostic

The ordinary IU-PCR and frozen DUFS-LIU solvers were refitted label-free on
both D1 and exact CIW inputs over all 24 cells. DUFS settings were inherited,
not tuned: seeds `(11,23,37)`, 80 epochs, `k=7`, and LIU `lambda=0.1`. Scores
and hashes were frozen before the label module was imported.

- IU-PCR on CIW input: cell-macro/equal-family AUROC
  `0.7739522561864316 / 0.7411060399368028`.
- DUFS-LIU on CIW input: cell-macro/equal-family AUROC
  `0.7743883889733002 / 0.7419007565436684`.
- DUFS-LIU minus IU-PCR on CIW: `+0.0004361327868686` cell-macro and
  `+0.0007947166068656` equal-family AUROC. The descriptive family-bootstrap
  interval for the latter crosses zero (`[-0.000214, +0.002072]`).
- DUFS-LIU on pre-CIW D1 input: cell-macro/equal-family AUROC
  `0.7754416158368906 / 0.7428118624691677`.
- CIW minus D1 for DUFS-LIU: `-0.0010532268635904` cell-macro and
  `-0.0009111059254994` equal-family AUROC; 15/24 cells lose beyond the
  `0.0005` tie tolerance.

Thus DUFS contributes a small positive correction to IU-PCR, but the CIW
feature transform loses more than that correction adds. CIW-DEEM itself
remains substantially higher than DUFS-LIU on CIW input (`0.782026` versus
`0.774388` cell-macro AUROC). The CIW input layer is not promoted as a generic
replacement feature contract for IU-PCR or DUFS-LIU.

## Application adapters

The registered score remains completed-response hallucination detection.
Separate frozen adapters now document external response transfer, response-plus-token
localization, causal-prefix early detection, and RAGTruth response detection.
They are not inferred from the 24-cell result and they retain their own task
units and comparators. See `docs/experiments/CIW_DEEM_MULTI_APPLICATION_V1.md`
and `results/ciw_deem_multi_application_v1/REPORT.md`.

Sentence/token/span/claim RAG tasks, EDIS with its incomplete feature roster,
stopping policies, and hidden-state white-box models are not relabeled as CIW.
Their exact compatibility status is recorded in the multi-application report.

## Canonical files

- `spectral_utils/ciw_deem.py`
- `spectral_utils/deem_b3_unsupervised_input_gate.py`
- `configs/ciw_deem_v1.json`
- `scripts/run_ciw_deem_v1.py`
- `scripts/evaluate_ciw_deem_v1.py`
- `scripts/diagnose_ciw_deem_supervised_lr.py`
- `scripts/diagnose_ciw_dufs_liu.py`
- `results/ciw_deem_v1/REPORT.md`
- `results/ciw_deem_v1/RESULT.json`
- `local_cache/deem_b3_moe_v1/unsupervised_input_gate_full/`
- `local_cache/deem_b3_moe_v1/unsupervised_input_gate_full_eval/`
