# CIW-DEEM v1 — compact result record

CIW-DEEM is the official Cross-fitted Innovation-Weighted DEEM challenger.
It is a structured, target-free input layer followed by unchanged continuous
B3. It is not a promoted champion.

## Registered CIW-DEEM result

- 24-cell macro AUROC/AUPRC: `0.7820255514493354 / 0.7517170841581265`.
- Equal-dataset-family AUROC/AUPRC:
  `0.7492330051057238 / 0.7791317276773182`.
- Equal-family AUROC delta versus frozen B3: `+0.0007316506044068283`.
- Exact eight-family one-sided sign-flip p-value: `0.13671875`.
- Promotion threshold: `+0.0025`; not met.

Cell-macro gives every cell equal weight. Equal-family first averages within
each dataset family and then weights the eight families equally. Both values
describe the same frozen 24-cell run.

## Closing diagnostics

Supervised group-OOF LR on exact CIW input scores `0.7827757140615349`
cell-macro and `0.7427084969104820` equal-family AUROC. The same LR before CIW
scores `0.7834087245664737 / 0.7433574384486479`; CIW does not improve linear
separability.

IU-PCR on exact CIW input scores `0.7739522561864316 / 0.7411060399368028`
cell-macro/equal-family AUROC. DUFS-LIU scores
`0.7743883889733002 / 0.7419007565436684`, a small incremental
`+0.0004361327868686 / +0.0007947166068656`. Pre-CIW D1 DUFS-LIU is higher at
`0.7754416158368906 / 0.7428118624691677`.

## Decision

The CIW representation is useful only as a small B3-specific challenger under
the evidence available here. It is not a generic replacement input contract
for supervised LR, IU-PCR, or DUFS-LIU. Completed-response scores do not imply
localization or causal early-detection performance.
