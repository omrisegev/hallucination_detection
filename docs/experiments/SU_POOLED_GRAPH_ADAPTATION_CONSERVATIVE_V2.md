# SU-aware pooled graph adaptation — conservative V2 report

**Status:** protocol-repair report over the already frozen, label-free V1 fit
bank.  No graph, covariance, contribution residual, or roughness moment is
refitted.

## Why V2 is required

The V1 sidecar's explicit reproduction gate failed: its observed-IU control
returned +0.452pp rather than the +0.251pp result of the canonical pooled
graph-roughness development protocol.  Inspection identified three protocol
differences, all fixed here before V2 scoring:

1. V1 searched graph topology and k; the canonical protocol fixes the residual
   graph to duplicate-safe union kNN with `k=7`.
2. V1 selected the maximum inner mean; the canonical primary uses a one-SE
   selector plus a worst-inner-family guard of -0.005 AUROC.
3. V1 used trust `{0.25, 0.5, 1}`; the canonical grid is `{0.5, 1, 2}`.

V1 remains an explicitly optimistic sensitivity and may not be used as the
headline or to choose an SU adaptation.

## V2 fixed selection

- Graph: union kNN, `k=7` only.
- Calibration lambda: `{0.03, 0.1, 0.3, 1, 3, 10, 30, 100}`.
- Trust: `{0.5, 1, 2}`.
- Cleaning alpha: `{0.25, 0.5, 1}` for the prespecified clean arms.
- Nested leave-dataset-family-out is unchanged.

For each outer fold, find the candidate with largest mean across inner held
families, compute its standard error, retain candidates within one SE, retain
the subset whose worst inner-family delta is at least -0.005 AUROC when that
subset is nonempty, then choose the smallest `(trust, lambda, alpha,
-mean_delta)`.  Alpha is placed after the two trust/regularization quantities
so cleaning cannot be increased merely to win a numerical tie.

The prespecified primary remains `IU rho + cell-specific cross-family sparse
cleaning + equal-group mean operator pooling`.  Direct primary-minus-current
outer-family contrasts and their paired family-bootstrap interval are required;
the primary is not supported merely because both arms separately beat IU.
