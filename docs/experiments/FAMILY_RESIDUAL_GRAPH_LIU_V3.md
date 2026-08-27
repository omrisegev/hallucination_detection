# Family-residual graph LIU v3 — final topology-policy rebuild

## Status

V3 is specified before any V2/V3 development label was opened. V2 completed
only a label-free topology diagnostic and found that an every-cell 95% largest-
component eligibility threshold was too strict for duplicate-heavy residual
coordinates. No V2 HPO or label report was run.

All representation, residualization, block scaling, union/adaptive graph
constructors, readouts, hyperparameter grid, source/score freeze, nested
leave-dataset-family-out selection, controls, success thresholds, and
retrospective external claim boundaries are inherited unchanged from
`FAMILY_RESIDUAL_GRAPH_LIU_V2.md`. Both specifications are source-hashed by the
fit.

## Health policy amendment

A disconnected graph is mathematically valid for Laplacian regularization: it
produces a block-diagonal Laplacian. Connectivity therefore is not used to
delete candidates before utility HPO.

Hard label-free eligibility requires, in every development cell:

- symmetric, finite, nonnegative graph weights;
- finite positive mean degree;
- zero isolated nodes (`degree_min > 0`, `isolated_fraction = 0`).

Connectivity is instead a separate mechanism-promotion gate inherited from
the earlier frozen topology audit: at least 90% of cells must have largest
component fraction at least 90% and isolated fraction at most 5%. A candidate
may still be evaluated for AUROC if this gate fails, but it cannot support a
claim that a coherent family-residual manifold was identified.

The strict primary remains self-safe positive-distinct union-kNN. Adaptive-k
and union-plus-adaptive selectors remain separately named retrospective
topology-rescue sensitivities. Mutual-kNN remains a fixed negative control.
No quotient/supernode, radius, or diffusion topology is introduced in V3.
