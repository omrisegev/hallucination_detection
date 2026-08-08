# Frozen atomic-operator premise audit

## Goal

This is Phase 0 of the proposed AOG-IU-PCR direction. It does not train an
atomic gate. It tests whether a fixed quantity computed without correctness
labels predicts which one-feature Laplacian improves IU-PCR.

The experiment must stop after this diagnosis. A global gate learner may be
built only if every registered continuation gate passes.

## Research question

For feature `j`, collapse samples with exactly equal feature values into one
quotient node. Build a graph on the sorted unique values, its normalized
Laplacian `L_j`, and the roughness operator

\[
R_j=F L_jF^\top/n.
\]

Here `F` is understood after count-weighted projection onto the quotient
nodes. Ordinary sparse k-NN breaks distance ties using sample order. The
quotient policy is invariant to row permutations and does not pretend that one
feature distinguishes tied samples. Features with fewer than three unique
values are invalid and cannot be selected. Each cell records unique count,
maximum tie size, tied-sample fraction, and effective `k`.

Let `U` contain the leading two IU-PCR covariance eigenvectors. Define the
trace-normalized projected operator

\[
S_j=\frac{U^\top R_jU}{\operatorname{tr}(U^\top R_jU)}.
\]

The question is not whether `S_j` is stable. The previous micro-view result
already showed that stable geometry can be harmful. The stronger question is:

> Does a reproducible atomic operator that agrees with an independently built
> IU-PCR pseudo-target tend to improve the final correctness ranking?

## Primary label-free proxy

### Cross-fitted alignment

For feature `j`, remove `j` and every feature whose absolute correlation with
it is at least 0.95. Fit ordinary two-component IU-PCR on the remaining
features and call its score `s_{-j}`. If fewer than three features remain, only
`j` is removed.

Measure its normalized energy on `j`'s graph:

\[
e_j=\frac{s_{-j}^\top L_js_{-j}}{s_{-j}^\top s_{-j}+\epsilon}.
\]

Using 16 deterministic sample permutations, compute the median null energy
`e_j^perm`. The signed alignment is

\[
a_j=\frac{\operatorname{median}(e_j^{perm})-e_j}
{\lvert\operatorname{median}(e_j^{perm})\rvert+\epsilon}.
\]

Positive `a_j` means the feature graph makes a score built without that
feature smoother than a node-permuted graph does. This is a pseudo-target
diagnostic, not pseudo-label training.

### Reproducibility and actuation

Use 40 deterministic 80% subsamples, capped at 1,500 examples. The original
full-cell standardization is kept fixed because the deployed method fits the
whole unlabeled cell.

For every subsample, rebuild `S_j`, `a_j`, ordinary IU-PCR, and the
lambda-one atomic-IU score. Embed `S_j` back into feature space as
`M_j=US_jU^T`. Define:

- operator reproducibility `r_j` as one minus the median Frobenius distance
  from the full `M_j`, divided by `sqrt(2)` and clipped to `[0,1]`;
- rank-change reproducibility `c_j` as the norm of the mean bootstrap rank
  change divided by its root-mean-square norm;
- actuation `d_j` as the median mean absolute rank change, divided by sample
  count;
- bounded relative actuation `u_j=min(d_j/median_l(d_l),1)` within the cell.

The registered primary proxy is

\[
p_j=\operatorname{median}_b(a_{jb})
\sqrt{r_jc_j}\,u_j.
\]

The sign comes only from cross-fitted alignment. Stability and actuation can
attenuate the proxy but cannot turn a negative alignment positive.

## Frozen parameters

| parameter | primary value | sensitivity values | reason |
|---|---:|---:|---|
| unique-value graph neighbours `k` | 15 | 7, 30 | preceding atomic scale with an order-invariant tie policy |
| Laplacian strength `lambda` | 1 | 0.3, 3 | trace matching makes 1 one covariance-scale unit |
| duplicate threshold | 0.95 | recorded diagnostics at 0.90 and 0.99 | prevent a near clone from leaking the held feature into `s_{-j}` |
| stability subsamples | 40 | convergence at 4, 8, 12, 20, 30, 40 | enough to diagnose convergence without neural fitting |
| subsample fraction | 0.80 | none | inherited from the fusion-aware stability study |
| sample cap | 1,500 | none | prevents large QA cells from dominating runtime |
| permutation nulls | 16 | none | deterministic label-free alignment reference |

Sensitivity paths are reported after freezing. No path may replace the primary
result after labels are opened.

## Leakage barrier

### Stage 1: fit and freeze

`scripts/atomic_operator_premise_fit.py` first creates a physically
label-stripped input bundle containing only feature matrices, feature names,
and frozen orientation metadata. It has no label argument and never reads
`__labels`. It writes:

- IU-PCR, projected-ridge, uniform-atomic, and every atomic score for the fixed
  `(k, lambda)` sensitivity grid;
- every label-free proxy component and graph diagnostic;
- proxy convergence checkpoints;
- a SHA-256 manifest for every score and diagnostic file.

### Stage 2: evaluate

`scripts/atomic_operator_premise_report.py` verifies source and artifact hashes
and creates an immutable score-freeze manifest before reading labels. It then
computes atomic usefulness as AUROC change versus IU-PCR.

The 24 cells are retrospective development data, not external confirmation.

## Primary evaluation

For `k=15` and `lambda=1`, report:

1. Spearman association between `p_j` and atomic usefulness within each cell;
2. the median cell association and an eight-family bootstrap interval;
3. top-proxy minus bottom-proxy quartile usefulness for every family;
4. partial rank association after controlling for graph edge mass, projected
   effective rank, duplicate density, and distance from isotropic ridge;
5. the AUROC of the highest-proxy atomic feature as a diagnostic, not as a
   promoted detector;
6. oracle-best atomic headroom, computed only after the freeze.

Inference respects the nesting. Alongside descriptive family-bootstrap
intervals, the report uses within-cell feature-identity permutations, a
nuisance-adjusted Freedman--Lane residual permutation, and exact sign-flip
tests over eight family estimates. Undefined associations fail closed. Top and
bottom quartiles include every threshold tie; overlapping tied groups are
undefined rather than ordered by array position.

Secondary proxies are alignment alone, operator stability alone,
stability-times-actuation without alignment, actuation alone, and anisotropy.
They are exploratory and cannot replace the primary proxy.

## Continuation gates

All must pass:

1. median within-cell Spearman association is positive;
2. the family-bootstrap lower bound for association is above zero;
3. top-proxy operators beat bottom-proxy operators in at least six of eight
   families;
4. the family-bootstrap lower bound of nuisance-adjusted partial association
   is above zero;
5. the proxy is not almost perfectly explained by distance from ridge: median
   absolute within-cell Spearman is below 0.8.
6. the registered feature-identity, Freedman--Lane, and family sign-flip tests
   pass their one-sided 0.05 thresholds;
7. the label-free top-proxy atom has a positive equal-family AUROC interval,
   improves at least 14 of 24 cells, and has no loss below -2pp;
8. the label-only oracle atom has a positive equal-family interval, showing
   that useful atomic headroom exists at all.

Failure means do not build AOG-IU-PCR from this proxy. The complete proxy is
recomputed for every registered `(k, lambda)` pair, but only `k=15, lambda=1`
is primary. A sensitivity cell that looks better after evaluation is a
hypothesis for a new registered experiment, not a rescue. The 0.90/0.99
duplicate thresholds change only the full-alignment component and appear in a
separate descriptive table.

The fit is label-free conditional on `fixed_stable_v1`. That feature contract
was developed historically using these same 24 cells. This is not a fully
unsupervised external confirmation experiment.

## Required outputs

- `REPORT.md` in simple English;
- atomic, cell, family, proxy, control, and sensitivity CSV tables;
- machine-readable continuation gates;
- proxy-versus-utility scatter;
- per-cell association plot;
- family top-minus-bottom plot;
- bootstrap convergence plot;
- `(k, lambda)` sensitivity heatmap;
- top-proxy, oracle, uniform, and projected-ridge headroom plot.

## Run commands

```bash
python scripts/test_atomic_operator_premise.py
python scripts/atomic_operator_premise_fit.py --resume
python scripts/atomic_operator_premise_report.py
```

The default output is `results/atomic_operator_premise_audit_v2/`. The `v2`
suffix distinguishes this preregistered quotient-graph protocol from the
earlier, incompatible draft that used ordinary sample-level one-dimensional
graphs.
