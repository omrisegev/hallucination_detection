# Pooled Graph-Roughness Direction V1

**Status:** frozen retrospective development protocol (2026-08-23).

Implementation note: the first score-bank lineage stopped before label access
after a provenance audit requested stricter registry/index verification.  The
scientifically identical integrity-hardened implementation is version V2.

## Question

Can the successful Family-NRM family residual representation improve IU-PCR
through a sample graph, without selecting the covariance eigenvector whose
eigenvalue is closest to one?

This experiment follows the closed Family-residual Graph-LIU V3 search.  Its
development labels and the PRMBench/HLE outcomes have already been inspected.
It is therefore a reconstruction/mechanism study, not prospective
confirmation.  ProcessBench and SemGrad are also retrospective transfer
panels.  A later claim of independent generalization requires a newly sealed
dataset family and, ideally, a new model family.

## Fixed representation and graph

The development roster is the 23 historically eligible original cells used by
Family-NRM: the 24-cell in-scope roster with
`spilled_triviaqa_llama8b` excluded by the already-frozen minimum-positive
eligibility rule.  The fit phase must use this literal roster and must not read
correctness labels to derive it.

For cell (e), ordinary mixed-v2 IU-PCR supplies a standardized baseline
(b_e\) and six provenance-family contributions.  Each contribution is
residualized against (b_e\) and standardized, yielding (R_e\).  The primary
sample graph is the reviewed duplicate-safe symmetric union-kNN graph on
(R_e\), with (k=7\).  Exact row indices are the target-blind tie keys.

Pure residual coordinates, union topology, and (k=7\) are fixed from the
previous V3/topology evidence; they are not searched here.  Because that
evidence was opened before this protocol, the choice is retrospective.

## Label-free graph calibration

Let (L_e\) be the symmetric normalized graph Laplacian.  Define

\[
A_e = R_e^\top L_eR_e/n_e, \qquad
c_e = R_e^\top L_eb_e/n_e.
\]

Both are multiplied by (G_e/\operatorname{tr}(A_e)\), aligned by the named
family registry, and embedded into the global six-family space.  Missing
families are represented by zero rows/columns; there is no pairwise
availability reweighting.  Moments are averaged equally across cells within a
dataset family and then equally across dataset families:

\[
(\bar A,\bar c)=\frac1{|\mathcal G|}\sum_g
  \frac1{|g|}\sum_{e\in g}(A_e,c_e).
\]

The correction direction is the regularized roughness-descent step

\[
d_\lambda=-\lambda(I+\lambda\bar A)^{-1}\bar c.
\]

For a target cell, the score is

\[
s=b+\frac{t}{G}\frac{Rd_\lambda}{\operatorname{sd}(Rd_\lambda)}.
\]

This has a direct quadratic-objective interpretation.  It contains no
eigenvalue-near-one selection and no semantic interpretation of a covariance
eigenvector.  Conditional on the graph and hyperparameters, calibration and
target scoring are label-free and transductive within each cell.

## Hyperparameter selection

Only these two axes are searched:

- λ: `{0.03, 0.1, 0.3, 1, 3, 10, 30, 100}`
- trust factor (t\): `{0.5, 1, 2}`

Primary evaluation is strict nested leave-dataset-family-out (LOFO).  For
outer held family (H\), every inner validation family (J\) is scored by a
direction fitted after excluding both (H\) and (J\).  Cell AUROCs are
averaged within (J\), and inner families receive equal weight.

The primary selector is conservative one-SE:

1. Find the candidate with largest mean inner-family AUROC delta versus IU.
2. Set the eligibility threshold to best mean minus the best candidate's
   standard error across inner families.
3. Prefer eligible candidates whose worst inner-family delta is at least
   -0.005 AUROC; if none exist, use all one-SE-eligible candidates.
4. Deterministic tie-break: smallest `(trust, lambda, -mean_delta)`.

A separately named nested max-mean sensitivity selects the largest inner mean
with tie-break `(trust, lambda)`.  It cannot replace the conservative primary
or determine the frozen external candidate silently.

After development reporting, the all-source frozen candidate is chosen with
the same primary rule applied to single-held-family label-free scores.  Donor
labels therefore select λ/trust; the method is meta-selected, not wholly
unsupervised.  No target labels may be used for direction fitting or scoring.

## Mechanism controls

All controls use the same source omissions and candidate grid.  Report both
capacity-matched nested selection and the primary fold's exact matched
hyperparameters.

1. **Node-permuted graph:** at least 20 deterministic per-cell permutations of
   (W\), preserving graph weights, degree sequence, and spectrum while
   breaking node-to-residual alignment.  Primary attribution compares the real
   graph to the permutation-null distribution at matched hyperparameters.
2. **DUFS graph:** replace the residual graph by the exact historical
   DUFS-coordinate graph; keep (R,b\) in the roughness operator.
3. **Cross-only:** use (d=-\bar c\), removing the
   ((I+\lambda\bar A)^{-1}\) preconditioner.  This is still graph-derived.
4. **Contribution graph:** build (W\) on standardized unresidualized family
   contributions while retaining residual (R\) in the readout.
5. **Equal-cell pooling:** replace hierarchical family balancing by a flat
   mean across cells.
6. **Family-axis permutation:** independently and deterministically relabel
   family axes in each cell, including the scored target, while leaving graph
   topology unchanged.
7. **Identity-L invariant:** verify (R^\top b/n\approx0\); this is a numerical
   invariant, not an HPO arm.

The graph mechanism is supported only if the primary beats IU and exceeds the
matched node-permutation null and DUFS-graph control.  A benefit over
cross-only is additionally required to attribute value to the (A\)
preconditioner rather than only to the graph cross-gradient.

## Development endpoints and gates

Primary endpoint: equal-dataset-family mean AUROC delta versus exact IU-PCR.
Secondary: AUPRC, cell wins, worst family, direction stability, and fraction of
the frozen Family-NRM AUROC gain recovered.

Use 200,000 equal-family paired bootstrap draws (seed 20260822).  Promotion
requires all of:

- lower 95% CI of delta versus IU > 0;
- point delta at least +0.10pp;
- at least 6/8 positive held families;
- worst held-family delta at least -0.50pp;
- point recovery of Family-NRM at least 50%;
- lower 95% CI of `D_0.30 = delta_new - 0.30*delta_NRM` at least zero;
- minimum cosine between outer-fit directions at least 0.80;
- graph-attribution controls as defined above.

Failure of the graph-attribution controls does not erase predictive recovery;
it changes the claim from a graph mechanism to a transferable family-space
direction discovered through a graph-derived objective.

## External evaluation

Freeze the all-source direction, selected configuration, source hashes, and
all target scores before opening each target's labels.  Evaluate ProcessBench
(Llama and Qwen), SemGrad, PRMBench, and HLE against exact IU-PCR and frozen
Family-NRM.  PRMBench uses its source-question clustered bootstrap.  HLE uses
the authenticated interim labels and a class-stratified bootstrap; its low
positive count is a power limitation.  All four panels are explicitly
retrospective known-outcome stress tests.

For every NRM recovery calculation, assert exact row identity and numerical
equality of the IU baseline between the new and frozen NRM artifacts.
