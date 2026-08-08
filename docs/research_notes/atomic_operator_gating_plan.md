# Next research phase: atomic operator gating for IU-PCR

**Date:** 2026-08-07
**Status:** Phase 0 completed and failed; Phase 1 is blocked
**Working name:** AOG-IU-PCR, Atomic Operator Gating for IU-PCR

## Why this is the next question

> **Post-run decision:** The registered Phase-0 proxy failed. Median
> within-cell association with atomic usefulness was -0.312, the practical
> top-proxy atom lost 0.838 AUROC percentage points cell-macro, and only 3 of
> 15 continuation gates passed. Do not implement the Phase-1 learner in this
> plan. See `atomic_operator_premise_audit_conclusion.md` and
> `results/atomic_operator_premise_audit_v2/REPORT.md`.

The frozen 24-cell benchmark rejected sample-specific CA-SpecRaGE and LOCO
micro-views. Balanced atomic views were safer than manual or micro views, but
they tied IU-PCR. Global and permuted alpha controls matched or beat
sample-specific alpha.

The remaining possibility is smaller and simpler:

> Atomic feature-induced roughness operators may contain useful information,
> but their weights should be global rather than sample-specific and should be
> learned directly for the IU-PCR subspace rather than through clustering.

This is inspired by DUFS's continuous differentiable gates. It is not a DUFS
reproduction and must not be presented as one.

## Research basis and claim boundary

The IU-PCR reliability estimate and two-component regression head come from
Tenzer et al., [Crowdsourcing Regression: A Spectral
Approach](https://proceedings.mlr.press/v151/tenzer22a.html), AISTATS 2022. The
idea of learning continuous feature gates with a label-free spectral objective
comes from Lindenbaum et al., [Differentiable Unsupervised Feature Selection
based on a Gated
Laplacian](https://proceedings.neurips.cc/paper/2021/hash/0bc10d8a74dbafbf242e30433e83aa56-Abstract.html),
NeurIPS 2021.

Neither paper defines AOG-IU-PCR. Atomic roughness operators, their global
mixture inside the IU-PCR solve, and the premise audit are proposed project
work. If implemented, the method must be described as DUFS-inspired rather
than attributed to either paper.

## Terms used in this plan

- An **atomic view** is one feature considered by itself.
- An **operator** is the small matrix that describes how a feature graph would
  penalize the two-dimensional IU-PCR solution.
- A **gate** is a nonnegative weight assigned to one operator.
- A **proxy** is a quantity computed without correctness labels and used to
  predict whether an operator is reliable.
- A **bootstrap** rebuilds an estimate after resampling examples. Similar
  estimates across bootstraps indicate reproducibility.
- **Leave-one-family-out** means designing on seven dataset families and
  checking transfer to the eighth. It is stronger than randomly splitting rows
  from the same dataset/model cell.
- **Actuation** means how much an operator actually changes IU-PCR weights or
  sample ranks. A stable operator that changes nothing cannot improve ranking.

## Mathematical object

For one cell, let `F` be the feature-by-sample matrix, `C=FF^T/n`, and let `U`
contain the two leading covariance eigenvectors used by IU-PCR. Feature `j`
creates a one-dimensional sample graph with normalized Laplacian `L_j`:

\[
R_j=F L_jF^\top/n,
\qquad
S_j=\frac{U^\top R_jU}
{\operatorname{tr}(U^\top R_jU)+\epsilon}.
\]

Each `S_j` is a two-by-two, trace-normalized operator. Global nonnegative gates
`q_j`, with `sum_j q_j=1`, define

\[
S(q)=\sum_j q_jS_j.
\]

The final IU-PCR equation would use the corresponding trace-matched aggregate
roughness. The gate is global within a cell: it does not vary by sample.

## The missing premise

The previous experiment proves that operator stability alone is insufficient:
the micro groups were stable and harmful. Before optimizing gates, we must test
a more specific premise:

> A label-free measure of an atomic operator's reproducibility predicts its
> contribution to the IU-PCR ranking after labels are opened.

No new gated model should be built until this premise test passes.

## Phase 0 — sealed atomic-operator premise audit

### 0.1 Units and splits

- Use the existing 24 cells only as retrospective development data.
- Preserve complete dataset/model cells; do not call a random row split an
  independent validation set.
- Use sample bootstraps only to measure estimator stability.
- Use leave-one-family-out analysis to measure transfer across the eight
  dataset families.
- Reserve genuinely new dataset/model families for later confirmation.

### 0.2 Label-free quantities, computed first

For every cell and feature:

1. build `S_j` on the full unlabeled cell;
2. rebuild it on registered sample bootstraps;
3. record operator-distance stability;
4. record the two-dimensional anisotropy direction
   `(S_11-S_22, 2S_12)`;
5. record graph degree, components, edge mass, and effective rank;
6. record similarity to other atomic operators and duplicate density;
7. record the infinitesimal and fixed-strength change that `S_j` causes in the
   IU-PCR weights and ranks, without reading correctness labels.

Freeze and hash these quantities before evaluation.

The exact primary proxy is now registered in
`docs/experiments/FROZEN_ATOMIC_OPERATOR_PREMISE_AUDIT.md`. It combines a
signed, cross-fitted smoothness test with operator reproducibility and
non-trivial but bounded IU-PCR actuation. Operator stability alone is
explicitly not an acceptable primary proxy because the completed micro-view
experiment already falsified that shortcut.

### 0.3 Evaluation quantities, opened second

After the freeze, labels may be used only to measure:

- each atomic operator's AUROC change versus IU-PCR;
- its worst-cell loss and family consistency;
- Spearman association between each label-free stability statistic and atomic
  usefulness;
- whether the top proxy quartile beats the bottom quartile within held-out
  families.

The result is a premise diagnosis, not a trained detector.

### 0.4 Phase-0 continuation gates

Continue only if all are true:

1. the chosen label-free proxy has positive median within-cell association
   with operator usefulness;
2. its family-bootstrap lower bound is above zero;
3. top-proxy operators beat bottom-proxy operators in at least six of eight
   families;
4. the result survives controls for graph edge mass, effective rank, and
   duplicate density;
5. the proxy is not merely a measure of distance from isotropic ridge.

The proxy and thresholds must be fixed before labels are opened. If these gates
fail, stop the AOG line. Do not replace the proxy after inspecting AUROC.

Passing this audit would show that the chosen diagnostic carries useful
selection information on these development families. It would not prove that
the final gating algorithm improves detection or generalizes to new data.

## Phase 1 — label-free global gate learner, only if Phase 0 passes

### 1.1 Objective

Use gates `q=softmax(theta)` and optimize the Phase-0 proxy across unlabeled
development cells and sample bootstraps. The objective should reward a
reproducible operator orientation inside the IU-PCR subspace.

Duplicate control must be conservative. A continuous density prior may stop a
large clone family from receiving extra mass, but it must not reward diversity
for its own sake. Earlier DPP and decorrelation experiments showed that forced
anti-redundancy can be strongly harmful.

### 1.2 Two transfer forms

Test them sequentially, not as an unrestricted model menu:

1. **Shared global gate:** learn one `q` on development families and apply it
   to a held-out family.
2. **Per-cell global gate:** if shared transfer fails but Phase 0 shows a valid
   within-cell proxy, learn one global `q_c` from each cell's unlabeled samples.

Do not add sample-specific weights. That mechanism has already been tested.

### 1.3 Minimal hyperparameters

- one-feature graph neighbours `k`;
- gate temperature;
- duplicate-density bandwidth and maximum prior correction;
- aggregate Laplacian strength `lambda`;
- bootstrap count and sample fraction.

Use a small registered grid. Select configurations with the label-free proxy
and stability only. Report the full path after freezing; do not replace the
headline with the best observed AUROC.

## Required controls

Every candidate must be compared with:

- deployed U-PCR;
- ordinary two-component IU-PCR;
- DUFS-LIU;
- uniform atomic operator average;
- the current CA global-alpha atomic control;
- trace-matched isotropic/projected ridge;
- duplicate-balanced prior weights without learning;
- feature-permuted gates;
- an oracle atomic gate used only as a headroom diagnostic after freezing.

The candidate is not novel if it only reproduces ridge shrinkage.

## Diagnosis metrics

### Gate mechanism

- gate entropy and effective feature count;
- gate cosine/Jensen-Shannon stability across seeds and bootstraps;
- leave-one-cell and leave-one-family weight stability;
- total mass assigned to near-duplicate features;
- difference from uniform, prior, and ridge controls.

### IU-PCR actuation

- projected operator anisotropy and condition number;
- cosine between candidate and IU-PCR weights;
- mean and tail rank displacement;
- score Laplacian energy;
- exact identity at `lambda=0`;
- whether the operator changes the ranking enough to test the hypothesis.

### Evaluation, after freezing

- AUROC and AUPRC for every cell;
- cell-, domain-, and family-macro means;
- paired cell and family bootstrap intervals;
- wins, ties, losses, and worst-cell change;
- Holm-adjusted tests for the small preregistered primary family;
- runtime and memory compared with ordinary IU-PCR.

## Promotion gate

Do not promote the method from the existing 24 development cells. A candidate
may advance to new-data confirmation only if it:

1. improves IU-PCR and DUFS-LIU by at least 0.5 AUROC percentage points;
2. has a positive family-bootstrap lower bound;
3. improves at least 14 of 24 cells;
4. has no loss worse than -2 pp;
5. beats uniform atomic and trace-matched ridge controls;
6. shows stable gates and non-trivial, well-conditioned IU-PCR actuation.

Final confirmation must use dataset/model families not used to design the
feature contract, proxy, gate objective, or hyperparameters.

## Failure interpretation

| observation | conclusion |
|---|---|
| Phase-0 proxy does not predict atomic utility | label-free target identifiability is missing; close AOG before model building |
| proxy predicts utility but learned gates are unstable | optimization/estimation failure; simplify the gate or increase unlabeled sample support |
| gates are stable but tie uniform and ridge | atomic graph information adds no specific value to IU-PCR; close graph regularization |
| development gain disappears on new families | cross-family overfitting; do not add a more flexible gate |
| mean gain hides a tail failure | reject or design a separate safety/abstention mechanism; do not average it away |

## Stop point for the next working session

The next implementation session should build only Phase 0 and its frozen
report. It should stop for human review after the premise plots and tables.
Phase 1 must not start automatically.
