# Global hallucination detection: contextual c-STG router diagnostic v1

## Question and claim boundary

This is the first task-level test of the c-STG idea.  It uses the exact frozen
24-cell completed-trace panel and the six frozen family experts from the family
relevance diagnostic.  The question is whether the already observed
`context_iu_rank` specialization (+2.833 AUROC percentage points of in-sample
quartile-oracle headroom) is accessible to a runnable, held-out router.

This is a retrospective supervised mechanism diagnostic.  It cannot establish
a label-free router, external generalization, or target identification.  c-STG
uses correctness labels in each training fold.  The contexts and expert scores
were produced before this diagnostic and are not refit with labels.

No inference, GPU, download, Drive mutation, or feature recomputation is
allowed.

## Frozen inputs

- Cells: `scripts.inscope_cells.INSCOPE` (24 completed-trace cells).
- Labels: the existing correctness vector for each cell.
- Explanatory inputs: the six `family_experts` frozen by
  `results/family_relevance_real_v1`.
- Primary context: `context_iu_rank` only.
- Extended context: IU rank, trace length, and family disagreement.
- Every row is one source question/trace; no token copies are created.

The explanatory directions are not re-oriented with labels.  The project c-STG
head uses non-negative explanatory weights and context has no direct prediction
path.  It can therefore change family leverage, but not flip a family merely
because a held-out label makes that sign convenient.

## Evaluation

Within every cell, use deterministic stratified five-fold CV.  Models are fit
on four folds and evaluated on the fifth.  AUROC and AUPRC are computed within
each held-out fold and averaged; out-of-fold scores from different fits are
never concatenated.

The c-STG configuration was fixed by the prior synthetic switching-world test:
one 16-unit ReLU layer, Gaussian gate sigma 0.5, sparsity 0.01, Adam learning
rate 0.005, weight decay 1e-4, at most 600 epochs, and seeds 11, 23, 47.  Their
held-out scores are averaged before each fold metric is computed.

Report cell macro and equal-dataset-family macro.  A deterministic 20,000-draw
bootstrap over the eight dataset-family mean deltas provides the primary 95%
interval.

## Models and controls

- `iu_pcr`: unchanged frozen unsupervised score.
- `fixed_expert_cv`: choose one family expert by training-fold AUROC.
- `global_lr`: balanced logistic regression over all six experts.
- `context_only_lr`: balanced logistic regression over the primary context.
- `augmented_lr`: balanced logistic regression over experts plus context.
- `context_core_only_lr` and `augmented_core_lr`: matched linear controls using
  the same three-coordinate extended context as `cstg_core`.
- `quartile_router_cv`: within each training-fold IU-rank quartile, select the
  best family expert and apply that frozen selection to the held-out fold.
- `cstg_iu_rank`: c-STG with IU rank as context (primary runnable router).
- `cstg_core`: c-STG with all three frozen contexts.
- `cstg_iu_rank_permuted`: identical c-STG after a deterministic row
  permutation breaks the context/sample association.
- `cstg_core_permuted`: the matched permutation control for the exploratory
  extended-context gate.

The original in-sample quartile oracle is reproduced only as a premise check;
it is never compared as a deployable held-out score.

## Decision

`GLOBAL_ROUTING_SIGNAL_ACCESSIBLE` requires all of:

1. `cstg_iu_rank - global_lr >= 0.005` equal-family AUROC;
2. its paired equal-family bootstrap lower bound is above zero;
3. it wins in at least five of eight dataset families;
4. it is better at the point estimate than the simple quartile router and
   augmented LR;
5. the permuted-context control does not reproduce its gain.

Otherwise the result is `GLOBAL_ORACLE_NOT_ACCESSIBLE_BY_CSTG`.  A failure means
that the observed specialization remains an oracle fact, not an implementable
router from the tested metadata.  A pass opens a separately frozen label-free
teacher/distillation or intervention study; it does not by itself produce a
deployable hallucination detector.
