# Direct DUFS graph-semantics audit v1

## Question

Does the sample-neighbourhood graph used by DUFS-LIU place hallucinated
examples near other hallucinated examples, or does it mainly organize samples
by nuisance variables such as answer length, row order, task, or model output
properties?

This is a retrospective mechanism diagnostic.  It is not a new blinded
benchmark and it does not promote or tune a method.

## Separate evaluation lanes

The audit never pools labels or metrics across tasks.

1. **Global answer correctness:** the frozen 24-cell panel.  The graph and
   fixed-stable feature matrix are reconstructed from the canonical bundle and
   frozen DUFS gates.  The target is answer hallucination (`1 - correctness`).
2. **ProcessBench:** eight model/dataset cells.  The frozen mixed-v2 answer
   graph is reconstructed from the original telemetry.  The target is the
   presence of an erroneous reasoning step, not answer correctness.
3. **RAGTruth:** development and test response-level graphs for the original-30
   full-context and evidence-aware hybrid variants.  The target is the official
   response hallucination label.  Development and test stay distinct.

## Frozen graph definition

For sample feature vector `f_i` and the continuous DUFS gate vector `g`, the
graph coordinates are `z_i = g * f_i`.  A self-tuning symmetric 7-nearest-
neighbour affinity graph `W` is constructed with the repository's frozen
implementation.  The exact symmetric normalized Laplacian used by LIU is

`L = I - D^(-1/2) W D^(-1/2)`.

No label enters feature construction, DUFS fitting, graph construction, or
LIU scoring.  Frozen gates are reused where they were persisted.  ProcessBench
gates are deterministically reconstructed and its frozen DUFS-LIU scores must
reproduce before the cell is accepted.

## Direct graph measurements

For a centered target or nuisance encoding `X`, graph smoothness is the
normalized Laplacian Rayleigh quotient

`R(X) = trace(X.T L X) / trace(X.T X)`.

The reported `smoothness_z` is `(mean(R_perm) - R_observed) / sd(R_perm)` over
200 deterministic row permutations.  Positive values mean the variable is
smoother on the graph than chance.  Categorical variables use a centered
one-hot encoding.  Weighted neighbour purity and its permutation z-score are
reported as a second, easier-to-read measure.

Every DUFS graph is compared with an ungated graph over the same features and
the same `k=7`.  Nuisances are lane-specific:

- all lanes: trace/response length and row order;
- ProcessBench: final-answer wrongness;
- RAGTruth: context length, chunk count, task type, source, and generator model.

Individual feature smoothness is also measured so that a graph organized by a
particular input feature is visible rather than silently called a hallucination
manifold.

The value `k=7` is inherited from the frozen DUFS-LIU method, not treated as a
geometric truth.  A topology sensitivity repeats the target-versus-length test
for `k in {3, 5, 7, 10, 15, 25}` under both the ordinary symmetric union-kNN
graph and a stricter mutual-kNN graph.  A manifold interpretation is unstable
when its conclusion changes across these reasonable local graph definitions.

## Relation to LIU

Within each eligible cell, the audit computes the AUROC change from ordinary
IU-PCR (`lambda=0`) to DUFS-LIU (`lambda=0.1`) using the lane's target.  It then
reports the within-lane Spearman association between target smoothness and the
LIU AUROC change.  This is descriptive mechanism evidence, not causal proof.

## Validation structure and interpretation

- Global: leave-dataset-family-out summaries over all 24 cells; the graph
  itself is fitted label-free within each cell.
- ProcessBench: Qwen3-4B is the historical development model and Qwen3-8B is
  the model-held validation panel.
- RAGTruth: development and test remain separate; test is the validation panel.

A lane is called `CONSISTENT_TARGET_ALIGNMENT` only when at least two thirds of
its validation cells have target `smoothness_z > 1.96` and more than half are
more target-aligned than length-aligned.  Otherwise it is
`NO_CONSISTENT_TARGET_MANIFOLD` or `TARGET_ALIGNED_BUT_NUISANCE_DOMINATED`.
These names describe the graph; they do not assert that hallucination itself is
a physical or uniquely identifiable manifold.
