# Direct DUFS explicit-length-drop ablation v1

## Question

Does the DUFS graph stop organizing samples by answer length when every
explicit `trace_length` feature is removed?

This is a retrospective mechanism ablation over the validation panels of the
direct DUFS graph-semantics audit.  It does not tune a new method.

## Validation panels

- Global: the 21 frozen-24 cells that contain an explicit `trace_length`
  coordinate.
- ProcessBench: the four Qwen3-8B model-held validation cells.
- RAGTruth: the original-30 full-context test graph.

The tasks and labels remain separate.  No cross-task metric is computed.

## Three graph conditions

1. `original`: the frozen DUFS graph with all features.
2. `drop_length_fixed_gates`: delete every explicit length coordinate while
   keeping the other frozen DUFS gates.  This isolates the direct geometric
   contribution of the length coordinate.
3. `drop_length_refit_gates`: delete length and refit DUFS on the remaining
   features using the frozen seeds, epochs, and k=7.  This is the method-level
   ablation.

For RAGTruth hybrid-style names, every column whose base feature is
`trace_length` would be removed.  The registered primary test here uses the
original-30 full-context graph, which contains one such coordinate.

## Measurements

The target and the held-out length variable are evaluated on every graph.  The
held-out length values are used only after graph construction.  Smoothness is
the symmetric-normalized-Laplacian energy reduction relative to 200 row
permutations.  Positive values mean smoother than chance.  Target-versus-length
comparisons use this normalized effect size, not the non-comparable raw z-score.

Ordinary IU-PCR and lambda=0.1 LIU are refit on the no-length matrix.  Their
target AUROC delta is reported separately; labels never enter graph or score
fitting.

## Interpretation gate

- `EXPLICIT_LENGTH_WAS_PRIMARY_CHANNEL`: after refitting, median held-out
  length smoothness is no greater than target smoothness in at least half of
  validation cells.
- `EXPLICIT_LENGTH_NOT_SOLE_NUISANCE_CHANNEL`: held-out length remains smoother
  than target in more than half of validation cells.

The second decision means that other spectral/confidence features encode length
indirectly.  It motivates a separately registered length-residualization or
matched-neighbour graph; it does not authorize calling those transforms
label-free improvements without a new evaluation.
