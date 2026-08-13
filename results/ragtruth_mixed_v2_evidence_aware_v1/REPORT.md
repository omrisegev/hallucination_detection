# RAGTruth original-30 evidence-aware experiment

**Status:** exploratory comparison, not a blinded confirmation.

## Result

The highest pooled exploratory test AUROC is 0.8013 for Original-30 noctx / DUFS. After giving QA and Data-to-Text equal weight, the highest task-macro AUROC is 0.7225 for GASP-top50. This difference is important because pooled AUROC can reward task identification rather than within-task grounding. No test result was used to revise a variant.

The largest evidence-versus-full-only change is
**+0.0385** with a
source-grouped 95% interval
**[+0.0262,
+0.0511]**.

With equal weight for QA and Data-to-Text, the largest pure Original-30
evidence gain is
**+0.1163** with
interval **[+0.0795,
+0.1544]**.

The largest DUFS-minus-IU change is
**+0.0028** with interval
**[+0.0017,
+0.0042]**.

The largest task-macro DUFS-minus-IU change is
**+0.0065** with
interval **[+0.0047,
+0.0085]**.

All 30 original features were available in every full, no-context and observed
LOO condition. No missing feature or chunk was imputed. The report separates
QA and Data-to-Text and includes condition permutations, graph permutations,
gate stability, fusion weights, confounds and chunk/task diagnostics.

See [`REPORT.html`](REPORT.html) for the visual report and [`METHODS.md`](METHODS.md)
for the mathematical definition.

## Claim boundary

RAGTruth labels were opened before this comparison. No labels enter fitting,
but a final method claim requires confirmation on a new benchmark or scorer.
