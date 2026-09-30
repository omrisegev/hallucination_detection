# Full benchmark pass2a: conditioning, graph and trajectory shortlist

2026-09-07. Continue the registered full benchmark on the same 13,769 rows,
corrected v3 labels, canonical source groups and metric contracts. Keep all
19 original anchors. Add the five routed condition100/equal-graph maps and
12 already-tested trajectory outputs in `fusion_full_shortlist.NEW_ARMS`.
This is not a new parameter search or an untouched test. The token/window
sampling shortlist remains pass2b; historical refits remain later stages.

Reuse the original moment/context feature matrices, normalization, groups,
DUFS gates and dual-bank route. Reconstruct C/v/u with the exact original
fixed-partition Joint fit (seed and 5 starts unchanged), not new clustering.
Before producing condition100 outputs, replay original condition1000 weights,
window/step scores, peaks, GMM decisions and BIC values for zero/real/permuted
graphs. Only compute DUFS gates for equal controls when the old failed Joint
fit did not leave gates. Preserve namespace, graph k7 and permutation seed.
The equal controls retain their original condition1000 implementation;
their regularized condition is below all tested condition caps.

Keep the original `moment_iu` fallback only when that was the original route.
Do not change routing or borrow data when a new readout fails. Keep all21
short unsupported records as failures. No extra model inference or label use
in scoring. Fusion stays central and every fitted quantity is answer-local.

Add mean/GLS for the five existing pairs: IU+Joint graph, IU+Joint zero,
IU+Joint permuted, equal+equal graph, equal+equal permuted. The IU+Joint graph
pair also retains chronological hold and IMM as negative controls. Use the
same dense window support, official-span maximum and GMM decision as before.
IMM filters only the independent chronological fit windows; hold handles
the optional output-only tail. No new gate or threshold is introduced.

Preflight re-executes all110 old pilot cases and compares all17 new maps and
decisions to frozen artifacts. Also exercise short records and long traces.
Freeze the runner, core, imported numerical sources, protocol, evaluation
adapter and source manifests before production. Save per-answer checkpoints,
three CPU workers / one BLAS thread each, and an8-hour scoring invocation cap.
Build evaluation/review/report before launch and call them automatically once
scoring freezes. Observation timeouts do not authorize restarting a live run.

The report keeps full/pilot panels and explicit coverage. PRMB reports pooled
and within-answer AUC, with common-valid paired supports. PB includes all
eight cells and separate Q4/Q8 macros; invalid predictions count as failures.
One set of1000 source-group bootstrap draws jointly weights all3483 groups,
including both scorers and cross-task duplicates. Pair each method with dual
IU and retain the original graph-zero/permutation controls. Also compare the
new graph against its zero/permutation/equal-graph controls and compare new
trajectory combinations against their actual components and matched controls.
Intervals are exploratory and not adjusted for multiple comparisons.
