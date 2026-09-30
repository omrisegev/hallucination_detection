# Full fixed shortlist: condition-100 Joint and graph controls

This is the second stage after `LOCALIZATION_FULL_BENCHMARK_V3.md` pass 1.
It uses the same 13,769 cached model-answer rows and the same corrected labels,
canonical groups and metric definitions. It is development evidence, not an
untouched confirmation set.

The shortlist asks one narrow question: does the newer condition-100 Joint
head, with its graph and equal-graph controls, improve both PRMBench ranking
and ProcessBench first-error/no-error decisions on the full population? The
original answer-only normalization, fixed feature matrix, answer-local Joint
grouping and DUFS graph recipe are retained. Only the registered condition
target and graph/equal control differ. No correctness labels, other answers,
or global fitted quantities enter scoring. A failed Joint fit is reported as a
failure in its native bank; the original route is not silently repaired.

Arms are routed moment/context `cond100`, `cond100_graph010`,
`cond100_graph_perm`, `equal_graph010` and `equal_graph_perm`, with the
original single/dual routing. The original condition-1000 anchors remain in
the same combined evaluation. The graph-permutation control preserves all
graph-node values and destroys only graph topology.

Before the full run, six current110 rows replay the existing condition-100
and equal-graph score arrays exactly. At completion, parent anchor arrays are
replayed bit-for-bit, all new score arrays are hash-frozen, and the same joint
canonical-source bootstrap is applied to the combined methods. Feature search,
HMM/BOCPD and trajectory readouts remain separately registered follow-ups;
they are not silently folded into this condition test.
