# Graph-local IU full result: 2026-09-13

Job255753 completed with RESULT_REVIEW PASS. All13,769 answers retained.
Eleven fetched files match remote SHA256; PB macro recomputed from eight cells.

| Method | PB % | PRMB within AUC | PRMScore |
|---|---:|---:|---:|
| RBM12 Logit reference | 36.271 | 0.74520 | 0.62222 |
| Native answer-local Shrinkage IU on RBM12 bank | 20.273 | 0.69059 | 0.56861 |
| Sliding-window IU | 35.547 | 0.74182 | 0.58590 |
| Graph-local IU | 21.165 | 0.70115 | 0.57174 |
| Permuted-graph IU control | 36.268 | 0.74945 | 0.58857 |

Primary graph-local minus sliding-window: PB -14.382pp,
98.333% source-group CI [-16.824,-11.949]pp; within AUC -0.04067,
CI [-0.04371,-0.03776]. The implemented affinity weighting is negative relative
to the uniform temporal neighborhood and shuffled graph. No missing-score
or optimizer nonconvergence exclusions explain this comparison.

Against RBM12, graph-local gains272 former misses and loses937 successes:
936 early,1 late,0 gate,0 invalid. This shows an early-selection failure,
not its definitive statistical cause. Graph weighting also changes effective
weight concentration and therefore local-prior shrinkage; the contrast does
not isolate graph topology alone. Do not infer that every token graph fails.

The permutation control is a diagnostic, not a promoted learned graph method.
The sliding window is useful relative to its weak native IU baseline, but does
not establish an overall advantage over RBM12. GraphTV separately regularizes
coefficient differences; this result does not decide that pending experiment.
No new method was launched. Raw SQLite/NPZ remain on AIRCC.
