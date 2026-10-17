# OG-SML Agent B — T0 report

Terminal status: **T0_FALSIFIED_STOP_BEFORE_STEPS_0_6**

## Result

The preregistered retrospective prediction is **FALSIFIED**.
The C-v2 ledger contains 18/18 single hard partitions and
0/18 overlapping selected families; provenance was a reference and
was not part of the fitted structure.

- Prior joint-gate passes: 3; admissible among them: 0.
- Prior joint-gate failures: 15; admissible among them: 6.
- Minimum selection-J among prior passes: 0.
- Maximum selection-J among prior failures: 0.0770585588.
- Strict J separation: False.
- C-v2 multistart PASS: 15/18; profiled-Jacobian PASS: 18/18; regularization-sensitivity PASS: 3/18.
- In this ledger `primary_gate_pass` equals the regularization-sensitivity verdict in all 18 lanes; it is not a pure optimizer-stability outcome.

Because the stop rule failed, Agent B does not implement Steps 0--6 or run
T1--T3 under this proposal.  This result does not show that graph-identifiable
fusion is impossible; it shows that Theorems 1--2, applied to the structures C-v2
actually fitted, do not explain its observed primary-gate outcomes.  It does not
falsify Theorems 1--2 themselves.

## Lane-level evidence

| Cell | Lane | K | Group sizes | Prior gate | Admissible | |H| | H components | H bipartite | weighted lambda2(H) | selection J | Blockers |
|---|---|---:|---|---|---|---:|---:|---|---:|---:|---|
| processbench_gsm8k_qwen3_4b | v2_active28 | 4 | 14,8,3,3 | FAIL | YES | 253 | 1 | NO | 0.0204873 | 0.0204873 | none |
| processbench_gsm8k_qwen3_4b | h2_24 | 3 | 14,8,2 | FAIL | NO | 156 | 1 | NO | 0.0227444 | 0 | GROUP_2_EXCLUSIVE_SUPPORT_LT3, GROUP_2_EXCLUSIVE_BIPARTITE |
| processbench_math_qwen3_4b | v2_active28 | 2 | 6,22 | PASS | NO | 132 | 1 | YES | 0.00869843 | 0 | FREE_GRAPH_BIPARTITE |
| processbench_math_qwen3_4b | h2_24 | 6 | 4,5,7,4,2,2 | FAIL | NO | 231 | 1 | NO | 0.0812579 | 0 | GROUP_4_EXCLUSIVE_SUPPORT_LT3, GROUP_4_EXCLUSIVE_BIPARTITE, GROUP_5_EXCLUSIVE_SUPPORT_LT3, GROUP_5_EXCLUSIVE_BIPARTITE |
| processbench_olympiadbench_qwen3_4b | v2_active28 | 2 | 6,22 | PASS | NO | 132 | 1 | YES | 0.00389684 | 0 | FREE_GRAPH_BIPARTITE |
| processbench_olympiadbench_qwen3_4b | h2_24 | 2 | 22,2 | FAIL | NO | 44 | 1 | YES | 0.00157743 | 0 | FREE_GRAPH_BIPARTITE, GROUP_1_EXCLUSIVE_SUPPORT_LT3, GROUP_1_EXCLUSIVE_BIPARTITE |
| processbench_omnimath_qwen3_4b | v2_active28 | 6 | 3,6,3,8,4,4 | FAIL | YES | 317 | 1 | NO | 0.0312513 | 0.0312513 | none |
| processbench_omnimath_qwen3_4b | h2_24 | 2 | 22,2 | FAIL | NO | 44 | 1 | YES | 0.00130005 | 0 | FREE_GRAPH_BIPARTITE, GROUP_1_EXCLUSIVE_SUPPORT_LT3, GROUP_1_EXCLUSIVE_BIPARTITE |
| processbench_gsm8k_qwen3_8b | v2_active28 | 3 | 2,18,8 | PASS | NO | 196 | 1 | NO | 0.0100456 | 0 | GROUP_0_EXCLUSIVE_SUPPORT_LT3, GROUP_0_EXCLUSIVE_BIPARTITE |
| processbench_gsm8k_qwen3_8b | h2_24 | 2 | 16,8 | FAIL | NO | 128 | 1 | YES | 0.0528825 | 0 | FREE_GRAPH_BIPARTITE |
| processbench_math_qwen3_8b | v2_active28 | 4 | 8,3,3,14 | FAIL | YES | 253 | 1 | NO | 0.0770586 | 0.0770586 | none |
| processbench_math_qwen3_8b | h2_24 | 2 | 16,8 | FAIL | NO | 128 | 1 | YES | 0.00848937 | 0 | FREE_GRAPH_BIPARTITE |
| processbench_olympiadbench_qwen3_8b | v2_active28 | 4 | 3,7,15,3 | FAIL | YES | 246 | 1 | NO | 0.00247402 | 0.00247402 | none |
| processbench_olympiadbench_qwen3_8b | h2_24 | 6 | 7,5,2,2,3,5 | FAIL | NO | 230 | 1 | NO | 0.0204043 | 0 | GROUP_2_EXCLUSIVE_SUPPORT_LT3, GROUP_2_EXCLUSIVE_BIPARTITE, GROUP_3_EXCLUSIVE_SUPPORT_LT3, GROUP_3_EXCLUSIVE_BIPARTITE |
| processbench_omnimath_qwen3_8b | v2_active28 | 4 | 8,3,3,14 | FAIL | YES | 253 | 1 | NO | 0.00570109 | 0.00570109 | none |
| processbench_omnimath_qwen3_8b | h2_24 | 2 | 16,8 | FAIL | NO | 128 | 1 | YES | 0.0231751 | 0 | FREE_GRAPH_BIPARTITE |
| prmbench_response_qwen3_8b | v2_active28 | 4 | 13,3,9,3 | FAIL | YES | 258 | 1 | NO | 0.046837 | 0.046837 | none |
| prmbench_response_qwen3_8b | h2_24 | 4 | 8,2,5,9 | FAIL | NO | 201 | 1 | NO | 0.00641301 | 0 | GROUP_1_EXCLUSIVE_SUPPORT_LT3, GROUP_1_EXCLUSIVE_BIPARTITE |

## Firewall

`labels_seen=false`, `targets_loaded=false`, `outcome_metrics_computed=false`,
and `fused_score_arrays_created=false`.  No localization outcome was evaluated.
