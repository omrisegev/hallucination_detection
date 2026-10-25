# PRM boosting: math rigor round 1 (2026-10-24)

Response to the math peer review (`docs/research_notes/math_peer_review_2026-10-24/`). Theory, history audit and
advisor tools are in `docs/research_notes/math_rigor_2026-10-24/`. CPU only, no inference, development data.
Labels are used only for evaluation and for diagnostics marked as such.

## Bottom line
1. **Stage 0 parity: PASS.** The 90 published metric rows of `STATUS_MAIN.csv` (within-AUC and GAP PRMScore,
   15 cells × 6 arms) reproduce exactly, and CHAIN_N16 refits from the exported channels within 2.1e-7 in 15/15
   cells (`stage0/PARITY_SUMMARY.json`).
2. **The improvement condition holds with labels.** The Gaussian plug-in formula (FRAMING_THEORY.md, P2) predicts
   the sign of CHAIN − PRM within-AUC in 13/15 cells, with correlation .979 between predicted and actual
   differences (`stage1/STAGE1_SUMMARY.json`).
3. **Most of the linear headroom is in the fuse-or-not choice.** With labels, Fisher weights on CHAIN's views gain
   −.003 to +.022 over CHAIN on PRMBench, and +.004 to +.036 over the PRM on Socratic for the four non-RLHFlow PRMs
   (`stage1/STAGE1_DIAGNOSTICS.csv`).
4. **The anchor assumption of the label-free identification fails on these views.**
   - Within-class |corr| between the residualized PRM and the telemetry group scores is .13–.35 on average.
   - Within-class structure is .82–1.00 of the off-diagonal covariance norm.
5. **The pre-registered label-free gate failed.**
   - It abstained in 15/15 cells, so it chose the PRM everywhere: primary 4/10, against 6/10 for always-fuse
     (`stage2/GATE_SUMMARY.json`, `NEGATIVE_RESULT.md`).
   - The review's rank-one gate chose fuse everywhere: 6/10, the same as always-fuse.

## Stages and files
| Stage | Script | Output | Labels? |
|---|---|---|---|
| 0 parity | `stage0_parity.py` | `stage0/PARITY_{METRICS,REFIT}.csv`, `PARITY_SUMMARY.json` (input sha256) | evaluation only |
| A synthetic selection | `synthetic/select_gate_estimator.py`, `synthetic/select_gate_level.py` | `synthetic/run_v1/{REGIMES.csv,SUMMARY.json,LEVEL_SUMMARY.json}` | simulated labels only |
| pre-registration | — | `GATE_PREREG.json` + `docs/.../GATE_PREREG.md`, committed in `6b7162f5` before stages 1–2 | — |
| 1 diagnostics | `stage1_diagnostics.py` | `stage1/STAGE1_DIAGNOSTICS.csv`, `STAGE1_SUMMARY.json` | yes, diagnostics only |
| 2 gate | `stage2_gate.py` | `stage2/GATE_RESULTS.csv`, `GATE_SUMMARY.json` | evaluation only |
| post hoc | `posthoc_gate_diagnostics.py` | `stage2/POSTHOC_CUMULANT_SIGNS.csv`, `POSTHOC_NUMBERS.json` | evaluation only |

Shared code:
- `common.py` rebuilds CHAIN_N16's cross-stage inputs and the PERM control exactly as ablation5 does.
  Parity: CHAIN 2.1e-7, CHAINPERM 9.1e-8.
- `spectral_utils/fusion_theory.py` holds the theory helpers. `tests/test_fusion_theory.py` has 9 synthetic
  tests, and all pass.

## Reproduce
```
python3 -m venv ~/.venvs/hd_math_rigor && ~/.venvs/hd_math_rigor/bin/pip install numpy scipy scikit-learn pandas matplotlib pytest
rclone copy gdrive:hallucination_detection/cluster_results/prm_boost_collection_v1/prmbench/ablation5_export_v1 <DATA>/prmbench
rclone copy gdrive:hallucination_detection/cluster_results/prm_boost_collection_v1/external/collect_socratic_v1 <DATA>/socratic
rclone copy gdrive:hallucination_detection/cluster_results/prm_boost_collection_v1/external/collect_h2v_v1 <DATA>/h2v
export COLLECTION_DATA=<DATA>
python results/prm_boost_math_rigor_v1/stage0_parity.py results/prm_boost_math_rigor_v1/stage0
python results/prm_boost_math_rigor_v1/synthetic/select_gate_estimator.py results/prm_boost_math_rigor_v1/synthetic/run_v1
python results/prm_boost_math_rigor_v1/synthetic/select_gate_level.py results/prm_boost_math_rigor_v1/synthetic/run_v1
python results/prm_boost_math_rigor_v1/stage1_diagnostics.py results/prm_boost_math_rigor_v1/stage1
python results/prm_boost_math_rigor_v1/stage2_gate.py results/prm_boost_math_rigor_v1/stage2
python results/prm_boost_math_rigor_v1/posthoc_gate_diagnostics.py results/prm_boost_math_rigor_v1/stage2
python results/prm_boost_math_rigor_v1/stage7_hygiene.py results/prm_boost_math_rigor_v1/stage7   # Step 474, ~25 min
python results/prm_boost_math_rigor_v1/stage8_auto_forced_k.py results/prm_boost_math_rigor_v1/stage8   # Step 475, ~8 min
python results/prm_boost_math_rigor_v1/stage9_auto_f.py results/prm_boost_math_rigor_v1/stage9   # Step 476
```
The scripts import `spectral_utils` without its package `__init__` (no torch needed), through
`tests/lever_imports.py`.

## Caveats
- Development data: PRMBench, Socratic and Hard2Verify have all been inspected before. Nothing here is untouched
  confirmation.
- The gate's outcomes (the 15 signs) were known before the pre-registration. The estimator and the interval level
  were chosen on synthetic data only, and the rule was committed before any real-data gate statistic.
- The Stage 1 Fisher oracle uses labels on the same steps it scores. It is headroom, not a method.
