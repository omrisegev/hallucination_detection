# CHAIN on Socratic-PRMBench (2026-10-18): it does not transfer either

CHAIN was the best label-free PRM boosting arm on PRMBench: Qwen-7B 68.25 with N16, 67.82 with N13
(`../../prm_boosting_bank_sweep_v1/README.md`). On Socratic it loses to the native PRM on 4 of 5 PRMs.
It wins only on RLHFlow, the weakest PRM, which is the same pattern as the frozen arms (`../README.md`).

## Status of this arm

CHAIN was defined on PRMBench on 2026-10-18 at 18:59 IDT, before the Socratic outcome was seen at 21:23
IDT. It is **not** in the frozen Socratic arm list, so this is a post-registration arm. It cannot change
the pre-registered verdict, which was already a fail.

## How it was run (local CPU, no labels during scoring)

- **Inputs:** `full_<prm>_socratic/bank_v2/{FEATURES,REWARDS}.npz` from AIRCC job 288317, copied to Drive
  under `gdrive:hallucination_detection/cluster_results/prm_boost_external_v1/`. The 5 `MANIFEST.json`
  files are byte-identical to the ones in git.
- **Scoring:** `score_chain_external.py` uses the same views, orientation and `lsml_continuous` as
  `../score_external.py`, plus the CHAIN, TELE and PRMG code of `lsml_bank_sweep.py` for N13 and N16.
  - The repo copy of `evaluate_amended.py` runs unchanged. Only its `__file__` is set to the AIRCC mirror
    location, because the module resolves its root from that path.
  - The scores are written to the scratchpad and are not committed, the same as the git-ignored
    `scores_socratic/*.npz`.
- **Evaluation:** `../eval_external.py socratic <scores> <gold dir> eval/`.
  - Gold labels are `results/remote_claims_replay_2026-10-15/inputs/socratic_gold.json`, 2,995 answers.
  - Paired bootstrap over answers, 1,000 draws.
- **Parity: PASS.** The recomputed LSML_N16_PRMG and PRM_native match `../eval_socratic/RESULTS_socratic.csv`
  on all 5 PRMs, for example Qwen-7B 0.6514 / 0.7525 and 0.6809 / 0.8236.

## Result (official PRMScore and within-AUC; delta vs PRM native, 95% CI)

| PRM | PRM native | CHAIN_N16 (3 digit ch.) | Δ PRMScore | CHAIN_N13 (digit-free) | Δ PRMScore | Δ within-AUC, N16 / N13 |
|---|---|---|---|---|---|---|
| Qwen-7B | 68.09 / 0.824 | 65.52 / 0.776 | −2.57 [−3.33, −1.80] | 62.73 / 0.721 | −5.36 [−6.19, −4.53] | −0.047 / −0.103 |
| Qwen-72B | 71.96 / 0.814 | 56.42 / 0.613 | −15.53 [−16.40, −14.69] | 49.80 / 0.460 | −22.16 [−22.89, −21.40] | −0.201 / −0.354 |
| PURE | 63.71 / 0.796 | 64.47 / 0.698 | +0.76 [−0.27, +1.75] | 63.44 / 0.674 | −0.27 [−1.37, +0.78] | −0.098 / −0.122 |
| RLHFlow | 48.12 / 0.549 | 56.97 / 0.590 | +8.85 [+7.87, +9.68] | 54.51 / 0.570 | +6.39 [+5.61, +7.17] | +0.041 / +0.021 |
| MathShepherd | 64.39 / 0.718 | 50.42 / 0.468 | −13.98 [−14.83, −13.11] | 50.42 / 0.468 | −13.98 [−14.83, −13.11] | −0.250 / −0.250 |

All numbers are from `eval/RESULTS_socratic.csv` and `eval/BOOTSTRAP_socratic.csv`, N = 2,995 answers per PRM.
On MathShepherd, N13 and N16 give identical numbers, as on PRMBench: the digit group gets zero weight.

## Reading

- **Telemetry dominates the fit.** On Socratic the fused CHAIN score has a correlation of only
  0.07–0.61 with the PRM (`score_chain_external.py` stdout).
  - On Qwen-72B, CHAIN_N13 reaches a within-AUC of 0.460, which is below chance.
  - The answer-local L-SML picks a direction that the telemetry structure dominates, and on Socratic that
    direction does not track step errors.
- **CHAIN gets closer to the PRM than PRMG does on Qwen-7B and PURE (N16), but the gain stops there.**
  Standardizing the group scores raises the PRM's share in the fit; it does not reverse the loss.
- **This is the same failure as the frozen arms.** The likely cause is the telemetry and fusion, not the
  grouping variant.
