# Math-Shepherd digit-id fix: AIRCC job 288932 (2026-10-19)

This follows `../HANDOFF_MS_DIGIT_FIX_2026-10-19.md`. It is CPU only, with no new inference, and it rebuilds Math-Shepherd's banks from the saved top-50 telemetry. Script: `run_digitfix.sh`. Log: `prmboost_digitfix_288932.out`.

## Checks
- **Tokenizer.** The Math-Shepherd `tokenizer.model` has the same sha256 as Mistral-7B-v0.1's (dadfd56d...). Its digits resolve to 28734, 28740, 28750, 28770, 28781, 28782, 28784, 28787, 28783, 28774. Qwen-PRM-7B, Qwen-PRM-72B, RLHFlow and Qwen3-8B resolve to 15..24. The ids are recorded in `manifests/`.
- **G3 parity.** The v2 builder on the Qwen3-8B cell gives `PARITY 0.0 0` (`manifests/PARITY_qwen3_8b_v2.json`).
- **Guard is a no-op without constant channels.** `guardcheck/` reruns qwen_prm_7b with the guarded L-SML and is identical to `../ABL5_RESULTS_qwen_prm_7b.csv` (49 arms, 0 differences).
- **Mirror deviation from the handoff.** The AIRCC mirror's `spectral_utils/fusion_utils.py` is older than the repo copy and differs by more than this change, so the file was not synced. Only `lsml_continuous_guarded` was appended (`guarded_append.py`), and `gates` is passed only when given. The original is backed up on AIRCC as `results/prm_boost_gen_v1/digitfix_v1/fusion_utils.mirror_backup_before_guard.py`.
- **Constant channels.** `digitfix/ABL5_DIAG_math_shepherd.json` lists no zero-variance channels in N13, N16 or S5. The number of L-SML groups is now 4 / 4 / 4 (before: 4 / 6 / 4).

## Math-Shepherd on PRMBench (within-AUC / PRMScore)
| arm | committed (zero digits) | guard only (old bank) | digit fixed |
|---|---|---|---|
| TELE_N16 | .7072 / 60.14 | .7159 / 60.88 | .7483 / 62.91 |
| MEAN_N16 | .7187 / 60.95 | .7187 / 60.95 | .7402 / 62.35 |
| CHAIN_N16 | .7119 / 60.31 | .7269 / 61.56 | .7783 / 64.37 |
| HALF_N16 | .7640 / 62.52 | .7640 / 62.52 | .7698 / 62.82 |
| EQ_N16 | .7367 / 62.06 | .7367 / 62.06 | .7539 / 63.05 |
| CHAIN_S5 | .7891 / 65.40 | .7928 / 65.64 | .8022 / 65.42 |

With the fix, the paired source-group bootstrap (`digitfix/ABL5_BOOTSTRAP_math_shepherd.csv`) gives:

| contrast | within-AUC | PRMScore |
|---|---|---|
| CHAIN_N16 - HALF_N16 | +.0085 [+.0026, +.0148] | +1.55 [+1.10, +2.02] |
| CHAIN_N16 - EQ_N16 | +.0244 | +1.32 |
| TELE_N16 - MEAN_N16 | +.0080 | +0.56 |
| CHAIN_N16 - PRM_Q80_PROB | +.0939 | +6.64 |
| CHAIN_N16 - PRMPOSLEN | +.0357 | +3.78 |

All of these intervals exclude 0. Only the N16 and S5 arms changed (25 of 49 arms); N13, the PRM rows and the controls are identical. The guard alone only removes the N16 penalty (N16 then equals N13). The gain comes from the real digit channels. This is development data and a single PRM.

## Math-Shepherd on Socratic (`socratic/`, scored locally; gold labels stay local)
Within-AUC / official metric. Non-N16 arms are identical to `../socratic/RESULTS_socratic.csv`.

| arm | before | digit fixed |
|---|---|---|
| TELE_N16 | .5617 / .5492 | .5821 / .5603 |
| EQ_N16 | .5889 / .5681 | .6028 / .5707 |
| CHAIN_N16 | .4677 / .5042 | .4272 / .4978 |
| HALF_N16 | .7354 / .6213 | .7352 / .6210 |
| PRM_native | .7181 / .6439 | unchanged |

On Socratic the fix helps telemetry alone and EQ a little. CHAIN on Math-Shepherd stays below chance on within-AUC, and lower than before. Socratic Math-Shepherd boosting is still a failure, and the digit fix does not repair it.

## Not done here
- HISTORY.md and PROGRESS.md are left to the machine that writes the page, to avoid colliding edits.
- The page has not been rebuilt with these numbers.
