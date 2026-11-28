# realized_z / chosen_surprisal / realized_drv stability, PB vs PRMBench (2026-11-28)

Diagnosis only. No new arms. CPU, stored raw banks.

## Data
- Model: standalone Qwen2.5-Math-7B-Instruct (base model of Qwen-PRM-7B), both benchmarks.
- PB: `hd_collection_data/c4u_r1_inputs/crop_ares_all/full_sa_qwen_prm_processbench/bank_v3/FEATURES.npz` + `crop_ares_v1/gold/processbench.json`.
- PRMBench: `prm_standalone_banks/bank_qwen_prm_7b/FEATURES.npz` (cell `prmbench_qwen3_8b`, 6,969 answers) + `results/advisor_completion_2026-10-11/inputs/prmbench_meta.json`.
- Step token counts and alternative readouts: `readout_steps_sa_v1/{processbench,prmbench}__qwen_prm_7b.npz` (row order asserted equal to the banks).
- Files are read from `/mnt/user-data/uploads/hd_collection_data/` (staged copy). Change `U` in `load.py` / `fast.py` for a local run.

## Metrics
- within-answer AUC: mean over answers with both classes, on the official pool (PB: steps up to and including the first error; PRMBench: all steps, control answers removed).
- position-matched first-error AUC (`pm.py`): for each step index j < 12, first-error steps at j vs steps at j that are still before the first error; mean weighted by positives.

## Results (`auc_table.json`, `fig1_feature_auc.png`, `fig2_position_trend.png`)
| feature | within PB | within PRMB | PM PB | PM PRMB |
|---|---|---|---|---|
| chosen_surprisal | .560 | .725 | .750 | .858 |
| realized_z | .523 | .595 | .524 | .700 |
| realized_drv | .501 | .719 | .690 | .836 |
| q15_H1 (reference) | .744 | .738 | .802 | .823 |
| POSITION (control) | 1.000 | .662 | | |

## Findings
1. On PB the within-answer pool puts the first error at the last kept step. Position alone gives AUC 1.000.
2. On PB correct steps, chosen_surprisal and realized_drv are high at steps 0-1 and fall (Spearman with relative position -.41, -.39; PRMBench -.04, -.12). With metric 1 this trend pushes their AUC to .50-.56.
3. PRMBench with the PB prefix pool gives higher AUC (.79/.68/.78). The pool rule alone does not cause the drop.
4. With the position-matched metric, chosen_surprisal and realized_drv recover to .75/.69. realized_z stays at .52.
5. realized_z on PB depends on the generator: Qwen2.5-Math family (same family as the scorer) .41, Llama .58, other Qwen .54. Hypothesis (not tested): excess surprisal (surprisal minus entropy) detects text that the scorer would not write. PRMBench errors are inserted by another model; PB errors are written by the generator itself.
6. Readout: chosen_surprisal median readout gives within .803/.671 and PM .727/.764 (smallest gap). Top10 is the bank readout.
7. Label-free OLS removal of relative position: within chosen_surprisal .638/.702, realized_drv .617/.728, realized_z .589/.629.
