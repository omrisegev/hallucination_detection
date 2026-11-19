Quick dev-only CPU checks run on 2026-11-19 (Omri's 'error direction' ideas), saved unchanged from the scratchpad.
Code base: commit 4d57e57 (results/ssl_one_experiment_v1/*.py), Python 3.13.3, HD_ROOT=analysis_v1/code.
pair_direction.py -> PAIR_DIRECTION.json: one mean-difference direction (synthetic minus own parent, 17 z-scored answer features) per V3/V4/V3+V4, cross-fit by gatefold; AUROC, BBE pi-hat on the projection, supervised LR ceiling.
pair_sml.py -> PAIR_SML.json: one direction per error type V1-V5, per-type scores, conditional correlations, SML / L-SML / equal-mean fusion.
Run: python pair_direction.py <code dir> <W> <processbench_gold.json> <out.json>   (same for pair_sml.py)
The console output was not saved separately; the JSON files hold the printed numbers.
