# External LR and PRM-7B rows (2026-10-12)

`RESULTS_relayed.csv`: values copied at 3 decimals from `results/external_lr_prm_v1/RESULTS.csv` on Omri's local checkout
(branch claude/project-thread-9ccmlt, commit adc32b5f, pushed 2026-10-12). Input of `build_v15.py` (5th argument).

- LR: trained on the 6,211 noncontrol PRMBench answers (N16 per-answer-z features, balanced LR, C by group-fold log-loss on PRMBench: 8B C=10, 4B C=100), applied to the targets without target labels.
- PRM: Qwen2.5-Math-PRM-7B on AIRCC jobs 282806 (Socratic) and 282807 (Hard2Verify), same answers.json and step splits as our methods (hash-checked).
- Check: the plain-average N16 / ANSWER_Z_Q80 rows were reproduced to < 1e-9.
- Hard2Verify: 200 answers, no CIs; 19 answers are longer than 4,096 tokens (their PRM AUC 0.636 vs 0.633 for the rest).

Checked against the pushed RESULTS.csv: all 8 rows match within 0.0005 (rounding).
