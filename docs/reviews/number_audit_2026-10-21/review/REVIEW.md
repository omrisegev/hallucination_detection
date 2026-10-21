# Independent review of the audit / error-count / competitor reports (2026-10-21)

Read-only. Scripts in this directory: `a2_count.py`, `a3.py`, `b_recompute.py` (official evaluator port
`spectral_utils.prmbench.prmbench_evaluate`, own GAP implementation), `b_bootstrap.py` (own 707-group paired
bootstrap, 4,000 draws), `show_diag.py`. Outputs: `B_RESULTS.json`, `B_BOOTSTRAP.json`, `gap_dist.txt`.

| claim | verdict | evidence |
|---|---|---|
| A1 think-block fix does not touch PRM-boosting PRMBench | CONFIRMED (code level) | `results/prm_boosting_v1/build_bank.py`: every `cells=='prmbench_qwen3_8b'` row gets `values`/`digit_active` from the PRM's own cell. `ablation5.py:47` keeps only that cell. `Population` (`monday_external_panel_v1/score.py:60-109`) uses only values/names/digit_active/offsets and standardizes per answer, so the base's ProcessBench rows cannot leak in. Templates (`spectral_utils/prm_adapters.py`): Qwen-PRM 7B/72B use the Qwen2.5 chat template, PURE and Math-Shepherd use raw text, RLHFlow uses the Llama-3.1 chat template. None is Qwen3, so there is no think block. The AIRCC telemetry manifests were not available locally. |
| A2 anchor flip fired in 5 of 20 fits, not 3 | CONFIRMED | 20 = 5 PRMs x {N13, N16} x {CHAIN, PRMG}. The flips are Math-Shepherd CHAIN and PRMG on both banks, plus 72B CHAIN N13. That is 5, or 3 distinct fits, because Math-Shepherd N13 and N16 have identical weights (zero digit channels before the fix). `sml README:22` itself lists all five under "3 of 20". |
| A2 an unflagged negative PRM weight at 72B N16 (-0.39) | REFUTED as a PRM weight | The -0.39 belongs to telemetry group g4. The 72B N16 CHAIN PRM weight is +0.095. The raw PRM weight is negative in exactly the 5 flipped fits, and after the flip it is positive in all 20. Negative telemetry-group weights without a flip appear in 5 fits: 72B N13 PRMG -0.28, 72B N16 CHAIN -0.39, 72B N16 PRMG -0.27, RLHFlow N16 CHAIN -0.045, RLHFlow N16 PRMG -0.012. The audit's list misses 72B N13 PRMG. |
| A3 range is now +0.037 to +0.112 | CONFIRMED | CHAIN_N16 minus PRM_Q80_PROB within-AUC: 7B +0.0365, 72B +0.0480, PURE +0.0528, RLHFlow +0.1125, MS +0.0939 after the digit fix (+0.0274 before it). |
| A4 mask1 N16: L-SML 40.02/.7792 vs mean 40.44/.7811 | CONFIRMED | `ABSOLUTE.csv:17,21`. `CONTRASTS.csv` gives within-AUC -0.0019 [-0.0037, -0.000002] and PB -0.42 pp [-1.26, +0.42]. The mean is marginally better on within-AUC; on PB the two tie. |
| B1 parity | CONFIRMED by a different evaluator | Official port: LSML_CONT 65.38088144924220 and ALL_EQ_ORIENTED 65.35297125781070. `>` and `>=` give the same flags. Recomputed within-AUC .7792 matches mask1, not old (.7958) or thinkfix (.7799), so `prmscore_inputs/bank_N16.npz` is the mask1 bank. Qwen-7B CHAIN_N16 68.2509. PRM native 65.4885 uses reward <= 0.5 = error; the PRM run's own stored labels give 65.4568, a tie rule. |
| B2 GAP +2.2 over Q80, below k=2 | CONFIRMED numerically; framing PARTLY | My GAP counts equal theirs on all 6,969 answers. GAP 67.567, Q80 65.381, k=2 67.920. GAP-Q80 = +2.19 [1.73, 2.63]. GAP-k=2 = -0.35 [-0.72, +0.004], so "below k=2" is a point estimate and the CI touches 0. GAP reads only the arm's own scores, no labels. Timing: DECLARATION mtime 09:54:46, RESULTS.csv 09:55:48, run.py edited 09:55:35, and parity.py (labels plus ORACLE counts) ran earlier, at 09:53. The cap was chosen knowing `count_structure_check_v1` (k=2 beats Q80 by +2.5; counts about 2 and independent of length); DECLARATION discloses this. The post hoc cap check shows fragility: with no cap (n-1) the score is 57.80 (-7.6 vs Q80), and with cap 4 it is 68.33 (> k=2). File times cannot exclude an earlier run that was overwritten. Nothing is committed. |
| B3 Q80 count tracks length (~.78-.80), true count does not (.09) | CONFIRMED | Spearman(count, n_steps), noncontrols: Q80 .775 / .785 / .795 (LSML / EQ / BOOST), true .092. With answer tokens instead: .67 vs .02. This was already reported in `count_structure_check_v1/README.md:9`, so it is not a new finding. |
| C1 PRISM 65.50->68.00; FPR 69.34->47.13; TPR 95.36->87.30 | CONFIRMED | arxiv.org/html/2606.09078 Table 2 (v1, 8 Jun 2026) |
| C1 CROP Trace+PRM AUROC .8336, kept 21.8% at alpha .05 | CONFIRMED | arxiv.org/html/2605.30085 Table 6: .8336±.0027, kept 21.8±0.9, risk 5.0±0.6; 5,000 traces, 10 seeds. The abstract page Comments field says "Accepted at EMNLP 2026 (Oral, Main)". |
| C1 ARES numbers; "plain PRM beats it" | Numbers CONFIRMED; "beats" PARTLY | Table 1 matches. Table A7: PRM F1 .749±.016 vs ARES .736±.014, within one fold SD; the paper calls the PRM "a strong baseline". The subset size is not stated. |
| C2 no label-free PRM+telemetry step fusion on PRMBench | Not contradicted (4 searches) | Nearest: CROP Trace+PRM (supervised LR fusion of PRM + trace features, on PRMBench); uPRM 2605.10158 (label-free PRM training, ProcessBench); 2610.18704 (training-free token rewards, no PRMBench). |

Could not verify: the PDFs (WebFetch summarizer only; the last 3-43k characters of each page were not read), the AIRCC telemetry manifests, and the PRISM category table.

## Corrections before reporting to Omri
1. Change "3 of 20" to "5 of 20 PRM-including fits (3 distinct)".
2. Do not call the 72B N16 -0.39 a PRM weight; it is a telemetry-group weight. Add 72B N13 PRMG to the list of negative weights without a flip.
3. Use "+0.037 to +0.112".
4. N16 mask1: "L-SML does not beat the plain mean (within -0.002 [-0.004, -0.000]; PB tie)". Give the CI.
5. GAP: "beats Q80 by +2.2 [1.7, 2.6] (140/140 telemetry scorer x bank cases, 4/4 boosting arms), does not beat constant k=2 (-0.35 [-0.72, +0.00])". Also disclose that the cap was informed by the known count structure, that removing the cap costs 7.6 points, and that the data are development data, uncommitted.
6. B3: cite `count_structure_check_v1` as the earlier source.
7. Name the PRM-native tie rule (65.49 vs 65.46).
8. ARES: "PRM slightly higher, within fold SD", not "beats".
9. Keep the novelty claim as "as far as searched", and name CROP Trace+PRM as the supervised analogue.
