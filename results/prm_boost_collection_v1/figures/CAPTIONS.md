# Figure captions

### prism_fig5b_fpr_fnr.png
PRMBench, 6,211 non-control answers (83,371 steps per PRM). FPR-FNR trade-off of one global threshold swept over the PRM risk (PRM trained by its authors) and over the answer-z score of CHAIN_N16 / CHAIN_N13 (label-free fusion) and LSML_CONT (label-selected channels). Points: native rule (PRM), the label-free Q80 and GAP rules. Star: PRISM (re-trained PRM, PRM800K labels), Qwen-7B only. Positive = valid step, as in PRISM. GAP is a per-answer count, not a global threshold, so its point need not lie on the curve.
Data: `figures/data/prism_fig5b_fpr_fnr.csv`.

### prism_fig7_score_histograms.png
PRMBench, Qwen2.5-Math-PRM-7B, 6,211 non-control answers. Score densities on error vs correct steps: PRM reward (trained PRM), CHAIN_N16 answer-z (label-free fusion), LSML_CONT answer-z (label-selected channels). Data for all 5 PRMs in the CSV.
Data: `figures/data/prism_fig7_score_histograms.csv`.

### prism_fig4_pairwise_by_margin.png
PRMBench, within-answer pairs (one error step, one valid step of the same answer). Accuracy = share of pairs where the error step scores higher, by decile of the answer-z score margin. No decision rule. Access: PRM trained by its authors; CHAIN label-free; LSML_CONT label-selected.
Data: `figures/data/prism_fig4_pairwise_by_margin.csv`.

### crop_fig3_kept_vs_alpha.png
PRMBench, CROP protocol: first 5,000 non-control traces (Hub order), 10 seeds, 60/20/20 splits; lambda calibrated with labels on the 20% split for EVERY proxy (CROP access). Prefix kept vs alpha. Direct PRM: trained PRM, fixed 101-point grid (dotted) and train-quantile grid; CHAIN label-free fusion; LSML_CONT label-selected; Telemetry13+PRM LR supervised. Qwen-7B is CROP's own PRM; the other 4 PRMs are our extension.
Data: `figures/data/crop_fig3_kept_vs_alpha.csv`.

### crop_fig4_auroc_vs_kept.png
PRMBench CROP protocol (5,000 traces, 10 seeds, alpha .05). One point per proxy x PRM (marker = PRM, colour = proxy). Stars: CROP Table 6 (Qwen-7B). Thresholds are label-calibrated for every proxy; LR proxies are supervised; our fused scores are label-free.
Data: `figures/data/crop_fig4_auroc_vs_kept.csv`.

### crop_fig5_by_length.png
PRMBench CROP protocol, alpha .05, test traces pooled over 10 seeds; retention (mean kept fraction) by trace length (steps). Risk per bin in the CSV. Direct PRM trained; CHAIN label-free; LSML_CONT label-selected; LR supervised.
Data: `figures/data/crop_fig5_by_length.csv`.

### crop_fig8_by_first_error.png
PRMBench CROP protocol, alpha .05, test traces with an error; share whose kept prefix contains an error, by relative position of the first error (thirds). Direct PRM trained; CHAIN label-free; LSML_CONT label-selected; LR supervised.
Data: `figures/data/crop_fig8_by_first_error.csv`.
