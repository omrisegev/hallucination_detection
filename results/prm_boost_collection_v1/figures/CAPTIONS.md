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

### status_official_by_prm_gap.png
Official metric per benchmark and PRM under the label-free GAP count rule (PRM native rule hatched). PRMBench: 6,211 non-control answers; Socratic: 2,995 answers; Hard2Verify: 177 answers every PRM can read. PRM same rule = the PRM answer-z score with the same GAP rule, on the better of logit(1-reward) / (1-reward) (chosen after scoring; favours the PRM). Access: PRM trained by its authors; TELE/CHAIN label-free fusion; LSML_CONT label-selected channels. Development data, already inspected.
Data: `figures/data/status_official_by_prm_gap.csv`.

### status_contrasts_gap.png
Paired bootstrap contrasts (2,000 draws; PRMBench by source group, external sets by answer), official metric points. PRM reference = PRM answer-z with GAP on the better scale (post hoc). All rules label-free.
Data: `figures/data/status_contrasts_gap.csv`.

### crop_controls_c6_kept_vs_alpha.png
PRMBench, CROP Table 6 protocol on the exact Hub-order first-5,000 subset (4,995 scored traces; C1), 10 seeds, test split. Prefix kept vs alpha on two grids (C2): train-quantile 201 and fixed {0,.01,..,1} after a train min-max map. Every proxy's lambda is calibrated with labels (CROP access). Direct PRM: trained PRM. LRs: supervised. CHAIN_N16 / LSML_CONT: label-free fusion (LSML_CONT channels label-selected); dashed = inductive fit on the 60% training traces (C4).
Data: `figures/data/crop_controls_c6_kept_vs_alpha.csv`.
