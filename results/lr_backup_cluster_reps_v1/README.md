# Per-cluster representations vs LR (PRMBench, N43 / N56 / L1-21), 2026-10-18

Backup material for the LR slide. Omri asked: on N43, N56 and the bank that LR selected, did we try
(1) the 2nd and 3rd eigenvectors of each cluster, (2) sum and difference per cluster, (3) the cluster mean?

## Inputs
- Bank matrix: rebuilt from `external_banks_v4/source/FEATURES.npz` (answer z-score, digit channels masked).
  Parity: the saved LR fold models in `results/monday_lr_reference_v1/cache/{N43,N56}.pkl` give the same
  per-answer AUC on the rebuilt bank (max difference 0; 0.8336 / 0.8355).
- Step labels: `results/localization_full_benchmark_v3/evaluation/JOINED.npz`. Folds and source groups:
  `step_evidence_v1/OOF_ANSWERS.csv` (branch `claude/readout-quickest-detection-v1`).
- DS survivors and L-SML groups: `fit_diag` in the same cache files.
- L1-21: the 21 channels that enter the N56 L1 path at C <= 0.0022 (`results/monday_lr_weights_v1/L1_PATH.csv`).
  The N56 groups are restricted to these channels; channels that are not N56 survivors are singletons.

## Representations (per cluster, each column answer-z)
- MEAN: cluster mean. PC1/PC2/PC3: projection on eigenvectors 1-3 of the cluster covariance.
- MEAN+DIFF: cluster mean plus each member minus the cluster mean (a pair: x1 - x2).

## Fusion
- EQ (label-free): equal weights; column sign from its correlation with the MEAN_EQ anchor
  (L1-21: the full-N56 MEAN_EQ anchor).
- SML (label-free): leading eigenvector of the column correlation matrix, sign aligned with the anchor.
- oracle sign: equal weights, sign from labels. oracle LR: balanced LR, C=1, 5 source folds, OOF.

## Result (within-answer AUC)
| representation | N43 | N56 |
|---|---|---|
| LR, all channels | 0.834 | 0.836 |
| best label-free before (HEM_MOM_cc / MEAN_MOM_cc) | 0.774 | 0.775 |
| PC1+PC2, SML | 0.779 | 0.784 |
| MEAN+DIFF, EQ | 0.783 | 0.785 |
| MEAN+DIFF, SML | 0.803 | 0.769 |
| oracle LR on cluster means | 0.777 | 0.780 |
| oracle LR on PC1+PC2 | 0.816 | 0.819 |
| oracle LR on MEAN+DIFF | 0.832 | 0.833 |

- With perfect weights, cluster means stay at 0.78. PC2 recovers about 70% of the gap to LR, MEAN+DIFF about 97%.
- Label-free, PC2 (SML) and MEAN+DIFF (EQ) add +0.005 to +0.010 on both banks (paired source-group
  bootstrap, 2,000 draws, CIs above 0). MEAN+DIFF with SML is not stable across banks.
- PRMScore (top-1) changes by at most +0.003.
- L1-21 fails label-free (0.46-0.75): the LR selection keeps the negative-weight partner channels,
  and the label-free orientation does not recover their sign.
- Arms were compared on the data they are reported on. No held-out test.

## Files
`data.py` (loader, metrics), `run2.py` (final run), `boot.py` (bootstrap), `RESULTS_v2.json`, `BOOT_v2.json`,
`run2.log`. `RESULTS_v1.json` = first run (identical for N43/N56; L1-21 used the subset's own anchor).
Backup document (Claude Doc): https://claude.ai/code/artifact/8cc0e1f7-082e-45ee-bd0f-d4fae5cff57f
