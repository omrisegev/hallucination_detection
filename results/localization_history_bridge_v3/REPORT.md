# Historical localization bridge — Step316

Review PASS.131 earlier method entries,25 repaired fallback anchors,199 contrasts and7 gate diagnostics.
Same original58 answers; current110/107 entries shown separately. No new fits or predictions.

| Lane | Arm | Valid PRMB | Corrected AUC | Within-answer AUC | PB F1 |
|---|---|---:|---:|---:|---:|
| readout | moments27_local8__iu@@parent_first | 12 | 0.64753 | 0.70397 | 0.00% |
| readout | moments27_local8__iu@@parent_peak | 12 | 0.64753 | 0.70397 | 17.71% |
| readout | moments27_local8__iu@@hmm_entry | 10 | 0.56022 | 0.65521 | 0.00% |
| readout | moments27_local8__iu@@kalman_level | 12 | 0.56703 | 0.60096 | 5.36% |
| readout | moments27_local8__iu@@imm_level | 12 | 0.61291 | 0.68192 | 14.79% |
| readout | moments27_local8__iu@@bocpd_rise | 12 | 0.58178 | 0.62798 | 13.69% |
| sampling | moments27_local8__iu@@risk_top | 12 | 0.66850 | 0.69799 | 19.05% |
| sampling | moments27_local8__iu@@dufs_transposed | 12 | 0.64817 | 0.68087 | 13.69% |
| sampling | moments27_local8__iu@@dufs_permuted | 12 | 0.64487 | 0.68935 | 21.88% |
| context | context__equal | 12 | 0.66090 | 0.69290 | 27.67% |
| context | context__joint0 | 9 | 0.63595 | 0.63104 | 27.43% |

Joint DUFS AUC0.76333 covers only4 answers; original Joint is0.75000 on those same4.
Entropy-risk IU shows0.66850/19.05% versus0.64753/17.71%, but both paired intervals include0, within-answer AUC falls, and PB loses one success/gains one. No winner.
An answer offset raises pooled IU AUC0.64753->0.69240 without changing within-answer ranking or decisions.
Next: bounded sampling replication on the current110 with original banks/routes, matched controls, budget feasibility and short-error retention reporting. Full research goal remains open.
