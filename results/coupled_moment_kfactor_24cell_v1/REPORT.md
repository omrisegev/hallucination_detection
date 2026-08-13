# Coupled-moment k-factor experiment

**Decision: DO NOT PROMOTE.**

This experiment asked whether non-Gaussian latent nuisance factors can be identified from the original mixed-v2 features, removed, and followed by the same IU-PCR/DUFS-LIU solvers.

## Main result

CM-deflated IU-PCR scored **0.7540** cell-macro AUROC, compared with **0.7761** for IU-PCR and **0.7766** for DUFS-LIU.
Its change versus IU-PCR was **-2.205 points** (0 wins, 5 losses; worst -20.223). The equal-family interval was [-8.513, -0.062] points.

Selected nuisance-factor counts were {0: 19, 1: 1, 2: 3, 3: 1, 4: 0}. 1 cells additionally failed a full-fit ambiguity guard.

## Headline table

| Method | Cell AUROC | Family AUROC | QA | Math | AUPRC |
|---|---:|---:|---:|---:|---:|
| Deployed-style U-PCR (mixed-v2) | 0.7740 | 0.7418 | 0.7588 | 0.7830 | 0.7100 |
| IU-PCR | 0.7761 | 0.7423 | 0.7597 | 0.7859 | 0.7102 |
| SU-PCR reproduction | 0.7714 | 0.7393 | 0.7489 | 0.7849 | 0.7081 |
| SDSF | 0.7326 | 0.7166 | 0.7146 | 0.7434 | 0.6635 |
| DUFS-LIU | 0.7766 | 0.7430 | 0.7604 | 0.7862 | 0.7127 |
| CM-LFF direct factor | 0.7530 | 0.7059 | 0.7042 | 0.7823 | 0.6793 |
| CM-deflated IU-PCR | 0.7540 | 0.7078 | 0.7064 | 0.7826 | 0.6801 |
| CM-deflated DUFS-LIU | 0.7544 | 0.7085 | 0.7070 | 0.7829 | 0.6827 |
| Second-order deflation control | 0.7751 | 0.7419 | 0.7568 | 0.7861 | 0.7073 |
| Permuted-moment control | 0.7726 | 0.7417 | 0.7607 | 0.7797 | 0.7059 |

## Promotion gates

- FAIL — gain at least half point vs iu
- FAIL — gain at least half point vs dufs
- FAIL — family interval positive vs iu
- FAIL — qa nonnegative
- FAIL — math nonnegative
- FAIL — at least 14 wins
- FAIL — worst loss no more than 2pp

## Interpretation boundary

CM-LFF is not tensor-identifiable under the categorical theorem in the source survey. A stable component may still represent confidence, length, or difficulty rather than correctness. This report evaluates retrospective development cells and cannot establish an external claim.

## Figures

![Headline](figures/headline_auroc.png)
![Rank path](figures/cm_iu_rank_path.png)
![Per-cell delta](figures/per_cell_delta.png)
![Moment stability](figures/moment_stability.png)
![Selected ranks](figures/selected_rank_counts.png)
