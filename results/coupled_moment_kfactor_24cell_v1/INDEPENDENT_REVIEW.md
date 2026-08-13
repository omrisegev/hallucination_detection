# Independent review of the CM-LFF result

## Verdict

The frozen experiment is valid and the tested method should not be promoted.
The result does not show that latent factors are absent. It shows that the
tested rule for naming and deleting them is wrong for these data.

## The decisive comparison

The label-free guards selected `k=0` in 19 cells. Those cells are exact
fallbacks to the existing IU-PCR and DUFS-LIU methods. Deflation was activated
in only five cells, and it reduced IU-PCR AUROC in all five:

| Cell | Removed factors `k` | CM-deflated IU-PCR minus IU-PCR |
|---|---:|---:|
| `epr_triviaqa_mistral24b` | 1 | -5.486 points |
| `se_squad_v2_llama8b` | 2 | -20.223 points |
| `seiclr_triviaqa_opt30b` | 2 | -5.587 points |
| `semenergy_triviaqa_qwen3_8b` | 3 | -16.683 points |
| `noise_gsm8k_mistral7b` | 2 | -4.944 points |

Across these five activated cells, mean AUROC fell from 0.79484 to 0.68900.
The conditional loss is 10.585 points. The headline loss of 2.205 points is
smaller only because it includes 19 unchanged fallbacks.

The diagnostic fixed-rank path also decreases monotonically:

| `k` | CM-deflated IU-PCR macro AUROC |
|---:|---:|
| 0 | 0.7761 |
| 1 | 0.7175 |
| 2 | 0.6732 |
| 3 | 0.6284 |
| 4 | 0.5734 |

## What failed

The method assumed that the component closest to the IU-PCR reliability vector
was the hallucination target and that every other component was nuisance. The
data do contain reproducible higher-order dependence, but good reconstruction
does not tell us the semantic meaning of a component. Post-hoc analysis shows
that removed components still contain substantial correctness information.
Hard deflation therefore deleted signal together with nuisance.

The PCA and feature-permutation controls do not explain away this result.
Selected same-rank PCA changes macro AUROC by only -0.098 points, while CM
deflation changes it by -2.205 points. The failure is specific to the attempted
component identification and subtraction, not merely to adding one more fit.

## Reporting cautions

- `CM-LFF direct factor` in the main report is a guarded score with IU-PCR
  fallback in 19 cells; it is not a direct factor in every cell.
- The paired 5/5 harm is more informative than overlapping unpaired headline
  intervals.
- The method fitted CP components; it did not establish that each component is
  an identifiable real-world cause.
- The high-`k` path is diagnostic only because some high-rank fits are unstable.

The small difference between the reproduced DUFS-LIU macro score here
(0.776560772) and the previous artifact (0.776561998) is 0.000123 AUROC points.
It comes from tied nearest-neighbour graph boundaries in cells with duplicate
feature rows. Gate values are identical and score correlations exceed
0.99999999. This is immaterial to the conclusion, but deterministic graph tie
breaking should be added before a publication run.

## Research consequence

Stop the CM-LFF hard-deflation branch. Keep mixed-v2 DUFS-LIU as the global
detection baseline. The ProcessBench HMM experiment is independent: it uses the
unchanged core-five IU-PCR token-risk curve and tests temporal onset decoding,
without deleting latent components.
