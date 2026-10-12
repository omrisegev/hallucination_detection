# Qwen3-8B vs Qwen3-4B, think-fix mask1 pipeline (2026-10-12)

Same code, labels, banks, scorers and evaluators for both models. Only the teacher-forcing model changes.
The 4B PRMBench tokens and step spans are identical to 8B on 6969/6969 answers (`SPANS_CHECK_4B.json`).
All numbers are point estimates. No paired 8B-vs-4B confidence interval was computed.

## Runs (AIRCC)
- GPU: PRMBench 281675/281676 (6969 rows), Hard2Verify 281677 (200), Socratic 281678 (2995). Alignment gates PASS.
- mask1: 0 failures in all three cells.
- PRMBench: bank 281681, re-score 281682, LR 281683, official PRMScore 281908 (4B) / 281909 (8B replay, same script).
- Externals: copy `thinkfix_ext_thinkfix_mask1_4b`, jobs 281694, 281706-281716.
- Externals keep the `_qwen3_8b` cell names, because the frozen scripts hardcode them.
- Frozen L-SML / BUNDLE and dev-panel selections were fitted on 8B, so 4B is a transfer test.
- ProcessBench rows of the source bank are unchanged (8B + 4B), so `pb_argmax_hit` is identical by construction.

## PRMBench within-answer AUC, bank N16 (`prmbench_panel_a.csv`)
| scorer | 8B | 4B |
|---|---|---|
| DS_FILTER_EQ | 0.7801 | 0.7764 |
| TCI_ONLY_ga | 0.7789 | 0.7677 |
| MEAN_MOM_cc | 0.7786 | 0.7763 |
| HEM_MOM_cc | 0.7773 | 0.7766 |
| MEAN_MOM_ga | 0.7767 | 0.7697 |
| HEM_MOM_ga | 0.7758 | 0.7697 |
| MEAN_EQ | 0.7743 | 0.7658 |
| HEM_EQ | 0.7731 | 0.7660 |
| ALL_EQ | 0.7708 | 0.7692 |

All 90 bank x scorer pairs:
- mean change (4B - 8B) -0.0005, range -0.020 to +0.019;
- bank order per scorer: median Spearman 0.92;
- scorer order per bank: median Spearman 0.73. The scorers are within about 0.01 of each other.

## Supervised LR (`prmbench_lr_gaps.csv`)
| bank | LR 8B | LR 4B | LR lead over best label-free, 8B | 4B |
|---|---|---|---|---|
| N16 | 0.8015 | 0.7967 | 0.0226 | 0.0200 |
| N56 | 0.8210 | 0.8162 | 0.0523 | 0.0478 |

LR drops 0.003-0.010 on 4B in all 10 banks. Its lead over the best label-free scorer narrows in 9 of 10 banks
(N14: 0.0191 -> 0.0194).

## Official PRMScore, N16 (`prmbench_official_prmscore.csv`, 700 rows)
| method | rule | overall 8B | overall 4B | simplicity 8B / 4B | soundness 8B / 4B | sensitivity 8B / 4B |
|---|---|---|---|---|---|---|
| DS_FILTER_EQ | R2 | 0.653 | 0.649 | 0.681 / 0.675 | 0.661 / 0.657 | 0.604 / 0.603 |
| LSML_CONT | R2 | 0.653 | 0.650 | 0.683 / 0.681 | 0.661 / 0.657 | 0.604 / 0.601 |
| MEAN_MOM_ga | R2 | 0.652 | 0.644 | 0.682 / 0.670 | 0.659 / 0.650 | 0.606 / 0.601 |
| any method | R1 | 0.653 | 0.650 | 0.677 / 0.675 | 0.663 / 0.658 | 0.607 / 0.605 |

Over all 700 rows, the mean change in overall PRMScore is -0.0007.

## Externals, locked L-SML (`external_lsml_locked.csv`, official metric)
| benchmark | arm | 8B | 4B |
|---|---|---|---|
| Hard2Verify | frozen L-SML | 0.421 | 0.433 |
| Hard2Verify | frozen equal | 0.395 | 0.397 |
| Hard2Verify | CT7 | 0.363 | 0.372 |
| Socratic | frozen L-SML | 0.628 | 0.631 |
| Socratic | frozen equal | 0.600 | 0.600 |
| Socratic | CT7 | 0.566 | 0.577 |

Panel / nested / cont external methods (`external_panel.csv`, `external_nested.csv`, `external_cont.csv`):

| file | rows | mean official-metric change | range |
|---|---|---|---|
| external_panel.csv | 442 | +0.002 | -0.026 to +0.050 |
| external_nested.csv | 550 | +0.004 | -0.027 to +0.050 |
| external_cont.csv | 40 | +0.003 | -0.014 to +0.026 |
