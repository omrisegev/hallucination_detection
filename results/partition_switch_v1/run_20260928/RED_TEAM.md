# Red team: partition_switch_v1 run_20260928 (outputs 0c90aa4e4, runner 3dfb66511, protocol 37d41f181)

- **A** independent recomputation (own AUC, own rule selection for all 80 targets, own 20,000-draw bootstrap) and a
  sibling-held-out rerun.
- **B** robustness: leave-family-out rule selection (the 8 banks are 4 families: 13/16, 20/23, 32/35, 51/54), threshold margins,
  3,280 perturbed decisions, breadth by fold / class / length, concentration.
- **C** nulls (500 permutations) and math: content vs position, whether the label-free statistic tracks conditional dependence and
  predicts the gain, the rank-one misfit statistic.

Scripts in the session scratchpad (`rt_ps_A`, `rt_ps_B`, `rt_ps_C`).

| CLAIM | VERDICT | EVIDENCE |
|---|---|---|
| 1. The leak-free switch uses the grouped method only on B32/B35 (all folds, statistic S1) and meets the frozen success criterion: no bank loss, 8-bank mean +0.0028 [0.0020, 0.0036] (hem version +0.0031) | **numbers confirmed; the success does not survive holding out the bank family** | **A/B**: all 80 decisions, gains and intervals reproduce exactly. With the target's whole FAMILY held out (for B32 also B35, etc.), the rule turns grouping on in 0/80 decisions; SW = BASE on 8/8 banks; 8-bank mean exactly 0. B32 and B35 are each other's only positive training evidence. The threshold used for most targets (0.171-0.173) is literally a B54 fold's S1; B54 lies within 0.02 of it in 5/5 folds |
| 2. The B32/B35 gain is content and broad | **confirmed, moderately concentrated** | **C**: content share 77% (B32; swap null +0.0030, z 6.4) and 90% (B35); **B**: positive in 5/5 folds and 3/3 length tertiles; top 1% of answers carry 36% / 45%; the confidence and redundancy classes carry 65-74%. **C**: by relative first-error position the B32 gain is -0.0119 / +0.0138 / +0.0355 (early to late) |
| 3. Mechanism (hypothesis): grouping with DS-estimate weights pays where the merged groups are close to independent (low S1) | **refuted beyond the one positive family** | **C**: S1 tracks the label-using class-conditional dependence faithfully (Spearman 0.97-0.99 over 40 bank-folds), but its relation to the gain reverses outside B32/B35 (+0.48 over 30 cells, p 0.008; the rule assumes negative); B13 has the highest dependence and gains in 5/5 folds; B54's mark-level dependence is the lowest of all banks and it does not gain. One positive family cannot separate independence from family-specific causes (its lf__ channels and their orientation, K = 6-7 with the smallest largest group) |
| 4. S3 (rank-one misfit) | **uninformative / confounded** | **C**: at K = 3 an exact rank-one fit exists whenever the three covariances have a positive product (all B13 cells: S3 ~ 5e-13); at K >= 4 it grows mechanically with K even under a true one-factor model (simulation 0 / .037 / .042 / .046 / .054 for K = 3..7; Spearman with K 0.84 over 40 cells) |
| 5. The blend (no parameter) | **fails** | **A**: B23 -0.0020, B51 -0.0053 (intervals below 0); both losses mostly positional (content share 24% / 9%) |
| Side notes | | **B**: the switch discards the grouped method's B13 gain (+0.0016) and the hem version's B13/B54 gains; ProcessBench SLA of the switch is lower than BASE on B32/B35 (-0.0072 / -0.0069) |

## What the design got wrong
The protocol protected against label leakage across banks (a target's fold never trains its own rule) but treated the 8 banks as
exchangeable units. They are 4 families of twins (each bank with and without the three digit channels), so leaving one bank out
still left its twin - with nearly the same statistic and the same gain - in the training set. The frozen criterion was met, but the
"rule" it produced is a lookup for the 32-channel family, and it transfers nothing when the family is held out.
