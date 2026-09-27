# Red team: expectation_realization_v1 stage B2, level reduction (run_20260927_stage_b2, commit f6b0da475)

Three independent agents, no summary files read:
- **A**: fresh recomputation from the raw inputs with its own Dawid-Skene EM (41 starts) and its own tie key.
- **B**: coverage audit from STEP_SCORES.npz and raw inputs, per fold, error class, length tertile, concentration, ProcessBench cells. Its side script for the position bank was stopped after 38 minutes (it decompressed the npz inside a per-answer loop); that content is covered by C.
- **C**: within-answer shuffle and whole-answer same-length swap nulls (500 permutations, own code), mechanism, math, position bank.

Scripts are in the session scratchpad (`rtb2_A`, `rtb2_B`, `rtb2_C`).

| CLAIM | VERDICT | EVIDENCE |
|---|---|---|
| 1. Counting the level family once (merge, no removal) makes the binary-property rule beat averaging all 13 channels: 0.7818 vs 0.7749 within-AUC, PRMScore 0.6587 vs 0.6527; median over the ten removals +0.0071 | **confirmed, but the gain is not the rule's** | **A**: 0.7818 / 0.6587 exact; median +0.0071 (range -0.0022 to +0.0115, 8/10). **B**: 5/5 folds, 7/8 classes (missing_condition -0.0080), 3/3 length tertiles; 146/146 arms fully scored. **C**: decomposition of the gain: the channel filter +0.0053 is positional (swap null +0.0064, 83% >= observed); removing or merging two level channels under plain averaging +0.0028 is not positional; the rule over plain averaging of the same channels contributes about 0 (removal median) to +0.0016 (merge) |
| 2. The estimated weights beat equal GROUP weights: median +0.0031 (10/10), merge +0.0035 | **numbers confirmed, interpretation weakened** | **C**: not positional (swap null -0.0011), but the whole gain is shrinking the incoherent change group (alone 0.669 within-AUC): a fixed 0.22 change weight beats the rule in 10/10 configurations, a label-free first-component variance-share weight in 9/10 and equals the oracle on merge (0.7823, post hoc). Equal group weights are a weak reference (below plain channel averaging). The position-bank +0.0155 is an artefact: there the partition splits the change family into two groups, so equal group weights give change half the weight |
| 3. The rule adds nothing over the plain average of the same channels | **confirmed for the removals; merge is a small, fragile exception** | **A**: removal median +0.0001 (range -0.0015 to +0.0010). **C**: removal median indistinguishable from both nulls; merge +0.0016 [0.0005, 0.0027], 5/5 folds, survives both nulls. **B**: the merge gain is directionally consistent (5/5 folds, 3/3 length tertiles, 6/8 classes, answer sign test p = 0.018 unadjusted) but concentrated (87% of answers unchanged, top 1% carry more than the whole mean, 5%-trimmed +0.0008): wording = small edge, not established; for the removals the null reading holds (6/10 positive, two of them below 0.0001; 2/5 folds) |
| 4. Estimates still biased but less unevenly: prevalence ~0.23 vs 0.14; estimated/oracle weight ratio level ~2.6 vs realization ~1.8 on merge (base 3.7 vs 1.5); clean-step group-mark correlation 0.52 -> 0.3 | **numbers confirmed, "less dependence" refuted** | **A**: per-fold prevalence 0.225-0.240, ratios level 2.58-2.66, realization 1.77-1.91. **C**: normalized weights depend only on the ratios of est/oracle factors (identity checked on 120 rows), so the common inflation cancels (angle to oracle 20 deg base -> 9-10 deg merge); the weakest group is identified in 60/60 folds but level is ranked above realization in 5/5 merge folds. The 0.52 -> 0.3 drop is mechanical: 0.52 was the correlation between the two level sub-groups, which merging deletes; between-family correlations are unchanged (realization-change 0.143, realization-level ~0.31) |
| 5. L-SML still loses to averaging the same channels (median -0.0073, 0/10); merge helps it (+0.0036 over L-SML on 11) but only to the level of averaging | **confirmed** | **A/B**: numbers exact; L_own below E_equal in 10/10 configurations and in 49/50 fold x configuration cells; L_grp[merge] - E_equal[base] positive in 1/5 folds |
| 6. ProcessBench: every arm below ct7 | **confirmed** | **B**: merge rule 0.3788 vs ct7 0.3989, above ct7 in 1/8 cells, above all-13 averaging in 4/8 |
| 7. Partitions: merge joins exactly the two level groups; 9/10 removals give three families identical in 5/5 folds (rm_H1_VE1 not) | **confirmed** | FIT_MANIFEST: level_groups_merged = 2 in 5/5 folds on both banks; **A**: rm_H1_VE1 gives 4 groups in folds 0-2 and 3 in folds 3-4. rm_H1_LM (0.7864) is the best of ten and label-selected; never report it as an unselected result |

## What the author's preliminary report got wrong (corrected before this table)

- It read "the rule beats equal group weights" as value from the estimates. It is a weak reference being beaten by down-weighting one weak group.
- It reported the position-bank +0.0155 as a strengthening. It is an artefact of a split change family.
- It read the correlation drop 0.52 -> 0.3 as reduced dependence. It is the deletion of one pair by construction.

## Added from agent B's final report

- **The position-bank replicate is not a replicate of the three-family design.** On that bank 12 channels survive (top50_js is kept), bocpd_p0 is always a singleton, and removals give 4 groups in 50/50 configuration-folds (three families in 0/50); merge gives 4 groups. Position-bank contrasts therefore test a structurally different setup.
- **Class exceptions:** missing_condition is negative on every rule contrast (merge vs all-13 average -0.0080, 0/10 removals positive); deception: rule below equal group weights.
- **ProcessBench:** merge rule below ct7 in 7/8 cells and 5/5 folds (1592 vs 1699 hits over 4,442 erroneous answers); above all-13 averaging in 4/8 cells.
- Official PRMScore per fold/class was not audited (per-fold thresholds are not saved outside the run).

## Corrections after external review (Codex and a side Claude agent, 2026-09-27)

- **Decomposition.** The merge-path gain 0.7818 - 0.7749 = +0.0069 is the DS filter (+0.0053) plus the rule over the plain average of the same 11 channels (+0.0016). The +0.0028 "level counted once under averaging" belongs to the REMOVAL path (median plain average of 9 channels minus plain average of the 11 survivors); adding it to the merge path double-counted.
- **Position.** On the original bank the swap null reproduces the filter gain, so no value beyond position is shown there; on the position-adjusted bank +0.0043 survives. "Entirely positional" was too strong.
- **"Ceiling".** B_oracle is the same rule with true psi/eta, not a ceiling for all weightings (a label-using correlation-aware weighting reached 0.7890). Being near it does not show that the estimates are accurate; the est/oracle factors became more uniform (max/min 1.4-1.5 vs 2.5), they did not cancel.
- **Manual merge.** The merge and the ten removals use a hard-coded list of the five level channels in `er_stage_b2_run.py`. Stage B2 is a mechanism test, not an automatic method; the 0.7818 row must always carry that label. An automatic merge rule is an open research question.
- **Open:** the paired PRMScore interval of the merge rule minus the plain average of the same 11 channels was not computed (per-fold PRMScore thresholds are not saved in STEP_SCORES).
