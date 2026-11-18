# ANSWERS TO THE MAC: questions 1-56 (2026-11-18)

Status: answers from the cloud project. The experiment is still a PROPOSAL. Do not start GPU work before Omri gives the OKs in section 0.3.
Numbering: the same order as the Mac's question list (A 1-10, prep 11-16, B 17-21, C 22-32, D 33-35, E 36-38, F 39-44, G 45-50, H 51-55, I 56).
Read with: PLAN_SUMMARY.md and RUN_REQUIREMENTS.md (both updated today with the corrections below).

## 0. Summary

### 0.1 Branches and locations: checked against origin (GitHub), 2026-11-18

| Branch (origin) | Head | Relation |
|---|---|---|
| claude/prm-boosting | a76e0ba0 | Contains 06ff15d4 (standalone driver change). |
| claude/simes-answer-gate-2026-11-15 | 43b2886e | a76e0ba0 is an ancestor. 25 commits ahead of prm-boosting (only additions). Driver file identical. Full tree (27,196 files). RECOMMENDED BASE. |
| claude/shuffle-gate-design-2026-11-16 | 71d19896 | Child of 43b2886e, but its commit DELETES 27,176 files (a sparse tree: only cluster/, docs/, 6 results dirs). Do NOT use it as a base. Copy only results/shuffle_gate_design_v1/. |
| claude/cross-model-gate-check-2026-11-14 | b41634ef | Parent of 43b2886e (steps/*.csv.gz are in the base). |
| claude/pb-first-error-rule-2026-11-10 | 6224d4ea | Not in the base. Copy results/pb_first_error_rule_v1/. |
| claude/sml-readout-calibration-2026-11-12 | d65c0143 | Not in the base. Copy results/sml_readout_calibration_v1/. |
| claude/sa-readouts-2026-11-12 | 97c07d1e | Not in the base. Copy results/bench_gap_diagnosis_v1/readout_alternatives/ and data_export/readout_steps_sa_v1/. |
| claude/gsp-ablation-baseline-2026-11-12 | d3346dc2 | Code is in the base already. |
| claude/gsp-status-review-2026-11-13 | b1cf890d | Reference only. |
| main | 3829eb0b | Not used. |

Drive (checked through the Drive connector): graft_rewards.py, prm_boost_build_bank_external_v2.py and the 1.12 GB prmbench_telemetry.pkl copies exist. The Mac's findings are correct.

New facts found in the repository today (important for the Mac):

| Fact | Where | Use |
|---|---|---|
| Corrected source-question groups exist: PRMBench 707, PB 2,776, total 3,483 (66 PB questions are identical to PRMBench questions and join the PRMBench group). | results/localization_source_group_audit_v1/ (REPORT.md, FOLDS_V2.json keys = group ids, group_strata). Recipe: scripts/audit_localization_source_groups.py. Both on the base branch. | Split and donor rules (answers 6-10). |
| PB standalone 7B, 16 channels in R0 per step (sa_N16, 25,697 steps), plus G16 and digit_active. | results/pb_first_error_rule_v1/cells/PB_qwen_prm_7b.npz (13.9 MB, real blob, not LFS). | PB parity for all 16 channels without the PC. |
| Standalone 7B top10 (and 6 other readouts) for the 11 token channels, PB (3,400 answers) and PRMBench (6,969 answers, 94,203 steps). Gate: equal to the stored banks (max abs diff 0). | data_export/readout_steps_sa_v1/ on claude/sa-readouts-2026-11-12 (10 MB and 36 MB, real blobs). | Token-channel parity for both benchmarks without the PC. |
| PB: 28 answers with 2 steps, 150 with 3 steps; median 7 steps; median first error = step 3 (one-based). Token count per step: median 72, p90 159, max 675; 0.4% of steps have < 10 tokens. | gold json + readout npz | V1 rule (answer 5), readout bounds (answer 27). |

### 0.2 Main decisions (details in the numbered answers)

| Topic | Decision |
|---|---|
| Base branch | New branch claude/ssl-one-experiment-2026-11-18 from origin/claude/simes-answer-gate-2026-11-15 (43b2886e). |
| Split | One 50/50 split over the 3,483 corrected groups, stratified by group_strata, seed 20261116. Split before V1/V2. |
| Readout | COUNT-based soft top-N (not a share). N = 10 with a hard edge = top10 exactly. |
| AUROC term | Paired: foreign step k in V2 vs the original step k in the parent C answer (same position, same prefix). |
| log(n) | Stratifier only. Not a view. |
| Marking with the gate | Rank rule: close the (1 - pi-hat) share of answers with the lowest answer score. No P >= .5 rule. No soft PRMScore. |
| PB change point | Free EM with the per-answer gate prior fixed (no pi re-estimation), on the fused R0 step score, B = 20. |
| Gate | BBE on the stage-A answer score (no PU classifier), cross-fitted. pi-hat per pool. |

### 0.3 Items that need Omri's OK (the Mac must wait for these)

| # | Item |
|---|---|
| a | Approve the experiment and the 50/50 dev/test split. |
| b | Download the model (15.23 GB, revision ef9926d7) to the internal disk. |
| c | Download about 3.5 GB from Drive to the Mac. |
| d | Upload hd_collection_data (about 2 GB) from the PC to Drive (Omri, or the cloud project through Remote Control on the PC). Needed only for the PRMBench pipeline parity (68.26) and for running simes run.py unchanged. PB parity can start without it (section 0.1). |
| e | Optional: raise the GPU wired-memory limit (sudo sysctl iogpu.wired_limit_mb=...). It resets at reboot. Only if the 12-row timing run shows swap. |
| f | GPU source. AIRCC is back (coordinator, 2026-11-18 16:00 Israel time): jobs only through the cycle4 spot account; a 1-GPU test job estimates start 2026-11-19 21:19. The cycle2 account is not valid. Recommendation: Mac first (12-row timing with the real model); AIRCC is the fallback if the long rows swap. |

## A. Inputs and split (questions 1-10)

| # | Answer | Why |
|---|---|---|
| 1 | Only step k is "known wrong". Steps k+1..n have an unknown status (they continue from a foreign step; PB's cascade convention would call them wrong, but we do not know it). Stage S uses exactly one step per V2 answer: step k. Steps k+1..n are not used in stage S. In stage A the whole V2 answer is "known wrong". | Only step k has a label that we made. |
| 2 | Yes. The sidecar records k as "known wrong step", never as a first-error label. The sidecar holds no gold labels. For dev diagnostics only, join the parent's gold first error later (for example, to report stage S on V2 rows whose parent is clean before k). | A parent error before k is common in PB (median first error = step 3). |
| 3 | Hard rules (level 0): same split side, different corrected group, same PB subset, relative position within +-.2, character length within +-20%. Inside the candidates, prefer the same generator (PB only) and donor steps not used yet; then pick at random (seed 20261116). If no candidate: L1 length +-40%; L2 position +-.4; L3 any PB subset. If still none: skip the row and log it. Report the count per level. Expect almost all rows at level 0; tell Omri if skips > 1%. | Keep realism first (length, position), then coverage. |
| 4 | Any answer to a different corrected group on the same split side. Correct or wrong donor answers are both allowed (the step is foreign in the new context, so it is wrong). A donor step can be used more than once, at most 3 times; prefer unused steps. Same generator: preferred, not required (PB has 12+ generators; style matters for the realism check). PRMBench has no generator field: ignore it there. | Wrong-in-context does not depend on the donor's own label. |
| 5 | Add one condition: the original last step must not stay last. For n >= 4: uniform random permutation of steps 2..n under that condition. For n = 3 (150 PB answers) the only option is the swap of steps 2 and 3; it meets the condition; keep it. n = 2 (28 PB answers): excluded. Add a sidecar flag "single_adjacent_swap" for diagnostics. | The last step holds the final answer; moving it breaks the answer. A swap of two independent middle steps alone can leave a valid answer. |
| 6 | Use the corrected group map, not source_id. PB group = sha256 of the whitespace-normalized question text (' '.join(problem.split())), id 'pb_source_v2:' + first 24 hex. If the same normalized text is a PRMBench question, the PB row takes that PRMBench group id. Recipe: scripts/audit_localization_source_groups.py (base branch). Check: your ids must equal the keys of FOLDS_V2.json; counts PB 2,776 + PRMBench 707 = 3,483. | source_id = uid in PB (as you found). The corrected map is verified (REPORT.md: 430 repeated PB questions, 988 answers). |
| 7 | PRMBench group = connected component of (a) the source seed, regex (prm_(?:train\|test)_p\d+_\d+)$ on source_idx, and (b) the exact whitespace-normalized question text. id 'prmb_source_v2:' + sha256('\|'.join(sorted seeds))[:24]. 758 seeds join into 707 groups. Same script as answer 6. | Old source_idx was a perturbation id, as CLAUDE.md says. |
| 8 | One joint split over all 3,483 groups (PB and PRMBench together), stratified by FOLDS_V2.json group_strata (pb_gsm8k, pb_math, pb_olympiadbench, pb_omnimath, prmbench, pb_math\|prmbench). Inside each stratum: shuffle the groups with seed 20261116; the first ceil(half) go to dev. Report answers, steps and true pi per side (labels only for this count). | A joint split keeps the 66 shared questions on the same side in both benchmarks. |
| 9 | Yes. Split first. Then build V1 and V2 inside each side; donors only from the same side. | No test steps inside dev answers. |
| 10 | Yes, all 6,969 rows are parents (758 original + 6,211 modified). Drop the category condition for PRMBench donors: donor = any step of an answer in a different corrected group on the same side, with the position and length rules. | The PRMBench category describes the injected error type of a modified answer, not the topic of the question. Original rows have none. |

### Prep (questions 11-16)

| # | Answer |
|---|---|
| 11 | Internal disk (282 GB free). Set HF_HOME to a folder there (for example ~/hf_home). Download with revision ef9926d7. Needs Omri's OK (0.3 b). |
| 12 | Branch claude/ssl-one-experiment-2026-11-18, cut from origin/claude/simes-answer-gate-2026-11-15 (43b2886e), not from prm-boosting. It contains a76e0ba0 and 06ff15d4, the same driver, and all analysis code (c4u_telemetry_only_v1, cross_model_gate_check_v1 with steps/, simes_answer_gate_v1, gsp_ablation_baseline_v1, prm_boost_processbench_v1, prm_standalone_telemetry_v1, prm_boost_external_v1, localization_source_group_audit_v1). Do not cut from claude/shuffle-gate-design-2026-11-16: its commit deletes most of the tree. Then copy in, with GIT_LFS_SKIP_SMUDGE=1: git checkout origin/claude/shuffle-gate-design-2026-11-16 -- results/shuffle_gate_design_v1; git checkout origin/claude/pb-first-error-rule-2026-11-10 -- results/pb_first_error_rule_v1; git checkout origin/claude/sml-readout-calibration-2026-11-12 -- results/sml_readout_calibration_v1; git checkout origin/claude/sa-readouts-2026-11-12 -- results/bench_gap_diagnosis_v1/readout_alternatives data_export/readout_steps_sa_v1. Push with --no-verify. No merge. |
| 13 | Yes to both. MPS: add a device flag (auto = cuda, then mps, then cpu). Keep the CUDA code path unchanged. On MPS use model.to("mps") without device_map, so accelerate is not needed. Chunk: add a flag, default 1024 (unchanged); use 256 on the Mac only after a 12-row check that the per-token values are identical between 1024 and 256. |
| 14 | Yes. --max-batch 1 for all 6 runs. Note: the stored runs used --max-batch-tokens 16384. This is why parity against the stored banks is Spearman only (answer 20 and PLAN_SUMMARY section 6). |
| 15 | Yes, after Omri's OK (0.3 c). Put the data outside the repository. |
| 16 | hd_collection_data: Omri uploads it (0.3 d). Until then, use the GitHub data in section 0.1 for PB parity: L-SML on sa_N16 + GAP + gold + pbeval.py (pb_first_error_rule_v1) should reproduce PB SLA 38.47; verify this first, and if it does not match, wait for hd_collection_data. E0: skip it in this run. |

Memory headroom (3.6 GiB): put the longest rows first in the 12-row timing run (PB 2,577 tokens, PRMBench 3,555 tokens). Watch swap (memory_pressure, vm_stat). If swap: chunk 128; then ask Omri about 0.3 e; else run PRMBench on AIRCC (0.3 f).

## B. Channels and normalization (questions 17-21)

| # | Answer | Why |
|---|---|---|
| 17 | Keep the 3 digit channels. | TELE16 is the method of record. Omri 11-12: no feature removal at this stage. |
| 18 | Correct. 11 token channels in that order; 5 step channels: realized_z, realized_drv, digit_alternative, digit_spread, digit_alternative_innovation. Stage S changes only the 11 token channels. | Matches the code (TOK11 = N16[:11]) and readout_steps_sa_v1 channel_names. |
| 19 | Yes. Risk-oriented token values (validate_telemetry), after mask1 (forced token 0 dropped). "Top" = the largest risk-oriented values. PB empty steps are dropped; PRMBench empty steps get the answer mean (as the stored builders do). | Same input as top10 today (extract_readouts.py, gate max abs diff 0). |
| 20 | Stop rule only on the raw token telemetry (per-token values of the steps before the change, max abs diff <= 1e-4; you measured 0.0 on MPS). At bank level: report the max abs diff per channel for steps before k; do not stop on it. A channel with a forward window can change the last step before k without a fault. | The equal-prefix property is a property of the forward pass. |
| 21 | Yes. R0 exactly as today: answer-z of the raw step value; digit channels: z over digit-active steps only, 0 elsewhere. | Method of record. |

## C. Readout, stage S (questions 22-32)

| # | Answer |
|---|---|
| 22 | Yes: r = sum_i w_i * v_(i) / sum_i w_i, v sorted from high to low, i = 1..n. Weight form: see 23. |
| 23 | Change to COUNT. w_i = sigmoid((N_c + 0.5 - i) / s), i = rank (1 = largest), s = 1 token, N_c continuous. With N_c = 10 and a hard edge this is top10 exactly (min(10, n) largest tokens), so the method of record is a point inside the family. Report the hard top-round(N_c) as a check. |
| 24 | All 120 pairs of the 16 channels. The 10 pairs between step channels are constant, so the gradient comes from the 110 pairs that contain a token channel. Report all 120. |
| 25 | All pairs (as in the plan). Report the cross-group-only sum as a diagnostic. Why: the groups are detected again after each readout change (a moving, non-smooth target); within-group SML also assumes a rank-1 structure; Omri's goal is independence given the label. |
| 26 | Paired: for each V2 answer, compare the raw readout value of the foreign step k with the raw readout value of the ORIGINAL step k in the parent C answer (same position, same prefix). AUROC_c = share of pairs where V2 > C (ties count .5). Gradient: sigmoid((x_V2 - x_C) / s_c), s_c = .1 x SD of channel c over C steps. Fixed risk orientation, no absolute value: AUROC_c < .5 lowers the term. Only the 11 token channels enter the AUROC term. Do not compare with all C steps: that mixes position and other answers. |
| 27 | Decide and document. Defaults: N_c = 1 + exp(theta_c), 1 <= N_c <= 200; inits N_c in {3, 10, 30} (all channels together), keep the lowest L; Adam, lr .05, 500 full-batch steps. corr = Pearson on R0 values (R0 is already z-scored; no soft rank). Fit on dev V2 half A (by group), check stability on half B. Report exact Spearman at the end (the dev metric). |
| 28 | Per benchmark. Among the lambda values that pass the guards (dev PRMScore / SLA not lower than top10 by more than 1), choose the lowest real conditional mean |corr| on dev. If two values are within .005, choose the higher PRMScore (PRMBench) or SLA (PB). |
| 29 | Yes. Mean |Spearman| over the 120 pairs on R0, computed separately in wrong steps and in clean steps; primary = mean of the two. PB: wrong = the first-error step; clean = steps before it plus all steps of clean answers; steps after the first error are excluded. PRMBench: step labels as given (v3 contract). CI: clustered bootstrap by corrected group, 2,000 draws. |
| 30 | Yes, scipy.optimize.milp, with the exact paired AUROC and exact Spearman. Candidates per token channel: top-N for N in {1, 2, 5, 10, 20}, mean, median, p90. Linearize each pair with y >= x_a + x_b - 1, y >= 0 (corr^2 >= 0 and we minimize, so this is exact). |
| 31 | Guard on real dev labels (same classes as 29): if a channel's S readout has AUROC < .52 and lower than its top10 AUROC, revert that channel to top10 and report it. The channel stays in (no removal). |
| 32 | S2: bottom 30% of all dev C steps together (global), ranked by the new s_g (L-SML on G16 = global rank-normal of the S readouts, no answer-z). Add (mean corr^2 inside the proxy, on R0) with weight 1, the same weight as the V2 term. Keep S2 only if the dev check of S improves (CI > 0). |

## D. Position and length (questions 33-35)

| # | Answer |
|---|---|
| 33 | Yes. No position or length correction in any score, also not at answer level. Only reports: within-class corr of each channel with position and length (before / after S), and length-stratified AUROC. Residualization lost on PRMBench 4/4 (11-12). |
| 34 | A check only. Not a term in the objective. |
| 35 | Yes. 5 strata by number of steps, quantile edges from dev C answers, per benchmark. |

## E. Fusion (questions 36-38)

| # | Answer |
|---|---|
| 36 | Detect the groups again after the new readout, with the method-of-record defaults (detect_dependent_groups, residual method, K = 2..8). Do not freeze the top10 groups. Report the group assignments before / after. |
| 37 | Yes, all unchanged. I checked the code: covariance over all steps of the pool; within a group the leading eigenvector of the zero-diagonal covariance, sign by majority (spectral_utils/fusion_utils.py sml_fuse_signed); between groups the same on the group scores, no re-standardization (lsml_continuous); final sign by corr with the channel mean (stage8_auto_forced_k.fit); constant views dropped (lsml_continuous_guarded). Import these functions; do not rewrite them. |
| 38 | Per pool. Dev work: fit on dev C, apply unchanged to dev V1/V2. Final test: fit the label-free parts on test C (as the method of record does per cell) and apply to test V1/V2. Every choice that uses gold labels (lambda, stage keep/drop, guards) is frozen from dev. The readout parameters N_c are frozen from dev. Exception: the P0 parity check fits on the full C pool, as the stored numbers do. |

## F. Marking (questions 39-44)

| # | Answer |
|---|---|
| 39 | ceil. evaluate_collection.gap_counts: km = min(ceil(m/2), m-1); k = argmax of the gaps among the top km+1 sorted scores, + 1; m = 1 gives k = 1. |
| 40 | Drop the soft version. PRMScore needs binary marks. The gate is a rank rule: close the (1 - pi-hat) share of answers with the lowest answer score (no marks there); GAP marks the open answers. |
| 41 | Option b, changed: free EM with a PER-ANSWER prior P(tau = inf) = 1 - P_i(wrong) from the gate, held fixed (pi is not re-estimated, so no collapse). Emission densities f0/f1/f2: estimated by EM on the dev C pool, histogram on quantile bins with the monotone (isotonic) constraint, as in pb_first_error_rule_v1 (em(), cp_rule.py). Init: free_init (as stored). Small code change: em() takes pi as an array. Not option a: in V2 the steps before k can have natural errors and the steps after k are the original continuation, so f0 and f2 would be wrong; f1 from a foreign step is the shifted distribution that the realism check tests. |
| 42 | The fused R0 step score s (1-D, the score GAP uses), B = 20 bins (as the PRM-anchored change point that worked). Not the 16 channels: a product over 16 channels assumes independence given the state, which is false here, and gives over-confident posteriors. The gate brings the answer level; the change point only places the error inside the answer. |
| 43 | Source: pb_first_error_rule_v1 RESULTS.csv, arm "FREE CP TELE16 standalone (EM)": input sa_N16 (16 R0 channels), B = 10, 30 iterations, pi re-estimated each iteration. Mind the Gap macro SLA 37.55 / 42.10 / 35.37 / 37.04 (7B / 72B / 7B base / Llama-8B). avg F1 = 0.00 in all 4: the EM collapsed and every answer got an error. With pi fixed at .5: F1 17.9 / 32.2 / 22.8 / 24.1, SLA 35.7 / 37.3 / 34.1 / 37.2. Hazard h: h_from() = maximum likelihood of a truncated geometric over the free_init first-flag positions, grid linspace(.01, .6, 60), label-free, fixed during EM. |
| 44 | No P >= .5. The same rank rule as answer 40: close the (1 - pi-hat) share with the lowest answer score; GAP in the open answers. This is the "GAP + gate" row next to the change-point row. |

## G. Answer level, stage A (questions 45-50)

| # | Answer |
|---|---|
| 45 | The stage S readout (top10 if stage S is dropped). |
| 46 | One arm: ECDF per channel, fitted on dev C step values (raw readout values, not R0), applied to all. Digit channels: ECDF over active steps, inactive steps = 0. No RAW arm. |
| 47 | Yes. Count-based soft top-N over the step ECDF values of the answer, one N per channel, all 16 channels; inits N in {1, 3, 10}. Then standardize each answer feature with mean / SD of dev C answers, and fit answer-level L-SML (16 aggregates + Simes = 17 views) on dev C answers. Objective: the same form as stage S: sum of corr^2 over the 136 pairs inside V2 answers - lambda x sum of (paired AUROC - .5), where the paired AUROC compares each V2 answer with its own parent C answer. |
| 48 | Simes of s_g (L-SML on G16, no answer-z), as in simes_answer_gate_v1, recomputed from the stage S readout; step p-values against the dev C steps. Not of R0 s: answer-z gives every answer mean 0 and removes the answer level. |
| 49 | Stratifier only (length strata of answer 35). Not a view. V2 and V1 have the same step count as their parent, so log(n) has the same distribution in the synthetic and the real pool: it cannot help BBE, and as a view it only adds a length prior that breaks the independence assumption. |
| 50 | Fit on V2 only. V1 is the held-out check (answer AUROC, pi-hat with V1 as positive). |

## H. Gate, stage G (questions 51-55)

| # | Answer |
|---|---|
| 51 | BBE (Garg et al. 2021) on the stage-A answer score. No PU classifier. Cross-fit: 2 folds by corrected group inside the pool; fit stage A with fold-1 synthetic answers, run BBE with fold-2 synthetic answers, and swap; average the two pi-hat values. Why no classifier: a classifier "synthetic vs pool" learns the artifacts of the synthetic errors, so BBE would estimate the share of answers that look synthetic, not wrong. |
| 52 | V2 is the primary positive. Report V1 as positive too, as a check. |
| 53 | P_i(wrong) = min(1, pi-hat x f_P(s_i) / f_U(s_i)) (the PU posterior). Estimate f_P / f_U with an isotonic (monotone) regression of "synthetic = 1, pool = 0" on the answer score, on a balanced, cross-fitted sample: f_P/f_U = [g/(1-g)] x (n_U/n_P). Under the PU model the mean of P on the pool is pi-hat before clipping; report the realized mean. P is used for the PB change-point prior and the F1 histograms; the marking rule uses pi-hat only (answers 40, 44). |
| 54 | Per pool, with the synthetic answers of that pool: dev pi-hat for dev work, test pi-hat for the test result. The check abs(pi-hat - pi) <= .10 uses the test estimate. |
| 55 | C2ST = logistic regression (sklearn, L2, default C), 5-fold cross-validation with folds by corrected group. Step level: V2 step k vs real wrong dev steps (PB first-error steps; PRMBench wrong steps), features = the 16 R0 channel values; balance the two samples by token-length quintile. Answer level: V2 answers vs real wrong dev answers, features = the 17 answer features; balance by step-count stratum. Pass: AUROC <= .65 and Spearman agreement of the 120 off-diagonal correlations (V2 step k vs real wrong steps) >= .5. If V2 fails: stop stages S, A and G for V2. Then A and G may use V1 as the positive only if V1 passes the answer-level C2ST; else report the shift and stop. |

## I. Evaluation (question 56)

| # | Answer |
|---|---|
| 56 | On test, only: the final pipeline (the stages kept on dev), the method of record (no gate), the baselines (equal mean of 16 channels + GAP, position only, length only, mean entropy, perplexity), the Simes reference, and the oracle references for F2 (best Simes alpha chosen with labels, label-perfect gate). Not every arm of every stage. |

## J. Next steps for the Mac (proposed order)

1. Wait for Omri's OKs (0.3 a-c, f).
2. Create the branch (answer 12). Write the MPS / chunk flags (answer 13).
3. 12-row timing run with the real model, longest rows first, all 6 inputs (C, V1, V2 x 2). Report s/row, peak memory, swap.
4. 12-row checks: equal prefix (raw tokens), PRMBench native path vs --answers, C vs stored (token channels: readout_steps_sa_v1; PB 16 channels: sa_N16).
5. Build the group map and the split (answers 6-9). Report counts. Then make_variants.py (answers 1-5, 10).
6. Full G1 runs: PB first, then PRMBench.
7. P0 parity on C. Then stages S, S2, A, G and marking on dev. Test once, at the end.
