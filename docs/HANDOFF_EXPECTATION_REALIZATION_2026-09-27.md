# HANDOFF: expectation vs realization fusion, SML binary properties for selection (Steps 450-452, 2026-09-27)

Read after `PROGRESS.md` and `LESSONS.md`, before planning anything on PRMBench fusion with SML / Dawid-Skene
estimates, discovered groups or L-SML on the 13-channel bank. It covers one Claude session (HISTORY Steps
450, 451, 452) on branch `claude/ssl-pseudolabel-residual-v1`, worktree `.worktrees/ssl-pseudolabel-residual-v1`,
plus review comments from Codex and a side Claude agent that changed how the results must be described.

All numbers are development evidence on the already-evaluated population (13,769 answers: 6,800
ProcessBench, 6,969 PRMBench, 6,030 PRMBench answers eligible for within-answer AUC). Nothing here is
untouched confirmation. Each number names its source folder under `results/expectation_realization_v1/`.

## 1. What Omri wanted, in his words

- Fuse classifiers of the model's EXPECTATION (bank11: level + change channels) with classifiers of the
  REALIZATION (the written token against a forecast): `chosen_surprisal`, `realized_z` (CT7's pooled
  de-spiked z-test), `realized_drv` (EMA-16 derivative of chosen_surprisal).
- Use the SML binary classifier properties (sensitivity psi, specificity eta; Parisi et al. 2014) to build
  "a decision rule better than an average over all features, with good benchmark scores", with groups found
  by the data, not by declared families. PRMBench decides; ProcessBench is reported beside; the winner goes to
  the external benchmarks later.
- Process: frozen protocol committed before scoring, independent subagent code review before the full run,
  three-agent red team after it, results reported only with their consensus, commits per stage.
  Chat in plain Hebrew, no U+200F/U+200E direction marks, numbers with a leading zero, no leading +/-.

## 2. Omri's decision at the end of the session

- **Keep** the label-free Dawid-Skene (DS) estimates + filtering as a first stage (section 4.1).
- **Keep** clustering on binary marks as an option worth studying (with the correction in 4.2).
- **Drop** weights computed from the estimates: they never added real value (section 4.3).
- Next: PRMScore decomposition (section 6.1) before any new algorithm change.

## 3. Runs, commits, artefacts

| Stage | Protocol (frozen commit) | Run folder | HISTORY | Red team |
|---|---|---|---|---|
| A: 13-channel fusion + are SML/DS estimates accurate? | `PROTOCOL.json` (73cca2308, amendment A1 a9ad01e88) | `run_20260927/` | Step 450 | `run_20260927/RED_TEAM.md` |
| B: DS filter -> discovered groups -> group DS weights | `PROTOCOL_STAGE_B.json` (2b3971383, amendment B1 7fa255e8f) | `run_20260927_stage_b/` | Step 451 | `run_20260927_stage_b/RED_TEAM.md` |
| B2: count the level family once (merge / remove 2 of 5) | `PROTOCOL_STAGE_B2.json` (028b313df, review fixes 9b57f4a09) | `run_20260927_stage_b2/` | Step 452 | `run_20260927_stage_b2/RED_TEAM.md` |

Code: `scripts/experiments/expectation_realization_run.py` (stage A), `er_stage_a.py` (estimators),
`er_stage_b.py` + `er_stage_b_run.py`, `er_stage_b2.py` + `er_stage_b2_run.py`; tests
`tests/test_er_stage_a.py`, `test_er_stage_b.py`, `test_er_stage_b2.py` (18 known-result tests, all pass).
Label-free diagnostics: `results/expectation_realization_v1/stage_b_diagnostics/`.
Local only (gitignored): every run's `STEP_SCORES.npz` and `BOOTSTRAP_DELTAS.npz`. Inputs are listed with
sha256 in each `INPUT_MANIFEST.json` (depth-feature-fusion-v1, token-probability-fusion-v1,
cumulative-vote-fusion-v2, readout-quickest-detection-v1 worktrees). Runtime: stage B2 698 s CPU (TIMING.json).

## 4. What holds, with the exact comparison each number refers to

PRMBench within-AUC / official PRMScore / ProcessBench SLA (macro 8 cells, gate-free).

| Row | within-AUC | PRMScore | PB SLA | Algorithmic? |
|---|---:|---:|---:|---|
| Average of all 13 channels | 0.7749 | 0.6527 | 0.3698 | yes |
| DS filter, then average of the 11 survivors (S_equal) | 0.7802 | 0.6565 | 0.3750 | yes |
| Simple filter (drop channels negatively correlated with the mean of the others), then average | 0.7803 | n/a | n/a | yes (red-team diagnostic, stage B) |
| DS filter + discovered groups + DS group weights, level split in two groups (stage B) | 0.7729 | 0.6492 | 0.3639 | yes |
| Same rule, level groups MERGED (stage B2) | 0.7818 | 0.6587 | 0.3788 | **no: the merge uses a hard-coded list of the five level channels** |
| Same rule with TRUE psi/eta (label-using diagnostic, not a ceiling for all weightings) | 0.7823 | 0.6607 | 0.3782 | label-using |
| Label-using weighting aware of between-group correlation (LDA on 4 groups, red team) | 0.7890 | n/a | n/a | label-using |
| Continuous L-SML on the 11 survivors | 0.7751 | 0.6491 | 0.3594 | yes |
| fam421 / CT7 | 0.7801 / 0.7724 | 0.6573 / 0.6458 | 0.3980 / 0.3989 | references |

### 4.1 The DS filter (keep, with caveats)
- Drops exactly `energy_innovation` and `top50_js` in 5/5 folds in both stage B and B2, and exactly the
  channels whose true balanced accuracy on the same rows is <= 0.5. Label-free. Tie keys do not change it.
- Gain over averaging all 13: +0.0053 within-AUC [0.0033, 0.0072], +0.0038 PRMScore; 5/5 folds, 7/8 classes.
- Caveats: (a) the whole gain is dropping `energy_innovation`; dropping `top50_js` adds nothing. (b) A trivial
  label-free rule (negative correlation with the mean of the other channels) finds the same channel and the
  same score, so the unique value of the SML/DS estimate over a simple filter is NOT established.
  (c) On the original bank a whole-answer label swap between answers of equal step count reproduces the gain,
  so this test cannot show value beyond error position; on the position-adjusted bank a +0.0043 gain
  survives (swap-null z 4.3, 5/5 folds). Both facts hold; "the filter is entirely positional" is too strong.
  (d) The 0.5 cut has no margin: a within-answer-permuted pure-noise channel is kept (pi_hat 0.502-0.507).

### 4.2 Clustering on binary marks (option, corrected)
- It is NOT established that binary-mark clustering is better than continuous clustering. What is true:
  - On all 13 channels, L-SML grouping of TIE-AWARE (fractional) top-20% marks over all fit-fold steps gives
    exactly the three families (level 5 / change 5 / realization 3) in 5/5 folds. With random-tie binary
    marks it gives 6 groups; continuous L-SML gives 5 groups (level split in two, bocpd_p0 alone). The
    three-family result depends on tie handling (change channels have many boundary ties).
  - After the DS filter, the tie-aware binary partition splits level into the same two sub-groups as the
    continuous one ({q15_H1, q15_VE1, logprob_margin} | {true_tail50, energy_level}) and keeps
    {top15_turnover, dominant_freq16, bocpd_p0} together; identical in 5/5 folds. Random-tie partitions
    change after the filter.
  - No experiment compared binary vs continuous partitions at equal weighting. That is an open, cheap test.

### 4.3 Weights from the estimates (dropped)
- With the level family split in two groups the DS weights hurt (0.7729 < 0.7749): the two level sub-groups
  are conditionally dependent (clean-step mark correlation 0.52), so DS overweights them about x3.
- With the level merged, the rule reaches 0.7818, but over the plain average of the same 11 channels it adds
  only +0.0016 within-AUC [0.0005, 0.0027] (5/5 folds, both nulls) and the gain is concentrated (87% of
  answers unchanged, 5%-trimmed +0.0008). Across the ten removal configurations the rule minus the plain
  average of the same 9 channels is +0.0001 (median; 6/10 positive, two of them below 0.0001).
- "Rule beats equal GROUP weights" (+0.0031, 10/10) is only down-weighting the incoherent change group:
  equal group weights are below plain channel averaging, a fixed 0.22 change weight does better.
- Estimates remain biased (prevalence 0.225-0.240 vs true 0.137-0.141). Normalized weights depend on the
  ratios of the est/oracle factors; after the merge these ratios are more similar (max/min 1.4-1.5 vs 2.5),
  which is why the rule stops hurting. The inflation does not "cancel"; it became more uniform. The weakest
  group is identified in 60/60 folds; level is ranked above realization in 5/5 merge folds (wrong order).

### 4.4 L-SML (important to Omri)
- Continuous L-SML never beat plain averaging on this bank: all 13 (0.7765 vs 0.7749 n.s.), survivors
  (-0.0050), after removing two level channels (-0.0073 median, 0/10), merged partition passed as groups
  (0.7788 vs 0.7802). Its cross-group SML step assumes independent groups and counts the two level
  sub-groups twice (level weight share 52%, realization 16% on the survivors).

### 4.5 Correct decomposition of the stage-B2 gain (Codex review: my first version double-counted)
- Merge path: 0.7818 - 0.7749 = +0.0069 = DS filter (+0.0053, S_equal - all-13) + rule over the plain
  average of the same 11 channels (+0.0016). Merging repairs the RULE (0.7729 -> 0.7818); it adds nothing to
  plain averaging, which never split the level family.
- Removal path (medians over ten configurations; medians do not add): rule - all-13 +0.0071; plain average of
  the 9 channels - plain average of the 11 survivors +0.0028 (not positional by the swap null); rule - plain
  average of the same 9 channels +0.0001.
- The earlier report's "filter + level-once + weights" sum mixed the two paths.

## 5. What is manual, post hoc or not algorithmic (must be stated whenever these numbers are cited)

- **The level merge and the ten removals use a hard-coded list** `LEVEL = [q15_H1, q15_VE1, logprob_margin,
  true_tail50, energy_level]` (`er_stage_b2_run.py`). The protocol justified it as "members of the two level
  sub-groups of the stable partition", which is true in every fold, but the code does not discover it. B2 is
  therefore a MECHANISM test ("what happens when level is counted once"), not an automatic method. This was
  not flagged in the first report of 0.7818.
- **Amendment B1** (tie-aware partition recipe) was added after the fold-0 smoke of stage B had been seen.
- **rm_H1_LM** (0.7864) is the best of ten configurations, i.e. label-selected; never report it alone.
- **Cohesion weight** (group weight proportional to lambda_1/p of the members' correlation matrix) reached
  0.7823 on the merge, post hoc, computed by a red-team agent, and on top of the manual merge. It is a hint,
  not a candidate. Codex's caution: lambda_1/p measures concentration of variance in one direction, not
  reliability or error; it depends on group size and composition; cohesion cannot tell "agree because of the
  error" from "agree because of a nuisance" (the level cartel).
- **Automatic merge by covariance residuals** (side-agent idea: fit the conditional-independence model on
  group representatives, merge the pair with the largest off-diagonal residual): not established. Under
  conditional independence only the OFF-diagonal part is rank one; a large residual does not identify the
  responsible pair with 3-4 groups; merging the most correlated pair may merge two accurate groups whose
  agreement comes from the truth. An open research question, not a missing implementation detail.

## 6. Next steps (ordered)

1. **PRMScore decomposition (Omri + Codex, next task).** On existing scores (no new fitting), compare
   all-13 average, DS filter + average, simple filter + average, the stage-B2 merge rule, L-SML, fam421 and
   CT7 by: official PRMBench categories; official PRMScore components (positive F1, negative F1); answer
   length and first-error position; paired per-answer changes (improved / worse / unchanged). Report N and
   paired source-group intervals in every cell. Note: per-fold PRMScore thresholds are not saved in
   STEP_SCORES; recompute them from the fold models (fixed-weight arms are trivial; the merge rule needs the
   per-fold partitions and weights from `FIT_MANIFEST.jsonl`) or rerun the runner with the thresholds saved.
   The simple-filter arm must be computed the same way (per fold, label-free).
2. **Paired PRMScore interval for the merge rule minus the plain average of the same 11 channels** (Codex:
   the +0.0016 within-AUC claim needs the direct paired test on the target metric). Same threshold issue.
3. **DS filter vs simple filter**, frozen, on PRMBench and on the external benchmarks: the unique value of
   the SML estimate for selection is the open claim.
4. **Binary vs continuous partition at equal weighting** (section 4.2), frozen, one comparison.
5. External transfer (Codex: ProcessBench alone is not a reason to refuse): if done, ONE frozen algorithmic
   candidate (DS filter + average) against the matched all-13 average, the simple filter and the existing
   external reference, on the external pipeline (`codex/lsml-external-generalization-v1`). The merge rule is
   not algorithmic and cannot be sent.
6. Only after that: an automatic merge rule or the cohesion weight, each as its own frozen single-variant
   protocol, compared with the plain average of the same channels (LESSONS 2026-09-27).
7. Still open from Step 450: re-score the PRMScore numbers of the Step 438-440 runners
   (`bank20_lsml_run.py`, `indbank_lsml_run.py`, `declared_joint_run.py`), which have the
   evaluation/calibration overwrite.

## 7. Traps (all in LESSONS.md, 2026-09-27)

- Copying a runner copies its role bugs: every fold-k model writes only its evaluation rows; thresholds from
  the same model on the calibration fold; write-once assert (`put()` in all stage runners).
- Every within-answer gain needs BOTH nulls (within-answer shuffle AND whole-answer swap within equal step
  count) plus a position-adjusted replicate; report step index alone (0.6617) as a reference.
- Compare a new weighting with the plain average of the SAME channels, not only with a structured control.
- A diagnostic that changes by construction (a merged pair deleted from a correlation table) is not evidence.
- Re-run every label-free premise on the exact frozen input recipe before freezing (tie handling changed the
  partition between stage A and B).
- Say which reference a number is relative to; do not add components computed on different paths.
- A red-team script that reads `np.load(npz)[key]` inside a per-answer loop decompresses the array each time
  (38 minutes wasted); load arrays once.
