# HANDOFF: PRMBench fusion levers, Mind the Gap reproduction and open leads (2026-09-23 to 2026-09-27)

This handoff is for the next research session. Read it after `PROGRESS.md` and before planning anything on
PRMBench fusion, bank extensions or Mind the Gap. It covers one Claude session: HISTORY Steps 437-440 and
447-449 on this branch, three runs relayed to background agents, and several questions answered in chat
without an experiment.

All numbers are development evidence on the already-evaluated population (13,769 answers: 6,800
ProcessBench, 6,969 PRMBench, of which 6,030 are eligible for within-answer AUC). None of them is
untouched confirmation. Each number names its source file.

**Project direction check.** The CLAUDE.md update of 2026-09-24 makes external telemetry (Hard2Verify,
Socratic-PRMBench, branch `codex/lsml-external-generalization-v1`) the current execution priority. This
session was the PRMBench source-side line under the 2026-09-23 decision (PRMBench primary, L-SML must add
value over equal fusion on the same bank, CPU only). The external line has shown three times that source
rank does not predict external rank, so nothing below is a candidate until it passes the external pipeline.

## 1. Where things are

| Branch | Worktree | Commits | Contents |
|---|---|---|---|
| `claude/ssl-pseudolabel-residual-v1` (pushed, remote tip 28787a563) | `.worktrees/ssl-pseudolabel-residual-v1` | 92163f5f1 (S5-CPU), fde3e79e9 (bank20), 2e88031ad (independent channels), 07ff827cc (declared Joint), 037c1d5d2 (partition ceiling), 7b208ee5f (Mind the Gap code audit), 28787a563 (Mind the Gap reproduction) | Steps 437-440, 447-449; each result folder has a frozen `PROTOCOL.json` committed before scoring |
| `claude/lsml-ct7-levers-v1` (pushed; local branch `lsml-ct7-levers-run`) | `.worktrees/lsml-ct7-levers-run` | 93ece8f4a (family-equal), 87fa781fe (token-level L-SML + window representation), b016ef6a1 (PRM vs CT7) | Runs done by background agents at Omri's request; HISTORY blocks tagged `[local][ct7-levers]` and `[local][prm-measure]` |

The same local branch also holds another session's Step 447 (`pb_tail_weights_v1`, commits
4cae70d8a/9ebe914c2), which is not part of this handoff.

**Step-number collisions (tagged, never renumber):** Step 437 is both `[Claude, SSL plan S5-CPU]` and
`[local][prm-measure]`; Step 447 is both `[Claude, partition ceiling]` and the other session's PB tail
weights; Step 434 is both `[Claude, SSL plan S1]` and `[local][ct7-levers]`.

### Local-only artifacts (gitignored `results/**/*.npz`; back up to Drive before removing a worktree)

| File | Size | Why it matters |
|---|---:|---|
| `results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz` (main checkout) | 447 MB | CT7's seven streams per token, all 6,968,779 tokens; took 1,118 s to extract |
| `.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz` | 19 MB | Holds the `chosen_surprisal` derivative, the strongest untested lead (section 4) |
| `results/partition_ceiling_prmbench_v1/run_20260927/EXHAUSTIVE_AUC.npz`, `BOOTSTRAP_DELTAS.npz` (this worktree) | 6 + 11 MB | Scores of all 237,018 partition profiles |
| `results/mtg_reproduction_v1/TOKEN_SIGNALS.npz` (this worktree) | 88 MB | Four Mind the Gap signals per ProcessBench token; rebuilt in 20 s by `mtg_reproduction_extract.py` |

## 2. Reference scoreboard (PRMBench within-answer AUC, 6,030 eligible answers)

| Row | Access | Within-AUC | Source |
|---|---|---:|---|
| Qwen2.5-Math-PRM-7B | **supervised** external verifier, reference only | .8012 | `prm_vs_ct7_prmbench_v1/MEASUREMENT.json` |
| `chosen_surprisal` derivative, single channel | label-free, no protocol yet | .7957 (.7707 with step length regressed out) | `DERIVATIVE_CHANNELS.npz`, section 4 |
| CT7 declared family partition 4/2/1 (`fam421_answer`) | label-free | .7801 | `ct7_family_equal_v1` |
| Best block-equal partition of bank11, exhaustive | **label-selected** ceiling | .7746 (held-out selection .7737) | `partition_ceiling_prmbench_v1` |
| CT7 (seven views, equal) | label-free | .7724 | frozen anchor, replayed exactly in every run |
| bank11 continuous L-SML | label-free | .7645 | `bank20_lsml_prmbench_v1` (fit-on-4 replay) |
| `energy_level` alone | label-free | .7612 | Step 438 diagnosis |
| bank11 equal | label-free | .7496 | same |

CT7 on ProcessBench: 39.89 SLA (macro over 8 cells, gate-free) / 41.19 F1.

## 3. What was tested, and the verdict

Each row had a frozen protocol and full-population scoring with paired source-group bootstrap intervals.

| Question (Omri) | Step, folder | Answer | Key number |
|---|---|---|---|
| Can learned attention inside a step (S5 without the GPU encoder) replace the frozen Top-k readout? | 437, `ssl_pseudolabel_residual_v1/S5/` | **No.** The per-channel top-5 order statistic carries localization; one shared softmax cannot express per-channel top-k | -9.23 pp PB, -.044 within vs the frozen readout |
| Does a family-equal readout of CT7's views help? | `[local][ct7-levers]`, `ct7_family_equal_v1/` | Only on PRMBench within-AUC; nothing on PB | fam421 - CT7 +.0077 [+.0055, +.0100]; PB SLA on 11+-step chains +0.36 pp [-3.13, +3.75] |
| Does L-SML on CT7's token streams beat equal weights? | `[local][ct7-levers]`, `ct7_token_lsml_v1/` | **No**, and worse on long chains | -0.83 pp PB [-1.93, +0.25]; -.0125 within (interval excludes 0) |
| (same run) side result | same | Building the 7th view CT7's own way at token level, then equal weight | `ct7_top10_equal7` +1.15 pp PB SLA over CT7 [+0.06, +2.25], Holm .79; within .7750 |
| Does a sliding-window representation help? | `[local][ct7-levers]`, `window_representation_b3_v1/` | **No.** The whole window family is about 10 pp below CT7 | best window arm 29.70 PB vs 39.89 |
| How does a supervised PRM compare with CT7 on PRMBench? | `[local][prm-measure]`, `prm_vs_ct7_prmbench_v1/` | PRM higher on average, but the two find different errors | +2.88 pp within [+2.14, +3.60]; union of argmax hits 79.9 %, PRM worse on redundancy (-19.6 pp), circular and domain-inconsistency |
| Can bank11 grow toward 20 channels while keeping the L-SML edge? | 438, `bank20_lsml_prmbench_v1/` | **No.** The bank11 L-SML edge is de-noising by one fixed partition that silences three anti-oriented channels; added channels break it | 20-channel L-SML -.0042 vs bank11 L-SML; edge over equal .0149 shrinks to .0071 |
| What about weak channels whose errors are independent of entropy? | 439, `indbank_lsml_prmbench_v1/` | **No.** L-SML rewards their independence without the accuracy to back it; orientation is not the issue | -.0921 within vs bank11 L-SML |
| Declared partitions and Joint L-SML on the step bank; Omri's four chosen channels | 440, `declared_joint_prmbench_v1/` | Declared partition hurts bank11; Joint on it gains; the four channels pay only under Joint. Nothing reaches CT7 | Joint - block-equal +.0130 [+.0088, +.0172]; best new arm .7587 |
| Why don't we find the optimal partition? | 447, `partition_ceiling_prmbench_v1/` | No rule we own searches for it. The optimum is stable, has a singleton that Joint forbids, and is worth little | ceiling .7746; held-out selected - CT7 +.0013 [-.0039, +.0066] |
| What is in the Mind the Gap code? | 448, `docs/reviews/MIND_THE_GAP_CODE_AUDIT_20260927.md` | No step-level or ProcessBench code; the release does not implement the published method (top-1 log-prob plus an undocumented running mean) | running mean weights token 20 21x over token 500 |
| Can we reproduce Mind the Gap's Table 3? | 449, `mtg_reproduction_v1/` | Pattern yes, numbers approximately, exact reproduction impossible (Table 3 is not k/N over one population) | Shannon Drop 37.00 vs 39.26 (mean deviation 3.33 per cell) with the published method; released code 18.42 away |

**Two methodological conclusions from the partition work (Steps 438-440, 447):**

1. L-SML's edge on bank11 is suppression of harmful channels, not exploitation of independent ones. Before
   extending any bank, read the fitted weights and each channel's single-stream AUC.
2. Joint L-SML needs K >= 3 and every group >= 3 (`fit_joint_lsml` raises; `discover_loao_consensus_groups`
   has a minimum group size of 3). This is why the baseline script never ran it on bank11, and why it cannot
   express the optimal structure (a channel standing alone).

## 4. Findings from chat questions (no experiment run)

- **Predicting the next token's uncertainty (full inventory in memory `project_prediction_residual_family_2026_09_27`).**
  The member that worked is the chosen-token z-test, CT7's 7th view: the model's own next-token distribution
  is the predictor (entropy = predicted surprisal, varentropy = its predicted variance). Fixed,
  parameter-free predictors beat learned ones on within-AUC even when they predict worse; Step 387's
  conclusion: "prediction loss cannot select semantic quality". Every learned member failed (S2 ridge -1.54 pp,
  AR(1) views -6.66 pp, C8 -0.51 pp, S5 attention -9.23 pp).
- **Why the z-test is not in bank11 or Family15.** It entered both, but always in the weak form: Top10 of the
  per-token `chosen_std_excess` (.6495 alone). CT7's actual 7th view is `despiked_chosen_token_z`, the z-test
  pooled over the step with step 0 neutralized (.7434 alone). The strong form was never tested in either bank.
- **Omri's self-supervision idea** (fuse at token level, learn to predict the next fused score, read out the
  residual). Assessment given: a learned predictor tends to "explain" the error once scores start rising, so
  the residual shrinks where the signal is; the zero predictor (the residual equals the level) is a strong
  baseline. The version most consistent with what worked: a **fixed** predictor with a predictive variance
  (BOCPD's Student-t predictive) on the fused token series, a pooled per-step z with step 0 neutralized,
  added **as a channel next to the level**. A learned version needs a variance output, low capacity, training
  on other answers only, and zero / running-mean / shuffled-order controls. Nothing was run.
- **The `chosen_surprisal` derivative** (EMA-16 of the realized token's surprisal, mean of the 3 sharpest rises
  per step) is the strongest single PRMBench channel found: .7957 vs its level .6975. Controls: step length
  alone .6181, length regressed out .7707, argmax on step 0 in only 0.7 % of answers, without step 0 .7748.
  It sat unexamined because the derivative experiment was judged only as a fused arm on ProcessBench.
  Caveat: best of 22 single-channel numbers, no bootstrap yet.
- **Mind the Gap on `true_tail50`** (Omri's question): already computed, slightly worse than the level
  (.7281 vs .7376). Their evidence series correlates .9996 with our `q15_H1`; what they cannot see is which
  token was realized.

## 5. Open leads, in the order I would take them (each needs Omri's go)

1. **Add CT7's pooled de-spiked z-test to bank11 as a channel.** Frozen protocol, same recipe and splits as
   Steps 438-440; the column already exists in CT7's frozen profiles, so this takes minutes. Caveat from
   Step 417: on the 20-stream bank most of its gain came from the position prior.
2. **Add the `chosen_surprisal` derivative as a channel** to bank11 and to CT7, with length and position
   controls and paired intervals. It is the largest untested single-channel number we have.
3. **`ct7_top10_equal7`** (+1.15 pp PB over CT7, Holm .79): a view-construction result that needs its own
   pre-registered test before anyone reads it as a gain.
4. **Omri's self-supervision idea, fixed-predictor version** (section 4). The fused token series already
   exists on disk for all 6.97 M tokens, so no new extraction is needed.
5. **PRM vs CT7 complementarity**: the PRM is better on semantic error types, CT7 on structural ones, and the
   union of hits is 79.9 %. Fusing with a supervised PRM changes the access class and was not authorized;
   decide whether a descriptive follow-up is worth it.
6. **Mind the Gap**: ask the authors for the ProcessBench script and prompt. It is the only thing that can
   close the remaining gap (OlympiadBench and Omni-MATH: ours about 34-35, theirs 37-43).
7. **S2 v1.1** (per-channel, per-answer standardized residual readout with a zero-prediction control) is
   registered and never run.

Anything that looks good on source must then go through the external pipeline before it is called a
candidate.

## 6. Traps to avoid

- **Mind the Gap is three different objects**: the published method, the released code and the reported
  numbers. Say which one a comparison uses. The best-matching configurations in Step 449 were chosen to match
  their table (8 numbers per column): they are candidate readings, not a method and not a fair comparator.
- **Counting only traces where a detector fires inflates SLA** from about 15 % to about 45 % (Step 449,
  first-q0.95 fires on 29-39 % of traces). Keep all erroneous traces in the denominator.
- **Partition ceiling (Step 447): the shuffled-label null is invalid.** With the pair set fixed, a within-answer
  label shuffle turns the objective into roughly its own negation, so it lands on the distribution minimum.
  Use the distribution median (.7455) as the random-partition reference.
- **Declared Joint (Step 440) is marked INCOMPLETE only because** a fit-on-4 replay deviates 1.28e-9 against a
  1e-9 tolerance (rounding from standardizing twice). No fit failed.
- **Exactness gates must compare in the stored dtype.** The first token-level L-SML attempt stopped because
  the gate cast a float64 bank to float32.
- **Token spans differ by source**: `step_token_spans` in the ProcessBench pickles are answer-relative; the
  29-stream historical bank (`localization_full_benchmark_v3/inputs`) stores absolute spans.
- **Shell**: heredocs that contain quotes fail with `unexpected EOF`, and then nothing in the chain ran.
  Write files with the Write tool and commit with `git commit -F`.
