# Which telemetry channels, and why: reasoning for the view-selection candidates (2026-10-26, revised)

## Status

- **A first version was written earlier today.** It was revised after Omri's review the same day, which asked
  for five things:
  1. Take into account the fork's finding that on Socratic the channels track step position, and its
     position-neutralization probe.
  2. Drop Math-Shepherd and work on the four other PRMs. This is Omri's decision.
  3. Explain what the pipeline does: GAP binarization, SML, HEM or L-SML, and clustering.
  4. Report how the channels perform alone and fused with the PRM.
  5. Ground the reasoning in Jaffe, Fetaya and Nadler (2016) and in error types, TPR/FPR and the competitor
     protocols.
- **Everything is development data, in-sample.** The stage-1 and stage-2 subsets were selected with PRMBench and
  Socratic labels.
- **New computations, all on CPU from existing caches:**
  - A descriptive evaluation, pre-declared before it was run in
    `results/prm_view_selection_v1/reasoning/PREDECLARATION_8CELL_EVAL.md` (sha256 recorded in `RULE_POS.json`).
    It ran no search.
  - A second pre-declared run: redundancy errors under SLIDER's protocol
    (`PREDECLARATION_REDUNDANCY_SLIDER.md`).
  - Post-hoc diagnostics, each labelled as such (§7, §7a, §3a).
  - Re-reads of the existing search files.
- **Later questions from Omri, answered the same day:**
  - position as a separate channel (§7a);
  - whether Socratic is a first-error benchmark (§3a, checked against its paper);
  - redundancy against SLIDER (§3a).
- **Parity checks:**
  - The position-neutral representation reproduces the fork's probe exactly in 18/18 subset-cells.
  - The competitor protocols reproduce `stage2/COMPETITORS.csv` exactly in 20/20 values.
  - Replays match `SEARCH.npz` in 31/31 values.
- **Scripts and outputs:** `results/prm_view_selection_v1/reasoning/`. They are listed in §13.

## 1. In brief

1. **Position explains the Socratic "wins".** On Socratic the error steps sit late in the answer: position alone,
   used as a score, reaches within-answer AUC .731 (PRMBench: .662). Stage 1 orients every channel by the sign of
   its correlation with the PRM. That turns the weak "dynamics" channels into position proxies: on the
   strong-PRM Socratic cells they correlate +.24 to +.60 with position. Together with the PRM, which also scores
   later steps as riskier, they keep DS-EM's latent class on the PRM.
   - 8 of the 9 subsets that beat both the PRM and their own mean in all 8 cells contain the same three dynamics
     channels (top15_turnover, dominant_freq16, bocpd_p0), against 7.8% of valid subsets.
   - When position is removed from the channels, the PRM's weight on those cells falls from .70–.77 to .15–.26.
     SUB5 then loses all three cells (−2.3 to −4.0).
2. **Without position, Jaffe's assumptions hold much better.**
   - Every channel is better than random in every cell; in the raw representation up to 6 channels per cell were
     below chance.
   - The PRM plus the rule's set fits Lemma 1's rank-one covariance closely: misfit ≤ .046 in every cell, against up
     to .33 raw.
3. **The label-free rule recovers SUB5's structure.** The rule is Jaffe's selection heuristic, with accuracy ranked
   by covariance with the PRM's vote, applied after position is removed.
   - Pooled over cells it picks S_R = {q15_H1, chosen_surprisal, top15_turnover, digit_spread}.
   - Cell by cell, chosen_surprisal wins its family in 8/8 cells and digit_spread in 8/8.
   - The dynamics slot has no stable winner: it rotates among top15_turnover, dominant_freq16 and bocpd_p0, which are
     within about .02 of each other. Those three are exactly SUB5's dynamics channels.
4. **With position neutralized, no evaluated set beats the PRM in all 8 cells.**
   - All sets beat the PRM on all 4 PRMBench cells and on Socratic RLHFlow (+11 to +12 for SUB5 and C3).
   - All lose on the three strong-PRM Socratic cells.
   - SUB5 still beats its own mean in 8/8 cells with the PRM and 7/8 without it.
5. **Why it fails there: weighting, not channel choice** (post-hoc diagnostic).
   - The only gain available on those cells is +0.02 to +0.51 PRMScore, and it requires the PRM to hold at least
     35–80% of the weight, depending on the cell.
   - Label-free DS-EM gives the PRM 13–26%, because it overestimates every channel's accuracy by +.10 to +.18.
   - Even the paper's Oracle-CI, the same rule given the true accuracies, gives the PRM only 30–36% and loses to
     the PRM.
6. **Error types.** Every fused set gains on the PRM's weakest PRMBench categories, redundancy (+8 to +22) and
   circular logic (+4 to +12). It loses on deception (up to −10), confidence, counterfactual and multi-solution
   errors.
7. **Position cannot be added back as an SML voter (§7a).** Removing position from a channel also removes the part of
   the error that position predicts. Within each class, the residualized channel is then negatively correlated with
   position: 60 of 64 checks. A position voter therefore violates Lemma 1 by construction. Position belongs in the
   prior P(error | position) of a latent-class model. An oracle check shows it is informative (+1.2 on Socratic
   Qwen-7B over the PRM alone), but it does not fix the weighting bottleneck.
8. **Socratic is used correctly (§3a).** Its paper (arXiv 2505.23474) builds it like PRMBench and scores it with
   step-level PRM-Score. Our labels are its `error_steps`.
   - Its position signal comes from design: where an error sits depends on the perturbed reasoning pattern. Position
     alone reaches .92–.98 in "integration" categories and .31–.51 in "repeat" and regather_redundancy.
   - Our native-threshold scores reproduce the paper's (Qwen-7B PRM 68.09 against 68.0).
9. **Redundancy under SLIDER's protocol (§3a).**
   - **PRMBench:** every telemetry arm beats SLIDER's 61.9 accuracy in all 4 PRMs (66–78). The value comes from the
     telemetry; the PRM is near chance in balanced accuracy (49–62).
   - **Socratic regather_redundancy** (position carries nothing there): position-neutral SUB5 + PRM beats the PRM in
     all 4 PRMs (balanced accuracy 86–89 against 55–85).
   - **Late patterns** (premise, integration): position and the PRM dominate.

## 2. What the pipeline does (Omri's questions)

| Question | What we did | Where it departs from the paper |
|---|---|---|
| **Binarization** | Yes. Every view, the PRM included, is turned into a per-step ±1 vote by GAP within each answer: sort the answer's answer-z values and flag the top k at the largest gap, with 1 ≤ k ≤ min(⌈n/2⌉, n − 1). SML and Dawid–Skene are defined for binary classifiers, and Lemma 1 is a statement about binary votes. | The paper assumes i.i.d. instances. Our steps are nested in answers, and GAP flags at least one step in every answer, even a fully correct one. |
| **Grouping** | L-SML Algorithm 1 on the vote covariance: score matrix (Eq. 15), spectral clustering, K chosen by the residual of Eq. 14. For subsets of at most 4 channels, all partitions are enumerated exactly. | As in the paper, K is searched over 2..m−1, so the conditionally independent case (each channel its own group) is excluded by construction. |
| **Parameters** | SML within groups (a group's virtual vote is the sign of its weighted sum); two-coin Dawid–Skene EM between groups, with the PRM as its own group and label switching resolved by the PRM. | The paper's Algorithm 2 allows any Dawid–Skene method. |
| **Prediction** | Primary: the estimated weights are applied to the continuous answer-z values, then GAP. Secondary: the weighted binary vote. | The paper predicts by maximum likelihood (Eq. 19), which is non-linear in general. For a set with one channel per group (K = m), Eq. 19 reduces to the linear log-odds vote. |
| **HEM** | The full latent-group EM of the 2016 model was pre-registered as a secondary estimator. It ran in stage 1 only, on 300 subsets: 0 successes in 10 cells and 0 in 8 cells (best min-margin −0.65). It was not run in stage 2. | — |
| **Clustering with 4 channels** | It was done: every subset in the searches was clustered, including 4-channel subsets. You are right that it is not meaningful there (next paragraph). For the sets proposed here (C3, S_R) there is no clustering on the small set. The groups come from the 16-channel bank, and the selected channels are fused as conditionally independent (F_CI: each channel its own group, Dawid–Skene). | — |

**Lemma 4 makes the clustering point precise.**
- Within-group separation in the score matrix comes only from 2×2 determinants whose other two indices lie in two
  other groups.
- In the symmetric case, the within-group lower bound is proportional to m²(1 − 3/K)·δ²·|c_ij|. It vanishes for
  K = 3, and no such determinants exist for K ≤ 2.
- So the guarantee needs at least 4 groups. With 4 channels, Algorithm 1 can only try K = 2 or 3. That is why every
  "one channel per family" set of 4 was rejected by stage 1's K ≥ 3 rule.

**Identifiability, which the paper does not state as a lemma:**
- **Dawid–Skene needs at least 3 conditionally independent voters.** The two-coin model with K voters has 2K + 1
  parameters against 2^K − 1 observable pattern probabilities. That gives 3 < 5 for K = 2 and 7 = 7 for K = 3
  (Kruskal-type identifiability; Allman, Matias and Rhodes 2009).
- **SML's rank-one completion needs m ≥ 3 as well.**
- **Consequence for C3.** C3 alone has only two independent groups: q15_H1 and chosen_surprisal share the entropy
  term (§4). So C3 cannot be identified as a telemetry-only ensemble, and the earlier proposal needs a third
  family.
- **Consequence for the K ≥ 3 rule.** This is also why stages 1–2 were right to require at least 3 telemetry groups.

## 3. Position

### Position as a score

Relative step position, used directly as a risk score (later = riskier), has within-answer AUC .662 on PRMBench
and .731 on Socratic. Source: `EXISTING_8CELL.json`.
- On Socratic, 34% of the steps in the pool are erroneous, against 16% on PRMBench.
- In 31% of Socratic answers that contain an error, every step after the first error is also labelled an error.
  On PRMBench the figure is 13%. This is not a "first error onward" labelling convention: §3a shows that the
  injected error block often reaches the end, because of where the perturbed reasoning pattern sits.

### How the channels and the PRM relate to position

Source: `POSITION_CORR.csv`. Values are correlations with relative position within the answer, after stage 1's
PRM-anchored orientation.
- **The PRMs.** +.24 to +.73. RLHFlow is the exception, at −.33.
- **Dynamics channels on the strong-PRM Socratic cells:**
  - top15_turnover +.49 to +.60;
  - top50_js +.53 to +.59;
  - energy_innovation +.45 to +.55;
  - dominant_freq16 +.24 to +.33.

  On PRMBench the same channels are weaker (−.22 to +.31), and top15_turnover is negative there.
- **The correlations are nuisance, not error signal.** Among correct steps alone they are about as large (for
  example top15_turnover +.53 to +.64). So position is a common cause of the votes that does not go through the
  error label.

### How this produces the stage-1 winners

- On Socratic the dynamics channels carry little error signal. The PRM anchor therefore orients them by their
  position component: top15_turnover and dominant_freq16 are flipped on the strong-PRM Socratic cells.
- They then vote with the PRM's own position trend. DS-EM's latent class settles on "PRM plus position", and the
  PRM holds .70–.77 of the weight.
- **Supporting evidence:**
  - The trio top15_turnover + dominant_freq16 + bocpd_p0 is in 8 of the 9 subsets that win all 8 cells, against
    7.8% of valid subsets.
  - Removing one trio member from SUB5 collapses the PRM's weight and loses cells (`NAMED_LOOKUP.csv`).

### The neutralization

**The procedure** is the fork's probe (`position_probe/probe.py`), copied unchanged; its output matches the fork's
in 18/18 cells.
- Every channel is residualized on a cubic in relative position, by pooled least squares, label-free.
- The PRM is left untouched.

**Why it suffices for estimation.** Write a channel as u·y + β·position + noise. After residualization,
cov(PRM, channel) = u_PRM·u_channel·Var(ỹ) plus shared noise, where ỹ is the part of the label not explained by
position. So every cross-covariance used by SML and DS-EM is free of position, although the PRM still contains it.

**What it does to SUB5** (prediction P1, holds):
- On the three strong-PRM Socratic cells, the PRM's share falls to .16, .26 and .15.
- SUB5 loses all three: −2.32, −3.42, −3.96.
- On Socratic RLHFlow, whose PRM is anti-position, the fused score rises from 50.80 to 61.38.

**A correction to the stage-1b proposal.** It says residualizing only the channels "leaves cov(PRM, channel)
unchanged". It does not: the covariance loses the β_PRM·β_channel·Var(position) term. That is exactly why the
orientation flips go away.

## 3a. What Socratic-PRMBench is built for (checked against the source, 2026-10-26)

**Omri's question.** Is Socratic really a first-error benchmark, like ProcessBench, rather than a step-level one like
PRMBench? If so, we may have been using it for the wrong task.

### What the source says

Li et al. 2025, "Socratic-PRMBench: Benchmarking Process Reward Models with Systematic Reasoning Patterns",
arXiv 2505.23474. The repository README is empty, so the paper is the reference.
- **Construction.** Each path starts from a solution verified as completely correct. GPT-4o then modifies it to
  introduce a chosen error type, and may adjust original steps or add new ones.
- **Labels.** `error_steps` are the steps that contain errors, and they are part of `modified_steps`.
- **Task.** Step-level detection of flawed steps.
- **Metric.** PRM-Score: 0.5·F1 on valid steps + 0.5·F1 on erroneous steps. The paper says it follows PRMBench and
  ProcessBench, and uses the PRMBench toolkit.
- **Error location by pattern.** The paper's Table 2 reports the average first-error step for each reasoning pattern:
  Transformation about step 1.5, Verification about step 6.9.

### What the paper evaluates and reports

**Metrics:**
- PRM-Score: 0.5·F1 on valid steps + 0.5·F1 on erroneous steps; random guessing scores 50.
- Accuracy on correct steps, on erroneous steps, and overall (their Table 4, used to show reward bias).
- PRM-Score per pattern and per sub-category (their Table 3 and Figure 3).
- Average first-error step per pattern (their Table 2).
- Predicted against true error positions (their Figure 4).

**Models evaluated:**
- **6 PRMs:** Skywork-PRM-7B, ReasonEval-7B, RLHFlow-PRM-Mistral-8B, RLHFlow-PRM-Deepseek-8B,
  MathShepherd-Mistral-7B, Qwen2.5-Math-PRM-7B.
- **5 LLM critics:** GPT-4o, Gemini-2.5-Pro, DeepSeek-R1, QwQ-32B, o3-mini.

**Data (their Table 2).** 2,995 paths, 8.7 steps and 3.0 error steps per path on average. Average first-error step
by pattern:

| Pattern | Paths | Average first-error step |
|---|---|---|
| Transformation | 313 | 1.5 |
| Decomposition | 463 | 3.0 |
| Regather | 463 | 3.1 |
| Deduction | 926 | 5.4 |
| Integration | 615 | 7.2 |
| Verification | 215 | 6.9 |

**Their findings:**
- **PRMs trail LLM critics.** The best PRM is Qwen2.5-Math-PRM-7B at 68.0; the best overall is o3-mini at 75.7.
  Skywork-PRM-7B scores 43.6 and RLHFlow-PRM-Mistral 48.8, at or below chance.
- **Redundancy errors are the hardest.**
- **Early patterns are harder.** Transformation, Decomposition and Regather are harder than Deduction, Integration
  and Verification. For example, the Qwen PRM scores about 80 on Integration but below 60 on Decomposition.
- **Some models detect errors late.** Qwen2.5-Math-PRM and o3-mini flag errors later than they occur; MathShepherd
  flags early correct steps.
- **Reward bias.** Qwen2.5-Math-PRM-7B is 90.8% accurate on correct steps but 42.9% on erroneous steps.

**Replication.** Our native-threshold PRM-Score reproduces the paper's for the two shared PRMs: Qwen2.5-Math-PRM-7B
68.09 against 68.0, and RLHFlow-PRM-Mistral 48.12 against 48.8 (`STATUS_MAIN.csv`).

**Relation to our findings:**
- Their pattern difficulty is the mirror image of our position effect. The late patterns, where position predicts
  the error, are the ones PRMs find easy. Their "late detection" finding is the PRM position trend we measured.
- Their hardest type is redundancy. On Socratic our advantage there is partial: it holds in 2 of the 4 redundancy
  categories (`SOCRATIC_BY_CATEGORY.csv`).

**Socratic redundancy categories, within-answer AUC** (strong PRMs Qwen-7B / Qwen-72B / PURE; position-neutral SUB5):

| Category | Position alone | PRM | Telemetry alone | Telemetry + PRM |
|---|---|---|---|---|
| regather_redundancy | .31 | .92 / .90 / .60 | .95 / .90 / .94 | .96 / .94 / .94 |
| decomposition_redundancy | .42 | .70 / .68 / .58 | .69 / .62 / .67 | .69 / .68 / .68 |
| premise_redundancy | .78 | .88 / .83 / .86 | .76 / .67 / .72 | .80 / .81 / .75 |
| integration_redundancy | .92 | .88 / .83 / .90 | .80 / .77 / .78 | .82 / .83 / .81 |

**Reading:**
- The telemetry adds value where position carries nothing: regather, and decomposition for PURE. RLHFlow is not in
  the table; on decomposition_redundancy it goes from .46 for the PRM to .63 for telemetry alone and .67 for
  telemetry + PRM.
- In the late, position-heavy redundancy categories (premise, integration), the PRM is stronger.

**For a paper:**
- Cite their "redundancy is hardest" finding as context.
- Claim our advantage on PRMBench redundancy (NR +9 to +22; SLIDER balanced accuracy 69–75 against PRMs at 53–57).
- Claim the early-pattern redundancy categories on Socratic, with the position baseline shown.

**SLIDER** (Step-RRI, arXiv 2610.00571):
- It reports only PRMBench's redundancy class (accuracy 61.9). There is no Socratic number and no code release.
- So a Socratic comparison would apply its protocol to the four redundancy categories above: our method against the
  PRMs and position alone. It would not be a head-to-head with Step-RRI.

### Redundancy errors under the SLIDER protocol (pre-declared, run 2026-10-26)

**Setup.** `PREDECLARATION_REDUNDANCY_SLIDER.md`, `redundancy_slider.py`, `REDUNDANCY_SLIDER.csv`.
- Parity with `COMP_SLIDER.csv`: 12/12 values exact.
- SLIDER itself cannot be run: there is no code, and it needs model inference. Its only number is PRMBench
  redundancy accuracy, 61.9.
- Values below are balanced accuracy under the protocol: a logistic threshold fitted on half the answers. Ranges are
  over the 4 PRMs.

| Arm | PRMBench redundancy | Socratic regather_redundancy | Socratic integration_redundancy |
|---|---|---|---|
| PRM alone | 49–62 | 55–85 | 56–85 |
| Position alone | 58 | 75 | **89** |
| CHAIN_N16 (original label-free boosting) | 68–76 | 75–89 | 46–73 |
| TELE_N16 (all 16 channels, no PRM) | 72–76 | 81–88 | 49–67 |
| LSML_CONT (original 3-channel boosting) | 68–75 | — | — |
| SUB5 + PRM, raw | 70–76 | 60–86 | 62–85 |
| SUB5 + PRM, position-neutral | 69–74 | **86–89** | 73–78 |
| SUB5 alone, position-neutral | 72–73 | 84–87 | 71–74 |

### Readings

**PRMBench:**
- Every telemetry arm beats SLIDER's 61.9 accuracy in all 4 PRMs; the range is 66–78. Prediction R1 holds.
- The PRM is near chance in balanced accuracy, even where its accuracy looks high (Qwen-72B: 81.3 accuracy, 61.7
  balanced).
- The value comes from the telemetry; adding the PRM does not help. The full bank alone is best (balanced 73–78).
  SUB5 is slightly lower, and LSML_CONT lower still.
- Position cleaning costs SUB5 + PRM 1.1–1.6 points. Prediction R3 holds.

**Socratic, by pattern:**
- **regather** (position .31): position-neutral SUB5 + PRM beats the PRM in all 4 PRMs (86.0–89.1 against
  55.1–84.5) and beats CHAIN_N16 in 3/4. Prediction R2, first half, holds.
- **decomposition:** the telemetry beats the PRM only for PURE and RLHFlow (about +10).
- **premise and integration:** the PRM and position dominate. Position alone reaches 78 and 89 balanced. Our
  position-neutral arms do not beat the strong PRMs there. Prediction R2, second half, holds.
  - Raw SUB5 + PRM edges the PRM on integration: 82.9 against 79.8 on Qwen-7B. That is the position coalition again.
- **All four redundancy categories pooled:**
  - Qwen-7B and Qwen-72B: tie with the PRM.
  - PURE: +5.5 (full bank + PRM, raw).
  - RLHFlow: +13 (position-neutral).

### What this means for our pipeline

- **Socratic is a PRMBench-style step-level benchmark, not a first-error benchmark.**
- Our adapter (`spectral_utils/external_generalization/contracts.py: adapt_socratic`) marks exactly the source's
  `error_steps` (one-based) as errors and propagates nothing. Our step-level PRMScore matches the official protocol.
- The dataset was used for its intended task.

### Label structure (`label_structure.py`, `LABEL_STRUCTURE.json`)

| | PRMBench (non-control answers) | Socratic |
|---|---|---|
| Answers containing an error | 97% (plus 758 fully correct controls) | 99.8% (6 of 2,995 have none) |
| Error steps per erroneous answer | 2.2 (33% single) | 3.0 (12% single) |
| Errors form one contiguous block | 72% | 68% |
| The block runs to the last step | 13% | 31% |
| The last step is an error | 19% | 44% |
| Position alone, within-answer AUC | .662 | .731 |

The two benchmarks share one design. Socratic has longer error blocks, and they end the answer more often.

### Where Socratic's position signal comes from: the reasoning pattern

Source: `SOCRATIC_BY_CATEGORY.csv`. Values are within-answer AUC of relative position, within each category.

The grouping into the paper's six patterns is read from the category names; the dataset does not state it.

| Category group (inferred pattern) | Position AUC |
|---|---|
| integration_* (Integration) | .92–.98 |
| detection_error, correction_error (Verification) | .86–.88 |
| conclusion_* (Deduction) | .85–.87 |
| premise_* (Deduction) | .78–.83 |
| decomposition_*, regather_* (Decomposition, Regather) | .31–.70; regather_redundancy .31 |
| repeat_* (probably Transformation) | .46–.51 |

**What follows:**
- **Position alone beats the mean strong-PRM AUC in 7 of 20 categories:** all four integration categories, both
  verification categories, and conclusion_invalidity.
- **The position-neutral SUB5 + PRM loses most against the PRM exactly where position is most informative.** Across
  the 20 categories, the Spearman correlation between position AUC and (SUB5 position-neutral − PRM) is −.69. The
  loss is up to −.17 within-AUC in the integration and conclusion categories.
- **It gains where position carries nothing:** regather_redundancy +.14, decomposition_redundancy +.03,
  regather_incompleteness +.04.
- **The telemetry alone is strongest on redundancy patterns:** regather_redundancy .93, integration_redundancy .78.
  This is the same redundancy strength seen on PRMBench (§8).

### Reading

The confusion did not come from using the wrong dataset. It came from Socratic's design:
- the error's location is tied to which reasoning pattern was perturbed;
- the strong PRMs are themselves highest in the late-pattern categories (up to .93–.95 in two integration
  categories).

So any step-level score can gain on Socratic by "knowing where a pattern lives".

**Report Socratic accordingly:**
- with a position-only baseline;
- per reasoning pattern, at least early patterns (repeat, decomposition, regather) against late patterns
  (conclusion, verification, integration);
- with the telemetry's contribution claimed only where it holds within patterns.

## 4. Jaffe, Fetaya and Nadler (2016): what the method assumes, and whether our data satisfy it

**What the diagnostics are.** D1–D3 below come from `EVAL8_DIAGNOSTICS.csv` and are computed per cell, raw against
position-neutral (POS). Labels enter D1–D2 as diagnostics only.

| Paper element | What it requires | Raw | Position-neutral |
|---|---|---|---|
| Model (Eq. 6–7) | Votes from different groups are conditionally independent given y. The PRM is its own group. | Violated. Position is a common cause; within class (mean of the valid and error classes), PRM–channel \|corr\| is .11–.17 and cross-family \|corr\| .12–.19. | Better, not exact. PRM–channel \|corr\| is lower in 7/8 cells (.07–.16). Cross-family is lower only on the 4 Socratic cells (.12–.15). On PRMBench it is unchanged (~.13). A shared backbone nuisance remains (FRAMING_THEORY P4). |
| Lemma 4 assumption | Every classifier better than random (2π − 1 > δ > 0). | Violated in 6/8 cells: 1–2 channels below chance, and 6 on Socratic RLHFlow. | Holds in 8/8 cells: all 16 channels are above chance. |
| Lemma 1 | Cross-group covariance is rank one: r_ij = (1 − b²)(2π_i − 1)(2π_j − 1). | Misfit of [PRM, S_R] is .03–.13, and .33 on Socratic RLHFlow. | Misfit ≤ .046 in 8/8 cells (lower than raw in 8/8). |
| Lemma 2 + Algorithm 1 | The score matrix separates groups; K searched over 2..m−1. | 16-channel grouping matches the code-defined families: Rand .78–1.00 against a null of .57–.70, p ≤ .004 in 8/8 cells. | Rand .74–.95 against the families. |
| Lemma 3 | Minimizing Eq. 14 is NP-hard in general. | Exact enumeration is used for ≤ 4 channels, which is tractable. | — |
| Lemma 4 | Separation needs accurate classifiers, strong within-group dependence, and K ≥ 4. | Vacuous for K ≤ 3 (§2). 16-channel K is 3–6 raw. | 16-channel K is 3–7. |
| Algorithm 2 | Dawid–Skene within groups, then between groups. | SML within groups, DS-EM between groups. | Same. |
| Eq. 19 | Maximum-likelihood prediction. | Linear weights on continuous values; HEM as a secondary. | For F_CI sets (K = m), Eq. 19 is the linear log-odds vote. |
| §4 selection heuristic | Most accurate classifier per group, M ≤ K. Evaluated in the paper on DREAM with a simple vote. | Our rule (§5), with accuracy ranked by covariance with the PRM. | Same. |

**Proposition A** (anchored ranking; the proof is Lemma 1 with i = PRM). If the PRM is better than random and is its
own group, then cov(PRM vote, channel k vote) = (1 − b²)(2π_PRM − 1)(2π_k − 1). So ranking channels by this
covariance ranks them by balanced accuracy, without labels and without using any telemetry–telemetry covariance.

**Proposition C** (chosen_surprisal and entropy are dependent by construction).
- s = H + e, where e is the excess surprisal. Under "the text comes from the model's own distribution",
  E[e | p] = 0, so cov(s, H) = Var(H).
- Consequence: chosen_surprisal and q15_H1 belong to one dependency group. L-SML merges them inside SUB5 in 8/8 raw
  cell fits (8/10 counting Math-Shepherd) and in 5/8 position-neutral fits. In the other 3 position-neutral fits,
  top15_turnover and dominant_freq16 merge instead.

## 5. The label-free rule and what it picks

### Rule R

Rule R was pre-declared before it was run.
1. **Families** are fixed from the code: uncertainty, realized, dynamics, digit (members in `eval8.py`).
2. **Orientation audit.** Keep a channel only if its PRM-anchored sign in the position-neutral representation equals
   its stored sign in all 8 cells. The set must also work without the PRM, and no rule inside the telemetry can repair
   a stored sign: the consensus sign agrees with the stored sign even for energy_innovation and top50_js, which are
   below chance.
3. **Accuracy proxy.** The covariance r_0k between the PRM's vote and channel k's vote (Proposition A), median over
   the 8 cells.
4. **Pick** one channel per family: the largest median r_0k.

### What it picks (`RULE_POS.json`)

- **Excluded by the audit:** energy_innovation (anchored sign agrees in 1/8 cells), top50_js (3/8), realized_z
  (7/8).
- **S_R** = {q15_H1, chosen_surprisal, top15_turnover, digit_spread}.

**How clear the picks are:**

| Family | Pooled pick | Runner-up | Per-cell winners |
|---|---|---|---|
| uncertainty | q15_H1 (.171) | true_tail50 (.165) | q15_H1 5, true_tail50 2, energy_level 1 |
| realized | chosen_surprisal (.162) | realized_drv (.139) | chosen_surprisal 8/8 |
| dynamics | top15_turnover (.064) | bocpd_p0 (.053), dominant_freq16 (.047) | top15_turnover 4, dominant_freq16 3, bocpd_p0 1 |
| digit | digit_spread (.158) | digit_alternative (.120) | digit_spread 8/8 |

### Reading: the rule recovers SUB5's structure

SUB5 = {q15_H1, chosen_surprisal, digit_spread} plus {top15_turnover, dominant_freq16, bocpd_p0}.
- The first three are the rule's stable picks.
- The second three are exactly the dynamics members the rule picks cell by cell. They are near-tied, and each is
  its own L-SML group in SUB5's fits.
- So SUB5 is "one per family, keeping all near-tied candidates of the one family without a stable winner". This
  reading was found after the fact, not pre-declared.

**Prediction P4 holds:** S_R contains digit_spread and chosen_surprisal. In the raw representation the same rule was
unstable: there, the dynamics slot went to sign-flipped energy_innovation or top50_js.

## 6. Performance on the 8 cells

**Setup.**
- Source: `EVAL8_ARMS.csv`, `EVAL8_SUMMARY.json`.
- Official PRMScore under GAP.
- Margins are against the stage-1 exact PRM comparator and the same-set plain mean.

**Which fusion each set uses.**
- **SUB5:** stage-1 L-SML with the PRM, stage-2 L-SML alone.
- **S_R and C3:** F_CI, i.e. Dawid–Skene with each channel its own group.

| Set, representation | With PRM: beats PRM | With PRM: beats own mean | Both (goal G1) | Alone: beats own mean (goal G2) | Alone: beats PRM |
|---|---|---|---|---|---|
| SUB5, raw | 8/8 | 8/8 | **8/8** | **8/8** | 4/8 |
| SUB5, position-neutral | 5/8 | 8/8 | 5/8 | 7/8 | 3/8 |
| S_R, raw | 5/8 | 2/8 | 2/8 | 6/8 | 3/8 |
| S_R, position-neutral | 4/8 | 4/8 | 2/8 | 6/8 | 3/8 |
| C3, raw | 5/8 | 2/8 | 2/8 | 3/8 | 3/8 |
| C3, position-neutral | 5/8 | 5/8 | 3/8 | 6/8 | 3/8 |

**Fused minus PRM, with the PRM, position-neutral.** Cells in order: PB Qwen-7B, Qwen-72B, PURE, RLHFlow; SO
Qwen-7B, Qwen-72B, PURE, RLHFlow.
- **SUB5:** +2.18, +2.00, +2.69, +7.09 | −2.32, −3.42, −3.96, +10.99.
- **S_R:** +1.37, +1.02, +2.56, +7.13 | −2.00, −4.50, −4.20, 0.00.
  - On Socratic RLHFlow the PRM is near chance (vote balanced accuracy .513).
  - Anchoring the label switch on a near-chance PRM makes every channel weight clip to 0, so the fused score
    equals the PRM.
- **C3:** +1.34, +1.29, +2.74, +7.71 | −1.21, −3.01, −3.49, +11.89.

**Within-answer AUC** (position-neutral, with the PRM):
- PRMBench: SUB5 .776–.835, against the PRM's .662–.801.
- Socratic strong cells: SUB5 .715–.768, against the PRM's .796–.824.

**Secondary checks:**
- **DS-EM probability floor.** It has no effect on the F_CI arms: 1e-4 against 1e-6 gives 0.00 difference in 8/8
  cells.
- **Binary scoring.** It is below continuous scoring for SUB5 on PRMBench (−0.7 to −1.6).
- **P6 holds.** S_R alone is within 1 point of its own mean in 6/8 cells.

## 7. Why no position-neutral set beats a strong PRM on Socratic (post-hoc diagnostic, labels used)

**The diagnostic.** Source: `weighting_diag.py`, `WEIGHTING_DIAG.csv`, `WEIGHTING_CURVE.csv`. It was not in the
pre-declaration. Following the paper's Section 5, label-free DS-EM is compared with "Oracle-CI": the same linear
log-odds rule (their Eq. 5), given the true sensitivity and specificity of every vote.

**1. The bias is in the channels' accuracy, not the PRM's.**
- DS-EM estimates the PRM's balanced accuracy almost exactly. For example, on Socratic Qwen-72B: .683 against a true
  .708.
- It overestimates the channels' accuracy. For S_R the mean overestimate is +.10 to +.12 on PRMBench and +.14 to
  +.16 on the strong-PRM Socratic cells; for C3 it is +.10 to +.18.
- This is stage 0's over-trust, now in a representation where Lemma 1's misfit is small. A shared telemetry
  nuisance makes the channels agree beyond y, and a conditional-independence estimator reads that agreement as
  accuracy. This is FRAMING_THEORY P4: second moments cannot see that the PRM is the strong view.

**2. Oracle-CI also loses to the PRM on the strong-PRM Socratic cells.**
- Oracle-CI scores 69.53, 69.21 and 67.90 against the PRM's 70.10, 71.72 and 70.59.
- It gives the PRM only .30–.36 of the weight.
- Binarizing by GAP compresses the PRM's advantage. Its vote's true balanced accuracy is .69–.71, against .54–.66
  for S_R's channels. Its continuous within-answer AUC is .80–.82, against .51–.72.

**3. Where along the PRM's share a any gain exists.**

The curve is fused = a·PRM + (1 − a)·telemetry composite of S_R.

| Cell | Range of a with a gain | Best gain |
|---|---|---|
| Socratic Qwen-7B | .45–.95 | +0.45 |
| Socratic Qwen-72B | .70–.95 | +0.35 |
| Socratic PURE | .80–1.0 | +0.02 |
| PRMBench (all 4 PRMs) | almost every a | +1.6 to +7.0, at a ≈ .1–.45 |
| Socratic RLHFlow | a ≈ 0 is best | +11 |

FRAMING_THEORY P2(iii) predicted a_min ≈ .52–.82 for Socratic Qwen-72B. This matches.

**Conclusion.** On strong-PRM Socratic the channels carry little that the PRM lacks: at most half a point, and only
when the PRM holds most of the weight (at least 35–45% on Qwen-7B, at least 70–80% on Qwen-72B and PURE). No set choice changes that. Getting there needs a different weighting: a
declared PRM anchor, or an identification of the PRM's continuous advantage. It cannot come from Jaffe's
conditional-independence weights, even with oracle accuracies. Raw SUB5 reached that weight range only through the
position coalition. That is why its Socratic margins were small (+0.15 to +0.64).

## 7a. Position as a separate channel? (Omri's question, 2026-10-26)

**The question.** Once position has been removed from the channels, why not add position itself back as one more
SML voter? That would reuse the information it carries.

### Why it cannot be an SML or Dawid–Skene voter

**The derivation.**
- Write a channel as x = u·y + β·p + e, where y is the error indicator, p is the position and e is independent
  noise.
- Errors come later in the answer, so m(p) = E[y | p] increases with p.
- Residualizing x on p removes the whole position trend: both β·p and u·m(p). What remains is
  x̃ = u·(y − m(p)) + e.

**Two consequences:**
1. Cov(x̃, p) = 0, by construction.
2. Given y, Cov(x̃, p | y) = −u·Cov(m(p), p | y), which is negative. Removing position overall creates a negative
   dependence within each class.

**Why that breaks the model.** Under conditional independence (Lemma 1), two informative views must have a positive
covariance, equal to the product of their accuracy terms. Here the covariance is zero although both views are
informative: the negative within-class dependence exactly cancels the shared signal. So:
- A position voter violates conditional independence against every position-neutral channel, by construction.
- SML and DS-EM judge a voter's accuracy by its agreement with the other voters. Position agrees with none of the
  position-neutral channels, so it would look uninformative.
- Its only agreement would be with the raw PRM. That agreement is mostly the PRM's own within-class position trend
  (+.23 to +.72 among correct steps; RLHFlow −.41 to −.46). This is the position coalition of §3 again.

### The check

Source: `position_as_view_diag.py`, `POSITION_AS_VIEW_DIAG.csv`. Labels are used as a diagnostic only.
- **Marginal correlation.** For the position-neutral S_R channels it is −.11 to +.06.
- **What independence predicts.** corr(p, y)·corr(x̃, y) is +.02 to +.06 on PRMBench and +.01 to +.12 on Socratic.
  The observed value is below this prediction in 31/32 channel-cells.
- **Within each class,** 60 of 64 correlations are negative, down to −.19 (chosen_surprisal, Socratic PURE, error
  steps).

### Where position belongs: the prior

**The model.** A latent-class model with position as a covariate:
- P(y = 1 | p) = π(p);
- the views are conditionally independent given both y and p;
- each view's within-class position trend is a nuisance to be estimated.

**The posterior is then**

    logit P(y = 1 | x, p) = logit π(p) + Σ_k [evidence of view k, with its within-class position trend removed].

**What this means:**
- This is Dawid–Skene with a class prior that depends on a covariate: Jaffe's class imbalance b becomes b(p).
  Identifiability holds at each position as long as there are at least 3 conditionally independent views.
- **The position term is the "separate channel" Omri proposed.** Its weight, however, is the prior given by the model,
  not something SML can estimate from agreement.
- **If the total-trend residualization is kept,** the position term must also add back Σ_k w_k·u_k·m(p), the part the
  residualization stripped from each channel. That too is a model quantity, invisible to SML.
- **In this model the PRM's own within-class position trend is a nuisance.** So position is counted once, not twice
  (once inside the PRM and once as a channel).

### How much position can add (post-hoc oracle; labels choose the weight)

The score is base + c·position, with c on a grid (`position_headroom_diag.py`, `POSITION_HEADROOM.csv`). Values are
the best gain over the PRM alone.

| Base | PRMBench, Qwen and PURE | Socratic Qwen-7B / Qwen-72B / PURE | RLHFlow, PB / SO |
|---|---|---|---|
| The PRM + position | about 0 PRMScore (within-AUC +.004 to +.021) | +1.16 / +0.48 / +0.04 | +4.4 / +14.2 |
| SUB5 position-neutral + PRM + position | +2.2 to +3.1 | −0.47 / −1.11 / −2.62 | +9.7 / +15.6 |

**Reading:**
- On Socratic, position carries information the strong PRMs do not fully use: up to +1.2 on Qwen-7B. On PURE, whose
  PRM is already strongly position-loaded, it adds about 0.
- Adding position to the position-neutral fused score does not recover the strong-PRM Socratic cells, even at the
  best weight. The telemetry already holds too much of the weight (§7).
- **Conclusion:** position is worth reusing, through the prior, but it does not remove the weighting bottleneck.

### Caveats

- **Position's value is a benchmark-design property.** On Socratic the error's location is tied to the perturbed
  reasoning pattern (§3a). It is not a first-error labelling convention.
- **It must be estimated without labels on the target data:** π(p) inside the model.
- **Report it as a component separate from the telemetry.**
- **Check it on Hard2Verify,** where errors are the solving models' own.

### Proposal (not run)

Pre-register a position-aware latent-class fusion:
- EM with π(p) logistic in a cubic of relative position;
- views = the PRM plus SUB5 or S_R;
- a within-class position trend per view;
- the posterior log-odds as the score, then GAP.

Arms, controls and context:
- **Arms:** with and without the π(p) term.
- **Controls:** the PRM, position alone, SUB5 raw and position-neutral.
- **Context:** the oracle PRM + position headroom.

Stated in advance: without a better PRM weight, this is not expected to beat the PRM on strong-PRM Socratic.

## 8. Error types and operating points (PRMBench, 4 PRMs)

### Error categories

Source: `EVAL8_ARMS.csv` (category columns). Values are the mean over 4 PRMs; ranges are fused minus PRM across the
4 PRMs.

| Category | PRM | SUB5 position-neutral, + PRM | S_R position-neutral, + PRM | S_R position-neutral, alone |
|---|---|---|---|---|
| NR redundancy | 54.5 | 66.6 (+9 to +17) | 70.4 (+12 to +19) | 71.9 (+13 to +21) |
| NCL circular | 61.7 | 69.5 (+5 to +11) | 71.2 (+8 to +12) | 70.9 |
| DC domain inconsistency | 65.8 | 72.3 (+4 to +12) | 72.6 | 71.3 |
| SC step contradiction | 67.6 | 72.6 (+3 to +8) | 71.5 | 69.1 |
| PS missing condition | 61.5 | 63.8 (+1 to +6) | 64.0 | 63.3 |
| ES counterfactual | 70.1 | 70.0 (−2 to +4) | 67.4 (−5 to +3) | 65.2 |
| CI confidence | 75.9 | 75.5 (−2 to +2) | 73.0 (−5 to +2) | 69.8 |
| DR deception | 67.9 | 66.4 (−4 to +1) | 62.9 (−7 to 0) | 60.0 |
| MS multi-solution | 93.7 | 90.9 (−4 to −1) | 89.6 | 88.8 |

**Reading:**
- **Where the telemetry helps.** It covers the PRM's "simplicity" blind spot. Redundant or circular steps are
  locally true, so a verifier that judges content passes them. The backbone is uncertain or surprised there.
- **Where it hurts.** Deception, confidence and counterfactual errors are fluent and confident: the backbone is not
  surprised, and the PRM catches them by content.
- **Position is not the driver on PRMBench.** Position alone is below the PRM in every category on average, by 4–18
  points.
- **The one-per-family sets trade harder** than SUB5: larger gains on NR/NCL, larger losses on DR/CI.

### Operating points

Source: `EVAL8_COMPETITORS.csv`. These are the competitors' own protocols. Official orientation: positive = valid
step, so FPR is the share of error steps passed as valid.

| Qwen-7B | PRM | CHAIN_N16 | SUB5 raw + PRM | SUB5 position-neutral + PRM | S_R position-neutral + PRM |
|---|---|---|---|---|---|
| TPR / FPR at GAP | 92.1 / 58.5 | 88.6 / 45.1 | 91.6 / 53.1 | 92.0 / 53.4 | 89.2 / 49.0 |
| FPR at matched TPR 87.3 (PRISM 47.13) | 47.70 | 47.78 | **46.31** | 47.07 | 50.44 |
| ARES-protocol macro-F1 | 68.17 | 68.10 | 68.81 | 68.46 | 67.22 |
| SLIDER redundancy, accuracy / balanced accuracy (Step-RRI 61.9) | 78.8 / 54.5 | 74.1 / 74.3 | 75.4 / 72.2 | 73.7 / 71.1 | 73.8 / 73.1 |
| CROP kept at α = 5% (paper 21.8) | 4.83 | 23.33 | 23.67 | 21.34 | 18.89 |
| CROP calibrated on PRMBench, Socratic risk (target 5%) | 2.07 | 6.71 | 12.82 | 5.41 | 6.94 |

**Pattern across the four PRMs:**
- **At the GAP operating point,** every fused set catches more error steps than the PRM (FPR lower by 2–16 points).
  The cost is at most 3 points of valid-step recall, and on PURE and RLHFlow recall rises.
- **At PRISM's matched TPR,** only SUB5 improves on the Qwen PRMs (Qwen-72B: 41.92 raw, 43.93 position-neutral,
  against 45.25). S_R and C3 are worse than the PRM there.
- **On redundancy,** the PRMs are near chance in balanced accuracy (52.7–57.2). Every fused set reaches 68.9–75.6.
- **The anchor can fail.** CROP cross-domain on Socratic RLHFlow for S_R gives risk 99.8, because that fused score
  is the near-chance PRM. A label-free abstain check on the anchor (FRAMING_THEORY P5(g)) is needed before any use.

## 9. The channels of SUB5, one by one

**q15_H1** (uncertainty; the entropy of the next-token distribution):
- Covers redundancy and circular-logic steps.
- In the position-neutral representation: the largest pooled anchored covariance in its family, a near tie with
  true_tail50, and orientation stable in 8/8 cells.
- It shares the entropy term with chosen_surprisal (Proposition C), so the two form one L-SML group.

**chosen_surprisal** (realized; how unexpected the token actually written is):
- The family winner in 8/8 cells after neutralization.
- The highest stored-orientation within-AUC on PRMBench (median .766 over 4 PRMs).
- **Main risk.** PRMBench and Socratic build errors by editing correct solutions, so edited tokens are not the
  backbone's own continuation. Hard2Verify, with errors the models made themselves, tests this.

**digit_spread** (digit; how spread the probability is over the ten digits):
- The family winner in 8/8 cells.
- Its own dependency group in every 16-channel grouping.
- Decoding-independent; consistent with Omri's 2026-09-30 decision to keep digit features in the label-free line.

**top15_turnover, dominant_freq16, bocpd_p0** (dynamics; change in the next-token distribution or the entropy
series):
- **Raw:** weak, and on Socratic their orientation follows position. They are the position coalition of §3.
- **Position-neutral:** all three keep their stored sign in 8/8 cells, and each wins the dynamics family in some
  cells.
- **Role in SUB5:** three near-tied weak singleton groups. They provide the third and later independent groups that
  the telemetry-only fit needs (§2). After neutralization they no longer steer the PRM's weight.

## 10. Predictions against results

| | Prediction | Result |
|---|---|---|
| P1 | SUB5 + PRM, position-neutral, loses to the PRM on ≥ 2 of 3 strong-PRM Socratic cells, with PRM share < .5 | **Holds:** 3/3; shares .16 / .26 / .15 |
| P2 | Neutralization lowers cross-family and PRM–channel conditional dependence in ≥ 6/8 cells | **Fails as stated.** PRM–channel 7/8; cross-family 5/8, Socratic only |
| P3 | Lemma-1 misfit of [PRM, S_R] lower position-neutral than raw in ≥ 6/8 | **Holds:** 8/8 |
| P4 | S_R contains digit_spread and chosen_surprisal | **Holds** |
| P5 | S_R + PRM beats the PRM on 4/4 PRMBench cells in both representations, and position-neutral not on ≥ 2 of 3 strong-PRM Socratic cells | **Holds:** 4/4 and 4/4; loses 3/3 |
| P6 | S_R alone within 1 point of its mean in ≥ 6/8 | **Holds:** 6/8 |

## 11. What can and cannot be claimed

**Can be claimed** (development data, in-sample):
- On PRMBench (4 PRMs) and on a weak PRM (Socratic RLHFlow), label-free fusion of SUB5's channels improves on the PRM
  and on the same-set mean. This holds with position removed.
- The gain is concentrated on redundancy and circular-logic errors.
- The set's structure follows from a label-free rule derived from Jaffe et al.'s selection heuristic and Lemma 1.

**Cannot be claimed:**
- That any set beats the PRM in all 8 cells without position. The raw 8/8 result depends on position.
- That the Socratic gains with strong PRMs are telemetry information.
- That the rule was formulated before SUB5 was known. It was not, so the agreement is not independent evidence.

**Assumptions that remain violated:**
- non-i.i.d. steps;
- residual cross-family dependence (a shared backbone nuisance);
- the PRM and the telemetry share a backbone (the anchor condition).

## 12. Decisions for Omri (nothing run)

1. **Representation.** Should the position-neutral representation become the default for this line? It is better
   justified (D1–D3) and does not depend on position. The cost is the raw Socratic wins.
2. **The goal "beat the PRM in all cells".** On strong-PRM Socratic this needs the PRM to hold ≥ 70% of the weight.
   That requires a change in weighting, not in channels.
   - The options are the stage-1b PRM weight anchor (a declared prior) or a label-free identification of the PRM's
     continuous advantage (FRAMING_THEORY P5, whose gate implementation failed earlier).
   - A position-neutral subset search for this goal is not recommended. §7 shows that any winner there would be
     tie-level and found by accident.
3. **Held-out test on Hard2Verify** (CPU, 177 answers, 4 PRMs). Pre-register before running.
   - Arms: SUB5 and S_R, with the PRM and alone, raw and position-neutral.
   - Comparators: the PRM alone, the same-set mean, CHAIN_N16, TELE_N16.
   - The label-free pre-check: the rule's picks and C3/S_R orientation on Hard2Verify, before labels.
   - Note in advance: aggregate Hard2Verify results already show the full bank near chance (TELE_N16 within-AUC
     .48–.56), and position's role there is unknown.

## 13. Files (`results/prm_view_selection_v1/reasoning/`)

- `PREDECLARATION_8CELL_EVAL.md`: the pre-declaration.
- `eval8.py`:
  - `cache`: the position-neutral caches;
  - `rule`: writes `RULE_POS.json`;
  - `run`: writes `EVAL8_ARMS.csv`, `EVAL8_DIAGNOSTICS.csv`, `EVAL8_SHARES.csv`, `scores/`;
  - `comp`: writes `EVAL8_COMPETITORS.csv`.
- `EVAL8_SUMMARY.json`.
- `weighting_diag.py` (post-hoc): `WEIGHTING_DIAG.csv`, `WEIGHTING_CURVE.csv`.
- `position_as_view_diag.py` (diagnostic): `POSITION_AS_VIEW_DIAG.csv`.
- `label_structure.py` (descriptive): `LABEL_STRUCTURE.json`, `SOCRATIC_BY_CATEGORY.csv`.
- `position_headroom_diag.py` (post-hoc oracle): `POSITION_HEADROOM.csv`.
- `existing_8cell.py`: `EXISTING_8CELL.json`, `POSITION_CORR.csv`.
- `checks.py` (8 cells): `CHANNEL_TABLE.csv`, `FAMILY_VS_LSML.csv`, `RULE_PICKS_PER_CELL.csv`, `NAMED_LOOKUP.csv`,
  `REPLAY_WEIGHTS.csv`, `CHECKS.json`.
- Logs: `*.log`.
- `cache_pos/` and `scores/` are large intermediates and should not be committed.

**Sources:**
- Jaffe, Fetaya, Nadler, Jiang and Kluger (2016): Lemmas 1–4, Eq. 14–15 and 19–20, Algorithms 1–2, §4 "Classifier
  selection", §5 Oracle-CI (`papers/extracted/unsupervised-ensemble-learning-with-dependent-classifiers.md`).
- Parisi et al. (2014).
- Allman, Matias and Rhodes (2009).
- `docs/research_notes/math_rigor_2026-10-24/FRAMING_THEORY.md`: P2–P5.
- Stage READMEs and `position_probe/`; the handoff `docs/handoffs/VIEW_SELECTION_REASONING_HANDOFF_2026-10-26.md`.
