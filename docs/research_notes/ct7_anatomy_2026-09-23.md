# What makes CT7 work: an anatomy from the recorded evidence (2026-09-23)

Omri's question: what in CT7 (features, readouts, thresholds, order of operations) makes it the
strongest locator? This note assembles the answer from the frozen record only; nothing was
recomputed. Sources: `spectral_utils/frozen_locator_ct7.py`, `results/chosen_token_calibration_v1/
FROZEN_CANDIDATE_CT7.json`, HISTORY Steps 334, 361-364, 372-373, 377, 387, 399, 413-420, 428-432
(branch `claude/competition-sync-2026-09-23`), and Codex's CT7-profile vote-fusion run.

## 1. The recipe, stage by stage

| stage | CT7 | eleven-channel bank (v2 / readout-family) |
|---|---|---|
| token views | H0lim, ve0, ve0.75, ve1 (shape statistics of the renormalized top-15 distribution), H0lim strict-prefix innovation | q15_H1, q15_VE1, chosen_surprisal, logprob_margin, true_tail50, energy_level, energy_innovation, top15_turnover, top50_js, dominant_freq16, bocpd_p0 |
| non-entropy views | signed BOCPD residual of the five streams (Gaussian, hazard 1/32); provided-token pooled z-test (entropy-free), step 0 set to the answer mean | bocpd_p0 (a change probability); raw chosen_surprisal (corr .34 with entropy) |
| step readout | Top10 mean of the largest token values, every view | top5 (frozen), label-selected among seven, or top30 |
| normalization | per view, within the answer (median/IQR, masked), missing = 0 | same idea (answer-local z) |
| fusion | equal 1/7, no fitted parameter, no sign anchor | equal / spectral / L-SML / DS / EM, fitted per source fold |
| locator | argmax step | argmax, or cumulative-vote mode |
| no-error decision | frozen tail15 (missing top-15 mass) Top10 whole-answer score, within-cell midrank >= .33 | same gate borrowed for the common-gate F1 |

Development scores (13,769 answers, v3 labels): PB macro-F1 41.19 under the gate, gate-free SLA
39.89, PRMB within-answer AUROC .7724. Effective conditionally independent views: 1.80 of 7.

## 2. Where the points came from (the ladder as recorded)

| step | change | PB (gated) / within |
|---|---|---|
| 334 | plain token entropy, Top10 readout (vs 8-token windows 31.76, step mean 27.66, max 28.90) | 35.44 / .7301 |
| 361-364 | four q15 Renyi/varentropy views, per-view Top10, raw equal; supervised simplex and centering rejected | 36.62 / .7534 |
| 372-373 | tail15 Top10 gate, q = .33 chosen on PB (plateau .31-.35) | 37.47 / .7534 |
| 377 | + H0lim prefix innovation (innovation5); duplication / centering / shuffled-prefix controls do not reproduce it | 39.83 / .7603 |
| 387 | + signed BOCPD residual (Ridge predictor scored higher, 40.85, not adopted) | 40.37 / .7632 |
| 399, 413 | 50- and 20-stream banks with every grouping/weighting rule: <= 40.04; the six streams at equal weight 40.27; IU on them 36.65, continuous L-SML 39.98 | "selection, not weighting" |
| 415-417 | entropy-free chosen-token statistic, length-free z readout, step-0 neutralization (view alone 19.62 -> 32.28) | 41.19 / .7724 |
| 420 | length-calibrated readout removes 8.16 pp; declared log-length view recovers 3.35 | length is evidence |

Every increment after Step 364 is under 1 pp on PB; the two large levers were the token-level
Top10 readout (+3.6 pp over windows) and the prefix innovation (+2.4). CT7 is a stack of small,
individually verified additions on a strong base, not one mechanism.

## 3. The mechanisms, as measured

1. **The views carry the most per-step information.** KL(first-error steps || error-free steps)
   on answer-standardized scores: CT7 .332, token L-SML .260, token equal .232, longest-step .178,
   Mind-the-Gap .079 (Step 428). CT7's KL rises with depth (.262 / .353 / .438), so its long-chain
   deficit is argmax competition, not signal loss. Per-view PRMB step AUC .689-.713 for the five
   entropy views, .647 BOCPD, .690 chosen-token (Step 418).
2. **Top10 is an aggregation that keeps step length as evidence.** The informative tokens inside
   a step are scattered, not adjacent (contiguous windows -6.65 pp, Step 426); whitening the
   series destroys signal (-2.24); larger k helps monotonically in pooled divergence (top30 >
   top10 > top5, Step 428) but top30 costs PRMB (-.053 within), whose steps are three times
   shorter; Top10 is the compromise that holds both benchmarks. Removing the length coupling
   costs 8.16 pp (Step 420) and the first error is the longest step in 29.7 % of answers (chance
   15.5 %): the coupling is largely legitimate.
3. **Equal weight over five near-copies is variance reduction, not fusion.** The entropy views
   correlate .75-.97 conditionally; averaging them smooths per-token measurement noise the way
   Top-k does. Four q15 views beat the best single one by about 0.9 pp (36.62 vs 35.68). With
   fewer than three independent views the L-SML eigen-stage is undetermined (Step 205 guard) and
   every fitted rule on the six is worse than equal (IU -3.6, L-SML -0.3).
4. **The two non-entropy views are the only genuinely new evidence**, and they are added as
   equal members, not by learned weight: BOCPD residual (+0.5 pp / +.003), chosen-token z
   (+0.92 pp [0.00, +1.86] / +.0135 [+.0118, +.0153]); conditional correlations .56-.65 and
   .28-.38 with the entropy family. Step 418's own reading: "new evidence, not fusion weighting".
5. **A boundary repair made the seventh view usable.** The chosen-token statistic spiked at step 0
   (+2.08 SD; view alone peaked there in 96 % of answers vs 12 % truth). Setting step 0 to the
   answer mean is a one-line rule with no fitted parameter and it moved the view from harmful to
   the largest single gain in the line. This is the same pathology the eleven-channel bank still
   carries (discrete channels voting step 0 on 83-86 % of answers; Step 429 tied-argmax 99-100 %).
6. **The gate is separate and strong.** Answer-level AUROC .80 from missing top-15 mass; the
   midrank rule is transductive within the cell. Gate-free SLA 39.89 shows the locator does not
   depend on it, but the F1 41.19 does (gate-dependent macro-F1 was the Step 423 trap).

## 4. Why the eleven-channel bank loses to it (measured, not argued)

- Two copies of the strong family instead of five, so equal weight gives the strong signal 2/11
  of the vote instead of 5/7. Adding a weak independent view at equal weight lowers fused PB
  (Step 415: -0.73 pp), and the bank carries four such channels (turnover, JS, dominant_freq16,
  energy_level) plus one gauge pair.
- Raw surprisal instead of the calibrated, entropy-free z-test; bocpd_p0 (a probability) instead
  of the signed residual; top5 instead of Top10; no step-0 rule. Codex's swap of the profiles alone
  (CT7 views, v2 recipe) gives soft equal 39.83 vs the bank's label-selected 36.03: +3.8 pp is
  views plus readouts, before any fusion rule.
- The bank's fitted rules do not repair this: with the ramp artefact removed (pmf fitting), learned
  weights equal equal weights (+0.19 / +0.31 pp, intervals include zero, Step 429).

## 5. What is development-selected in CT7 (the honest ledger)

From the frozen manifest's own caveats: the five entropy streams and the BOCPD residual were
chosen on this population (Steps 361-387, 413); the gate's q on PB labels (Step 373); the
step-0 neutralization was motivated by a label-using diagnostic and was the fifth readout of the
same statistic evaluated on these answers (Step 417); CT7's PB interval over the six touches
zero. The comparison with the eleven-channel arms is therefore development-vs-development, with
CT7 having had more rounds on the same answers. Untouched confirmation does not exist for either.
Two structural biases are measured: peaks on the longest step 43 % (truth 29.7 %) and a
start-of-answer excess (+0.67 SD at step 0; CT7 slope -1.17 SD per unit position, Step 430),
which produces its early misses (27 %); its late misses on long chains are genuine excursions.

## 6. The decomposition nobody has run as one table

All on the frozen `profiles.npy` (145,597 x 7, Codex's `ct7_profiles_v1`), same folds, gate and
bootstrap; no new inference, no fit except where stated:

1. Leave-one-view-out and single-view rows for all seven, equal weight (only six-vs-seven and
   five-vs-six exist today).
2. Copies control: one entropy view repeated five times (weight 5/7 on ve0 or ve1) versus the five
   distinct views. Separates "five copies smooth noise" from "five views add information".
3. Readout on CT7's views: top5 / top10 / top30 with everything else fixed, both benchmarks
   (Step 429 measured this only on the bank).
4. Recipe transplant: the eleven-channel bank under CT7's recipe (Top10, signed BOCPD residual,
   calibrated chosen-token z, step-0 neutralization of the spiking channels, degenerate channels
   dropped), equal weight. If this reaches ~39-40, the gap is recipe; if not, it is the views.
5. The reverse is already done (CT7 views under the v2 recipe: hard equal 37.77, soft equal 39.83).

Ordering: 2 and 4 first; they decide whether the next candidate is "better views" or "better
boundary rules", which is the question the whole readout line has been circling.

## 7. Failure points, set against the successes (added 2026-09-23)

All numbers from the frozen tables (`ct7_profiles_v1/PB_METRICS.json`, `competition_diagnostics_v1/
A1_*.csv`, `readout_family_v1/LENGTH_POSITION_DIAGNOSTICS.csv`, `quickest_detection_diagnostics_v1/
B1_SUMMARY.csv`, Steps 390, 414-420, 428-432). Micro SLA over the 4,442 erroneous answers is 38.25;
macro over eight cells 39.89.

| success | the failure sitting next to it |
|---|---|
| per-step evidence is the strongest measured (KL .332; first-error step beats a random clean step in 73-78 % of answers, rising with depth) | the argmax cannot pick it among more steps: exact .476 / .359 / .299 and MAE 0.79 / 1.60 / 3.53 steps for 2-5 / 6-10 / 11+ steps; the error is in the top two in only 45 % of 11+ answers; tolerance-one .626, so 37 % of misses are more than one step away |
| the seven views cancel each other's bias at equal weight: level views peak late (late .39-.42), innovation and BOCPD peak early (early .44 / .38), fused early .27 / late .34 | the balance is fixed, not length-aware: on the 11+ stratum late rises to .42, and the innovation view alone is early on 57-61 % of the long q8 cells |
| late misses (1,526, 34 %) sit where the label-free null is 0.46 SD *lower*, i.e. they are genuine high-uncertainty steps | they are excursions *after* the error (prediction at relative position .67, target at .27): a level detector marks where the model is most confused, which on long chains is downstream of the first error; no position null can move them |
| early misses (1,217, 27 %) are explained by the start-of-answer excess (+0.67 SD at step 0; argmax at step 0 on 32 % of gate-closed answers vs 18 % uniform, 40 % on short answers) | the true first-error hazard at step 0 is .023 and at step 1 .065; a position-conditional null repairs these but trades them one-for-one for late misses (Step 432) |
| Top10 keeps step length as evidence and that is worth 8 pp | the locator names the longest step in 55 / 41 / 30 % of answers by stratum (truth 29.7 %); log length alone scores 35.14 of the gated 41.19; the misses no method finds are short error steps inside long answers (error-step fraction .105 vs .199, Step 390) |
| gate-free SLA 39.89 shows the locator stands on its own | the gate is the larger loss on F1: clean accuracy .50 / .49 on GSM8K, .58 MATH, .60 OlympiadBench, .72 Omni-MATH (false alarms on up to half the clean answers), and it closes on 20-25 % of erroneous answers (gated error accuracy .28-.39 against SLA .37-.48); it opens on 0 of 6,969 PRMBench answers; a step-level first-crossing rule cannot replace it (at a 30 % false-alarm budget CT7 detects 16 %, exact 9 %) |
| equal weight is the correct fusion for 1.80 effective independent views; +4 pp over the best single view (ve0.75, 35.78) | that is also the ceiling: no weighting can add anything, a weak independent view at equal weight subtracts (Step 415: -0.73 pp), so the only way up is a strong *and* independent view, and after one pass over the output distribution there is no obvious source of one (Step 414) |
| PRMBench within-answer AUROC .7724, the best in the project | pooled step AUROC .723 / AUPRC .34; 28 % of erroneous PRMBench answers have more than one error run and a single argmax is the wrong decision object there; Top10 on steps of median 24 tokens is close to the step mean |
| every increment was verified with paired intervals on the full population | all of it was chosen on that population: five entropy streams, the BOCPD residual, the gate's q, the step-0 rule (fifth readout of the same statistic); the CT7-over-six PB interval touches zero; there is no untouched confirmation for any part |
| one teacher-forced pass, offline, no labels | not causal: answer standardization needs the whole answer, the gate needs the other answers of the cell (within-cell midrank); the recipe assumes a top-50 log-probability trace |

Reading: CT7 is limited by decision, not by measurement. Its three failure classes (post-error
excursions, start-of-answer excess, short error steps in long answers) are all properties of
"argmax of a level score with a length-coupled readout", and the gate adds a fourth. Anything that
keeps the level+argmax design inherits them; the decomposition in Section 6 tells which of the two
levers (views or boundary rules) is worth the next candidate.
