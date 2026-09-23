# Readout family on the frozen eleven-channel bank — 2026-09-22 (Step 429)

Question: does a label-free choice among a broader family of within-step readouts,
including temporal statistics of the step's tokens and a sequential first-crossing rule,
improve first-error localization over the frozen top5 readout, and does the learned fusion
add anything over equal weighting once the readout changes?

Authorized experiment, branch `claude/readout-quickest-detection-v1`, base = consolidation
`2e7758158` (master + token-probability-fusion-v1 + token-axis-fusion-sampling +
cumulative-vote-fusion-v2). No acceptance threshold and no automatic promotion. No new
inference. The frozen v2 profiles, `results/cumulative_vote_fusion_v2/`, CT7 and every
historical score stay immutable; new artifacts live in `results/readout_family_v1/`.

## Population and provenance

The cumulative-vote-v2 population unchanged: 13,769 answers = 6,800 ProcessBench (4,442
erroneous; Qwen3-4B and Qwen3-8B x gsm8k / math / olympiadbench / omnimath) + 6,969
PRMBench (Qwen3-8B), v3 labels, source-question folds v2 (`FOLDS_V2.json`), the frozen
CT7 gate for the common-gate F1. `prepare` rebuilds the seven frozen readouts from the token
matrices and refuses to continue unless `profiles.npy` and `shuffled_top5.npy` reproduce the
reference sha256 (`f932ad81…f5f3`, `60d6ba3a…3065`) bit for bit; the extended columns are
appended in `profiles_full.npy` with the frozen seven copied unchanged, and the shuffle
control reuses the identical per-answer `SeedSequence([seed, i])` permutations. The
`Dataset` anchor asserts (token L-SML .3592, token equal .3259, CT7 F1 .41188745848863717,
CT7 within-AUROC .7723966352864217) run before any fit.

## Readouts

Seven frozen: top5, top10, max, mean, log_top5, cusum_top5, onset80. Ten extended, all
computed on the same answer-standardised tokens (`spectral_utils/step_readouts_v1.py`,
tested against `cvf_v2.core.profiles` for identity of the shared standardisation):

| readout | what it reads | why |
|---|---|---|
| top30 | mean of the 30 largest tokens | the Top-k ladder anchor of the FUSE note; a transparent comparison, not a confirmed choice |
| std, iqr | spread inside the step | ActMap-style temporal statistics: aggregation over the step, not a peak |
| frac_above_z | fraction of tokens at or above z = 1 | same, robust to step length |
| slope | in-step OLS trend | direction of change inside the step |
| jump | step mean minus previous step mean (step 0 vs the answer median) | change against the neighbourhood; index 0 reachable |
| q90 | 0.9 quantile | a softer peak than max |
| first_token | the step's first token | the first-token effect of Snel and Oh at step level |
| boxcar8_max | best contiguous 8-token window | the matched-aggregation control Step 427 asked for beside any sequential rule |
| page_wmax | per-step maximum of the one-sided Page statistic, k = 0.5, on the channel re-centred by the answer mean and scaled by its std | the sequential statistic; a threshold on it is the token-level first crossing. Mean centring is required: the median-centred tokens leave right-skewed channels with a positive mean, and a CUSUM on them drifts with no change (found and fixed before any label was read; the first five jobs of the run were discarded) |
| page_cross (dynamic) | page_wmax up to the first step reaching h, -inf after | h = training-fold quantile 0.5 of the per-answer maximum over ALL training answers; labels unused; no crossing -> argmax fallback, rate reported |

k = 0.5 and q_h = 0.5 are fixed before running; the diagnostics stage reports k in {0.25,
1.0} and does not feed back. onset80 and page_cross are ProcessBench-only (PRMBench ranking
cannot use suffix-masked profiles).

## Rosters and arms

| roster | choice rule | role |
|---|---|---|
| top5 | fixed | frozen incumbent readout; must replay the v2 numbers |
| top30 | fixed | ladder anchor |
| onset80 | fixed, every channel | the untested anti-late roster (review of v2, section 6.3) |
| page_cross | fixed, every channel | sequential first crossing, label-free threshold |
| consensus | **label-free**: per channel, the readout whose argmax agrees most often with the equal soft fusion of the other channels at their current readouts; two coordinate-descent sweeps from top5; per-cell macro weighting on ProcessBench | primary candidate |
| consensus10 | same on the ten-channel bank without energy_level (Step 425: exactly anti-correlated with energy_innovation) | sensitivity |
| selected | label-selected over the seven frozen readouts (frozen v2 rule) | ceiling, never called unsupervised |
| selected_all | label-selected over all seventeen static readouts | ceiling |
| shuffle | within-answer token permutation, top5 | null |
| shuffle_consensus | same permutation, consensus rule | null for the consensus rule |
| top5 on erroneous answers only | fixed | inherited control |

Arms per roster: the nine frozen arms (hard equal / spectral / binary L-SML / continuous
L-SML bridge / DS / hierarchical EM; soft equal / spectral / continuous L-SML on cumulative
curves, tau = 1) plus three **pmf** arms (equal / spectral / continuous L-SML fitted on the
per-step softmax mass, the jumps of the cumulative curve; review of v2, section 3). The
binary encodings keep the frozen earliest-wins argmax vote; the tie-split vote proposed in
the review changes the input contract of the binary EM models and is deferred, and the
tied-argmax rate of every chosen readout is recorded per job instead so that the pathology
is measured rather than repaired silently. 300 outer jobs (2 stages x 5 folds x (11 + 11 +
8 task-rosters)), 320 inner jobs for the PRMScore thresholds.

The consensus rule is not the FUSE triplet statistic: Codex's 2026-09-22 execution closed
that statistic as a selector (soft-cumulative sign reversal, q8/fold4 collapse), and this
protocol keeps only its Top-30 anchor. The consensus rule is transductive over the training
answers of each fold and reads no error label.

## Endpoints, anchors, uncertainty

Primary: gate-free ProcessBench SLA (macro over eight cells), macro-F1 under the frozen
CT7 gate, PRMBench within-answer AUROC. Secondary: tolerance-one, early / late, MAE, late
fraction on the four long cells (OlympiadBench and Omni-MATH), step-count strata
(`LENGTH_POSITION_DIAGNOSTICS.csv`), PRMScore at q = 0.8 and with inner-fold thresholds,
coverage, fallbacks, page_cross fallback rate per fold, tied-argmax rate per channel.
Reference rows: CT7 (.3989 / .4119 / .7724), token L-SML (.3592 / .3732 / .7532), token
equal, Mind-the-Gap replay (.2225), the longest-step control (.3178), relative-position
controls, single channels at top5, and the historical dual / context arms. `ANCHOR_PARITY.json`
must show every method shared with the reference run reproduced before any new roster is
tabulated; the report raises otherwise. Tolerance 1e-12 on every ProcessBench number and on
the within-answer AUROC of continuous arms; 1e-4 on the within-answer AUROC of the binary
(`hard__`) arms only, because their discrete scores carry exact ties on PRMBench steps
(1.8 % of within-answer pairs) that 1e-13 platform noise in the saved weights re-orders in
the rank-based AUROC (verified on `prm__fold0__top5`: weights 2.6e-14, scores 2e-13, AUROC
8e-5; the soft arms have no ties and replay exactly).

10,000 shared source-question bootstrap draws; planned contrasts: soft vs binary, pmf vs
cumulative, learned vs equal, continuous / binary bridge, EM vs spectral initialisation,
each roster vs top5, consensus vs selected, selected_all vs selected, ten-channel vs eleven,
shuffle vs original (and shuffle_consensus vs consensus), each new arm vs CT7 and
Mind-the-Gap. One Holm family across the three primary endpoints.

## Execution and immutability

`scripts/experiments/readout_family_v1.py --stage prepare | spectral | em | inner | report`,
config `configs/readout_family_v1.json`, BLAS threads 1. `RUN_FREEZE.json` records the
sha of every fitting module including `step_readouts_v1.py`; the `fusion_utils.py` sha equals
the reference run's. The `cvf_v2` package changes are backward compatible: with the frozen
v2 config every code path reduces to the frozen behaviour and its ten unit tests still pass.

Out of scope: per-channel step offsets, fusion before the readout (HANDOFF_TOKEN_PROBABILITIES
5.1, still open), pseudo-labels, adding CT7 features to the bank, running `cvf_v2/ct7.py`,
retuning k or q_h, and any claim of confirmation: the population is development data.
