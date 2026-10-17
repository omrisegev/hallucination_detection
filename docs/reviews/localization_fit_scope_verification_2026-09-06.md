# Verified fit scope versus the intended one-answer matrix

Verified 2026-09-06, by source inspection and label-free bookkeeping only. Worktree: `C:/Users/omris/TAU/hd_jlsml_v2_wt`, branch `claude/joint-lsml-optimization-v2`, HEAD `ff800082468ecceca4ef15217c5b045314b5fea6`. The tracked worktree was clean. Windows process 137068 remained present with command `scripts\joint_lsml_optimization_v2\run_v2.py structure --workers 4`, started September 5 at 17:46:32. On-disk source and completed artifacts agree about the fit scope; this inspection did not introspect Python's in-memory module objects.

## User's clarified design

For each individual question/answer pair, divide that answer into N steps or token windows. Extract the same P chosen feature definitions from each segment. Build an N-by-P matrix and fit fusion using only that answer's observations. Repeat the recipe independently for each answer. Do not confuse repeated evaluation over many answers with fitting a shared model across those answers.

The running v2 does not implement this fit scope. A within-answer-centered pooled donor model, proposed in the previous review, would also be a different design.

## Concrete completed-fold evidence

Cell `pb_gsm8k_q4`, outer fold 0:

| Bookkeeping item | Observed value |
|---|---:|
| Total response rows | 400 |
| Outer-training response rows | 320 |
| Eligible tokens in those training answers, before the cap | 91,954 |
| Module-A fit token cap | 60,000 |
| Module-A retained feature count | 23 |
| Steps in those training answers, before the length filter | 1,671 |
| Recorded Module-B full-length fitting steps | 1,667 |
| Module-B ranked token-risk slots per step | 10 |

Therefore Module A constructs a 60,000-by-23 donor fit for this fold. Module B's recorded fit is 1,667-by-10, pooling steps from the training responses. Neither is a separate N-by-P fit for each answer.

Cell bundle header shapes: `raw.npy` = (114502,29), `row_ids.npy` = (400,), `step_starts.npy` = (2082,). Read only `group_ids`, `token_offsets`, `step_row_offsets`, array headers, the group-fold map and `moduleb_meta.json`'s `b1.n_fit_steps`. No correctness labels, token-risk arrays, fitted weights, or evaluation outcomes were read. No fit or evaluation was run.

## Code chain

All paths below are relative to the v2 worktree.

- `scripts/joint_lsml_optimization_v2/run_v2.py:99–123`: `_write_cell` stacks response token matrices with `np.vstack`, retaining response and step offsets.
- `run_v2.py:255–259`: `_row_masks` selects all response rows outside the held group fold.
- `spectral_utils/joint_lsml_localization.py:118–147`: `prepare_active23` concatenates eligible token indices across selected response rows, caps them at 60,000, and standardizes the combined fit.
- `spectral_utils/joint_lsml_v2_localization.py:268–300`: `fit_v2_arms` fits weights from that combined `standardized_fit`.
- `run_v2.py:351–367`: `_run_cell` makes one outer fit per fold, then applies Module B; there is no per-answer fusion fit loop here.
- `run_v2.py:320–330`: `_module_b` uses answer ownership to select all training steps, then calls `fit_orderstat_weights` once on their combined matrix.
- `spectral_utils/trajectory_reducer.py:35–64,129–149`: descending top-ten extraction, followed by a fit across full-length steps. Columns are ranks of already-fused token risks, not the original P telemetry features.
- `scripts/joint_lsml_optimization_v2/second_pass_amendments.py:171–210`: R1 likewise selects full-length training steps, standardizes slots jointly and passes owner IDs to the grouping procedure. Owner IDs provide grouped stability checks; they do not make this a separate fit per answer.

“Training” here means a label-free structure-fit population. It does not imply that Module A or the unsupervised Module-B rows fit to correctness labels.

## Why ten?

`docs/experiments/REASONING_LOCALIZATION_03662_ANCHOR_V1.md:1146–1181` records the earlier top-ten checkpoint. In that contract, top-ten PB macro F1 was 0.3581627690 versus top-five 0.3452267399, an increment of 0.0129360292. The completed eleven-contrast interval at lines 1380–1388 is [0.0014438677,0.0252246398]. It was the raw best development reducer but did not clear the lower-bound practical-benefit requirement of 0.005. It was not a universally optimal or independently confirmed choice.

v2 protocol Section 6B explicitly reuses that incumbent: B0 averages the largest min(10,L) token risks on PB, while B1 learns weights for those ranked slots. PRMB's original B0 is official-span maximum. Ten does not mean ten adjacent tokens, ten original features or ten steps. It is not required by fusion or by the user's intended matrix.

## Fixed-window implications

Fixed width w produces about ceil(T/w) rows for T tokens, with a declared final-partial-window policy. N can vary across answers; only P needs to remain consistent. Variable-length reasoning steps also produce a rectangular N-by-P matrix if each yields P scalar features. Fixed width is useful for measurement comparability, not required for matrix rectangularity.

Some response features need sufficient window length or lose variation in fixed-length windows; for example, window length itself is constant. Spectral features may be unreliable on very short segments. The exact feature mapping and degenerate-feature handling must be stated rather than silently replacing P.

Fusion learns feature weights and assigns a score to each window. Learning window positions, widths or stride is another task. Within-answer centering yields relative evidence and requires a separate explicit no-error decision. Offline fitting over the whole answer must not be described as causal streaming. With N centered observations, sample covariance rank is at most min(P,N-1); overlapping windows do not create independent observations. Regularization can permit a numerical fit without establishing stability.

This is a verified architectural mismatch and a proposed clarification, not an implementation change. The current run remains untouched.
