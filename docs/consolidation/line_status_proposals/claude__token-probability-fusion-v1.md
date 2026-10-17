# Line status proposal: claude/token-probability-fusion-v1

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status:
- Token matrices and derived channel files (`TOKEN_MATRICES.npz`, `DERIVATIVE_CHANNELS.npz`, `CT7_DEV_SCORES.npz` and related): TOOL-ONLY (inputs read by the ssl, decision-rule, estimator, self-generated-label, readout and depth lines)
- Step readouts on the eleven-channel token series (readout width calibration, adaptive per-answer readout, change-point readouts, AR whitening, contiguous windows; Steps 424, 426, 427 and the undated readout commits): SATURATED for the tested implementations (the readout family of Step 429 on `claude/readout-quickest-detection-v1` was also negative)
- Token-level L-SML over the eleven channels (Steps 423 and 425 qualify Step 422): IMPLEMENTATION-NEGATIVE (direction open), see the `codex/claude-feature-bank-token-lsml-v1` status file

Owner / last updated: no owner session. Tip 5cf01f95a (2026-09-19), identical on origin; merged into `consolidation/fusion-2026-09-22` and so contained in `claude/readout-quickest-detection-v1`. Not contained in the ssl line or `lsml-ct7-levers-run` as commits (their HISTORY files carry Steps 423-427 as text; `results/token_probability_fusion_v1/` is on neither). Worktree `.worktrees/token-probability-fusion-v1`.

## Steps on this branch

Own commits on top of the ssl base 72d8235b4 (cb2358514 is the same commit as Codex's 890459866, cherry-picked):
- Step 423 [Claude] - Correction to Step 422: the gate made the comparison invalid but did not hide a token-level win; CT7's gate-free localization 39.89 against token L-SML 35.92, +3.96 [+2.28, +5.59]; the original 7.11-point gap splits exactly into about half locator and half gate; the frozen tail-15 gate beats LOCO-5 for both locators. Source: `results/token_probability_fusion_v1/GATE_HOLD_STAGE_A.json`, `docs/experiments/TOKEN_PROBABILITY_FUSION_V1_STAGE_A.md`. (Collides with `Step 423 [Claude]` cumulative vote and `Step 423 [Codex cross-branch audit]`; keep all, tagged.)
- Undated commits 2026-09-18/19 without step numbers (Stage B two-by-two, length-axis attacks, readout calibration, adaptive per-answer readout, spectral screen, fitting scope and sliding-window variance, feature behaviour, noise floor): learned weighting beats equal only when fusion happens before the step readout; both attacks on the long-chain deficit failed; a fully shuffled pipeline still localizes at 25-30% against a 16.6% chance floor. Source: `docs/experiments/TOKEN_PROBABILITY_FUSION_V1_SESSION_REPORT.md`, `results/token_probability_fusion_v1/STAGE_B_2X2.json`, `READOUT_CALIBRATION*.json`, `ADAPTIVE_READOUT.json`, `PR_NOISE_FLOOR.json`.
- Step 424 [Claude] - Change-point readouts (CUSUM, BOCPD, first-crossing) over our level and the Mind-the-Gap evidence-drop series: nothing reaches 35.92; best non-incumbent 33.67. Source: `results/token_probability_fusion_v1/CHANGEPOINT_READOUT.json`, `docs/experiments/CHANGEPOINT_STEP_READOUT_V1_RESULTS.md`. (Collides with `Step 424 [Claude]` raw-channel readouts on the token-axis line.)
- Step 425 [Claude] - The energy pair: `energy_level` is free to drop (35.90, -0.03 [-0.79, +0.74]); about two thirds of L-SML's margin over equal weighting was L-SML repairing that redundant pair. Source: `results/token_probability_fusion_v1/ENERGY_PAIR_ABLATION.json`.
- Step 426 [Claude] - The step-measurement stage wants more aggregation, not a sharper filter: AR whitening (-2.24 [-3.91, -0.57] at matched width) and contiguous windows (-6.65 [-8.56, -4.72]) both lose; per-model fitting changes nothing. Source: `results/token_probability_fusion_v1/STEP_MEASURE.json`, `docs/experiments/STEP_MEASURE_V1_RESULTS.md`.
- Step 427 [Claude] - Corrections after two independent audits: two code bugs fixed, four claims withdrawn, and a missing length control found (choosing the longest step scores 31.78; residualizing on log step length costs -10.94 points at K = 10). Source: `docs/experiments/AUDIT_CORRECTIONS_20260919.md`.

## Evidence

- `results/token_probability_fusion_v1/*.json` @ `claude/token-probability-fusion-v1`; population: 4,442 erroneous ProcessBench answers over 8 cells for gate-free localization, 13,769 answers overall; 10,000 paired source-group draws; anchor 35.92 replayed with difference 0.0 in every script (Step 427).
- Step 427 block: margin over the step-length heuristic is +4.1 points for the deployed readout and +5.2 for the grid maximum (CT7 +8.1); under the length control the "Top-20 gain" is +0.24 [-0.89, +1.39].
- `results/token_probability_fusion_v1/DERIVATIVE_CHANNEL_EVAL.json`: on ProcessBench the derivative-only bank scores 0.2796 (equal) and 0.2370 (L-SML) against 0.3484 / 0.3495 for the level bank and 0.3989 for CT7; level plus derivative does not beat level.

## What failed (implementation) vs what is still open (direction)

Exhausted on these channels: every attempt to sharpen the step statistic (whitening, contiguous windows, change-point rules, adaptive width) and the readout-choice axis generally (with Step 429's readout family on the readout line). The surviving direction from Step 426/427 is "more aggregation", with its size and mechanism withdrawn.

Still open:
- Learned pooling of a step's tokens (weighting, not localizing more precisely). A graph or self-supervised step encoder was named for this; GPU encoder work (S5) is outside scope per Omri's 2026-09-23 decision, so only a CPU form would be admissible.
- Whether a change-point readout helps a better token-level series (Step 424 left this explicitly untested).
- The step-length confound in this line's readouts (Step 427): open. The white-box step-length follow-up (Stage 1b section 5) is OUT OF SCOPE FOR NOW (Omri, 2026-10-02; not closed). Whether a general step-length check stays a separate checklist item is for Omri.
- Derivative channels used per channel rather than as a fused bank: the ssl line's 13-channel bank adds the realized-token derivative (`realized_drv`) to the eleven level channels it reads from `DERIVATIVE_CHANNELS.npz`, with `TOKEN_MATRICES.npz` as a second input (`expectation_realization_run.py` lines 42 and 93), so this is carried there.

## Reopening conditions

- Readouts: a new token series (new view or second model) worth reading sequentially, pre-registered with the step-length heuristic and a boxcar aggregation control as mandatory references (both were missing in Steps 424-426 and caused the withdrawals).
- Learned pooling: Omri's go for a CPU-only pooling experiment with the same length control.

## Dependencies other lines have on it

Read by absolute path from `.worktrees/token-probability-fusion-v1/results/` by:
- `claude/ssl-pseudolabel-residual-v1`: 21 script files outside source snapshots (for example `expectation_realization_run.py`, `er_stage_b_run.py`, `algorithm_decisions_run.py`, `lsml_merge_step_run.py`, `bank20_lsml_run.py`, `fit_external_source_bundle.py`, `mtg_reproduction_extract.py`).
- `claude/estimator-provenance-collection-2026-09-29`: `tensor_mom_stage_a_run.py` (`DERIVATIVE_CHANNELS.npz`).
- `claude/decision-rule-v1`: its status file lists `TOKEN_MATRICES.npz` and `DERIVATIVE_CHANNELS.npz` as upstream inputs.
- `claude/readout-quickest-detection-v1` (Steps 428-432: `TOKEN_MATRICES.npz`, `CT7_DEV_SCORES.npz`, `OOF_SCORES.npz`) and `claude/self-generated-step-labels-v1` (`b16_fit.py`, reproduction checks).
- `claude/depth-feature-fusion-v1` runners (`DERIVATIVE_CHANNELS.npz`, `TOKEN_MATRICES.npz`, `CT7_DEV_SCORES.npz`).

## Outside git but needed

About 537 MB of git-ignored files in `.worktrees/token-probability-fusion-v1/results/` (113 ignored paths in the worktree), none with a verified Drive archive record found in this check (the 2026-10-01 reconciliation lists other worktrees' ignored files as "not yet uploaded"; the decision-rule status file says these two are "not archived by this line"):
- `token_probability_fusion_v1/TOKEN_MATRICES.npz` (307.9 MB; eleven-channel risk-oriented token bank, 6,968,779 tokens; about 95 minutes to rebuild per `docs/HANDOFF_MIND_THE_GAP_MECHANISM.md`).
- `token_probability_fusion_v1/EVIDENCE_DROP.npz` (57.0 MB), `C1_TOKEN_SERIES.npz` (55.9 MB), `DERIVATIVE_CHANNELS.npz` (19.1 MB), `FITTING_SCOPE_SCORES.npz` (15.5 MB), `STAGE_B_SCORES.npz` (8.9 MB), `DERIVATIVE_ARM_SCORES.npz` (7.8 MB), `STEP_VIEWS_12CH.npz` (5.9 MB), `ADAPTIVE_READOUT_SCORES.npz` (5.6 MB), `CHANGEPOINT_PEAKS.npz` (0.9 MB).
- `chosen_token_calibration_v1/OOF.npz` (15.6 MB), `CT7_DEV_SCORES.npz` (1.1 MB; restored from the Drive backup `gdrive:hallucination_detection/consolidated_results/local_backup_2026-09-17/atlas/`, frozen-candidate JSON byte-identical to master's), `extracted/` and `extracted_sufficient/` files.
- `length_explicit_ct7_v1/bank/*.npz` and `bocpd.npz` (about 12 MB), `localization_full_benchmark_v3/evaluation/JOINED.npz` (10.6 MB).
The raw inputs for rebuilding are in the 2026-09-17 Drive backup (`docs/HANDOFF_TOKEN_PROBABILITIES.md`: "the Drive backup has the inputs").
