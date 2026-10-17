# Line status proposal: codex/cumulative-vote-fusion-v2

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status:
- Cumulative-vote fusion of first-error decisions (binary and soft, SML, L-SML, Dawid-Skene EM, exact hierarchical latent-group EM), on the eleven-channel bank and on CT7's seven profiles: SATURATED
- The `cvf_v2` estimation package (`scripts/experiments/cvf_v2/`, including the Dawid-Skene and hierarchical EM in `em.py`) and the frozen CT7 profiles (`results/cumulative_vote_fusion_v2/ct7_profiles_v1/profiles.npy`): TOOL-ONLY (imported or read by every newer label-free runner)

Owner / last updated: no owner session. Tip bcf5a4bd8 (2026-09-22), identical on origin; contained in `lsml-ct7-levers-run`, `origin/claude/lsml-ct7-levers-v1` and `claude/readout-quickest-detection-v1`. Worktree `.worktrees/cumulative-vote-fusion-v2`.

## Steps on this branch

Own commits on top of a snapshot of the consolidated base (0c7d68265):
- 2026-09-22 [Codex] Cumulative vote fusion v2 (untagged HISTORY block, no step number) - Full-population run of the cumulative-vote design proposed in token-axis Steps 423-424: 230 fit jobs, all selected EM fits converged, no fallbacks, 10 unit tests. Selected soft L-SML stays below CT7: 36.184 against 39.886 gate-free localization, 38.313 against 41.189 common-gate F1, 0.761480 against 0.772397 PRMBench within-answer AUC; learned fusion adds +0.152 over selected soft equal (interval includes 0). Source: `results/cumulative_vote_fusion_v2/` (SUMMARY.csv, PAIRED_CONTRASTS.csv, REPORT_HE.html), `docs/experiments/CUMULATIVE_VOTE_FUSION_V2_SESSION_REPORT.md`.
- Step 421 [Codex] - Repeat on CT7's frozen seven profiles: soft continuous L-SML 39.996 / 41.260 / 0.768632 against CT7 39.886 / 41.189 / 0.772397; localization +0.109 [-0.600, +0.829]; PRMBench within-answer AUC -0.003765 [-0.004880, -0.002680], Holm p = 0.0232. Source: `results/cumulative_vote_fusion_v2/ct7_profiles_v1/` (CT7_CONTRASTS.json, UNCERTAINTY.json, REPORT_HE.html), `docs/experiments/CT7_VOTE_FUSION_V1_SESSION_REPORT.md`. (Collides with `Step 421 [Claude]` white-box extraction; keep both, tagged.)

## Evidence

- `HISTORY.md` blocks above @ `codex/cumulative-vote-fusion-v2`; N = 13,769 answers on corrected source folds with the frozen gate; 10,000 source bootstraps; 232 Holm contrasts for the CT7-profile run (50 outer + 40 inner jobs, 405 models; 1,215 model-prediction replays). The seven-view mean reproduces CT7 to 1.33e-15 with identical peaks.
- `docs/research_notes/lsml_against_ct7_failures_2026-09-23.md` @ `origin/claude/token-axis-fusion-sampling-3i9r2u`, failure "Argmax competition": cumulative-vote L-SML / Dawid-Skene "reproduced the incumbent (Steps 423, v2, CT7-profile run); no non-max rule beats argmax (Step 430 A3) ... closed for weighting".

## What failed (implementation) vs what is still open (direction)

Exhausted on current inputs: re-weighting first-error votes from localizers or profiles that share one output distribution. Every learned variant ties its equal counterpart on ProcessBench and the CT7-profile version loses slightly on PRMBench within-answer ranking.

Not failed: the estimation code. Its Dawid-Skene EM is the "DS estimate" of the whole later label-free line (provenance note, `docs/research_notes/ESTIMATOR_PROVENANCE_2026-09-29_HE.md` section 1, on the estimator branch), and its hierarchical EM is the "HEM" that Omri's 2026-10-01 decision names as the within-group weight of the documented starting point.

## Reopening conditions

Votes from localizers with genuinely different error patterns (a new view or a second model), so that the consensus is not the incumbent.

## Dependencies other lines have on it

- Code imported by absolute path from `.worktrees/cumulative-vote-fusion-v2/scripts/experiments` (`cvf_v2.core`, `cvf_v2.em`, `cvf_v2.scoring`): `er_stage_a.py` (and through it the stage-A/B runners) on `claude/ssl-pseudolabel-residual-v1`, `tensor_mom_stage_a_run.py` on the estimator branch, and (through `cvf_v2.scoring.prm_metrics` / `prmscores`) Step 437 on `lsml-ct7-levers-run`. The collection handoff (`docs/HANDOFF_COLLECTION_2026-09-29.md` section 1) names this as the reason lineage A is not reproducible from its own checkout.
- `results/cumulative_vote_fusion_v2/ct7_profiles_v1/profiles.npy` (8.2 MB, git-ignored): input to Steps 434 and 437 (a copy sits in the main checkout, sha matching the tracked `PROFILE_VALIDATION.json`), to `expectation_realization_run.py` and the other stage runners, and to `b16_fit.py` on `claude/self-generated-step-labels-v1` (there only for reproduction checks, per that line's status file). Script files on the ssl branch that mention this worktree: `er_stage_a.py`, `expectation_realization_run.py`, `er_stage_b_run.py`, `er_stage_b2_run.py`, `er_generality_run.py`, `algorithm_decisions_run.py`, `lsml_merge_step_run.py`, `index_month_lsml_prm_results.py`, `verify_external_banks_v4_source.py`.
- `claude/readout-quickest-detection-v1` reads the frozen profiles and reference metrics from this worktree.

## Outside git but needed

About 404 MB of git-ignored files in `.worktrees/cumulative-vote-fusion-v2` (1,237 files under `results/cumulative_vote_fusion_v2/`), including `profiles.npy` (89.7 MB), `OOF_STEP_SCORES.npz` (38.6 MB), `shuffled_top5.npy` (12.8 MB), `BOOTSTRAP_PRIMARY_DRAWS.npz` (12.7 MB), `channel_merits.npy` (8.5 MB), `OOF_ANSWERS.csv` (4.8 MB), and in `ct7_profiles_v1/`: `BOOTSTRAP_PRIMARY_DRAWS.npz` (11.7 MB), `profiles.npy` (8.2 MB), `OOF_SCORES.npz` (6.0 MB). A `jobs/browser_profile/` cache folder is also inside `ct7_profiles_v1/` (not research data). No Drive archive record for these was found in this check; the 2026-10-01 review lists this worktree's ignored files as not yet uploaded. `RUN_FREEZE.json` and `PROFILE_VALIDATION.json` (tracked) hold the hashes to check a regenerated copy against.
