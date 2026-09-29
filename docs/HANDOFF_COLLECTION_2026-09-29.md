# Handoff: collecting the last two weeks of experiments into one line (2026-09-29)

Read after `PROGRESS.md` and `LESSONS.md`. Written on branch `claude/estimator-provenance-collection-2026-09-29`, based on
`claude/ssl-pseudolabel-residual-v1` at Step 462 (716892a8), HISTORY Step 463.
Purpose (Omri, 2026-09-29): package this session and give the next branches a map from which to
**collect every experiment of 2026-09-15 to 2026-09-29 into one line and continue in one direction**.
Everything below was checked with `git` on the remote branches on 2026-09-29. Nothing here was merged yet.

## 1. The situation in one paragraph

There are **two lineages** that diverged on 2026-08-23 (merge-base f7f7801a):

- **Lineage A, the consolidated line.** `claude/ssl-pseudolabel-residual-v1` holds Steps 421-462: SSL plan,
  bank20 and bank-extension L-SML, the expectation/realization stages, the DS filter, algorithm decisions,
  external checks and position. The external line `codex/lsml-external-generalization-v1`,
  `claude/er-generality-v1` and `codex/token-local-fusion-optimization-v1` are already inside it.
- **Lineage B, the Codex/competition line.** It has 124 result directories that are **not** in lineage A.
  Its tip, `claude/lsml-ct7-levers-v1`, contains `claude/competition-sync-2026-09-23`,
  `codex/cumulative-vote-fusion-v2` and `origin/master` (2026-08-26). Every result directory of
  `codex/fusion-independence-atlas-v1`, `codex/temporal-research-20260915` and
  `codex/renyi-position-temporal-fusion-v1` is also present in it.

Lineage A's runners **read lineage-B outputs through absolute Windows worktree paths**. For example,
`expectation_realization_run.py` reads `MAIN/.worktrees/cumulative-vote-fusion-v2/.../ct7_profiles_v1` and
`.worktrees/token-probability-fusion-v1/...`, and `er_stage_a.py` imports `cvf_v2` from
`.worktrees/cumulative-vote-fusion-v2`. So lineage A is **not reproducible from its own checkout**. This
is the practical reason to collect.

A full consolidation into a local `master` already happened on 2026-09-17. It covered all 27 worktree
branches (`docs/HANDOFF_localization_2026-09-17_evening.md` on the atlas branch, section B1), but it was
**never pushed**: the repository is over its GitHub LFS budget, and that `master` references about 17 GB of
LFS-tracked `dataset_cache/repgrid`. `origin/master` is still at 2026-08-26.

## 2. Branch map (activity since 2026-09-15)

"Own commits" is `git rev-list --count origin/claude/ssl-pseudolabel-residual-v1..<branch>`. The conflict
count comes from `git merge-tree --write-tree` against this branch.

| Branch | Last commit | Relation to lineage A | Own commits | Result dirs not in A | Conflicting files on merge |
|---|---|---|---|---|---|
| `claude/ssl-pseudolabel-residual-v1` | 09-29 | **the base (A)** | 0 | 0 | - |
| `claude/estimator-provenance-collection-2026-09-29` | 09-29 | A + Step 463 (**this branch**) | Step 463 only | 0 (code + docs, no run yet) | - |
| `claude/tensor-mom-estimator-v1` | 09-29 | earlier prefix of this branch (estimator code only) | - | 0 | - |
| `claude/er-generality-v1` | 09-28 | merged into A | 0 | 0 | - |
| `codex/lsml-external-generalization-v1` | 09-24 | merged into A | 0 | 0 | - |
| `codex/token-local-fusion-optimization-v1` | 09-23 | merged into A | 0 | 0 | - |
| `claude/lsml-ct7-levers-v1` | 09-23 | **tip of B** | 15 + lineage B | **124** | 11: `.gitattributes`, `.gitignore`, `CLAUDE.md`, `HISTORY.md`, `PROGRESS.md`, `Research_Directions.md`, `docs/HANDOFF_TOKEN_PROBABILITIES.md`, `papers/index.md`, `scripts/test_token_local_fusion.py`, `spectral_utils/fusion_utils.py`, `spectral_utils/token_local_fusion.py` |
| `claude/competition-sync-2026-09-23` | 09-23 | inside `lsml-ct7-levers-v1` | 8 | (subset) | - |
| `codex/cumulative-vote-fusion-v2` | 09-22 | inside `lsml-ct7-levers-v1` | 7 | (subset) | - |
| `codex/fusion-independence-atlas-v1` | 09-17 | B; not an ancestor of the tip, but all its result dirs are in the tip | 186 (232 vs tip) | 112 (all also in tip) | 8 |
| `codex/temporal-research-20260915` | 09-16 | ancestor of atlas | 158 | (subset) | - |
| `codex/renyi-position-temporal-fusion-v1` | 09-15 | ancestor of temporal | 137 | (subset) | - |
| `claude/token-probability-fusion-v1` | 09-19 | branched from A at 72d8235b (09-18) | 30 | 4, of which `results/token_probability_fusion_v1` is in neither A nor B's tip | 1: `HISTORY.md` |
| `claude/token-axis-fusion-sampling-3i9r2u` | 09-23 | branched from A at 72d8235b | 11 (65 vs B's tip) | 2: `results/cumulative_vote_fusion_v1`, `results/raw_channel_readout_fusion_v1` | 2: `HISTORY.md`, `PROGRESS.md` |
| `codex/claude-feature-bank-token-lsml-v1` | 09-18 | its one own commit also appears in `token-probability-fusion-v1` | 5 | 0 | - |

## 3. What ran, by lineage (HISTORY headings, two weeks)

Step numbers collide between lineages. CLAUDE.md says to tag, never renumber; keep both blocks.

**Lineage B (Codex/competition, 09-15 to 09-23), on `claude/lsml-ct7-levers-v1`:**
- 390-397 [Codex]: shared missed errors, readout/impossibility audit, step-evidence fusion, digit
  disagreement (later excluded, 09-17 decision), tail normalization audit, alternative views, fusion independence atlas.
- 398-399 [Claude/Codex]: Joint L-SML pair route, redundancy robustness, v-free within-group readout; native SLA.
- 399-412 [Codex, 09-17]: digit-free broad50, gate/locator attribution, Joint feature selection + BOCPD,
  structured stress and membership (406-409 are also in A).
- 413-420 [Claude, 09-17]: digit-free 20-stream ladder, fewer than 3 conditionally independent signals,
  chosen-token calibration, de-spiked entropy-free view, **CT7 frozen (Step 418)**, position/length as evidence.
- 421-424 [Codex/Claude]: cumulative-vote fusion (v1 on `token-axis`, v2 on `cumulative-vote-fusion-v2`), raw-channel readouts.
- 428-432 [Claude, readout-qd/competition]: quickest-detection diagnostics, readout family, competition
  diagnostics, per-channel likelihood-ratio evidence.
- 433-437 [Claude/local, ct7-levers]: three L-SML levers (family-equal, token-level L-SML, gated window),
  gate correction A1, **supervised PRM measured beside CT7** (Step 437 [prm-measure]).

**Token-probability line (09-18 to 09-19), on `claude/token-probability-fusion-v1`:** Steps 423-427
[Claude]: correction to 422, change-point readouts, the energy pair, more aggregation not a sharper filter,
corrections after an independent audit.

**Lineage A (09-21 to 09-29), on `claude/ssl-pseudolabel-residual-v1`:** Steps 421-422 (white box,
token-level L-SML), 423/430/431/432 [Codex audits and plans], 433-437 [SSL plan S0-S5], 438-443 and 445
(bank20, weak channels, declared partitions, family/tail, calfix, tail threshold, maximal-step tail),
444/446 (family-tail external), 447-449 (partition ceiling, Mind-the-Gap audit and reproduction),
447-448 [Codex digit alternative/numeric family], 450-452 (expectation vs realization A/B/B2),
453 (calfix of six runners), 454 (PRMScore decomposition), 455 (er_generality), 456 (merge step),
457 (algorithm decisions), 458 (partition switch), 459 (external), 460 (position channel),
461 (position prior), 462 (per-dataset fit), **463 (this: estimator provenance, tensor MoM, this map)**.

## 4. The one direction, as of Step 462

This is the state of lineage A, which is the most recent development line. Details are in
`docs/research_notes/ESTIMATOR_PROVENANCE_2026-09-29_HE.md`, section 4.

- **Method:** answer-standardized channels, then top-20% within-answer marks, then the label-free
  **Dawid-Skene filter** (π̂ > 0.5), then the **plain average** of the surviving continuous channels.
  The step index is added as a channel (Step 460). Fit per model per dataset (Step 462). Optional:
  grouped with DS group weights and the **position prior** (Step 461).
- **Best PRMBench within-AUC:** 0.8002 plain + position channel and 0.8056 grouped + prior, both on 13+digit.
  The cross-fitted slope reaches 0.8089 but was not adopted. References: CT7 0.7724, fam421 0.7801.
- **Open problems:**
  - No label-free estimate of how much position should count.
  - ProcessBench (first error) wants no position, PRMBench wants a lot.
  - The per-cell prior latches onto a step-0 telemetry artefact on ProcessBench.
  - The external check is deferred. MedPRMBench is still the untouched test.
- **Standing constraints (CLAUDE.md):**
  - Digit-disagreement features are excluded (09-17). The three decoding-independent digit features need
    Omri's scope decision.
  - No new GPU training.
  - PRMBench is primary and ProcessBench secondary (09-23).
  - Ask Omri before choosing which method to evaluate.

## 5. Suggested collection procedure for the next branch

1. **Start from this branch, not from `main`.** `main` is the unrelated early package skeleton.
   Name the new branch after its purpose, for example `claude/collection-2026-09-29`.
2. **Merge in order, one merge commit each.** Never rebase another agent's branch.
   - a. `claude/token-probability-fusion-v1`. Only `HISTORY.md` conflicts. It brings
     `results/token_probability_fusion_v1`, which A's runners read from a worktree.
   - b. `claude/token-axis-fusion-sampling-3i9r2u`. `HISTORY.md` and `PROGRESS.md` conflict.
   - c. `claude/lsml-ct7-levers-v1`, the big one: lineage B, 124 result dirs and CT7's origin.
3. **Resolve the conflicts:**
   - `HISTORY.md`, `PROGRESS.md`, `Research_Directions.md`: **union**, then count `### Step` blocks on both
     sides against the result (CLAUDE.md; merge commit `cd423ab` is the worked example). Never use `--ours`/`--theirs`.
   - `.gitattributes` / `.gitignore`: union of both rule sets. Lineage B also sets `HISTORY.md`,
     `PROGRESS.md` and `Research_Directions.md` to `merge=union`, which is worth keeping. Keep A's
     byte-identity rules (`spectral_utils/external_generalization/** text eol=lf`, the digit and family rules).
   - `CLAUDE.md`: keep both dated update blocks, newest first.
   - `spectral_utils/fusion_utils.py`, `spectral_utils/token_local_fusion.py`,
     `scripts/test_token_local_fusion.py`: **these are code, and need a real read.** Run both lineages'
     tests after resolving. A frozen result whose `SOURCE_SNAPSHOT` hash no longer matches must stay
     frozen: do not "fix" it.
4. **Push blocker (LFS):**
   - Lineage B's `.gitattributes` tracks `dataset_cache/**/*.pkl` in LFS. Check with
     `git lfs ls-files` which LFS objects the merged branch references before pushing.
   - The 09-17 `master` failed on exactly this, at about 17 GB. The decision is Omri's: raise the budget,
     or stop tracking `dataset_cache` on the collection branch.
   - Ordinary branches without new LFS content still push.
5. **Make lineage A reproducible from one checkout.** After the merge, replace the
   `MAIN / '.worktrees/<branch>/...'` paths with repo-relative paths **in new runners only**. Frozen runners
   keep their recorded paths: their `CODE_MANIFEST`/`SOURCE_SNAPSHOT` hashes are the record.
6. **Verify before pushing:**
   - No conflict markers.
   - Step-block counts reconcile.
   - `pytest tests/` passes as far as it can without the local data. Tests that import `cvf_v2` from the
     Windows worktree fail in a cloud checkout; this is pre-existing (3 in `tests/test_er_stage_a.py`).
   - The frozen-candidate modules parse.

## 6. Traps met in this session

- **Two files are permanently "modified" in a Linux checkout of A:**
  `spectral_utils/external_generalization/_bank11/__init__.py` and `.../chosen_token_calibration.py`.
  They are committed with CRLF, but `.gitattributes` says `text eol=lf`. The diff is line endings only
  (`git diff --ignore-cr-at-eol` is empty).
  - They are **hash-locked** (the "preserve byte identities of the externally evaluated method" rule), so
    do not commit a renormalized copy.
  - Locally, `git update-index --assume-unchanged <file>` quiets them.
  - Do not use `git stash` in such a checkout: the pop is refused, and the repo's `guard_git` hook blocks
    `stash drop`.
- **"DS" in a report means Dawid-Skene EM (1979).** It is not the Jaffe-Nadler-Kluger (2015) tensor
  method, which Step 463 implemented but has not yet run on data. See the provenance note.
