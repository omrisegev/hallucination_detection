# Handoff: localization line, 2026-09-17 (evening)

Branch `codex/fusion-independence-atlas-v1`, worktree `.worktrees/fusion-independence-atlas-v1`.
Head at writing: `135bb4da0`, in sync with origin. 23 commits today (Steps 413-420 [Claude]) on top of
Codex's Steps 397-412. Everything below is DEVELOPMENT evidence on the frozen 13,769-answer population;
nothing is confirmed.

## 1. What this session established, in one line each

1. **Step 413** - on the digit-free 20-stream bank, no fusion rule beats block averaging: partition,
   SML at every K from 3 to 8, Continuous L-SML, Joint reliability weights, IU-PCR. Equal weight 39.19 PB.
2. **Step 414/415** - the reason is structural: the evidence families carry **2.46** conditionally
   independent signals (labels used for measurement only). Below three, the L-SML eigen-stage is
   undetermined (Step 205) and its registered guard returns equal weights, so averaging IS the method's
   correct output. Tail = entropy (.64), prefix innovations stay .87-.96 with their parent family.
3. **Step 415** - first evidence channel built to satisfy the assumption: the chosen-token standardized
   excess surprisal `(-log q(x) - H)/sqrt(VE)` is entropy-free on real data (token-level correlation
   with entropy -.016, versus .337 for raw surprisal).
4. **Step 416/417** - as a length-free pooled step z-test with its step-0 spike removed, it improves the
   six-stream locator on **both** benchmarks. The step-0 spike (+2 answer-SD at the first step) was the
   whole PB/PRMB disagreement. Control: the token evidence beats the position prior alone by +1.59 PB.
5. **Step 418** - candidate **CT7** frozen: 41.19 PB / .7724 within. Its seven views are **1.80**
   effective independent signals, so this is new evidence, not a fusion-weighting result.
6. **Step 419** - position and length as extra views: measured, not built. One position feature exists
   within an answer; answer length is between-answer only (gate material).
7. **Step 420** - Omri's proposal to make step length explicit: **rejected by the data**. Calibrating the
   length out of the six streams costs 8.16 PB points and .059 within; a declared log-length view returns
   only 3.35 of them. The first error IS the longest step in 29.7% of PB error answers (chance 15.5%).
   Length is evidence, not a hidden prior, and the coupling is not additively separable.

## 2. Current state

* **Frozen candidate CT7** (`CT7-six-entropy-plus-despiked-chosen-token-z-equal-v1`): manifest with code
  and input hashes in `results/chosen_token_calibration_v1/FROZEN_CANDIDATE_CT7.json`, code
  `spectral_utils/frozen_locator_ct7.py`, exact-replay freeze script
  `scripts/freeze_candidate_ct7_v1.py` (refuses drift; run it to verify nothing moved).
  **Do not modify it.** Any change is a new candidate id.
* Comparators under the same gate: six streams equal 40.27/.7589; frozen BOCPD-corrected innovation5
  40.37/.7632; equal20 39.19/.7558; H1 alone 36.35/.7301.
* CLAUDE.md, Research_Directions.md, PROGRESS.md and HISTORY.md on this branch are up to date.

## 3. The open question Omri and I converged on

L-SML has never been tested where it is defined (more than three conditionally independent views).
Two coherent configurations:

| configuration | rows per answer | can fit per answer? | effective views measured |
|---|---|---|---|
| pooled fitting + step level (today) | 8 median steps | no (p > n) | 1.8 to 2.5 |
| answer-local fitting + 8-token windows | 49 median windows | yes, with shrinkage in the short tail | moment bank 3.4 to 3.9 (only 24 answers) |

The proposed next measurement (NOT authorized yet, one variant): extract a ~10-view moment-style bank
(base streams crossed with level / sd / slope) on 8-token windows for the full population and measure the
participation ratio at window and step level. Level, sd and slope are genuinely different functionals,
unlike smoothings and innovations of one series, which we have now shown add nothing. If it clears 3, we
finally have a setting where L-SML versus averaging is a fair test, and it also satisfies the thesis's
answer-only objective. Shrinkage (`spectral_utils/shrinkage_iu.py`) belongs there, with a label-free
alpha and a declared distance from the average; at step level it is forced toward the diagonal, i.e.
toward averaging.

## 4. How to reproduce the data on another machine (important)

The code and documents travel with git. **The data does not**: every npz under `results/` is gitignored.
To rebuild from the raw pickles in `dataset_cache/` (times measured on this machine):

```bash
python scripts/run_digitfree_broad50_v1.py          # bank extraction, ~270 s (extract() only if you just need features)
python scripts/run_chosen_token_calibration_v1.py   # chosen-token step values, ~35 s
python scripts/run_chosen_token_calibration_steps_v2.py  # per-step sufficient statistics, ~27 s
python scripts/freeze_candidate_ct7_v1.py           # verifies the frozen candidate replays exactly
```

The BOCPD residual additionally needs the temporal worktree
`.worktrees/temporal-research-20260915` (branch `codex/temporal-research-20260915`): its
`results/aligned_context_predictors_v1/SCORES_FROZEN.npz` and `results/temporal_context_data_v1/`
(features.npy, METADATA.json, step_spans.npy) are gitignored too, and the token-level recomputation in
`scripts/run_length_calibrated_streams_v1.py` reads them directly.
**Before any sparse-checkout or branch surgery, back these up to Drive** (`rclone`, remote `gdrive:`).

## 5. Traps this session paid for

* `git sparse-checkout add/set/reapply` in this worktree **deletes ignored files outside the cone**. It
  wiped Codex's local-only extractions; they were rebuilt from the raw pickles. Use `git add --sparse`.
* The frozen bank extraction is stored as **float32**. A float64 recomputation matches only after casting;
  exactness gates must say so (Step 420 amendment 1).
* Indexing an `np.load` archive inside a per-answer loop decompresses the whole array every time. Two
  loaders had this; fixed. It turned 30-second analyses into 20-minute ones.
* Codex's `RUN_STATE.json` files are tracked: re-running an extraction overwrites them. Restore with
  `git checkout --` afterwards.

## 6. What not to do

Do not modify CT7. Do not read the historical per-bank participation numbers as quality (some rest on 24
answers, different populations and models). Do not open a broad sweep: one variant, one discussion, then
build. Omri wants to be consulted on direction before new builds.

---

# Part B: session close, repository consolidation and open decisions (added late 2026-09-17)

## B1. Repository state

* **master is now the consolidation of everything.** All 27 branches that had a worktree were merged
  into it, oldest first (`backup/master-20260917-2143` tags the state before). Verification after the
  merges: no conflict markers in tracked text, 791 `### Step` blocks in HISTORY.md with no branch losing
  a block, the frozen-candidate modules present and parsing, their hashes matching the manifest.
* **Conflicts and how they were resolved** (only three): `.gitattributes` and `.gitignore` by union of
  both rule sets, and both files are now `merge=union` along with HISTORY.md, PROGRESS.md and
  Research_Directions.md; `docs/experiments/CIW_DEEM_V1.md` took the newer branch text, whose
  "Application adapters" section explicitly supersedes the older "Task boundary" section;
  `docs/research_notes/CONDITIONAL_IU_SECOND_MACHINE.md` was edited independently on two branches for a
  Git LFS checkout and for a git-only checkout, and **both routes were kept**, the second under its own
  heading.
* **master is NOT pushed. This is the one open blocker.** GitHub reports the repository is over its Git
  LFS budget and the pre-receive hook declines the push. Twelve LFS snapshots that existed only locally
  were removed in their own commit (their history stays on `codex/conditional-iu-followups-v1`,
  commit 54fc3fa99), which was not enough: master still references the LFS-tracked
  `dataset_cache/repgrid` caches, about 17 GB. Ordinary branches without LFS content still push fine.
  **Decision needed from Omri**: raise the LFS budget, or stop tracking `dataset_cache` in master.
  Until then master exists only locally, merged and verified, with the backup tag above.

## B2. Data locations after the cleanup

Backed up to Drive at `gdrive:hallucination_detection/consolidated_results/local_backup_2026-09-17/`:

* `atlas/` - today's extractions and scores: the digit-free bank extraction, the chosen-token step values
  and per-step sufficient statistics, the BOCPD input and historical scores, the length-calibrated
  readouts, the frozen CT7 development scores, and the atlas evaluation and baseline-replay archives.
* `temporal/` - the token-level telemetry the BOCPD channel is built from
  (`temporal_context_data_v1`: features.npy, METADATA.json, step_spans.npy, MANIFEST.json) and the frozen
  predictor and baseline score archives.

Deliberately **not** backed up, and lost with their worktrees: a partial varentropy run (3.7 GB, stopped
at 576 of 13,769 answers on 2026-09-12) and roughly 4 GB of older RBM-family results whose findings are
already recorded in tracked JSON and HISTORY blocks now on master.

Worktrees: the main checkout and `.worktrees/a6-s0b` (which holds master) are kept; the other 25 are
removed. **Deleting a worktree does not delete its branch**, so every lineage remains reachable by name.
Two worktrees were locked with git's default "initializing" reason, not to protect anything; one
worktree's RUN_STATE said RUNNING but had not been touched for five days and no process existed.

## B3. The plan agreed for the next session (measurement first, not authorized to run yet)

The live hypothesis after Steps 413-420: we have been testing L-SML outside its domain of definition,
and the way in is the **representation**, not the fitting method.

Evidence: a step-level answer has a median of 8 steps against 20+ features, so an answer-local covariance
is not estimable; 8-token windows give a median of 49 rows per answer. Measured effective independent
views are 1.8 to 2.5 for entropy-transform banks at step level, but 3.4 to 3.9 for the moment bank
(level / sd / slope), because those are different functionals rather than smoothings of one series.
Prediction residuals and prefix innovations add at most 0.15, and transforms of one stream add nothing.

Proposed single bounded measurement: build a roughly ten-view bank on 8-token windows for the full
13,769-answer population (one view per measured evidence family, plus sd and slope moments of two or
three base streams) and measure the conditional participation ratio at window and step level. Only if it
clears three does an L-SML versus equal-weight test become meaningful; that test would then use
answer-local fitting with shrinkage (`spectral_utils/shrinkage_iu.py`), a label-free alpha, and a
reported distance from the equal-weight solution. At step level shrinkage is forced toward the diagonal,
i.e. toward averaging, so it only makes sense in the window representation.

## B4. Open decisions waiting for Omri

1. The LFS blocker above, which is what keeps master unpushed.
2. Whether to run the window-representation measurement in B3, and with which ten views.
3. Whether CT7 goes to an untouched confirmation now, and on which data, since the cached population is
   already development data.
