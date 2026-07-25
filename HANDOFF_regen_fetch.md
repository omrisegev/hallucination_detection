# HANDOFF — Fetch the 46-view-coverage regen wave, map to legacy cell keys

**Written**: 2026-07-20. **Goal owner**: next session.

**STATUS (2026-07-20, Step 190): items 1-4 DONE.** 33/35 preset dirs fetched + integrated into
`cache/repgrid/<preset_id>/` (mechanical move; `internalstates_gsm8k_qwen25_7b` archived-and-swapped
per the collision noted below). gpqa T-mislabel propagation checked — it's bigger than assumed,
see HISTORY Step 190 for the `Qwen-7B_T1.0` finding. **Still open**: item 5 (fetch the 2 remaining
`_mn4096` dirs once `sacct` shows COMPLETED — they were ~32-36% done as of this check) and item 6
(this file is the HISTORY writeup source — Step 190 already covers it, no separate action needed).
Phase 5 was deliberately NOT started (Step 189's re-scoping gate is still standing). Rest of this
file is the original plan, kept for reference.

## Where we are

The full 32-preset "46-view coverage" regen wave (repeated backfill + Colab-tier
regeneration effort tracked since `HANDOFF_full_coverage.md`) is **done on the cluster**.
Additionally, 3 new "long-cap" flavour presets were added on top — **additive, not
replacing** the originals (Omri: "we will add another cell, not replace. we shou;d have
both flavours"). This session did the submission + one comparison pass; **nothing has been
fetched to local yet**. That's the next session's job.

### 1. Backfill stream — closed (prior session, unrelated to this wave)
19 repgrid cells backfilled to full Z_n coverage, Gate C Δ=0.0000 on all. All Colab-era
cells including the two CoT qa cells (trivia_qa_cot 98.7% coverage, webq_cot 98.0%,
`--tolerate-skips`) are swapped in under `cache/repgrid/`. Not this handoff's concern —
already landed, just noted for completeness.

### 2. Full-N regen wave — 32/32 DONE on cluster, 0 real failures
All 32 presets (20 RAG, 3 math500 T=1.5, 5 gpqa T=1.0, `internalstates_gsm8k_qwen25_7b`,
3 tier-3 trace regens) finished. Every "FAILED exit 85" in `sacct` across the whole wave
is the walltime-resume convention (job hits the 8h wall, checkpoints, exits 85, chained
successor picks up) — every one has a completed successor, zero genuine failures anywhere.
Output lives on the cluster at `$S/results/regen/<preset>/` (`$S` =
`/shared/cycle2_tau_averbuch_prj/omrisegev1`), each dir holding a `raw_*.pkl` + `manifest.json`.

**Known caveats already decided, carry them into the fetch/score step:**
- `math500_qwenmath7b`: acc_band widened to `(0.10, 0.90)` in `cluster/presets.py`
  (uncommitted) — the T=1.5 legacy protocol genuinely produces ~17% positives (degenerate
  rambling, cap-hitter acc 0.025), not a pipeline bug. Score as-is, no re-run.
- `math500_dsmath7b`: landed 0.190 accuracy, 0.01 under the default floor. Decision already
  made: **score with a documented gate note** at rebuild (57 positives at N=300 is fine for
  AUROC) — do not re-run.
- `llama8b` RAG cells on **natural_questions** (5 positives) and **2wikimultihopqa** (11
  positives): label-starved but this matches the legacy Colab distribution almost exactly
  (verified this session, see §4 below) — **score with documented caveat, not REJECT.**
- `gpqa_llama8b` / `gpqa_r1distill8b` / `gpqa_mistral7b` / `gpqa_qwen72b` / `gpqa_llama70b`:
  these regen at the **true advertised T=1.0**. The old Colab-era gpqa cells labeled `T1.0`
  in the sweep pool were actually T=1.5 data (fingerprint-proven mislabel, see presets.py
  comment near `# gpqa regens at TRUE T=1.0`). The new cells **retire that mislabel** —
  don't try to reconcile old and new gpqa cell keys as the same condition; the 72B cell also
  changes identity (AWQ int4 → plain bf16 Qwen2.5-72B-Instruct).

### 3. Long-cap flavour cells — added this session, 1/3 done, 2/3 running
Motivated by R1-style models frequently hitting the 2048 cap mid-`<think>` (see §4 below
for the legacy-truncation evidence that justified this). Presets defined in
`cluster/presets.py` right before `get_preset()` (search `# Long-cap flavours`):

| Preset | Base preset it clones | Status at last check (2026-07-20) |
|---|---|---|
| `math500_r1distill8b_mn4096` | `math500_r1distill8b` (T=1.5, k=1, n=300) | **DONE** — 300/300, `raw_math500_T1.5.pkl` (685 MB) + manifest in `$S/results/regen/math500_r1distill8b_mn4096/`. Finished in the very first job leg (5:12h), no chain needed. |
| `gpqa_r1distill8b_mn4096` | `gpqa_r1distill8b` (T=1.0, k=10, n=198) | RUNNING, job 126275 chain, was at ~60-67/198 (~90s/candidate at 4096 tokens — the slow one). |
| `trace_gpqa_r1qwen7b_mn4096` | `trace_gpqa_r1qwen7b` (T=1.0, k=10, n=198) | RUNNING, job 126279 chain, similar pace to the above. |

Chain successors are queued (jobs 126271-3, 126276-7, 126280-1) so these should finish
unattended over the next day or two. **`math500_qwenmath7b` was deliberately excluded**
from the long-cap set — its cap-hitting is T=1.5 degenerate rambling, a longer cap would
feed it, not fix it.

All three new presets passed `scripts/smoke_preset.py` before submission (mandatory CPU
gate per CLAUDE.md) and code was synced via `bash cluster/sync_code.sh` before submitting.

### 4. Legacy-vs-regen comparison — done this session (informal, local scripts only, not committed anywhere)
Ran ad-hoc local scripts against `local_cache/` to sanity-check the regen's label
statistics against the original Colab runs, in response to Omri asking "is it similar?".
**Answer: yes, closely** — this is informal evidence, not a formal report; if useful,
turn it into a committed script/CSV during the fetch pass rather than re-deriving by hand.

- **RAG grid**: legacy sample-level grounding (via `is_grounded_lciteeval` applied to the
  legacy `{idx, output:{full_text,...}, row}` pkl schema under `local_cache/phase10_main/raw/`)
  matches regen full-N rates closely across all 16 model×dataset combos, several exact
  (e.g. llama8b-NQ 5/160 both legacy and regen). Same N caps (240/160) both sides. This
  resolves the earlier "pilot vs full-N accuracy drop" scare as a **distribution effect**
  (pilots run the first-30 rows only, which are easier) — not a regen protocol bug.
- **math500 T=1.5**: dsmath7b near-exact (0.197→0.190), qwenmath7b close (0.280→0.227),
  **r1distill8b has the one real gap (0.410→0.287)** — was hypothesized as 2048-cap
  truncation, but the legacy cap (1536) was actually *tighter* than the regen's 2048, so
  truncation was never fixed by the regen at 2048; **the just-landed `_mn4096` cell is the
  first real test of whether raising the cap recovers accuracy** — fetch and compare
  `math500_r1distill8b` (2048) vs `math500_r1distill8b_mn4096` (4096) vs legacy (1536) next
  session.
- **gpqa small models**: regen (true T=1.0) systematically higher than legacy (actually
  T=1.5), as expected: llama8b 0.268→0.323, mistral7b 0.253→0.273, r1distill8b 0.242→0.350.
  72B AWQ legacy 0.404 → bf16 pilot 0.457 (close); llama70b 0.480 is new.
- **internalstates gsm8k**: 0.306→0.294, match.
- **trace cells**: legacy gsm8k trace exactly 100/200 (0.500) — looks like a balanced-subset
  artifact in the old pkl, not a real 50% base rate; regen pilot 0.603 is probably closer to
  the model's natural rate. legacy gpqa_r1_7b trace cell was capped at 1024 with 98.7% of
  traces at cap (basically no real signal in the legacy version) vs regen pilot 0.287→0.317.

## What's left to do (next session, in order)

1. **Check status** — one-shot `cluster-ops` agent check (no polling loops, standing
   instruction) on the 2 still-running long-cap jobs: `gpqa_r1distill8b_mn4096` (126275
   chain) and `trace_gpqa_r1qwen7b_mn4096` (126279 chain). If both DONE, proceed with fetch.
   If not, fetch everything else now and circle back for these two.

2. **Fetch all 35 preset dirs** from `$S/results/regen/<preset>/` to local. There is no
   existing "fetch a `results/regen/*` preset dir" script — the project's `fetch_backfill.py`
   is scoped to backfill (append-only key patching), not this kind of fresh-cell fetch. Check
   whether an aircc-fetch-style flow already handles `results/regen/` (search for "regen" in
   `scripts/` and `.claude/skills/aircc-fetch/` if it exists) before writing a new one — if
   nothing fits, a plain `scp -r` per preset dir into a staging area, then a manual
   integration step, is fine; this is a one-time bulk fetch, not a repeating pattern that
   needs a reusable script.

3. **Map each regen cell to a legacy cell key** and integrate into the analysis pool. Key
   decisions already made (carry forward, don't re-litigate):
   - 72B AWQ → bf16: **new cell key**, not a replacement of the AWQ one (AWQ output is
     retained as historical, per "backfill preferred / regen only where authorized" and
     since the AWQ cell already has its own legacy comparisons).
   - gpqa small-model cells: **new true-T1.0 cell keys**, retiring the old T1.5-mislabeled-
     as-T1.0 cells from the *sweep pool* (the old cells' raw data isn't deleted, just no
     longer treated as "T=1.0" in any comparison).
   - `math500_dsmath7b` (0.190 acc) and `llama8b` RAG NQ/2wiki (5/11 positives): score
     normally, attach the documented caveat noted in §2 above — do not REJECT.
   - `math500_qwenmath7b`: score under the widened band, no other special handling.
   - Long-cap `_mn4096` cells: **additive new cell keys** (e.g.
     `math500_r1distill8b_mn4096`), never overwrite the 2048 cell's key.

4. **gpqa T-mislabel propagation**: any *repgrid* cells (not just the Colab gpqa cells)
   that were KEPT as T1.5-labeled-T1.0 need their label corrected at unified-rebuild time —
   this was flagged as a to-do in an earlier session and is still open. Check
   `Research_Directions.md` / prior HISTORY steps for which specific cells this touches
   before assuming it's only the 5 Colab gpqa cells handled in this wave.

5. **Once integrated**: this feeds directly into the still-open **Phase 5** of
   `HANDOFF_full_coverage.md` (unified featcache, `cache/unified/` schema, legacy
   `cell_key` mapping so `subset_sweep.py`'s `PKL_NAMES`/`iter_cells` need only a
   prefer-unified-path change, selector re-bench on the now-uniform pool, full report
   regen chain). Per PROGRESS.md Step 189, the selector-bench punch list (split-half
   oracle finding — the real headroom is ~1-2pp not ~7-8pp) should land **before** Phase 5
   per standing priority order; check the latest PROGRESS.md entry for whether that's still
   true when you pick this up.

6. **HISTORY.md**: this session's work (band-widen note already in presets.py comments,
   long-cap preset additions, the legacy-vs-regen comparison) has not been written up as a
   HISTORY step yet — do that once the fetch+score lands so the step tells the whole story
   (submit → land → compare → integrate) rather than being split across two step entries.

## Uncommitted changes to be aware of

`git status` shows a large pre-existing uncommitted diff spanning `results/`, `scripts/`,
`spectral_utils/` etc. — **this predates this session** (Step 189 selector-bench work,
per PROGRESS.md) and is not something to resolve here; per project convention, commits
await Omri. This session's own changes are additive and small, both in `cluster/presets.py`:
- the `math500_qwenmath7b` acc_band widen (`(0.10, 0.90)`) with its rationale comment
- the 3 `_mn4096` long-cap preset definitions with rationale comment

Both are already synced to the cluster (`bash cluster/sync_code.sh`) so the submitted jobs
are running the current code. Nothing needs to be committed before the fetch step — do it
whenever it's convenient, no urgency.

## Paste-ready prompt for the new session

> Read HANDOFF_regen_fetch.md and PROGRESS.md (latest Step entry). The 32-preset full-N
> 46-view-coverage regen wave is done on AIRCC (`$S/results/regen/<preset>/`), plus 1 of 3
> new long-cap `_mn4096` flavour presets is also done, 2 still running (check status first,
> one-shot cluster-ops, no polling). Fetch all landed preset dirs to local, map each to a
> legacy or new cell key per the decisions already recorded in the handoff (72B AWQ→bf16 is
> a new key, gpqa small models retire the T1.5-mislabeled-as-T1.0 legacy key, dsmath7b/
> qwenmath7b/llama8b-NQ/2wiki score with documented caveats not REJECT, `_mn4096` cells are
> additive new keys), then check the still-open gpqa T-mislabel propagation item before
> moving into Phase 5 (unified featcache) of HANDOFF_full_coverage.md.
