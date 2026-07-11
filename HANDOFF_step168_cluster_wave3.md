# Handoff — Step-168 → Wave-3 cluster-execution session

*Written 2026-07-11. Everything below is READY — presets edited, bugs fixed, smoke `--all` 23/23 PASS.
Your job: submit, babysit, fetch, score, integrate. Read `PROGRESS.md` + the CLAUDE.md cluster rules first.
Background: `HANDOFF_step166.md` §7–8 (wave-2 postmortem) and HISTORY Steps 162–168.*

## Why wave 3 exists (30 seconds of context)

Wave-2 Qwen3 jobs (101075/101076) hit the 8h wall, checkpointed, exited 0 → Slurm recorded
COMPLETED and never requeued → stuck at partial N (440/500, 280/500). Worse: 13% (GSM8K) / 45%
(MATH-500) of traces were pinned at `max_new=4096` — truncated generations grade "wrong", so
cap-hitting correlates with the label = **length-leakage; those runs are NOT citable**. Also the
ARS and Internal-States presets ran at T=1.0, the wrong operating point vs the papers.
**Omri's decisions (2026-07-11): fresh re-runs at the papers' decoding configs, original N=500,
`max_new=8192`; plus run everything from round 1 that never ran.**

Fixes already in the repo (do not redo):
- `cluster/run_inference.py` exits **85** on any incomplete checkpoint (was 0) — sacct now shows
  FAILED(85) = "resume me". Slurm still won't auto-requeue; use the chain-submit below.
- `cluster/submit_inference.sbatch.template` header documents the chain-submit pattern.
- Presets corrected from primary sources (verbatim-quote verified 2026-07-11):
  ARS §5.1 = **greedy decoding** → `ars_{gsm8k,math500}_qwen3_8b` + `ars_gsm8k_r1distill8b` now
  `temps=[0.0], max_new=8192`. Internal-States §3.1 = **T=0.8** (max 300 tok — we keep 2048,
  annotated) → `internalstates_gsm8k_qwen25_7b` now `temps=[0.8]`.

## Pre-flight (do all before any submit)

1. VPN up (a hanging ssh = TAU VPN down).
2. **HF_TOKEN check**: `ssh aircc "grep -c REPLACE_ME $SHARED/code/cluster/submit_inference.sbatch"`
   must be 0. `sync_code.sh` may have clobbered the live token with the template. Required for every
   gated cell (llama3b, nemo, mistral24b, NI mistral7b, NI gemma2b). If clobbered, re-inject the real
   token on the cluster copy (Omri has it) — NEVER commit it.
3. `bash cluster/sync_code.sh` — ships the exit-85 driver + corrected presets.
4. `$SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1` throughout.

## Chain-submit pattern (any run that may outlive one 8h wall)

```bash
jid=$(sbatch -p power-gpu --qos=owner_880 cluster/submit_inference.sbatch --preset <ID> --out $SHARED/results/repgrid/<ID> | awk '{print $4}')
jid=$(sbatch --dependency=afterany:$jid -p power-gpu --qos=owner_880 cluster/submit_inference.sbatch --preset <ID> --out $SHARED/results/repgrid/<ID> | awk '{print $4}')
# repeat one line per extra wall
```
A resume of a finished cell is an idempotent no-op (~10 min: loads checkpoint, refreshes manifest,
exits 0). Better one wasted no-op than another stalled cell.

## Wave A — re-runs (corrected operating points)

| # | Cell | Config | Walls | Pre-step on cluster |
|---|---|---|---|---|
| A1 | `truthfulqa_llama8b` **--regrade** | judge Qwen2.5-7B, no generation | <1 | none |
| A2 | `ars_gsm8k_qwen3_8b` | greedy, mn8192, N=500 | ~2 | `mv $SHARED/results/repgrid/ars_gsm8k_qwen3_8b{,_mn4096_partial}` |
| A3 | `ars_math500_qwen3_8b` | greedy, mn8192, N=500 | ~3 | `mv $SHARED/results/repgrid/ars_math500_qwen3_8b{,_mn4096_partial}` |
| A4 | `internalstates_gsm8k_qwen25_7b` | T=0.8, N=500 | ~1 | `mkdir -p $SHARED/results/repgrid/internalstates_gsm8k_qwen25_7b/archive_T1.0 && mv $SHARED/results/repgrid/internalstates_gsm8k_qwen25_7b/raw_gsm8k_T1.0.pkl $SHARED/results/repgrid/internalstates_gsm8k_qwen25_7b/archive_T1.0/` |

A1 command (sbatch wrapper, single wall):
`sbatch -p power-gpu --qos=owner_880 cluster/submit_inference.sbatch --preset truthfulqa_llama8b --regrade --judge Qwen/Qwen2.5-7B-Instruct --out $SHARED/results/repgrid/truthfulqa_llama8b`

The `mv` pre-steps matter because `score_repgrid.py` globs **all** `raw_*.pkl` in a cell dir — a
stale partial/T1.0 pkl would silently pollute the re-score. Local mirror hygiene: the local caches
`cache/repgrid/ars_*_qwen3_8b` are already renamed `*_mn4096_partial` (Step 168); do the same for
the local `internalstates` T1.0 pkl before fetching A4.

Watch on the A2/A3 pilots: greedy + Qwen3-thinking has a known repetition-loop risk — check the
pilot's trace-length distribution for cap-pinning (`inspect_cell.py`) before scaling. A2 stays a
likely **ceiling cell** (T=1.0 partial had acc 0.904; greedy may go higher) — a wide-CI outcome is
expected and reportable, not a failure.

## Wave B — round-1 presets that never ran (priority order)

| # | Preset | Model gated? | Notes |
|---|---|---|---|
| B1 | `ars_gsm8k_r1distill8b` | no | the missing ARS point; greedy/mn8192 now; ~2 walls |
| B2 | `lapeigvals_gsm8k_llama3b` | **yes** | triple-anchor cell (AttentionScore 71.7 / NI 82.70 / probe 87.0), N=1319 |
| B3 | `noise_gsm8k_phi3mini` | no | NI head-to-head, N=1319 |
| B4 | `noise_gsm8k_mistral7b` | **yes** | NI head-to-head, N=1319 |
| B5 | `noise_gsm8k_gemma2b` | **yes** | acc-floor risk — pilot may land below [0.20, 0.85]; a REJECT is itself the reportable outcome (document, don't scale) |
| B6 | `lapeigvals_gsm8k_nemo` | **yes** | LapEigvals sweep remainder |
| B7 | `lapeigvals_gsm8k_mistral24b` | **yes** | LapEigvals sweep remainder |

Concurrent submission is safe (Step-162 fixes: node-local PYTHONUSERBASE, anonymous containers);
owner_880 historically has zero queue wait — submit Wave A together, then Wave B in pairs.

## Per-cell loop (unchanged from Steps 163–167)

smoke (already green) → **N=30 pilot** (judge ONLY by acc in [0.20, 0.85] + trace not pinned at
max_new; `gate_ok=false` at N=30 is expected — `min_minority=30` is unsatisfiable) → full N with
chain-submit → fetch to `cache/repgrid/<id>/` (background; mn8192 pkls can exceed 1 GB) →
`python scripts/inspect_cell.py <dir>` → `python scripts/score_repgrid.py --cells <id>`
(**background** for >100 MB; merge-safe on write since Step 167) →
`python scripts/score_ubaselines.py` for the new cells (survey baselines on our traces) →
append `results/reasoning_benchmark.csv` → `python scripts/advisor_report.py`.

Rules that keep the session cheap (Step-163/164 retro, CLAUDE.md):
- ALL polling via `/aircc-status` or the `cluster-ops` subagent — never raw ssh loops inline.
- Score/fetch big cells in background with generous timeouts.
- `reasoning_benchmark.csv` model strings must match `advisor_report.py`'s order list exactly:
  `Qwen3-8B`, `DeepSeek-R1-Distill-Llama-8B`, `Qwen2.5-7B`, `Phi-3.5-mini-instruct`,
  `Llama-3.2-3B-Instruct`, `Phi-3-mini-4k-instruct`, `Mistral-7B-Instruct-v0.3`, `Gemma-2B-it`.
- Terminology bans in anything advisor-facing: no "Nadler" alone (method = L-SML), no "MV_EPR",
  no "recommended".

## Do NOT

- Do NOT import the wave-2 Qwen3 scratch scores (HANDOFF_step166 §7: GSM8K 0.938/0.962,
  MATH-500 0.795/0.834) into `scores_lsml_upcr.csv` or the benchmark CSV — provisional,
  truncation-confounded, superseded by A2/A3.
- Do NOT resume the archived `*_mn4096_partial` checkpoints — mixing mn4096 and mn8192 traces
  re-creates the truncation confound the re-run exists to remove.
- Do NOT lower `max_new` to make cells fit a wall — chain walls instead (truncation is the enemy).
- Do NOT judge-regrade the GSM8K T=1.0 cells (HANDOFF_step166 §4a: errors are genuine).

## End of session

`/update-docs` (HISTORY step + PROGRESS + commit). Push stays with Omri (credential comes and
goes; one fail-fast attempt max).
