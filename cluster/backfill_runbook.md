# View-backfill runbook — repgrid Phase 1–2 (full-coverage plan)

Goal: append the missing probability-derived keys (`token_logsumexp`,
`top_k_logprobs_raw`, and any missing `token_spilled_energies`/`top_k_logprobs`) to
the 12 repgrid cells that lack them, via teacher-forced forward passes. Labels,
traces, published numbers untouched (append-only, gated). Design + decisions:
`HANDOFF_full_coverage.md` + the approved plan (2026-07-18).

**State (verified 2026-07-18)**
- 7 cells COMPLETE (have Z_n) → they are the Gate-A validation set:
  `spilled_triviaqa_llama8b, se_nq_open_llama8b, se_squad_v2_llama8b,
  truthfulqa_llama8b, epr_triviaqa_mistral24b, seiclr_triviaqa_opt30b,
  semenergy_triviaqa_qwen3_8b`
- 12 cells TIER-2 (have `gen_token_ids`, lack Z_n) → backfill targets:
  `ars_gsm8k_r1distill8b, inside_coqa_llama7b, internalstates_gsm8k_qwen25_7b,
  lapeigvals_gsm8k_llama3b, lapeigvals_gsm8k_llama8b, lapeigvals_gsm8k_mistral24b,
  lapeigvals_gsm8k_nemo, lapeigvals_gsm8k_phi35, losnet_hotpotqa_mistral7b,
  noise_gsm8k_mistral7b, noise_gsm8k_phi3mini, sciq_llama8b`
  (~3.9 M generation tokens total → a few GPU-hours incl. 11 model loads)
- Cluster data: 19/19 cells present at `$SHARED/results/repgrid/<cell_id>/`
  (raw pkl + manifest each). Local coverage table: `results/repgrid_local_coverage.csv`.
- `python scripts/smoke_backfill.py` → **17/17 PASS** (required before submission).

## Step 1 — sync code

```bash
bash cluster/sync_code.sh
```

## Step 2 — Gate A: validate-only on the 7 complete cells (writes NOTHING)

Measures the bf16 decode-vs-teacher-forced noise floor by recomputing Z_n (and the
post-warp entropies) and comparing to the stored values. Covers 5 model families
including the OPT-30B raw-prompt path.

```bash
sbatch -p power-gpu --qos=owner_880 cluster/submit_backfill.sbatch \
    --cells spilled_triviaqa_llama8b,se_nq_open_llama8b,se_squad_v2_llama8b,truthfulqa_llama8b,epr_triviaqa_mistral24b,seiclr_triviaqa_opt30b,semenergy_triviaqa_qwen3_8b \
    --validate-only
```

Then fetch the reports (small JSONs, no pkls move) and review:

```bash
for c in spilled_triviaqa_llama8b se_nq_open_llama8b se_squad_v2_llama8b truthfulqa_llama8b epr_triviaqa_mistral24b seiclr_triviaqa_opt30b semenergy_triviaqa_qwen3_8b; do
  scp aircc:/shared/cycle2_tau_averbuch_prj/omrisegev1/results/repgrid/$c/backfill_report.json results/backfill_reports/$c.json
done
```

**Decision gate — MEASURED (job 123504, 2026-07-18)**: Gate-A Z_n PASSED on all 7
cells (median|ΔZ| 1.5e-3..1e-2, per-trace r ≥ 0.9997; teacher-forcing is faithful).
Gate-B's original thresholds were mis-calibrated for bf16: median|ΔH| came in at
1e-5..3e-3 (prompt reconstruction CORRECT everywhere) but ~1% of tokens jump ~0.1
nat — kernel noise (incremental decode vs full-sequence forward) flipping tokens at
the top-k/top-15 boundaries, which also fires the first-token check spuriously on
broad first-token distributions. Gate B was therefore recalibrated to median-based
statistics (whole-trace median ≤ 2e-2, first-token MEDIAN ≤ 5e-2, ≥90% of tokens
within 0.05; p99 and per-trace r demoted to informational). A true prompt mismatch
shifts every token by 0.1–1+ nat and still fails all three.
Known oddity, informational: `semenergy_triviaqa_qwen3_8b` shows a systematic
Z_n offset (median|ΔZ| 0.123 vs ~5e-3 elsewhere) with H essentially exact — a
uniform logit shift (bf16 lm-head accumulation on Qwen3's large-logit head);
probability-derived quantities are unaffected by a uniform shift.
Recalibration verified by rerun job 123561 before Step 3.

## Step 3 — backfill the 12 cells (chained pair, resume-safe)

One job processes all 12 (models load sequentially, grouped). Chain a second job
with `--dependency=afterany` — if the first hits the 8 h wall it checkpoints and
exits 85; the resume is idempotent (key presence = resume marker).

```bash
JOB=$(sbatch --parsable -p power-gpu --qos=owner_880 cluster/submit_backfill.sbatch \
    --cells ars_gsm8k_r1distill8b,inside_coqa_llama7b,internalstates_gsm8k_qwen25_7b,lapeigvals_gsm8k_llama3b,lapeigvals_gsm8k_llama8b,lapeigvals_gsm8k_mistral24b,lapeigvals_gsm8k_nemo,lapeigvals_gsm8k_phi35,losnet_hotpotqa_mistral7b,noise_gsm8k_mistral7b,noise_gsm8k_phi3mini,sciq_llama8b)
sbatch -p power-gpu --qos=owner_880 --dependency=afterany:$JOB cluster/submit_backfill.sbatch \
    --cells ars_gsm8k_r1distill8b,inside_coqa_llama7b,internalstates_gsm8k_qwen25_7b,lapeigvals_gsm8k_llama3b,lapeigvals_gsm8k_llama8b,lapeigvals_gsm8k_mistral24b,lapeigvals_gsm8k_nemo,lapeigvals_gsm8k_phi35,losnet_hotpotqa_mistral7b,noise_gsm8k_mistral7b,noise_gsm8k_phi3mini,sciq_llama8b
```

Per cell the driver: gates on the first 50 problems (Gate B blocking — a failed cell
is SKIPPED, nothing written, reasons in its report) → appends missing keys with
atomic checkpoints → writes `backfill_report.json` + a `backfill` provenance block
into `manifest.json`.

Monitor via `/aircc-status` (never raw ssh loops).

## Step 4 — land results locally (validating fetch)

```bash
python scripts/fetch_backfill.py --cells <all 12, comma-separated>
```

Per cell: scp → `cache/_incoming/` → validates (cluster Gate-B verdicts, all keys
present on every candidate, frozen keys `label`/`full_text`/`token_entropies`/
`gen_token_ids` byte-identical to the local pre-backfill copy) → backs up old pkls
to `cache/_backup/<date>/` → swaps in. A failed cell is left in `_incoming/` and
NOT swapped.

## Step 5 — feature-level continuity (Gate C)

```bash
python scripts/inspect_cell.py cache/repgrid/sciq_llama8b        # spot: energy=yes
python scripts/build_repgrid_featcache.py                        # Δ≤0.005 gate vs canonical CSV
```

The featcache gate re-proves GOOD_5/GOOD_6 numbers are untouched. After it passes,
the 19-cell grid is uniformly 46-view → selector bench / subset-sweep augmentation
can re-run on it (Phase 5 of the plan — AFTER the Step-186/188 punch-list items land,
per the agreed sequencing).

---

# Phase 3 — Colab-era cells: classification from the Drive audits (2026-07-18)

Sources: `results/coverage_audit.csv` (full Drive walk) + `schema_dump.json` (structure
dump of the ~40 relevant pkls) + notebook archaeology (`notebooks/Spectral_Analysis_
Phase4/5/10_Main_RAG.ipynb`). Verdict per sweep-pool cell family:

| cells | raw cache (Drive) | schema/keys | tier | prompt recipe | warp |
|---|---|---|---|---|---|
| math500 ×4 (`*_T1.5`) | `epr_spectral_phase4/<model>__math500/inference_cache.pkl` | `{idx:{full_text, all_entropies, correct, gold}}` | **2r** | phase4 `math_prompt` template (verbatim in notebook); idx = `load_math500(300)` order (lighteval/MATH_500 test); `gold` validates alignment | T=1.5, top_k=50, fp16 origin |
| gpqa Mistral-7B, Qwen-7B (`*_T1.0`) | `epr_spectral_phase5/<model>__gpqa/` | same + `gold`=letter | **2r** | `gpqa_prompt_and_answer` clone — options shuffled by `np.random.default_rng(idx).permutation(4)`, fully deterministic; GPQA Diamond in HF order | T=1.0, top_k=50 |
| gpqa Llama-8B, R1-Distill (`*_T1.0`) | only `epr_spectral_phase4/<model>__gpqa/` exists (**T=1.5 pipeline**) | same | **2r** | same | **⚠ suspected T-mislabel** (math500-Step-184's sibling): phase5 never ran these two. Resolve by label-sequence fingerprint vs `gpqa_res.pkl` before backfill; if phase4 → relabel cells T1.5 |
| gpqa Qwen-72B-AWQ | `epr_spectral_gpqa_72b/Qwen2.5-72B-Instruct-AWQ__gpqa_T1.0/` | + `correct_letter`, `has_answer` | **2r** | same gpqa recipe | T=1.0 |
| gsm8k Llama-8B | `epr_spectral_gsm8k_vs_lapei/Llama-3.1-8B-Instruct__gsm8k_T1.0/` | + `question` saved | **2r** | notebook gsm8k template on saved `question` | T=1.0 |
| qa ×3 (phase9 trivia/webq cot+plain) | `spectral_phase9_cache/*_traces.pkl` | `[{text, ents, correct, item{question,...}}]` list schema | **2r** | phase9 CoT/plain templates on `item.question` (extract from Phase9 notebook) | T=1.0 (per legacy cell key) |
| rag ×16 | `phase10_main/raw/<model>__<ds>.pkl` (+ `data_<ds>_n500.pkl` row caches) | `[{idx, row{question,docs,answers,raw_row}, output{full_text, token_entropies, token_offsets}}]` | **2r — NOT tier-3**: full docs + question saved; prompt = `spectral_utils.data_loaders.lciteeval_prompt(row)` (still in the package, default variant) | T=1.0, max_new=1024, top_k=50 |
| trace ×3 (phase12 K=10) | local `local_cache/raw_traces/*.pkl` | `token_entropies` ONLY (no text, no ids) | **3** | irreconstructible — stays H16-only or re-generate (Omri decision) | — |
| (bonus) phase15 math500 qwen7b T0.3–2.0 ×9 | `phase15_temperature/math500_qwen7b_T*.pkl` | CANONICAL keys incl. `gen_token_ids` | **2 (clean)** | dataset_fn | per filename |

Key facts: **no Colab-era cell except phase15 saved `gen_token_ids`** → tier-2r =
gen ids come from re-tokenizing `full_text` (decoded with skip_special_tokens +
strip, so expect len(retok) ≈ len(all_entropies) − 1: the trailing EOS entropy has no
text; Gate B compares over the aligned prefix and records the length delta).
Old caches were generated in **fp16** (notebook load_model), same chat-template +
`add_special_tokens` path as today, `do_sample=True, top_k=50`, no top_p, entropy
formula identical (top-15). Total backfill cost ≈ 4M forward tokens (few GPU-hours,
72B-AWQ legs included).

**Drive → local download list for Omri** (into `cache/colab_src/`, ~1.2 GB total):
`epr_spectral_phase4/` (~20 MB) · `epr_spectral_phase5/` (~7 MB) ·
`epr_spectral_gpqa_72b/` (~2 MB) · `epr_spectral_gsm8k_vs_lapei/` (~4 MB) ·
`spectral_phase9_cache/` (~1 MB) · `hallucination_detection/cache/phase10_main/`
(`raw/` + `data_*_n500.pkl`, ~580 MB) · `hallucination_detection/cache/
phase15_temperature/*.pkl` (~570 MB).

Implementation queue (code side, before the data lands): key-alias support in the
backfill loader (`all_entropies`/`ents`→token_entropies, `text`→full_text,
`correct`→label), a `stored_text_roundtrip` recipe (gen ids from re-tokenized
full_text + Gate-B length-delta policy), a `gpqa_phase4` deterministic-shuffle
prompt callable, list-schema iteration, and the phase4↔phase5 label-fingerprint
check for the two suspect gpqa cells.
