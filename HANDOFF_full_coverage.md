# HANDOFF — Full 46-view coverage: one unified dataset, every cell, every feature

**Written**: 2026-07-18 (Step 188). **Goal owner**: next session.
**Omri's directive**: no Colab-era / cluster-era split. One big dataset including the RAG and
GPQA runs. Every method evaluable on ALL cells, not subsets of them.

---

## What we have today, per task (verified 2026-07-18)

Feature groups: **H16** = 16 H(n) spectral/time features · **ΔE** = 4 spilled-energy views ·
**lp** = 3 top-K logprob views · **xlp** = 3 extended logprob views · **Z_n** = 4 raw-energy
views (needs `token_logsumexp`).

| domain | cells | H16 | ΔE | lp | xlp | Z_n | raw pkls local? |
|---|---|---|---|---|---|---|---|
| gpqa | 5 (incl. Qwen-72B-AWQ) | 5 | 0 | 0 | 0 | 0 | no — Drive |
| gsm8k | 1 | 1 | 0 | 0 | 0 | 0 | no — Drive |
| math500 | 4 | 4 | 0 | 0 | 0 | 0 | no — Drive |
| qa (trivia/webq) | 3 | 3 | 0 | 0 | 0 | 0 | no — Drive |
| rag | 16 (4 models × 4 datasets) | 16 | 0 | 0 | 0 | 0 | no — Drive |
| trace | 3 | 3 | 0 | 0 | 0 | 0 | partial (npz only) |
| repgrid | 19 | 19 | 19 | 19 | 19 | **7** | yes — `cache/repgrid/` (7.5 GB) |

So: 32 Colab-era cells are H16-only; 12 repgrid cells lack only Z_n; 7 repgrid cells are complete.

**Caveat**: the Colab-era row reflects the *sweep-pool* (npz manifests), i.e. what was extracted —
not necessarily what the raw Drive pkls contain. Some later Colab runs may have saved
`token_spilled_energies` or more. **First step of the plan must be a Drive raw-pkl key audit**
(one Colab/`ssh`-free session: mount Drive, loop over cache dirs, print key presence per cell —
mirror the `scripts/inspect_cell.py` report).

## The three recovery tiers (cheapest first)

1. **Offline, free**: anything derivable from saved `top_k_logprobs` (lp, xlp at K≤50) and saved
   `token_spilled_energies`. If the Drive audit finds these keys in any Colab-era pkl, extraction
   is a local/Colab CPU pass — no GPU.
2. **Teacher-forced backfill (GPU forward pass, no re-generation)**: a cell with
   `gen_token_ids` (or reconstructible token ids from `full_text` via re-tokenization — verify
   roundtrip per tokenizer) can have **everything** recovered — full-vocab logits at every
   position give Z_n, top-50 logprobs, ΔE, H(n) at any K. Labels, traces, and all published
   numbers stay untouched; we only append keys to existing pkls. This is the Step-188 discovery:
   **all 12 Z_n-missing repgrid cells qualify** (verified `gen_token_ids` present).
3. **Full re-generation (last resort)**: cells whose raw pkls lack both token ids and a clean
   `full_text` roundtrip, or whose prompts can't be reconstructed (RAG cells: were the retrieved
   passages saved in the pkl? — audit item). Re-generation creates NEW samples/labels → breaks
   comparability with legacy headline numbers (MATH-500 90.0 etc.). Avoid where tier 2 works.

## Models needed (tier-2/3 GPU work, all fit B200)

Llama-3.1-8B, Llama-3.2-3B, llama-7b (huggyllama), Mistral-7B v0.2/v0.3, Mistral-Nemo-12B,
Mistral-Small-24B (2501 + 3.1), Phi-3-mini, Phi-3.5-mini, Qwen2.5-7B, Qwen2.5-Math-1.5B/7B,
deepseek-math-7b, R1-Distill-Llama-8B, Qwen3-8B, OPT-30B, **Qwen2.5-72B-AWQ** (gpqa + rag —
needs the AWQ/gptqmodel load path from CLAUDE.md).

## Plan skeleton for the next session

1. **Drive audit** (blocking): per Colab-era cell, record keys present (`token_entropies`,
   `token_spilled_energies`, `gen_token_ids`, `top_k_logprobs`, `token_logsumexp`, `full_text`,
   prompt/context for RAG). Output: a coverage CSV committed to `results/`.
2. Classify every cell into tier 1/2/3; estimate GPU hours (tier 2 ≈ one forward pass per
   candidate; sum tokens per cell from trace lengths).
3. Write `cluster/backfill_logsumexp.py` (teacher-forcing mode; checkpoint/resume + SIGTERM per
   cluster rules; append-only writes via `save_cache_atomic`) — start with the 12 local repgrid
   cells as the pilot (data already in `cache/repgrid/`, no Drive dependency).
4. Colab-era pkls: sync Drive → cluster (or run backfill on Colab where the model is small),
   same script.
5. Rebuild featcache + npz manifests over the unified pool; re-run selector bench + subset sweep
   augmentation on ALL cells; regenerate reports (`selector_deep_report`, `chosen_sets`,
   action-items chain per the report-regen-chain memo).
6. Decide the unified-dataset format: one manifest schema (the repgrid rich-save schema is the
   target), one loader (`load_repgrid_cell`-style) for every cell regardless of origin.

**Open decisions for Omri**: (a) tier-3 cells — rerun or drop from the unified pool? (b) is
bit-exactness of teacher-forced logits vs generation-time capture a concern (bf16 kernel
variation — Z_n is smooth, expected negligible; can validate on the 7 cells that HAVE captured
Z_n by recomputing and comparing)? (c) priority order: repgrid-12 first (cheap, unblocks
uniform 46-pool bench) vs RAG/GPQA first (unblocks the "one big dataset" goal).

## Paste-ready prompt for the new session

> Read HANDOFF_full_coverage.md and PROGRESS.md. We are planning full 46-view coverage: one
> unified dataset over ALL cells (Colab-era + cluster, including RAG and GPQA). Start with the
> Drive raw-pkl key audit design, then classify cells into the three recovery tiers, then draft
> cluster/backfill_logsumexp.py starting with the 12 repgrid cells in cache/repgrid/ as pilot.
> Validate teacher-forcing on the 7 cells that already have Z_n before scaling.
