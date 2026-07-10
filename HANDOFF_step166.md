# Handoff — Step-166 agent → survey-benchmark agent

*Written 2026-07-10 by the agent that staged the Step-166 reasoning replication-grid and launched
the first cluster wave. Your plan (`use-this-source-generated-elegant-crescent.md`) supersedes and
absorbs this work. This doc = what I did, what's live, and the things that will bite you if you don't
know them.*

---

## 1. TL;DR — what's live right now

- **4 full-N cluster jobs I submitted** (do NOT resubmit — your plan already lists them):
  - `101074 lapeigvals_gsm8k_phi35` — **DONE, fetched, scored → WIN** (details §3)
  - `101077 internalstates_gsm8k_qwen25_7b` — DONE, fetched, scored (0.639, flagged — **read §4, your Phase-3 assumption is wrong**)
  - `101075 ars_gsm8k_qwen3_8b` — **RUNNING** (~6h40m in; ceiling cell, see §4)
  - `101076 ars_math500_qwen3_8b` — **RUNNING** (~6h40m; the long one, requeues past the 8h wall)
- Fetched pkls live in `cache/repgrid/{lapeigvals_gsm8k_phi35,internalstates_gsm8k_qwen25_7b}/` (gitignored).
- Scores in `results/repgrid/scores_lsml_upcr.csv` (the two done cells).
- I **removed my duplicate `Phi-3.5-mini` rows** from `reasoning_benchmark.csv` — your `Phi-3.5-mini-instruct`
  rows (the ones matching `advisor_report.py`'s model-order list) are canonical. I left line 19 (the qwen2.5
  "our result" row) since you integrated it with the `category` column.

## 2. Coordination / who-owns-what (to avoid clobbering)

I am **standing down** from the shared files your plan owns: `cluster/presets.py`,
`results/reasoning_benchmark.csv`, `scripts/advisor_report.py`, the judge-regrade path. I will not edit them
further unless asked. **Do not revert** the foundations your plan depends on:
- `cluster/presets.py` — the 7 Step-166 reasoning presets (LapEigvals sweep + ARS Qwen3 + Internal-States)
  are mine; your plan reuses them.
- `scripts/smoke_preset.py` — I added `gsm8k_family` + `math_family` grader fixtures (incl. the
  `<think>`→`\boxed{}` case). `--all` was 20/20 green before your 3 NI presets.
- `scripts/score_edis.py`, `cluster/reasoning_grid_runbook.md` — mine, new.
- `HISTORY.md` Steps 165–166, `PROGRESS.md` — mine.

## 3. Results produced (citable / not)

**Phi-3.5-mini / GSM8K (N=1319, job 101074) — CLEAN WIN, citable:**
- our L-SML **GOOD_5 0.803** (GOOD_5+logprob 0.808) vs LapEigvals **unsup AttentionScore 0.666** → **+13.7pp**.
- Below the supervised probe 0.885 (ceiling), as expected. `valid_rate 1.00`, acc 0.848 (plausible for Phi-3.5,
  so grading is fine here — this is the control that proves the qwen2.5 problem below is model-specific).

**Qwen2.5-7B / GSM8K (N=500, job 101077) — flagged NOT citable:** our L-SML 0.639 (GOOD_5+logprob 0.660) vs
Internal-States sup 0.7915. **But the comparison is confounded — see §4.**

## 4. ⚠️ CRITICAL — corrections to your plan's assumptions

**(a) The qwen2.5 acc-0.284 is NOT a "broken lexical grader." Your Phase-3 / verification line 109 will fail.**
Your plan says the judge re-grade should lift accuracy from "broken lexical 0.284" to "~0.8". I diagnosed the
actual pkl (`cache/repgrid/internalstates_gsm8k_qwen25_7b/`): **99% of the wrong answers DO have a `\boxed{}`**,
and the ones I inspected are **genuinely wrong** (T=1.0 arithmetic errors, self-contradiction). So an LLM judge
will label them wrong too — **accuracy stays ~0.28, not 0.8.** The root cause is **T=1.0 decoding**, not
grading. A judge re-grade will *not* unblock this cell.
- Implication: to make the Internal-States (and ARS-Qwen3) comparison apples-to-apples, you need a
  **temperature-matched re-run** at the paper's decoding T (near-greedy), not a re-grade. The presets I staged
  hard-code `temps=[1.0]` — that's wrong for these supervised-paper comparisons (I matched our own lapeigvals
  cell's T, before realizing the operating-point issue). **LapEigvals cells are unaffected** (that paper is
  temperature-robust; phi35 landed at acc 0.848 with a clean win).
- The judge re-grade is still worth doing for `truthfulqa_llama8b` (ROUGE-L proxy → real labels) — that one is
  a genuine label-quality fix. Just not for the GSM8K T=1.0 cells.

**(b) `ars_gsm8k_qwen3_8b` (101075) is a CEILING cell.** Pilot acc was 0.967 (Qwen3-8B thinking is near-perfect
on GSM8K). At full N=500 you'll get ~17 negatives → minority < 30 → the band gate will flag it and the AUROC CI
will be wide/unstable. Expect a REJECT-style outcome; the MATH-500/Qwen3 cell (101076) is the usable ARS/Qwen3
comparison.

**(c) Gated-token sbatch may be clobbered.** My `bash cluster/sync_code.sh` tars the working tree over
`$SHARED/code`, which includes `cluster/submit_inference.sbatch`. If the local copy is the `REPLACE_ME` template
(not the live HF_TOKEN version), the cluster's token file was overwritten. **Verify `HF_TOKEN` is real in
`$SHARED/code/cluster/submit_inference.sbatch` before submitting any gated cell** (Llama-3.2-3B, Mistral-Nemo,
Mistral-Small-24B, and your NI Mistral-7B). The 4 non-gated jobs were unaffected.

## 5. Cluster mechanics gotchas (learned this session)

- **Pilot `gate_ok=false` at N=30 is EXPECTED and not a failure** — `min_minority=30` is unsatisfiable with 30
  samples. At pilot stage judge only by **accuracy in [0.20, 0.85]**; `gate_ok` becomes meaningful at full N.
- **VPN drops mid-session.** A hanging/timeout ssh = TAU VPN down (happened once here). Jobs keep running +
  checkpoint regardless; only your ability to query is affected.
- **Raw ssh reprints the login banner** into context every call — I filter with
  `| grep -vE "Welcome|restricted|monitored|Unauthorized|disciplinary|logging|acceptable|====="`. Prefer
  `/aircc-status` / the `cluster-ops` subagent for polling (CLAUDE.md rule; I bent it for one-shot checks).
- **Score cells >100 MB in the background** — the phi35 pkl is 177 MB (N=1319) and timed out a 2-min foreground
  `score_repgrid.py`; re-run with `run_in_background`.
- Fetch mirrors `cache/repgrid/<id>/{manifest.json, raw_*.pkl}` (the layout `score_repgrid.py` expects).

## 6. Suggested immediate next actions for you

1. When 101075/101076 finish: `scp` to `cache/repgrid/<id>/`, `scripts/inspect_cell.py`, then
   `scripts/score_repgrid.py --cells <id>` (background for the MATH-500 one — long traces).
2. Reconcile the qwen2.5/Internal-States story per §4(a) — decide on a temperature-matched re-run vs annotating
   it as a different-operating-point caveat. Drop the "judge regrade fixes acc" expectation.
3. Verify the gated-token sbatch (§4c) before your `lapeigvals_gsm8k_{llama3b,nemo,mistral24b}` and NI cells.
4. Your `advisor_report.py` model-order list uses `Phi-3.5-mini-instruct`, `Llama-3.2-3B-Instruct`,
   `Mistral-7B-Instruct-v0.3` etc. — make sure the preset `model=` strings and the CSV `model` column match
   those exact strings or the rows won't render.
