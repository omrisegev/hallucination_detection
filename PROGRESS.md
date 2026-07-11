# Spectral Hallucination Detection — Session Progress Handoff

**Date**: 2026-07-11
**Last updated**: Step 169 cont. — **8 of 11 Wave-3 cells now SCORED in the canonical CSVs; only A2 (ars_gsm8k_qwen3, ETA ~01:30), A3 (ars_math500_qwen3 mn16384, ETA 2026-07-12 afternoon/evening) and C1 (inside_coqa + chained judge-regrade job 106293) still running — chains healthy, nothing to babysit.** Scored full-N this session (all in `reasoning_benchmark.csv` + `scores_lsml_upcr.csv` + `ubaseline_scores.csv`; advisor report regenerated guardrail-clean; Nemo/Mistral-24B added to the report order list, NQ-Open added to the QA head-to-head table): **B1 r1distill GSM8K greedy (acc 0.728): unsup GOOD_5 75.0 [70.4,79.7] BEATS the supervised ARS anchor 74.72 same-model** and beats every ARS unsup baseline by 13pp+ (caveat: seqlp ubaseline 77.2); **A4 internalstates T=0.8 (acc 0.306): U-PCR GOOD_5 69.1 [64.0,73.8] edges SelfCheckGPT 67.98**, supervised IS-probe 79.15 stays above; **B6 nemo (acc 0.829): GOOD_5 78.2 beats same-model LapEigvals unsup AttentionScore 63.0 by +15.2pp WIN** (+logprob 80.0; seqlp 80.3 caveat); **B7 mistral24b (acc 0.917 CEILING, 109 negatives): GOOD_5 80.1 vs AttentionScore 57.6 = +22.5pp WIN with ceiling caveat** (U-PCR 83.5; seqlp 85.7); **B4 NI mistral7b (acc 0.499 balanced): GOOD_5 78.5 = EXACT TIE with NI K=10 78.5 at 1-pass** (~10x less compute; seqlp 79.6 caveat); **B3 NI phi3mini (acc 0.710): 66.4 honest loss vs NI 72.51 (−6.1pp) but beats NI's own no-noise answer-entropy baseline 65.86**; **B2 llama3b (acc 0.445, unsloth mirror): 68.5 narrow loss vs AttentionScore 71.7 (−3.2pp; GOOD_5+logprob 71.1 = −0.6pp)**; **C2 se_nq_open on judge labels (acc 0.501 perfectly balanced, N=1000 K=10, valid 0.85): GOOD_5 71.8 / U-PCR 73.2 / +logprob 74.2, LNPE 74.4 slightly above — no clean published same-model anchor, rendered as such in the QA table**. Wave-3 tally so far: **3 WINs (r1distill-vs-SUPERVISED, nemo, mistral24b), 1 exact tie (mistral7b), 1 edge (internalstates vs SelfCheckGPT), 3 honest losses (llama3b narrow, phi3mini, truthfulqa), 1 floor-REJECT (gemma2b), 1 ceiling-caveated score (sciq)**. Stale CSV notes refreshed (queued→scored, gemma2b REJECT, A2/A3 job ids). **Next session**: fetch+score A2 (ceiling check — full-N acc ≥0.98 ⇒ greedy-ceiling-unreportable), A3 (verify no 16384 cap-pinning in inspect_cell), C1 + its regrade; then write the Step-170 HISTORY entry that closes the benchmarking desk. Commits ahead of origin — Omri pushes. **[Step-169 submit-session status below.]** Step 169 was — **Wave 3 EXECUTED: 30 jobs (103531–103544, 106275–106308); 10 cells running full-N chain-protected; truthfulqa + sciq SCORED; gemma2b floor-REJECT documented; NEXT SESSION = per-cell fetch→inspect→score.** All 10 pilots gated; 3 pilot failures fixed in-session: `unsloth/Llama-3.2-3B-Instruct` + `unsloth/gemma-2b-it` **mirror swaps** for gated-403s (meta-llama-3.2 and google gates reject our token; same pattern as huggyllama — Omri can request HF access to retire the deviation, or keep mirrors: byte-identical weights, documented in presets) and `sentencepiece` added to `cluster/requirements.txt` (Mistral-v0.3 slow-tokenizer crash). **Desk-clean (Omri's directive — every cell ends scored-in-CSV or documented-REJECT)**: judge-regrades flipped both paused QA cells into band (inside_coqa lexical 0.183→judge 0.223; se_nq_open 0.067→**0.663** — the lexical EM grader was the blocker, not the model) → both running full-N with a chained post-inference judge-regrade; **sciq scored** L-SML 0.738 / U-PCR 0.744 (double caveat: ceiling acc 0.877 + only 20% of MCQ traces ≥8 tok); **gemma2b acc 0.000 (0/30) = documented floor-REJECT** (the NI-anticipated reportable outcome — not scaled). **truthfulqa re-scored on REAL judge labels** (acc 0.116; judge-vs-lexical agreement 0.762): L-SML GOOD_5 0.660 / U-PCR 0.673 vs TSV semi-sup 84.2 (honest lose; seq-logprob ubaseline 0.693 edges GOOD_5 — caveat in CSV). **A3 ars_math500_qwen3_8b**: mn8192 pilot had 6/30 traces capped with **3 of 4 negatives capped** (leakage persists at 8192) but **NO repetition loops** (tail repeat-frac ≤0.08 — genuinely long reasoning) → preset now **max_new=16384**, pilot archived `ars_math500_qwen3_8b_mn8192_pilot` cluster+local (**NEVER resume** — cap-mixing confound), fresh full-N on 4 chained walls (106305–08; the long pole, ~24h). **A2 ars_gsm8k_qwen3_8b pilot acc 1.000** (0 negatives; worse ceiling than the 0.904 forecast) — scaled per handoff; if full-N acc ≥0.98 the cell documents as greedy-ceiling-unreportable. Other pilot accs (all scaled): internalstates-T0.8 **0.333** (in band; the T=1.0 collapse only partially recovers at T=0.8 — expected low-acc operating point), r1distill 0.633, llama3b 0.367, phi3mini 0.633, mistral7b 0.333, nemo 0.800, mistral24b 0.900 (ceiling caveat per the sciq precedent). **Follow-up session, per cell when its chain finishes**: `/aircc-fetch` → `inspect_cell.py` (on A3 verify no 16384 cap-pinning) → `score_repgrid.py --cells <id>` (background >100 MB) → `score_ubaselines.py` → `reasoning_benchmark.csv` (exact model strings from advisor_report order list) → `advisor_report.py`. Chains auto-resume; sacct FAILED(85) = "checkpointed, resume pending" — normal mid-chain. Full detail: HISTORY Step 169. **[Step-168 status below — its execution tail is DONE (Step 169).]** Step 168 was — **Wave-2 postmortem + Wave-3 staged via `HANDOFF_step168_cluster_wave3.md`.** Wave-2 Qwen3 jobs 101075/101076 hit the 8h wall, checkpointed, **exited 0 → Slurm recorded COMPLETED, no requeue → stalled partial** (440/500 GSM8K, 279/500 MATH-500). Deeper confound (HANDOFF_step166 §7–8, the Step-166 agent scored the partials to scratch): **13% / 45% of traces pinned at max_new=4096 → truncation-label leakage**; wave-2 Qwen3 numbers are PROVISIONAL and kept OUT of canonical CSVs (scratch: GSM8K GOOD_5 0.938 / U-PCR 0.962 vs ARS sup 0.904; MATH-500 0.795 / 0.834 vs 0.787; `GOOD_5+logprob` HURTS MATH-500/Qwen3). Decoding configs now **verified from primary sources** (Haiku subagent, verbatim quotes): **ARS §5.1 = greedy** ("By default, greedy decoding is used to generate model answers"); **Internal-States §3.1 = T=0.8 / max 300 tok**. **Fixes landed (committed)**: `run_inference.py` exits **85** on incomplete checkpoint (exit-0 was the no-requeue root cause); chain-submit resume pattern (`sbatch --dependency=afterany`) in the sbatch template header; presets `ars_{gsm8k,math500}_qwen3_8b` + `ars_gsm8k_r1distill8b` → **greedy / max_new=8192** (do NOT lower max_new — chain walls instead), `internalstates_gsm8k_qwen25_7b` → **T=0.8**; local partial caches renamed `*_mn4096_partial` (score_repgrid globs ALL raw_*.pkl — archive stale pkls before any re-fetch). smoke `--all` **23/23 PASS**. **NEXT SESSION executes the handoff**: pre-flight (VPN, HF_TOKEN REPLACE_ME check, sync_code) → Wave A re-runs (truthfulqa `--regrade`; ars_gsm8k_qwen3_8b ~2 chained walls; ars_math500_qwen3_8b ~3; internalstates T=0.8 — each with its archive-mv pre-step) → Wave B never-ran presets (`ars_gsm8k_r1distill8b`, `lapeigvals_gsm8k_llama3b`, `noise_gsm8k_{phi3mini,mistral7b,gemma2b}`, `lapeigvals_gsm8k_{nemo,mistral24b}`; gemma2b pilot may REJECT = reportable) → per-cell fetch→`inspect_cell`→`score_repgrid` (background >100 MB)→`score_ubaselines`→`reasoning_benchmark.csv`→`advisor_report.py`. Full detail: HISTORY Step 168. **[Step-167 status below; its cluster tail is SUPERSEDED by the handoff.]** Step 167 — **Survey-driven benchmarking pass: every anchor VERIFIED from primary arXiv sources; survey baselines scored on our traces; Noise-Injection sweep staged; judge-regrade mode shipped.** Source = `papers/State of the Art in LLM Hallucination Detection for Reasoning Tasks (as of July 2026)...md`. **All survey numbers verified** (Haiku web-subagent, verbatim-quote protocol): Noise Injection 2502.03799 **v4** Table 3 (Llama-3.2-3B 76.53→82.70 — v4 only, stale fetches miss it; Phi-3-mini 65.86→72.51; Mistral-7B-v0.3 75.85→78.50; Gemma-2B-it 51.36→57.11; N=1319 K=10 T=0.5 majority-vote question-level), ARS 2601.17467 **Table 2** unsup baselines (EigenScore Qwen3-8B: GSM8K 63.40 / MATH-500 81.38; R1-Distill GSM8K 52.98/61.98/58.48, MATH-500 75.89/43.60/40.96 → **our GOOD_5 84.4 beats every published unsup baseline on MATH-500/R1-Distill**), Internal-States 2510.11529 (SelfCheckGPT 67.98±1.28 = fair unsup Y on GSM8K/Qwen2.5-7B), TSV 84.2 semi-sup TruthfulQA/Llama-3.1-8B, HaloScope 78.64, Janiak 2508.08285 quotes. **New `scripts/score_ubaselines.py`** (perplexity/seq-logprob/naive-entropy per candidate + LN-PE/PE question-level for K≥2, one cheap pass, dual-label AUROCs) ran on 13 cells → `results/repgrid/ubaseline_scores.csv`: GSM8K/Llama-8B seq-logprob **80.4** vs GOOD_5 81.5 (close — honest caveat in CSV); **dual-label swing up to 35pp** (EPR cell naive-entropy 70.2 judge vs 35.6 lexical) = in-house Janiak confirmation. **Fresh phi35 cell scored** (job 101074, N=1319, acc 0.848): **GOOD_5 80.3 vs LapEigvals unsup AttentionScore 66.6 → +13.7pp WIN** (2nd sweep point). **Infra fix**: `score_repgrid.py` now merge-on-write (a concurrent session's phi35 run had silently overwritten the 11 Step-163 CSV cells; restored, GOOD_5 llama8b 0.8152 intact). **presets.py**: 7 presets anchor-enriched + 3 NI presets (`noise_gsm8k_{phi3mini,mistral7b,gemma2b}`, Gemma pilot-gated acc-floor risk); smoke 23/23. **`run_inference.py --regrade --judge <id>`** relabels an existing fetched run dir (no generation; `label_lexical` preserved; resumable) — for truthfulqa (ROUGE proxy → real labels). **CORRECTED per HANDOFF_step166.md: the internalstates acc-0.284 is NOT a grading artifact** — 99% of wrong answers have `\boxed{}` and are genuinely wrong (T=1.0 sampling collapse); a judge regrade will NOT unblock that cell — it needs a **temperature-matched re-run at the paper's near-greedy decoding T** (decision with Omri; the staged preset hard-codes T=1.0). `reasoning_benchmark.csv` 19→49 rows + `category` column (UGB/BB/WB/SUP); advisor report regenerated guardrail-clean (math-reasoning-gap box, category badges, judge-vs-lexical section). ProcessBench/MR-GSM8K **deferred** → Research_Directions.md Extension F. **Cluster tail (user-run, in order)**: (0) **verify HF_TOKEN is real in `$SHARED/code/cluster/submit_inference.sbatch`** before any gated cell — sync_code.sh tars the working tree and may have clobbered the live token with the REPLACE_ME template (HANDOFF_step166 §4c; gated = llama3b, nemo, mistral24b, NI mistral7b, NI gemma2b); (1) regrade job for **truthfulqa only** — `python cluster/run_inference.py --preset truthfulqa_llama8b --regrade --judge Qwen/Qwen2.5-7B-Instruct --out $SHARED/results/repgrid/truthfulqa_llama8b` → re-fetch → re-score (internalstates is NOT regrade-fixable — see correction above; T-matched re-run decision with Omri); (2) `ars_gsm8k_r1distill8b`; (3) `lapeigvals_gsm8k_llama3b` (triple-anchor: AttentionScore 71.7 / NI 82.70 / probe 87.0); (4) `noise_gsm8k_phi3mini` + `noise_gsm8k_mistral7b`; (5) `noise_gsm8k_gemma2b` (pilot may REJECT — that's the reportable outcome); (6) `lapeigvals_gsm8k_{nemo,mistral24b}`. **In flight, do NOT resubmit**: 101075 ars_gsm8k_qwen3_8b (~8h; **ceiling cell — pilot acc 0.967 → expect gate REJECT / wide CI at full N**, the MATH-500 cell is the usable ARS/Qwen3 point), 101076 ars_math500_qwen3_8b (~14h, requeues past 8h wall) — on completion `/aircc-fetch` → `inspect_cell.py` → `score_repgrid.py --cells <id>` (merge-safe now; background for the MATH-500 pkl) → CSV → report. Full detail: HISTORY Step 167. **[Step-166 status below.]** Step 166 — **Reasoning replication-grid presets staged: 7 new inference-only cells (our L-SML vs a competitor's PUBLISHED reasoning AUROC).** Reviewed `BENCHMARKING_COMPETITOR_GUIDE.md`, verified each method against its paper, and staged presets to fill the real reasoning gaps (same pattern as Steps 162–163: run inference on the paper's exact X/Y/N, score OUR L-SML offline, compare to their published Y — no competitor detector reproduced). **Paper verification corrects the guide**: EPR (2509.04492) is **QA-only** (not reasoning) → excluded; **LapEigvals (2502.17598)** evaluated **GSM8K only (N=1319)** on 5 models with published unsup AttentionScore + sup probe AUROC (we had only Llama-3.1-8B); INSIDE/LOS-Net are QA-domain; FG-PRM/FUSE report best-of-N accuracy not detection AUROC. **7 presets added** (`cluster/presets.py`, all `smoke_preset.py`-PASS, K=1, default capture): Tier 1 LapEigvals GSM8K sweep — `lapeigvals_gsm8k_{llama3b,phi35,nemo,mistral24b}` (fair Y = unsup AttentionScore 0.717/0.666/0.630/0.576; sup ceilings 0.870/0.885/0.890/0.925); Tier 2 — `ars_gsm8k_qwen3_8b` (vs 90.37), `ars_math500_qwen3_8b` (vs 78.66), `internalstates_gsm8k_qwen25_7b` (vs 79.15). Added GSM8K+MATH grader fixtures to `scripts/smoke_preset.py` so the CPU gate now validates the math graders incl. the `<think>`-then-`\boxed{}` case (R1/Qwen3) + `\frac` normalization; `--all` = 20/20 PASS. **Cluster async tail (user-run, VPN+queue)**: per cell `bash cluster/sync_code.sh` → `/aircc-submit <id>` N=30 pilot (acc in [0.20,0.85], trace not pinned; strong models may ceiling on GSM8K) → full N → `/aircc-fetch` → `scripts/score_repgrid.py --cells <id>` → append `results/reasoning_benchmark.csv` → `scripts/advisor_report.py`. Files: `cluster/presets.py`, `scripts/smoke_preset.py`. Not committed (await Omri). Full detail: HISTORY Step 166. **[Step-165 status below.]** Step 165 — **Reasoning-first advisor report rebuilt from CSV + missing reasoning comparisons filled.** Replaced the Gemini `results/Advisors_Action_Items_Report.html` with a generated, fact-checked one (`scripts/advisor_report.py`; every numeric cell sourced from a CSV; built-in terminology-guardrail scan passes). **Reasoning story now leads** (`results/reasoning_benchmark.csv`): MATH-500/Qwen-Math-7B **94.4** unsup 1-pass (= GOOD_5 subset-sweep exactly); **R1-Distill/MATH-500 GOOD_5 84.4 ≈ ARS supervised 86.38 on the SAME model** (already had the npz — no cluster run needed); GSM8K beats **LapEigvals-unsup 72.0** (A1: same-model anchor fixed in `cluster/presets.py`; 92.5 was the cross-model Mistral-24B sup number); **EDIS scored on our own trace** = 0.809 but redundant with L-SML (ρ=0.87) via new `scripts/score_edis.py` → `results/repgrid/edis_scores.csv` (all 11 cells). Verified-from-arXiv anchors: **ARS 2601.17467** (sup; GSM8K 90.37 / R1-Distill 74.72, MATH-500 78.66 / R1-Distill 86.38), **Internal-States 2510.11529** (sup; GSM8K 79.15). Report corrections: Semantic Energy = **Chen et al. 2508.14496** (not Farquhar), dropped "Minut et al." from EPR, EPR X labeled **U-PCR+logprob**, **fusion 0.768→0.758** (Step 152), NLI-truncation reframed suspected/unresolved (SE 87.7 / SC 87.2 flagged not-citable), selection-bias caveats (spilled n_pos=6, se_squad valid 0.29), closed-subset-per-domain table. Two ready-to-run ARS presets added (`ars_math500_r1distill8b`, `ars_gsm8k_r1distill8b`; smoke-passed) — GSM8K/R1-Distill cluster run is the async tail. Not committed (await Omri). **Next open items**: (1) reconcile the SE/SC NLI-truncation drop to make the old-cache reasoning baselines citable [Step-152 P1]; (2) run the GSM8K/R1-Distill ARS cell; (3) score MATH-500 EDIS on Colab (50 MB Drive pkl); (4) keep spilled/se_squad as selection-biased. Full detail: HISTORY Step 165. **[Step-164 status below.]** Step 164 — **Workflow token-economy tooling (from the Step-163 retro): two CPU-only scripts + three CLAUDE.md rules.** `scripts/smoke_preset.py <id>` = CPU-only pre-submit validator running the preset's REAL prompt/grader/judge helpers on fixtures (no model/dataset) — catches the pure-CPU pilot bugs (Qwen3 empty-`<think>`, OPT ramble, judge-parse ordering) offline in seconds; verified all 3 new presets PASS + a tampered fixture forces FAIL. `scripts/inspect_cell.py <pkl|dir>` = standard schema report (N/K, label dist + judge-vs-lexical agreement, trace lengths, base/energy/judge key presence, extractable features + valid-rate) — replaces ad-hoc `python -c`; verified on semenergy (K=10) + losnet (899 MB K=1). **3 CLAUDE.md rules**: (1) cluster polling only via `/aircc-status`/`cluster-ops`, never raw `ssh` in main context; (2) new preset MUST pass `smoke_preset.py` before submit (gate: local smoke → N=30 pilot → full N); (3) score/extract cells >100 MB or K≥10 in the background, inspect with `inspect_cell.py` first. **Deferred to a separate design pass (Omri's call): PDF text-cache + RAG** — extract `papers/*.pdf` → committable `papers/extracted/*.md` (PyMuPDF installed; `*.pdf` gitignored so `.md` persists) then latent-space search (fork: sklearn TF-IDF vs sentence-transformers; no embed libs installed yet). **Next**: design the PDF-cache + RAG pass; and the open analysis threads from Step 163 still stand (decide the 3 paused pilot cells; optional Gemma judge re-run if access lands). Full detail: HISTORY Step 164. **[Step-163 status below.]** Step 163 — **Replication grid SCORED (Phase 2, local CPU): OUR L-SML continuous + U-PCR vs the papers' PUBLISHED numbers, same-scenario head-to-head.** First re-ran 3 mismatched papers on their EXACT model (Phase 1, cluster inference only) so X sits next to Y with only the method differing. **Headline (our best X vs published Y, all `head_to_head=SAME-MODEL`)**: Semantic Energy/Qwen3-8B/TriviaQA **X=0.801 (GOOD_5 L-SML) vs Y=0.748 → +0.05, we beat it**; EPR/Mistral-24B/TriviaQA **X=0.736 (GOOD_5+logprob U-PCR) vs Y=0.746 → −0.01, tie**; SE-ICLR/OPT-30B/TriviaQA X=0.630 vs Y=0.83 → −0.20 lose (per-question K=10 semantic-sampling regime, different units — annotated); LOS-Net/Mistral-7B-v0.2/HotpotQA 0.583 vs 0.729 lose (supervised probe). So against the two strong single-answer baselines on TriviaQA our unsupervised spectral method **ties EPR and beats Semantic Energy**. **6 pilot bugs fixed** (multimodal Mistral3 load, Qwen3 empty-`<think>`, OPT-30B torch.load guard + base-model raw_prompt/few-shot, list-content chat template, judge swap Gemma/general-verifier→**Qwen2.5-7B-Instruct** — documented deviation). **4 whatis**: (A) fresh GSM8K/Llama-8B GOOD_5 L-SML 0.815 vs old 0.756 = +0.059; (B) MACRO 0.714 (n=9) vs old 0.636 (n=29); (C) energy/logprob views help QA (EPR +0.024, SQuAD +0.031) not reasoning, and HURT SemEnergy (−0.095); (D) more features ≠ better on short QA — AUROC peaks at 4-5 feats then declines everywhere. Deliverables: `results/Replication_Grid_Report.html` + `results/repgrid/*.csv` (committable); raw pkls gitignored under `cache/repgrid/`. Full detail: HISTORY Step 163.  **[Older status below.]**  Step 162 was: **Replication grid EXECUTED on AIRCC (inference-only). 5 VALID full-N cells produced + fetched + schema-validated locally; 3 cells paused out-of-band; two cluster infra bugs fixed.** Gate-first waves (N=30 pilots -> auto-scale in-band). VALID cells, each at cluster `$SHARED/results/repgrid/<preset>/` + local `cache/repgrid/<preset>/` (gitignored, ~1.5 GB): **losnet_hotpotqa**/Mistral-7B-v0.2 (acc 0.338, top-1000 logprobs; LOS-Net 72.92 anchor), **lapeigvals_gsm8k**/Llama-8B (acc 0.724, trace 168 - strongest cell), **spilled_triviaqa**/Llama-8B (acc 0.320, energy capture), **se_squad_v2**/Llama-8B K=10 (acc 0.606), **truthfulqa**/Llama-8B K=10 (acc 0.222, ROUGE-L proxy label - re-grade offline). PAUSED (N=30 pilots, out-of-band, re-gradeable offline since full_text saved): **inside_coqa**/llama-7b-base (acc 0.183 floor - base model rambles, ROUGE-L>0.3 grader misses; hidden_middle_last captured), **se_nq_open** (0.067 floor), **sciq** (0.900 ceiling). **Infra fixes** (first wave crashed): (1) concurrent `pip install` into shared `$HOME/.local` corrupted dill (`version.parse(None)`) -> node-local per-job `PYTHONUSERBASE` + `pip --user`; (2) pyxis `--container-name` collision on shared nodes -> dropped the static name. Both in `cluster/submit_inference.sbatch(.template)`, resynced, re-validated by 3 concurrent jobs. **Step-161 energy fix confirmed on cluster data**: raw-vs-warped logsumexp gap 22.8/29.1/24.2 nats; all 4 capture paths (base/raw-energy/wide-logprob/hidden) validated; 0 non-finite features. **Next**: offline scoring (local CPU) - L-SML continuous GOOD_5 + logprob/energy AUROCs on the 5 VALID cells vs published anchors; decide the 3 paused cells. Docs + sbatch fix not committed (await Omri). Full detail: HISTORY Step 162.

**Prior**: Step 161 — Reviewed the Step-160 replication-grid impl; Step-159 consolidation verified complete; two protocol-fidelity fixes (now shipped + validated by the Step-162 run): (1) energy capture uses RAW full-vocab logits (`generate_full(capture_logsumexp=True)` sets `output_logits=True`; `token_logsumexp`=true Z_n + `top_k_logprobs_raw`, not the temperature/top-k-masked `out.scores`); (2) CoQA conditions on dialogue history. Notes: INSIDE grades ROUGE-L>0.3 not >0.5 (re-gradeable, full_text saved); LapEigvals `published`=92.5 is the supervised Mistral-24B probe, not the cell's unsupervised Llama-8B (~72 anchor). Files: `spectral_utils/model_utils.py`, `spectral_utils/data_loaders.py`.

**Prior**: Step 160 — **Replication-grid plan reviewed after the EDIS/AIME24 test run; implementation landed (unit-tested locally, not yet run on cluster).** Decision (Omri): **keep each paper's exact terse protocol — no CoT/long-trace change** (apples-to-apples with published tables; CoT wouldn't rescue single-fact QA anyway). The plan's structure stands; the AIME24 demo (Qwen-1.5B, MAX_NEW=1024, **acc 1.7–2.9%** → AUROC uncomputable) only forced guardrails. Landed in `spectral_utils` + `cluster`: (1) `generate_full` default-OFF capture flags — `token_logsumexp` (energy papers), `hidden_middle_last` (INSIDE int(L/2)), `gen_top_p`/`gen_top_k` (INSIDE top-p=0.99/top-k=5); `capture_attention`/`capture_layer_fft` raise until the LapEigvals/HSAD reducers land. (2) 5 QA loaders + **paper-terse** prompts + graders (CoQA/SQuAD v2/NQ-Open/TruthfulQA/SciQ) + dependency-free `rouge_l`. (3) `cluster/presets.py` — per-paper preset table (source of truth for MAX_NEW per preset [reasoning 2048, short QA 256–512], N≥few-hundred, paper-T, capture flags) with the 4 high-impact cells + 4 QA-extension cells. (4) `run_inference.py` preset-driven with all QA datasets registered + a per-cell **accuracy-band gate** (VALID/REJECT; REJECTs the AIME24 floor and a 92% ceiling, VALIDs a healthy 55% cell) + `manifest.json` provenance. GOOD_5/spectral path unchanged (all flags off by default). **Next**: `bash cluster/sync_code.sh` → submit the 4 cells as N=30 gate pilots (read the GATE line before scaling to full N) → offline scoring scripts (with `anchor_orient`). Deferred: LapEigvals attention-Laplacian + HSAD layer-FFT on-GPU reducers.

**Prior**: Step 159 — **Branch consolidation: everything merged to `master`; work on `master` from now on.** Merged `experiment/bocpd-features` (Steps 151–156: pivot pilot, Phase-12-Corrected, subset sweep, AIRCC onboarding + verification, replication-grid plan) and `experiment/item6-temperature` (Steps 157–158, renumbered from branch-local 152/153). **Item 6 (temperature variation) is COMPLETE — both gates FAIL, and the negative result is the finding**: temperature diversity HURTS multi-pass fusion (paired B−A = −5.3pp [−10.3,−1.1]); more same-T=1.0 passes HELP (K=5 L-SML 0.912 vs single-pass 0.851, +6.1pp CI excl. 0) → the multi-pass lift is variance reduction at one good temperature, not diversity. Q1 AUROC-vs-T is an inverted-U confounded by accuracy collapse (80%→4%). Two method flags: `spectral_entropy` sign is temperature-dependent (flips at hot T); label-free L-SML underperforms best-single-feature at every T (weak `epr` anchor at low T) — anchor robustness is follow-up #3. **Data debt repaid**: Phase-15 T=1.0 run0 = canonical MATH-500/Qwen-7B raw-trace cache (N=200, full raw schema incl. top-50 logprobs) — unblocks streaming Extension E. 8 CPU follow-ups on the 9 cached runs (top: SC/SE baseline over the 5 same-T passes — also closes Item 5). Merge resolutions: master's float32 `extract_top_k_logprobs` kept (item6's float16 `topk_logprobs_from_scores` retired; Phase-15 Drive caches remain valid); `multipass_lsml_continuous` + Phase-15 notebook merged in. Branches deleted after merge: `pivot-alternatives` (contained in bocpd), `theorem-validation` + `lsml-variants` (already merged), `bocpd-features`, `item6-temperature`. ⚠ Omri: GitHub default branch is still `main` (2 stale initial commits) — switch to `master` in repo Settings → Branches.

**Prior**: Step 156 — **AIRCC verification ladder complete (Stages 2–4): Docker→Pyxis fix, AIME24 demo job 97309 done (2h52m), pkls fetched and validated (acc 2.9/2.5/1.7%), cluster skills updated.** Goal: run our **L-SML continuous GOOD_5** (Steps 134–136 baseline — NOT Step-100/107 numbers) on the **exact** (dataset, model, protocol) grids of the competitor papers, so every thesis number is directly comparable to a published table. Plan finalized (web-agent protocol research + Gemini review + inference-only scoping): **9 protocol cards** — SE-ICLR'23 (arXiv 2302.09664: OPT, CoQA dev 8K + TriviaQA train 8K, K=10 T=0.5, ROUGE-L>0.3, DeBERTa-MNLI), SE-Nature'23 (adapt, don't replicate — GPT-4 judge proprietary), INSIDE/EigenScore (2402.03744: K=10 T=0.5 top-p=0.99 top-k=5, middle-layer int(L/2) hidden states), LapEigvals (2502.17598: all-layer×head attention Laplacian eigvals → PCA-512 → LR probe), LOS-Net (2503.14043: top-K=1000 logprobs + ~1M-param Transformer probe; **G3 gate 72.92 ± 0.45 CONFIRMED correct**), HSAD (2509.13154: layer-axis FFT — 2 GB/sample raw ⇒ compute on-GPU, store per-layer scalars), EPR, Semantic Energy, Spilled Energy (energy papers **blocked on new `token_logsumexp` capture field** — raw logit = logprob + logsumexp; spec in plan). **Inference-only boundary**: cluster = generation + capture ONLY; all scoring local CPU. Data org: `local_cache/replication_grid/{preset_id}/` + `manifest.json` provenance (paper/model/dataset/split/N/K/T/capture flags/job id). 4 high-impact cells first: HotpotQA×Mistral-7B-**v0.2** (LOS-Net head-to-head), GSM8K×Llama-3.1-8B (LapEigvals 92.5 supervised), TriviaQA×Llama-3.1-8B, CoQA×LLaMA-7B-base (INSIDE 80.4 + SE-ICLR). Storage ~160 GB / 10 TB quota. **HF token live on cluster**: hardcoded in gitignored `cluster/submit_inference.sbatch` (untracked via `git rm --cached`; tracked `.template` has REPLACE_ME; synced + verified + chmod 600; note — `ssh aircc "<cmd>"` bypasses the login menu). Implementation follow-up (next sessions): `generate_full` extensions (token_logsumexp, hidden-state capture, at-capture attention/FFT reducers), 5 QA loaders (CoQA, SQuAD v2, NQ-Open, TruthfulQA, SciQ), cluster preset system, offline scoring scripts. **Item 3 dataset priority corrected: CoQA > SQuAD v2 > TruthfulQA** (published SE/SC baselines exist; AmbigQA/PopQA have none).

**Prior**: Step 154 — **Exhaustive L-SML subset sweep (branch `experiment/bocpd-features`): 1.66M subset fits over 32 cells.** Headline: **honest (LOCO) subset selection does NOT beat GOOD_5** (0.6295 vs 0.636 macro; in-cell best-of-65k ceiling 0.7205 = +8.5pp pure selection bias) — GOOD_5 is validated, feature selection stays a minor tweak. All-cell consensus best = {spectral_entropy, sw_var_peak, cusum_max, cusum_shift_idx} (+0.9pp, in-sample). **Every pivot signal HURTS as an added fusion view** (anomaly views −4.9..−7.9pp with 120–179 sig-negative bases; bocpd_ecp −4.8pp; bocpd_ecp_spilled = BOCPD-on-logprobs standalone 0.726 on gsm8k-trace but −1.0pp fused) → Step-151 17th-view thread CLOSED. **ρ≥0.75 subset filter refuted for continuous L-SML** (high-ρ subsets average HIGHER AUROC 0.600 vs 0.556 — clustering absorbs dependence). Sweep GOOD_5 matches table1 CONT 29/29 to ≤0.001; label-free anchor misorients 3/29 cells (the honest cost, now quantified). ⚠ Spilled signs inverted on gsm8k/Llama-8B trace cell (0.27–0.31 oriented) — cross-model sign instability, recheck at Step 132. New: `spectral_utils/subset_sweep.py`, `scripts/{build_derived_views,run_subset_sweep,subset_sweep_report}.py`, `results/Subset_Sweep_Report.html` + `results/subset_sweep/` CSVs/manifests; sweep is chunked+resumable (`--with-trace-cells --workers 7 --yes` resumes anywhere).

**Prior**: Step 152 — **Phase 12 Corrected finished on Colab (Items 4+5).** GSM8K/Llama-8B: **L-SML 1-pass 0.754 beats every multi-pass baseline** (best: SelfCheckGPT-official K=5 0.701; D-SE/LW-SE/SC K=10 all 0.61); third independent run at 75.4–76.0. MATH-500/Qwen-Math-7B: L-SML 0.230 = **global sign flip** (notebook lacks `anchor_orient`; flipped ≡ 0.770 — still far below the 94.4 old-cache reference, unresolved); SC K=10 wins the cell at 0.863. GPQA: all sampling baselines at chance, VC 0.428, L-SML 0.553 best. RAG×4: SelfCheckGPT **below chance** everywhere (official 0.24–0.44, worse than hard). Fresh-cache baselines collapse vs old Phase 12 (GSM8K SC 78.5→60.8, SE 77.4→61.4; GPQA SE 70.6→50.1; MATH SE 87.7→63.0 — NLI-truncation suspect; old table no longer citable until reconciled). **Item 5 verdict: fusion gate NOT passed** — ρ low everywhere but gains ≤+2.0pp; SE K=10 adds ≈nothing over 1-pass spectral, while spectral adds +14.5pp over LW-SE (GSM8K). Follow-ups = new Priority 1 (anchor_orient re-analysis, MATH discrepancy, RAG below-chance, SE-drop reconciliation). Results: notebook Cell 25 + Drive `cache/phase12_corrected/phase12_corrected_results.pkl`.

**Prior**: Step 151 — **Pivot-alternatives pilot (branch `experiment/pivot-alternatives`): both gates FAIL → no pivot.** Assessed the 5 Gemini pivot options (`docs/research_notes/thesis_pivot_options.md`) in `docs/research_notes/thesis_pivot_assessment.md` and piloted the survivors locally with pre-registered gates. Track A (6 anomaly scorers — Mahalanobis/GMM/KDE/IForest/AE/PRAE — as L-SML replacements over the same 16 features, 29-cell battery): ALL FAIL, best gmm2 0.553 vs L-SML continuous 0.651; even label-peeked oracle orientation tops at ~0.60; PRAE ≤ plain AE ≈ Mahalanobis. Track B (2-state HMM, BOCPD, AR/Kalman innovations on raw traces, gsm8k/Llama-8B): none beats DeepConf 0.735 / lsml5 0.754; innovations are entropy-level repackaging (ρ 0.93–0.97) → **KalmanNet NO-GO**; LOCA/IMM/hybrid dropped in assessment. Positive residue: consensus-direction fusion measurably beats direction-free anomaly scoring (strengthens the signal-first FUSE defense), and **bocpd_ecp is a level-orthogonal signal (ρ≈−0.07, 0.685 AUROC alone)** — candidate 17th view, null in 1-cell fusion, re-check free on the queued raw-trace re-inference. New: `spectral_utils/anomaly_utils.py`, `spectral_utils/temporal_models.py`, `paired_boot_delta_auc`, `iter_trace_records`, `scripts/pivot_track{A,B}.py`, `scripts/pivot_report.py`; results `results/pivot_track{A,B}.pkl` + 3 figs. Branch not merged — merge decision with advisors.

**Prior**: Step 148 — **Streaming pivot pilot (local CPU, 4 cells): G1 PASS / G2 FAIL.** Prefix-AUROC + DeepConf shoot-out + online monitor on GSM8K/Llama-8B + MATH-500/Qwen-1.5B (clean) and 2 truncated R1/GPQA cells. Early signal is real — 50% of the trace gives ≥95% of full-trace AUROC (G1 PASS); but fused L-SML does NOT beat the best DeepConf window by the pre-registered +2pp at ≥2 absolute budgets on ≥2 clean cells (G2 FAIL). Only significant spectral edge: earliest 10% of trace, BOTH clean cells (+9.8pp / +4.6pp paired bootstrap). Context: our unsupervised gsm8k 75.4 vs their SUPERVISED hidden-state probe 72.69 on the same model family (arXiv:2601.02170; different label protocol). New `spectral_utils/streaming_utils.py` (incl. `anchor_orient` — per-budget L-SML global-sign coin-flip fix), `scripts/streaming_pilot.py`, `scripts/streaming_pilot_report.py`; figures in `results/figs/`; advisor-ready explainer `results/Streaming_Pilot_Explainer.html`; Extension E (streaming) added to Research_Directions.md with updated priority order (raw-trace regeneration is the next Colab item). Data gaps found: MATH-500/Qwen-7B has NO raw-trace cache anywhere (Phase-12 K10 files are texts-only); no clean R1 cell (all traces capped at 1024). Verdict: streaming pivot NOT supported in current framing; the earliest-prefix edge is the thread to pull, and it needs a re-inference run saving raw traces first.

**Prior**: Step 147 — Bracha reply + Ofir FUSE concern. LR-oracle re-validated on a strict common-cell basis (the ~1pp macro artifact fixed): corrected gaps LR vs L-SML = +4.7 / +3.8 / +3.6pp for 5/9/16 features (LR 68.9/66.8/67.8 vs CONT 64.2/62.9/64.1; in-sample ceilings 70.5/73.7/79.3). New `scripts/oracle_report.py`, `lr_convergence.py`, `lr_weight_analysis.py`; convergence + weight-agreement figures; `logistic_oracle.png` bar chart corrected to common-cell. FUSE positioned (signal + task + dependence-handling differ). 4-point advisor reply drafted (not sent). All local — no model re-runs.


---

## TL;DR — where we are today

**Recommended method** (established by the Step-134 method comparison, 12 variants × 29 cells):
`lsml_continuous_pipeline(feats_dict, GOOD_FEATURES, FEATURE_SIGNS)` — **L-SML continuous** (previously called "CONT" — that term is retired).
- Macro AUROC **70.1%** vs the old binary PROD pipeline 65.2% (**+4.9pp**); **78.3%** on the reasoning regime {MATH-500, GSM8K, QA}.
- On reasoning it beats a simple average (+2.2pp) and even the per-cell oracle best-single-feature (+0.7pp).

**Old production method** (binary, Steps 100–131 — now superseded as the recommendation):
`binarize_classifiers(feats_dict, FEATURE_SIGNS)` → filter to `GOOD_FEATURES` → `lsml_fuse(...)` — the `np.sign()` binarization was the single biggest source of lost signal.

**Key conclusions from Step 134** (independently co-signed by Gemini, `LSML_IMPLEMENTATION_REPORT.md` §13–17):
- **Encoding is the dominant lever**, not features or signs. Continuous beats binary by +4.9pp macro / +7.2pp reasoning.
- **Feature selection is a minor tweak**: continuous L-SML on *all 16* features (`lsml16c`) = 69.2%, within 0.9pp of the selected 5-feature CONT. It helps on reasoning, hurts GPQA. (Answers Bracha Q1.)
- **FEATURE_SIGNS = one global orientation bit**, not a learned dictionary (all 5 GOOD_5 signs equal → a single global flip). Required for deployment orientation; adds zero separability. The paper's internal sign algorithm fails on our error-predicting features (~14% concordance).
- **Robustness (R4) hypothesis rejected**: grouping does *not* insulate against volatile features — avg5 is the most cross-domain-stable (8.9pp std), CONT the least (10.9pp). Fusion's justification is in-regime peak accuracy, not robustness.
- **Operating regime**: spectral L-SML is a reasoning-trace method. GPQA (forced-choice MCQ) and RAG (retrieval-grounded) lack the temporal structure; there a simple average is as good or better.
- **Deliverables**: `Bracha_Reply_Jun2026.md` (answers her 3 Jun-8 questions), updated `results/method_comparison_report.html` (§13–16: lsml16c, R4 robustness, reasoning-only, per-cluster AUC).

**Step 135 — grid completion + benchmarking + narrative report**:
- Full design grid done (5/9/16 × binary/continuous × flat/L-SML + avg). **Continuous beats binary in every cell.** L-SML clustering helps only with many features (5 feat: ties flat; 16 feat: +6.1). Flat-SML-continuous collapses 70→63 as features added; L-SML holds 68–70.
- **Benchmarking (model-matched, CONT, 1-pass)**: MATH 94.4 (win vs SE 87.7/SC 87.2), GSM8K 75.6 (competitive; beats LapEigvals-unsup 72.0), GPQA 52.3 (loss vs SE 70.6), RAG beats SelfCheckGPT 3/4.
- ⚠ **Do NOT reuse Step-117 "ours" numbers** (96.7/71.3/88.1 — leaked supervised). ⚠ **EDIS Phase-13 invalid** (7.7% acc = `\boxed{}` grading bug); fix before citing.
- New: **`results/Spectral_LSML_Report.html`** — story-driven advisor report (this is the one to attach, not method_comparison_report.html).

**Step 136 — cross-cluster weights + full correlation + report v2 (report sent to advisors)**:
- **Across-group fusion weight now stored** per cluster (`cross_weight` col in table2/JSON). Mechanism: it is the leading eigenvector of the clusters' off-diagonal covariance = each cluster's estimated reliability, **not** an average.
  - **K=2 → always 0.50/0.50** (structural — 2×2 zero-diag covariance). So a 2-cluster even split is NOT evidence of adaptive weighting.
  - **K≥3 → weights separate**; a weak isolated cluster gets ≈0 (e.g. pe_mean 0.02 on 16-feat MATH-500). A true average would give it 0.25.
- **pe_mean is domain-dependent — do NOT hard-delete it**: isolated + weight 0.02–0.05 where weak (MATH-500, both QA-CoT cells), but joins a useful `epr,pe_mean` cluster (67.7%, weight 0.24) on GSM8K. L-SML's weighting suppresses it adaptively, only where it should.
- **Full 16×16 dependence matrix** → `results/feature_correlation_16.csv` (new `scripts/feature_correlation_full.py`): band-power block ρ 0.77–0.88, median pair 0.25, pe_mean near-independent. This is the structure L-SML exploits / flat SML ignores.
- **No feature is both strong and stable**: strong features (epr/cusum_max/sw_var_peak) swing ~30pp across domains; stable features (pe_mean range 8.5) are weak everywhere.
- Report v2: removed exec summary; added terminology + aggregation note + 9-feature data + 3 graphs (dependence heatmap, stability scatter, per-domain ranking heatmap). Self-contained except Chart.js CDN.
- **Open**: fix EDIS grading + re-run; complete Phase 14 (GPQA/DeepSeek-R1-8B).

**Step 142 — U-PCR algorithm correction + re-run**:
- Fixed two bugs: (1) weight formula `w_k = (v1@rho/lam1)*v1` hardcoded to v1 even for n_components=2 — corrected to `Σ_c (vc@rho/lamc)*vc`; (2) no λ₂ auto-threshold — added `auto_components=True, lambda2_threshold=0.1`.
- Re-run (29 cells, 3 feat sets): U-PCR-auto gets +0.5pp over old U-PCR-1 on 16-feat (63.0% vs 62.5%), still below L-SML continuous (65.1%). On 5/9 feat the correction slightly hurts (−0.6pp, −0.8pp) because v₂ captures structured noise for low-correlation feature sets.
- λ₂/Trace = 9–34% across cells (28/29 exceed 10% threshold). The paper's 10% threshold is too permissive — 15–20% would be more appropriate for our curated feature sets.
- **v₂ as soft clustering**: (v₁[i], v₂[i]) are continuous cluster coordinates for each feature; U-PCR uses them directly instead of L-SML's hard group assignment. Same structural idea, different tradeoff.
- Updated `results/upcr_comparison.pkl` + new `results/upcr_comparison.png` (3-panel visualization).

**Step 141 — Deep literature review: FUSE, Deep L-SML, STDR, U-PCR**:
- **FUSE finding**: our closed-form eigenvector weights (`w@F`) underperform naive averaging in 7/10 FUSE benchmark settings (Figure 3). Fix: replace with pseudo-label logistic regression on MoM-estimated triplet posteriors `p̂(r_i)` — still fully unsupervised. **Highest-priority next experiment.**
- **RBM = L-SML equivalence** (Lemma 4.1, Shaham et al. 2016): our covariance+eigenvector step IS a single-hidden-node RBM trained by MoM. Stacked RBM (Deep L-SML) handles correlated features without exclusion; relevant for 16-feat expansion where band-power pairs (ρ 0.77–0.88) trigger heavy filtering.
- STDR (Fiedler vector tree recovery): not relevant at current feature counts.
- Step 140 numbers now explained: U-PCR ≈ L-SML on 5/9 features (low-corr regime matches assumption); L-SML wins on 16 because clustering handles band-power block violation.

**Steps 139–140 — U-PCR literature + implementation + empirical comparison**:
- U-PCR (`upcr_fuse`, `upcr_pipeline`) implemented in `spectral_utils/fusion_utils.py` (Tenzer et al. 2022).
- Comparison run across 29 cells, 5/9/16 feature sets. Results (macro AUROC):

| Feature set | L-SML continuous | U-PCR | Delta |
|-------------|-----------------|-------|-------|
| 5-feat | 65.3% | 65.7% | +0.4pp |
| 9-feat | 63.9% | 65.0% | +1.1pp |
| 16-feat | 65.1% | 62.5% | −2.5pp |

- **Conclusion**: U-PCR ≈ L-SML continuous on low-correlation feature sets (5, 9 feat — the assumption E[h_i h_j]=0 approximately holds). L-SML continuous wins on 16 features where correlated features (band-power block ρ 0.77–0.88) violate U-PCR's assumption; clustering handles this, plain eigenvector weighting doesn't.
- Provides the theoretical citation for Step 134: L-SML continuous ↔ U-PCR's ρ̂-proportional weighting. Cite Tenzer et al. (2022) instead of "workaround for Lemma 1".
- Advisor meeting Item 1 (lit search) ✅ complete.

**Prior session (Step 131)**:
- GSM8K cross-dataset verification: spilled energy transfers well (cusum_max_spilled = 0.725 best individual)
- Verbalized confidence: **null result on 1.5B**; adding VC hurts L-SML (−1.77pp)
- Structural finding: within_H/cross ratio = 0.04 (MATH-500) vs 0.99 (GSM8K) — H features are near-independent views on long traces but redundant on short traces
- All changes on branch `experiment/lsml-variants` (commit `f4bc5e8`)

---

## MEETING ACTION ITEMS — Jun 17, 2026 (Ofir, Bracha, Amir)

*Email thread: Omri → Ofir/Bracha/Amir, Jun 17 2026, confirmed by Ofir same day.*

These 6 items are the current priority order. They supersede the old Step 132 GPU-first priority (Step 132 is still pending but de-prioritized until these are underway).

| # | Action | Status |
|---|--------|--------|
| 1 | **L-SML literature search** — find Nadler post-2016 follow-up work extending or improving L-SML | ✅ Complete (Step 141) |
| 2 | **Logistic regression oracle** — supervised LR on 5/9/16 feature sets → upper bound on fusion AUROC (5-fold CV, no in-sample leakage) | ✅ Complete (Steps 142–143 corrected; Step 147 common-cell re-validation + convergence + weight-agreement experiments) |
| 3 | **Extend QA evaluation** — priority corrected (Step 155): **CoQA > SQuAD v2 > TruthfulQA** — these have published SE/SC baselines to compare against; AmbigQA/PopQA have none. Folded into the replication-grid plan (loaders + presets) | In progress — actively running (Steps 160–169); TruthfulQA + SciQ freshly scored, CoQA/NQ-Open mid-flight |
| 4 | **Benchmarking completion** — model-matched comparisons for MATH-500, GSM8K, QA vs SE/SC/SelfCheckGPT | In progress — actively running (Steps 160–169); 3 cells still in queue, Wave-3 scored with 3 wins + 1 tie + 1 edge |
| 5 | **Experiment 1 — Sampling fusion** — fuse SE (K=10) with single-pass spectral features; measure AUROC gain vs each alone | ✅ Complete (Step 152) — gate NOT passed: SE K=10 adds ≤+2.0pp over 1-pass L-SML; spectral adds +14.5pp over LW-SE (GSM8K) |
| 6 | **Experiment 2 — Temperature variation** — run same model at T∈{0.3,0.6,1.0,1.5,2.0}; does higher T improve detectability? Ablate: T-diversity vs just more passes | ✅ Complete (Step 158) — temperature diversity hurts fusion (−5.3pp, CI excludes 0); same-T sampling helps (+6.1pp) |

See `Research_Directions.md` § "Meeting Action Items — Jun 17, 2026" for full experimental designs.

---

## Current best-candidate pipeline constants (not finalised — pending meeting experiments and merge decision)

```python
GOOD_FEATURES = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']

FEATURE_SIGNS = {
    'epr': -1, 'trace_length': 1, 'spectral_entropy': -1,
    'low_band_power': -1, 'high_band_power': -1, 'hl_ratio': -1,
    'dominant_freq': -1, 'spectral_centroid': -1,
    'stft_max_high_power': -1, 'stft_spectral_entropy': -1,
    'rpdi': -1, 'sw_var_peak': -1,
    'pe_mean': -1, 'hurst_exponent': 1,
    'cusum_max': -1, 'cusum_shift_idx': 1,
    # Spilled energy signs — validated on GSM8K; confirm on MATH-500 in Step 132
    'epr_spilled': -1, 'sw_var_peak_spilled': -1,
    'cusum_max_spilled': -1, 'min_spilled': -1,
    # Verbalized confidence — null on 1.5B; may work on 7B+
    'verb_conf': +1, 'verb_conf_1p': +1,
}
```

Note: `min_spilled` sign updated from initial `+1` estimate to `-1` — validated on GSM8K Cell 12 sign check.

---

---

## AIRCC CLUSTER — current state (Step 162, 2026-07-08)

Account `omrisegev1`, group `cycle2_tau_averbuch_prj`. VPN required for all access (`ssh aircc`).

| Resource | Value |
|---|---|
| Owner partition | `power-gpu` (36 nodes, no time limit) / QoS `owner_880` |
| Sandbox partition | `sandbox` (1 node, 2 h limit) / QoS `sandbox_owner_880` |
| Allocation | 5760 GPU-h (1237 used by group) |
| Shared dir | `/shared/cycle2_tau_averbuch_prj/omrisegev1/{code,hf_cache,results,logs,pip_cache}` |
| Model cache | Qwen2.5-Math-1.5B-Instruct prefetched ✓ (snapshot at `$SHARED/hf_cache/hub/...`) |
| NGC image | `nvcr.io/nvidia/pytorch:25.01-py3` — **Pyxis only** (rootless Docker dead on power-gpu since 2026-07-01, cgroup v2 BPF block); **Step 162: `--container-name` removed** — a persistent named container makes two jobs on one node collide on first-create; anonymous per-job containers now, image squashfs still enroot-cached (~8 min first import per node, cheap after) |

**Verification ladder** — ✅ ALL COMPLETE:
- [x] Stage −1: first login + key
- [x] Stage 0: partition/QoS discovered → `cluster/aircc.env`
- [x] Stage 1: sandbox smoke test job 97148 → **PASS** (B200 (10,0), bf16 OK, spectral_utils OK)
- [x] Stage 2: owner-queue smoke test job 97306 → **PASS** (power-gpu/owner_880, Pyxis fix required)
- [x] Stage 3: AIME24 demo job 97309 → **COMPLETED** 2h52m, 30×8×3 temps, gpu-node-05
- [x] Stage 4: `/aircc-fetch edis_aime24` → **VALID** — acc 2.9%/2.5%/1.7% per-candidate (T=0.2/0.6/1.0), 20/20 features finite

**Code sync**: `bash cluster/sync_code.sh` (tar-over-ssh, push-independent). After any local change, re-sync before submitting. Omri must push commits A–C (bf708a5, 88bca56, 0be44e4) from his own terminal (GitHub credential needed).

**Replication grid — EXECUTED (Step 162)**: 5 VALID full-N cells at `$SHARED/results/repgrid/<preset>/` (+ local `cache/repgrid/`, gitignored): losnet_hotpotqa (500, top-1000 logprobs), lapeigvals_gsm8k (500), spilled_triviaqa (500, energy), se_squad_v2 (1000×10, energy), truthfulqa (817×10, energy). 3 paused out-of-band pilots (re-gradeable offline): inside_coqa (0.183), se_nq_open (0.067), sciq (0.900). Two sbatch fixes this run (in `submit_inference.sbatch`): node-local per-job `PYTHONUSERBASE`+`pip --user` (shared-$HOME dill race) and removed `--container-name` (pyxis collision) — concurrent submit is now safe. Submit via `--preset`; pilot+full share `--out` (idx-aligned resume). Reading the gate: `min_minority=30` always REJECTs a K=1 pilot at N=30 — judge by acc-in-band + untruncated trace. Job trail 98667–99090. Detail: HISTORY Step 162.

---

## IMMEDIATE NEXT ACTIONS

*Priority order from the Jun 17 meeting. See table in "MEETING ACTION ITEMS" section above for status.*

### Priority 1 — Phase 12 Corrected follow-ups (Step 152 — run DONE, 4 open issues before numbers are citable)

The run completed (results in Step 152 HISTORY entry + `cache/phase12_corrected/phase12_corrected_results.pkl`). GSM8K is a clean headline (L-SML 1-pass 0.754 > all K=10 baselines). The other cells have open issues:

1. **`anchor_orient` re-analysis** (cheap — analysis cells only, inference caches on Drive, no GPU re-inference): the notebook calls `lsml_continuous_pipeline` without the Step-148 label-free orientation fix. MATH-500 flipped (0.230 → 0.770); the MATH fusion number (0.232) is invalid for the same reason. Add `anchor_orient` to the three analysis cells and re-run Cells 9/10, 15, 20, 25, 26.
2. **MATH-500 discrepancy**: flipped 0.770 vs 94.4 CONT reference (Step 135, old T=1.0 cache). Fresh traces at MAX_NEW=2048, data_loaders prompt. Compare trace-length distributions and per-feature AUCs old-cache vs new-cache before citing either number.
3. **RAG SelfCheckGPT below chance** (official 0.243–0.442, worse than hard on all 4 datasets): check score orientation and grading in the long-context L-CiteEval setting.
4. **SE baseline drops vs old Phase 12** (GSM8K 77.4→61.4, GPQA 70.6→50.1, MATH 87.7→63.0 while SC on MATH is stable): NLI cross-encoder truncation on long traces is the prime suspect; also different question subsets. Old Phase-12 competitor table is not citable until reconciled.

**Still needed separately**:
- Phase 14 full rerun (GPQA/DeepSeek-R1, `Spectral_Analysis_Phase14_GPQA_Comparison.ipynb`): fixed in Step 144, needs fresh Colab A100 (~4–5 hrs).
- QA datasets (Item 3): **CoQA > SQuAD v2 > TruthfulQA** (priority corrected Step 155 — published SE/SC baselines exist; AmbigQA/PopQA have none). Loaders + cluster presets are part of the replication-grid implementation.

### Priority 2 — Temperature variation experiment (Item 6, Colab GPU)

Run Qwen2.5-Math-7B on MATH-500 at T∈{0.3, 0.6, 1.0, 1.5, 2.0}. (T=1.0 and T=1.5 caches already exist.)

### Priority 3 — Temperature variation experiment (Item 6, Colab GPU)

Run Qwen2.5-Math-7B on MATH-500 at T∈{0.3, 0.6, 1.0, 1.5, 2.0} + 4 extra runs at T=1.0 for the ablation. (T=1.0 and T=1.5 caches already exist.)

### Priority 4 — Extend QA evaluation (Item 3, AIRCC cluster)

Priority corrected (Step 155): **CoQA > SQuAD v2 > TruthfulQA** (published SE/SC baselines exist; AmbigQA/PopQA have none). Runs as part of the replication grid on AIRCC — loaders (CoQA, SQuAD v2, NQ-Open, TruthfulQA, SciQ) + per-paper presets are the implementation follow-up to the Step-155 plan.

### De-prioritized (was Priority 1 before meeting)

- **Step 132** (MATH-500 SpilledEnergy GPU run) — still pending, still valid, but not the current focus. Run when a Colab session is available between other GPU tasks.
- **Merge decision** (continuous L-SML + spilled energy → master) — contingent on Step 132.
- **Phase 13** (EDIS vs L-SML on AMC23/AIME24) — EDIS grading bug must be fixed first.
- **Verbalized confidence on 7B** — low priority relative to meeting items.

---

## Research directions and open questions

### What we know works
- **Spectral features of H(n)** work on reasoning-heavy domains (MATH-500, GPQA). GOOD_5, continuous L-SML: best published unsupervised single-pass numbers on these domains.
- **Spilled energy ΔE(n)** cross-dataset validated: competitive individual AUROCs on both MATH-500 and GSM8K, corr(H,ΔE) = 0.984–0.989.
- **Not general-purpose**: short factual QA traces (TriviaQA, WebQ) are structurally incompatible.
- **Continuous L-SML** (+3.53pp over binarized, 25/29 cells) — merge to master pending Step 132.
- **within_H/cross ratio** is a dataset-level diagnostic for L-SML benefit: long reasoning = 0.04 (near-independent, L-SML gains a lot); short traces = 0.99 (redundant, gains less).
- **GOOD_5 is LOCO-validated** (Step 153): held-out subset selection over the full 65k-subset landscape cannot beat it (0.6295 vs 0.636). Candidate tweaks if ever revisited: `low_band_power`→`hl_ratio` (18/29 cells, +0.4pp) or the consensus {spectral_entropy, sw_var_peak, cusum_max, cusum_shift_idx} (+0.9pp in-sample). `cusum_shift_idx` (shift timing) is the only new feature that keeps earning a place.
- **The ρ≥0.75 correlation filter is unnecessary for continuous L-SML** (Step 153): subsets with violating pairs average higher AUROC; the clustering handles dependence (consistent with Steps 135/141).

### Verbalized confidence — model-size gated
- 1.5B: null result confirmed (Step 131). Model doesn't follow "Confidence: X" instruction.
- `parse_verbalized_confidence` is now correct — ready to test on 7B+.
- **Do not include verb_conf in GOOD_FEATURES for 1.5B runs.**

### Open: M=9 orthogonal feature set design
If Step 132 confirms Pearson corr(epr_H, epr_ΔE) < 0.6 on MATH-500, proceed with:
```
Group A — H(n): epr, cusum_max, sw_var_peak
Group B — ΔE(n): epr_spilled, cusum_max_spilled, min_spilled
Group C — structural: rpdi, dominant_freq, stft_max_high_power
```

### What was ruled out
- **Pivot signals as extra fusion views** (Step 153, closes the Step-151 thread): anomaly scorers (Mahalanobis/GMM/KDE/IForest/AE/PRAE), BOCPD (on H(n) AND on the ΔE logprob trace), HMM, AR/Kalman innovations — all reduce AUROC when added to any good subset (paired bootstrap, 32 cells). bocpd_ecp_spilled is a fine standalone signal (0.726 gsm8k-trace) but adds nothing to the fusion.
- **Subset search as a lever**: best-of-65k in-cell is +8.5pp of selection bias; LOCO-honest selection ≤ GOOD_5. Do not chase subsets.
- **Verbalized confidence on 1.5B**: null result (Step 131). Model-size gated.
- **Hedging count**: not formalized, domain-dependent, weaker than spectral. Do not implement.
- **NLI/semantic entropy methods**: require additional model inference. Out of scope for zero-extra-compute.
- **Quantile calibration**: null result. Median binarization only.

---

## Branch situation

| Branch | Status | Contents |
|--------|--------|----------|
| `master` | **Current — all work here** | All Steps through 146; continuous L-SML, baselines.py corrections, Phase 12 Corrected notebook |
| `experiment/pivot-alternatives` | Open — strictly ahead of master (fast-forwardable); merge decision with advisors | Step 151: pivot-options assessment memo + anomaly/temporal pilot (both gates FAIL, no pivot). Step 152: Phase 12 Corrected results + docs (committed here — master fast-forward picks them up) |
| `experiment/lsml-variants` | Superseded (can delete) | Continuous L-SML development — all merged into master via `analysis/theorem-validation` |
| `analysis/theorem-validation` | Merged — can delete | Steps 131–146; fast-forward merged to master (Step 146) |

Colab clones `master` by default. All new work should be committed directly to master or a short-lived feature branch merged quickly.

---

## Running experiments (Colab — GPU needed)

- **Step 132** (priority): `Spectral_Analysis_SpilledEnergy_Verify.ipynb` on `experiment/lsml-variants`
- **Phase 13**: `Spectral_Analysis_MathComp_Phase13.ipynb` — L-SML vs EDIS, Qwen2.5-Math-1.5B
- **Phase 14**: `Spectral_Analysis_Phase14_GPQA_Comparison.ipynb` — L-SML vs VC/SC, GPQA Diamond
  - **Fixed (Step 144)**: MAX_NEW→4096, FORCE_RECOMPUTE=True, Cell 9 uses `lsml_continuous_pipeline`, `lsml_ci` bug fixed. Upload fixed notebook to Drive and run from Cell 1.

---

## Completed phases

| Phase | Notebook / Script | Status | Key result |
|-------|------------------|--------|------------|
| Step 100 | Consolidated_Results | ✅ | Old supervised numbers — do not use |
| Step 107 | Consolidated_Results_LSML | ✅ | L-SML with assumption (iii) — superseded |
| Step 110 | LSML_Diagnostics | ✅ | Consensus FEATURE_SIGNS derived |
| Step 113 | Pilot_RAG_Prompt_Variants | ✅ | V4 prompt wins (+18.6pp), RAG direction dropped |
| Phase 12 | Phase12_Benchmarking | ✅ | SE/SC/VC/SelfCheckGPT baselines computed |
| Step 121–123 | LSML_Optimized | ✅ | 5-feature GOOD_FEATURES finalized, median binarization |
| Step 124–125 | Consolidated_Results_LSML_v2 | ✅ | 5-feat; 29/29 beat chance; HTML updated |
| Step 126 | run_lsml_local.py | ✅ | −5.7pp fusion lift; feature sign instability diagnosed |
| Step 127 | analyze_features.py | ✅ | Cluster structure mapped; trace_length suppression confirmed |
| Step 128 | verify_lsml_paper.py | ✅ | Implementation correct; K_range bug confirmed + fixed |
| Step 129 | experiment/lsml-variants | ✅ | Continuous L-SML: +3.53pp, 25/29 wins — pending merge |
| Step 130 | model_utils + feature_utils | ✅ | Spilled Energy implemented; verification notebook created |
| Step 131 | GSM8K_SpilledEnergy_Verify | ✅ | Spilled energy cross-dataset confirmed; VC null on 1.5B |
| Step 132 | SpilledEnergy_Verify.ipynb | ⏳ | **NEXT — MATH-500 run, needs Colab GPU** |

---

## Best results (reference, do not use Step 100 supervised numbers)

| Setup | L-SML AUC | Notes |
|-------|-----------|-------|
| MATH-500 / Qwen-7B / T=1.0 | **88.2%** | PROD (binary L-SML). CONT (continuous L-SML) achieves **94.4%**. |
| MATH-500 / Qwen-1.5B / T=1.5 | 88.3% | |
| GSM8K / Llama-3.1-8B | 76.0% | vs LapEigvals unsupervised 72.0% |
| GSM8K / Qwen2.5-Math-1.5B | 70.8% | L-SML GOOD_5; best individual 72.5% (cusum_max_spilled) |
| GPQA / Mistral-7B / T=1.0 | 65.4% | Phase 4 best — beaten by 72B (Phase 8) |
| HotpotQA / Mistral-7B | 59.5% | spectral doesn't transfer to multi-hop QA |

---

## Available competitor numbers

### Phase 12 Corrected (Step 152) — paper-accurate methods, fresh shared caches at T=1.0

⚠ MATH-500 L-SML is sign-flipped (see Priority 1); RAG SelfCheckGPT below chance — pending investigation before citing.

| Domain | Model | L-SML 1-pass | Best competitor | Other baselines | Fusion (L-SML+LW-SE) |
|--------|-------|--------------|-----------------|-----------------|----------------------|
| GSM8K | Llama-3.1-8B | **0.754** [0.66,0.84] | SCGPT-official K=5: 0.701 | D-SE 0.614 / LW-SE 0.613 / SC 0.608 / SCGPT-hard 0.601 | 0.758 (+0.4pp, ρ=0.26) |
| MATH-500 | Qwen-Math-7B | 0.230 ⚠flip (≡0.770) | **SC K=10: 0.863** [0.80,0.91] | D-SE 0.630 / LW-SE 0.625 / SCGPT 0.549/0.593 | 0.232 (invalid — flip) |
| GPQA | Qwen2.5-7B | **0.553** [0.45,0.66] | SCGPT-official K=5: 0.512 | D-SE 0.504 / LW-SE 0.501 / SC 0.504 / VC 0.428 | 0.573 (+2.0pp, ρ=−0.19) |
| RAG hotpotqa | Qwen2.5-7B | — | SCGPT hard 0.317 / official 0.243 ⚠ | | |
| RAG NQ | Qwen2.5-7B | — | SCGPT hard 0.393 / official 0.322 ⚠ | | |
| RAG 2Wiki | Qwen2.5-7B | — | SCGPT hard 0.354 / official 0.306 ⚠ | | |
| RAG NarrativeQA | Qwen2.5-7B | — | SCGPT hard 0.477 / official 0.442 ⚠ | | |

### Old Phase 12 (superseded — do NOT cite until reconciled with Step 152; large unexplained SE/SC drops on fresh caches)

| Domain | Model | Competitor | AUROC |
|--------|-------|------------|-------|
| GSM8K | Llama-3.1-8B | SC K=10 | 78.5% [72.0,84.5] |
| GSM8K | Llama-3.1-8B | SE NLI K=10 | 77.4% [70.9,83.5] |
| MATH-500 | Qwen2.5-Math-7B | SC K=10 | 87.2% [72.1,98.4] |
| MATH-500 | Qwen2.5-Math-7B | SE NLI K=10 | 87.7% [79.7,93.9] |
| GPQA | Qwen2.5-7B | SE NLI K=10 | 70.6% [43.6,93.3] |
| GPQA | Qwen2.5-7B | VC K=1 | 67.9% [49.5,83.3] |
| GPQA | Qwen2.5-7B | SC K=10 | 33.6% [11.0,58.2] |
| RAG HotpotQA | Qwen2.5-7B | SelfCheckGPT K=5 | 51.4% [41.5,62.9] |
| RAG NQ | Qwen2.5-7B | SelfCheckGPT K=5 | 57.1% [42.9,70.5] |
| RAG 2Wiki | Qwen2.5-7B | SelfCheckGPT K=5 | 55.3% [35.7,78.3] |
| RAG NarrativeQA | Qwen2.5-7B | SelfCheckGPT K=5 | 52.4% [41.7,65.4] |

---

## Key decisions (permanent — do not revisit)

1. No old supervised numbers — Step 100 historical only.
2. GOOD_FEATURES = `['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']` — final 5.
3. Median binarization only — quantile calibration dropped (null result).
4. HTML table: per domain, per model, same-task/same-model/same-dataset only.
5. Cite Jaffé-Fetaya-Nadler 2016. Never say "Nadler" alone. Method name = L-SML.
6. Never say "MV_EPR" — the method is spectral/L-SML.
7. Branch cleanup done — all work on `master`. `analysis/theorem-validation` and `experiment/lsml-variants` are superseded and can be deleted.
8. Hedging count: ruled out — not formalized, domain-dependent, weaker than spectral.
9. Continuous L-SML (`lsml_continuous_pipeline`) is the candidate replacement — pending Step 132 validation before merge.
10. Verbalized confidence on 1.5B: null result (Step 131). Do not include in GOOD_FEATURES for 1.5B runs.
11. `min_spilled` sign = −1. Validated GSM8K Cell 12.
12. "CONT" is retired. Say "L-SML continuous" (with feature count when relevant: "L-SML continuous 5").
13. LR oracle corrected conclusion (Step 147, common-cell basis): supervised LR beats L-SML by **+4.7 / +3.8 / +3.6pp** on 5/9/16 features (LR 68.9/66.8/67.8 vs CONT 64.2/62.9/64.1); in-sample ceilings 70.5/73.7/79.3. Gap largest on GPQA (+4.9pp) and RAG+QA (+5.8pp), ~0 on reasoning (both near ceiling). "5 best" = named sets non-nested (STABLE_H9 drops spectral_entropy) + overfitting (CV flat while ceiling climbs). LR vs L-SML weights correlate weakly (Spearman ~0.1–0.2). See `SUPERVISED_ORACLE_CORRECTION.md` for evaluation rules; reproduce with `python scripts/oracle_report.py`.

---

## Deferred

- Verbalized confidence on 7B+ — parser is ready; needs one inference run with Qwen2.5-Math-7B on GSM8K
- Phase 10 RAG re-run with variant=4 prompt — low priority
- LapEigvals integration into spectral_utils — potential Group D feature for M=12, low priority
- M=9 orthogonal feature set experiment — contingent on Step 132 confirming ΔE group independence on MATH-500
