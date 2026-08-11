# HANDOFF — Review Codex's hallucination detection + localization advisor report

**Created**: 2026-08-11 (end of the Step 242 session)
**For**: the next session, whose sole job is to **review** an advisor-facing HTML report that
Codex is building on a different machine, find what is wrong and what is right, and drive it to
publishable quality.
**Status when this was written**: every cluster job is finished, all raw data is on Drive and
fetched locally, and **nothing has been committed**.

---

## 0. Scope — read this first, it is wider than "localization"

Codex's report covers **both hallucination DETECTION and hallucination LOCALIZATION**, in **both
reasoning and RAG**, over **all the datasets collected on the cluster**. It takes its visual and
structural cues from **our existing benchmarking HTML** (see §3).

So the deliverable is a 2×2 plus background, not a single localization story:

| | **Detection** (does this answer contain a hallucination?) | **Localization** (where is it?) |
|---|---|---|
| **Reasoning** | answer-level grid: GSM8K / MATH / TriviaQA / NQ-Open / CoQA / SQuAD / SciQ / TruthfulQA cells vs the published roster | ProcessBench (first erroneous step) · PRMBench (every step) |
| **RAG** | RAGTruth response-level evidence contrast | RAGTruth token/char span · GASP sentence · RefChecker claim |

**Background, not panels**: HLE (answer-level transfer, currently ungraded) and SemGrad
(SciQ / TruthfulQA collection).

Omri's two explicit requirements:

1. The HTML is **for advisors** — modelled on our existing benchmark pages.
2. **For each reproduction it must state exactly what it ran** — model + revision, dataset +
   revision, prompt, decoding, metric implementation, and the command. Not "we ran LettuceDetect";
   rather the checkpoint, the entry point, the `max_length`, and the metric source.

Omri's words: *"I want it to be perfect."* Treat this as an audit, not a proofread.

Governing specs: `docs/experiments/FOUR_LOCALIZATION_BENCHMARKS_CLUSTER_HANDOFF.md` (§4
apples-to-apples + record-keeping, §11 score freezing, §12 report structure, §13 acceptance) for
the localization half; `Research_Directions.md` + `BENCHMARKING_COMPETITOR_GUIDE.md` for the
detection half.

## 1. How to get Codex's output

Omri will ping with the location. It is on **another machine**, so it is probably a git branch, a
pushed remote, or a pasted directory. Establish this first — do not guess:

```bash
git fetch --all --prune && git branch -a --sort=-committerdate | head -20
git log --oneline --all --since="2026-08-11" | head -30
```

If it arrives as a branch, review it **without merging** (`git diff master...<branch>`, or
`git worktree add`). Do not merge or commit: **Omri reviews and commits.**

## 2. Ground rules that do not change

- **Read `PROGRESS.md` first** (top addendum = Step 242 state), then HISTORY.md Step 242, then this.
- **Do NOT commit or push.**
- **Anti-hallucination is the whole point.** If a number cannot be traced to a source file, mark it
  **UNVERIFIED** and surface it. Never invent, never round-to-plausible, never reconstruct from
  memory. A missing number is a finding, not a hole to paper over.
- **Terminology bans (advisor-facing, strict)**: never "Nadler" (the method is **L-SML** /
  continuous L-SML), never "MV_EPR", never the word "recommended" in advisor mail/HTML, never a
  comparison against the supervised `best_nadler_on` (that was a bug). `scripts/advisor_report.py`
  has a **built-in guardrail scan** — any regenerated advisor page must pass it clean.
- **Never call the Qwen3-8B judge control "uPRM"** — it is *our own no-training LLM-as-a-Judge
  reconstruction*. Real uPRM needs RL training (~44 GPU-h, no public code).
- **Which method is "ours" — do not infer it.** Hand-picked subsets (`GOOD_5`, `GOOD_6`, `LOCO_5`)
  are **reference/compatibility rows, never the contribution**. The maintained U-PCR arm goes
  through `spectral_utils.upcr.upcr_fit` over the full pool with `sign(ρ̂)` polarity
  (`scripts/labelfree_standing_report.py::upcr_rho_oriented`) — **not**
  `fusion_utils.upcr_pipeline` / `eval_subset_flex(fusion='upcr')`, which is the legacy path. If
  the report heads a table with a fixed subset as "our method", that is a **fatal** framing error.
- **Report format**: ours-vs-theirs is **one grid**, columns per method, rows per metric, direction
  in the row label, winner marked explicitly. No shorthand labels — never "R1/R2", "channel A/B",
  "T1–T4"; name each experiment by what it does and what it found.
- **Published-roster headline**: advisor-facing comparisons lead with the cited-paper scoreboard.
  The seq-logprob / perplexity / normalized-entropy audit is appendix material, never the headline.

## 3. Style references (what "inspired by our benchmarking HTML" means)

| Artifact | Why it is the reference |
|---|---|
| `results/action_items/index.html` + `item4_benchmarking.html` + `advisor_scrutiny.html` | the canonical advisor benchmarking pages |
| `results/action_items_jul2026/index.html` and its `item1_failure_deepdive/cell_*.html` | per-cell deep dives with worked examples |
| `results/ours_only_localization_v1/REPORT.html` / `.md` / `ADVISOR_BRIEF.md` | the localization deliverable (`scripts/build_gl_liu_report.py`) |
| `results/gl_liu_factorial_v2/REPORT.md` | how a controlled factorial is written up honestly |
| `results/hard_filter_dufs_liu_24cell/REPORT.md` | the 24-cell answer-detection background |
| `scripts/advisor_report.py`, `scripts/repgrid_report.py`, `scripts/inscope_report.py` | the generators; note the regen chain below |

**Regen chain** (a real failure mode): the QA table reads `results/repgrid/headline_X_vs_Y.csv`,
produced by `score_repgrid → repgrid_report → advisor_report`. A new cell that is not added to
`advisor_report.py`'s order-list with its **exact** CSV model string is **silently dropped** from
the page. If Codex added cells, verify they actually appear.

---

## 4. THE REVIEW CHECKLIST

### 4.1 Structural

- [ ] Detection and localization are **clearly separated**, and reasoning vs RAG within each. A
      reader must never have to guess which question a number answers.
- [ ] **NO cross-task macro average across incompatible metrics.** Response AUROC, ProcessBench
      F1, PRMBench step F1 and character-overlap F1 do not average. A "macro summary" must be a
      **status table of each task's own primary metric**.
- [ ] **RefChecker's three settings are never pooled** (zero/noisy/accurate context differ by 0.23
      macro F1 — they are three different tasks).
- [ ] **Fidelity label (1–4) on every competitor row**: 1 exact reproduction · 2 protocol
      reproduction · 3 adaptation · 4 published context only. Levels 2–4 must **never** be called
      an exact reproduction.
- [ ] Supervision / access / compute / inference-pass table present, so a supervised ceiling
      (LettuceDetect, Qwen2.5-Math-PRM-7B) is never displayed as a peer of a label-free score.
- [ ] Any field failing spec §13 reads **pending** / **protocol mismatch** / **blocked** — never a
      published number borrowed from another model or split.

### 4.2 Score freezing and label isolation

- [ ] Fitting code never saw evaluation labels. Mirror `scripts/gl_liu_external_v1/run.py`:
      import scorers unchanged → fit → apply → `_score_hash` (explicit `<f8`/`<i8` dtype tags) →
      `FREEZE_MANIFEST.json` with `labels_seen_during_fit: false` → labels only in a **separate**
      evaluate command. Stronger variant to demand:
      `hard_filter_dufs_liu_benchmark.py::verify_freeze`, which re-hashes and **raises if any
      score key contains the substring "label"**.
- [ ] **Thresholds only from declared non-test calibration.** RAGTruth has an unused dev slice at
      `dataset_cache/ragtruth_ec_full/dev/` (150 source_ids, 5,724 items, 900 responses), never
      scored. A threshold tuned on test is **fatal**.
- [ ] **Bootstrap grouping**: RAGTruth by `source_id`; PRMBench/ProcessBench by **complete
      problems**. Evidence conditions, sentences, claims and steps from one source are not
      independent.
- [ ] IU-PCR reported **beside** DUFS-LIU, with the **λ=0 exact-identity invariant** asserted
      (`laplacian_iu_path` at λ=0 returns ordinary IU-PCR weights verbatim). Without it any claimed
      Laplacian gain is unfalsifiable.
- [ ] Frozen settings unchanged: 2 components, k=7, λ_global=0.1, λ_local=0.3, DUFS seeds
      (11, 23, 37), 80 epochs. Tuning these on the new panels is method selection smuggled into a
      measurement cycle — spec §15 forbids it.

### 4.3 DETECTION traps

**Answer-level grid (reasoning + QA)**
- [ ] **Low `valid_rate` cells must carry their coverage.** `sciq_llama8b` valid_rate **0.198**
      (n=198), `se_squad_v2_llama8b` **0.293** (n=2933), `spilled_triviaqa_llama8b` **0.512**
      (n=256), `epr_triviaqa_mistral24b` **0.621**. Quoting `spilled_triviaqa` **0.9620** as a
      headline without its 51% coverage and n=256 would be **materially misleading** — this is the
      single most likely detection-side overclaim.
- [ ] `se_squad_v2_llama8b` has **no published Y value** (NaN). It cannot appear in a
      ours-vs-theirs win column; it is an ours-only row.
- [ ] Win/loss counts must be computed, not asserted. In `headline_X_vs_Y.csv` we win clearly on
      the Mistral/Nemo/Phi LapEigvals cells and lose clearly on CoQA, NQ-Open, TruthfulQA,
      SEICLR-TriviaQA and LOSNet-HotpotQA. **A page that shows only wins is a finding.**
- [ ] The 24-cell DUFS-LIU vs IU-PCR difference (0.776562 vs 0.776087) is **+0.0005 with
      uncertainty including zero**. It must be presented as *current implementation standard*, not
      as a proven Laplacian gain.
- [ ] Scope: **GPQA and multi-hop RAG accuracy are out of thesis scope**; GPQA features measured
      uniformly at chance (0.51–0.55). If they appear as results rather than as documented
      negatives, that is wrong.

**RAG detection (RAGTruth evidence contrast)** — the highest-risk claim in the whole report:
- [ ] **The preregistered novelty test does NOT pass.** Best arm
      `ec_dufs_liu_evidence_graph` beats `fusion_isolation_naive_avg` by **+2.51pp with 95% CI
      [−0.58, +5.72] and P(Δ≤0) = 0.066** — the CI crosses zero. `ec_upcr` (P=0.3875) and
      `ec_dufs_liu_temporal` (P=0.399) are statistically indistinguishable from naive averaging.
      **Any wording that presents this as a confirmed win over naive averaging is fatal.**
- [ ] What *is* confirmed: the evidence-contrast **intervention design** works —
      `full_context_only_dufs_liu` (P(Δ≤0)=1.0) and `likelihood_drop` (P=0.98) are significantly
      **worse** than naive averaging. Say that; it is the real finding.
- [ ] Arm 5b (`ec_dufs_liu_evidence_graph`) is **one reading of the preregistration's graph
      description, flagged unconfirmed** since Step 237. It must not be presented as a validated
      mechanism.
- [ ] A sign bug in `anchor_orient` once put three arms below chance (AUROC 0.25–0.31); fixed via
      `anchor_sign` with a regression test in `smoke()`. Any arm scoring below 0.5 is a bug signal,
      not a weak result.

### 4.4 LOCALIZATION traps — each silently produces a plausible wrong number

**Token/span (RAGTruth / LettuceDetect)**
- [ ] Entry point is `predict_prompt(prompt, answer)`, **not** `predict(context=[...], question=...)`
      — the latter re-wraps input in the library's own `"passage N:"` template, a train/test
      mismatch worth 3.4 F1 points. First thing to check if the number looks off.
- [ ] Predicted spans are **answer-relative** char offsets and RAGTruth gold indexes into
      `response` — directly comparable, no offset surgery. Any re-basing is a bug.
- [ ] Our arm must **not** re-run LM inference — it reuses `dataset_cache/ragtruth_ec_full/test/`
      (1.23 GB, 16,200 items / 2,700 responses), recovering char offsets by re-tokenizing
      `row["response"]` (`add_special_tokens=False, return_offsets_mapping=True`).

**Sentence (GASP)**
- [ ] Target is **0.713 response AUC / 0.673 span AUC** (Qwen2.5-1.5B). **~0.73 / ~0.67 is the
      paper's cross-scorer average** over three scorers, two of which we never ran — and older
      manifests in this repo contain that mistake.
- [ ] GASP is **fidelity level 2**: arXiv-only, no code release or response-ID list located, so our
      own seed and our own sentence splitter.
- [ ] Exact-vs-approximate JSD must be compared **on identical rows**;
      `scripts/rag_ec_v1/gasp.py::jsd_source` records which source produced each score. A panel
      must never silently mix them.
- [ ] 1,119 context-cap and 705 answer-cap hits are **the paper's protocol as specified**, not
      truncation damage.

**Claim (RefChecker)**
- [ ] Must state this measures the **checking stage only** — human labels attach to
      **Claude-2-extracted** triplets, so extraction is out of scope by construction. Calling it an
      end-to-end RefChecker reproduction is a false claim.
- [ ] Our scalar score appears **only** under the binary collapse (Contradiction + Neutral →
      unsupported), never in the paper's three-way column.
- [ ] Proprietary GPT-4 / Claude-2 configurations are quoted as **published context only** (level 4).

**Every-step (PRMBench)** — three details that invert the panel:
- [ ] Evaluated question is **`modified_question`**, not the dataset's own `question` field.
- [ ] **`labels[i] == 1` means the step is VALID** — the positive class of the official F1 is
      *correct* steps, so a risk-oriented score must be **inverted**.
- [ ] The all-correct control class is **constructed** by the loader from `redundency` rows, is
      scored, and is **not pooled** into totals.
- [ ] Corpus: the paper's **83,456** is raw; the loader drops 5 duplicate `multi_solutions` rows →
      **83,371** evaluated. 100 rows annotate an out-of-range error step — upstream treats them as
      **inert**; keep them, do not drop or "repair".
- [ ] **The availability constraint must appear**: 71.0% of steps are shorter than 32 tokens
      (median 24, 3.5% under 8), while `compute_stft_features` needs 32 and
      `compute_spectral_features` needs 8. A weak every-step number without this is under-explained.
- [ ] `redundency`/`circular` use the **validity fallback** (no redundancy head) — a declared
      adaptation; `used_redundancy_head == False` and `adaptation_note` must be surfaced.

**First-error (ProcessBench)**
- [ ] **Unit mismatch**: `processbench.first_error_f1` returns **percentages (0–100)**;
      `localization_metrics.processbench_f1` returns **fractions (0–1)**. Both in one table = a
      silent 100×.
- [ ] Our ProcessBench inference must **not** have been re-run; reuse
      `results/localization/processbench_qwen3_8b/` and `results/gl_liu_external_v1/llama31_8b/`.
- [ ] The critic row is **"ProcessBench protocol reproduction with a different critic model"** —
      QwQ-32B-Preview was deliberately not run.
- [ ] **Do not quote the N=30 pilot numbers** (see §5).
- [ ] GL-LIU is a **calibrated unsupervised** method, not fully label-free: labels are used on two
      declared development cells for component selection and inside each calibration half for the
      threshold. Claiming "fully label-free decision policy" is wrong.

### 4.5 "What exactly did it run"

Per spec §4, every reproduction records: dataset + revision · paper + official-code revision ·
model + checkpoint revision · prompt, tokenizer, truncation, decoding · generator internals seen? ·
separately trained evaluator? · another generation or teacher forcing only? · training labels
human/synthetic/none · threshold used dev labels? · runtime, accelerators, peak memory, model
passes.

Check against the Stage-A protocol locks in `cluster/manifests/`, written **before** submission:
`ragtruth_lettuce_large_v1`, `gasp_exact_v1`, `prmbench_v1`, `refchecker_v1`,
`pb_prm_qwen25math7b_v1`, `pb_critic_qwen72b_v1`, `pb_uprm_baseline_qwen3_8b_v1`,
`pb_llama31_8b_external_v1`, `ragtruth_ec_v1`, `hle_pilot_v1`, `semgrad_pilot_v1`.

⚠ **Known artifact**: a chained resume wall **overwrites its predecessor's `elapsed_sec` and
`job_id`**. `gasp_exact` reports 4.2 s / job 177895; the truth is job **177894's 2 m 39 s**. Same
for `prmbench_*` (177823/177824 are resume walls). **Runtime claims must come from `sacct`.**

---

## 5. GROUND-TRUTH NUMBERS — diff every claim against this

If Codex disagrees with a number here, one of the two is wrong; resolve it, never average.

### 5.1 Answer-level detection — `results/repgrid/headline_X_vs_Y.csv`
20 cells × 2 methods (`lsml`, `upcr`); **X = ours, Y = published competitor**, `delta = X − Y`.
Clear wins: `lapeigvals_gsm8k_mistral24b` +0.2483 (lsml) / +0.2828 (upcr) · `lapeigvals_gsm8k_nemo`
+0.1713 · `lapeigvals_gsm8k_phi35` +0.1465 · `spilled_triviaqa_llama8b` +0.0995 ·
`semenergy_triviaqa_qwen3_8b` +0.0680 · `ars_gsm8k_r1distill8b` +0.0151.
Clear losses: `seiclr_triviaqa_opt30b` −0.1996 · `truthfulqa_llama8b` −0.1710 ·
`se_nq_open_llama8b` −0.1627 · `losnet_hotpotqa_mistral7b` −0.1482 · `inside_coqa_llama7b` −0.1199 ·
`sciq_llama8b` −0.1127 · `lapeigvals_gsm8k_llama8b` −0.1064 · `internalstates_gsm8k_qwen25_7b`
−0.0961.
**Coverage caveats**: `sciq_llama8b` valid_rate 0.198 · `se_squad_v2_llama8b` 0.293 (and **Y is
NaN** — no published comparator) · `spilled_triviaqa_llama8b` 0.512, n=256 ·
`epr_triviaqa_mistral24b` 0.621.

24-cell answer detection: DUFS-LIU **0.776562** · IU-PCR **0.776087** · deployed U-PCR
**0.773528** (difference's uncertainty includes zero).
Reference subsets, not the contribution: GOOD_6 0.7594 · LOCO_5 0.7705 (24 cells) ·
a6.pl_dufs 0.7524 (the label-free selector of record).

### 5.2 RAG detection — `results/rag_ec_v1/full_test_split_result.json`
N=2,700 responses, 450 `source_id`s, grouped bootstrap.

| Arm | AUROC | 95% CI |
|---|---|---|
| `ec_dufs_liu_evidence_graph` | 0.7536423 | [0.7318264, 0.7752513] |
| `ec_upcr` | 0.7340944 | [0.7117809, 0.7553145] |
| `ec_dufs_liu_temporal` | 0.7329217 | [0.7121275, 0.7549839] |
| `fusion_isolation_naive_avg` | 0.7289883 | [0.7053392, 0.7518408] |
| `gasp_reproduction` | 0.7137323 | [0.6914524, 0.7368903] |
| `likelihood_drop` | 0.6946219 | [0.6717700, 0.7181041] |
| `full_context_only_dufs_liu` | 0.6424144 | [0.6179456, 0.6649879] |

**Preregistered test vs `fusion_isolation_naive_avg`** (this is the claim the campaign rests on):
- `ec_dufs_liu_evidence_graph` **+0.025054, CI [−0.005837, +0.057210], P(Δ≤0) = 0.066** ← crosses zero
- `ec_upcr` +0.005091, P = 0.3875 · `ec_dufs_liu_temporal` +0.004401, P = 0.399 (both ties)
- `gasp_reproduction` −0.014605, P = 0.824 · `likelihood_drop` −0.033880, CI [−0.066045, −0.001492]

### 5.3 LettuceDetect-large — RAGTruth test, 2,700 responses (fidelity 1)
`predict_prompt`, `max_length=4096`, `min_confidence=0.0`
- **F1 0.7928994082840236**, P 0.8045851528384279, R 0.7815482502651113
- tp 737 · fp 179 · fn 206 · tn 1578 · gold-hallucinated 943 · overlap hits 702
- **n_truncated 0** · published 0.7922 · **gate delta +0.0006994** · runtime 175.8 s
- the `max_length=8192` arm is **byte-identical** — a null result, not a second number

### 5.4 GASP exact — RAGTruth (fidelity 2)
`Qwen/Qwen2.5-1.5B-Instruct`, K=5 sentence-grouped, caps 700/200, seed 0
- 2,508 items · **400 responses (exactly 200/200)** · Data2txt 214 / Summary 186
- conditions per response min 4, max 7 · pools 783 hallucinated / 1,017 clean
- context-cap hits 1,119 · answer-cap hits 705
- mean full-vocab JSD **0.06982875807791554** · max 0.3391658 · ln 2 = 0.6931472
- **target 0.713 / 0.673**, NOT 0.73/0.67 · real runtime **2 m 39 s** (job 177894)

### 5.5 RefChecker — 10,733 claims, 3/3 settings, `n_missing_files: 0`
competitor `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli` (fidelity 2); ours Qwen3-8B (3)

| Setting | 3-way accuracy | Macro F1 | n |
|---|---|---|---|
| overall | 0.6931892294791764 | 0.580474087761565 | 10,733 |
| zero_context (NQ) | 0.7336547152756855 | 0.6922888894353139 | 3,319 |
| noisy_context (MS MARCO) | 0.7619883040935672 | 0.46159028032139154 | 3,420 |
| accurate_context (Dolly) | 0.600650976464697 | 0.4336316229263278 | 3,994 |

⚠ Per-class figures (Entailment F1 0.8091 / Neutral 0.2771 / Contradiction 0.2457, supports
6129/804/481) are from the **earlier 7,414-claim, 2-setting** pass. **Do not mix with the 10,733
totals**; recompute if shown.

### 5.6 PRMBench — Qwen2.5-Math-PRM-7B ceiling (fidelity 1)
6,969 meta rows · `n_reward_count_mismatch 0` · `validity_rate 1.0` · 758 control rows ·
`used_redundancy_head false`
- correct_step_acc **0.9543447922303552** · wrong_step_acc **0.30466195147919994**
- total_step_acc 0.8518789507142771 · first_error_acc 0.3768019884009942 · similarity 0.0171
- precision 0.879948528735934 · recall 0.9543447922303552 · **f1 0.9156379584782177**
- negative_precision 0.5554631170271769 · **negative_f1 0.3934973724276804**
- by category (f1): simplicity **0.8830890537877457** · soundness 0.9267223213723843 ·
  sensitivity 0.9345175380314651
- corpus: raw 6,216 rows / **83,456** steps · 5 dup rows dropped · **83,371** evaluated ·
  94,203 incl. controls · 160 empty error_steps · 100 out-of-range

**Our telemetry** (Qwen3-8B): 6,969 rows · 94,203 spans · mean 13.52 steps · mean 370.5 tokens ·
**n_unmapped_steps 0** · 3 misaligned · **frac_steps_lt_8 0.0354** · **frac_steps_lt_32 0.7101** ·
median step 24 tokens

### 5.7 ProcessBench, full N = 3,400

Qwen2.5-Math-PRM-7B (supervised ceiling):

| Subset | error acc | correct acc | F1 | n_error / n_correct |
|---|---|---|---|---|
| gsm8k | 71.0145 | 96.3731 | **81.7729** | 207 / 193 |
| math | 67.0034 | 91.1330 | **77.2272** | 594 / 406 |
| olympiadbench | 54.7655 | 85.2507 | **66.6894** | 661 / 339 |
| omnimath | 54.6772 | 83.4025 | **66.0519** | 759 / 241 |

Qwen2.5-72B-Instruct critic (`max_new=8192`):

| Subset | error acc | correct acc | F1 | trunc / unparsed |
|---|---|---|---|---|
| gsm8k | 61.35 | 95.85 | **74.82** | 0 / 0 |
| math | 45.62 | 90.64 | **60.70** | 1 / 1 |
| olympiadbench | 35.40 | 88.50 | **50.57** | 1 / 1 |
| omnimath | 36.50 | 87.55 | **51.52** | 6 / 6 |
| **macro** | | | **59.40** | 8 / 8 (0.24% each) |

Our no-training LLM-as-a-Judge control, Qwen3-8B (**NOT uPRM**), `n_failed_tokenization 0`:
gsm8k **15.27** · math **10.38** · olympiadbench **5.13** · omnimath **7.79**

**⚠ The Step-241 N=30 pilots were misleading; do not quote them.** Critic omnimath fell **−14.4**
at full N (65.9 → 51.52), math rose **+10.7** (50.0 → 60.70); the PRM ceiling's omnimath fell
**−6.9** (73.0 → 66.05).

### 5.8 Our own reasoning-localization numbers
- GL-LIU v1 ProcessBench **F1 31.36%**, exact-loc 21.79%, tol-1 46.76%, clean-acc 57.99%
- unified core-five DUFS-LIU **31.72%** · broad-28 local **29.03%** · Mind the Gap control **25.71%**
- external family (Llama-3.1-8B): gl_liu_v1_frozen **31.71** · unified_core_five_dufs 31.62 ·
  **baseline_max_entropy (no fusion) 31.50** · mindgap_control 25.45
  → we clearly beat Mind the Gap (+5–10pp per subset) but **do not** clearly beat the simplest
  transparent baseline; the margin flips sign per subset and 0.21pp is noise at ~850 rows/subset.
  A page claiming a win over max-entropy is wrong.

### 5.9 Background (not panels)
- **HLE**: 2,158 items, Qwen2.5-72B-Instruct, accuracy **0.076** (164/1,994) under the
  **placeholder ROUGE-L grader**, `gate_ok: false` (outside [0.2, 0.85]). A floor, not the real
  accuracy. **Do not compute AUROC or fit anything on this cell** until a real judge regrade lands.
- **SemGrad**: SciQ 1,000 rows (accuracy 0.648) · TruthfulQA 817 rows (accuracy 0.308),
  Qwen3-4B-Instruct-2507.
- **HUB and ReDe remain BLOCKED** (no controlled generation protocol / no official code).

## 6. Cluster corpus inventory — 59 GB of results, what each is for

| Directory | Role |
|---|---|
| `repgrid` (17 G), `regen` (30 G), `pilots`, `pilot_cap` | the answer-level **detection** grid + regenerations |
| `pb_qwen3_4b`, `pb_qwen3_8b` | **our** ProcessBench telemetry (first-error) |
| `pb_llama31_8b_full` | external-family ProcessBench confirmation |
| `pb_prm_qwen25math7b_full`, `pb_critic_qwen72b_full`, `pb_uprm_baseline_qwen3_8b_full` | ProcessBench **competitors** (new) |
| `prmbench_qwen25math7b_full`, `prmbench_qwen3_8b_telemetry_full` | **every-step** panel (new) |
| `ragtruth_ec_qwen25_15b_test` (1.2 G), `_dev` (427 M) | **RAG detection** evidence contrast (dev slice never scored) |
| `ragtruth_lettuce_large_span_full`, `_ml8192` | RAG **span** competitor (new) |
| `gasp_ragtruth_exact_qwen15b_full` | RAG **sentence** competitor (new) |
| `refchecker_knowhalbench_open_full` | RAG **claim** panel (new) |
| `evdrop_{gsm8k,math}_qwen3_{4b,8b}` | evidence-drop reasoning cells |
| `gateb_gsm8k_*` | Gate B prompt/template validation cells |
| `hle_full`, `semgrad_full` | background answer-level collection |

Local copies of the 10 new dirs: `dataset_cache/four_localization/` (2.3 GB, plus
`ALL_MANIFESTS.json` with every manifest and a byte inventory). Same on Drive at
`hallucination_detection/cluster_results/`, byte-verified.

## 7. Independent verification commands

```bash
python -m spectral_utils.prmbench --corpus   # 9 checks incl. the 6,216 / 83,456 reproduction
python -m spectral_utils.refchecker           # 4 checks
python -m spectral_utils.ragtruth              # 9 checks incl. the frozen-chunker invariant
python -m spectral_utils.processbench          # 12 checks incl. the 0.0-vs-None F1 regression
python scripts/rag_ec_v1/gasp.py --smoke       # 6 checks
python scripts/smoke_selectors.py
python scripts/build_glossary.py --allow-gaps  # confirm the guardrail scan is clean
```

## 8. Known-open items Codex may not have handled (not its fault)

- `scripts/build_glossary.py` **fails its own coverage gate** on four pre-existing selector
  families (`a8_lscae`, `a9_dpp`, `a10_mmdufs`, `a11_rfae_scfs`).
- **Resume walls overwrite manifest timing** (§4.5) — a cost table built from manifests is wrong.
- The uPRM control was scaled to full N against the spec's own rule; correct handling is the right
  name plus a limitations note.
- No `verify_freeze` existed for the new panels before this handoff — check Codex built one.

## 9. Deliverable of the review

A table: **finding · severity (fatal / material / cosmetic) · file:line · evidence · suggested
fix**. Lead with what is *right* as well as wrong — Omri asked for both.

**Fatal** = anything that would put a wrong number in front of an advisor: a cross-task average; a
mislabelled fidelity level; a threshold tuned on test; an inverted PRMBench polarity; a supervised
ceiling shown as a peer; a hand-picked subset headed as "our method"; **the RAGTruth novelty claim
presented as confirmed when its CI crosses zero**; a win claimed over `baseline_max_entropy` on
ProcessBench; or a low-coverage cell (`spilled_triviaqa` 0.9620 at valid_rate 0.512) quoted without
its coverage.

Fix what is safely fixable, leave the rest as a punch list, and **do not commit**.

---

## 10. Paste-ready prompt for the new session

> Read `PROGRESS.md`, then HISTORY.md Step 242, then
> `HANDOFF_codex_localization_report_review.md` in full.
>
> Codex has produced an advisor-facing HTML report covering hallucination **detection and
> localization**, in **reasoning and RAG**, over all the datasets we collected on the cluster,
> built on a different machine. Its location: **<PASTE LOCATION>**.
>
> Your job is to review it and make it perfect. Audit it against the checklist in §4 of that
> handoff, diff every number against the ground-truth tables in §5, confirm the cluster corpus
> coverage in §6, and verify that each reproduction states exactly what was run (§4.5). Run the
> independent checks in §7 rather than trusting the report.
>
> Pay particular attention to the claims that are NOT confirmed and must not be presented as if
> they were: the RAGTruth evidence-contrast novelty test (CI crosses zero), our ProcessBench margin
> over the plain max-entropy baseline, and any low-coverage detection cell quoted without its
> valid_rate.
>
> Report findings as: finding · severity · file:line · evidence · fix. Tell me what is good as well
> as what is broken. Fix what is safely fixable and leave the rest as a punch list.
> **Do not commit or push.**
