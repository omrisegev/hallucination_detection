# Handoff: an honest, adversarial review of the PRM boosting line (2026-10-25)

## Why this session exists

Omri, 2026-10-25 (Hebrew, verbatim): "אני רוצה review איכותי אמיתי על מצב הפרוייקט. מרגיש לי שאנחנו מתחפרים
פה במשהו שאי אפשר להציג למנחים בכנות ואי אפשר לפרסם. חשוב לבחון את זה ולהודות מראש".

He asks for a real, high-quality review of where the project stands. His concern: we are digging into something that
cannot be presented honestly to the advisors, and cannot be published. He wants this examined and admitted up front.

**Your job is to judge the line, not to defend it.** The previous session (the one that wrote this file) built much
of it. Read section 3 as a list of suspicions to verify or refute, not as conclusions.

## 1. Rules for the reviewer

- **Language.** Chat with Omri in Hebrew, with plain-language explanations first (see CLAUDE.md, Communication style).
  Advisor-facing documents are in English.
- **Start read-only.** No new experiments, GPU or AIRCC work without Omri's explicit OK. Cheap CPU checks that decide
  a question are fine once you have said what they test.
- **Every number names its file and the N checked.** Re-verify from the tables; do not trust this handoff's numbers.
- **Get an unanchored read.** Use independent subagents. Give them the code, tables and papers, NOT this file's
  conclusions. The `/red-team` pattern is appropriate for any headline number.
- **Steelman before you dismiss.** For each red flag, state the strongest defence, then whether the evidence beats it.
- **Admit sunk cost openly.** The project has about 477 HISTORY steps. Many directions closed negative (CT7, Joint,
  DUFS-LIU, AUTO, AUTO-F, tensor MoM, temporal readouts and more). "We invested a lot" is not evidence.

## 2. The claim as it currently stands

**PRM boosting:**
- A frozen PRM scores each step.
- In the same teacher-forced pass, 16 token-probability "telemetry" channels are read from the PRM's backbone.
- Each view is z-scored within its answer and fused with label-free L-SML (internal name CHAIN_N16, a fixed partition
  in which the PRM sits in its own group).
- Steps are flagged per answer with the GAP rule.
- Metric: official PRMBench PRMScore.

Advisor documents:
- `docs/meetings/advisor_status_2026-10-21/PRM_BOOSTING_STATUS.html`
- `docs/meetings/advisor_status_2026-10-21/PRM_BOOSTING_POSITIONING.html`

Both are published as private artifacts (see the previous session). The advisor email of 2026-10-22 presented 3
PRMScore tables and the ablation.

## 3. Red flags already visible (verify each one)

All numbers below come from `results/prm_boost_collection_v1/tables/STATUS_MAIN.csv` and `STATUS_CONTRASTS.csv`
(paired bootstrap by source group, 2000 draws), unless stated otherwise.

### 3.1 The gain does not transfer to the external benchmarks

Socratic-PRMBench, 2,995 answers. PRMScore with GAP, CHAIN_N16 against the PRM alone under the same rule:

| PRM | PRM alone | CHAIN_N16 | Δ [95% CI] |
|---|---|---|---|
| Qwen-7B | 70.10 | 66.89 | −3.21 [−3.88, −2.49] |
| Qwen-72B | 71.72 | 57.43 | −14.29 [−15.18, −13.42] |
| PURE | 70.59 | 64.99 | −5.59 [−6.49, −4.72] |
| RLHFlow | 50.38 | 57.06 | +6.67 [+5.79, +7.61] |
| Math-Shepherd | 65.52 | 49.43 | −16.09 [−16.91, −15.31] |

- On Socratic, telemetry alone is also far below the PRM on the four usable PRMs.
- CHAIN's within-answer AUC falls to .43 on Math-Shepherd, below chance. Relevant: the sign-rule inversion on Socratic
  (Step 467).
- **Hard2Verify** (177 answers, wide CIs) is mixed: CHAIN − PRM is −1.55 (Qwen-7B), +7.85 (72B), −10.79 (PURE),
  +27.95 (RLHFlow, whose PRM is near zero there), +1.65 (Math-Shepherd).
- **ProcessBench** (natural model errors) was declared "out of focus" because boosting showed no gain over the plain
  average there (memory note; `results/prm_boost_processbench_v1/`). Verify what it shows. "No gain on natural errors"
  is evidence, not a scoping choice.
- Calibrated on PRMBench and tested on Socratic, the CROP-style certificate breaks its 5% target: 6.7% on Qwen-7B,
  8.2–25.9% on the other PRMs.

**Question:** is the PRMBench gain a property of PRMBench rather than of reasoning errors?

### 3.2 Where the PRMBench gain comes from

Qwen-7B on PRMBench, PRMScore:

| Configuration | PRMScore |
|---|---|
| PRM, native threshold | 65.49 |
| PRM + GAP | 68.01 |
| plain mean of telemetry + GAP | 68.60 |
| telemetry alone + GAP | 69.26 |
| EQ (PRM + telemetry mean) + GAP | 69.32 |
| CHAIN_N16 + GAP | 70.40 |

- CHAIN − PRM under the same rule is +2.39 [1.90, 2.90].
- On the other PRMs: +1.60 (72B), +3.27 (PURE), +6.86 (RLHFlow), +6.18 (Math-Shepherd).
- **Caveats.** The flag rule (GAP, chosen 2026-10-21 after looking at this data) contributes a gain as large as the
  fusion on the strong PRMs. The PRM reference scale is "chosen post hoc" (the note in STATUS_CONTRASTS.csv).
- Gains are largest on the weak PRMs (RLHFlow, Math-Shepherd), which no practitioner would pick. On the strong ones
  they are about 1.6–2.4 points.

### 3.3 Benchmark validity: perturbation artifact (hypothesis, untested)

- PRMBench answers are LLM-perturbed versions of PRM800K solutions. Teacher-forced telemetry measures how surprising
  each step is to the backbone. It may detect "this text was rewritten by another LLM", not "this step is wrong".
- The failure on Socratic and the absence of gain on ProcessBench are consistent with this; neither proves it.
- Step 460 found that PRMBench has a fixed-error-count construction prior. GAP (a per-answer flag count) may exploit
  it.

**Design a cheap test**, for example:
- telemetry on rewritten-but-correct steps, if identifiable;
- telemetry against edit distance or style features;
- a style-only baseline.

### 3.4 Label-free only with a hand-picked anchor

- The thesis principle (CLAUDE.md) is label-free with no hand-picked prior knowledge.
- CHAIN works because the PRM is placed in its own group by hand. The label-free partition (AUTO) loses to CHAIN on
  5/5 PRMs (Step 475). AUTO-F loses more (Step 476).
- Step 476's reading: the PRM is "the identifying anchor". So the method is "trust the PRM, add telemetry around it",
  with a manual design choice at its core.

### 3.5 Supervised fusion beats it

- Plain supervised linear LR beats L-SML by +.008 to +.034 within-answer AUC on 5/5 PRMs (Step 477,
  `results/telemetry_class_correlation_v1/QUADRATIC_HEADROOM.csv`).
- Step 476 found the same (+.007 to +.031).
- Bracha (advisor, 2026-10-24/25) suggested an LR trained on the PRM's own fine-tuning data as the natural arm. If
  label-free is not the selling point, the method has to beat that.

### 3.6 Competitors: ties, not wins

From the positioning page and `results/prm_boost_collection_v1/tables/COMP_*`:
- **PRISM:** 70.40 against 68.0, but at PRISM's TPR the FPR ties (47.70 PRM vs 47.78 L-SML). The edge comes from the
  GAP rule, not the fusion.
- **CROP:** a tie on prefix kept (22.3% vs 21.8%), lower AUROC (.792 vs .834).
- **ARES:** a tie on both Qwen PRMs, ahead only on the weak PRMs.
- **SLIDER:** above it under its own metric, but so is the raw PRM.
- Competitor explainers: `docs/research_notes/competitor_explainers_he_2026-10-24/`.

### 3.7 Evidence hygiene

- Everything is development data. PRMBench has been reused across hundreds of steps and many forking choices: bank,
  normalization, GAP rule, PRM scale, partition.
- There is no untouched confirmation set.
- Step 474 checked CHAIN's statistical hygiene (refit bootstrap, partition stability, split-half) and passed. That
  covers estimation noise, not selection on the benchmark.

### 3.8 What looks genuinely solid (steelman material)

- Telemetry carries a real step-level signal on PRMBench: within-AUC .82 for telemetry alone on Qwen-7B vs .80 for the
  PRM; much higher on the weak PRMs.
- Standalone base-model telemetry beats the PRM's PRMScore on 3/5 PRMs (`results/prm_standalone_telemetry_v1/`).
- The flag-count observation may be a contribution in itself. A per-answer count rule moves PRMScore about as much as
  PRISM's retraining (65.49 → 68.01 for the same PRM). It could support an evaluation critique of PRMScore and
  threshold choice.
- The class-conditional correlation finding (Step 477) is robust, but of small practical value.

## 4. Questions the review must answer

1. What is the strongest claim that is (a) true on the evidence, (b) survives transfer to at least one benchmark not
   used for design, and (c) is not already published (see the competitor explainers and the literature sweep)?
2. Is the Socratic failure fatal, or diagnosable? Distinguish the method (sign/orientation, Step 467; fitting scope)
   from the benchmark (artifact, section 3.3). Name the cheapest test that separates them.
3. Can the current advisor material be presented honestly as is? List every sentence in the two advisor pages and the
   2026-10-22 email that overstates. Quote them.
4. Options, each with its honest pitch, what it needs, and the risk:
   - (A) an analysis/negative paper, e.g. "telemetry boosting works on PRMBench but not on natural errors";
   - (B) a pivot to an evaluation critique (flag rule, threshold sensitivity, construction priors in PRM benchmarks);
   - (C) supervised fusion à la Bracha;
   - (D) close the line and return to another thesis direction.

   Give a recommendation; do not just survey.
5. What should go to the advisors at the next meeting (Tuesday morning was proposed by both)?

## 5. Deliverables

- A frank Hebrew verdict in chat, first, in Omri's style: one sentence, the key number, the next step.
- An English, advisor-ready "honest status" document: what holds, what does not, what we propose. Publish it as an
  artifact only after Omri has read the chat verdict.
- A HISTORY step (`[project review]` tag) and a LESSONS.md entry if the review finds a systematic mistake, e.g.
  scoping out the benchmarks where the method fails.

## 6. Pointers

**Status and ablation:**
- `results/prm_boost_collection_v1/tables/` (STATUS_MAIN, STATUS_CONTRASTS, ABLATION_GAP, COMP_*, CROP_*)
- `results/prm_boost_collection_v1/evaluate_collection.py` (the canonical evaluator; data in `COLLECTION_DATA`,
  default `/Users/osegev/Desktop/hd_collection_data`)

**Math peer review and its stages:**
- `docs/research_notes/math_peer_review_2026-10-24/MATH_PEER_REVIEW_HE.md`
- `docs/research_notes/math_rigor_2026-10-24/` (GAPS_STATUS_HE.md, FRAMING_THEORY.md, HISTORY_AUDIT.md)
- `results/prm_boost_math_rigor_v1/` (NEGATIVE_RESULT*.md)

**Steps 469–477:** HISTORY.md. Steps 469–476 are tagged [PRM math rigor]; 477 is tagged [PRM boosting].

**Other results:**
- Standalone telemetry: `results/prm_standalone_telemetry_v1/README.md`
- SLIDER bridge: `results/slider_cosine_bridge_v1/`
- Class-conditional correlation: `results/telemetry_class_correlation_v1/`

**Literature:**
- `docs/research_notes/literature_sweep_2026-10-21/`
- `docs/research_notes/competitor_primer_2026-10-22/`
- `papers/` (PDFs, some untracked)

**Earlier honest audits worth reading first:**
- `docs/reviews/PRMBENCH_LITERATURE_AND_CLAIM_AUDIT_20260923_HE.md`
- Step 460 (PRMBench construction prior)
- Step 464 [LR-gap closed]

## 7. Repository state at handoff

- **Branch** `claude/prm-boosting`, pushed. HEAD is the commit that adds this file, on top of `55f44696` (Step 477).
- **Merged:** `claude/prm-boost-math-rigor-2026-10-24` (447d717d). HISTORY.md and PROGRESS.md are CRLF; keep them so.
- **Not merged, on purpose:** `claude/dufs-liu-pcr-prmbench-2026-10-24`. Omri: that line failed. It exists only
  locally and is not pushed.
- **Untracked at the repo root, not ours to commit without asking:** `papers/*.pdf`, `results/family_representatives_v1/`,
  `docs/experiments/FAMILY_REPRESENTATIVES_V1_HE.md`, `.agents/`, `.codex/`, `.worktrees/`, and a few others.
- **Other live sessions:** "ביקורת מתמטית PRM boosting" (math rigor; idle, its branch merged).
- **Pending with the advisors:** the meeting slot (Tuesday morning suits both); Ofir asked whether competitors release
  code; Bracha's LR suggestion.
