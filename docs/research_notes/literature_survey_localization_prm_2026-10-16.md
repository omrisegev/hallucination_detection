# Literature survey: hallucination localization + PRM trends (2026-10-16)

Scope: work since roughly late 2025, with emphasis on May–Oct 2026. Excludes what is already
covered in `rag_localization_methods_and_benchmarks_2026.md` and
`reasoning_localization_methods_and_benchmarks_2026.md` (GASP, RAGTruth, ProcessBench, PRMBench,
uPRM, FUSE, Mind the Gap, …) except where a new number or follow-up matters.

**Verification.** Every arXiv ID was opened on its abstract page by the research agents. Most
abstracts give no numbers ("–" below). Items marked *(unverified)* were seen only in search
snippets or citations. Before citing any of these in the thesis, read the paper itself
(`/paper-digest`).

---

## Part A — Hallucination localization

### A1. New methods

| Method | arXiv | Date | Level | Supervision | Signal | Key result |
|---|---|---|---|---|---|---|
| **CORTEX** (Furumai et al.) | 2606.31033 | Jun 2026 | token, then span | unclear (likely light head) | hidden-state contrast **with vs. without retrieved docs**; grounding propagates to the tokens after it | "substantial" token-level gains, 2 RAG benchmarks × 3 LLMs |
| **Temporal Multi-Signal Fusion** (Itkin) | 2608.18115 | Jun 2026 | token/span | supervised BiGRU | 33-dim per-token stream: surprisal + NLI + text stats | RAGTruth token AUC **0.840** (+11pp over per-token LR); RNN, Mamba and attention all plateau near 0.845, so the features are the bottleneck |
| **External Observers** (Gupta & Buscaldi) | 2610.02066 | Oct 2026 | span onset + continuation | supervised probes | another model reads the generator's hidden states | the external observer ≥ self-probing, even when it is smaller |
| **TokenHD** (Min et al.) | 2605.12384 | May 2026 | token, free-form | supervised (synthetic data engine) | trained detector | a 0.6B detector beats QwQ-32B |
| **LettuceDetect-Qwen-2B** (Kovács et al.) | 2607.00895 | Jul 2026 | character span | supervised (injected spans) | token classifier | RAGTruth 81.8 example-F1; unified span-F1 0.689 |
| **Input-Side Evidence Alignment** (Yamada & Arase) | 2608.15804 | Aug 2026 | span + alignment | self-supervised masked prediction | an output token that is hard to predict **from the input** is ungrounded | qualitative + human eval |
| **TeQHallu** (Vukovic et al.) | 2609.05025 | Sep 2026 | claim | unsupervised (neurosymbolic) | builds a SQL DB from the reference and reasons over it | competitive on RAGTruth and DiaHalu |
| **HallDetect** (Oukelmoun et al.) | 2608.05823 | Aug 2026 | claim → source span | black-box NLI | contrastive entailment over multi-scale chunks; one strongly contradicted claim flags the response | best on 3/4 benchmarks |
| **TPA** (Lu et al.) | 2512.07515, ACL 2026 | Dec 2025 | token, aggregated by POS | unclear | splits each token's probability into 7 sources (query, context, past, self, FFN, LayerNorm, embedding) | SOTA claim |
| **FRANQ** (Fadeeva et al.) | 2505.21072 | v5 Apr 2026 | claim | UQ is unsupervised | claim UQ conditioned on faithfulness to the context | beats unsupervised UQ and supervised classifiers on long-form RAG |
| **IUQ** (Fan et al.) | 2604.15109, ACL 2026 | Apr 2026 | claim | sampling, training-free | claim → question → re-ask the model; consistency across samples | – |
| **SHARS** (Li, …, Gal) | 2606.03628, ICML 2026 | Jun 2026 | segment, online | training-free | segment-level semantic uncertainty, then reject and resample | – |
| **Transport Geometry** (Alvarez & Baheri) | 2605.13772 | May 2026 | step | label-conditioned teacher → BiLSTM student | contrastive PCA + transport costs on hidden-state trajectories | degrades under shift |
| **Sanity Checks / TRACT** (Zollicoffer et al.) | 2605.08346 | May 2026 | trace (evaluation method) | lightweight | Force/Remove perturbations of the final answer | **many trace detectors exploit final-answer artefacts** |
| HDM-4B-RL (Liu et al.) | 2601.09734, AAAI | Dec 2025 | span + explanation | supervised + GRPO | – | HaluEval SOTA |
| TOHA (Bazarova et al.) | 2504.10063, ACL 2026 | 2025/26 | response | minimal labels | attention-graph topology | not a span localizer |

### A2. New benchmarks
- **Unified span benchmark** (2607.00895): injected character spans in code, tool output,
  structured docs and RAG. Zero-shot judges reach ≤0.22 span-F1 on code.
- **HalluPeer** (2609.03580, EMNLP Findings 2026): peer reviews, with detection, classification
  and localization tasks.
- **FullCite** (2606.07130): snippet-level citation. Snippet-F1 is far below Doc-F1, so span
  localization is the bottleneck.
- **PsiloQA** (2510.04849): multilingual span benchmark, now used as a standard test set.

**RAGTruth reference points:** supervised BiGRU over token signals reaches 0.84 token AUC; the
best training-free result is still GASP at about 0.67 span AUC. **No published training-free
token-level result beats GASP**, so this space is open.

### A3. Trends
1. **With-vs-without-context contrast is the main RAG primitive.** It appears in hidden states
   (CORTEX), likelihood (GASP), predictability from the input (2608.15804) and probability
   attribution (TPA). This supports our Evidence-Contrast direction.
2. **Token → span through temporal structure.** Smoothing and propagation of support, and
   BiGRU/onset modelling. Neighbouring context matters more than the per-token classifier.
3. **Supervised detectors trained on synthetic or injected labels lead span-F1**; training-free
   methods lag clearly.
4. **Unsupervised step localization is viable** (uPRM).
5. **More scrutiny of evaluation shortcuts** (Sanity Checks).
6. **Claim-level UQ conditioned on faithfulness** (FRANQ, IUQ).

---

## Part B — Process Reward Models: trends

### B1. Generative "thinking" PRMs trained with outcome-only RL (strongest trend)
| Paper | arXiv | Date | Mechanism | Number |
|---|---|---|---|---|
| **TIPS** (Fan, …, Yankai Lin) | 2609.36641 | Sep 2026 | generative PRM (CoT → step labels → outcome); GRPO reward is only whether the outcome is correct | **ProcessBench 85.2** (Qwen3-4B-Thinking, 3.2K trajectories); highest open trained-PRM number found |
| StepWiser (Meta FAIR) | 2508.19229 | Aug 2025 | stepwise generative judge, RL on relative rollout outcomes | about 61.9 *(snippet)* |
| ThinkPRM | 2504.16828 | Apr 2025 | long-CoT verifier, about 1K synthetic chains | beats LLM-judge |
| GenPRM | 2504.00891, AAAI-26 | Apr 2025 | CoT + code verification + test-time scaling of the PRM itself | 7B beats Qwen-PRM-72B |

### B2. PRMs without ground-truth labels
| Paper | arXiv | Date | Mechanism | Number |
|---|---|---|---|---|
| uPRM (anchor) | 2605.10158 | May 2026 | cross-batch scoring of first-error position from P(+/−); LoRA trained without labels | ProcessBench F1 **58.3 / 52.6 / 42.7 / 39.8** (GSM8K/MATH/Olympiad/Omni), avg about 48; no PRMBench |
| **ScalePRM** (Rahman, …, Peng) | 2512.03244 | v2 Aug 2026 | many independent verifications per step → aggregated pseudo-labels → generative PRM | ProcessBench **67.5**, above its version with ground truth (66.4) |
| FreePRM | 2506.03570 | Jun 2025 | pseudo step labels | 53.0 |
| Noise-aware PRM (Xie, …) | 2601.12748 | Jan 2026 | MC labels treated as noise; LLM relabeling + self-confidence refinement | up to +27 F1 |

### B3. Gray-box verifiers (hidden states, entropy)
| Paper | arXiv | Date | Mechanism | Number |
|---|---|---|---|---|
| **Hidden Error Awareness** (Yuan, …, Yue Zhao) | 2605.09502 | May 2026 | linear probe on hidden states | AUROC 0.95, **0.79 from the first step**, text-only 0.59; using the signal to intervene fails |
| **HSRM** (Li & Zhu) | 2608.30841, EMNLP 2026 | Aug 2026 | 2M-parameter reader of frozen hidden states at step boundaries, outcome labels only | ≥ text verifier in 15/16 settings |
| PAIR | 2605.17877, NeurIPS 2026 | May 2026 | belief-consistency probe as process reward for agents | – |
| **REWIND** (Ding et al.) | 2610.19561 | Oct 2026 | **training-free**: instability before entropy peaks → cut and regenerate | wrong traces have about 112% higher mean entropy; **no localization evaluation** |
| *(unverified)* ReProbe, VerifySteer 2605.20745, ReSET 2606.13233 | | | per-step token features; strictness encoded at the delimiter token; token entropy dominated by step-level uncertainty | |

### B4. Process rewards in RL, and their hacking
- **GRPO is Secretly a PRM** (2509.21154, ICML 2026): outcome-reward GRPO with shared prefixes
  equals an MC-PRM objective; proposes λ-GRPO.
- **Reward Under Attack** (2603.06621): policies reach PRM reward >0.9 with AIME accuracy <4%;
  about 43% of the gain comes from style. PRMs act as "fluency detectors".
- **PRPO** (2601.07182): PRM scores → token advantages aligned to the outcome advantage.

### B5. Agents, tools, computer use
**AgentProcessBench** (Fan, …, Yankai Lin; 2603.14465, Mar 2026; data
`hf.co/datasets/LulaCola/AgentProcessBench`): 1,000 trajectories × 8,509 human step labels
{+1, 0, −1} over HotpotQA, GAIA, BFCL and τ²-Bench, from Qwen3-4B/30B, DeepSeek-V3.2 and
GPT-5(-mini). A step is one assistant message. Error propagation: dependent steps after an error
are labeled −1. Metrics are StepAcc and **FirstErrAcc**. Best: Gemini-3-Flash-Thinking
81.6/65.8; best open: Qwen3-30B-A3B-Thinking 68.5/52.0. Only LLM judges are evaluated, with no
trained PRMs and no label-free methods. ToolPRMBench (2601.12294), PRO-CUA (2605.29119), AgentPRM promise+progress (2511.08325),
Web-Shepherd (2505.15277).

### B6. Benchmarks beyond math
LSR-Ben (2605.01203; science and logic, where PRMs are much weaker), MedPRMBench (2604.17282;
113K clinical step labels), PRMs Meet Planning (2604.17957; about 1M PDDL steps with exact labels).

### B7. Segmentation without "Step k" markers
Splitting at semantic or entropy points (PRPO, SAPO, AdaptiveStep — mostly unverified). No
consensus yet.

### B8. Unsupervised ensembling of verifiers
FUSE (2604.18547, Candès) and Weaver (2506.18203) work **at the response level only**.

### SOTA summary
- **ProcessBench:** TIPS 85.2 > Qwen2.5-Math-PRM-72B about 78 > ActPRM 75.0 > ScalePRM 67.5
  (label-free with an LLM judge) > uPRM about 48 (label-free, logprob). Baselines vary between
  papers; compare only within a table.
- **PRMBench:** no 2026 update found; the original leaderboard (Gemini-2-Thinking 68.8) still
  stands. **New PRMs barely report on it.**

### Open problems
Hackability and fluency bias; whether a separate PRM is needed at all for RL; domain transfer;
MC label noise; ad-hoc segmentation; diagnostic ≠ causal; inconsistent metrics.

---

## Part C — Implications for the current line (revised 2026-10-17)

**Revision note.** The first version of this section was written against a stale branch
(Step 290: Unified-28 / U-PCR, ProcessBench lane). It has been rewritten against the current line
(`codex/lsml-cluster-contrast-analysis-v1`, Steps ≤451, PROGRESS 2026-10-15):

- Since the 2026-10-08 advisor meeting (Step 451) the focus is **PRMBench only**.
- The working fusion is label-free CONT L-SML with the DS filter.
- The main new result is **PRM boosting**: Qwen2.5-Math-PRM-7B combined with three telemetry
  channels by label-free L-SML raises PRMBench PRMScore from 65.49 to 69.48 [+3.31, +4.67], and
  the gain holds across five PRMs.

1. **PRM boosting fits the hottest gap in the literature.** Label-free verifier ensembling is
   published only at the response level:
   - FUSE (Candès group, already used in this line) and Weaver aggregate whole responses.
   - ScalePRM aggregates step verdicts by majority vote.
   - uPRM is a single view.

   Step-level label-free fusion of a trained PRM with gray-box telemetry is not in the
   literature. Position the boosting result against FUSE, Weaver and ScalePRM explicitly.
2. **Natural next voters are generative PRMs.** Add a "thinking" PRM's step verdict as one more
   L-SML input: ThinkPRM, GenPRM (open checkpoints), or TIPS once released. This means P(−) on the
   step token, or its generated label. The current ProcessBench SOTA is TIPS, 85.2.
   `claude/prm-voters-processbench-pilot` (2026-10-16) already started a PRM-voters pilot; check
   it before building anything.
3. **PRMBench is barely reported in 2026, and our numbers sit near its leaderboard.** The
   original leaderboard has Gemini-2-Thinking 68.8, Qwen2.5-Math-PRM-7B about 65.5, and humans
   83.8; the boosted row is at 69.48. Before any claim, check comparability: the same official
   PRMScore, the same subset, and the Step-313 label-shift correction.
4. **The robustness angle matches the boosting gains.** Reward Under Attack (2603.06621) finds text
   PRMs act as fluency detectors. The boosting gains are largest on redundancy (49.0 → 67.7) and
   circular reasoning (54.9 → 71.8), which are the PRMBench Simplicity categories where fluency
   bias should hurt most. This is a testable story: gray-box telemetry corrects a text-PRM bias.
5. **Supervised gray-box ceilings to cite next to the LR-gap analysis:** Hidden Error Awareness
   (0.79 AUROC from the first step) and HSRM (outcome-label-only hidden-state reader).
6. **ProcessBench (de-prioritized).** The REWIND cut point was checked as a locator (Step 463).
   It confirms Step 331: the gain is the mean-entropy detector, and the cut point is no better
   than the raw peak. No further ProcessBench work is proposed here. For any Mind the Gap
   comparison, keep the Step-333 caveat: the public code does not implement the paper's step
   aggregation.
7. **Agents (scope decision for Omri).** AgentProcessBench (2603.14465) has human step labels
   and a first-error index. Only LLM judges have been evaluated on it, the best open one at 52.0
   FirstErrAcc. It is the natural external benchmark for the agent-telemetry line
   (`codex/agent-step-telemetry-v1`, Step 452), provided full trajectories are in the release.
8. **RAG (application priority 2).** CORTEX's with/without-evidence contrast plus temporal
   propagation is the closest published analogue to Evidence-Contrast. The training-free
   token-level space on RAGTruth is still open; GASP is at about 0.67 span AUC.
9. **Sanity check before any step-level claim:** apply the Force/Remove perturbations of
   2605.08346 to our telemetry features.
