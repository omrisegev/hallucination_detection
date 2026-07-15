"""
Replication-grid presets — the single source of truth for the thesis grid cells.

Each preset bundles a competitor paper's exact (model, dataset, protocol) so a cell is
submitted by `preset_id` and every paper-matched parameter comes from here, not from
hand-passed flags. This encodes the Step-155 protocol cards plus the guardrails the
EDIS/AIME24 test run forced (see the replication-grid plan):

  U1  per-preset `max_new` (reasoning 2048, short-answer QA 256–512) — never a blanket
      1024, which truncates the entropy trace our spectral features live on.
  U3  per-preset `n_samples` >= a few hundred for stable AUROC CIs (the demo's 30 was
      a smoke-test size).
  U4  `temps` = the paper's own protocol temperature (apples-to-apples). Offline scoring
      of our L-SML must use `anchor_orient` because the unsupervised anchor is fragile at
      low T (Phase-15). Never mix temperatures in one cell (diversity hurts, −5.3pp).
  U5  QA cells are wired here to real loaders + paper-terse prompts (registered in
      run_inference.py DATASETS), NOT chain-of-thought.

This module is pure data — no torch / transformers / datasets imports — so both the
cluster driver and the local offline scoring scripts can import it cheaply.

`dataset` is a KEY into run_inference.DATASETS (resolved there, to avoid a circular
import). `capture` flags map to generate_full(capture_*=...).
"""

# Accuracy-band gate defaults (U2). A cell's AUROC is only trustworthy when both classes
# are populated: overall accuracy inside the band AND the minority class large enough for
# a stable bootstrap CI. AIME24 x Qwen-1.5B floored at ~2% acc — that config is
# verification-only, never a data cell.
DEFAULT_ACC_BAND = (0.20, 0.85)
DEFAULT_MIN_MINORITY = 30


def _preset(**kw):
    """Fill preset defaults so every cell has the full guardrail set."""
    kw.setdefault("gated", False)
    kw.setdefault("split", "validation")
    kw.setdefault("k", 1)
    kw.setdefault("temps", [1.0])
    kw.setdefault("gen_top_p", None)
    kw.setdefault("gen_top_k", 50)
    kw.setdefault("repetition_penalty", None)
    kw.setdefault("no_repeat_ngram_size", None)
    kw.setdefault("logprob_top_k", 50)
    kw.setdefault("capture", {})
    kw.setdefault("acc_band", DEFAULT_ACC_BAND)
    kw.setdefault("min_minority", DEFAULT_MIN_MINORITY)
    kw.setdefault("published", {})
    kw.setdefault("judge", None)          # LLM-judge HF id for correctness labels; None -> dataset's lexical grader
    kw.setdefault("head_to_head", None)   # "SAME-MODEL" when our model matches the paper's exactly
    kw.setdefault("prompt_suffix", "")    # appended to every user message (e.g. Qwen3 "/no_think")
    kw.setdefault("raw_prompt", False)    # skip chat template (base LMs like OPT-30B + few-shot)
    kw.setdefault("notes", "")
    return kw


# ── The four high-impact cells (Step 155) ─────────────────────────────────────
PRESETS = {

    # 1. LOS-Net head-to-head. LOS-Net scores a single generation's top-K logprob
    #    statistics (TDS) with a supervised Transformer probe. We generate one greedy
    #    candidate "as they did", capture top-1000 logprobs, and score our L-SML on the
    #    same trace. G3 gate: beat LOS-Net 72.92 ± 0.45.
    "losnet_hotpotqa_mistral7b": _preset(
        paper="LOS-Net (arXiv 2503.14043)",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        dataset="hotpotqa", split="validation", n_samples=500,
        k=1, temps=[0.0],                 # single greedy generation, as LOS-Net does
        max_new=512, logprob_top_k=1000,  # TDS wants the wide top-K
        published={"method": "LOS-Net", "metric": "AUROC", "value": 72.92, "pm": 0.45},
        notes="Greedy single-pass. LapEigvals-style attention not needed here. "
              "Our L-SML anchor is greedy (argmax) — anchor_orient offline.",
    ),

    # 2. LapEigvals cell. LapEigvals is a SUPERVISED attention-Laplacian probe; its
    #    on-GPU capture is not implemented yet, so this cell runs our L-SML now on a
    #    long CoT GSM8K trace (our validated regime) and defers the LapEigvals number.
    "lapeigvals_gsm8k_llama8b": _preset(
        paper="LapEigvals (arXiv 2502.17598)",
        model="meta-llama/Llama-3.1-8B-Instruct", gated=True,
        dataset="gsm8k", split="test", n_samples=500,
        k=1, temps=[1.0], max_new=2048,   # U1: reasoning trace, do not truncate
        capture={"attention": True, "attention_top_k": 100},  # LapEigvals reducer (Step-182)
        published={"method": "LapEigvals", "metric": "AUROC",
                   # SAME-MODEL anchors (GSM8K / Llama-3.1-8B, HISTORY Steps 66-69):
                   "value": 72.0,   # unsupervised AttentionScore (white-box, no labels) — our head-to-head Y
                   "supervised": 87.2,   # supervised attention-Laplacian probe (80% labeled train split)
                   "cross_model": 92.5,  # supervised probe on Mistral-Small-24B (cross-model, NOT comparable)
                   "model_note": "GSM8K/Llama-3.1-8B: unsup AttentionScore 72.0, sup probe 87.2. "
                                 "92.5 is the paper's cross-model Mistral-Small-24B supervised number."},
        notes="capture_attention NOW LIVE (Step 182 reducer, scripts/test_attn_laplacian.py "
              "PASS): one extra teacher-forced forward pass -> attn_lap_eigvals [L,H,100] + "
              "attn_diag_logmean [L,H]. Offline: scripts/score_lapeigvals.py builds the paper's "
              "OWN supervised Y (top-k eigvals -> PCA-512 -> balanced LR 5-fold CV) AND an "
              "unsupervised AttentionScore from attn_diag_logmean, so the cell finally carries a "
              "self-reproduced LapEigvals number, not just the cited 72.0/87.2. THIS IS A RE-RUN: "
              "the current fetched cache has no attention -> archive it (mv *_noattn) before "
              "re-fetch. gsm8k_prompt is the boxed CoT prompt, not LapEigvals Listing-5 — note "
              "when citing. Same-model unsup AttentionScore=72.0 stays the fair head-to-head Y "
              "for our unsupervised L-SML. MEMORY: attention capture on 8B at T<=2048 peaks ~20GB "
              "(eager maps, all layers) — B200-sized, verify no OOM at the N=30 pilot before full. "
              "The other lapeigvals_gsm8k_* cells stay capture-free until this one validates on GPU "
              "(24B would need far more; flip per-cell after checking peak mem).",
    ),

    # 3. Energy baselines (EPR / Semantic Energy / Spilled Energy) + LapEigvals on a
    #    single-fact QA set. Single-fact traces stay short — this is a BOUNDARY cell
    #    (expected weak for our method), kept to map where the spectral signal vanishes.
    "spilled_triviaqa_llama8b": _preset(
        paper="Spilled/Semantic Energy + LapEigvals",
        model="meta-llama/Llama-3.1-8B-Instruct", gated=True,
        dataset="trivia_qa", split="validation", n_samples=500,
        k=1, temps=[1.0], max_new=256,    # short factual answer
        capture={"logsumexp": True},      # energy papers need Z_n
        published={"method": "HCPD (arXiv 2606.12900)", "metric": "AUROC",
                   "value": 86.25,
                   "supervision": "unsupervised (zero-source, agentic multi-criteria probing)",
                   "model_note": "TriviaQA / Llama-3.1-8B-Instruct: HCPD 86.25 (Table 2). "
                                 "Same-paper baselines: Perplexity 80.62, Semantic Entropy 78.71, "
                                 "SAPLMA(sup) 78.51, TSV(sup) 79.78 -- full table in "
                                 "published_baselines.csv. HARP's cross-model TriviaQA anchor "
                                 "92.8 (model unspecified) tracked separately, same file."},
        head_to_head="SAME-MODEL",
        notes="Boundary cell: single-fact QA -> short trace -> spectral has little signal.",
    ),

    # 4. INSIDE/EigenScore + SE-ICLR'23 on CoQA. INSIDE: K=10, T=0.5, top-p=0.99,
    #    top-k=5, int(L/2) last-token hidden state; ROUGE-L>0.5. We capture the hidden
    #    state for INSIDE and score our L-SML on the K generations.
    "inside_coqa_llama7b": _preset(
        paper="INSIDE/EigenScore (2402.03744) + SE-ICLR'23 (2302.09664)",
        model="huggyllama/llama-7b",       # LLaMA-7B base mirror (INSIDE model)
        dataset="coqa", split="validation", n_samples=500,
        k=10, temps=[0.5], gen_top_p=0.99, gen_top_k=5, max_new=256,
        capture={"hidden": True},          # INSIDE sentence embedding
        published={"method": "INSIDE", "metric": "AUROC", "value": 80.4},
        notes="INSIDE K=10 sampling; SE-ICLR ROUGE-L>0.3 grader is is_correct_coqa.",
    ),

    # ── QA extension (Item 3) — SE-ICLR-style terse protocol, ready to run ────────
    "se_squad_v2_llama8b": _preset(
        paper="SE-ICLR'23 protocol (adapted)",
        model="meta-llama/Llama-3.1-8B-Instruct", gated=True,
        dataset="squad_v2", split="validation", n_samples=1000,
        k=10, temps=[0.5], max_new=256, capture={"logsumexp": True},
        notes="SQuAD v2 includes unanswerable — grader rewards correct abstention.",
    ),
    "se_nq_open_llama8b": _preset(
        paper="SE-ICLR'23 protocol (adapted)",
        model="meta-llama/Llama-3.1-8B-Instruct", gated=True,
        dataset="nq_open", split="validation", n_samples=1000,
        k=10, temps=[0.5], max_new=128, capture={"logsumexp": True},
        published={"method": "HCPD (arXiv 2606.12900)", "metric": "AUROC",
                   "value": 90.38,
                   "supervision": "unsupervised (zero-source, agentic multi-criteria probing)",
                   "model_note": "NQ-Open / Llama-3.1-8B-Instruct: HCPD 90.38 (Table 2). "
                                 "Same-paper baselines: Perplexity 57.92, Semantic Entropy 61.04, "
                                 "SAPLMA(sup) 76.23, TSV(sup) 70.17 -- full table in "
                                 "published_baselines.csv."},
        head_to_head="SAME-MODEL",
        notes="Boundary cell: open-domain single-fact -> short trace.",
    ),
    "truthfulqa_llama8b": _preset(
        paper="TruthfulQA (generation)",
        model="meta-llama/Llama-3.1-8B-Instruct", gated=True,
        dataset="truthfulqa", split="validation", n_samples=817,
        k=10, temps=[0.5], max_new=128, capture={"logsumexp": True},
        published={"method": "TSV (arXiv 2503.01917)", "metric": "AUROC",
                   "value": 84.2,                    # SAME-MODEL Llama-3.1-8B, but SEMI-SUPERVISED
                   "supervision": "semi-supervised (32 labeled exemplars + OT pseudo-labels)",
                   "supervised": 85.5,               # TSV fully-supervised upper bound
                   "haloscope": 78.64,               # HaloScope, LLaMA-2-7b-chat — DIFFERENT model
                   "model_note": "TruthfulQA / Llama-3.1-8B: TSV 84.2±0.2 (semi-sup), sup ceiling 85.5. "
                                 "HaloScope 78.64 is Llama-2-7b-chat (cross-model, annotate)."},
        head_to_head="SAME-MODEL",
        notes="At-inference label is a ROUGE-L proxy; judge-regrade offline before citing. TSV anchor is "
              "semi-supervised (not a fair unsup Y — report as ceiling-ish reference).",
    ),
    "sciq_llama8b": _preset(
        paper="SciQ (4-way MCQ)",
        model="meta-llama/Llama-3.1-8B-Instruct", gated=True,
        dataset="sciq", split="validation", n_samples=1000,
        k=1, temps=[1.0], max_new=512,
        published={"method": "HCPD (arXiv 2606.12900)", "metric": "AUROC",
                   "value": 86.04,
                   "supervision": "unsupervised (zero-source, agentic multi-criteria probing)",
                   "model_note": "SciQ / Llama-3.1-8B-Instruct: HCPD 86.04 (Table 2). "
                                 "Same-paper baselines: Perplexity 66.12, Semantic Entropy 77.81, "
                                 "SAPLMA(sup) 85.63, TSV(sup) 80.01 -- full table in "
                                 "published_baselines.csv."},
        head_to_head="SAME-MODEL",
        notes="MCQ -> suppressed entropy dynamics (GPQA-like); expect a modest cell.",
    ),

    # HCPD CoQA on the CORRECT model. inside_coqa_llama7b runs CoQA on huggyllama/llama-7b
    # BASE — a model mismatch vs HCPD/Automatic-Layer-Selection, which both use the 3.1-8B
    # INSTRUCT checkpoint. This closes HCPD's 4-dataset same-model grid (TriviaQA/SciQ/
    # NQ-Open already scored above) and gives Automatic Layer Selection a same-model CoQA
    # anchor too (see papers/digests/automatic-layer-selection-for-hallucination-detection.md).
    # Decoding: HCPD's extraction (line ~920) states responses are generated with plain
    # greedy decoding; a separate "greedy decoding with 5 beam search" mention (line ~2214)
    # is a DIFFERENT section (RL training-data construction, citing Kuhn et al. 2023's
    # protocol for a canonical reference answer) — not the Table-2 detection-eval protocol,
    # verified 2026-07-13 before writing this preset.
    "hcpd_coqa_llama8b": _preset(
        paper="HCPD (arXiv 2606.12900) + Automatic Layer Selection (arXiv 2605.26366)",
        model="meta-llama/Llama-3.1-8B-Instruct", gated=True,
        dataset="coqa", split="validation", n_samples=500,   # matches inside_coqa for comparability
        k=1, temps=[0.0], max_new=256,        # HCPD: single response per query, greedy decoding
        capture={"logsumexp": True},          # energy views (Step 163 whatis C)
        judge="Qwen/Qwen2.5-7B-Instruct",     # HCPD labels via LLM-judge/BLEURT>0.5 — deviation,
                                               # same uniform-judge pattern as EPR/SemEnergy cells.
                                               # is_correct_coqa (ROUGE-L>0.3) still runs at-inference
                                               # as the lexical label; judge label is the citable one.
        head_to_head="SAME-MODEL",
        published={"method": "HCPD (arXiv 2606.12900)", "metric": "AUROC",
                   "value": 90.07,
                   "supervision": "unsupervised (zero-source, agentic multi-criteria probing)",
                   "als_fepoid_ceiling": 67.05,   # Automatic Layer Selection's own method — SUPERVISED
                                                   # MLP-probe ceiling (9k-example trained probe), not
                                                   # a fair unsup Y; see published_baselines.csv.
                   "model_note": "CoQA / Llama-3.1-8B-Instruct: HCPD 90.07 (Table 2). Same-paper "
                                 "baselines: Perplexity 81.41, Semantic Entropy 75.26, SAPLMA(sup) "
                                 "71.58, TSV(sup) 69.31. Automatic Layer Selection's own FEPoID reaches "
                                 "67.05 but is a supervised probe ceiling, not comparable head-to-head; "
                                 "their unsupervised baselines (Pred.Entropy 58.33, Semantic Entropy "
                                 "50.03, Lexical Similarity 67.80) are the fair comparison set — full "
                                 "table in published_baselines.csv once this cell is scored."},
        notes="Replaces inside_coqa_llama7b as the SAME-MODEL CoQA cell (that cell stays for the "
              "INSIDE/llama-7b-base comparison it was built for). is_correct_coqa ROUGE-L>0.3 grader "
              "unchanged from inside_coqa_llama7b.",
    ),

    # ── Exact-scenario re-runs (Step 163) — same model+dataset+protocol+grading as the ──
    #    paper, so OUR L-SML/U-PCR AUROC (X) sits next to the paper's PUBLISHED AUROC (Y)
    #    and the only difference is the method. Cluster does inference + judge-LABELING
    #    only; competitor detectors are NOT reproduced (their Y comes from the paper).

    # EPR (Entropy Production Rate). Exact model+dataset: Mistral-Small-3.1-24B on TriviaQA
    # (Wikipedia domain), non-greedy T=1.0, single generation. Correctness graded by an
    # LLM judge (paper used Gemma-3-12b-it). Published EPR baseline AUROC = 74.6.
    "epr_triviaqa_mistral24b": _preset(
        paper="EPR — Entropy Production Rate",
        model="mistralai/Mistral-Small-3.1-24B-Instruct-2503", gated=True,
        dataset="trivia_qa_wiki", split="validation", n_samples=1000,
        k=1, temps=[1.0], max_new=64,
        capture={"logsumexp": True}, logprob_top_k=50,
        judge="Qwen/Qwen2.5-7B-Instruct",
        head_to_head="SAME-MODEL",
        published={"method": "EPR", "metric": "AUROC", "value": 74.6,
                   "model_note": "Mistral-Small-3.1-24B / TriviaQA (unsupervised EPR baseline; "
                                 "WEPR supervised is higher)"},
        notes="Exact EPR scenario (model/dataset/decoding). JUDGE DEVIATION: paper used Gemma-3-12b-it "
              "(access pending Google review), so we use the open, reliable Qwen2.5-7B-Instruct as a "
              "uniform LLM-judge for correctness labels. N=1000. capture logsumexp -> our energy views.",
    ),

    # Semantic Energy. Exact model+dataset: Qwen3-8B on TriviaQA, sampling T=0.6, K=10.
    # Correctness graded by an LLM judge (paper used TIGER-Lab/general-verifier). Published:
    # Semantic Entropy baseline 69.6, Semantic Energy 74.8. Qwen3 thinking mode is disabled
    # via the "/no_think" prompt suffix (short factual answers, matches the main-table setup).
    "semenergy_triviaqa_qwen3_8b": _preset(
        paper="Semantic Energy",
        model="Qwen/Qwen3-8B",
        dataset="trivia_qa", split="validation", n_samples=500,
        k=10, temps=[0.6], max_new=64,
        capture={"logsumexp": True}, logprob_top_k=50,
        judge="Qwen/Qwen2.5-7B-Instruct",
        head_to_head="SAME-MODEL",
        prompt_suffix=" /no_think",
        published={"method": "Semantic Energy", "metric": "AUROC", "value": 74.8,
                   "baseline_semantic_entropy": 69.6,
                   "model_note": "Qwen3-8B / TriviaQA (English; CSQA Chinese skipped)"},
        notes="Exact Semantic Energy scenario (model/dataset/decoding). JUDGE DEVIATION: paper used "
              "TIGER-Lab/general-verifier (bespoke format incompatible with a clean correctness "
              "prompt), so we use Qwen2.5-7B-Instruct as a uniform LLM-judge. K=10; Qwen3 /no_think.",
    ),

    # SE-ICLR'23 (Semantic Uncertainty, Kuhn/Gal/Farquhar). Exact scenario: OPT-30B (base)
    # on closed-book TriviaQA, K=10, T=0.5, correctness = ROUGE-L>0.3 (NO LLM judge — the
    # paper's own grader). Few-shot prompt for the base model. Published Semantic Entropy
    # AUROC = 0.83 (TriviaQA/OPT-30B); CoQA 0.77.
    "seiclr_triviaqa_opt30b": _preset(
        paper="Semantic Uncertainty (SE-ICLR'23, arXiv 2302.09664)",
        model="facebook/opt-30b",
        dataset="trivia_qa_rougel", split="validation", n_samples=500,
        k=10, temps=[0.5], max_new=64,
        capture={"logsumexp": True}, logprob_top_k=50,
        raw_prompt=True,   # OPT-30B base model -> few-shot prompt goes in raw, no chat template
        head_to_head="SAME-MODEL",
        published={"method": "Semantic Entropy (SE-ICLR'23)", "metric": "AUROC", "value": 83.0,
                   "coqa_value": 77.0,
                   "model_note": "OPT-30B / TriviaQA closed-book, ROUGE-L>0.3 grader"},
        notes="Exact SE-ICLR scenario. Grader=ROUGE-L>0.3 (is_correct_trivia_qa_rougel), no judge. "
              "OPT-30B base -> few-shot prompt (trivia_qa_fewshot_prompt). Paper used an 8k subset; "
              "N=500 is AUROC-stable. Pilot check: base model may ramble -> watch accuracy.",
    ),

    # ── ARS-matched reasoning cells (arXiv 2601.17467, "Answer-agreement Representation
    #    Shaping"). ARS is a SUPERVISED trace-embedding-shaping detector. We run the exact
    #    (model, dataset) so our UNSUPERVISED L-SML sits next to ARS's supervised Y.
    #    MATH-500/R1-Distill already exists offline (subset_sweep npz, GOOD_5=84.4 vs ARS 86.38);
    #    this preset documents that cell + adds GSM8K/R1-Distill (the missing matched point).
    "ars_math500_r1distill8b": _preset(
        paper="ARS (arXiv 2601.17467)",
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        dataset="math500", split=None, n_samples=300,
        k=1, temps=[1.0], max_new=4096,   # U1: R1-Distill emits very long CoT; do not truncate
        head_to_head="SAME-MODEL",
        published={"method": "ARS (CCS)", "metric": "AUROC", "value": 86.38,
                   "supervision": "supervised (representation shaping)",
                   "best_baseline": "G-Detector 64.45",
                   # ARS Table 2 unsupervised baselines on this exact cell — our GOOD_5 84.4
                   # beats every one of them:
                   "eigenscore_unsup": 75.89, "semantic_entropy_unsup": 43.60,
                   "perplexity_unsup": 40.96,
                   "model_note": "MATH-500 / DeepSeek-R1-Distill-Llama-8B (ARS Table 1). "
                                 "Our unsupervised L-SML GOOD_5=84.4 on the same cell (subset_sweep). "
                                 "ARS Table 2 unsup: EigenScore 75.89, SE 43.60, Perplexity 40.96."},
        notes="ALREADY HAVE offline data (math500__DeepSeek-R1-Distill-Llama-8B_T1.0.npz, N=300, "
              "GOOD_5=0.844, best-subset=0.861). Preset documents provenance + allows a raw-trace "
              "re-run for EDIS/energy features. is_correct_math extracts \\boxed{} after </think>.",
    ),
    "ars_gsm8k_r1distill8b": _preset(
        paper="ARS (arXiv 2601.17467)",
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        dataset="gsm8k", split="test", n_samples=500,
        # Greedy per ARS §5.1 ("By default, greedy decoding is used"); mn8192 because
        # R1-Distill thinking traces exceed 4096 like Qwen3's did (13-45% cap-pinning
        # on jobs 101075/101076 = truncation-label leakage). Never ran at T=1.0.
        k=1, temps=[0.0], max_new=8192,
        head_to_head="SAME-MODEL",
        published={"method": "ARS (CCS)", "metric": "AUROC", "value": 74.72,
                   "supervision": "supervised (representation shaping)",
                   "best_baseline": "G-Detector 70.38",
                   # ARS Table 2 unsupervised baselines on this exact cell:
                   "eigenscore_unsup": 52.98, "semantic_entropy_unsup": 61.98,
                   "perplexity_unsup": 58.48,
                   "model_note": "GSM8K / DeepSeek-R1-Distill-Llama-8B (ARS Table 1). "
                                 "ARS Table 2 unsup: EigenScore 52.98, SE 61.98, Perplexity 58.48."},
        notes="MISSING matched cell — the one ARS point we do not yet have offline. Run this to "
              "place our unsupervised L-SML next to ARS supervised 74.72 on GSM8K/R1-Distill. "
              "Gate: smoke -> N=30 pilot (watch trace not pinned at max_new, acc in [0.20,0.85]) -> full.",
    ),

    # ── LapEigvals GSM8K model-sweep (arXiv 2502.17598, EMNLP'25). LapEigvals published
    #    AUROC on GSM8K (test split, N=1319, exact-match) for FIVE models, each with an
    #    UNSUPERVISED "AttentionScore" baseline and the SUPERVISED LapEigvals probe. We have
    #    Llama-3.1-8B (lapeigvals_gsm8k_llama8b, our L-SML 0.815 vs AttentionScore 0.720). These
    #    four fill the rest of the sweep: our unsupervised L-SML vs their unsupervised
    #    AttentionScore (fair Y), with the supervised probe as the ceiling. Inference-only —
    #    L-SML needs only token_entropies; capture_attention (their signal) is NOT reproduced.
    "lapeigvals_gsm8k_llama3b": _preset(
        paper="LapEigvals (arXiv 2502.17598)",
        # unsloth mirror = byte-identical weights; meta-llama/Llama-3.2-3B-Instruct 403s
        # (Llama-3.2 gate is separate from the 3.1 approval our token has; job 103536).
        # Same mirror pattern as huggyllama/llama-7b in inside_coqa_llama7b.
        model="unsloth/Llama-3.2-3B-Instruct",
        dataset="gsm8k", split="test", n_samples=1319,   # full test split, as LapEigvals
        k=1, temps=[1.0], max_new=2048,
        head_to_head="SAME-MODEL",
        published={"method": "LapEigvals", "metric": "AUROC",
                   "value": 71.7,          # unsupervised AttentionScore — the fair head-to-head Y
                   "supervised": 87.0,      # supervised attention-Laplacian probe (ceiling)
                   # Noise Injection (arXiv 2502.03799 v4, Table 3) — SAME model+dataset,
                   # unsupervised gray-box, but K=10 T=0.5 with question-level majority-vote
                   # labels (vs our K=1 answer-level): annotate the protocol gap when citing.
                   "noise_injection_k10": 82.70,
                   "answer_entropy_k10": 76.53,   # NI's no-noise baseline (answer entropy, K=10)
                   "model_note": "GSM8K / Llama-3.2-3B (LapEigvals Table 1): AttentionScore 71.7, probe 87.0. "
                                 "Noise Injection v4 Table 3 (same cell, K=10 question-level): "
                                 "answer entropy 76.53 -> +noise 82.70."},
        notes="LapEigvals GSM8K model-sweep + Noise Injection triple-anchor cell. Fair K=1 Y = unsup "
              "AttentionScore 71.7; NI 82.70 is K=10 question-level (protocol gap). Prompt = our boxed-CoT "
              "gsm8k_prompt (not LapEigvals Listing-5) — annotate when citing. Grader is_correct_gsm8k.",
    ),
    "lapeigvals_gsm8k_phi35": _preset(
        paper="LapEigvals (arXiv 2502.17598)",
        model="microsoft/Phi-3.5-mini-instruct",
        dataset="gsm8k", split="test", n_samples=1319,
        k=1, temps=[1.0], max_new=2048,
        head_to_head="SAME-MODEL",
        published={"method": "LapEigvals", "metric": "AUROC",
                   "value": 66.6,          # unsupervised AttentionScore
                   "supervised": 88.5,
                   "model_note": "GSM8K / Phi-3.5-mini (LapEigvals Table 1): AttentionScore 66.6, probe 88.5."},
        notes="LapEigvals GSM8K model-sweep. Fair Y = unsup AttentionScore 66.6. boxed-CoT prompt; "
              "is_correct_gsm8k grader. Phi-3.5 not gated.",
    ),
    "lapeigvals_gsm8k_nemo": _preset(
        paper="LapEigvals (arXiv 2502.17598)",
        model="mistralai/Mistral-Nemo-Instruct-2407", gated=True,
        dataset="gsm8k", split="test", n_samples=1319,
        k=1, temps=[1.0], max_new=2048,
        head_to_head="SAME-MODEL",
        published={"method": "LapEigvals", "metric": "AUROC",
                   "value": 63.0,          # unsupervised AttentionScore
                   "supervised": 89.0,
                   "model_note": "GSM8K / Mistral-Nemo-12B (LapEigvals Table 1): AttentionScore 63.0, probe 89.0."},
        notes="LapEigvals GSM8K model-sweep. Fair Y = unsup AttentionScore 63.0. boxed-CoT prompt; "
              "is_correct_gsm8k grader.",
    ),
    "lapeigvals_gsm8k_mistral24b": _preset(
        paper="LapEigvals (arXiv 2502.17598)",
        model="mistralai/Mistral-Small-24B-Instruct-2501", gated=True,
        dataset="gsm8k", split="test", n_samples=1319,
        k=1, temps=[1.0], max_new=2048,
        head_to_head="SAME-MODEL",
        published={"method": "LapEigvals", "metric": "AUROC",
                   "value": 57.6,          # unsupervised AttentionScore
                   "supervised": 92.5,
                   "model_note": "GSM8K / Mistral-Small-24B (LapEigvals Table 1): AttentionScore 57.6, probe 92.5."},
        notes="LapEigvals GSM8K model-sweep. Fair Y = unsup AttentionScore 57.6; probe 92.5 is the same "
              "92.5 previously miscited as the Llama-8B anchor. Strong model -> watch acc ceiling on the "
              "N=30 pilot (band [0.20,0.85]). boxed-CoT prompt; is_correct_gsm8k grader.",
    ),

    # ── ARS Qwen3-8B cells (arXiv 2601.17467) — ARS's OWN headline models. Supervised Y.
    "ars_gsm8k_qwen3_8b": _preset(
        paper="ARS (arXiv 2601.17467)",
        model="Qwen/Qwen3-8B",
        dataset="gsm8k", split="test", n_samples=500,
        # Greedy per ARS §5.1: "By default, greedy decoding is used to generate model
        # answers" (verified from arXiv 2026-07-11). max_new=8192: the T=1.0/mn4096 run
        # (job 101075) had 13% of traces pinned at the 4096 cap -> truncation-label
        # leakage; Qwen3 thinking routinely exceeds 4096. Do NOT lower max_new.
        k=1, temps=[0.0], max_new=8192,   # Qwen3 thinking ON; do NOT add /no_think here
        head_to_head="SAME-MODEL",
        published={"method": "ARS (CCS)", "metric": "AUROC", "value": 90.37,
                   "supervision": "supervised (representation shaping)",
                   "best_baseline": "Semantic Entropy 72.51",
                   "eigenscore_unsup": 63.40,   # ARS Table 2 vanilla EigenScore — fair UNSUP Y
                   "model_note": "GSM8K / Qwen3-8B (ARS Table 1) — ARS's strongest published cell. "
                                 "Unsup anchors from ARS Table 2: EigenScore 63.40."},
        notes="Reasoning cell: KEEP Qwen3 thinking ON (no /no_think) — ARS scores reasoning trajectories. "
              "is_correct_gsm8k extracts the boxed/final answer after </think>. CEILING CELL: T=1.0 run "
              "(job 101075, partial N=440) had acc 0.904 (~42 negatives); greedy may push higher -> watch "
              "the pilot. Greedy+thinking repetition risk: check pilot trace-length dist for cap-pinning. "
              "RE-RUN v2 (greedy/mn8192): archive the old partial first "
              "(mv $SHARED/results/repgrid/ars_gsm8k_qwen3_8b{,_mn4096_partial}); needs ~2 chained walls "
              "(sbatch --dependency=afterany, see submit_inference.sbatch header).",
    ),
    "ars_math500_qwen3_8b": _preset(
        paper="ARS (arXiv 2601.17467)",
        model="Qwen/Qwen3-8B",
        dataset="math500", split=None, n_samples=500,
        # Greedy per ARS §5.1 (see gsm8k cell). max_new=16384: at mn4096 (job 101076,
        # T=1.0) 45% of traces pinned; at mn8192 (pilot 103533, greedy) still 6/30 (20%)
        # pinned AND 3 of the 4 negatives were capped -> the truncation-label confound
        # survives 8192. Pilot tails show NO repetition loops (repeat-frac <= 0.08) --
        # the capped traces are genuinely long reasoning, so more room fixes them.
        # Do NOT resume the mn8192 pilot checkpoint (cap-mixing); it is archived as
        # ars_math500_qwen3_8b_mn8192_pilot.
        k=1, temps=[0.0], max_new=16384,
        head_to_head="SAME-MODEL",
        published={"method": "ARS (CCS)", "metric": "AUROC", "value": 78.66,
                   "supervision": "supervised (representation shaping)",
                   "best_baseline": "TSV 63.12",
                   "eigenscore_unsup": 81.38,   # ARS Table 2 vanilla EigenScore — fair UNSUP Y
                   "model_note": "MATH-500 / Qwen3-8B (ARS Table 1). "
                                 "Unsup anchors from ARS Table 2: EigenScore 81.38."},
        notes="Reasoning cell: Qwen3 thinking ON. is_correct_math extracts \\boxed{} after </think>. "
              "Slowest cell of the grid (long thinking traces): RE-RUN v3 (greedy/mn16384) needs ~4 chained "
              "walls at N=500 (mn8192 pilot: 30 problems in 70 min, mean 4584 tok). Stale runs archived as "
              "*_mn4096_partial and *_mn8192_pilot — never resume either (cap-mixing confound).",
    ),

    # ── Internal-States + Reasoning-Consistency (arXiv 2510.11529) — supervised. GSM8K/Qwen2.5-7B.
    "internalstates_gsm8k_qwen25_7b": _preset(
        paper="Internal-States + Reasoning-Consistency (arXiv 2510.11529)",
        model="Qwen/Qwen2.5-7B-Instruct",
        dataset="gsm8k", split="test", n_samples=500,
        # T=0.8 per the paper §3.1: "all decoding at a fixed temperature of 0.8 and a
        # maximum length of 300 tokens" (verified from arXiv 2026-07-11). We keep
        # max_new=2048 (their 300-token cap would truncate our entropy traces; the cap
        # is non-binding for GSM8K CoT ~150-250 tok — annotated protocol difference).
        k=1, temps=[0.8], max_new=2048,
        head_to_head="SAME-MODEL",
        published={"method": "Internal-States+RC", "metric": "AUROC", "value": 79.15,
                   "supervision": "supervised (LLM-judge labels, multi-path)",
                   "best_baseline": "V-STaR 76.55",
                   # Same paper's Table 1 UNSUPERVISED baselines — the fair Y for our L-SML:
                   "selfcheckgpt_unsup": 67.98,   # ± 1.28
                   "semantic_entropy_unsup": 58.36, "saplma": 59.72,
                   "model_note": "GSM8K / Qwen2.5-7B (arXiv 2510.11529). Fair unsup Y = SelfCheckGPT "
                                 "67.98±1.28; also SE 58.36, SAPLMA 59.72 (Table 1)."},
        notes="Supervised ceiling 79.15 + fair unsupervised Y SelfCheckGPT 67.98 on GSM8K/Qwen2.5-7B. "
              "boxed-CoT prompt; is_correct_gsm8k grader. RE-RUN v2 at the paper's T=0.8: the T=1.0 run "
              "(job 101077, HANDOFF_step166 diagnosis) collapsed Qwen2.5-7B to acc 0.284 with GENUINE "
              "errors (99% of wrongs still \\boxed{}) - operating-point confound, judge regrade won't fix. "
              "New T -> new pkl name (raw_gsm8k_T0.8.pkl), so the same --out dir is safe, but archive the "
              "T1.0 pkl before fetching (score_repgrid globs ALL raw_*.pkl in a cell dir). Labels: paper "
              "uses dual-LLM judge (GPT-4.1+Gemini) - ours stays lexical boxed-match, annotated.",
    ),

    # ── Noise Injection GSM8K sweep (arXiv 2502.03799 v4, Table 3). NI is the strongest
    #    published UNSUPERVISED GRAY-BOX GSM8K detector (answer entropy over K=10 samples,
    #    lifted by activation-noise injection). We run each of the paper's remaining models
    #    K=1 (our regime) on GSM8K and put our L-SML next to their K=10 numbers — protocol
    #    gap (K + question-level majority-vote labels vs our answer-level) annotated, same
    #    pattern as the SE-ICLR cell. Llama-3.2-3B is covered by lapeigvals_gsm8k_llama3b.
    #    NI protocol verified from v4: N=1319 (full test), K=10, T=0.5, majority-vote labels.
    "noise_gsm8k_phi3mini": _preset(
        paper="Noise Injection (arXiv 2502.03799)",
        model="microsoft/Phi-3-mini-4k-instruct",
        dataset="gsm8k", split="test", n_samples=1319,
        k=1, temps=[1.0], max_new=2048,
        head_to_head="SAME-MODEL",
        published={"method": "Noise Injection", "metric": "AUROC",
                   "value": 72.51,               # +noise, K=10 question-level
                   "answer_entropy_k10": 65.86,  # NI's no-noise baseline
                   "model_note": "GSM8K / Phi-3-mini-4k-instruct (NI v4 Table 3): "
                                 "answer entropy 65.86 -> +noise 72.51. K=10 question-level."},
        notes="NI GSM8K sweep. NOT the staged Phi-3.5 (that one is LapEigvals'). Protocol gap: "
              "NI is K=10 T=0.5 majority-vote question-level; ours K=1 answer-level. boxed-CoT "
              "prompt; is_correct_gsm8k grader.",
    ),
    "noise_gsm8k_mistral7b": _preset(
        paper="Noise Injection (arXiv 2502.03799)",
        model="mistralai/Mistral-7B-Instruct-v0.3", gated=True,
        dataset="gsm8k", split="test", n_samples=1319,
        k=1, temps=[1.0], max_new=2048,
        head_to_head="SAME-MODEL",
        published={"method": "Noise Injection", "metric": "AUROC",
                   "value": 78.50,               # +noise, K=10 question-level
                   "answer_entropy_k10": 75.85,
                   "model_note": "GSM8K / Mistral-7B-Instruct-v0.3 (NI v4 Table 3): "
                                 "answer entropy 75.85 -> +noise 78.50. K=10 question-level."},
        notes="NI GSM8K sweep. v0.3 exactly (paper-verified). Protocol gap as phi3mini. "
              "boxed-CoT prompt; is_correct_gsm8k grader.",
    ),
    "noise_gsm8k_gemma2b": _preset(
        paper="Noise Injection (arXiv 2502.03799)",
        # unsloth mirror = byte-identical weights; google/gemma-2b-it access request
        # stuck "awaiting review" (job 103542). Same mirror pattern as inside_coqa.
        model="unsloth/gemma-2b-it",
        dataset="gsm8k", split="test", n_samples=1319,
        k=1, temps=[1.0], max_new=2048,
        head_to_head="SAME-MODEL",
        published={"method": "Noise Injection", "metric": "AUROC",
                   "value": 57.11,               # +noise, K=10 question-level
                   "answer_entropy_k10": 51.36,
                   "model_note": "GSM8K / Gemma-2B-it (NI v4 Table 3): answer entropy 51.36 -> "
                                 "+noise 57.11. K=10 question-level."},
        notes="NI GSM8K sweep — HIGH ACC-FLOOR RISK (2B on GSM8K; NI's own near-chance AUROC "
              "suggests very low accuracy). N=30 pilot gate decides; a REJECT is itself the "
              "reportable outcome. gemma-2b-it NOT gemma-2-2b. boxed-CoT prompt; is_correct_gsm8k.",
    ),
    # ── EDIS grid (arXiv 2602.01288, Zhu et al.) — base Qwen2.5-Math-1.5B, the paper's own
    #    §5.3 protocol: GSM8K+MATH+AMC23+AIME24, T in {0.2,0.6,1.0}, K=8 responses/problem
    #    treated as independent points. Published: EDIS AUC=0.804 vs mean-entropy AUC=0.673
    #    (pooled, 26356 valid responses). Two prior attempts failed on class starvation /
    #    the wrong grading regex (HISTORY Steps 36/41/42, Phase 13, Step 156) — this sweep
    #    fixes both: is_correct_math's balanced-brace \boxed{} extraction (already fixed
    #    post-Phase-13) plus a new standing policy of over-collecting so both classes clear
    #    min_minority=100 per (dataset,T) cell, sized from the paper's OWN Table-1 "Mean"
    #    (unfiltered) baseline accuracy per dataset: GSM8K 36.0, MATH 30.0, AMC23 38.9,
    #    AIME24 7.2. Qwen2.5-Math-1.5B is a BASE checkpoint but its HF tokenizer_config.json
    #    ships a full ChatML chat_template tuned for boxed CoT math (verified) — raw_prompt
    #    stays False, same boxed-CoT prompts as every other math preset in this file.
    #
    #    PILOT FINDING, ROUND 1 (N=30 pilots on all 4, first attempt): 27-47% of responses
    #    hit the max_new cap with NO boxed answer — genuine degenerate-loop collapse (e.g.
    #    "Assistant\nAssistant\n..." or a single repeated token for the rest of the budget).
    #    ROUND 2 (misdiagnosed fix): tried repetition_penalty=1.15 + no_repeat_ngram_size=4.
    #    Loops stopped but accuracy got WORSE (7-9% vs a 30-39% expected baseline) — the hard
    #    n-gram ban was corrupting legitimate token repeats (variable names, digits) into
    #    garbled substitutes. Both generation-time knobs also bake into the entropy trace
    #    (LogitsProcessors run pre-sampling), which is a real fidelity problem for a project
    #    whose whole methodology depends on faithful H(n) — reverted, not just wrong-but-costly.
    #    ROOT CAUSE (found by checking the actual HF Hub configs, not guessing): Qwen2.5-Math-
    #    1.5B is a base checkpoint whose tokenizer_config.json ships a full ChatML
    #    chat_template (ending every turn with <|im_end|>, id 151645) but whose
    #    generation_config.json lists eos_token_id=151643 (<|endoftext|>) ONLY. generate()
    #    stops on generation_config.eos_token_id, not on whatever the chat template happens
    #    to close a turn with — so the model correctly finishes its answer, emits <|im_end|>,
    #    and generate() doesn't recognize that as done, so it keeps sampling into territory
    #    it was never trained to continue. That is exactly the observed symptom. FIX (this
    #    round): generate_full() now unions generation_config's eos_token_id with the chat
    #    template's turn-end marker (spectral_utils.model_utils.chat_turn_end_token_ids) —
    #    a pure stopping-criterion fix, doesn't touch the sampling distribution at all, so
    #    entropy fidelity is preserved. repetition_penalty/no_repeat_ngram_size reverted to
    #    None (paper-faithful — the paper doesn't describe any anti-repetition sampling, and
    #    now there's no reason to want it). Re-piloted before committing to a full run.
    #    Prompt content: the paper's extracted text (all 16 pages) never states an exact
    #    prompt string. Qwen's own shipped chat_template for this model defaults to the
    #    system message "Please reason step by step, and put your final answer within
    #    \boxed{}." — near-identical to this project's existing boxed-CoT prompt phrasing,
    #    so no prompt change was made; raw_prompt stays False (uses that chat_template).
    "edis_gsm8k_qwenmath15b": _preset(
        paper="EDIS (arXiv 2602.01288)",
        model="Qwen/Qwen2.5-Math-1.5B",
        dataset="gsm8k", split="test", n_samples=500,
        k=8, temps=[0.2, 0.6, 1.0], max_new=3072,
        capture={"full_entropy": True},
        min_minority=100,
        head_to_head="SAME-MODEL",
        published={"method": "EDIS", "metric": "AUROC (pooled, all 4 datasets x 3 temps)",
                   "value": 80.4, "mean_entropy_baseline": 67.3, "n_responses_paper": 26356,
                   "baseline_acc_gsm8k": 36.0,   # Table 1 "Mean" column (unfiltered accuracy)
                   "model_note": "GSM8K+MATH+AMC23+AIME24 / Qwen2.5-Math-1.5B (base), pooled "
                                 "T={0.2,0.6,1.0}. Table-1 Mean baseline acc: GSM8K 36.0, "
                                 "MATH 30.0, AMC23 38.9, AIME24 7.2."},
        notes="Section-5.3 protocol cell (GSM8K leg). n_samples=500/k=8 is ~5x the paper's own "
              "N=100 problems (standing over-collection policy: both classes must clear "
              "min_minority=100 per (dataset,T) cell). capture full_entropy for the paper's exact "
              "full-vocab H_t (Eq.1); our own top-K=15 token_entropies is captured unconditionally "
              "and used for L-SML GOOD_5. No-answer (unboxed) responses grade incorrect here rather "
              "than being excluded (the paper excludes them from its 26356) — protocol annotation, "
              "not a bug. max_new 2048->3072 for headroom; degenerate-loop fix is the eos_token_id "
              "stopping-criterion patch in generate_full, not a sampling-distribution change — see "
              "PILOT FINDING above. is_correct_gsm8k grader.",
    ),
    "edis_math500_qwenmath15b": _preset(
        paper="EDIS (arXiv 2602.01288)",
        model="Qwen/Qwen2.5-Math-1.5B",
        dataset="math500", split=None, n_samples=500,
        k=8, temps=[0.2, 0.6, 1.0], max_new=3072,
        capture={"full_entropy": True},
        min_minority=100,
        head_to_head="SAME-MODEL",
        published={"method": "EDIS", "metric": "AUROC (pooled, all 4 datasets x 3 temps)",
                   "value": 80.4, "mean_entropy_baseline": 67.3, "n_responses_paper": 26356,
                   "baseline_acc_math500": 30.0,
                   "model_note": "See edis_gsm8k_qwenmath15b for the full pooled protocol note."},
        notes="Section-5.3 protocol cell (MATH leg). Same sizing/capture rationale as "
              "edis_gsm8k_qwenmath15b, incl. the eos_token_id stopping-criterion fix (see PILOT "
              "FINDING above). is_correct_math grader (balanced-brace \\boxed{} extraction).",
    ),
    "edis_amc23_qwenmath15b": _preset(
        paper="EDIS (arXiv 2602.01288)",
        model="Qwen/Qwen2.5-Math-1.5B",
        dataset="amc23", split=None, n_samples=40,   # full AMC23 set, as the paper uses
        k=32, temps=[0.2, 0.6, 1.0], max_new=3072,    # k is the only lever: n_samples caps at 40
        capture={"full_entropy": True},
        min_minority=100,
        head_to_head="SAME-MODEL",
        published={"method": "EDIS", "metric": "AUROC (pooled, all 4 datasets x 3 temps)",
                   "value": 80.4, "mean_entropy_baseline": 67.3, "n_responses_paper": 26356,
                   "baseline_acc_amc23": 38.9,
                   "model_note": "See edis_gsm8k_qwenmath15b for the full pooled protocol note."},
        notes="Section-5.3 protocol cell (AMC23 leg). Full 40-problem set (paper-exact) x k=32 gives "
              "~500 minority-class responses at the paper's own 38.9% baseline acc — comfortably over "
              "the 100 target even at k=8; k=32 kept for CI tightness under the over-collection policy. "
              "This was the WORST cell in the pre-fix pilot (12.2% acc vs 38.9% expected, 36% no-boxed) "
              "— see PILOT FINDING above for the eos_token_id stopping-criterion fix, re-piloted before "
              "full. is_correct_amc23 grader (numeric boxed match, letter-match fallback unused here).",
    ),
    "edis_aime24_qwenmath15b": _preset(
        paper="EDIS (arXiv 2602.01288)",
        model="Qwen/Qwen2.5-Math-1.5B",
        dataset="aime24", split=None, n_samples=30,   # full AIME24 set, as the paper uses
        k=64, temps=[0.2, 0.6, 1.0], max_new=4096,    # extra headroom: hardest cell, base model
        capture={"full_entropy": True},
        min_minority=100,
        acc_band=(0.02, 0.85),   # paper's own baseline is ~7.2% — EXPECTED FLOOR, not a REJECT
        head_to_head="SAME-MODEL",
        published={"method": "EDIS", "metric": "AUROC (pooled, all 4 datasets x 3 temps)",
                   "value": 80.4, "mean_entropy_baseline": 67.3, "n_responses_paper": 26356,
                   "baseline_acc_aime24": 7.2,
                   "model_note": "See edis_gsm8k_qwenmath15b for the full pooled protocol note. AIME24 "
                                 "is the paper's own hardest cell for this model (Table 1 Mean=7.2 at m=4)."},
        notes="Section-5.3 protocol cell (AIME24 leg) — EXPECTED FLOOR, not a pipeline bug: the paper's "
              "own Table-1 baseline accuracy for Qwen2.5-Math-1.5B/AIME24 is ~7.2%. k=64 (vs 8/32 "
              "elsewhere) is sized from the min_minority=100 rule at that accuracy (100/(0.072*30)≈46, "
              "64 for margin against per-T variance) — may need re-sizing once the eos_token_id fix's "
              "effect on measured accuracy is known (see PILOT FINDING above; other 3 legs' accuracy "
              "was measured artificially low by degenerate-loop non-answers). Two-tier gate policy: "
              "scored + FLOOR flag, never REJECTed, as long as inspect_cell.py shows no cap-pinned "
              "leakage at max_new=4096. This is the cell that floored at Step 156 (1.7-2.9% acc, "
              "MAX_NEW=1024) — 4096 + the paper-verified 7.2% baseline should fix the truncation "
              "confound and the class-starvation failure mode from Steps 41/42; the eos_token_id fix "
              "addresses the distinct degenerate-loop failure mode found in this session's pilot. "
              "is_correct_aime24 grader.",
    ),

    # ── Punch-list cells (HANDOFF_punchlist_and_reruns.md, staged 2026-07-14) ──────

    # Item 6 — second cell for the Step-174 Item-5 fusion result. MATH-500/Qwen-Math-7B
    # showed L-SML 1-pass x answer-agreement SC K=5 -> 95.2 [91.8,98.0] (+10.1pp over the
    # best single arm, rho +0.23). Single-cell so far. This cell mirrors the Phase-15
    # protocol (K same-T raw passes, full rich-save schema) on GSM8K/Llama-8B, where the
    # fresh 1-pass GOOD_5 L-SML arm is already known (0.815, lapeigvals_gsm8k_llama8b).
    "fusion_gsm8k_llama8b_k5": _preset(
        paper="internal — Item-5 fusion replication (HISTORY Steps 174/181)",
        model="meta-llama/Llama-3.1-8B-Instruct", gated=True,
        dataset="gsm8k", split="test", n_samples=500,
        k=5, temps=[1.0], max_new=2048,
        notes="NOT a competitor cell — no published Y. Offline rescore mirrors "
              "scripts/rescore_phase15_selfconsistency.py: answer-agreement SC over the "
              "K=5 same-T passes, fused with the 1-pass L-SML GOOD_5 arm; gate = fused "
              "beats best single arm by >=1pp with CI excluding 0. K=5 at T=1.0 exactly "
              "(never mix temperatures — Step 158: diversity hurts -5.3pp). Expected acc "
              "~0.72 (in band) from the K=1 sibling cell.",
    ),

    # Item 7 — verbalized confidence on a 7B+ instruct model. The Step-131 null result
    # (model ignored the instruction) is only known on Qwen2.5-Math-1.5B; flagged
    # "expected to work on 7B+, untested" ever since. One-pass elicitation via
    # prompt_suffix; spectral capture unchanged, so the same trace also carries L-SML.
    "verbconf_gsm8k_llama8b": _preset(
        paper="internal — verbalized-confidence 7B+ test (HISTORY Step 131 follow-up)",
        model="meta-llama/Llama-3.1-8B-Instruct", gated=True,
        dataset="gsm8k", split="test", n_samples=500,
        k=1, temps=[1.0], max_new=2048,
        prompt_suffix='\n\nAfter your final boxed answer, end with one extra line '
                      '"Confidence: X" where X is an integer 0-100 giving your '
                      'confidence that the answer is correct.',
        notes="Offline scoring: spectral_utils.baselines.parse_verbalized_confidence on "
              "full_text (label-style regex first, last-integer fallback) -> report parse "
              "rate, VC AUROC, and rho(VC, GOOD_5 L-SML) for fusion viability (Extension B "
              "check). Grader unaffected by the trailing Confidence line: is_correct_gsm8k "
              "extracts \\boxed{} first and the fallback is pattern-anchored, not "
              "last-number (verified + smoke fixture). If parse rate < ~0.7 at pilot, the "
              "Step-131 null extends to 7B+ and that is the reportable outcome.",
    ),
}


def get_preset(preset_id: str) -> dict:
    if preset_id not in PRESETS:
        raise KeyError(f"unknown preset {preset_id!r}; known: {sorted(PRESETS)}")
    return dict(PRESETS[preset_id])


def list_presets() -> list:
    return sorted(PRESETS)
