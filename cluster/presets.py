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
        published={"method": "LapEigvals", "metric": "AUROC", "value": 92.5,
                   "model_note": "supervised probe on Mistral-Small-24B, not Llama-8B"},
        notes="capture_attention deferred (NotImplemented). gsm8k_prompt is the boxed "
              "CoT prompt, not LapEigvals Listing-5 — note when citing.",
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
        notes="Boundary cell: open-domain single-fact -> short trace.",
    ),
    "truthfulqa_llama8b": _preset(
        paper="TruthfulQA (generation)",
        model="meta-llama/Llama-3.1-8B-Instruct", gated=True,
        dataset="truthfulqa", split="validation", n_samples=817,
        k=10, temps=[0.5], max_new=128, capture={"logsumexp": True},
        notes="At-inference label is a ROUGE-L proxy; re-label offline with a judge.",
    ),
    "sciq_llama8b": _preset(
        paper="SciQ (4-way MCQ)",
        model="meta-llama/Llama-3.1-8B-Instruct", gated=True,
        dataset="sciq", split="validation", n_samples=1000,
        k=1, temps=[1.0], max_new=512,
        notes="MCQ -> suppressed entropy dynamics (GPQA-like); expect a modest cell.",
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
}


def get_preset(preset_id: str) -> dict:
    if preset_id not in PRESETS:
        raise KeyError(f"unknown preset {preset_id!r}; known: {sorted(PRESETS)}")
    return dict(PRESETS[preset_id])


def list_presets() -> list:
    return sorted(PRESETS)
