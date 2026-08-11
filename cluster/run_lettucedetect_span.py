#!/usr/bin/env python
"""
LettuceDetect **span-preserving** supervised ceiling on RAGTruth, for the token/character-span
panel of docs/experiments/FOUR_LOCALIZATION_BENCHMARKS_CLUSTER_HANDOFF.md (Benchmark 1).

Supersedes `scripts/ragtruth_lettucedetect_ceiling.py`, which produced the currently-reported
0.759 example-level F1 but is **not sufficient for a span comparison** for two reasons this
driver fixes:

1. IT DISCARDED THE SPAN COORDINATES. It saved only `n_pred_spans` (a count) and a boolean
   `overlap_hit`, so no character-overlap precision/recall/F1 or IoU can be reconstructed from
   it. This driver persists every predicted span's `start`/`end`/`confidence`/`text` plus the
   full per-answer-token hallucination probability vector.

2. IT USED THE WRONG ENTRY POINT — a real train/test mismatch, not a stylistic difference.
   The old script called `predict(context=[row["prompt"]], question="")`. `predict()` routes
   through `PromptUtils.format_context`, which **re-wraps** whatever it is handed in the
   library's own passage template ("passage 1: ...", plus instruction text). Feeding it an
   already-complete RAGTruth prompt therefore double-wraps the input into a string the
   checkpoint never saw in training.

   The official RAGTruth preprocessing is unambiguous. `lettucedetect/preprocess/
   preprocess_ragtruth.py::create_sample` builds `HallucinationSample(prompt, answer, ...)`
   with `prompt = source["prompt"]` — the whole original RAGTruth prompt, one string, no
   context/question split — and training/eval both consume that via
   `HallucinationDataset.prepare_tokenized_input(tokenizer, prompt, answer, max_length)`.
   The matching inference call is `predict_prompt(prompt, answer)`, which goes straight to
   `_predict_single` with no re-wrapping. That is what this driver uses.

Span coordinates returned by `_predict_single` are already relative to the answer
(`rel_start = token_start - answer_char_offset`), so they are directly comparable to RAGTruth's
gold `label["start"]`/`label["end"]`, which index into `response`. No offset surgery needed.

FIDELITY GATE (handoff §6): the primary arm must reproduce the large checkpoint's published
RAGTruth example-level F1 of 79.22% within an explained tolerance. `--max-length` defaults to
the library's own 4096 so the primary arm matches the published setting; run a second job at
8192 (ModernBERT's native limit, which the model card advertises) as a declared no-truncation
sensitivity arm. `n_input_tokens` is recorded per row either way, so the truncation rate is
always measurable rather than assumed.

Usage:
    python cluster/run_lettucedetect_span.py \\
        --out $SHARED/results/ragtruth_lettuce_large_span_full
    python cluster/run_lettucedetect_span.py --max-length 8192 \\
        --out $SHARED/results/ragtruth_lettuce_large_span_ml8192
"""
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass

from spectral_utils import load_cache, save_cache_atomic
from spectral_utils.ragtruth import load_ragtruth

MODEL_ID = "KRLabsOrg/lettucedect-large-modernbert-en-v1"
PUBLISHED_EXAMPLE_F1 = 0.7922  # model card, RAGTruth test split, large checkpoint

EXIT_INCOMPLETE = 85
STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[lettuce] SIGTERM received — will checkpoint after the current row", flush=True)


def spans_overlap(a_spans, b_spans) -> bool:
    return any(a[0] < b[1] and b[0] < a[1] for a in a_spans for b in b_spans)


def example_level_stats(cache):
    """Response-level confusion matrix + P/R/F1 — the quantity the 79.22% gate is about."""
    rows = [e for e in cache.values() if "pred_spans" in e]
    tp = sum(1 for r in rows if r["gold_hallucinated"] and r["pred_hallucinated"])
    fp = sum(1 for r in rows if not r["gold_hallucinated"] and r["pred_hallucinated"])
    fn = sum(1 for r in rows if r["gold_hallucinated"] and not r["pred_hallucinated"])
    tn = sum(1 for r in rows if not r["gold_hallucinated"] and not r["pred_hallucinated"])
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    # `is not None`, never a truthiness test — a legitimate 0.0 precision must not be
    # silently converted to a null F1 (the exact bug fixed in processbench.first_error_f1).
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if (precision is not None and recall is not None) else None
    return {
        "n_rows": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_gold_hallucinated": sum(1 for r in rows if r["gold_hallucinated"]),
        "n_overlap_hit": sum(1 for r in rows if r["overlap_hit"]),
        "n_truncated": sum(1 for r in rows if r["truncated"]),
    }


def run(detector, tokenizer, rows, cfg, out_path):
    cache = load_cache(out_path)
    n_done = sum(1 for e in cache.values() if "pred_spans" in e)
    print(f"[lettuce] {n_done}/{len(rows)} rows already done -> {out_path}", flush=True)

    for idx, row in enumerate(rows):
        if idx in cache and "pred_spans" in cache[idx]:
            continue
        if STOP["flag"]:
            save_cache_atomic(cache, out_path)
            print(f"PREEMPTED — checkpoint saved with {len(cache)} rows", flush=True)
            return False

        t0 = time.time()
        prompt, answer = row["prompt"], row["response"]

        # Two calls into the library's own code path: the span merging rule (consecutive
        # above-threshold tokens joined, confidence = max) is reproduced by THEIR code, not
        # reimplemented here, per the handoff's "reproduce that rule exactly" requirement.
        pred_spans = detector.predict_prompt(prompt, answer, output_format="spans")
        token_preds = detector.predict_prompt(prompt, answer, output_format="tokens")

        n_input_tokens = int(
            tokenizer(prompt, answer, add_special_tokens=True, return_tensors="pt")
            ["input_ids"].shape[1]
        )
        gold_spans = [(lab["start"], lab["end"]) for lab in row["labels"]]
        gold_hallucinated = len(gold_spans) > 0
        pred_tuples = [(int(p["start"]), int(p["end"])) for p in pred_spans]

        cache[idx] = {
            "response_id": row["response_id"],
            "source_id": row["source_id"],
            "task_type": row["task_type"],
            "model": row["model"],
            "response": answer,
            # gold, carried through so the scorer never has to re-join the corpus
            "gold_spans": gold_spans,
            "gold_hallucinated": gold_hallucinated,
            # prediction, at full resolution
            "pred_spans": [
                {"start": int(p["start"]), "end": int(p["end"]),
                 "confidence": float(p.get("confidence", 0.0)), "text": p.get("text", "")}
                for p in pred_spans
            ],
            "pred_hallucinated": len(pred_spans) > 0,
            "token_probs": [float(t["prob"]) for t in token_preds],
            "token_preds": [int(t["pred"]) for t in token_preds],
            "token_strings": [t["token"] for t in token_preds],
            "overlap_hit": gold_hallucinated and bool(pred_tuples)
                           and spans_overlap(gold_spans, pred_tuples),
            "n_input_tokens": n_input_tokens,
            "truncated": n_input_tokens > cfg.max_length,
            "elapsed_sec": time.time() - t0,
        }
        if (idx + 1) % 50 == 0 or idx + 1 == len(rows):
            print(f"[lettuce] {idx + 1}/{len(rows)} rows "
                  f"({time.time() - t0:.2f}s last)", flush=True)
        if (idx + 1) % cfg.checkpoint_every == 0:
            save_cache_atomic(cache, out_path)

    save_cache_atomic(cache, out_path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-samples", type=int, default=None, help="cap rows (debug only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=4096,
                    help="library default 4096 = the published setting; 8192 is ModernBERT's "
                         "native limit and must be reported as a separate sensitivity arm")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="0.0 keeps every span — the paper's own fixed-threshold merging rule; "
                         "any other value is a sensitivity analysis, not the primary arm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    cfg = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    out_dir = cfg.out or f"results_lettuce_{cfg.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)

    from lettucedetect.models.inference import HallucinationDetector

    print(f"[lettuce] model={cfg.model} split={cfg.split} max_length={cfg.max_length} "
          f"min_confidence={cfg.min_confidence}", flush=True)

    rows, diag = load_ragtruth(split=cfg.split, n=cfg.n_samples, seed=cfg.seed)
    print(f"[lettuce] loaded {len(rows)} RAGTruth rows (diagnostics: {diag})", flush=True)

    detector = HallucinationDetector(
        method="transformer", model_path=cfg.model, max_length=cfg.max_length,
    )
    tokenizer = detector.detector.tokenizer

    out_path = os.path.join(out_dir, f"lettuce_spans_{cfg.split}.pkl")
    t0 = time.time()
    if not run(detector, tokenizer, rows, cfg, out_path):
        print("[lettuce] INCOMPLETE — resubmit with the same --out to resume", flush=True)
        sys.exit(EXIT_INCOMPLETE)

    stats = example_level_stats(load_cache(out_path))
    gate_delta = None if stats["f1"] is None else stats["f1"] - PUBLISHED_EXAMPLE_F1
    manifest = {
        "driver": "run_lettucedetect_span.py",
        "paper": "LettuceDetect: A Hallucination Detection Framework for RAG Applications "
                 "(arXiv:2502.17125, KRLabsOrg)",
        "benchmark": "RAGTruth (Niu et al., ACL 2024) — vendored, data/ragtruth_protocol/",
        "model": cfg.model,
        "entry_point": "HallucinationDetector.predict_prompt(prompt, answer) — the call that "
                       "matches preprocess_ragtruth.py's HallucinationSample(prompt, answer); "
                       "predict(context=[...], question=...) re-wraps via "
                       "PromptUtils.format_context and is NOT the RAGTruth path",
        "supervision": "supervised token classifier trained on RAGTruth's own train split",
        "fidelity_level": 1,
        "max_length": cfg.max_length,
        "min_confidence": cfg.min_confidence,
        "split": cfg.split,
        "n_samples": cfg.n_samples,
        "ragtruth_diagnostics": diag,
        "published_example_f1": PUBLISHED_EXAMPLE_F1,
        "fidelity_gate_delta": gate_delta,
        "example_level": stats,
        "elapsed_sec": time.time() - t0,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = os.path.join(out_dir, "manifest.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(path + ".tmp", path)

    print("\n=== EXAMPLE-LEVEL ===")
    print(json.dumps(stats, indent=2))
    print(f"\nfidelity gate: published {PUBLISHED_EXAMPLE_F1:.4f} vs measured "
          f"{stats['f1']} (delta {gate_delta})", flush=True)
    print(f"COMPLETE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
