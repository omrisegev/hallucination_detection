#!/usr/bin/env python
"""
LettuceDetect supervised ceiling on RAGTruth (RAG localization benchmark, Stage-1 item 6:
docs/research_notes/rag_localization_methods_and_benchmarks_2026.md).

Runs KRLabsOrg/lettucedect-base-modernbert-en-v1 (arXiv:2502.17125, MIT-licensed, trained as a
token classifier on RAGTruth) on our own `load_ragtruth()` rows -- the EXACT original published
(prompt, response, span-label) triples, so this is Comparison Level 1 (same outputs) with no
teacher-forcing or re-generation needed. Runs locally (ModernBERT-base is small); no cluster job.

Their own model card's `HallucinationDetector.predict(context=[...], question=..., answer=...)`
is the documented user-facing API; we pass the ENTIRE original RAGTruth prompt as `context` and
an empty `question` (RAGTruth's own instruction is already embedded in `prompt` for every task
type -- QA, Summary, Data2txt -- so there is no separate question string to extract without
guessing at their internal tokenization, which the documented API is the safe way to avoid).

Usage:
    python scripts/ragtruth_lettucedetect_ceiling.py --n 30           # pilot
    python scripts/ragtruth_lettucedetect_ceiling.py                  # full test split
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from spectral_utils.ragtruth import load_ragtruth

MODEL_ID = "KRLabsOrg/lettucedect-base-modernbert-en-v1"


def char_spans_overlap(a_spans, b_spans) -> bool:
    return any(a[0] < b[1] and b[0] < a[1] for a in a_spans for b in b_spans)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=None, help="cap rows (pilot); default = full test split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    cfg = ap.parse_args()

    from lettucedetect.models.inference import HallucinationDetector

    rows, diag = load_ragtruth(split="test", n=cfg.n, seed=cfg.seed)
    print(f"[lettuce] loaded {len(rows)} RAGTruth test rows (diagnostics: {diag})", flush=True)

    detector = HallucinationDetector(method="transformer", model_path=MODEL_ID)

    per_row = []
    t0 = time.time()
    for i, row in enumerate(rows):
        gold_spans = [(lab["start"], lab["end"]) for lab in row["labels"]]
        gold_hallucinated = len(gold_spans) > 0

        preds = detector.predict(
            context=[row["prompt"]], question="", answer=row["response"],
            output_format="spans",
        )
        pred_spans = [(p["start"], p["end"]) for p in preds]
        pred_hallucinated = len(pred_spans) > 0

        # Response-level: does the detector flag ANY span in a response with at least one gold
        # span, restricted to responses where the flagged span overlaps a real gold span (this
        # rewards localizing the right region, not merely guessing "hallucinated" on a long
        # multi-span response) -- reported alongside the coarser any-span response accuracy.
        overlap_hit = gold_hallucinated and pred_hallucinated and char_spans_overlap(gold_spans, pred_spans)

        per_row.append({
            "response_id": row["response_id"], "task_type": row["task_type"],
            "gold_hallucinated": gold_hallucinated, "pred_hallucinated": pred_hallucinated,
            "n_gold_spans": len(gold_spans), "n_pred_spans": len(pred_spans),
            "overlap_hit": overlap_hit,
        })
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"[lettuce] {i + 1}/{len(rows)} rows scored ({time.time() - t0:.1f}s elapsed)",
                  flush=True)

    tp = sum(1 for r in per_row if r["gold_hallucinated"] and r["pred_hallucinated"])
    fp = sum(1 for r in per_row if not r["gold_hallucinated"] and r["pred_hallucinated"])
    fn = sum(1 for r in per_row if r["gold_hallucinated"] and not r["pred_hallucinated"])
    tn = sum(1 for r in per_row if not r["gold_hallucinated"] and not r["pred_hallucinated"])
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else None
    n_overlap_hit = sum(1 for r in per_row if r["overlap_hit"])

    summary = {
        "model": MODEL_ID,
        "paper": "LettuceDetect (arXiv:2502.17125, KRLabsOrg) -- supervised token-classifier ceiling",
        "comparison_level": 1,
        "n_rows": len(rows),
        "diagnostics": diag,
        "example_level": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                           "precision": precision, "recall": recall, "f1": f1},
        "n_gold_hallucinated": sum(1 for r in per_row if r["gold_hallucinated"]),
        "n_overlap_hit": n_overlap_hit,
        "elapsed_sec": time.time() - t0,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    out_path = cfg.out or str(REPO_ROOT / "results" / "ragtruth_lettucedetect_ceiling.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_row": per_row}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
