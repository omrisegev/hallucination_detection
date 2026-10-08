"""
refchecker.py — loader, claim rendering, and metric for the **RefChecker benchmark**
(Hu et al., "Refchecker: Reference-based fine-grained hallucination checker and benchmark for
large language models" / Knowledge-Centric Hallucination Detection,
aclanthology.org/2024.emnlp-main.395/; code: github.com/amazon-science/RefChecker).

Benchmark 2b of docs/experiments/FOUR_LOCALIZATION_BENCHMARKS_CLUSTER_HANDOFF.md: unsupported
**CLAIM** localization. GASP (Benchmark 2a) treats a sentence as the localized unit; this panel
tests whether the same score still works when the unit is an explicit semantic claim, and it
exposes a real limitation of a scalar risk score — it separates supported from unsupported, but
it does not by itself distinguish a CONTRADICTION from MISSING evidence.

THE CLAIM SET IS FIXED, AND THAT IS A DELIBERATE DESIGN CHOICE
--------------------------------------------------------------------------------------
The benchmark ships human labels attached to triplets that were extracted by **Claude 2**
(`claude2_response_kg`: a list of `{"triplet": [head, relation, tail], "human_label": ...}` per
response). Running a different extractor would produce a different claim set, which the shipped
human labels do not cover — so an "open extractor" arm could not be scored against this gold at
all without a new annotation effort.

This module therefore fixes the claim set to the shipped, human-labelled triplets, and the panel
compares **checkers** on identical claims:

  - competitor: an open checker classifies each fixed triplet against the reference, 3-way;
  - ours:       the same fixed triplet is teacher-forced under the official context conditions,
                producing a scalar risk, scored only under the binary collapse.

Both judge the identical claim set against identical gold, which is what the handoff's
apples-to-apples rule requires. **Claim EXTRACTION is explicitly out of scope for this panel**
and must be recorded as such — we are not reproducing RefChecker end to end, only its checking
stage. The paper's strongest configuration additionally uses proprietary models (GPT-4 /
Claude 2), which stay quoted as published context (fidelity level 4), never reproduced.

THE THREE SETTINGS ARE NOT THREE DIFFICULTIES OF ONE TASK
--------------------------------------------------------------------------------------
`zero_context` (NQ): the generator saw only a question; the reference is NQ's own long answer,
retrieved afterwards purely for checking. `noisy_context` (MS MARCO): retrieved passages, some
irrelevant. `accurate_context` (Dolly): a given document the answer should rewrite. They are
reported separately, never pooled into one number.
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE.parent / "data" / "refchecker_protocol"

CONTEXT_SETTINGS = ("zero_context", "noisy_context", "accurate_context")
DATASET_OF = {"zero_context": "nq", "noisy_context": "msmarco", "accurate_context": "dolly"}
GENERATORS = ("alpaca_7B", "chatgpt", "claude2", "davinci001",
              "falcon_40B_instruct", "gpt4", "llama2_70b_chat")

# RefChecker's own three-way claim verdict.
LABELS = ("Entailment", "Neutral", "Contradiction")
SUPPORTED_LABEL = "Entailment"


def triplet_text(triplet) -> str:
    """Deterministic textual rendering of a `[head, relation, tail]` triplet.

    Deterministic on purpose: this string is teacher-forced, so any variation in spacing or
    punctuation changes the token sequence and therefore the telemetry. Kept deliberately plain
    — no articles inserted, no case normalisation — so the rendering adds no information the
    triplet did not already carry.
    """
    parts = [str(p).strip() for p in triplet]
    return " ".join(p for p in parts if p).rstrip(".") + "."


def binary_collapse(label: str) -> int:
    """1 == UNSUPPORTED. Collapses `Contradiction` and `Neutral` together, per the handoff:
    our scalar score is reported only under this collapse and never placed in the paper's
    three-way column."""
    return 0 if label == SUPPORTED_LABEL else 1


def load_refchecker(data_dir=None, settings=None, generators=None):
    """Join the human annotations with their questions/references into flat CLAIM rows.

    Expects `data_dir` (default `data/refchecker_protocol/`) laid out as the official repo's
    `benchmark/` folder after `download_data.sh`:

        <setting>/<dataset>.json                     questions + context per example id
        <setting>/<dataset>_<generator>_answers.json responses + claude2_response_kg

    Returns `(claims, diagnostics)`. Each claim row:
        example_id, setting, dataset, generator, response, triplet, claim_text,
        human_label, label_unsupported, question, context (list[str]), claim_index

    A response whose example id is missing from the context file is dropped and counted, never
    silently scored without its reference.
    """
    data_dir = Path(data_dir) if data_dir else _DATA_DIR
    settings = tuple(settings) if settings else CONTEXT_SETTINGS
    generators = tuple(generators) if generators else GENERATORS

    claims = []
    n_responses, n_dropped_no_context, n_dropped_no_kg = 0, 0, 0
    missing_files, label_counts, per_setting = [], {}, {}

    for setting in settings:
        dataset = DATASET_OF[setting]
        context_path = data_dir / setting / f"{dataset}.json"
        if not context_path.exists():
            missing_files.append(str(context_path))
            continue
        by_id = {str(ex["id"]): ex for ex in json.loads(context_path.read_text(encoding="utf-8"))}

        for generator in generators:
            answers_path = data_dir / setting / f"{dataset}_{generator}_answers.json"
            if not answers_path.exists():
                missing_files.append(str(answers_path))
                continue
            for row in json.loads(answers_path.read_text(encoding="utf-8")):
                n_responses += 1
                example = by_id.get(str(row["id"]))
                if example is None:
                    n_dropped_no_context += 1
                    continue
                kg = row.get("claude2_response_kg")
                if not kg:
                    n_dropped_no_kg += 1
                    continue
                for claim_index, item in enumerate(kg):
                    label = item.get("human_label")
                    if label not in LABELS:
                        continue
                    label_counts[label] = label_counts.get(label, 0) + 1
                    per_setting[setting] = per_setting.get(setting, 0) + 1
                    claims.append({
                        "example_id": str(row["id"]),
                        "setting": setting,
                        "dataset": dataset,
                        "generator": generator,
                        "response": row["response"],
                        "triplet": list(item["triplet"]),
                        "claim_text": triplet_text(item["triplet"]),
                        "human_label": label,
                        "label_unsupported": binary_collapse(label),
                        "question": example.get("question", ""),
                        "context": list(example.get("context") or []),
                        "claim_index": claim_index,
                    })

    diagnostics = {
        "data_dir": str(data_dir),
        "settings": list(settings),
        "n_responses_seen": n_responses,
        "n_claims": len(claims),
        "n_dropped_no_context": n_dropped_no_context,
        "n_dropped_no_kg": n_dropped_no_kg,
        "n_missing_files": len(missing_files),
        "missing_files": missing_files[:10],
        "label_counts": label_counts,
        "claims_per_setting": per_setting,
    }
    return claims, diagnostics


# ── metrics ──────────────────────────────────────────────────────────────────────────

def three_way_metrics(gold, pred, labels=LABELS):
    """Official-style three-way accuracy + per-class and macro F1.

    Used ONLY for a checker that emits a three-way verdict. A scalar risk score has no
    three-way output and must never be scored here — see `binary_metrics`.
    """
    gold, pred = list(gold), list(pred)
    assert len(gold) == len(pred), (len(gold), len(pred))
    n = len(gold)
    accuracy = sum(1 for g, p in zip(gold, pred) if g == p) / n if n else None

    per_class, f1s = {}, []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if (precision is not None or recall is not None) else None
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": sum(1 for g in gold if g == label)}
        if f1 is not None:
            f1s.append(f1)
    return {"accuracy": accuracy, "macro_f1": sum(f1s) / len(f1s) if f1s else None,
            "per_class": per_class, "n": n}


def binary_metrics(gold_unsupported, scores, threshold=None):
    """Threshold-free ranking metrics for a scalar risk score under the supported/unsupported
    collapse, plus thresholded P/R/F1 when a threshold frozen on a non-test split is supplied.

    `scores` must be risk-oriented: HIGHER = more likely unsupported.
    """
    import numpy as np
    from .fusion_utils import boot_auc

    gold = np.asarray(gold_unsupported, dtype=int)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    gold, scores = gold[ok], scores[ok]
    out = {"n": int(gold.size), "n_unsupported": int(gold.sum()),
           "n_dropped_non_finite": int((~ok).sum())}
    if gold.size == 0 or len(set(gold.tolist())) < 2:
        out.update({"auroc": None, "auprc": None})
        return out

    auc, lo, hi = boot_auc(gold, scores)
    out.update({"auroc": float(auc), "auroc_ci_lo": float(lo), "auroc_ci_hi": float(hi)})

    order = np.argsort(-scores)
    hits = gold[order]
    precision_at_k = np.cumsum(hits) / np.arange(1, hits.size + 1)
    out["auprc"] = float((precision_at_k * hits).sum() / hits.sum()) if hits.sum() else None

    if threshold is not None:
        pred = (scores >= threshold).astype(int)
        tp = int(((pred == 1) & (gold == 1)).sum())
        fp = int(((pred == 1) & (gold == 0)).sum())
        fn = int(((pred == 0) & (gold == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        out.update({"threshold": float(threshold), "precision": precision,
                    "recall": recall, "f1": f1})
    return out


# ── known-answer tests ───────────────────────────────────────────────────────────────

def smoke() -> None:
    checks = 0

    assert triplet_text(["The Mamas & The Papas", "sang", "I Dig Rock and Roll Music"]) == \
        "The Mamas & The Papas sang I Dig Rock and Roll Music."
    assert triplet_text([" A ", "", "B."]) == "A B."          # blanks dropped, one trailing dot
    checks += 1

    assert binary_collapse("Entailment") == 0
    assert binary_collapse("Neutral") == 1 and binary_collapse("Contradiction") == 1
    checks += 1

    gold = ["Entailment", "Entailment", "Neutral", "Contradiction"]
    perfect = three_way_metrics(gold, gold)
    assert perfect["accuracy"] == 1.0 and perfect["macro_f1"] == 1.0, perfect
    shifted = three_way_metrics(gold, ["Neutral"] * 4)
    assert shifted["accuracy"] == 0.25, shifted
    # A class the predictor never emits scores 0.0, not None — the same zero-vs-missing
    # distinction that produced a real null-F1 bug in processbench.first_error_f1.
    assert shifted["per_class"]["Contradiction"]["f1"] == 0.0, shifted["per_class"]
    checks += 1

    import numpy as np
    ranked = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert ranked["auroc"] == 1.0, ranked
    inverted = binary_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert inverted["auroc"] == 0.0, inverted
    thresholded = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    assert thresholded["precision"] == 1.0 and thresholded["recall"] == 1.0
    single = binary_metrics([1, 1], [0.3, 0.4])
    assert single["auroc"] is None, single      # one class only -> undefined, not 0.5
    nonfinite = binary_metrics([0, 1, 1], [0.1, np.nan, 0.9])
    assert nonfinite["n"] == 2 and nonfinite["n_dropped_non_finite"] == 1, nonfinite
    checks += 1

    print(f"refchecker.smoke: PASS ({checks} checks: triplet rendering, binary collapse, "
          f"three-way metrics, ranking metrics)")


if __name__ == "__main__":
    smoke()
