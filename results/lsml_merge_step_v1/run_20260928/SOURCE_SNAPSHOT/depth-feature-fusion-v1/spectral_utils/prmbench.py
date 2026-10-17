"""
prmbench.py — loader, official metric port, and alignment helpers for **PRMBench**
(Song et al., "PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward
Models", arXiv:2501.03124, ACL 2025).

Benchmark 3 of docs/experiments/FOUR_LOCALIZATION_BENCHMARKS_CLUSTER_HANDOFF.md: **correctness
of EVERY reasoning step**. ProcessBench only annotates the FIRST wrong step and certifies
nothing after it, so it cannot measure an every-step classifier; PRMBench supplies the missing
per-step ground truth — 6,216 problems / 83,456 step labels (both counts reproduced exactly by
`load_prmbench`, see `smoke()`).

WHY THIS IS A PORT AND NOT AN INTERPRETATION
--------------------------------------------------------------------------------------
Everything below mirrors the official `mr_eval` task `prmtest_classified`
(github.com/ssmisya/PRMBench, `mr_eval/tasks/prmtest_classified/task.py` +
`mr_eval/utils/task_utils.py::eval_on_hallucination_step`, both fetched 2026-08-10). Four
details there are *not* guessable from the dataset card, and each one silently changes the
score if you reinvent it:

1. **The evaluated question is `modified_question`, not `question`.** The raw dataset carries
   three question fields. The official loader takes `modified_question` + `modified_process`.
   (`question` differs from BOTH `original_question` and `modified_question` on ~250-500 rows
   per class, so picking it would quietly mis-condition thousands of traces.)

2. **A synthetic `"correct"` class is CONSTRUCTED, not shipped.** For every `redundency` row the
   loader appends an extra sample built from `original_question` + `original_process` with
   `error_steps=[]` and `classification="correct"`. The evaluated corpus is therefore LARGER
   than the 6,216 rows on the Hub, and the all-steps-correct control class exists only because
   this function creates it.

3. **`label == 1` means the model asserts the step is VALID.** In
   `eval_on_hallucination_step`, `POSITIVE_LABEL = 1` and TP counts *non-error* steps the model
   called valid. The "positive" class of the official F1 is **correct steps**, not errors — so a
   detector that scores risk must be inverted before its labels enter this metric.

4. **Out-of-range `error_steps` are inert upstream, so they must stay inert here.** 100 of the
   6,216 rows annotate a step index past the end of their own `modified_process` (e.g. idx
   `confidence_prm_test_p1_29`: 53 steps, `error_steps=[52, 54]`). The official loop only ever
   tests `idx in hallucination_steps` for `idx` in `range(len(labels))`, so those indices simply
   never match. We reproduce that — the rows are KEPT, the stray indices contribute nothing, and
   the count is reported in diagnostics rather than silently dropped or silently repaired.

Additionally, 165 rows (all of class `multi_solutions`) ship with an EMPTY `error_steps`: a
valid alternative solution where every step is correct. That is signal, not corruption.

WHAT WE CANNOT REPRODUCE FAITHFULLY, AND SAY SO
--------------------------------------------------------------------------------------
For `redundency` and `circular`, the official evaluator prefers a model's
`step_level_redundancy_labels` head and only falls back to `step_level_validity_labels` when
that head is absent. Our score has no redundancy head, so those two classes go through the
fallback path. That is a declared **adaptation**, recorded by
`prmbench_evaluate(...)["used_redundancy_head"]`, and must be labelled as such in any report —
it is not an exact reproduction for those two subcategories.
"""
from __future__ import annotations

import numpy as np

PRMBENCH_DATASET_ID = "hitsmy/PRMBench_Preview"
PRMBENCH_SPLIT = "train"

# The class whose rows seed the synthetic all-correct control (official: `correct_sample_classification`).
CORRECT_SEED_CLASSIFICATION = "redundency"
CORRECT_CLASSIFICATION = "correct"

# The nine shipped error classes, mapped onto the paper's three published categories.
# Names on the left are the dataset's own `classification` strings, including its spelling of
# "redundency" — do not "fix" it, it is the join key.
CATEGORY_OF = {
    "redundency": "simplicity",            # Non-Redundancy (NR)
    "circular": "simplicity",              # Non-Circular Logic (NCL)
    "counterfactual": "soundness",         # Empirical Soundness (ES)
    "step_contradiction": "soundness",     # Step Consistency (SC)
    "domain_inconsistency": "soundness",   # Domain Inconsistency (DC)
    "confidence": "soundness",             # Confidence Invariance (CI)
    "missing_condition": "sensitivity",    # Prerequisite Sensitivity (PS)
    "deception": "sensitivity",            # Deception Resistance (DR)
    "multi_solutions": "sensitivity",      # Multi-Solution Consistency (MS)
    CORRECT_CLASSIFICATION: "control",
}
CATEGORIES = ("simplicity", "soundness", "sensitivity")

METRIC_TYPES = ("correct_step_acc", "wrong_step_acc", "total_step_acc", "first_error_acc")

STEP_SEP = "\n\n"


# ── loading ──────────────────────────────────────────────────────────────────────────

def load_prmbench(n_samples: int = None, seed: int = 0, dataset_id: str = PRMBENCH_DATASET_ID,
                  split: str = PRMBENCH_SPLIT, revision: str = None):
    """Port of `mr_eval/tasks/prmtest_classified/task.py::load_data_function`.

    Returns `(meta_data, diagnostics)`. Each meta row: `idx, question, steps, error_steps,
    classification, category`, plus `source_idx` (the raw row's own `idx`, so a prediction can
    always be traced back to the Hub record).

    `n_samples` caps the RAW rows before the synthetic-correct expansion, so a capped draw stays
    reproducible and still contains both classes; it is a debug aid, never the reported run.
    """
    from datasets import load_dataset

    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    raw = load_dataset(dataset_id, **kwargs)

    if n_samples is not None and n_samples < len(raw):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(raw), size=n_samples, replace=False))
        raw = raw.select(keep)

    meta, seen = [], set()
    n_oob, n_empty, n_raw_steps, n_dup = 0, 0, 0, 0
    for item in raw:
        item_idx = item["idx"]
        classification = item["classification"]
        n_raw_steps += len(item["modified_process"])

        # The synthetic all-correct control, seeded off `redundency` rows only.
        if classification == CORRECT_SEED_CLASSIFICATION:
            correct_idx = f"correct_{item_idx}"
            if correct_idx not in seen:
                seen.add(correct_idx)
                meta.append({
                    "idx": correct_idx, "source_idx": item_idx,
                    "question": item["original_question"], "steps": list(item["original_process"]),
                    "error_steps": [], "classification": CORRECT_CLASSIFICATION,
                    "category": CATEGORY_OF[CORRECT_CLASSIFICATION],
                })

        classification_idx = f"{classification}_{item_idx}"
        if classification_idx in seen:
            n_dup += 1
            continue
        seen.add(classification_idx)

        steps = list(item["modified_process"])
        error_steps = list(item["error_steps"])
        if not error_steps:
            n_empty += 1
        if any(i < 1 or i > len(steps) for i in error_steps):
            n_oob += 1

        meta.append({
            "idx": classification_idx, "source_idx": item_idx,
            "question": item["modified_question"],  # NOT item["question"] — see module docstring
            "steps": steps, "error_steps": error_steps,
            "classification": classification,
            "category": CATEGORY_OF.get(classification, "unknown"),
        })

    counts = {}
    for row in meta:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    diagnostics = {
        "dataset_id": dataset_id, "split": split, "revision": revision,
        "n_raw_rows": len(raw), "n_meta_rows": len(meta),
        # The paper's headline "83,456 step labels" counts the RAW corpus. The official loader's
        # dedup on f"{classification}_{idx}" then removes 5 rows (all multi_solutions, 85 steps),
        # so the corpus actually evaluated carries 83,371 error-class steps. Both are reported;
        # neither is silently substituted for the other.
        "n_raw_steps": n_raw_steps,
        "n_duplicate_rows_dropped": n_dup,
        "n_steps_total": sum(len(r["steps"]) for r in meta),
        "n_steps_error_classes": sum(len(r["steps"]) for r in meta
                                     if r["classification"] != CORRECT_CLASSIFICATION),
        "n_rows_empty_error_steps": n_empty,
        "n_rows_out_of_range_error_steps": n_oob,
        "counts_by_classification": counts,
    }
    return meta, diagnostics


# ── the official per-row step metric ──────────────────────────────────────────────────

def eval_on_hallucination_step(hallucination_steps, labels, redundancy_label: bool = False):
    """Verbatim port of `mr_eval/utils/task_utils.py::eval_on_hallucination_step`.

    `hallucination_steps` are 1-indexed on input and shifted to 0-indexed here, exactly as
    upstream. `labels[i]` is the model's assertion about step i, where — for the default
    `redundancy_label=False` — **1 means "this step is VALID"**. TP therefore counts correct
    steps the model accepted; TN counts error steps the model rejected.

    Indices in `hallucination_steps` that fall outside `range(len(labels))` never match and are
    inert, which is upstream behaviour and the reason 100 out-of-range rows are kept rather than
    dropped (see module docstring).
    """
    hallucination_steps = [i - 1 for i in hallucination_steps]
    positive, negative = (0, 1) if redundancy_label else (1, 0)

    correct_step_acc, wrong_step_acc, total_step_acc = [], [], []
    tp = fp = tn = fn = 0

    first_error_location = min(hallucination_steps) if hallucination_steps else -1
    first_error_acc = None

    for idx in range(len(labels)):
        if idx == first_error_location:
            first_error_acc = 1 if labels[idx] == negative else 0

        if idx in hallucination_steps:
            if labels[idx] == positive:
                wrong_step_acc.append(0)
                total_step_acc.append(0)
                fp += 1
            else:
                wrong_step_acc.append(1)
                total_step_acc.append(1)
                tn += 1
        else:
            if labels[idx] == positive:
                correct_step_acc.append(1)
                total_step_acc.append(1)
                tp += 1
            else:
                correct_step_acc.append(0)
                total_step_acc.append(0)
                fn += 1

    def _mean(values):
        return sum(values) / len(values) if values else -1

    model_response_acc = _mean(list(labels)) if len(labels) else -1
    return {
        "correct_step_acc": _mean(correct_step_acc),
        "wrong_step_acc": _mean(wrong_step_acc),
        "total_step_acc": _mean(total_step_acc),
        "first_error_acc": first_error_acc,
        "model_response_acc": model_response_acc,
        "f1_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "correct_step_acc_list": correct_step_acc,
        "wrong_step_acc_list": wrong_step_acc,
        "total_step_acc_list": total_step_acc,
        "first_error_acc_list": [first_error_acc] if first_error_acc is not None else [],
        "model_response_acc_list": [model_response_acc] if model_response_acc != -1 else [],
    }


def _prf(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) else -1
    recall = tp / (tp + fn) if (tp + fn) else -1
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else -1
    neg_precision = tn / (tn + fn) if (tn + fn) else -1
    neg_recall = tn / (tn + fp) if (tn + fp) else -1
    neg_f1 = ((2 * neg_precision * neg_recall / (neg_precision + neg_recall))
              if (neg_precision + neg_recall) else -1)
    return {"precision": precision, "recall": recall, "f1": f1,
            "negative_precision": neg_precision, "negative_recall": neg_recall,
            "negative_f1": neg_f1}


def prmbench_evaluate(predictions, meta_data, use_redundancy_head: bool = False):
    """Port of `prmtest_classified/task.py::evaluate_function`.

    `predictions` is an iterable of `{"idx": str, "labels": list[int]}` — `labels[i] == 1` means
    the scorer asserts step i is VALID. Extra keys are ignored.

    Reproduces upstream's structure exactly: the synthetic `correct` rows are scored FIRST and
    are excluded from the pooled totals (they only seed the per-row `similarity` term, which is
    `|model_response_acc(modified) - model_response_acc(matching correct)|`); every other class
    contributes to both the pooled and the per-classification tables.

    `use_redundancy_head=False` (our case — we have no redundancy head) routes `redundency` and
    `circular` through the validity fallback. That is an adaptation, flagged in the return value.
    """
    meta_by_idx = {m["idx"]: m for m in meta_data}
    classifications = sorted({m["classification"] for m in meta_data})

    per_class = {metric: {c: [] for c in classifications}
                 for metric in list(METRIC_TYPES) + ["similarity"]}
    per_class["f1_matrix"] = {c: dict(TP=0, FP=0, TN=0, FN=0) for c in classifications}
    totals = {metric: [] for metric in list(METRIC_TYPES) + ["similarity"]}
    totals["f1_matrix"] = dict(TP=0, FP=0, TN=0, FN=0)

    # Upstream drops repeated idx values and anything not in meta_data.
    seen, filtered = set(), []
    for pred in predictions:
        idx = pred["idx"]
        if idx not in seen and idx in meta_by_idx:
            seen.add(idx)
            filtered.append(pred)

    correct_ids = {m["idx"] for m in meta_data if m["classification"] == CORRECT_CLASSIFICATION}
    correct_preds = [p for p in filtered if p["idx"] in correct_ids]
    other_preds = [p for p in filtered if p["idx"] not in correct_ids]

    def _accumulate(res, classification, pool):
        for metric in METRIC_TYPES:
            per_class[metric][classification].extend(res[f"{metric}_list"])
            if pool:
                totals[metric].extend(res[f"{metric}_list"])
        for key in ("TP", "FP", "TN", "FN"):
            per_class["f1_matrix"][classification][key] += res["f1_matrix"][key]
            if pool:
                totals["f1_matrix"][key] += res["f1_matrix"][key]

    correct_response_acc = {}
    for pred in correct_preds:
        meta = meta_by_idx[pred["idx"]]
        res = eval_on_hallucination_step(meta["error_steps"], pred["labels"])
        _accumulate(res, meta["classification"], pool=False)
        correct_response_acc[pred["idx"]] = res["model_response_acc"]

    n_valid = 0
    for pred in other_preds:
        meta = meta_by_idx[pred["idx"]]
        classification = meta["classification"]
        n_valid += 1
        redundancy = use_redundancy_head and classification in ("redundency", "circular")
        res = eval_on_hallucination_step(meta["error_steps"], pred["labels"],
                                         redundancy_label=redundancy)
        _accumulate(res, classification, pool=True)

        correct_idx = "correct_" + pred["idx"][len(f"{classification}_"):]
        base = correct_response_acc.get(correct_idx)
        if base is not None and base != -1 and res["model_response_acc"] != -1:
            similarity = abs(res["model_response_acc"] - base)
            totals["similarity"].append(similarity)
            per_class["similarity"][classification].append(similarity)

    def _mean(values):
        return sum(values) / len(values) if values else -1

    total_results = {m: _mean(totals[m]) for m in list(METRIC_TYPES) + ["similarity"]}
    total_results.update(_prf(**{k.lower(): v for k, v in totals["f1_matrix"].items()}))

    by_class = {m: {c: _mean(v) for c, v in per_class[m].items()}
                for m in list(METRIC_TYPES) + ["similarity"]}
    for key in ("precision", "recall", "f1", "negative_precision", "negative_recall", "negative_f1"):
        by_class[key] = {}
    for classification in classifications:
        prf = _prf(**{k.lower(): v for k, v in per_class["f1_matrix"][classification].items()})
        for key, value in prf.items():
            by_class[key][classification] = value

    # The paper's three published categories, averaged over their member subcategories. The
    # synthetic `correct` control is a control, not a category, and never enters this average.
    by_category = {}
    for category in CATEGORIES:
        members = [c for c in classifications if CATEGORY_OF.get(c) == category]
        by_category[category] = {
            metric: _mean([by_class[metric][c] for c in members
                           if by_class[metric].get(c) not in (None, -1)])
            for metric in list(METRIC_TYPES) + ["f1", "similarity"]
        }
        by_category[category]["members"] = members

    n_scoreable = len(meta_data) - len(correct_ids)
    return {
        "total": total_results,
        "by_classification": by_class,
        "by_category": by_category,
        "validity_rate": n_valid / n_scoreable if n_scoreable else -1,
        "n_predictions_scored": len(filtered),
        "n_meta_rows": len(meta_data),
        "n_correct_control_rows": len(correct_ids),
        "used_redundancy_head": bool(use_redundancy_head),
        "adaptation_note": (
            "redundency/circular scored through the validity fallback because this scorer has no "
            "step_level_redundancy_labels head; upstream prefers that head when present."
            if not use_redundancy_head else ""
        ),
    }


# ── conditioning prompt + alignment (shared with the ProcessBench machinery) ──────────

def prmbench_prompt(row: dict, thinking_suffix: str = None) -> str:
    """Conditioning prompt for the teacher-forced telemetry pass.

    Deliberately routed through the SAME `data_loaders.math_prompt` + `/no_think` suffix as
    `spectral_utils.processbench.processbench_prompt`, so the frozen local head sees the same
    conditioning distribution it was developed under. Any deviation must be recorded in the
    run manifest.
    """
    from .data_loaders import math_prompt
    from .processbench import NO_THINK_SUFFIX
    suffix = NO_THINK_SUFFIX if thinking_suffix is None else thinking_suffix
    return math_prompt({"problem": row["question"]}) + (suffix or "")


def build_chain(steps, sep: str = STEP_SEP):
    """Re-exported from `spectral_utils.processbench` — identical contract, one implementation."""
    from .processbench import build_chain as _build_chain
    return _build_chain(steps, sep=sep)


def step_token_spans(tok, text: str, char_spans):
    """Re-exported from `spectral_utils.processbench` — identical contract, one implementation."""
    from .processbench import step_token_spans as _step_token_spans
    return _step_token_spans(tok, text, char_spans)


def assert_alignment(token_ids, spans, steps, strict: bool = True):
    """Re-exported from `spectral_utils.processbench` — identical contract, one implementation."""
    from .processbench import assert_alignment as _assert_alignment
    return _assert_alignment(token_ids, spans, steps, strict=strict)


# ── known-answer tests ───────────────────────────────────────────────────────────────

def smoke(check_corpus: bool = False) -> None:
    checks = 0

    # 1. The label convention: 1 == "step is VALID". A perfect scorer on a 4-step trace whose
    #    step 2 (1-indexed) is wrong therefore emits [1, 0, 1, 1].
    res = eval_on_hallucination_step([2], [1, 0, 1, 1])
    assert res["f1_matrix"] == {"TP": 3, "FP": 0, "TN": 1, "FN": 0}, res["f1_matrix"]
    assert res["correct_step_acc"] == 1.0 and res["wrong_step_acc"] == 1.0
    assert res["total_step_acc"] == 1.0 and res["first_error_acc"] == 1
    checks += 1

    # 2. The inverted scorer on the same row: every judgement wrong.
    res = eval_on_hallucination_step([2], [0, 1, 0, 0])
    assert res["f1_matrix"] == {"TP": 0, "FP": 1, "TN": 0, "FN": 3}, res["f1_matrix"]
    assert res["first_error_acc"] == 0
    checks += 1

    # 3. Out-of-range annotations are INERT, not an error and not a drop — 5 exceeds the trace.
    inert = eval_on_hallucination_step([5], [1, 1, 1])
    allcorrect = eval_on_hallucination_step([], [1, 1, 1])
    assert inert["f1_matrix"] == allcorrect["f1_matrix"] == {"TP": 3, "FP": 0, "TN": 0, "FN": 0}
    assert inert["first_error_acc"] is None, inert["first_error_acc"]
    checks += 1

    # 4. Empty error_steps (the multi_solutions case) has no first-error term to report.
    assert allcorrect["first_error_acc"] is None and allcorrect["wrong_step_acc"] == -1
    checks += 1

    # 5. redundancy_label flips the polarity of the positive class.
    flipped = eval_on_hallucination_step([2], [0, 1, 0, 0], redundancy_label=True)
    assert flipped["f1_matrix"] == {"TP": 3, "FP": 0, "TN": 1, "FN": 0}, flipped["f1_matrix"]
    checks += 1

    # 6. The aggregator: two error rows scored perfectly, plus one synthetic correct control.
    meta = [
        {"idx": "confidence_a", "source_idx": "a", "question": "q", "steps": ["s1", "s2"],
         "error_steps": [2], "classification": "confidence", "category": "soundness"},
        {"idx": "redundency_b", "source_idx": "b", "question": "q", "steps": ["s1", "s2"],
         "error_steps": [1], "classification": "redundency", "category": "simplicity"},
        {"idx": "correct_b", "source_idx": "b", "question": "q", "steps": ["s1", "s2"],
         "error_steps": [], "classification": "correct", "category": "control"},
    ]
    preds = [{"idx": "confidence_a", "labels": [1, 0]},
             {"idx": "redundency_b", "labels": [0, 1]},
             {"idx": "correct_b", "labels": [1, 1]}]
    out = prmbench_evaluate(preds, meta)
    assert out["total"]["total_step_acc"] == 1.0, out["total"]
    # the control row must NOT be pooled: 2 error rows x 2 steps = 4 pooled steps, not 6
    assert out["total"]["f1"] == 1.0, out["total"]["f1"]
    assert out["n_correct_control_rows"] == 1
    assert out["by_category"]["soundness"]["total_step_acc"] == 1.0
    assert out["used_redundancy_head"] is False and out["adaptation_note"]
    checks += 1

    # 7. Category map covers exactly the nine shipped classes plus the synthetic control.
    assert len(CATEGORY_OF) == 10 and set(CATEGORIES) <= set(CATEGORY_OF.values())
    checks += 1

    # 8. build_chain / step_token_spans stay identical to the ProcessBench implementations.
    from .processbench import build_chain as pb_build_chain
    assert build_chain(["a", "bb"]) == pb_build_chain(["a", "bb"])
    checks += 1

    if check_corpus:
        # 9. The published corpus statistics, reproduced from the real Hub download, INCLUDING
        #    the gap between the paper's headline count and what the official loader evaluates.
        meta_rows, diag = load_prmbench()
        assert diag["n_raw_rows"] == 6216, diag["n_raw_rows"]
        assert diag["n_raw_steps"] == 83456, diag["n_raw_steps"]        # the paper's number
        assert diag["n_duplicate_rows_dropped"] == 5, diag["n_duplicate_rows_dropped"]
        assert diag["n_steps_error_classes"] == 83371, diag["n_steps_error_classes"]
        # 165 multi_solutions rows carry an empty error_steps; 5 of them are the dropped dups.
        assert diag["n_rows_empty_error_steps"] == 160, diag["n_rows_empty_error_steps"]
        assert diag["n_rows_out_of_range_error_steps"] == 100, diag
        assert diag["counts_by_classification"][CORRECT_CLASSIFICATION] == 758, diag
        n_control = sum(1 for r in meta_rows if r["classification"] == CORRECT_CLASSIFICATION)
        assert n_control == 758 and diag["n_meta_rows"] == 6216 - 5 + 758, diag["n_meta_rows"]
        checks += 1

    print(f"prmbench.smoke: PASS ({checks} checks"
          f"{' incl. corpus' if check_corpus else ''})")


if __name__ == "__main__":
    import sys
    smoke(check_corpus="--corpus" in sys.argv)
