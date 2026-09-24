"""Independent Agent A audit: raw shards + evaluator-only labels, never root summaries."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/family_tail_external_v1"
DEST = OUT / "independent_metrics"
PRIVATE = ROOT / "scratch/external_generalization_private"
CELLS = {"hard2verify_qwen3_8b": ("hard2verify", 200),
         "socratic_qwen3_8b": ("socratic", 2995),
         "socratic_qwq32b": ("socratic", 2995)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def count(y, p, keep):
    if not len(y) == len(p) == len(keep):
        raise ValueError("unaligned gold/prediction/mask")
    c = [0, 0, 0, 0]
    for truth, pred, included in zip(y, p, keep):
        if pred not in (0, 1) or truth not in (0, 1):
            raise ValueError("nonbinary raw decision")
        if included:
            c[{(1, 1): 0, (0, 1): 1, (0, 0): 2, (1, 0): 3}[int(truth), int(pred)]] += 1
    return c


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def calculate(c, benchmark):
    tp, fp, tn, fn = c
    def div(a, b):
        return a / b if b else -1.0
    pc, rc = div(tp, tp + fp), div(tp, tp + fn)
    pe, re = div(tn, tn + fn), div(tn, tn + fp)
    fc, fe = div(2 * pc * rc, pc + rc), div(2 * pe * re, pe + re)
    # Deliberately preserve author sentinel behavior; no new policy is substituted.
    primary = (fc + fe) / 2 if benchmark == "socratic" else (
        2 * rc * re / (rc + re) if rc >= 0 and re >= 0 and rc + re > 0 else
        0.0 if rc >= 0 and re >= 0 else None)
    return {"metric": primary, "confusion": c, "steps": sum(c),
            "precision_correct": pc, "recall_correct": rc, "f1_correct": fc,
            "precision_error": pe, "recall_error": re, "f1_error": fe}


def author_functions(path, names, namespace):
    module = ast.parse(path.read_text(encoding="utf8"))
    nodes = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if set(n.name for n in nodes) != set(names):
        raise ValueError("missing author function")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def selftest():
    c = count([1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
              [1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [True] * 10)
    assert c == [2, 1, 3, 4]
    s = calculate(c, "socratic")
    assert abs(s["f1_correct"] - 4 / 9) < 1e-15
    assert abs(s["f1_error"] - 6 / 11) < 1e-15
    assert abs(s["metric"] - (4 / 9 + 6 / 11) / 2) < 1e-15
    assert calculate(c, "hard2verify")["metric"] == 6 / 13
    assert count([1, 0], [0, 1], [False, True]) == [0, 1, 0, 0]
    assert calculate([0, 0, 2, 2], "socratic")["precision_correct"] == -1
    print("Independent synthetic confusion/component checks PASS; no external labels read.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    if args.synthetic_only:
        selftest()
        return
    started = time.perf_counter()
    allseal_path = OUT / "ALL_CELLS_SEALED.json"
    allseal = read(allseal_path)  # Absent means fail before reading any external labels.
    plan = read(OUT / "EXECUTION_PLAN.json")
    arms = plan["methods"]
    if len(arms) != 10 or set(allseal["seals"]) != set(CELLS):
        raise ValueError("missing registered arms/cell seals")
    lockpath = ROOT / "results/family_tail_transfer_v1/TRANSFER_LOCK_V1.json"
    if sha(lockpath) != plan["lock_sha256"] or allseal["lock_sha256"] != sha(lockpath):
        raise ValueError("source lock mismatch")
    ledger, rows, seal_info = [], {}, {}
    # Load and verify every prediction seal before any gold file.
    for cell, (_, expected) in CELLS.items():
        shardpaths = sorted((OUT / cell).glob("shard_*/*.record.json"))
        if len(shardpaths) != expected:
            raise ValueError("population count mismatch: " + cell)
        payloads, runids = {}, set()
        for path in shardpaths:
            if (path.parent / "WRITER.lock").exists():
                raise ValueError("active writer")
            record = read(path)
            uid = record["uid"]
            if uid in payloads:
                raise ValueError("duplicate uid")
            payloads[uid] = record["payload"]
            runids.add(record["run_identity"])
            ledger.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path)})
        seal = read(OUT / cell / "SEAL.json")
        if seal != allseal["seals"][cell] or seal["prediction_sha256"] != digest(payloads):
            raise ValueError("prediction seal mismatch")
        if len(runids) != 1 or set(seal["arms"]) != set(arms) or seal["answers"] != expected:
            raise ValueError("run/arm/seal mismatch")
        rows[cell] = payloads
        seal_info[cell] = {"sha256": sha(OUT / cell / "SEAL.json"),
                           "prediction_digest": digest(payloads), "answers": expected}
    source = PRIVATE / "sources"
    hp, sp = source / "hard2verify/utils.py", source / "prmeval_classified_task.py"
    from sklearn.metrics import recall_score
    author_hard = author_functions(hp, {"calculate_metrics"}, {"recall_score": recall_score})
    author_soc = author_functions(sp, {"evaluate_function", "eval_on_hallucination_step"}, {})
    totals, comparisons, metric_rows, labels_hashes = {}, [], [], {}
    oob_answers = oob_indices = checked = step_count = 0
    per_answer = []
    for cell, (benchmark, expected) in CELLS.items():
        gp = PRIVATE / "inputs/evaluator_only" / (benchmark + ".json")
        labels_hashes[benchmark] = sha(gp)
        goldlist = read(gp)
        gold = {r["uid"]: r for r in goldlist}
        if len(goldlist) != expected or len(gold) != expected or set(gold) != set(rows[cell]):
            raise ValueError("gold population mismatch")
        aggregate = {a: [0] * 4 for a in arms}
        categories = {}
        empty_steps = excluded_steps = 0
        for uid in sorted(gold):
            g, r = gold[uid], rows[cell][uid]
            if set(r["predictions"]) != set(arms):
                raise ValueError("arm missing")
            if not len(g["correct"]) == len(g["include"]) == len(r["nonempty"]):
                raise ValueError("mask alignment")
            empty_steps += sum(not x for x in r["nonempty"])
            excluded_steps += sum(not x for x in g["include"])
            category = str(g["category"])
            categories.setdefault(category, {a: [0] * 4 for a in arms})
            outside = g.get("out_of_range_error_indices", [])
            if any(1 <= i <= len(g["correct"]) for i in outside):
                raise ValueError("declared out-of-range annotation is in range")
            oob_answers += bool(outside)
            oob_indices += len(outside)
            for arm in arms:
                c = count(g["correct"], r["predictions"][arm], g["include"])
                aggregate[arm] = add(aggregate[arm], c)
                categories[category][arm] = add(categories[category][arm], c)
                per_answer.append({"cell": cell, "uid": uid, "arm": arm,
                                   "TP": c[0], "FP": c[1], "TN": c[2], "FN": c[3]})
            checked += 1
        output = {"answers": expected, "empty_steps": empty_steps, "excluded_steps": excluded_steps,
                  "arms": {a: calculate(c, benchmark) for a, c in aggregate.items()},
                  "categories": {cat: {a: calculate(c, benchmark) for a, c in byarm.items()}
                                 for cat, byarm in categories.items()}, "official_replay": {}}
        for arm in arms:
            ours = output["arms"][arm]
            if benchmark == "hard2verify":
                y, pred = [], []
                for uid, g in gold.items():
                    y.extend(int(v) for v, keep in zip(g["correct"], g["include"]) if keep)
                    pred.extend(int(v) for v, keep in zip(rows[cell][uid]["predictions"][arm], g["include"]) if keep)
                direct = author_hard["calculate_metrics"](pred, y)["balanced_f1_score"]
                if abs(direct - round(100 * ours["metric"], 2)) > 1e-10:
                    raise ValueError("independent Hard2 score vs author mismatch")
                output["official_replay"][arm] = {"pass": True, "rounded_percent": direct}
            else:
                meta, predictions = [], []
                for uid, g in gold.items():
                    if not all(g["include"]):
                        raise ValueError("Socratic author replay inclusion contract changed")
                    meta.append({"idx": uid, "classification": g["category"],
                                 "error_steps": [i + 1 for i, c in enumerate(g["correct"]) if not c]
                                 + g.get("out_of_range_error_indices", [])})
                    predictions.append({"idx": uid, "scores": {
                        "step_level_validity_labels": rows[cell][uid]["predictions"][arm]}})
                official = author_soc["evaluate_function"](predictions, meta)
                max_error = 0.0
                for ok, ik in {"precision": "precision_correct", "recall": "recall_correct",
                               "f1": "f1_correct", "negative_precision": "precision_error",
                               "negative_recall": "recall_error", "negative_f1": "f1_error"}.items():
                    max_error = max(max_error, abs(ours[ik] - official["total_hallucination_results"][ok]))
                    if set(official["hallucination_type_results"][ok]) != set(categories):
                        raise ValueError("author category coverage mismatch")
                    for cat in categories:
                        max_error = max(max_error, abs(output["categories"][cat][arm][ik]
                                        - official["hallucination_type_results"][ok][cat]))
                if max_error > 1e-12:
                    raise ValueError("independent Socratic components vs author mismatch")
                output["official_replay"][arm] = {"pass": True, "max_full_category_error": max_error}
            metric_rows.append({"cell": cell, "arm": arm, **{k: v for k, v in ours.items() if k != "confusion"}})
        for left, right in read(lockpath)["external_primary_contrasts"]:
            comparisons.append({"cell": cell, "left": left, "right": right,
                                "delta": output["arms"][left]["metric"] - output["arms"][right]["metric"]})
        output["steps"] = sum(aggregate[arms[0]])
        step_count += output["steps"]
        totals[cell] = output
        print("Agent A independently verified", cell, expected, "answers", output["steps"], "steps", flush=True)
    if checked != 6190 or len(metric_rows) != 30 or len(comparisons) != 18:
        raise ValueError("full-population accounting failed")
    DEST.mkdir(exist_ok=True)
    result = {"status": "PASS", "n_checked": checked, "n_total": 6190, "steps": step_count,
              "root_summaries_read": False, "other_reviewer_outputs_read": False,
              "source_lock_sha256": sha(lockpath), "all_cells_seal_sha256": sha(allseal_path),
              "seals": seal_info, "labels_sha256": labels_hashes,
              "official_sources": {str(p.relative_to(ROOT)): sha(p) for p in (hp, sp)},
              "oob_answer_backbone_records": oob_answers, "oob_index_backbone_records": oob_indices,
              "out_of_range_policy": "inert, never adds a step or changes its inclusion",
              "cells": totals, "contrast_point_deltas": comparisons,
              "command": sys.executable + " " + str(Path(__file__).resolve()),
              "script_sha256": sha(__file__), "elapsed_seconds": time.perf_counter() - started}
    (DEST / "INDEPENDENT_METRICS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (DEST / "RAW_HASHES.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf8")
    for filename, values in (("METRIC_ROWS.csv", metric_rows), ("ANSWER_COUNTS.csv", per_answer)):
        with (DEST / filename).open("w", encoding="utf8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    print("Independent metric audit PASS:", checked, "records,", len(metric_rows), "method/cell rows")


if __name__ == "__main__":
    main()
