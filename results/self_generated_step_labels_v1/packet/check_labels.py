"""Validate one judge's label files against the packet (format and consistency only).

Usage:  python packet/check_labels.py labels/<JUDGE_ID> [--shards A-B]

Checks every assigned shard: the output file exists, has one JSON object per input item in the
same order, every field has the required type and range, and the fields agree with each other.
It reads no answer key and reports no label statistics. Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = {"calculation", "misread_problem", "concept", "logic", "unjustified_claim",
         "incomplete", "final_answer", "other"}
FIELDS = ("item_id", "judge_id", "n_steps", "candidate_final_answer",
          "final_answer_matches_reference", "first_error_step", "error_steps",
          "first_error_type", "self_corrected", "reference_suspect", "confidence", "rationale")


def check_line(lab, item, judge_ids):
    errs = []
    missing = [f for f in FIELDS if f not in lab]
    extra = [f for f in lab if f not in FIELDS]
    if missing or extra:
        return [f"missing fields {missing}, unexpected fields {extra}"]
    n = len(item["steps"])
    if lab["item_id"] != item["item_id"]:
        errs.append(f"item_id {lab['item_id']!r} != input {item['item_id']!r} (order must match)")
    if not isinstance(lab["judge_id"], str) or not lab["judge_id"]:
        errs.append("judge_id must be a non-empty string")
    judge_ids.add(lab["judge_id"])
    if lab["n_steps"] != n:
        errs.append(f"n_steps {lab['n_steps']} != {n}")
    if lab["candidate_final_answer"] is not None and not isinstance(lab["candidate_final_answer"], str):
        errs.append("candidate_final_answer must be a string or null")
    for f in ("final_answer_matches_reference", "self_corrected", "reference_suspect"):
        if not isinstance(lab[f], bool):
            errs.append(f"{f} must be true/false")
    fe, es, ty = lab["first_error_step"], lab["error_steps"], lab["first_error_type"]
    if not isinstance(fe, int) or isinstance(fe, bool) or not (fe == -1 or 0 <= fe < n):
        errs.append(f"first_error_step {fe!r} must be -1 or in 0..{n - 1}")
    elif not isinstance(es, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in es):
        errs.append("error_steps must be a list of integers")
    elif es != sorted(set(es)) or any(not 0 <= x < n for x in es):
        errs.append(f"error_steps {es} must be sorted, unique and in 0..{n - 1}")
    elif fe == -1:
        if es:
            errs.append("error_steps must be empty when first_error_step is -1")
        if ty != "none":
            errs.append("first_error_type must be 'none' when first_error_step is -1")
        if lab["self_corrected"] is True:
            errs.append("self_corrected must be false when first_error_step is -1")
        if lab["final_answer_matches_reference"] is False and lab["reference_suspect"] is False:
            errs.append("final answer does not match the reference, so first_error_step cannot be -1 "
                        "(unless reference_suspect is true)")
    else:
        if not es or es[0] != fe:
            errs.append(f"smallest error_steps element must equal first_error_step {fe}")
        if ty not in TYPES:
            errs.append(f"first_error_type {ty!r} must be one of {sorted(TYPES)}")
    if lab["confidence"] not in ("high", "medium", "low"):
        errs.append("confidence must be high, medium or low")
    r = lab["rationale"]
    if not isinstance(r, str) or not r.strip():
        errs.append("rationale must be a non-empty string")
    elif len(r.split()) > 80:
        errs.append(f"rationale has {len(r.split())} words (limit 50, hard cap 80)")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label_dir")
    ap.add_argument("--shards", default=None, help="inclusive range A-B, e.g. 0-5")
    a = ap.parse_args()
    manifest = json.load(open(os.path.join(HERE, "PACKET_MANIFEST.json"), encoding="utf-8"))
    names = [s["shard"] for s in manifest["shards"]]
    if a.shards:
        lo, hi = (int(x) for x in a.shards.split("-"))
        names = [n for n in names if lo <= int(n[6:9]) <= hi]
    problems, judge_ids, n_items = [], set(), 0
    for name in names:
        items = [json.loads(l) for l in open(os.path.join(HERE, "shards", name), encoding="utf-8")]
        out = os.path.join(a.label_dir, name)
        if not os.path.exists(out):
            problems.append(f"{name}: output file missing")
            continue
        raw = [l for l in open(out, encoding="utf-8").read().split("\n") if l.strip()]
        if len(raw) != len(items):
            problems.append(f"{name}: {len(raw)} lines, expected {len(items)}")
            continue
        for k, (line, item) in enumerate(zip(raw, items)):
            try:
                lab = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"{name} line {k + 1}: not valid JSON ({e})")
                continue
            if not isinstance(lab, dict):
                problems.append(f"{name} line {k + 1}: not a JSON object")
                continue
            problems += [f"{name} line {k + 1} ({item['item_id']}): {m}"
                         for m in check_line(lab, item, judge_ids)]
            n_items += 1
    if len(judge_ids) > 1:
        problems.append(f"more than one judge_id in this directory: {sorted(judge_ids)}")
    if problems:
        print(f"FAIL: {len(problems)} problem(s) in {len(names)} shard(s)")
        for p in problems[:200]:
            print("  " + p)
        sys.exit(1)
    print(f"PASS: {len(names)} shard(s), {n_items} item(s) checked")


if __name__ == "__main__":
    main()
