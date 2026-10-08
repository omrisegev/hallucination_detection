"""Validate a classifier output file against its input shard: python check_labels.py <in.jsonl> <out.jsonl>"""
import json
import sys

KEYS = {"redundency", "circular", "counterfactual", "step_contradiction", "domain_inconsistency", "confidence",
        "missing_condition", "deception", "calculation_error", "misread_problem", "wrong_method", "other"}
PRM = sorted(KEYS - {"calculation_error", "misread_problem", "wrong_method", "other"})


def main(inp, out):
    ids = [json.loads(l)["item_id"] for l in open(inp, encoding="utf-8")]
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    errs = []
    if [r.get("item_id") for r in rows] != ids:
        errs.append(f"item_id list/order mismatch: {len(rows)} rows vs {len(ids)} inputs")
    for r in rows:
        i = r.get("item_id")
        if r.get("category") not in KEYS: errs.append(f"{i}: bad category {r.get('category')}")
        if r.get("nearest_prmbench") not in PRM: errs.append(f"{i}: bad nearest_prmbench {r.get('nearest_prmbench')}")
        if r.get("prmbench_fit") not in ("good", "partial", "none"): errs.append(f"{i}: bad prmbench_fit")
        if r.get("category") in PRM and r.get("nearest_prmbench") != r.get("category"):
            errs.append(f"{i}: nearest_prmbench must equal a PRMBench category")
        if r.get("category") in PRM and r.get("prmbench_fit") == "none":
            errs.append(f"{i}: a PRMBench category cannot have fit 'none'")
        if r.get("confidence") not in ("high", "medium", "low"): errs.append(f"{i}: bad confidence")
        if not isinstance(r.get("disagree_with_step"), bool): errs.append(f"{i}: disagree_with_step must be bool")
        j = r.get("justification", "")
        if not j or len(j.split()) > 45: errs.append(f"{i}: justification missing or > 45 words")
    print("PASS" if not errs else "FAIL\n" + "\n".join(errs[:40]))
    return not errs


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
