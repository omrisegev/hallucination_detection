"""Build the classifier input files for own_error_types_v1 (no scores, no judge rationales, no cell ids).

Reads the blind judge packet and FINAL_LABELS_V1.jsonl from the self-generated-step-labels worktree
(read-only) and writes inputs/ITEMS_ALL.jsonl, pass1/pass2 shard inputs, and inputs/PASS2_IDS.json.
"""
import json
from pathlib import Path

import numpy as np

SRC = Path(r"C:\Users\omris\TAU\hallucination_detection\.worktrees\self-generated-step-labels-v1"
           r"\results\self_generated_step_labels_v1")
OUT = Path(__file__).resolve().parent
SEED, N_PASS2, P1_SHARDS, P2_SHARDS = 20261007, 120, 12, 4


def main():
    final = {}
    for line in open(SRC / "FINAL_LABELS_V1.jsonl", encoding="utf-8"):
        x = json.loads(line)
        if x["first_error_step"] is not None and x["first_error_step"] >= 0:
            final[x["item_id"]] = x["first_error_step"]
    items = {}
    for shard in sorted((SRC / "packet/shards").glob("shard_*.jsonl")):
        for line in open(shard, encoding="utf-8"):
            x = json.loads(line)
            if x["item_id"] in final:
                items[x["item_id"]] = {"item_id": x["item_id"], "problem": x["problem"],
                                       "reference_solution": x["reference_solution"],
                                       "reference_final_answer": x["reference_final_answer"],
                                       "first_error_step": final[x["item_id"]], "steps": x["steps"]}
    assert len(final) == 417 and set(items) == set(final), (len(final), len(items))
    for it in items.values():
        assert 0 <= it["first_error_step"] < len(it["steps"])
    ids = sorted(items)
    (OUT / "inputs").mkdir(exist_ok=True)
    with open(OUT / "inputs/ITEMS_ALL.jsonl", "w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps(items[i], ensure_ascii=False) + "\n")
    pass2 = sorted(np.random.default_rng(SEED).choice(ids, N_PASS2, replace=False).tolist())
    json.dump({"seed": SEED, "n": N_PASS2, "item_ids": pass2}, open(OUT / "inputs/PASS2_IDS.json", "w"), indent=1)
    for tag, sel, k in (("pass1", ids, P1_SHARDS), ("pass2", pass2, P2_SHARDS)):
        for s, chunk in enumerate(np.array_split(np.array(sel), k)):
            with open(OUT / f"inputs/{tag}_in_{s:02d}.jsonl", "w", encoding="utf-8") as f:
                for i in chunk:
                    f.write(json.dumps(items[str(i)], ensure_ascii=False) + "\n")
    print(len(ids), "items;", len(pass2), "pass2 ids")


if __name__ == "__main__":
    main()
