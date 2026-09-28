"""Build the blinded judge packet for self-generated step labels (v1).

Inputs
  * results/self_generated_step_labels_v1/private/own_answers.jsonl  (extract_own_answers.py)
  * Qwen/ProcessBench gsm8k + math splits (human first-error labels; judge validation items)
  * openai/gsm8k test and the full MATH test split (reference solutions / final answers)

Composition (fixed before any judge output exists; see docs/experiments/SELF_GENERATED_STEP_LABELS_V1.md)
  * every own answer graded wrong at inference time that is not truncated at the token cap
  * OWN_CORRECT_PER_CELL randomly drawn own answers graded correct (not truncated), per cell
  * PB_PER_STRATUM ProcessBench items per (subset x {error, no-error}) as judge validation

Outputs (under results/self_generated_step_labels_v1/)
  packet/shards/shard_XXX.jsonl   what the judges see: item_id, problem, reference, steps
  private/ITEM_KEY.jsonl          what they must not see: source, cell, labels, step spans
  packet/PACKET_MANIFEST.json     counts, seed, rule id, per-shard sha256
The shuffle is seeded, so any prefix of shards is a random sample of the whole packet.
"""
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from spectral_utils.self_generated_steps import SEGMENTATION_RULE_ID, segment_answer  # noqa: E402
from spectral_utils.data_loaders import _extract_boxed  # noqa: E402
from spectral_utils.localization_data import load_math_full  # noqa: E402

ROOT = "results/self_generated_step_labels_v1"
SEED = 20260928
OWN_CORRECT_PER_CELL = 100
PB_PER_STRATUM = 60
SHARD_SIZE = 25
PACKET_VERSION = "self-generated-step-labels-packet-v1"

_ws = re.compile(r"\s+")


def norm(q: str) -> str:
    return _ws.sub(" ", q).strip()


def gsm8k_reference(answer_field: str):
    sol, final = answer_field.split("####")
    sol = re.sub(r"<<[^>]*>>", "", sol).strip()
    return sol, final.strip().replace(",", "")


def math_reference(solution: str):
    final = _extract_boxed(solution)
    if final is None:
        raise ValueError("MATH solution without \\boxed{}")
    return solution.strip(), final


def load_references():
    from datasets import load_dataset
    refs = {}
    for r in load_dataset("openai/gsm8k", "main", split="test"):
        refs[("gsm8k", norm(r["question"]))] = gsm8k_reference(r["answer"])
    for r in load_math_full(n_samples=100000):
        refs[("math", norm(r["problem"]))] = math_reference(r["solution"])
    return refs


def own_items(rng, refs):
    rows = [json.loads(l) for l in open(f"{ROOT}/private/own_answers.jsonl", encoding="utf-8")]
    items, excluded = [], {"truncated": 0}
    by_cell = {}
    for r in rows:
        if r["n_tokens"] >= r["max_new"]:
            excluded["truncated"] += 1
            continue
        by_cell.setdefault(r["cell"], []).append(r)
    for cell in sorted(by_cell):
        cand = by_cell[cell]
        wrong = [r for r in cand if not r["auto_label"]]
        right = sorted((r for r in cand if r["auto_label"]), key=lambda r: r["src_idx"])
        chosen = wrong + rng.sample(right, OWN_CORRECT_PER_CELL)
        for r in chosen:
            ds = r["dataset"]
            ref_sol, ref_final = refs[(ds, norm(r["question"]))]
            body_start, spans = segment_answer(r["full_text"])
            items.append({
                "view": {"problem": r["question"], "reference_solution": ref_sol,
                         "reference_final_answer": ref_final,
                         "steps": [r["full_text"][s:e] for s, e in spans]},
                "key": {"source": "own", "cell": cell, "model": r["model"], "dataset": ds,
                        "src_idx": r["src_idx"], "auto_label_correct": r["auto_label"],
                        "segmentation_rule": SEGMENTATION_RULE_ID, "body_start": body_start,
                        "step_char_spans": spans},
            })
    return items, excluded


def pb_items(rng, refs):
    from datasets import load_dataset
    items = []
    for subset in ("gsm8k", "math"):
        ds = load_dataset("Qwen/ProcessBench", split=subset)
        rows = sorted((dict(r) for r in ds), key=lambda r: r["id"])
        for has_error in (True, False):
            pool = [r for r in rows if (int(r["label"]) != -1) == has_error]
            for r in rng.sample(pool, PB_PER_STRATUM):
                ref_sol, ref_final = refs[(subset, norm(r["problem"]))]
                items.append({
                    "view": {"problem": r["problem"], "reference_solution": ref_sol,
                             "reference_final_answer": ref_final, "steps": list(r["steps"])},
                    "key": {"source": "processbench", "subset": subset, "pb_id": r["id"],
                            "generator": r["generator"], "pb_label": int(r["label"]),
                            "pb_final_answer_correct": bool(r["final_answer_correct"])},
                })
    return items


def main():
    rng = random.Random(SEED)
    refs = load_references()
    own, excluded = own_items(rng, refs)
    pb = pb_items(rng, refs)
    items = own + pb
    rng.shuffle(items)

    os.makedirs(f"{ROOT}/packet/shards", exist_ok=True)
    shards, key_lines = [], []
    for s in range(0, len(items), SHARD_SIZE):
        name = f"shard_{s // SHARD_SIZE:03d}.jsonl"
        lines = []
        for j, it in enumerate(items[s:s + SHARD_SIZE]):
            item_id = f"J{s + j + 1:04d}"
            v = it["view"]
            lines.append(json.dumps({
                "item_id": item_id, "problem": v["problem"],
                "reference_solution": v["reference_solution"],
                "reference_final_answer": v["reference_final_answer"],
                "steps": [{"index": i, "text": t} for i, t in enumerate(v["steps"])],
            }, ensure_ascii=False))
            key_lines.append(json.dumps({"item_id": item_id, "shard": name,
                                         "n_steps": len(v["steps"]), **it["key"]},
                                        ensure_ascii=False))
        body = "\n".join(lines) + "\n"
        path = f"{ROOT}/packet/shards/{name}"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        shards.append({"shard": name, "n_items": len(lines),
                       "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()})
    with open(f"{ROOT}/private/ITEM_KEY.jsonl", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(key_lines) + "\n")

    count = {}
    for it in items:
        k = it["key"]
        tag = (f"own/{k['cell']}/{'correct' if k['auto_label_correct'] else 'wrong'}"
               if k["source"] == "own" else
               f"processbench/{k['subset']}/{'error' if k['pb_label'] != -1 else 'no_error'}")
        count[tag] = count.get(tag, 0) + 1
    # The judge-visible manifest carries no composition (base rates would leak a prior).
    manifest = {"packet_version": PACKET_VERSION, "shard_size": SHARD_SIZE,
                "n_items": len(items), "n_shards": len(shards), "shards": shards}
    with open(f"{ROOT}/packet/PACKET_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    private = {"packet_version": PACKET_VERSION, "seed": SEED,
               "segmentation_rule": SEGMENTATION_RULE_ID,
               "own_correct_per_cell": OWN_CORRECT_PER_CELL, "pb_per_stratum": PB_PER_STRATUM,
               "excluded_own": excluded, "composition": dict(sorted(count.items()))}
    with open(f"{ROOT}/private/COMPOSITION.json", "w", encoding="utf-8") as f:
        json.dump(private, f, indent=1)
    print(json.dumps({**private, "n_items": len(items), "n_shards": len(shards)}, indent=1))


if __name__ == "__main__":
    main()
