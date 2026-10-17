"""Extract text-only fields of the four evdrop Qwen3 self-generated cells into one JSONL.

Reads the raw rich-save pickles from the MAIN checkout (LFS objects, not in this worktree) and
writes results/self_generated_step_labels_v1/private/own_answers.jsonl: cell, source index,
question, gold row, full_text, the inference-time grade, token count and the generation cap.
No telemetry is copied; the step-to-token mapping is recomputed later from the raw pickles.
"""
import json
import os
import pickle

MAIN = os.environ.get("HD_MAIN_CHECKOUT", r"C:\Users\omris\TAU\hallucination_detection")
CELLS = {
    "evdrop_gsm8k_qwen3_4b": ("raw_gsm8k_T0.0.pkl", 1024),
    "evdrop_gsm8k_qwen3_8b": ("raw_gsm8k_T0.0.pkl", 1024),
    "evdrop_math_qwen3_4b": ("raw_math_T0.0.pkl", 2048),
    "evdrop_math_qwen3_8b": ("raw_math_T0.0.pkl", 2048),
}
OUT = "results/self_generated_step_labels_v1/private/own_answers.jsonl"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for cell, (pkl, max_new) in CELLS.items():
            man = json.load(open(os.path.join(MAIN, "dataset_cache/repgrid", cell, "manifest.json")))
            assert man["max_new"] == max_new and man["temps"] == [0.0] and man["k"] == 1, cell
            d = pickle.load(open(os.path.join(MAIN, "dataset_cache/repgrid", cell, pkl), "rb"))
            for idx in sorted(d):
                row = d[idx]
                assert len(row["candidates"]) == 1, (cell, idx)
                c = row["candidates"][0]
                f.write(json.dumps({
                    "cell": cell, "src_idx": int(idx), "model": man["model"],
                    "dataset": man["dataset"], "question": row["question"],
                    "gold_row": {k: v for k, v in row["gold_row"].items()
                                 if isinstance(v, (str, int, float))},
                    "full_text": c["full_text"], "auto_label": bool(c["label"]),
                    "n_tokens": len(c["gen_token_ids"]), "max_new": max_new,
                }, ensure_ascii=False) + "\n")
                n += 1
            print(cell, len(d), flush=True)
            del d
    print("wrote", n, "rows to", OUT)


if __name__ == "__main__":
    main()
