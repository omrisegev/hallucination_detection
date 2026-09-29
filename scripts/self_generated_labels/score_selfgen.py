"""Label-blind scoring of the self-generated packet answers with the FROZEN external methods.

Applies `spectral_utils.external_generalization.scoring.score_answer` (seven locked arms) with
the frozen source bundle used for Hard2Verify / Socratic, unchanged: no refit, no recalibration.
Telemetry is the generation-time rich save of the evdrop cells (Qwen3, greedy, non-thinking):
raw top-50 log-probs, top-15 entropies, chosen-token surprisal, full-vocab logsumexp. These are
the model's own distributions over its own tokens, i.e. on-policy.

Step token spans come from the judge-visible step character spans: a token belongs to the step in
which its first character lies (`token_offsets` are character offsets into `full_text`; the final
end-of-turn token has no offset and belongs to no step; `---` separators belong to no step).

Reads only cell / src_idx / step spans from the item key, never a label. Writes
results/self_generated_step_labels_v1/scoring/SCORES.jsonl and SCORING_PROVENANCE.json.
"""
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from spectral_utils.external_generalization.scoring import score_answer, ALL_ARMS  # noqa: E402

MAIN = os.environ.get("HD_MAIN_CHECKOUT", r"C:\Users\omris\TAU\hallucination_detection")
BUNDLE = os.path.join(MAIN, "results/lsml_external_generalization_v1/evaluation/source/BUNDLE.json")
BUNDLE_SHA256 = "b96939fcd0c3fb0db24e22caa6e902d202d1712ae7129a0c0ad416e2cd309cef"  # METHOD_FREEZE.json
ROOT = "results/self_generated_step_labels_v1"
PKL = {
    "evdrop_gsm8k_qwen3_4b": "raw_gsm8k_T0.0.pkl", "evdrop_gsm8k_qwen3_8b": "raw_gsm8k_T0.0.pkl",
    "evdrop_math_qwen3_4b": "raw_math_T0.0.pkl", "evdrop_math_qwen3_8b": "raw_math_T0.0.pkl",
}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def token_spans(offsets, char_spans):
    starts = np.asarray([a for a, _ in offsets])
    if (np.diff(starts) < 0).any():
        raise ValueError("token offsets are not monotone")
    out = []
    for s, e in char_spans:
        idx = np.flatnonzero((starts >= s) & (starts < e))
        out.append([int(idx[0]), int(idx[-1]) + 1] if len(idx) else [0, 0])
        if len(idx) and idx[-1] - idx[0] + 1 != len(idx):
            raise ValueError("non-contiguous step tokens")
    return out


def main():
    if sha256(BUNDLE) != BUNDLE_SHA256:
        raise SystemExit("frozen bundle hash mismatch")
    bundle = json.load(open(BUNDLE))
    key = [json.loads(l) for l in open(f"{ROOT}/private/ITEM_KEY.jsonl", encoding="utf-8")]
    own = [{k: r[k] for k in ("item_id", "cell", "src_idx", "step_char_spans")}
           for r in key if r["source"] == "own"]
    os.makedirs(f"{ROOT}/scoring", exist_ok=True)
    out_path = f"{ROOT}/scoring/SCORES.jsonl"
    t0 = time.time()
    n = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for cell in sorted(PKL):
            items = [r for r in own if r["cell"] == cell]
            d = pickle.load(open(os.path.join(MAIN, "dataset_cache/repgrid", cell, PKL[cell]), "rb"))
            for r in sorted(items, key=lambda r: r["item_id"]):
                c = d[r["src_idx"]]["candidates"][0]
                spans = token_spans(c["token_offsets"], r["step_char_spans"])
                row = {"gen_token_ids": c["gen_token_ids"], "top_k_logprobs": c["top_k_logprobs_raw"],
                       "token_entropies": c["token_entropies"],
                       "token_spilled_energies": c["token_spilled_energies"],
                       "token_logsumexp": c["token_logsumexp"], "step_token_spans": spans}
                res = score_answer(row, bundle)
                res.update(item_id=r["item_id"], cell=cell, step_token_spans=spans)
                f.write(json.dumps(res) + "\n")
                n += 1
            print(cell, len(items), f"{time.time() - t0:.0f}s", flush=True)
            del d
    code = "spectral_utils/external_generalization"
    prov = {"bundle": BUNDLE, "bundle_sha256": BUNDLE_SHA256, "arms": list(ALL_ARMS), "answers": n,
            "telemetry": "generation-time rich save (top_k_logprobs_raw), evdrop cells",
            "span_rule": "token belongs to the step containing its first character",
            "scores_sha256": sha256(out_path), "elapsed_s": round(time.time() - t0, 1),
            "code_sha256": {os.path.join(dp, fn).replace(os.sep, "/"): sha256(os.path.join(dp, fn))
                            for dp, _, fns in os.walk(code) for fn in sorted(fns) if fn.endswith(".py")},
            "labels_read": False}
    json.dump(prov, open(f"{ROOT}/scoring/SCORING_PROVENANCE.json", "w"), indent=1)
    print(json.dumps({k: prov[k] for k in ("answers", "elapsed_s", "scores_sha256")}))


if __name__ == "__main__":
    main()
