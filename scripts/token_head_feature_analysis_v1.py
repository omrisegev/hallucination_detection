"""Token-head feature analysis v1 — descriptive/diagnostic only.

Replays the frozen Unified-28 causal bank (`base7_full28`, loaded via
`load_unified28_model`, SHA-verified) over locally cached per-token telemetry
and persists, for every trace:

  * the 9 raw risk-oriented base streams (T x 9, model-reference order);
  * the 28 oriented model coordinates the token head actually fuses (T x 28);
  * the frozen per-token evidence curve (T,);
  * a token-level label mask (ProcessBench first-error step tokens, or
    RAGTruth annotated hallucination-span tokens).

Stage `analyze` then computes correlation / eigen-structure / conditional
dependence diagnostics and the descriptive comparison of fusion weight
vectors (frozen IU-PCR, refit IU-PCR, SU-PCR reproduction, SDSF structured,
simple average, first PC).

Token labels are used for conditioning and descriptive diagnostics only.
Nothing here changes, selects, or promotes any frozen method.

Usage:
  python scripts/token_head_feature_analysis_v1.py compute [--dataset gsm8k|math|ragtruth|all] [--limit N]
  python scripts/token_head_feature_analysis_v1.py analyze
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.fair_comparisons.twentyfour import load_unified28_model
from spectral_utils.unified_causal_iu import base_matrix, causal_feature_matrix
from spectral_utils.unified_causal_evaluation import first_error_mask, final_wrong

OUT_DIR = os.path.join(REPO, "results", "token_head_feature_analysis_v1")
PB_DIR = os.path.join(REPO, "dataset_cache", "repgrid", "pb_llama31_8b")
RT_PILOT = os.path.join(REPO, "dataset_cache", "ragtruth_ec_pilot", "ragtruth_ec_test.pkl")
MATH_SUBSAMPLE = 300
SEED = 20260828


def _load_pkl(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _per_row_arrays(model, row):
    """Raw base streams, oriented 28 coordinates, and frozen evidence for one trace."""
    raw = base_matrix(row, model.reference.names)                  # T x 9 (raw values)
    full = causal_feature_matrix(row, model.reference, raw_base=raw)  # T x 252
    selected = full[:, model.feature_indices]
    clean = np.where(np.isfinite(selected), selected, model.feature_medians)
    z = (clean - model.feature_centres) / model.feature_scales * model.feature_signs
    evidence = (z @ model.weights - model.evidence_centre) / model.evidence_scale
    return raw, z, evidence, full


def _verify_against_model(model, full, evidence):
    reference_evidence = model.evidence_from_feature_matrix(full)
    if not np.allclose(evidence, reference_evidence, atol=1e-10, rtol=0.0):
        raise AssertionError("vectorized evidence deviates from the frozen scorer")


def _save_dataset(name, metas, raws, zs, evids, masks):
    os.makedirs(OUT_DIR, exist_ok=True)
    lengths = [len(evid) for evid in evids]
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    np.savez_compressed(
        os.path.join(OUT_DIR, f"tokens_{name}.npz"),
        offsets=offsets,
        raw=np.concatenate(raws).astype(np.float32) if raws else np.zeros((0, 9), np.float32),
        z=np.concatenate(zs).astype(np.float32) if zs else np.zeros((0, 28), np.float32),
        evidence=np.concatenate(evids).astype(np.float32) if evids else np.zeros(0, np.float32),
        mask=np.concatenate(masks).astype(np.int8) if masks else np.zeros(0, np.int8),
    )
    with open(os.path.join(OUT_DIR, f"meta_{name}.json"), "w", encoding="utf-8") as handle:
        json.dump(metas, handle)
    print(f"[saved] {name}: {len(metas)} traces, {int(offsets[-1])} tokens", flush=True)


def compute_processbench(model, subset, limit=None):
    path = os.path.join(PB_DIR, f"processbench_{subset}.pkl")
    rows = _load_pkl(path)
    keys = sorted(rows)
    if subset == "math":
        rng = np.random.default_rng(SEED)
        keys = sorted(rng.choice(len(keys), size=min(MATH_SUBSAMPLE, len(keys)), replace=False).tolist())
        keys = [sorted(rows)[index] for index in keys]
    if limit:
        keys = keys[:limit]
    metas, raws, zs, evids, masks = [], [], [], [], []
    start = time.time()
    verified = False
    for count, key in enumerate(keys):
        row = rows[key]
        if row.get("align_diag", {}).get("problems"):
            continue
        raw, z, evidence, full = _per_row_arrays(model, row)
        if not verified:
            _verify_against_model(model, full, evidence)
            verified = True
            print("[verify] vectorized evidence matches frozen scorer bit-close", flush=True)
        n_tokens = len(evidence)
        label = int(row.get("label", -1))
        spans = [list(map(int, span)) for span in (row.get("step_token_spans") or []) if span is not None]
        metas.append({
            "id": str(row.get("id", key)),
            "label": label,
            "final_wrong": final_wrong(row),
            "n_tokens": n_tokens,
            "n_steps": len(spans),
            "step_spans": spans,
            "error_span": spans[label] if 0 <= label < len(spans) else None,
        })
        raws.append(raw)
        zs.append(z)
        evids.append(evidence)
        masks.append(first_error_mask(row, n_tokens))
        if (count + 1) % 25 == 0:
            elapsed = time.time() - start
            print(f"[{subset}] {count + 1}/{len(keys)} traces, {elapsed:.0f}s elapsed", flush=True)
        if (count + 1) % 100 == 0:
            _save_dataset(subset, metas, raws, zs, evids, masks)
    _save_dataset(subset, metas, raws, zs, evids, masks)


def compute_ragtruth(model, limit=None):
    rows = _load_pkl(RT_PILOT)
    keys = sorted(key for key in rows if key.endswith("::full") or key.endswith("::noctx"))
    if limit:
        keys = keys[:limit]
    metas, raws, zs, evids, masks = [], [], [], [], []
    start = time.time()
    for count, key in enumerate(keys):
        row = rows[key]
        raw, z, evidence, _ = _per_row_arrays(model, row)
        n_tokens = len(evidence)
        spans = [list(map(int, span)) for span in (row.get("span_token_spans") or [])]
        mask = np.zeros(n_tokens, dtype=np.int8)
        for begin, end in spans:
            mask[max(0, begin):min(n_tokens, end)] = 1
        metas.append({
            "id": key,
            "condition": str(row.get("condition")),
            "task_type": str(row.get("task_type")),
            "response_label": bool(row.get("response_label")),
            "n_tokens": n_tokens,
            "halluc_spans": spans,
            "label_types": [str(item.get("label_type")) for item in (row.get("span_labels") or [])],
        })
        raws.append(raw)
        zs.append(z)
        evids.append(evidence)
        masks.append(mask)
        if (count + 1) % 25 == 0:
            print(f"[ragtruth] {count + 1}/{len(keys)}, {time.time() - start:.0f}s", flush=True)
    _save_dataset("ragtruth_pilot", metas, raws, zs, evids, masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["compute", "analyze"])
    parser.add_argument("--dataset", default="all",
                        choices=["gsm8k", "math", "ragtruth", "all"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.stage == "compute":
        model, sha = load_unified28_model(REPO)
        print(f"[model] frozen base7_full28 loaded, artifact sha256 {sha[:16]}...", flush=True)
        if args.dataset in ("gsm8k", "all"):
            compute_processbench(model, "gsm8k", limit=args.limit)
        if args.dataset in ("math", "all"):
            compute_processbench(model, "math", limit=args.limit)
        if args.dataset in ("ragtruth", "all"):
            compute_ragtruth(model, limit=args.limit)
    else:
        from scripts.token_head_analysis_stage2 import run_analysis
        run_analysis()


if __name__ == "__main__":
    main()
