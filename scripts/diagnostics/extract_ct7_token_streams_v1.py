#!/usr/bin/env python
"""Item 3 extraction (2026-09-23): CT7's seven streams at TOKEN level for the whole localization
population, aligned to the roster (`JOINED.json` order, `token_offsets` = cumsum of record tokens,
`step_spans` 0-based within the answer, as in `results/token_probability_fusion_v1/TOKEN_MATRICES.npz`).

Writes `CT7_TOKEN_MATRICES.npz` with
  tokens [N x 7] float32   H0lim, ve0, ve0.75, ve1, H0lim_prefix_innovation, bocpd_residual, chosen_std_excess
  valid  [N x 7] bool      token 0 of the innovation is invalid; everything else valid
  token_offsets [answers+1], step_spans [steps x 2], step0_token_mask [N], channels, provenance.

Exactness gates (asserts; the run stops on failure), see docs/experiments/CT7_TOKEN_LSML_V1.md:
  (i)  masked Top10 of the five bank columns, cast to float32, equals the frozen bank extraction
       `length_explicit_ct7_v1/bank/<cell>.npz['top10']` exactly (when --bank-dir is given);
  (ii) the answer-standardized masked Top10 of the BOCPD column replays CT7's view 5
       (`profiles.npy[:, 5]`) to 1e-8 for the verbatim temporal source, 1e-6 for the fallback
       recomputation (when --profiles is given).
No labels are read. CPU only; about two hours for the nine cells from the raw pickles.

    python -B scripts/diagnostics/extract_ct7_token_streams_v1.py --source-root <repo with dataset_cache> \
        --temporal <.../results/temporal_context_data_v1> --bank-dir <.../results/length_explicit_ct7_v1/bank> \
        --profiles <.../ct7_profiles_v1/profiles.npy> --out results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import ct7_levers_common as L  # noqa: E402

L.ensure_spectral_package()
from spectral_utils.aligned_context_predictors import bocpd_mean  # noqa: E402
from spectral_utils.ct7_token_streams import (  # noqa: E402
    HAZARD, STREAMS, answer_streams, bocpd_residual_temporal_recipe, masked_step_top10, step0_token_mask,
)
from spectral_utils.digitfree_broad50 import masked_answer_standardize  # noqa: E402

SCHEMA = "ct7-token-matrices-v1"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_temporal(temporal: Path):
    manifest = json.loads((temporal / "MANIFEST.json").read_text(encoding="utf8"))
    feats = np.load(temporal / "features.npy", mmap_mode="r")
    cols = manifest["banks"]["innovation5"]
    meta = json.loads((temporal / "METADATA.json").read_text(encoding="utf8"))
    return feats, cols, meta, manifest


def bocpd_verbatim(feats, cols, meta_i):
    """Step 420 / run_length_calibrated_streams_v1._bocpd_one, minus the readout."""
    a = meta_i["offset"]; T = meta_i["tokens"]
    raw = np.asarray(feats[a:a + T], float)[:, cols]
    z = (raw - np.asarray(meta_i["mean"], float)) / np.asarray(meta_i["scale"], float)
    return (z - bocpd_mean(z, hazard=HAZARD)).mean(axis=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True, help="checkout holding dataset_cache/ and results/localization_full_benchmark_v3")
    p.add_argument("--temporal", help="temporal_context_data_v1 directory (verbatim BOCPD source)")
    p.add_argument("--bank-dir", help="length_explicit_ct7_v1/bank directory for gate (i)")
    p.add_argument("--profiles", help="ct7_profiles_v1/profiles.npy for gate (ii)")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", type=int, default=0, help="answers per cell (0 = all)")
    args = p.parse_args()
    source = Path(args.source_root).resolve(); out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    sys.path.insert(0, str(ROOT))
    from scripts import run_direct_probability_temporal as e
    e.old.configure_source_root(source)
    try:
        from scripts.run_fusion_independence_atlas_v1 import RAW_SOURCE_HASHES
    except Exception:
        RAW_SOURCE_HASHES = {}

    roster = source / "results/localization_full_benchmark_v3/evaluation"
    records = json.loads((roster / "JOINED.json").read_text(encoding="utf8"))["records"]
    offsets = np.asarray(np.load(roster / "JOINED.npz")["offsets"], int)
    n = len(records); total_steps = int(offsets[-1])
    token_counts = np.asarray([int(r["tokens"]) for r in records]); token_offsets = np.concatenate([[0], np.cumsum(token_counts)])
    N = int(token_offsets[-1])
    tokens = np.zeros((N, 7), np.float64); valid = np.zeros((N, 7), bool)
    spans_all = np.zeros((total_steps, 2), np.int32); step0 = np.zeros(N, bool); done = np.zeros(n, bool)

    temporal = None
    if args.temporal:
        temporal = load_temporal(Path(args.temporal))
        assert len(temporal[2]) == n, "temporal METADATA does not match the roster"
    by_cell = {}
    for i, r in enumerate(records):
        by_cell.setdefault(r["cell"], []).append(i)
    sources = {}
    for cell, path, kind, dataset in e.source_specs():
        indexes = by_cell.get(cell, [])
        if not indexes:
            continue
        rel = Path(path).resolve().relative_to(source).as_posix() if str(Path(path).resolve()).startswith(str(source)) else str(path)
        h = sha(path); sources[rel] = h
        if rel in RAW_SOURCE_HASHES and RAW_SOURCE_HASHES[rel] != h:
            raise ValueError(f"raw source hash differs: {rel}")
        rows = e.old._source_row_map(e.old.load_pickle(path), kind=kind, dataset=dataset)
        if args.smoke:
            indexes = indexes[:args.smoke]
        for i in indexes:
            rec = records[i]; row = rows[rec["row_id"]]
            payload = e.old._topk_payload(row); spans = np.asarray(row["step_token_spans"], int)
            a, b = offsets[i], offsets[i + 1]; ta, tb = token_offsets[i], token_offsets[i + 1]
            if len(payload["logprobs"]) != tb - ta or spans.shape != (b - a, 2) or spans[0, 0] != 0 or spans[-1, 1] != tb - ta:
                raise ValueError(f"answer {i}: token/step mismatch")
            boc = (bocpd_verbatim(temporal[0], temporal[1], temporal[2][i]) if temporal is not None
                   else bocpd_residual_temporal_recipe(payload["logprobs"], row["token_entropies"]))
            x, v = answer_streams(payload["logprobs"], payload["ids"], row["gen_token_ids"], row["token_spilled_energies"], bocpd=boc)
            tokens[ta:tb] = x; valid[ta:tb] = v; spans_all[a:b] = spans
            step0[ta:tb] = step0_token_mask(spans, tb - ta); done[i] = True
        del rows; gc.collect()
        print(f"[cell] {cell}: {len(indexes)} answers, {time.time() - started:.0f}s", flush=True)
    if not args.smoke:
        assert done.all(), "incomplete extraction"
    keep = np.flatnonzero(done)
    provenance = ("temporal_context_data_v1 verbatim (Step 420 recipe)" if temporal is not None
                  else "rebuilt from the raw row by the temporal_context_data_v1 recipe (bocpd_residual_temporal_recipe)")

    gates = {}
    if args.bank_dir:
        # CT7 consumes the bank as top10.astype(float32) (cvf_v2/ct7.py::prepare), so exactness is
        # required at that precision on BOTH sides; the float64 difference is recorded descriptively.
        folder = Path(args.bank_dir); mism = 0; checked = 0; max64 = 0.0
        for cell, indexes in by_cell.items():
            f = folder / f"{cell}.npz"
            if not f.exists():
                continue
            z = np.load(f); idx = list(z["indexes"]); top = z["top10"]; cursor = 0
            for i in idx:
                if not done[i]:
                    cursor += offsets[i + 1] - offsets[i]; continue
                a, b = offsets[i], offsets[i + 1]; ta, tb = token_offsets[i], token_offsets[i + 1]
                got = np.column_stack([masked_step_top10(tokens[ta:tb, j].astype(float), valid[ta:tb, j], spans_all[a:b]) for j in range(5)])
                want = top[cursor:cursor + (b - a)]; cursor += b - a
                fin = np.isfinite(got) & np.isfinite(want)
                if fin.any():
                    max64 = max(max64, float(np.max(np.abs(got[fin] - want[fin]))))
                got = got.astype(np.float32); want = np.asarray(want).astype(np.float32)
                same = np.array_equal(np.isnan(got), np.isnan(want)) and np.array_equal(got[~np.isnan(got)], want[~np.isnan(want)])
                mism += int(not same); checked += 1
        gates["i_bank_top10_exact"] = {"checked_answers": checked, "mismatches": mism, "compared_as": "float32 (CT7 precision)",
                                       "max_abs_difference_float64": max64}
        assert mism == 0, f"gate (i) failed on {mism} answers"
    if args.profiles:
        prof = np.load(args.profiles); assert prof.shape == (total_steps, 7)
        t5 = np.full(total_steps, np.nan)
        for i in keep:
            a, b = offsets[i], offsets[i + 1]; ta, tb = token_offsets[i], token_offsets[i + 1]
            t5[a:b] = masked_step_top10(tokens[ta:tb, 5].astype(float), valid[ta:tb, 5], spans_all[a:b])
        rows_ok = np.repeat(done, np.diff(offsets))
        z5 = masked_answer_standardize(t5[:, None], np.isfinite(t5)[:, None], offsets)[:, 0]
        delta = float(np.max(np.abs(z5[rows_ok] - prof[rows_ok, 5])))
        tol = 1e-8
        gates["ii_bocpd_view_replay"] = {"max_abs_difference": delta, "tolerance": tol, "provenance": provenance}
        assert delta < tol, f"gate (ii) failed: {delta} >= {tol} ({provenance})"

    np.savez(out, tokens=tokens, valid=valid, token_offsets=token_offsets, step_spans=spans_all, step0_token_mask=step0,
             channels=np.asarray(STREAMS, dtype=str), done=done)
    L.dump(out.with_suffix(".MANIFEST.json"), {"schema": SCHEMA, "answers": int(done.sum()), "tokens": N, "steps": total_steps,
           "channels": list(STREAMS), "bocpd_provenance": provenance, "sources_sha256": sources, "gates": gates,
           "seconds": time.time() - started, "smoke": args.smoke, "development_only": True,
           "npz_sha256": L.digest(out)})
    print("EXTRACT_DONE", gates, f"{time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
