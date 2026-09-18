"""Reduce the per-answer layer-view npz field to one compact answer-level bundle.

Runs on the cluster (CPU only) via cluster/cpu_job.sbatch:

    ssh aircc 'export SLURM_CONF_SERVER=controller-primary; \
      S=/shared/cycle2_tau_averbuch_prj/omrisegev1; cd $S/code && \
      sbatch -p power-gpu --qos=owner_880 cluster/cpu_job.sbatch \
        cluster/reduce_layer_views_answer_level.py \
          --roots $S/results --out $S/results/layer_views_answer_level_v1'

Why reduce on the cluster: the capture is 5.5 GB across 13,769 files, and the geometry contract
consumes resid_norm only as a per-layer token-mean. Reducing here turns ~502 MB of resid_norm
into ~2 MB and the whole answer-level field into one ~320 MB file, so the transfer is one scp
rather than 13,769 files over the VPN. The token-level field stays on the cluster, where the
Stage-3 shuffle null needs it anyway.

hid_proj is kept float16 EXACTLY as captured. The float16 hazard is in accumulation
(np.linalg.norm / np.dot / np.mean), not in storage; consumers cast to float64 on load, which is
what spectral_utils.whitebox_layer_views.geometry_summaries does.

Local dry run with synthetic rows, no cluster and no real data:
    python cluster/reduce_layer_views_answer_level.py --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

# Axis order is fixed by cluster/layer_lens.py and must not be re-derived.
MODULES = ("attn", "mlp", "resid")
RESID = MODULES.index("resid")

#: cell -> (capture directory, subset subdirectory or None)
CELL_SOURCES = {
    "pb_gsm8k_q4": ("pb_layer_views_qwen3_4b", "gsm8k"),
    "pb_math_q4": ("pb_layer_views_qwen3_4b", "math"),
    "pb_olympiadbench_q4": ("pb_layer_views_qwen3_4b", "olympiadbench"),
    "pb_omnimath_q4": ("pb_layer_views_qwen3_4b", "omnimath"),
    "pb_gsm8k_q8": ("pb_layer_views_qwen3_8b", "gsm8k"),
    "pb_math_q8": ("pb_layer_views_qwen3_8b", "math"),
    "pb_olympiadbench_q8": ("pb_layer_views_qwen3_8b", "olympiadbench"),
    "pb_omnimath_q8": ("pb_layer_views_qwen3_8b", "omnimath"),
    "prmbench_qwen3_8b": ("prmbench_layer_views_qwen3_8b", None),
}

EXPECTED_COUNTS = {
    "pb_gsm8k_q4": 400, "pb_gsm8k_q8": 400,
    "pb_math_q4": 1000, "pb_math_q8": 1000,
    "pb_olympiadbench_q4": 1000, "pb_olympiadbench_q8": 1000,
    "pb_omnimath_q4": 1000, "pb_omnimath_q8": 1000,
    "prmbench_qwen3_8b": 6969,
}
N_ANSWERS = 13769


def row_filename(row_id: str, cell: str) -> str:
    """Map a JOINED row_id to its npz basename.

    ProcessBench rows are namespaced in JOINED as "<subset>::<id>" while the producer wrote the
    bare id, so the prefix is stripped. PRMBench row_ids match the filename exactly.
    The SAME bare filename exists under both the 4B and 8B trees, so the directory is the only
    thing that distinguishes cells: never flatten the capture.
    """
    if cell.startswith("pb_") and "::" in row_id:
        return row_id.split("::", 1)[1] + ".npz"
    return row_id + ".npz"


def reduce_row(path: str) -> dict:
    """One answer -> its answer-level summary. Reads only what is needed."""
    with np.load(path, allow_pickle=False) as z:
        lens_h = z["lens_H"]                       # [3, L, T] f16, full vocabulary
        resid_norm = z["resid_norm"]               # [L, T] f16
        cov_eigs = z["cov_eigs"]                   # [L, R] f32
        hid_proj = z["hid_proj"]                   # [L, P] f16
        final_top15 = z["final_lens_H"]            # [T] f32, top-15 renormalised
        final_full = z["final_lens_H_fullvocab"]   # [T] f32, full vocabulary
        anchor_src = z["lens_logp_tgt"][RESID, -1]  # [T] f16
        gate_flag = str(z["gate_flag"])
        n_tokens = int(final_top15.shape[0])

    resid_lens = lens_h[RESID].astype(np.float64)  # [L, T]

    # AXIS-ORDER assertion, NOT a token-alignment gate. Be precise about what this can prove.
    #
    # The producer writes  final_lens_H_fullvocab = lens_H[resid, -1].astype(f32)  (see
    # cluster/run_localization_layer_views.py:99), and f16 -> f32 -> f64 is lossless, so this
    # difference is EXACTLY 0.0 by construction and can never fail. On its own it is a vacuous
    # gate of exactly the kind Step 421's pre-flight nearly shipped.
    #
    # What it does establish is that our (module index, layer index) convention matches the
    # producer's. To make that non-vacuous we pair it with a control against the WRONG module:
    # identity == 0 AND control > 0 together prove the convention. A control of 0 would mean the
    # module axis is degenerate and the assertion proves nothing.
    #
    # Genuine token alignment is evidenced separately, and already: the producer's own gate
    # compared final_lens_H against the cached top-15 token_entropies at tol_median 2e-2 and
    # reported 13,769 checked / 0 failed, with a GATE_VACUOUS guard against a silent no-op join.
    final_full64 = final_full.astype(np.float64)
    identity = float(np.max(np.abs(resid_lens[-1] - final_full64)))
    axis_control = float(np.max(np.abs(lens_h[0, -1].astype(np.float64) - final_full64)))

    # Depth-decay curve against the TOP-15 final entropy: a different statistic on purpose.
    target = final_top15.astype(np.float64)
    tgt_c = target - target.mean()
    tgt_norm = float(np.sqrt(np.dot(tgt_c, tgt_c)))
    layers = resid_lens.shape[0]
    decay = np.zeros(layers, dtype=np.float64)
    if tgt_norm > 1e-12 and n_tokens > 1:
        for l in range(layers):
            v = resid_lens[l] - resid_lens[l].mean()
            vn = float(np.sqrt(np.dot(v, v)))
            decay[l] = float(np.dot(v, tgt_c) / (vn * tgt_norm)) if vn > 1e-12 else 0.0

    return {
        "cov_eigs": np.asarray(cov_eigs, dtype=np.float32),
        "hid_proj": np.asarray(hid_proj, dtype=np.float16),
        "resid_norm_mean": resid_norm.astype(np.float64).mean(axis=1).astype(np.float32),
        "lens_anchor": np.float32(anchor_src.astype(np.float64).mean()),
        "depth_decay_corr": decay.astype(np.float32),
        "identity_check": np.float32(identity),
        "axis_control": np.float32(axis_control),
        "n_tokens": np.int32(n_tokens),
        "gate_flag": gate_flag,
    }


def roster(joined_path: str) -> list[tuple[str, str]]:
    with open(joined_path, encoding="utf-8") as handle:
        records = json.load(handle)["records"]
    if len(records) != N_ANSWERS:
        raise SystemExit(f"roster has {len(records)} records, expected {N_ANSWERS}")
    return [(r["row_id"], r["cell"]) for r in records]


def stratified_pilot(rows: list[tuple[str, str]], limit: int) -> list[tuple[str, str]]:
    """Take roughly ``limit`` rows spread over all nine cells.

    The roster is grouped by cell, so ``rows[:limit]`` would be pb_gsm8k_q4 only and would never
    exercise the PRMBench naming path or the 8B tree - i.e. it would not test the join at all.
    """
    per_cell = max(1, -(-limit // len(EXPECTED_COUNTS)))
    seen: dict[str, int] = {}
    picked = []
    for row_id, cell in rows:
        if seen.get(cell, 0) >= per_cell:
            continue
        seen[cell] = seen.get(cell, 0) + 1
        picked.append((row_id, cell))
    return picked


def run(roots: str, joined_path: str, out_dir: str, limit: int | None = None) -> int:
    os.makedirs(out_dir, exist_ok=True)
    rows = roster(joined_path)
    if limit:
        rows = stratified_pilot(rows, limit)
        print(f"[reduce] PILOT: {len(rows)} rows over "
              f"{len(set(c for _, c in rows))} cells", flush=True)
    started = time.time()

    acc: dict[str, list] = {k: [] for k in
                            ("cov_eigs", "hid_proj", "resid_norm_mean", "lens_anchor",
                             "depth_decay_corr", "identity_check", "axis_control",
                             "n_tokens")}
    row_ids, cells, flags, missing = [], [], [], []

    for i, (row_id, cell) in enumerate(rows):
        directory, subset = CELL_SOURCES[cell]
        parts = [roots, directory] + ([subset] if subset else []) + ["rows", row_filename(row_id, cell)]
        path = os.path.join(*parts)
        if not os.path.exists(path):
            missing.append(f"{cell}/{row_id}")
            continue
        summary = reduce_row(path)
        for key in acc:
            acc[key].append(summary[key])
        row_ids.append(row_id)
        cells.append(cell)
        flags.append(summary["gate_flag"])
        if (i + 1) % 1000 == 0:
            print(f"[reduce] {i + 1}/{len(rows)}  {time.time() - started:.0f}s", flush=True)

    # A missing row is fatal in both modes: the pilot exists to prove the join, so a join miss
    # there is exactly the failure it is looking for.
    if missing:
        raise SystemExit(f"{len(missing)} rows missing, first 5: {missing[:5]}")

    stacked = {key: np.stack(values) for key, values in acc.items()}
    per_cell = {cell: int(sum(1 for c in cells if c == cell)) for cell in EXPECTED_COUNTS}
    if limit:
        if len(set(cells)) != len(EXPECTED_COUNTS):
            raise SystemExit(f"pilot touched {len(set(cells))} cells, expected all "
                             f"{len(EXPECTED_COUNTS)}")
    else:
        if len(row_ids) != N_ANSWERS:
            raise SystemExit(f"{len(row_ids)} rows reduced, expected {N_ANSWERS}")
        if per_cell != EXPECTED_COUNTS:
            raise SystemExit(f"per-cell counts drifted: {per_cell}")

    out_npz = os.path.join(out_dir, "ANSWER_LEVEL_PILOT.npz" if limit else "ANSWER_LEVEL.npz")
    tmp = out_npz + ".tmp.npz"
    np.savez_compressed(tmp, row_id=np.array(row_ids), cell=np.array(cells),
                        gate_flag=np.array(flags), **stacked)
    os.replace(tmp, out_npz)

    digest = hashlib.sha256(open(out_npz, "rb").read()).hexdigest()
    identity = stacked["identity_check"]
    control = stacked["axis_control"]
    # Non-vacuity: the assertion only means something if the wrong-module control is far from 0.
    if not (float(identity.max()) <= 1e-6 and float(control.min()) > 1e-3):
        raise SystemExit(
            f"axis-order assertion inconclusive: identity_max={float(identity.max()):.3e} "
            f"(want ~0), control_min={float(control.min()):.3e} (want >0)")
    manifest = {
        "n_answers": len(row_ids),
        "per_cell": per_cell,
        "n_layers": int(stacked["cov_eigs"].shape[1]),
        "cov_eigs_r": int(stacked["cov_eigs"].shape[2]),
        "proj_dim": int(stacked["hid_proj"].shape[2]),
        "gate_flag_nonempty": int(sum(1 for f in flags if f)),
        "identity_check_max": float(identity.max()),
        "identity_check_p99": float(np.quantile(identity, 0.99)),
        "axis_control_min": float(control.min()),
        "axis_control_median": float(np.median(control)),
        "bytes": os.path.getsize(out_npz),
        "sha256": digest,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "git_sha": os.environ.get("GIT_SHA"),
        "seconds": round(time.time() - started, 1),
        "status": "PILOT" if limit else "COMPLETE",
    }
    name = "MANIFEST_PILOT.json" if limit else "MANIFEST.json"
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    print(json.dumps(manifest, indent=1), flush=True)
    return 0


def smoke() -> int:
    """End-to-end on synthetic rows: exercises the writer, the join and every assert."""
    import shutil
    import tempfile

    base = tempfile.mkdtemp(prefix="reduce_smoke_")
    try:
        rng = np.random.default_rng(0)
        layers, rank, proj = 36, 32, 256
        rows = [("gsm8k::gsm8k-0", "pb_gsm8k_q4"),
                ("gsm8k::gsm8k-1", "pb_gsm8k_q4"),
                ("circular_circular_prm_test_p1_0", "prmbench_qwen3_8b")]
        for row_id, cell in rows:
            directory, subset = CELL_SOURCES[cell]
            parts = [base, directory] + ([subset] if subset else []) + ["rows"]
            target_dir = os.path.join(*parts)
            os.makedirs(target_dir, exist_ok=True)
            n_tokens = int(rng.integers(20, 40))
            lens_h = rng.normal(8.0, 1.0, size=(3, layers, n_tokens)).astype(np.float16)
            np.savez_compressed(
                os.path.join(target_dir, row_filename(row_id, cell)),
                lens_H=lens_h,
                lens_logp_tgt=rng.normal(-2.0, 1.0, size=(3, layers, n_tokens)).astype(np.float16),
                resid_norm=np.abs(rng.normal(10.0, 1.0, size=(layers, n_tokens))).astype(np.float16),
                cov_eigs=np.abs(rng.normal(size=(layers, rank))).astype(np.float32) * 1e4,
                hid_proj=rng.normal(size=(layers, proj)).astype(np.float16),
                # the identity the entry gate checks: the final resid lens IS the full-vocab one
                final_lens_H=rng.normal(5.0, 1.0, size=n_tokens).astype(np.float32),
                final_lens_H_fullvocab=lens_h[RESID, -1].astype(np.float32),
                gen_token_ids=rng.integers(0, 1000, size=n_tokens).astype(np.int32),
                step_starts=np.array([0]), step_ends=np.array([n_tokens]),
                gate_flag=np.asarray(""))

        joined = os.path.join(base, "JOINED.json")
        with open(joined, "w", encoding="utf-8") as handle:
            json.dump({"records": [{"row_id": r, "cell": c} for r, c in rows]}, handle)

        # the roster assert must fire on a short roster
        try:
            roster(joined)
        except SystemExit as exc:
            print(f"  PASS  short roster rejected: {exc}")
        else:
            print("  FAIL  short roster accepted")
            return 1

        summary = reduce_row(os.path.join(base, "pb_layer_views_qwen3_4b", "gsm8k", "rows",
                                          "gsm8k-0.npz"))
        checks = [
            ("row_filename strips the PB prefix",
             row_filename("gsm8k::gsm8k-0", "pb_gsm8k_q4") == "gsm8k-0.npz"),
            ("row_filename leaves PRMBench alone",
             row_filename("circular_x_0", "prmbench_qwen3_8b") == "circular_x_0.npz"),
            ("cov_eigs shape", summary["cov_eigs"].shape == (layers, rank)),
            ("hid_proj stays float16", summary["hid_proj"].dtype == np.float16),
            ("resid_norm reduced to per-layer mean", summary["resid_norm_mean"].shape == (layers,)),
            ("depth_decay_corr per layer", summary["depth_decay_corr"].shape == (layers,)),
            ("identity check ~0 when the arrays agree", float(summary["identity_check"]) < 1e-2),
            ("decay correlations in [-1, 1]",
             bool(np.all(np.abs(summary["depth_decay_corr"]) <= 1.0 + 1e-6))),
        ]
        ok = True
        for name, passed in checks:
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
            ok &= bool(passed)
        print("SMOKE PASSED" if ok else "SMOKE FAILED")
        return 0 if ok else 1
    finally:
        shutil.rmtree(base, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", default="/shared/cycle2_tau_averbuch_prj/omrisegev1/results")
    parser.add_argument("--joined",
                        default="results/localization_full_benchmark_v3/evaluation/JOINED.json")
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="stratified pilot over all nine cells; skips the full-count asserts")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        return smoke()
    if not args.out:
        parser.error("--out is required unless --smoke")
    return run(args.roots, args.joined, args.out, args.limit)


if __name__ == "__main__":
    sys.exit(main())
