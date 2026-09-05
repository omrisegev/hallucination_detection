"""Label-free window feasibility from existing v2 telemetry bundles.

Reads only raw telemetry and offsets, never labels or frozen risk predictions.
Each task contains one answer. Outputs are small resumable JSON diagnostics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
import platform
from pathlib import Path
import signal
import sys
import time
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
import scipy
from spectral_utils.window_localization import (
    DEFAULT_WIDTHS, build_window_matrix, feasible_widths, make_window_plan, matrix_diagnostics,
)

STOP = False


def stop_requested(signum, frame):
    global STOP
    STOP = True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def cell_geometry(path):
    with np.load(path, allow_pickle=False) as bundle:
        offsets = np.asarray(bundle["token_offsets"], dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError(f"invalid token offsets: {path}")
    return offsets


def valid_checkpoint(path, digest, cell, row):
    """An existing file is resumable only when its identity and payload agree."""
    path = Path(path)
    if not path.exists():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("config_sha256") != digest or result.get("cell") != cell
            or result.get("row") != row or result.get("labels_accessed") is not False
            or not isinstance(result.get("widths"), list)):
        raise RuntimeError(f"invalid checkpoint identity: {path}")
    return True


def selected_traces(path, offsets, selected, stream_names):
    """Stream the compressed raw NPY; keep at most one raw answer in memory."""
    with zipfile.ZipFile(path) as archive, archive.open("raw.npy") as handle:
        version = np.lib.format.read_magic(handle)
        shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
        if fortran or dtype.hasobject or shape != (int(offsets[-1]), len(stream_names)):
            raise ValueError("unexpected raw telemetry NPY shape/order/dtype")
        position = 0
        for row in sorted(selected):
            start, end = int(offsets[row]), int(offsets[row + 1])
            skip = (start - position) * shape[1] * dtype.itemsize
            while skip:
                chunk = handle.read(min(skip, 4 * 1024 * 1024))
                if not chunk:
                    raise EOFError("truncated telemetry")
                skip -= len(chunk)
            count = (end - start) * shape[1] * dtype.itemsize
            data = handle.read(count)
            if len(data) != count:
                raise EOFError("truncated answer telemetry")
            values = np.frombuffer(data, dtype=dtype).reshape(end - start, shape[1]).copy()
            yield row, values
            position = end


def inspect_answer(task):
    cell, row, raw, names, widths, minimum = task
    started = time.perf_counter()
    diagnostics = []
    for geometry in feasible_widths(len(raw), widths=widths, min_fit_windows=minimum):
        entry = dict(geometry)
        if geometry["width"] >= 32 and geometry["fit_windows"] >= 2:
            try:
                # Nonoverlapping extraction for this pilot; scoring-grid density
                # is a separate mapping choice and does not improve fit capacity.
                plan = make_window_plan(len(raw), geometry["width"])
                matrix = build_window_matrix(raw, names, plan)
                entry["matrix"] = matrix_diagnostics(matrix)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                entry["error"] = f"{type(error).__name__}: {error}"
        diagnostics.append(entry)
    return {"cell": cell, "row": int(row), "tokens": len(raw), "widths": diagnostics,
            "elapsed_seconds": time.perf_counter() - started, "labels_accessed": False,
            "interpretation": "feature/geometry audit only; no fusion or error evaluation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-dir", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", action="append")
    parser.add_argument("--rows-per-cell", type=int, default=30, help="0 = every answer")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    parser.add_argument("--min-fit-windows", type=int, default=8)
    args = parser.parse_args()
    if (args.workers < 1 or args.rows_per_cell < 0 or args.min_fit_windows < 2
            or any(w < 1 for w in args.widths) or len(set(args.widths)) != len(args.widths)):
        parser.error("positive workers/unique widths, nonnegative row limit and minimum >= 2 required")
    names = tuple(json.loads(args.schema.read_text(encoding="utf-8"))["stream_names"])
    paths = sorted(args.cells_dir.glob("*.npz"))
    if args.cell:
        paths = [path for path in paths if path.stem in args.cell]
        if set(args.cell) != {path.stem for path in paths}:
            parser.error("a requested cell is missing")
    if not paths:
        parser.error("no telemetry cells found")
    out = args.output.resolve()
    if out == args.cells_dir.resolve() or out.is_relative_to(args.cells_dir.resolve()):
        parser.error("output must be outside the input cell directory")
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "window-feasibility-v1", "stream_names": names, "widths": args.widths,
        "minimum_fit_windows": args.min_fit_windows, "rows_per_cell": args.rows_per_cell,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "cells": {path.stem: sha(path) for path in paths},
        "source": {name: sha(REPO / name) for name in (
            "spectral_utils/feature_utils.py", "spectral_utils/window_localization.py",
            "scripts/per_answer_localization/feasibility.py")},
        "labels_accessed": False,
    }
    config = json.loads(json.dumps(config))
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (out / "CONFIG.json").exists():
        if json.loads((out / "CONFIG.json").read_text(encoding="utf-8")) != config:
            raise RuntimeError("resume configuration/source/input mismatch")
    else:
        atomic_json(out / "CONFIG.json", config)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_requested)
    start = time.perf_counter()
    geometry = []

    def tasks():
        for path in paths:
            offsets = cell_geometry(path)
            lengths = np.diff(offsets)
            geometry.append({
                "cell": path.stem, "answers": len(lengths), "median_tokens": float(np.median(lengths)),
                "widths": [{"width": int(w), "answers_meeting_count_floor": int(np.sum(lengths // w >= args.min_fit_windows))}
                           for w in args.widths],
            })
            order = np.argsort(lengths, kind="stable")
            count = len(order) if args.rows_per_cell == 0 else min(len(order), args.rows_per_cell)
            selected = order[np.linspace(0, len(order) - 1, count, dtype=int)]
            remaining = [int(row) for row in selected if not valid_checkpoint(
                out / f"{path.stem}__{int(row)}.json", digest, path.stem, int(row))]
            for row, raw in selected_traces(path, offsets, remaining, names):
                if STOP:
                    return
                yield path.stem, row, raw, names, args.widths, args.min_fit_windows

    completed = 0
    iterator = iter(tasks())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while not STOP and not exhausted and len(pending) < 2 * args.workers:
                try:
                    pending.add(pool.submit(inspect_answer, next(iterator)))
                except StopIteration:
                    exhausted = True
            if STOP:
                exhausted = True
            if not pending:
                break
            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                result["config_sha256"] = digest
                atomic_json(out / f"{result['cell']}__{result['row']}.json", result)
                completed += 1
                print(f"{result['cell']} row{result['row']}: {result['tokens']} tokens, {result['elapsed_seconds']:.2f}s", flush=True)
    atomic_json(out / "SUMMARY.json", {
        "status": "CHECKPOINTED" if STOP else "COMPLETE", "new_answers": completed,
        "workers": args.workers, "elapsed_seconds": time.perf_counter() - start,
        "geometry": geometry, "config_sha256": digest, "labels_accessed": False,
    })
    if STOP:
        raise SystemExit(85)


if __name__ == "__main__":
    main()
