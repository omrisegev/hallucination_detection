"""Prepare, run and evaluate one bounded, source-frozen answer-local pilot."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CAPSULE = ROOT / "local_cache/short_cycle01_code"
OUT = ROOT / "results/localization_short_cycle01"
LIVE = ROOT.parent / "hd_jlsml_v2_wt"
DATA = LIVE / "results/joint_lsml_optimization_v2"
sys.path.insert(0, str(CAPSULE))
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def extract_definition(source, name):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return ast.get_source_segment(source, node)


def prepare():
    if (OUT / "FREEZE.json").exists():
        raise RuntimeError("Existing freeze: inspect, do not overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("inputs", "scores", "logs"):
        (OUT / sub).mkdir(exist_ok=True)
    core = ROOT / "spectral_utils/short_cycle_localization.py"
    (CAPSULE / "spectral_utils/short_cycle_localization.py").write_bytes(core.read_bytes())
    # Exact extraction avoids unrelated package initialization and any mutable
    # imports from Claude's worktree during a numerical run.
    ref = "codex/per-answer-localization-v1:scripts/per_answer_localization/feasibility.py"
    src = subprocess.run(["git", "-C", str(ROOT), "show", ref], capture_output=True, check=True).stdout.decode("utf-8")
    (CAPSULE / "io_snapshot.py").write_text(
        "import zipfile\nimport numpy as np\n" + extract_definition(src, "selected_traces") + "\n", encoding="utf-8")
    eval_source = (LIVE / "scripts/joint_lsml_optimization_v2/evaluate_v2.py").read_text(encoding="utf-8")
    (CAPSULE / "metric_snapshot.py").write_text(
        "from __future__ import annotations\nimport numpy as np\n" +
        extract_definition(eval_source, "_auroc") + "\n\n" +
        extract_definition(eval_source, "_prm_paired_bootstrap") + "\n", encoding="utf-8")
    views_source = (LIVE / "spectral_utils/token_feature_views.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(views_source).body if isinstance(n, ast.Assign)
                and any(isinstance(x, ast.Name) and x.id == "BROAD_TOKEN_VIEWS" for x in n.targets))
    names = ("trace_length_series",) + tuple(ast.literal_eval(node.value))
    source = DATA / "cells/prmbench_qwen3_8b.npz"
    with np.load(source, allow_pickle=False) as z:
        offsets, ids, groups = (z[k].copy() for k in ("token_offsets", "row_ids", "group_ids"))
        starts, ends, step_offsets = (z[k].copy() for k in ("step_starts", "step_ends", "step_row_offsets"))
    lengths = np.diff(offsets)
    eligible = np.flatnonzero((lengths >= 1024) & (lengths <= 2048))
    order = sorted(eligible, key=lambda i: hashlib.sha256(("short-cycle01:" + str(ids[i])).encode()).hexdigest())
    selected, seen = [], set()
    for i in order:
        if str(groups[i]) not in seen:
            selected.append(int(i)); seen.add(str(groups[i]))
        if len(selected) == 30: break
    if len(selected) < 30:
        raise RuntimeError("Insufficient source-distinct eligible answers")
    fold_file = DATA / "folds/folds.json"
    folds = json.loads(fold_file.read_text(encoding="utf-8"))["prmbench"]["outer"]
    cohort = [{"row": i, "row_id": str(ids[i]), "group_id": str(groups[i]),
               "tokens": int(lengths[i]), "outer_fold": int(folds[str(groups[i])])} for i in selected]
    save_json(OUT / "COHORT.json", cohort)
    save_json(OUT / "STREAM_NAMES.json", names)
    from io_snapshot import selected_traces
    for i, raw in selected_traces(source, offsets, selected, names):
        lo, hi = int(step_offsets[i]), int(step_offsets[i + 1])
        np.savez_compressed(OUT / "inputs" / f"row{i}.npz", raw=raw,
                            step_starts=starts[lo:hi] - offsets[i], step_ends=ends[lo:hi] - offsets[i],
                            global_step_indices=np.arange(lo, hi))
    from spectral_utils.short_cycle_localization import SETTINGS
    import scipy, sklearn
    freeze = {
        "created_utc": utc(), "labels_accessed": False, "settings": SETTINGS,
        "cell": "prmbench_qwen3_8b", "wall_cap_seconds": 1200,
        "per_answer_cap_seconds": 90, "source_data": {"path": str(source), "sha256": digest(source)},
        "fold_file_sha256": digest(fold_file), "cohort_sha256": digest(OUT / "COHORT.json"),
        "stream_names_sha256": digest(OUT / "STREAM_NAMES.json"),
        "inputs": {x.name: digest(x) for x in (OUT / "inputs").glob("*.npz")},
        "source_files": {str(x.relative_to(CAPSULE)): digest(x) for x in CAPSULE.rglob("*.py")},
        "runner_sha256": digest(__file__), "upstream_sources_sha256": digest(CAPSULE / "UPSTREAM_SOURCES.json"),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "evaluation": {"primary_compatibility": "v2 pooled step AUROC on identical pilot IDs",
                       "additional": "fold-wise AUROC and within-answer ranking; no no-error decision",
                       "paired_bootstrap": 2000, "bootstrap_seed": 2026090602,
                       "coverage": "full reference coverage and joint/common supported subset both explicit"},
    }
    save_json(OUT / "FREEZE.json", freeze)
    print(json.dumps({"prepared": len(cohort), "labels_accessed": False,
                      "cohort_sha256": freeze["cohort_sha256"], "input_mb": sum(x.stat().st_size for x in (OUT / "inputs").glob("*.npz")) / 1e6}), flush=True)


def verify_freeze():
    f = json.loads((OUT / "FREEZE.json").read_text(encoding="utf-8"))
    checks = [(Path(__file__), f["runner_sha256"]), (OUT / "COHORT.json", f["cohort_sha256"]),
              (OUT / "STREAM_NAMES.json", f["stream_names_sha256"]),
              (CAPSULE / "UPSTREAM_SOURCES.json", f["upstream_sources_sha256"])]
    checks += [(CAPSULE / p, h) for p, h in f["source_files"].items()]
    checks += [(OUT / "inputs" / p, h) for p, h in f["inputs"].items()]
    for p, h in checks:
        if digest(p) != h: raise RuntimeError("Frozen content changed: " + str(p))
    return f


def worker(row):
    from spectral_utils.short_cycle_localization import fit_answer
    with np.load(OUT / "inputs" / f"row{row}.npz", allow_pickle=False) as z:
        raw, starts, ends = z["raw"], z["step_starts"], z["step_ends"]
    names = json.loads((OUT / "STREAM_NAMES.json").read_text())
    arrays, report = fit_answer(raw, names, starts, ends)
    report["row"] = row
    target = OUT / "scores" / f"row{row}.npz"
    with target.with_suffix(".npz.tmp").open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(target.with_suffix(".npz.tmp"), target)
    report["score_sha256"] = digest(target)
    save_json(OUT / "scores" / f"row{row}.json", report)


def run():
    freeze = verify_freeze()
    if (OUT / "RUN.json").exists(): raise RuntimeError("One bounded stage only; inspect existing run")
    cohort = json.loads((OUT / "COHORT.json").read_text())
    started = time.monotonic()
    runlog = {"started_utc": utc(), "rows": [], "labels_accessed": False}
    save_json(OUT / "RUN.json", runlog)
    for item in cohort:
        remaining = freeze["wall_cap_seconds"] - (time.monotonic() - started)
        if remaining <= 0: break
        row = item["row"]
        with (OUT / "logs" / f"row{row}.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--row", str(row)],
                                     stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                code = child.wait(timeout=min(remaining, freeze["per_answer_cap_seconds"]))
                state = "COMPLETE" if code == 0 else "WORKER_FAILED"
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(); state = "TIMEOUT"
        rec = {**item, "status": state}
        runlog["rows"].append(rec)
        runlog["elapsed_seconds"] = time.monotonic() - started
        save_json(OUT / "RUN.json", runlog)
        methods = {}
        f = OUT / "scores" / f"row{row}.json"
        if state == "COMPLETE":
            detail = json.loads(f.read_text())
            methods = {k: v["status"] for k, v in detail["methods"].items()}
        print(json.dumps({"done": len(runlog["rows"]), "row": row, "status": state,
                          "methods": methods, "elapsed_seconds": round(runlog["elapsed_seconds"], 2)}), flush=True)
        # First-answer runtime is recorded before continuing the SAME frozen pilot.
    runlog["finished_utc"] = utc()
    runlog["elapsed_seconds"] = time.monotonic() - started
    runlog["unattempted_rows"] = [x["row"] for x in cohort[len(runlog["rows"]):]]
    save_json(OUT / "RUN.json", runlog)
    verify_freeze()
    save_json(OUT / "SCORE_FREEZE.json", {
        "created_utc": utc(), "labels_accessed": False,
        "source_freeze_sha256": digest(OUT / "FREEZE.json"), "run_sha256": digest(OUT / "RUN.json"),
        "files": {str(x.relative_to(OUT)): digest(x) for x in (OUT / "scores").iterdir() if x.suffix in (".npz", ".json")},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "worker"))
    parser.add_argument("--row", type=int)
    args = parser.parse_args()
    if args.stage == "prepare": prepare()
    elif args.stage == "run": run()
    elif args.stage == "worker": worker(args.row)
