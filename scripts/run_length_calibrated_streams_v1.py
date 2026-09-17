"""Step 420 extraction: plain and length-calibrated Top10 readouts for CT7's six token streams.

Pass A: the five digit-free bank streams (H0lim, ve0, ve0.75, ve1, H0lim prefix innovation) from the
raw pickles, via digitfree_broad50.token_bank, same sources and alignment checks as the bank extraction.
Pass B: the BOCPD token residual recomputed from temporal_context_data_v1 (innovation5 bank), 8 processes.
No labels are read. Exactness gates are applied in the analysis script before any arm is scored.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import hashlib
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1]
TEMPORAL = SOURCE / ".worktrees/temporal-research-20260915"
sys.path.insert(0, str(ROOT))
from spectral_utils.digitfree_broad50 import NAMES as BANK50, token_bank  # noqa: E402
from spectral_utils.length_calibrated_readout import step_topk_and_calibrated  # noqa: E402

OUT = ROOT / "results/length_explicit_ct7_v1"
STREAMS = ("H0lim", "ve0", "ve0.75", "ve1", "H0lim_prefix_innovation")
COLS = [list(BANK50).index(n) for n in STREAMS]
ATLAS = ROOT / "results/fusion_independence_atlas_v1"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pass_bank():
    from scripts import run_direct_probability_temporal as e
    from scripts.run_fusion_independence_atlas_v1 import RAW_SOURCE_HASHES
    e.old.configure_source_root(SOURCE)
    records = json.loads((SOURCE / "results/localization_full_benchmark_v3/evaluation/JOINED.json").read_text())["records"]
    metadata = json.loads((ATLAS / "bundle/data/METADATA.json").read_text())
    folder = OUT / "bank"; folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for cell, path, kind, dataset in e.source_specs():
        dest = folder / f"{cell}.npz"
        if dest.exists():
            continue
        relative = path.relative_to(SOURCE).as_posix()
        if sha(path) != RAW_SOURCE_HASHES[relative]:
            raise ValueError(f"raw source hash differs: {relative}")
        source = e.old._source_row_map(e.old.load_pickle(path), kind=kind, dataset=dataset)
        indexes = [i for i, r in enumerate(records) if r["cell"] == cell]
        top, cal, cnt = [], [], []
        for i in indexes:
            record = records[i]; row = source[record["row_id"]]
            if metadata[i]["uid"] != record["uid"]:
                raise ValueError("metadata identity mismatch")
            payload = e.old._topk_payload(row)
            spans = np.asarray(row["step_token_spans"], int)
            if len(payload["logprobs"]) != record["tokens"] or spans.shape != (record["steps"], 2):
                raise ValueError("token or step count mismatch")
            x, valid, _ = token_bank(payload["logprobs"], payload["ids"], row["gen_token_ids"], row["token_spilled_energies"])
            t_ans = np.empty((len(spans), len(COLS))); c_ans = np.empty_like(t_ans); n_ans = np.empty_like(t_ans)
            for j, col in enumerate(COLS):
                t_ans[:, j], c_ans[:, j], n_ans[:, j] = step_topk_and_calibrated(x[:, col], valid[:, col], spans)
            top.append(t_ans); cal.append(c_ans); cnt.append(n_ans)
        with dest.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(f, indexes=indexes, top10=np.vstack(top), calibrated=np.vstack(cal), counts=np.vstack(cnt))
        os.replace(dest.with_suffix(".tmp"), dest)
        del source
        print(f"bank {cell}: {len(indexes)} answers, {time.time() - started:.0f}s", flush=True)


_ACP = None; _FEATS = None; _META = None; _SPANS = None


def _init():
    global _ACP, _FEATS, _META, _SPANS
    spec = importlib.util.spec_from_file_location("acp", TEMPORAL / "spectral_utils/aligned_context_predictors.py")
    _ACP = importlib.util.module_from_spec(spec); spec.loader.exec_module(_ACP)
    data = TEMPORAL / "results/temporal_context_data_v1"
    manifest = json.loads((data / "MANIFEST.json").read_text())
    _FEATS = np.load(data / "features.npy", mmap_mode="r")[:, manifest["banks"]["innovation5"]]
    _META = json.loads((data / "METADATA.json").read_text())
    _SPANS = np.load(data / "step_spans.npy", mmap_mode="r")


def _bocpd_one(i):
    m = _META[i]; a = m["offset"]; T = m["tokens"]
    raw = np.asarray(_FEATS[a:a + T], float)
    z = (raw - np.asarray(m["mean"], float)) / np.asarray(m["scale"], float)
    signed = (z - _ACP.bocpd_mean(z)).mean(axis=1)
    spans = np.asarray(_SPANS[m["step_start"]:m["step_stop"]], int) - a
    top, cal, _ = step_topk_and_calibrated(signed, np.ones(T, bool), spans)
    return i, top, cal


def pass_bocpd():
    dest = OUT / "bocpd.npz"
    if dest.exists():
        return
    _init()
    order = np.argsort([-m["tokens"] for m in _META])
    n_steps = sum(m["step_stop"] - m["step_start"] for m in _META)
    top = np.full(n_steps, np.nan); cal = np.full(n_steps, np.nan)
    started = time.time(); done = 0
    with Pool(8, initializer=_init) as pool:
        for i, t, c in pool.imap_unordered(_bocpd_one, [int(k) for k in order], chunksize=8):
            m = _META[i]; top[m["step_start"]:m["step_stop"]] = t; cal[m["step_start"]:m["step_stop"]] = c
            done += 1
            if done % 2000 == 0:
                print(f"bocpd {done}/{len(_META)} {time.time() - started:.0f}s", flush=True)
    if not (np.isfinite(top).all() and np.isfinite(cal).all()):
        raise ValueError("incomplete BOCPD pass")
    np.savez_compressed(dest, top10=top, calibrated=cal,
                        features_sha256=json.loads((TEMPORAL / "results/temporal_context_data_v1/MANIFEST.json").read_text())["files"]["features.npy"])
    print(f"bocpd done {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pass_bocpd()
    pass_bank()
    print("EXTRACT_DONE", flush=True)
