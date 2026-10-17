"""Extract per-step SUFFICIENT statistics of the chosen-token calibration streams (Step 416).

Same sources and alignment as run_chosen_token_calibration_v1; stores token counts and sums so that
the per-step z-test readouts (sum/sqrt(n), pooled-variance test) can be formed without re-reading tokens.

Original v1 docstring follows.

Same raw sources, hash checks, token alignment and Top10 step readout as
`run_digitfree_broad50_v1.extract`. Also records, per answer, the token-level Pearson correlation
of each statistic (and of raw surprisal) with the renormalized top-50 entropy, which is the direct
real-data check of the distribution-free construction. No labels are read.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.chosen_token_calibration import SUFFICIENT, step_sufficient_stats, token_calibration  # noqa: E402

OUT = ROOT / "results/chosen_token_calibration_v1"
ATLAS = ROOT / "results/fusion_independence_atlas_v1"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def corr_rows(x, h):
    """Token-level Pearson correlation of each column of x with h (nan if degenerate)."""
    hc = h - h.mean(); hn = np.sqrt((hc ** 2).sum())
    xc = x - x.mean(axis=0); xn = np.sqrt((xc ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (xc * hc[:, None]).sum(axis=0) / (xn * hn)


def main():
    from scripts import run_direct_probability_temporal as e
    from scripts.run_fusion_independence_atlas_v1 import RAW_SOURCE_HASHES
    e.old.configure_source_root(SOURCE)
    records = json.loads((SOURCE / "results/localization_full_benchmark_v3/evaluation/JOINED.json").read_text())["records"]
    metadata = json.loads((ATLAS / "bundle/data/METADATA.json").read_text())
    assert len(records) == len(metadata) == 13769
    folder = OUT / "extracted_sufficient"; folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for cell, path, kind, dataset in e.source_specs():
        dest = folder / f"{cell}.npz"
        if dest.exists():
            continue
        relative = path.relative_to(SOURCE).as_posix()
        digest = sha(path)
        if digest != RAW_SOURCE_HASHES[relative]:
            raise ValueError(f"raw source hash differs: {relative}")
        source = e.old._source_row_map(e.old.load_pickle(path), kind=kind, dataset=dataset)
        indexes = [i for i, r in enumerate(records) if r["cell"] == cell]
        steps, censored, tokens, corr = [], [], [], []
        for i in indexes:
            record = records[i]; row = source[record["row_id"]]
            if metadata[i]["uid"] != record["uid"]:
                raise ValueError("metadata identity mismatch")
            payload = e.old._topk_payload(row)
            if len(payload["logprobs"]) != record["tokens"]:
                raise ValueError("token count mismatch")
            spans = np.asarray(row["step_token_spans"], int)
            if spans.shape != (record["steps"], 2):
                raise ValueError("step count mismatch")
            x, cens, diag = token_calibration(payload["logprobs"], payload["ids"], row["gen_token_ids"],
                                              row["token_spilled_energies"])
            steps.append(step_sufficient_stats(x, cens, diag, spans))
            censored.append(int(cens.sum())); tokens.append(len(x))
            corr.append(corr_rows(np.column_stack([diag[:, 2], x]), diag[:, 0]))
        with dest.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(f, indexes=indexes, values=np.vstack(steps).astype(np.float64), columns=np.asarray(SUFFICIENT),
                                censored=np.asarray(censored), tokens=np.asarray(tokens),
                                token_entropy_corr=np.vstack(corr), source_sha256=digest)
        os.replace(dest.with_suffix(".tmp"), dest)
        del source
        print(f"{cell}: {len(indexes)} answers, censored tokens {sum(censored)}/{sum(tokens)}, "
              f"{time.time() - started:.0f}s", flush=True)
    print("EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
