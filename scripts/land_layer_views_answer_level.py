"""Validate the landed answer-level bundle before anything scientific reads it.

Checks hash, shapes, dtypes, per-cell counts, the row_id join against JOINED, the axis-order
assertion and its non-vacuity control, and the gate-flag distribution.

    python scripts/land_layer_views_answer_level.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_utils.whitebox_layer_views import (  # noqa: E402
    EXPECTED_CELL_COUNTS,
    N_ANSWERS,
    load_joined,
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results" / "whitebox_layer_views_localization_v1"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


manifest = json.loads((BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))
path = BUNDLE / "ANSWER_LEVEL.npz"

print("[1] integrity")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
check("sha256 matches the manifest", digest == manifest["sha256"],
      f"{digest[:16]}... vs {manifest['sha256'][:16]}...")
check("byte count matches", path.stat().st_size == manifest["bytes"],
      f"{path.stat().st_size} vs {manifest['bytes']}")
check("manifest status COMPLETE", manifest["status"] == "COMPLETE", manifest["status"])

print("[2] shapes and dtypes")
z = np.load(path, allow_pickle=False)
L, R, P = manifest["n_layers"], manifest["cov_eigs_r"], manifest["proj_dim"]
check("n_layers / cov_eigs_r / proj_dim", (L, R, P) == (36, 32, 256), f"{L}/{R}/{P}")
expect = {
    "cov_eigs": ((N_ANSWERS, L, R), np.float32),
    "hid_proj": ((N_ANSWERS, L, P), np.float16),
    "resid_norm_mean": ((N_ANSWERS, L), np.float32),
    "depth_decay_corr": ((N_ANSWERS, L), np.float32),
    "lens_anchor": ((N_ANSWERS,), np.float32),
    "identity_check": ((N_ANSWERS,), np.float32),
    "axis_control": ((N_ANSWERS,), np.float32),
    "n_tokens": ((N_ANSWERS,), np.int32),
}
for key, (shape, dtype) in expect.items():
    arr = z[key]
    check(f"{key} {shape} {np.dtype(dtype).name}",
          arr.shape == shape and arr.dtype == dtype, f"{arr.shape} {arr.dtype}")
check("hid_proj kept float16 at rest", z["hid_proj"].dtype == np.float16)

print("[3] finiteness")
for key in ("cov_eigs", "resid_norm_mean", "depth_decay_corr", "lens_anchor"):
    arr = z[key].astype(np.float64)
    check(f"{key} all finite", bool(np.isfinite(arr).all()),
          f"{int((~np.isfinite(arr)).sum())} non-finite")
check("hid_proj all finite (after cast)", bool(np.isfinite(z["hid_proj"].astype(np.float64)).all()))
# cov_eigs are eigenvalues of a centred token covariance, so exact non-negativity is not a
# property the eigensolver guarantees on a rank-deficient matrix. Quantify instead of asserting:
# what matters is that any negative is round-off relative to the spectrum, and that the consumer
# clips. _covariance_summaries does clip (np.maximum(eig, 0.0)), so the summaries are protected.
_cov = z["cov_eigs"].astype(np.float64)
_neg = _cov[_cov < 0]
_ratio = (abs(_neg.min()) / float(np.median(_cov[:, :, 0]))) if _neg.size else 0.0
check("cov_eigs negatives are round-off, not structure",
      _neg.size == 0 or (_neg.size / _cov.size < 1e-4 and _ratio < 1e-2),
      f"{_neg.size} of {_cov.size:,} ({100 * _neg.size / _cov.size:.5f}%), "
      f"min {_neg.min():.3g} = {_ratio:.2e} of the median top eigenvalue"
      if _neg.size else "none")

print("[4] axis-order assertion and its non-vacuity control")
identity, control = z["identity_check"], z["axis_control"]
check("identity ~ 0 over every row", float(identity.max()) <= 1e-6,
      f"max {float(identity.max()):.3e}")
check("wrong-module control bounded away from 0 (NON-VACUITY)",
      float(control.min()) > 1e-3,
      f"min {float(control.min()):.4f}, median {float(np.median(control)):.4f}")

print("[5] join against JOINED")
data = load_joined(ROOT)
row_id, cell = z["row_id"].astype(str), z["cell"].astype(str)
check("row order identical to the roster", bool(np.array_equal(row_id, data["row_id"])),
      "bundle rows are aligned index-for-index with JOINED records")
check("cells identical to the roster", bool(np.array_equal(cell, data["cells"])))
counts = {c: int((cell == c).sum()) for c in EXPECTED_CELL_COUNTS}
check("per-cell counts exact", counts == EXPECTED_CELL_COUNTS,
      "" if counts == EXPECTED_CELL_COUNTS else str(counts))
# row_id alone is NOT unique: the same ProcessBench answer appears under both the 4B and 8B
# cells with the same bare id. That is the reason the capture must never be flattened, so the
# uniqueness key is (row_id, cell) - which is also what JOINED's own `uid` encodes.
pairs = set(zip(row_id.tolist(), cell.tolist()))
check("(row_id, cell) pairs unique", len(pairs) == N_ANSWERS, f"{len(pairs)} distinct pairs")
# ProcessBench is 6,800 answers over 3,400 distinct questions scored under two models, so
# exactly 3,400 row_ids appear twice: 13,769 - 3,400 = 10,369 distinct.
check("row_id duplication is exactly the q4/q8 split",
      len(set(row_id.tolist())) == N_ANSWERS - 3400,
      f"{len(set(row_id.tolist()))} distinct row_ids, "
      f"{N_ANSWERS - len(set(row_id.tolist()))} appear in two cells")

print("[6] gate flags and token counts")
flags = z["gate_flag"].astype(str)
nonempty = int((flags != "").sum())
check("no row carries a gate flag", nonempty == 0, f"{nonempty} flagged")
steps = np.diff(data["offsets"])
check("token counts positive", int(z["n_tokens"].min()) > 0, f"min {int(z['n_tokens'].min())}")
check("total tokens ~ 6.97M", abs(int(z["n_tokens"].sum()) - 6_968_779) < 1,
      f"{int(z['n_tokens'].sum())}")
check("steps roster intact", int(steps.sum()) == 145597, f"{int(steps.sum())}")

print()
print(f"depth_decay_corr: layer 0 median {float(np.median(z['depth_decay_corr'][:, 0])):+.4f}, "
      f"layer 35 median {float(np.median(z['depth_decay_corr'][:, -1])):+.4f}")
if failures:
    print(f"\nFAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("\nLANDING VALIDATED")
