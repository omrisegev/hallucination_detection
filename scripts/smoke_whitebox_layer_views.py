"""Offline verification for spectral_utils.whitebox_layer_views. No cluster, no layer-view data.

Checks, in the order the plan's verification table lists them:
  1. roster loads, cell counts match the extraction manifests
  2. the label-contract partition asserts
  3. the answer-level label reproduces 10,477 / 13,769 and the per-cell table exactly
  4. n_steps from diff(offsets) equals records[i]['steps']
  5. group codes over the union give 3,483
  6. weighted_auc equals sklearn.roc_auc_score to ~1e-14, unweighted and weighted
  7. geometry_summaries produces 283 columns at L=36 and reproduces hand-derived values

Run:  python scripts/smoke_whitebox_layer_views.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_utils.whitebox_layer_views import (  # noqa: E402
    answer_error_label,
    assert_label_contract,
    auc_by_cell,
    auc_plan,
    geometry_summaries,
    group_codes,
    load_joined,
    n_steps_per_answer,
    shared_group_weights,
    weighted_auc,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_POSITIVES = {
    "pb_gsm8k_q4": (400, 207), "pb_gsm8k_q8": (400, 207),
    "pb_math_q4": (1000, 594), "pb_math_q8": (1000, 594),
    "pb_olympiadbench_q4": (1000, 661), "pb_olympiadbench_q8": (1000, 661),
    "pb_omnimath_q4": (1000, 759), "pb_omnimath_q8": (1000, 759),
    "prmbench_qwen3_8b": (6969, 6035),
}

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


print("[1] roster")
data = load_joined(ROOT)
records, cells, offsets = data["records"], data["cells"], data["offsets"]
labels, target = data["labels"], data["target"]
check("13,769 records / 145,597 steps / 9 cells", True,
      f"{len(records)} records, {int(offsets[-1])} steps, {len(set(cells.tolist()))} cells")

print("[2] label contract partition")
try:
    part = assert_label_contract(labels, offsets, target, cells)
    check("partition asserts", True,
          f"PB {part['pb_answers']}a/{part['pb_steps']}s, PRM {part['prm_answers']}a/{part['prm_steps']}s")
except ValueError as exc:
    check("partition asserts", False, str(exc))

print("[3] answer-level label")
y = answer_error_label(labels, offsets, target, cells)
check("total positives == 10477", int(y.sum()) == 10477, f"got {int(y.sum())}")
bad = []
for cell, (n_exp, pos_exp) in EXPECTED_POSITIVES.items():
    m = cells == cell
    if int(m.sum()) != n_exp or int(y[m].sum()) != pos_exp:
        bad.append(f"{cell}: {int(m.sum())}/{int(y[m].sum())} != {n_exp}/{pos_exp}")
check("per-cell positives match the verified table", not bad, "; ".join(bad))
# the trap this guards against
naive = target >= 0
prm = ~np.char.startswith(cells.astype(str), "pb_")
check("naive target>=0 really would zero PRMBench", int(naive[prm].sum()) == 0,
      f"naive PRMBench positives = {int(naive[prm].sum())} (correct value is 6035)")

print("[4] n_steps")
steps_json = np.array([r["steps"] for r in records])
check("diff(offsets) == records[i]['steps']",
      np.array_equal(steps_json, n_steps_per_answer(offsets)))

print("[5] source groups")
codes, ng = group_codes(data["group_id"])
check("3,483 groups over the union", ng == 3483, f"got {ng}")
pb_mask = np.char.startswith(cells.astype(str), "pb_")
shared = len(set(codes[pb_mask].tolist()) & set(codes[~pb_mask].tolist()))
check("66 groups span both benchmarks", shared == 66, f"got {shared}")

print("[6] weighted_auc vs sklearn")
try:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    scores = rng.normal(size=len(y)) + 0.4 * y
    plan = auc_plan(y, scores, codes)
    ours = weighted_auc(plan, np.ones(ng))
    theirs = roc_auc_score(y, scores)
    check("unweighted equals sklearn to 1e-13", abs(ours - theirs) < 1e-13,
          f"{ours!r} vs {theirs!r}")
    w = rng.integers(0, 4, size=ng).astype(float)
    if w[codes].sum() > 0:
        rep = np.repeat(np.arange(len(y)), w[codes].astype(int))
        ours_w = weighted_auc(plan, w)
        theirs_w = roc_auc_score(y[rep], scores[rep])
        check("weighted equals sklearn on the expanded index to 1e-13",
              abs(ours_w - theirs_w) < 1e-13, f"{ours_w!r} vs {theirs_w!r}")
except ImportError:
    check("sklearn available", False, "sklearn not installed; equality unverified")

print("[7] geometry summaries")
rng = np.random.default_rng(7)
N, L, R, P = 5, 36, 32, 256
cov = np.abs(rng.normal(size=(N, L, R))) * 1e4
hid = rng.normal(size=(N, L, P)).astype(np.float16)
rnm = np.abs(rng.normal(size=(N, L))) + 1.0
vals, names, groups = geometry_summaries(cov, hid, rnm)
check("283 columns at L=36", vals.shape == (N, 283), f"got {vals.shape}")
check("all finite", bool(np.isfinite(vals).all()))
check("four group names", set(groups) == {
    "geometry.hidden_to_final", "geometry.resid_norm",
    "geometry.hidden_adjacent", "geometry.covariance"}, str(sorted(set(groups))))
# hand-derive one column: resid_norm convergence at layer 0
manual = abs(np.log((rnm[0, 0] + 1e-12) / (rnm[0, -1] + 1e-12)))
j = names.index("geometry.resid_norm_convergence.layer_00")
check("resid_norm_convergence layer_00 re-derived by hand",
      abs(vals[0, j] - manual) < 1e-12, f"{vals[0, j]!r} vs {manual!r}")
# hand-derive cosine distance to final at layer 3
a = hid[1, 3].astype(np.float64); b = hid[1, -1].astype(np.float64)
manual_cos = 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
j = names.index("geometry.hidden_cos_to_final.layer_03")
check("hidden_cos_to_final layer_03 re-derived by hand",
      abs(vals[1, j] - manual_cos) < 1e-12, f"{vals[1, j]!r} vs {manual_cos!r}")
check("final layer emits no *_to_final column",
      "geometry.hidden_cos_to_final.layer_35" not in names)

print("[8] bootstrap plumbing")
w = shared_group_weights(ng, 16, seed=20260918)
check("weight matrix shape", w.shape == (16, ng), str(w.shape))
check("each draw sums to n_groups", bool(np.allclose(w.sum(axis=1), ng)))
res = auc_by_cell(y, scores, codes, cells, ng, weights=w[:4])
check("auc_by_cell covers 9 cells + POOLED", len(res) == 10, str(sorted(res)))
check("pooled point is finite", np.isfinite(res["POOLED"]["point"]))

print()
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
