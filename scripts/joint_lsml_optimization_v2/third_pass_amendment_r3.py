"""Amendment R3 third pass — Hook 3a/3b lambda=0 inertness reference row (label-free).

Runs after ALL structure + R1 processes have ended (module edit ordering constraint, see
docs/experiments/JOINT_LSML_OPTIMIZATION_V2_AMENDMENT_R3.md). Per (cell x OUTER fold) it
re-derives the shared INTERNAL joint fit through `fit_v2_arms` with the single descriptive
row `internal_joint_modelinv_lam0` (same seed / grouping / preparation as the frozen R4 and
R10-R13 rows) and writes ONLY additive artifacts:

    scores_amend_r3.npz      internal_joint_modelinv_lam0__{w,top10,spanmax,detector}
    meta_amend_r3.json       row meta + fallback events + labels_accessed=False
    MANIFEST_AMEND_R3.json   sha256 of the two files above (exact relative paths)

    python scripts/joint_lsml_optimization_v2/third_pass_amendment_r3.py [--cells ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from second_pass_amendments import (  # noqa: E402
    OUT, N_OUTER, PRM_CELL, RETAINED_23, _cells, _load_cell, _sha, _step_scores,
)
from spectral_utils.feature_contract import confidence_sign_vector  # noqa: E402
from spectral_utils.fixed_application_pipelines import (  # noqa: E402
    SHARED_GLOBAL_FEATURES, SHARED_TOKEN_VIEWS,
)
from spectral_utils.joint_lsml_localization import prepare_active23  # noqa: E402
from spectral_utils.joint_lsml_v2_localization import (  # noqa: E402
    HOOK3_LAMBDA0_REFERENCE_ROW, fit_v2_arms,
)

RUN_SEED = 20260905   # run_v2.SEED — the frozen outer lanes used SEED + 100*k
ROW = HOOK3_LAMBDA0_REFERENCE_ROW


def _run_fold(cell_id: str, cell, outer_map: dict[str, int], k: int) -> None:
    out_dir = OUT / "structure" / cell_id / f"outer{k}"
    if not (out_dir / "COMPLETE.json").exists():
        print(f"[{cell_id}] outer{k}: structure not frozen yet — skipped", flush=True)
        return
    manifest = out_dir / "MANIFEST_AMEND_R3.json"
    if manifest.exists():
        return
    groups = [str(g) for g in cell["group_ids"]]
    mask_train = np.asarray([outer_map.get(g, -1) not in (k, -1) for g in groups], bool)
    prep = prepare_active23(
        cell["raw"], cell["token_offsets"], [str(r) for r in cell["row_ids"]],
        retained_indices=list(RETAINED_23),
        confidence_signs_29=confidence_sign_vector(SHARED_GLOBAL_FEATURES),
        stream_names_29=SHARED_TOKEN_VIEWS,
        raw_feature_names_29=SHARED_GLOBAL_FEATURES,
        fit_row_mask=mask_train,
    )
    fitted = fit_v2_arms(
        prep, seed=RUN_SEED + 100 * k, cell_key=cell_id,
        domain="prmbench" if cell_id == PRM_CELL else "processbench",
        rows=[ROW], include_iu=False,
    )
    arrays: dict[str, np.ndarray] = {}
    if ROW in fitted["weights"]:
        weight = np.asarray(fitted["weights"][ROW], dtype=np.float64)
        risk = prep.token_risk(weight)
        top10, span_max, detector = _step_scores(risk, cell)
        arrays[f"{ROW}__w"] = weight
        arrays[f"{ROW}__top10"] = top10.astype(np.float32)
        arrays[f"{ROW}__spanmax"] = span_max.astype(np.float32)
        arrays[f"{ROW}__detector"] = detector.astype(np.float32)
    np.savez_compressed(out_dir / "scores_amend_r3.npz", **arrays)
    # fidelity check against the frozen R4 row: same shared fit => same grouping source
    frozen_meta = json.loads((out_dir / "meta_outer.json").read_text(encoding="utf-8"))
    frozen_src = frozen_meta.get("row_meta", {}).get("internal_joint", {}).get("grouping")
    (out_dir / "meta_amend_r3.json").write_text(json.dumps({
        "cell": cell_id, "outer": k, "seed": RUN_SEED + 100 * k, "row": ROW,
        "row_meta": fitted["row_meta"].get(ROW), "failures": fitted["failures"],
        "fallback_events": fitted["fallback_events"],
        "frozen_internal_joint_grouping": frozen_src,
        "grouping_matches_frozen_r4": (fitted["row_meta"].get(ROW, {}).get("grouping") == frozen_src),
        "labels_accessed": False,
    }, indent=1, default=str), encoding="utf-8")
    manifest.write_text(json.dumps({
        "scores_amend_r3.npz": _sha(out_dir / "scores_amend_r3.npz"),
        "meta_amend_r3.json": _sha(out_dir / "meta_amend_r3.json"),
    }, indent=1), encoding="utf-8")
    status = "frozen" if ROW in fitted["weights"] else f"FAILED {fitted['failures']}"
    print(f"[{cell_id}] outer{k}: R3 reference row {status}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="*", default=None)
    args = parser.parse_args()
    folds = json.loads((OUT / "folds" / "folds.json").read_text(encoding="utf-8"))
    started = time.time()
    for cell_id in (args.cells or _cells()):
        cell = _load_cell(cell_id)
        panel = "prmbench" if cell_id == PRM_CELL else "processbench"
        for k in range(N_OUTER):
            _run_fold(cell_id, cell, folds[panel]["outer"], k)
    print(f"third pass done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
