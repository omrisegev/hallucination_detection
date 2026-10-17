"""Amendment R1 second pass — continuity row + Module-B 3x3 grid (label-free).

Runs after `run_v2.py structure` folds are frozen (safe per completed fold).
Writes ONLY additive artifacts per (cell x outer fold):

    scores_continuity.npz    fixed_family_cont_unguarded__{w,top10,spanmax,detector}
    moduleb_grid.npz         per (substrate x fuser): step scores + weights + status
    MANIFEST_AMEND_R1.json   sha256 of the two files above

    python scripts/joint_lsml_optimization_v2/second_pass_amendments.py [--cells ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from spectral_utils.feature_contract import confidence_sign_vector  # noqa: E402
from spectral_utils.fixed_application_pipelines import (  # noqa: E402
    SHARED_GLOBAL_FEATURES,
    SHARED_TOKEN_VIEWS,
)
from spectral_utils.fusion_utils import lsml_continuous, sml_fuse_signed  # noqa: E402
from spectral_utils.joint_lsml import (  # noqa: E402
    continuous_lsml_weight_vector,
    covariance_matrix,
    discover_loao_consensus_groups,
    fit_joint_lsml,
    hierarchical_joint_weights,
)
from spectral_utils.joint_lsml_localization import prepare_active23  # noqa: E402
from spectral_utils.joint_lsml_v2_localization import (  # noqa: E402
    DEPLOYED_IU_ROW,
    SUCCESSOR_S1,
    SUCCESSOR_S2,
    donor_scale_orient,
    provenance_labels,
)
from spectral_utils.token_local_fusion import IU_CONFIG  # noqa: E402
from spectral_utils.trajectory_reducer import (  # noqa: E402
    ORDERSTAT_K,
    reduce_with_weights,
    step_order_statistics,
)
from spectral_utils.upcr import upcr_fit  # noqa: E402

OUT = REPO / "results" / "joint_lsml_optimization_v2"
RETAINED_23 = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 19, 20, 21, 23, 24, 25, 26, 27, 28)
PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
PB_MODELS = ("q4", "q8")
PRM_CELL = "prmbench_qwen3_8b"
N_OUTER = 5
SEED = 20260907
SUBSTRATES = (DEPLOYED_IU_ROW, SUCCESSOR_S2, SUCCESSOR_S1)
FUSERS = ("sml", "iu", "joint")
CONTINUITY_ROW = "fixed_family_cont_unguarded"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cells() -> list[str]:
    return [f"pb_{s}_{t}" for s in PB_SUBSETS for t in PB_MODELS] + [PRM_CELL]


def _load_cell(cell_id: str):
    bundle = np.load(OUT / "cells" / f"{cell_id}.npz", allow_pickle=False)
    return {key: bundle[key] for key in bundle.files}


def _step_scores(risk: np.ndarray, cell) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts, ends = cell["step_starts"], cell["step_ends"]
    top10 = np.empty(len(starts))
    span_max = np.empty(len(starts))
    for index, (lo, hi) in enumerate(zip(starts, ends)):
        values = risk[int(lo):int(hi)]
        values = values[np.isfinite(values)]
        if values.size == 0:
            top10[index] = span_max[index] = np.nan
            continue
        k = min(10, len(values))
        top10[index] = float(np.partition(values, -k)[-k:].mean())
        span_max[index] = float(values.max())
    offsets = cell["token_offsets"]
    detector = np.asarray([
        float(np.nanmax(risk[int(offsets[i]):int(offsets[i + 1])]))
        for i in range(len(offsets) - 1)
    ])
    return top10, span_max, detector


def _trajectory_fuser_weights(
    fuser: str, Z_train_full: np.ndarray, owners_train_full: np.ndarray, *, seed: int
) -> tuple[np.ndarray | None, dict]:
    """Fit one trajectory fuser over the 10 z-scored order-stat views."""
    k = Z_train_full.shape[1]
    if fuser == "sml":
        _, weights = sml_fuse_signed(*[Z_train_full[:, i] for i in range(k)])
        return np.asarray(weights, float), {"fuser": "sml"}
    if fuser == "iu":
        fitted = upcr_fit(Z_train_full.T, **dict(IU_CONFIG))
        return np.asarray(fitted.w, float), {
            "fuser": "iu", "g2_hat": float(fitted.g2_hat),
        }
    if fuser == "joint":
        grouping = discover_loao_consensus_groups(
            Z_train_full, owners_train_full, k_range=(3,), seed=seed,
            minimum_group_size=3, pairwise_diagnostic_cap=32768,
            minimum_held_admissible_fraction=0.95, use_minimum_ari_tiebreak=True,
        )
        if grouping["status"] != "SELECTED":
            return None, {"fuser": "joint", "status": "BLOCKED_NO_ADMISSIBLE_PARTITION"}
        labels = np.asarray(grouping["labels"], dtype=np.int64)
        cov = covariance_matrix(Z_train_full)
        fit = fit_joint_lsml(cov, labels, anchor_index=0, seed=seed + 1)
        _, weights, meta = hierarchical_joint_weights(
            Z_train_full, labels, fit.global_loading, anchor_index=0, small_m_guard=True
        )
        return np.asarray(weights, float), {
            "fuser": "joint", "status": "SELECTED", "K": int(grouping["K"]),
            "group_sizes": [int(x) for x in grouping["group_sizes"]],
            "cross_small_m_guarded": bool(meta.get("cross_small_m_guarded", False)),
            "joint_converged": bool(fit.converged),
        }
    raise ValueError(fuser)


def _run_fold(cell_id: str, cell, outer_map: dict[str, int], k: int) -> None:
    out_dir = OUT / "structure" / cell_id / f"outer{k}"
    if not (out_dir / "COMPLETE.json").exists():
        print(f"[{cell_id}] outer{k}: structure not frozen yet — skipped", flush=True)
        return
    amend_manifest = out_dir / "MANIFEST_AMEND_R1.json"
    if amend_manifest.exists():
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
    values = np.asarray(prep.standardized_fit)
    entropy_index = prep.feature_names.index("entropy_series")

    # ── continuity row: exact historical fit (unguarded), v2 boundary ────────
    prov = provenance_labels(prep.family_names)
    _, meta = lsml_continuous(
        *[values[:, i] for i in range(values.shape[1])],
        groups=prov, compute_score_matrix=False, small_m_guard=False,
    )
    weight = continuous_lsml_weight_vector(meta, values.shape[1])
    weight, scale_meta = donor_scale_orient(weight, values, entropy_index=entropy_index)
    risk = prep.token_risk(weight)
    top10, span_max, detector = _step_scores(risk, cell)
    np.savez_compressed(
        out_dir / "scores_continuity.npz",
        **{
            f"{CONTINUITY_ROW}__w": weight,
            f"{CONTINUITY_ROW}__top10": top10.astype(np.float32),
            f"{CONTINUITY_ROW}__spanmax": span_max.astype(np.float32),
            f"{CONTINUITY_ROW}__detector": detector.astype(np.float32),
        },
    )

    # ── Module-B 3x3 grid ────────────────────────────────────────────────────
    frozen = np.load(out_dir / "scores_outer.npz", allow_pickle=False)
    step_rows = np.repeat(np.arange(len(cell["row_ids"])), np.diff(cell["step_row_offsets"]))
    train_steps = mask_train[step_rows]
    grid_arrays: dict[str, np.ndarray] = {}
    grid_meta: dict[str, dict] = {}
    for substrate in SUBSTRATES:
        key = f"{substrate}__w"
        if key not in frozen.files:
            grid_meta[substrate] = {"status": "SUBSTRATE_MISSING"}
            continue
        sub_risk = prep.token_risk(np.asarray(frozen[key], float))
        matrix, lengths = step_order_statistics(sub_risk, cell["step_starts"], cell["step_ends"])
        full = train_steps & (lengths >= ORDERSTAT_K)
        if int(full.sum()) < 50:
            grid_meta[substrate] = {"status": "TOO_FEW_FULL_STEPS", "n": int(full.sum())}
            continue
        mu = matrix[full].mean(axis=0)
        sd = matrix[full].std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Z_all = (matrix - mu[None, :]) / sd[None, :]
        owners_full = step_rows[full]
        grid_arrays[f"{substrate}__b0"] = reduce_with_weights(
            matrix, lengths, np.ones(ORDERSTAT_K) / ORDERSTAT_K
        ).astype(np.float32)
        for fuser in FUSERS:
            combo = f"{substrate}__{fuser}"
            try:
                weights10, fuser_meta = _trajectory_fuser_weights(
                    fuser, Z_all[full], owners_full, seed=SEED + 17 * k
                )
                if weights10 is None:
                    grid_meta[combo] = fuser_meta
                    continue
                score_full = Z_all[full] @ weights10
                sd_score = float(score_full.std())
                if not np.isfinite(sd_score) or sd_score < 1e-8:
                    grid_meta[combo] = {**fuser_meta, "status": "SD_FLOOR"}
                    continue
                weights10 = weights10 / sd_score
                anchor = Z_all[full].mean(axis=1)
                corr = float(np.corrcoef(Z_all[full] @ weights10, anchor)[0, 1])
                if np.isfinite(corr) and corr < 0.0:
                    weights10 = -weights10
                scores = reduce_with_weights(Z_all, lengths, weights10)
                grid_arrays[f"{combo}__scores"] = scores.astype(np.float32)
                grid_arrays[f"{combo}__weights"] = weights10
                grid_meta[combo] = {**fuser_meta, "status": fuser_meta.get("status", "OK"),
                                    "anchor_correlation": corr}
            except Exception as error:
                grid_meta[combo] = {"fuser": fuser, "status": f"{type(error).__name__}: {error}"}
    np.savez_compressed(out_dir / "moduleb_grid.npz", **grid_arrays)
    (out_dir / "moduleb_grid_meta.json").write_text(
        json.dumps({"grid": grid_meta, "continuity_scale": scale_meta,
                    "labels_accessed": False}, indent=1, default=str),
        encoding="utf-8",
    )
    amend_manifest.write_text(json.dumps({
        "scores_continuity.npz": _sha(out_dir / "scores_continuity.npz"),
        "moduleb_grid.npz": _sha(out_dir / "moduleb_grid.npz"),
        "moduleb_grid_meta.json": _sha(out_dir / "moduleb_grid_meta.json"),
    }, indent=1), encoding="utf-8")
    print(f"[{cell_id}] outer{k}: amendment artifacts frozen", flush=True)


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
    print(f"second pass done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
