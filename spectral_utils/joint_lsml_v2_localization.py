"""Joint L-SML optimization v2 — fold-scoped arm fitting for the tuned study.

Protocol: docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V2.md
Prior-order audit: docs/experiments/PRIOR_ORDER_AUDIT_JOINT_LSML_OPTIMIZATION_V2.md

This module has no target/label API.  It consumes a fold-scoped
:class:`~spectral_utils.joint_lsml_localization.Active23Preparation` (built
with ``fit_row_mask`` = the outer/inner training rows) and returns one
SD=1-normalized, consistently oriented (23,) weight vector per registered arm.

Every learned arm passes through :func:`donor_scale_orient`:

1. rescale by 1 / SD(Z_fit @ w) on the fit population (floor 1e-8, fail-closed);
2. orient by the standardized-rowmean Pearson rule; fallback to the
   entropy_series Spearman rule when |corr| < 0.02; both undefined -> the arm
   is inadmissible on this lane (``ORIENTATION_UNDETERMINED``).

Roster (Section 4.2 of the protocol): R1-R16 as registered.  The small-m guard
(Steps 203/205) replaces any 3-unit SML eigen-stage with equal weights over
SD-standardized units; provenance-family CONT arms must be untouched by it
(their families are validated to contain no 3-stream group only via the
registered exemption assert in the runner).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .adapted_dufs import adapted_dufs_soft_gates
from .fusion_utils import lsml_continuous
from .joint_lsml import (
    continuous_lsml_weight_vector,
    covariance_matrix,
    discover_loao_consensus_groups,
    effective_gates,
    fit_joint_lsml,
    gated_joint_hierarchical_fit,
    hierarchical_joint_weights,
    regularized_joint_map_weights,
)
from .joint_lsml_localization import (
    Active23Preparation,
    K_RANGE,
    MINIMUM_HELD_ADMISSIBLE_FRACTION,
    PAIRWISE_DIAGNOSTIC_CAP,
)
from .specrage_views import VIEW_ORDER
from .token_local_fusion import IU_CONFIG
from .upcr import upcr_fit


SD_FLOOR = 1e-8
ORIENTATION_PEARSON_FLOOR = 0.02
DUFS_MIN_SURVIVORS = 9
GATE_SEED_STD_CAP = 0.15

# ── IU family: the 16-row full cross (Section 4.1) ───────────────────────────

IU_ROSTER: tuple[tuple[str, dict[str, Any]], ...] = tuple(
    (
        f"iu_c{components}_s{int(scale * 100):02d}_{loss}_{'exon' if exclusion else 'exoff'}",
        {
            "n_components": components,
            "auto_components": False,
            "scale_ratio": scale,
            "loss": loss,
            "g2_projection_k": 1,
            "exclusion": exclusion,
            "recompute_after_exclusion": exclusion,
            "simple_avg_fallback": False,
            "difficulty_gate": False,
        },
    )
    for components in (2, 1)
    for scale in (0.25, 0.10)
    for loss in ("l2", "l1")
    for exclusion in (False, True)
)

DEPLOYED_IU_ROW = "iu_c2_s25_l2_exoff"       # == IU_CONFIG
DEPLOYED_UPCR_PORT_ROW = "iu_c2_s25_l2_exon"  # Step-341 P3D1 recipe at token level

# ── L-SML family: the 16 registered rows (Section 4.2) ───────────────────────

LSML_ROSTER: tuple[str, ...] = (
    "prov5_cont",              # R1  (lambda=0 anchor; == fixed-family control estimator)
    "prov5_joint",             # R2  (provenance-merged groups, see provenance_merged_labels)
    "internal_cont",           # R3  (= S2)
    "internal_joint",          # R4  (= S1)
    "prov5_cont_gate050",      # R5
    "prov5_cont_gate100",      # R6
    "internal_cont_gate100",   # R7
    "internal_joint_gate050",  # R8
    "internal_joint_gate100",  # R9
    "internal_joint_liu010",   # R10
    "internal_joint_liu050",   # R11
    "internal_joint_diag010",  # R12
    "internal_joint_diag050",  # R13
    "internal_gaff_cont",      # R14
    "internal_gaff_joint",     # R15
    "dufs_pf_lsml",            # R16
)

SUCCESSOR_S1 = "internal_joint"
SUCCESSOR_S2 = "internal_cont"

EQUAL_ALL23_METHOD = "equal_all23"
EQUAL_FAMILY_METHOD = "equal_family_active23"
FIXED_FAMILY_METHOD = "fixed_family_continuous_lsml_active23"


def donor_scale_orient(
    weight: Sequence[float],
    fit_values: np.ndarray,
    *,
    entropy_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """SD=1 + unified orientation invariant (protocol Sections 3.1-3.2)."""
    matrix = np.asarray(fit_values, dtype=np.float64)
    output = np.asarray(weight, dtype=np.float64).copy()
    if output.shape != (matrix.shape[1],) or not np.isfinite(output).all():
        raise ValueError("weight vector is malformed")
    score = matrix @ output
    sd = float(score.std())
    if not np.isfinite(sd) or sd < SD_FLOOR:
        raise RuntimeError("SD_FLOOR: donor fused-score SD is degenerate")
    output = output / sd
    score = score / sd
    anchor = matrix.mean(axis=1)
    pearson = float(np.corrcoef(score, anchor)[0, 1]) if anchor.std() > 0 else float("nan")
    rule = "rowmean_pearson"
    correlation = pearson
    if not np.isfinite(pearson) or abs(pearson) < ORIENTATION_PEARSON_FLOOR:
        entropy = matrix[:, int(entropy_index)]
        if entropy.std() > 0 and score.std() > 0:
            from scipy.stats import spearmanr

            correlation = float(spearmanr(score, entropy).statistic)
            rule = "entropy_spearman_fallback"
        else:
            correlation = float("nan")
    if not np.isfinite(correlation):
        raise RuntimeError("ORIENTATION_UNDETERMINED: both anchors undefined")
    flipped = bool(correlation < 0.0)
    if flipped:
        output = -output
    return output, {
        "scale_sd": sd,
        "orientation_rule": rule,
        "anchor_correlation": correlation,
        "orientation_flipped": flipped,
        "rowmean_pearson": pearson,
    }


def provenance_labels(family_names: Sequence[str]) -> np.ndarray:
    """The frozen provenance-family partition (order = VIEW_ORDER)."""
    order = [name for name in VIEW_ORDER if name in family_names]
    mapping = {name: index for index, name in enumerate(order)}
    return np.asarray([mapping[str(name)] for name in family_names], dtype=np.int64)


def provenance_merged_labels(family_names: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Provenance families with every <3-stream family merged into ONE group.

    Deterministic, data-independent (depends only on the frozen roster), and
    pre-registered: `fit_joint_lsml` requires every group size >= 3, which the
    raw provenance families violate (they contain a singleton and a pair on
    the active-23 roster).  All undersized families form a single merged group.
    """
    base = provenance_labels(family_names)
    sizes = {int(label): int(np.sum(base == label)) for label in np.unique(base)}
    small = sorted(label for label, size in sizes.items() if size < 3)
    if not small:
        return base, {"merged_families": [], "merged_group_size": 0}
    merged = base.copy()
    target = small[0]
    for label in small[1:]:
        merged[base == label] = target
    relabeled = np.unique(merged, return_inverse=True)[1].astype(np.int64)
    merged_size = int(np.sum(np.isin(base, small)))
    if merged_size < 3:
        raise RuntimeError(
            "provenance-merged joint groups still contain a group smaller than three"
        )
    return relabeled, {
        "merged_families": [int(label) for label in small],
        "merged_group_size": merged_size,
    }


def compute_soft_gates(fit_values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """RMS-normalized nonnegative soft gates on the fit population (frozen learner)."""
    gates, diagnostics = adapted_dufs_soft_gates(np.asarray(fit_values, dtype=np.float64).T)
    gates = np.asarray(gates, dtype=np.float64)
    if not np.isfinite(gates).all() or np.any(gates < 0.0):
        raise RuntimeError("soft gates are not finite nonnegative")
    return gates, {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in dict(diagnostics).items()
        if key != "training_history"
    }


def dufs_hard_survivors(
    fit_values: np.ndarray, *, cell_key: str, domain: str, seed: int = 0
) -> tuple[np.ndarray, dict[str, Any]]:
    """R16: the exact historical a2.dufs_pf hard rule (signed mu > 0)."""
    from .selectors.a2_groupfs import dufs_pf_cell_rng, dufs_pf_gates

    rng = dufs_pf_cell_rng(str(cell_key), str(domain), seed=int(seed))
    mu = np.asarray(dufs_pf_gates(np.asarray(fit_values, dtype=np.float64), rng), dtype=np.float64)
    if mu.shape != (fit_values.shape[1],) or not np.isfinite(mu).all():
        raise RuntimeError("parameter-free DUFS returned invalid gates")
    survivors = np.flatnonzero(mu > 0.0)
    if len(survivors) < DUFS_MIN_SURVIVORS:
        raise RuntimeError(
            f"DUFS_PF_FAIL_CLOSED: {len(survivors)} survivors < {DUFS_MIN_SURVIVORS}"
        )
    return survivors, {"gate_means": mu.tolist(), "threshold": 0.0, "cell_rng_seed": int(seed)}


def _cont_weight(
    fit_values: np.ndarray,
    labels: np.ndarray,
    *,
    gates: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    _, meta = lsml_continuous(
        *[fit_values[:, index] for index in range(fit_values.shape[1])],
        groups=labels, compute_score_matrix=False,
        gates=gates, small_m_guard=True,
    )
    weight = continuous_lsml_weight_vector(meta, fit_values.shape[1])
    return weight, {
        "K": int(meta["K"]),
        "gates_applied": bool(meta.get("gates_applied", False)),
        "small_m_flags": list(meta.get("small_m_flags", [])),
        "small_m_guarded": list(meta.get("small_m_guarded", [])),
    }


def _joint_weight(
    fit_values: np.ndarray,
    labels: np.ndarray,
    *,
    anchor_index: int,
    seed: int,
    gates_effective: np.ndarray | None,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    if gates_effective is None:
        covariance = covariance_matrix(fit_values)
        fit = fit_joint_lsml(covariance, labels, anchor_index=anchor_index, seed=seed)
        _, weight, meta = hierarchical_joint_weights(
            fit_values, labels, fit.global_loading,
            anchor_index=anchor_index, small_m_guard=True,
        )
        return np.asarray(weight, dtype=np.float64), fit, dict(meta)
    weight, fit, meta = gated_joint_hierarchical_fit(
        fit_values, labels, gates_effective,
        anchor_index=anchor_index, seed=seed, small_m_guard=True,
    )
    return weight, fit, meta


def fit_v2_arms(
    preparation: Active23Preparation,
    *,
    seed: int,
    cell_key: str,
    domain: str,
    rows: Sequence[str] | None = None,
    include_iu: bool = True,
    gate_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit every registered v2 arm on one fold-scoped preparation.

    Returns weights + per-row metadata; a row that fails structurally is
    recorded (status + reason) without blocking the other rows.  Grouping,
    gates, and joint fits are shared across rows via internal caches.
    """
    values = np.asarray(preparation.standardized_fit, dtype=np.float64)
    entropy_index = preparation.feature_names.index("entropy_series")
    requested = tuple(rows) if rows is not None else LSML_ROSTER
    unknown = set(requested) - set(LSML_ROSTER)
    if unknown:
        raise ValueError(f"unknown roster rows: {sorted(unknown)}")

    weights: dict[str, np.ndarray] = {}
    row_meta: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    fallback_events: list[dict[str, Any]] = []

    def _admit(row_id: str, raw_weight: np.ndarray, meta: Mapping[str, Any]) -> None:
        oriented, scale_meta = donor_scale_orient(raw_weight, values, entropy_index=entropy_index)
        weights[row_id] = oriented
        row_meta[row_id] = {**dict(meta), **scale_meta}

    # ── shared structure ────────────────────────────────────────────────────
    needs_gates = any("gate" in row or "liu" in row or "diag" in row or "gaff" in row for row in requested)
    gates = None
    gate_diag: dict[str, Any] = {}
    if needs_gates:
        cache_key = "soft_gates"
        if gate_cache is not None and cache_key in gate_cache:
            gates, gate_diag = gate_cache[cache_key]
        else:
            gates, gate_diag = compute_soft_gates(values)
            if gate_cache is not None:
                gate_cache[cache_key] = (gates, gate_diag)

    prov_labels = provenance_labels(preparation.family_names)
    prov_merged, merge_diag = provenance_merged_labels(preparation.family_names)

    internal = None
    if any(row.startswith("internal_") and "gaff" not in row for row in requested):
        internal = discover_loao_consensus_groups(
            values, preparation.fit_row_indices, k_range=K_RANGE, seed=int(seed),
            minimum_group_size=3, pairwise_diagnostic_cap=PAIRWISE_DIAGNOSTIC_CAP,
            minimum_held_admissible_fraction=MINIMUM_HELD_ADMISSIBLE_FRACTION,
            use_minimum_ari_tiebreak=True,
        )
    gaff = None
    if any("gaff" in row for row in requested):
        gaff = discover_loao_consensus_groups(
            values * gates[None, :], preparation.fit_row_indices, k_range=K_RANGE,
            seed=int(seed) + 777, minimum_group_size=3,
            pairwise_diagnostic_cap=PAIRWISE_DIAGNOSTIC_CAP,
            minimum_held_admissible_fraction=MINIMUM_HELD_ADMISSIBLE_FRACTION,
            use_minimum_ari_tiebreak=True,
        )

    def _resolve_labels(grouping: Mapping[str, Any] | None, *, joint_map: bool, row_id: str) -> tuple[np.ndarray, str]:
        """INTERNAL labels, falling back to provenance (same map type) when blocked."""
        if grouping is not None and grouping.get("status") == "SELECTED":
            return np.asarray(grouping["labels"], dtype=np.int64), "internal"
        fallback_events.append({
            "row": row_id,
            "reason": "BLOCKED_NO_ADMISSIBLE_PARTITION" if grouping is not None else "GROUPING_NOT_COMPUTED",
            "fallback": "provenance_merged" if joint_map else "provenance",
        })
        return (prov_merged if joint_map else prov_labels), (
            "provenance_merged_fallback" if joint_map else "provenance_fallback"
        )

    joint_fit_cache: dict[str, Any] = {}

    def _internal_joint_fit() -> tuple[Any, np.ndarray, str]:
        """The ungated INTERNAL joint fit (shared by R4 and R10-R13)."""
        if "internal_joint" not in joint_fit_cache:
            labels, source = _resolve_labels(internal, joint_map=True, row_id="internal_joint_shared")
            covariance = covariance_matrix(values)
            fit = fit_joint_lsml(covariance, labels, anchor_index=entropy_index, seed=int(seed) + 10_000)
            joint_fit_cache["internal_joint"] = (fit, labels, source)
        return joint_fit_cache["internal_joint"]

    # ── the sixteen rows ────────────────────────────────────────────────────
    for row_id in requested:
        try:
            if row_id == "prov5_cont":
                weight, meta = _cont_weight(values, prov_labels, gates=None)
                _admit(row_id, weight, {**meta, "grouping": "provenance"})
            elif row_id == "prov5_joint":
                weight, fit, meta = _joint_weight(
                    values, prov_merged, anchor_index=entropy_index,
                    seed=int(seed) + 11_000, gates_effective=None,
                )
                _admit(row_id, weight, {**meta, "grouping": "provenance_merged", **merge_diag,
                                        "joint_converged": bool(fit.converged)})
            elif row_id == "internal_cont":
                labels, source = _resolve_labels(internal, joint_map=False, row_id=row_id)
                weight, meta = _cont_weight(values, labels, gates=None)
                _admit(row_id, weight, {**meta, "grouping": source})
            elif row_id == "internal_joint":
                fit, labels, source = _internal_joint_fit()
                _, weight, meta = hierarchical_joint_weights(
                    values, labels, fit.global_loading,
                    anchor_index=entropy_index, small_m_guard=True,
                )
                _admit(row_id, np.asarray(weight, dtype=np.float64),
                       {**dict(meta), "grouping": source, "joint_converged": bool(fit.converged)})
            elif row_id in ("prov5_cont_gate050", "prov5_cont_gate100"):
                lam = 0.5 if row_id.endswith("050") else 1.0
                q_eff = effective_gates(gates, lam, values.shape[1])
                weight, meta = _cont_weight(values, prov_labels, gates=q_eff)
                _admit(row_id, weight, {**meta, "grouping": "provenance", "hook": "hook2", "lambda": lam})
            elif row_id == "internal_cont_gate100":
                labels, source = _resolve_labels(internal, joint_map=False, row_id=row_id)
                q_eff = effective_gates(gates, 1.0, values.shape[1])
                weight, meta = _cont_weight(values, labels, gates=q_eff)
                _admit(row_id, weight, {**meta, "grouping": source, "hook": "hook2", "lambda": 1.0})
            elif row_id in ("internal_joint_gate050", "internal_joint_gate100"):
                lam = 0.5 if row_id.endswith("050") else 1.0
                labels, source = _resolve_labels(internal, joint_map=True, row_id=row_id)
                q_eff = effective_gates(gates, lam, values.shape[1])
                weight, fit, meta = _joint_weight(
                    values, labels, anchor_index=entropy_index,
                    seed=int(seed) + 12_000, gates_effective=q_eff,
                )
                _admit(row_id, weight, {**meta, "grouping": source, "hook": "hook2_joint",
                                        "lambda": lam, "joint_converged": bool(fit.converged)})
            elif row_id in ("internal_joint_liu010", "internal_joint_liu050",
                            "internal_joint_diag010", "internal_joint_diag050"):
                mode = "liu" if "liu" in row_id else "diag"
                lam = 0.1 if row_id.endswith("010") else 0.5
                fit, labels, source = _internal_joint_fit()
                weight, meta = regularized_joint_map_weights(
                    values, fit.model_covariance, fit.global_loading,
                    mode=mode, lam=lam, gates=gates,
                )
                _admit(row_id, weight, {**meta, "grouping": source,
                                        "hook": f"hook3{'a' if mode == 'liu' else 'b'}"})
            elif row_id in ("internal_gaff_cont", "internal_gaff_joint"):
                joint_map = row_id.endswith("joint")
                labels, source = _resolve_labels(gaff, joint_map=joint_map, row_id=row_id)
                if joint_map:
                    weight, fit, meta = _joint_weight(
                        values, labels, anchor_index=entropy_index,
                        seed=int(seed) + 13_000, gates_effective=None,
                    )
                    meta = {**meta, "joint_converged": bool(fit.converged)}
                else:
                    weight, meta = _cont_weight(values, labels, gates=None)
                _admit(row_id, weight, {**meta, "grouping": f"gated_affinity::{source}", "hook": "hook1"})
            elif row_id == "dufs_pf_lsml":
                survivors, dufs_meta = dufs_hard_survivors(
                    values, cell_key=cell_key, domain=domain, seed=0
                )
                sub_labels = provenance_labels(
                    [preparation.family_names[index] for index in survivors]
                )
                sub_weight, meta = _cont_weight(values[:, survivors], sub_labels, gates=None)
                weight = np.zeros(values.shape[1], dtype=np.float64)
                weight[survivors] = sub_weight
                _admit(row_id, weight, {**meta, **dufs_meta,
                                        "grouping": "provenance_survivors",
                                        "n_selected": int(len(survivors)), "hook": "hard_selector"})
            else:  # pragma: no cover - roster is validated above
                raise RuntimeError(f"unhandled roster row {row_id}")
        except Exception as error:  # fail-closed per row, never per study
            failures[row_id] = f"{type(error).__name__}: {error}"

    # ── IU family + named controls ──────────────────────────────────────────
    if include_iu:
        for row_id, config in IU_ROSTER:
            try:
                fitted = upcr_fit(values.T, **config)
                _admit(row_id, np.asarray(fitted.w, dtype=np.float64), {
                    "family": "iu", "config": dict(config),
                    "g2_hat": float(fitted.g2_hat),
                    "projection_residual": float(fitted.proj_residual),
                })
            except Exception as error:
                failures[row_id] = f"{type(error).__name__}: {error}"

    try:
        equal = np.ones(values.shape[1], dtype=np.float64) / values.shape[1]
        _admit(EQUAL_ALL23_METHOD, equal, {"family": "control"})
    except Exception as error:
        failures[EQUAL_ALL23_METHOD] = f"{type(error).__name__}: {error}"
    try:
        present = [name for name in VIEW_ORDER if name in preparation.family_names]
        family_weight = np.zeros(values.shape[1], dtype=np.float64)
        for family in present:
            indices = np.flatnonzero(np.asarray(preparation.family_names) == family)
            family_weight[indices] = 1.0 / (len(present) * len(indices))
        _admit(EQUAL_FAMILY_METHOD, family_weight, {"family": "control", "present_families": present})
    except Exception as error:
        failures[EQUAL_FAMILY_METHOD] = f"{type(error).__name__}: {error}"

    # deployed IU_CONFIG sanity: the named deployed row must equal the grid row
    deployed_matches_grid = None
    if include_iu and DEPLOYED_IU_ROW in weights:
        try:
            reference = upcr_fit(values.T, **dict(IU_CONFIG))
            oriented, _ = donor_scale_orient(reference.w, values, entropy_index=entropy_index)
            deployed_matches_grid = bool(np.allclose(oriented, weights[DEPLOYED_IU_ROW], atol=1e-10))
        except Exception:
            deployed_matches_grid = False

    return {
        "weights": weights,
        "row_meta": row_meta,
        "failures": failures,
        "fallback_events": fallback_events,
        "internal_grouping_status": None if internal is None else str(internal.get("status")),
        "internal_K": (int(internal["K"]) if internal is not None and internal.get("status") == "SELECTED" else None),
        "gated_affinity_grouping_status": None if gaff is None else str(gaff.get("status")),
        "gate_diagnostics": gate_diag,
        "gate_seed_std_cap": GATE_SEED_STD_CAP,
        "provenance_merge": merge_diag,
        "deployed_iu_matches_grid": deployed_matches_grid,
        "labels_accessed": False,
    }


__all__ = [
    "DEPLOYED_IU_ROW", "DEPLOYED_UPCR_PORT_ROW", "DUFS_MIN_SURVIVORS",
    "EQUAL_ALL23_METHOD", "EQUAL_FAMILY_METHOD", "FIXED_FAMILY_METHOD",
    "GATE_SEED_STD_CAP", "IU_ROSTER", "LSML_ROSTER", "ORIENTATION_PEARSON_FLOOR",
    "SD_FLOOR", "SUCCESSOR_S1", "SUCCESSOR_S2", "compute_soft_gates",
    "donor_scale_orient", "dufs_hard_survivors", "fit_v2_arms",
    "provenance_labels", "provenance_merged_labels",
]
