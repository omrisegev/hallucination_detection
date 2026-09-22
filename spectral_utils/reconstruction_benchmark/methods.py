"""Unified, label-free implementations of the 13 primary core arms.

This module is a thin orchestration layer over the project's canonical method
implementations.  It does not load data, preprocess feature columns, open
targets, or compute evaluation metrics.  Its sole input is a
:class:`~spectral_utils.reconstruction_benchmark.contracts.PreparedCell`.

Family-NRM-A and PGRD-A are intentionally marked as new within-cell ablations.
They use neither donor cells nor target labels.  Their fixed rules are kept
here, next to their exact configuration hashes, so a later run cannot silently
turn them into the older cross-dataset variants.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import importlib
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import (
    CONTRACT_VERSION,
    FitStatus,
    MethodSpec,
    PreparedCell,
    ScoreResult,
    prepared_matrix_sha256,
)


_EPS = 1e-12
_ORIENTATION_TOLERANCE = 1e-6

_UPCR_CONFIG = {
    "loss": "l2",
    "exclusion": True,
    "difficulty_gate": False,
    "simple_avg_fallback": True,
    "recompute_after_exclusion": True,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
    "min_frac": 0.05,
    "exclude_frac": 3.0,
}

_IU_CONFIG = {
    "loss": "l2",
    "exclusion": False,
    "difficulty_gate": False,
    "simple_avg_fallback": False,
    "recompute_after_exclusion": False,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
    "n_components": 2,
    "auto_components": False,
}

_DUFS_LIU_CONFIG = {
    "seeds": (11, 23, 37),
    "epochs": 80,
    "graph_k": 7,
    "lambda": 0.1,
    "duplicate_safe_graph": True,
    "tie_break": "lexicographic row_id rank",
}

_SU_PCR_CONFIG = {
    "scale_ratio": 0.25,
    "rank": 2,
    "n_components": 2,
    "g2_projection_components": 1,
    "g2_grid": 300,
    "threshold_multiplier": 1.0,
    "max_iter": 100,
    "inner_completion_iter": 40,
    "decomposition_tol": 1e-8,
    "max_sparse_fraction": None,
    "target_condition": 100.0,
}

_CA_CONFIG = {
    "output_dim": 2,
    "n_neighbors": 15,
    "temperature": 1.0,
    "learning_rate": 1e-2,
    "batch_size": 128,
    "max_epochs": 60,
    "min_epochs": 60,
    "patience": 61,
    "lr_patience": 20,
    "encoder_hidden": (32,),
    "fusion_hidden": (50,),
    "checkpoint_mode": "final",
    "orthogonalization": "svd_floor",
    "orthogonal_floor": 1e-3,
    "agreement_strength": 2.0,
    "agreement_temperature": 0.08,
    "edge_mass_strength": 0.1,
    "view_mass_normalization": True,
    "fit_sample_cap": 1500,
    "model_seeds": (11, 23),
    "liu_lambda": 10.0,
    "atomic_prior": "equal provenance-family mass; equal within family",
    "duplicate_safe_graph": True,
    "tie_break": "lexicographic row_id rank",
}

_DEEM_CONFIG = {
    "seeds": (0, 1, 2, 3, 4),
    "family_width": 8,
    "epochs": 100,
    "learning_rate": 1e-3,
    "momentum": 0.0,
    "mala_delta": 0.10,
    "mala_steps": 5,
    "replay_refresh": 0.05,
    "dtype": "float64",
    "device": "cpu",
    "anchor_tolerance": 1e-6,
    "posterior_sd_min": 1e-3,
    "init_sd": 0.005,
    "deterministic": True,
    "minimum_median_seed_spearman": 0.90,
    "aggregation": "mean of all five healthy seed posteriors; no survivor averaging",
}

_FAMILY_NRM_A_CONFIG = {
    "regime": "A_within_cell_fully_unsupervised",
    "base": "iu_pcr",
    "minimum_present_families": 3,
    "residual_covariance_denominator": "n",
    "mode_rule": "eigenspace whose eigenvalue is nearest 1",
    "tie_atol": 1e-10,
    "tie_rtol": 1e-8,
    "tie_direction": "projection of equal-present-family unit vector",
    "correction_sd": "1 / n_present_families",
    "donors": False,
    "targets": False,
}

_PGRD_A_CONFIG = {
    "regime": "A_within_cell_fully_unsupervised",
    "base": "iu_pcr",
    "minimum_present_families": 3,
    "graph": "duplicate-safe symmetric union self-tuning kNN",
    "graph_k": 7,
    "tie_break": "lexicographic row_id rank",
    "laplacian": "symmetric_normalized",
    "moment": "cross_only",
    "trace_normalization": "G / trace(R.T @ L @ R / n)",
    "direction": "-(G/trace(A0)) * R.T @ L @ b / n",
    "correction_sd": "1 / n_present_families",
    "donors": False,
    "targets": False,
}


def _config(**values: Any) -> Mapping[str, Any]:
    return MappingProxyType({
        "feature_contract": CONTRACT_VERSION,
        "prepared_matrix_semantics": "higher_is_confidence",
        "output_score_semantics": "higher_is_incorrect",
        "global_sign_anchor": "equal_family_mean_of_prepared_confidence_coordinates",
        "global_sign_rule": (
            "if a native confidence score has unresolved global sign, multiply by "
            "sign(Pearson(score, equal_family_confidence_anchor)); require |r| > 1e-6"
        ),
        "per_feature_reorientation_after_preparation": "forbidden",
        **values,
    })


PRIMARY_METHOD_SPECS = OrderedDict((spec.method_id, spec) for spec in (
    MethodSpec(
        "equal_feature_mean",
        "equal-feature-mean-mixed-v2-v1",
        "Equal-feature mean",
        _config(weighting="one equal weight per present feature"),
        "primary_control",
        "local adapter",
    ),
    MethodSpec(
        "equal_family_mean",
        "equal-family-mean-provenance-v1",
        "Equal-family mean",
        _config(
            families="frozen six provenance families",
            weighting="equal mass per present family, then per feature",
        ),
        "primary_control",
        "spectral_utils.specrage_views",
    ),
    MethodSpec(
        "continuous_lsml",
        "continuous-lsml-full-pool-mixed-v2-v1",
        "Continuous L-SML",
        _config(pool="all present features", grouping="residual"),
        "primary_reconstruction",
        "spectral_utils.fusion_utils.lsml_continuous",
    ),
    MethodSpec(
        "dufs_pf_lsml",
        "dufs-eq7-parameter-free-to-continuous-lsml-v1",
        "Parameter-free DUFS to L-SML",
        _config(
            selector="a2.dufs_pf",
            cell_rng_seed=0,
            threshold="mean gate > 0",
            fewer_than_three="full-pool L-SML fallback",
        ),
        "primary_reconstruction",
        "spectral_utils.selectors.a2_groupfs.dufs_pf_gates",
    ),
    MethodSpec(
        "dufs_stability_lsml",
        "stability-selected-dufs-to-continuous-lsml-v1",
        "Stability-selected DUFS to L-SML",
        _config(
            selector="a2.dufs",
            lambda_grid="lambda0 * {0.5, 1, 2}",
            seeds=5,
            choice="maximum pairwise Jaccard among admissible selections",
            threshold="mean gate > 0",
            fewer_than_three="full-pool L-SML fallback",
        ),
        "primary_reconstruction",
        "spectral_utils.selectors.a2_groupfs.dufs_stability_selection",
    ),
    MethodSpec(
        "upcr",
        "deployed-style-upcr-mixed-v2-v1",
        "U-PCR",
        _config(**_UPCR_CONFIG),
        "primary_reconstruction",
        "spectral_utils.upcr.upcr_fit",
    ),
    MethodSpec(
        "iu_pcr",
        "iu-pcr-full-pool-two-component-mixed-v2-v1",
        "IU-PCR",
        _config(**_IU_CONFIG),
        "primary_control",
        "spectral_utils.laplacian_upcr.IU_FIT_DEFAULTS",
    ),
    MethodSpec(
        "dufs_liu",
        "dufs-liu-duplicate-safe-k7-lambda01-mixed-v2-v2",
        "DUFS-LIU",
        _config(**_DUFS_LIU_CONFIG),
        "primary_reconstruction",
        "spectral_utils.laplacian_upcr",
    ),
    MethodSpec(
        "su_pcr",
        "su-pcr-low-rank-plus-sparse-reproduction-v1",
        "SU-PCR",
        _config(**_SU_PCR_CONFIG),
        "primary_reproduction",
        "spectral_utils.dependency_fusion.sparse_upcr_fit",
    ),
    MethodSpec(
        "ca_specrage_atomic",
        "ca-specrage-atomic-provenance-balanced-duplicate-safe-v2",
        "CA-SpecRaGE atomic",
        _config(**_CA_CONFIG),
        "primary_aligned_rerun",
        "spectral_utils.specrage_laplacian.fit_specrage_graph",
    ),
    MethodSpec(
        "deem_b3",
        "continuous-additive-deem-b3-five-seed-v1",
        "DEEM-B3 continuous additive adapter",
        _config(**_DEEM_CONFIG),
        "primary_in_house_adapter_not_paper_reproduction",
        "spectral_utils.residual_graph_deem.fit_continuous_deem",
    ),
    MethodSpec(
        "family_nrm_a",
        "family-nrm-a-within-cell-unrun-ablation-v1",
        "Family-NRM-A within-cell",
        _config(**_FAMILY_NRM_A_CONFIG),
        "new_unrun_ablation",
        "local rule over spectral_utils.contribution_subspace",
    ),
    MethodSpec(
        "pgrd_a",
        "pgrd-a-within-cell-cross-gradient-unrun-ablation-v1",
        "PGRD-A within-cell",
        _config(**_PGRD_A_CONFIG),
        "new_unrun_ablation",
        "local rule over contribution_subspace and graph_topology",
    ),
))

PRIMARY_METHOD_IDS = tuple(PRIMARY_METHOD_SPECS)


class MethodFitError(RuntimeError):
    """A scientific/mechanical gate failed after input validation."""


@dataclass(frozen=True)
class _FitContext:
    cell: PreparedCell
    X_confidence: np.ndarray
    X_risk: np.ndarray
    F_confidence: np.ndarray
    confidence_anchor: np.ndarray
    family_members: Mapping[str, tuple[int, ...]]


@dataclass
class _RawFit:
    score: np.ndarray
    native_semantics: str = "confidence"
    globally_aligned: bool = False
    status: FitStatus = FitStatus.OK
    fallback_reason: str | None = None
    selected_features: tuple[str, ...] = ()
    diagnostics: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None


def _family_members(feature_names: tuple[str, ...]) -> OrderedDict[str, tuple[int, ...]]:
    from ..specrage_views import FEATURE_TO_VIEW, VIEW_ORDER

    output: OrderedDict[str, tuple[int, ...]] = OrderedDict()
    for family in VIEW_ORDER:
        indices = tuple(
            index for index, name in enumerate(feature_names)
            if FEATURE_TO_VIEW[name] == family
        )
        if indices:
            output[family] = indices
    if not output:
        raise ValueError("no frozen provenance family is present")
    return output


def _make_context(cell: PreparedCell) -> _FitContext:
    # This is the sole common adapter boundary at which risk coordinates exist.
    # The confidence matrix itself remains immutable and unchanged.
    X_confidence = cell.matrix
    X_risk = np.negative(X_confidence)
    # Keep this private copy writable: continuous DEEM deliberately uses
    # torch.as_tensor (zero-copy), and PyTorch warns when its NumPy backing is
    # read-only.  The prepared confidence matrix remains read-only, and its
    # hash is rechecked after every method fit.
    F_confidence = X_confidence.T
    F_confidence.setflags(write=False)
    members = _family_members(cell.feature_names)
    family_scores = [
        X_confidence[:, indices].mean(axis=1)
        for indices in members.values()
    ]
    anchor = np.mean(family_scores, axis=0)
    if not np.isfinite(anchor).all():
        raise ValueError("equal-family confidence anchor is non-finite")
    anchor.setflags(write=False)
    return _FitContext(
        cell=cell,
        X_confidence=X_confidence,
        X_risk=X_risk,
        F_confidence=F_confidence,
        confidence_anchor=anchor,
        family_members=members,
    )


def _orientation_multiplier(
    score: np.ndarray,
    confidence_anchor: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(score, dtype=float)
    anchor = np.asarray(confidence_anchor, dtype=float)
    if values.shape != anchor.shape or values.ndim != 1:
        raise MethodFitError("score and global confidence anchor disagree")
    if not np.isfinite(values).all():
        raise MethodFitError("method produced a non-finite score")
    if np.std(values) <= _EPS or np.std(anchor) <= _EPS:
        raise MethodFitError("global score orientation is undefined for a constant score")
    correlation = float(np.corrcoef(values, anchor)[0, 1])
    if not np.isfinite(correlation) or abs(correlation) <= _ORIENTATION_TOLERANCE:
        raise MethodFitError(
            "global score orientation is ambiguous against the frozen equal-family anchor"
        )
    return (1.0 if correlation > 0 else -1.0), correlation


def _selected_names(context: _FitContext, indices: np.ndarray) -> tuple[str, ...]:
    return tuple(context.cell.feature_names[int(index)] for index in indices)


def _run_equal_feature_mean(context: _FitContext) -> _RawFit:
    return _RawFit(
        score=np.mean(context.X_confidence, axis=1),
        globally_aligned=True,
        selected_features=context.cell.feature_names,
        diagnostics={"n_features": context.X_confidence.shape[1]},
    )


def _run_equal_family_mean(context: _FitContext) -> _RawFit:
    return _RawFit(
        score=context.confidence_anchor.copy(),
        globally_aligned=True,
        selected_features=context.cell.feature_names,
        diagnostics={
            "present_families": list(context.family_members),
            "family_sizes": {
                family: len(indices)
                for family, indices in context.family_members.items()
            },
        },
    )


def _continuous_lsml(
    context: _FitContext,
    selected: np.ndarray,
    *,
    selector_diagnostics: Mapping[str, Any] | None = None,
    fallback_reason: str | None = None,
) -> _RawFit:
    from ..fusion_utils import lsml_continuous

    selected = np.asarray(selected, dtype=int)
    if selected.ndim != 1 or len(selected) < 3:
        raise MethodFitError("continuous L-SML needs at least three selected features")
    if len(np.unique(selected)) != len(selected):
        raise MethodFitError("selector returned duplicate columns")
    if np.any(selected < 0) or np.any(selected >= context.X_confidence.shape[1]):
        raise MethodFitError("selector returned an out-of-range column")
    columns = [context.X_confidence[:, index] for index in selected]
    score, meta = lsml_continuous(*columns)
    diagnostics = {
        "lsml": meta,
        "n_selected": int(len(selected)),
    }
    if selector_diagnostics is not None:
        diagnostics["selector"] = dict(selector_diagnostics)
    return _RawFit(
        score=np.asarray(score, dtype=float),
        selected_features=_selected_names(context, selected),
        status=FitStatus.OK_FALLBACK if fallback_reason else FitStatus.OK,
        fallback_reason=fallback_reason,
        diagnostics=diagnostics,
    )


def _run_continuous_lsml(context: _FitContext) -> _RawFit:
    return _continuous_lsml(
        context,
        np.arange(context.X_confidence.shape[1], dtype=int),
    )


def _run_dufs_pf_lsml(context: _FitContext) -> _RawFit:
    from ..selectors.a2_groupfs import dufs_pf_cell_rng, dufs_pf_gates

    rng = dufs_pf_cell_rng(context.cell.cell_id, context.cell.domain, seed=0)
    gates = np.asarray(dufs_pf_gates(context.X_confidence, rng), dtype=float)
    if gates.shape != (context.X_confidence.shape[1],) or not np.isfinite(gates).all():
        raise MethodFitError("parameter-free DUFS returned invalid gates")
    selected = np.flatnonzero(gates > 0.0)
    fallback_reason = None
    if len(selected) < 3:
        selected = np.arange(context.X_confidence.shape[1], dtype=int)
        fallback_reason = "parameter-free DUFS selected fewer than three features"
    return _continuous_lsml(
        context,
        selected,
        selector_diagnostics={
            "variant": "a2.dufs_pf",
            "gate_means": gates,
            "threshold": 0.0,
            "cell_rng_seed": 0,
        },
        fallback_reason=fallback_reason,
    )


def _run_dufs_stability_lsml(context: _FitContext) -> _RawFit:
    from ..selectors.a2_groupfs import (
        dufs_pf_cell_rng,
        dufs_stability_selection,
    )

    rng = dufs_pf_cell_rng(context.cell.cell_id, context.cell.domain, seed=0)
    selected_row = dufs_stability_selection(context.X_confidence, rng)
    selected = np.asarray(selected_row.get("cols", ()), dtype=int)
    fallback = bool(selected_row.get("fallback", False))
    if len(selected) < 3:
        fallback = True
        selected = np.arange(context.X_confidence.shape[1], dtype=int)
    reason = None
    if fallback:
        selector_diag = selected_row.get("diag", {})
        reason = str(
            selector_diag.get("fallback_reason")
            or selector_diag.get("error")
            or "stability-selected DUFS used its full-pool fallback"
        )
    return _continuous_lsml(
        context,
        selected,
        selector_diagnostics={
            "variant": "a2.dufs",
            **dict(selected_row.get("diag", {})),
        },
        fallback_reason=reason,
    )


def _upcr_diagnostics(result: Any) -> dict[str, Any]:
    return {
        "g2_hat": float(result.g2_hat),
        "var_y": float(result.var_y),
        "keep": np.asarray(result.keep, dtype=bool),
        "n_kept": int(np.sum(result.keep)),
        "abstained": bool(result.abstained),
        "used_simple_average": bool(result.used_simple_average),
        "n_components_used": int(result.n_components_used),
        "lambda2_frac": float(result.lambda2_frac),
        "projection_residual": float(result.proj_residual),
        "g2_at_ceiling": bool(result.g2_at_ceiling),
        "g2_frac_of_var_y": float(result.g2_frac_of_var_y),
        "meta": dict(result.meta),
    }


def _run_upcr(context: _FitContext) -> _RawFit:
    from ..upcr import upcr_fit

    result = upcr_fit(context.F_confidence, **_UPCR_CONFIG)
    fallback = bool(result.used_simple_average)
    reason = (
        "U-PCR exclusion left too few features and invoked its registered simple mean"
        if fallback else None
    )
    return _RawFit(
        score=np.asarray(result.w @ context.F_confidence, dtype=float),
        selected_features=tuple(
            name for name, keep in zip(context.cell.feature_names, result.keep) if keep
        ),
        status=FitStatus.OK_FALLBACK if fallback else FitStatus.OK,
        fallback_reason=reason,
        diagnostics=_upcr_diagnostics(result),
        artifacts={
            "weights": np.asarray(result.w, dtype=float),
            "rho_hat": np.asarray(result.rho_hat, dtype=float),
            "rho_hat_full": np.asarray(result.rho_hat_full, dtype=float),
        },
    )


def _fit_iu(context: _FitContext):
    from ..laplacian_upcr import IU_FIT_DEFAULTS
    from ..upcr import upcr_fit

    if dict(IU_FIT_DEFAULTS) != _IU_CONFIG:
        raise MethodFitError("canonical IU_FIT_DEFAULTS drifted from the registered config")
    return upcr_fit(context.F_confidence, **IU_FIT_DEFAULTS)


def _run_iu_pcr(context: _FitContext) -> _RawFit:
    result = _fit_iu(context)
    return _RawFit(
        score=np.asarray(result.w @ context.F_confidence, dtype=float),
        selected_features=context.cell.feature_names,
        diagnostics=_upcr_diagnostics(result),
        artifacts={
            "weights": np.asarray(result.w, dtype=float),
            "rho_hat": np.asarray(result.rho_hat, dtype=float),
        },
    )


def _run_dufs_liu(context: _FitContext) -> _RawFit:
    from ..graph_topology import self_safe_knn_graph
    from ..laplacian_upcr import (
        dufs_soft_gates,
        laplacian_iu_path,
    )

    gates, gate_diagnostics = dufs_soft_gates(
        context.F_confidence,
        seeds=_DUFS_LIU_CONFIG["seeds"],
        epochs=_DUFS_LIU_CONFIG["epochs"],
    )
    graph = self_safe_knn_graph(
        context.F_confidence.T * gates[None, :],
        k=_DUFS_LIU_CONFIG["graph_k"],
        tie_keys=_row_tie_keys(context.cell.row_ids),
    )
    path = laplacian_iu_path(
        context.F_confidence,
        (0.0, _DUFS_LIU_CONFIG["lambda"]),
        graph=graph,
    )
    fit = path[_DUFS_LIU_CONFIG["lambda"]]
    return _RawFit(
        score=np.asarray(fit.w @ context.F_confidence, dtype=float),
        selected_features=context.cell.feature_names,
        diagnostics={
            "dufs_gates": gate_diagnostics,
            "liu": dict(fit.diagnostics),
            "graph_builder": "duplicate-safe distance-then-row-id union-kNN",
            "lambda_zero_weight_identity_max_abs": float(np.max(np.abs(
                path[0.0].w - path[0.0].baseline.w
            ))),
        },
        artifacts={
            "gates": np.asarray(gates, dtype=float),
            "gate_probabilities_per_seed": np.asarray(
                gate_diagnostics["per_seed_probabilities"], dtype=float
            ),
            "graph": fit.graph,
            "laplacian": fit.laplacian,
            "weights": np.asarray(fit.w, dtype=float),
            "iu_weights": np.asarray(fit.baseline.w, dtype=float),
            "roughness": np.asarray(fit.roughness, dtype=float),
        },
    )


def _run_su_pcr(context: _FitContext) -> _RawFit:
    from ..dependency_fusion import sparse_upcr_fit

    result = sparse_upcr_fit(context.F_confidence, **_SU_PCR_CONFIG)
    decomposition = result.decomposition
    return _RawFit(
        score=np.asarray(result.w_pcr @ context.F_confidence, dtype=float),
        selected_features=context.cell.feature_names,
        diagnostics={
            "g2_hat": float(result.g2_hat),
            "var_y": float(result.var_y),
            "projection_residual": float(result.projection_residual),
            "pcr_eigenvalues": np.asarray(result.pcr_eigenvalues, dtype=float),
            "decomposition_converged": bool(decomposition.converged),
            "decomposition_iterations": int(decomposition.n_iter),
            "sparse_fraction": float(decomposition.sparse_fraction),
            "decomposition_relative_residual": float(decomposition.relative_residual),
            "theorem_support_ok": bool(decomposition.theorem_support_ok),
            "meta": dict(result.meta),
        },
        artifacts={
            "weights": np.asarray(result.w_pcr, dtype=float),
            "rho_hat": np.asarray(result.rho_hat, dtype=float),
            "low_rank": np.asarray(decomposition.low_rank, dtype=float),
            "sparse": np.asarray(decomposition.sparse, dtype=float),
            "sparse_support": np.asarray(decomposition.support, dtype=bool),
        },
    )


def _run_ca_specrage_atomic(context: _FitContext) -> _RawFit:
    from ..fusion_aware_views import atomic_views, group_balanced_atomic_prior
    from ..laplacian_upcr import laplacian_iu_fit
    from ..specrage_laplacian import SpecRaGEConfig, fit_specrage_graph
    from ..specrage_views import view_members

    config_values = {
        key: value for key, value in _CA_CONFIG.items()
        if key not in {
            "model_seeds", "liu_lambda", "atomic_prior",
            "duplicate_safe_graph", "tie_break",
        }
    }
    config = SpecRaGEConfig(**config_values)
    views = atomic_views(context.X_confidence, context.cell.feature_names)
    # This primary within-cell rerun uses the frozen provenance families as the
    # balancing partition.  It does not import the old donor-derived LOCO micro
    # partition under the same name.
    partition = view_members(context.cell.feature_names)
    prior = group_balanced_atomic_prior(context.cell.feature_names, partition)
    graph_fit = fit_specrage_graph(
        views,
        config=config,
        seeds=_CA_CONFIG["model_seeds"],
        view_prior=prior,
        tie_keys=_row_tie_keys(context.cell.row_ids),
    )
    fit = laplacian_iu_fit(
        context.F_confidence,
        lambda_=_CA_CONFIG["liu_lambda"],
        graph=graph_fit.graph,
    )
    return _RawFit(
        score=np.asarray(fit.w @ context.F_confidence, dtype=float),
        selected_features=context.cell.feature_names,
        diagnostics={
            "specrage": dict(graph_fit.diagnostics),
            "liu": dict(fit.diagnostics),
            "atomic_prior": dict(prior),
            "prior_partition": partition,
            "historical_loco_micro_prior_reused": False,
            "graph_builder": "duplicate-safe distance-then-row-id SpecRaGE affinities",
        },
        artifacts={
            "graph": graph_fit.graph,
            "embedding_graph": graph_fit.embedding_graph,
            "base_graphs": {
                name: graph
                for name, graph in zip(graph_fit.view_names, graph_fit.base_graphs)
            },
            "seed_graphs": {
                str(seed_result.seed): seed_result.graph
                for seed_result in graph_fit.seed_results
            },
            "alpha": np.asarray(graph_fit.alpha, dtype=float),
            "alpha_per_seed": np.stack([
                np.asarray(seed_result.alpha, dtype=float)
                for seed_result in graph_fit.seed_results
            ]),
            "view_names": tuple(graph_fit.view_names),
            "view_prior": np.asarray(graph_fit.view_prior, dtype=float),
            "laplacian": fit.laplacian,
            "weights": np.asarray(fit.w, dtype=float),
            "iu_weights": np.asarray(fit.baseline.w, dtype=float),
        },
    )


def _run_deem_b3(context: _FitContext) -> _RawFit:
    module = importlib.import_module("spectral_utils.residual_graph_deem")
    config_values = {
        key: value for key, value in _DEEM_CONFIG.items()
        if key not in {
            "seeds", "aggregation", "minimum_median_seed_spearman"
        }
    }
    config = module.ContinuousDeemConfig(**config_values)
    per_seed = []
    fit_results = []
    health = []
    for seed in _DEEM_CONFIG["seeds"]:
        result = module.fit_continuous_deem(
            context.X_risk,
            context.cell.feature_names,
            seed=int(seed),
            config=config,
        )
        seed_health = dict(result.health)
        # Wall-clock time is operational telemetry, not part of the canonical
        # scientific record.  Keeping it here would break independent A/B byte
        # identity even when every fitted score and parameter is identical.
        canonical_seed_health = {
            key: value for key, value in seed_health.items()
            if key != "runtime_seconds"
        }
        health.append({"seed": int(seed), **canonical_seed_health})
        valid = bool(seed_health.get("healthy", False))
        valid = valid and bool(seed_health.get("finite", False))
        valid = valid and float(seed_health.get("posterior_sd", 0.0)) >= config.posterior_sd_min
        valid = valid and float(
            seed_health.get("contribution_reconstruction_max_abs", np.inf)
        ) <= 1e-8
        valid = valid and int(seed_health.get("epochs_completed", -1)) == config.epochs
        score = np.asarray(result.score, dtype=float)
        valid = valid and score.shape == (context.X_risk.shape[0],)
        valid = valid and bool(np.isfinite(score).all())
        if not valid:
            raise MethodFitError(
                f"DEEM-B3 seed {seed} failed the frozen all-seed health gate"
            )
        per_seed.append(score)
        fit_results.append(result)
    stacked = np.stack(per_seed, axis=1)
    ensemble_score, stability = module.ensemble_seed_scores(fit_results)
    median_seed_spearman = float(stability.get("median_abs_spearman", np.nan))
    if (
        not np.isfinite(median_seed_spearman)
        or median_seed_spearman < _DEEM_CONFIG["minimum_median_seed_spearman"]
    ):
        raise MethodFitError(
            "DEEM-B3 failed the frozen cross-seed stability gate: "
            f"median |Spearman|={median_seed_spearman:.6f} < "
            f"{_DEEM_CONFIG['minimum_median_seed_spearman']:.2f}"
        )
    return _RawFit(
        score=np.asarray(ensemble_score, dtype=float),
        native_semantics="risk",
        globally_aligned=True,
        selected_features=context.cell.feature_names,
        diagnostics={
            "seed_health": health,
            "n_required_seeds": len(_DEEM_CONFIG["seeds"]),
            "n_healthy_seeds": len(per_seed),
            "aggregation": _DEEM_CONFIG["aggregation"],
            "seed_stability": stability,
            "paper_reproduction": False,
            "description": "in-house continuous additive adapter inspired by DEEM",
        },
        artifacts={"per_seed_risk_scores": stacked},
    )


@dataclass(frozen=True)
class _ContributionCoordinates:
    feature_names: tuple[str, ...]
    iu_fit: Any
    oriented_weights: np.ndarray
    orientation_multiplier: float
    orientation_correlation: float
    space: Any
    transform: Any
    baseline: np.ndarray
    residuals: np.ndarray


def _usable_family_residuals(
    coordinates: _ContributionCoordinates,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, tuple[str, ...]]:
    """Drop residual-family columns that carry no usable local variation.

    The family count ``G`` in both new within-cell ablations is defined *after*
    this mechanical, label-free gate.  Keeping an all-zero residual as a family
    would make the nearest-unit eigenspace and the PGRD trust scaling depend on
    a coordinate that contains no information.
    """

    residuals = np.asarray(coordinates.residuals, dtype=float)
    scales = np.std(residuals, axis=0, ddof=0)
    usable = np.logical_and(np.isfinite(scales), scales > _EPS)
    indices = np.flatnonzero(usable).astype(np.int64)
    families = tuple(
        str(coordinates.space.families[int(index)]) for index in indices
    )
    dropped = tuple(
        str(coordinates.space.families[int(index)])
        for index in np.flatnonzero(~usable)
    )
    return families, residuals[:, indices], indices, dropped


def _iu_contribution_coordinates(context: _FitContext) -> _ContributionCoordinates:
    from ..contribution_subspace import (
        fit_contribution_transform,
        iu_family_contributions,
    )

    iu_fit = _fit_iu(context)
    raw_score = np.asarray(iu_fit.w @ context.F_confidence, dtype=float)
    multiplier, correlation = _orientation_multiplier(
        raw_score, context.confidence_anchor
    )
    oriented_weights = multiplier * np.asarray(iu_fit.w, dtype=float)
    space = iu_family_contributions(
        context.F_confidence,
        context.cell.feature_names,
        oriented_weights,
    )
    transform = fit_contribution_transform(
        space,
        np.arange(context.X_confidence.shape[0], dtype=int),
    )
    baseline, residuals = transform.apply(
        space.baseline_score,
        space.contributions,
    )
    return _ContributionCoordinates(
        feature_names=context.cell.feature_names,
        iu_fit=iu_fit,
        oriented_weights=oriented_weights,
        orientation_multiplier=multiplier,
        orientation_correlation=correlation,
        space=space,
        transform=transform,
        baseline=np.asarray(baseline, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
    )


def _contribution_fallback(
    coordinates: _ContributionCoordinates,
    reason: str,
    *,
    diagnostics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> _RawFit:
    return _RawFit(
        score=coordinates.baseline.copy(),
        globally_aligned=True,
        status=FitStatus.OK_FALLBACK,
        fallback_reason=reason,
        selected_features=coordinates.feature_names,
        diagnostics={
            "present_families": list(coordinates.space.families),
            "n_present_families": len(coordinates.space.families),
            "base_iu_orientation_correlation": coordinates.orientation_correlation,
            **dict(diagnostics or {}),
        },
        artifacts={
            "baseline_standardized": coordinates.baseline,
            "residuals": coordinates.residuals,
            **dict(artifacts or {}),
        },
    )


def _run_family_nrm_a(context: _FitContext) -> _RawFit:
    coordinates = _iu_contribution_coordinates(context)
    families, residuals, usable_indices, dropped_families = (
        _usable_family_residuals(coordinates)
    )
    n, family_count = residuals.shape
    if family_count < _FAMILY_NRM_A_CONFIG["minimum_present_families"]:
        return _contribution_fallback(
            coordinates,
            "fewer than three provenance families are present",
            diagnostics={
                "usable_families": list(families),
                "dropped_degenerate_families": list(dropped_families),
            },
        )

    covariance = residuals.T @ residuals / n
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    distance = np.abs(eigenvalues - 1.0)
    minimum = float(np.min(distance))
    tied = np.flatnonzero(np.isclose(
        distance,
        minimum,
        atol=_FAMILY_NRM_A_CONFIG["tie_atol"],
        rtol=_FAMILY_NRM_A_CONFIG["tie_rtol"],
    ))
    basis = eigenvectors[:, tied]
    equal_family = np.ones(family_count, dtype=float) / np.sqrt(family_count)
    direction = basis @ (basis.T @ equal_family)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= _EPS:
        return _contribution_fallback(
            coordinates,
            "nearest-unit residual eigenspace has negligible equal-family projection",
            diagnostics={
                "residual_eigenvalues": eigenvalues,
                "selected_eigen_indices": tied,
            },
            artifacts={"residual_covariance": covariance},
        )
    direction = direction / direction_norm
    raw_correction = residuals @ direction
    correction_sd = float(np.std(raw_correction))
    if correction_sd <= _EPS:
        return _contribution_fallback(
            coordinates,
            "nearest-unit residual correction has negligible variance",
            diagnostics={
                "residual_eigenvalues": eigenvalues,
                "selected_eigen_indices": tied,
            },
            artifacts={
                "residual_covariance": covariance,
                "residual_direction": direction,
            },
        )
    correction = raw_correction / (family_count * correction_sd)
    return _RawFit(
        score=coordinates.baseline + correction,
        globally_aligned=True,
        selected_features=context.cell.feature_names,
        diagnostics={
            "regime": "A_within_cell_fully_unsupervised",
            "new_unrun_ablation": True,
            "present_families": list(families),
            "n_present_families": family_count,
            "dropped_degenerate_families": list(dropped_families),
            "residual_eigenvalues": eigenvalues,
            "selected_eigen_indices": tied,
            "selected_distance_from_unit": minimum,
            "direction_equal_family_cosine": float(np.dot(direction, equal_family)),
            "correction_sd_before_scaling": correction_sd,
            "correction_sd_after_scaling": float(np.std(correction)),
            "base_iu_orientation_correlation": coordinates.orientation_correlation,
            "donor_cells_used": 0,
            "targets_used": False,
        },
        artifacts={
            "baseline_standardized": coordinates.baseline,
            "family_contributions": coordinates.space.contributions,
            "residuals": residuals,
            "usable_family_indices": usable_indices,
            "residual_covariance": covariance,
            "residual_direction": direction,
            "correction": correction,
            "oriented_iu_weights": coordinates.oriented_weights,
        },
    )


def _row_tie_keys(row_ids: tuple[str, ...]) -> np.ndarray:
    values = np.asarray(row_ids, dtype=str)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _run_pgrd_a(context: _FitContext) -> _RawFit:
    from ..graph_topology import self_safe_knn_graph
    from ..laplacian_upcr import graph_diagnostics, symmetric_normalized_laplacian

    coordinates = _iu_contribution_coordinates(context)
    families, residuals, usable_indices, dropped_families = (
        _usable_family_residuals(coordinates)
    )
    baseline = coordinates.baseline
    n, family_count = residuals.shape
    if family_count < _PGRD_A_CONFIG["minimum_present_families"]:
        return _contribution_fallback(
            coordinates,
            "fewer than three provenance families are present",
            diagnostics={
                "usable_families": list(families),
                "dropped_degenerate_families": list(dropped_families),
            },
        )

    graph = self_safe_knn_graph(
        residuals,
        k=_PGRD_A_CONFIG["graph_k"],
        tie_keys=_row_tie_keys(context.cell.row_ids),
    )
    laplacian = symmetric_normalized_laplacian(graph)
    a0 = np.asarray(residuals.T @ (laplacian @ residuals) / n, dtype=float)
    a0 = 0.5 * (a0 + a0.T)
    c0 = np.asarray(residuals.T @ (laplacian @ baseline) / n, dtype=float)
    trace_a0 = float(np.trace(a0))
    graph_health = graph_diagnostics(graph, laplacian)
    if not np.isfinite(trace_a0) or trace_a0 <= _EPS:
        return _contribution_fallback(
            coordinates,
            "residual graph roughness has nonpositive trace",
            diagnostics={"graph_health": graph_health, "trace_a0": trace_a0},
            artifacts={"graph": graph, "laplacian": laplacian, "A0": a0, "c0": c0},
        )
    trace_scale = family_count / trace_a0
    c = trace_scale * c0
    direction = -c
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm <= _EPS:
        return _contribution_fallback(
            coordinates,
            "residual graph cross-gradient is negligible",
            diagnostics={
                "graph_health": graph_health,
                "trace_a0": trace_a0,
                "trace_scale": trace_scale,
            },
            artifacts={
                "graph": graph,
                "laplacian": laplacian,
                "A0": a0,
                "c0": c0,
                "c": c,
            },
        )
    raw_correction = residuals @ direction
    correction_sd = float(np.std(raw_correction))
    if not np.isfinite(correction_sd) or correction_sd <= _EPS:
        return _contribution_fallback(
            coordinates,
            "residual graph correction has negligible variance",
            diagnostics={
                "graph_health": graph_health,
                "trace_a0": trace_a0,
                "trace_scale": trace_scale,
            },
            artifacts={
                "graph": graph,
                "laplacian": laplacian,
                "A0": a0,
                "c0": c0,
                "c": c,
                "direction": direction,
            },
        )
    correction = raw_correction / (family_count * correction_sd)
    return _RawFit(
        score=baseline + correction,
        globally_aligned=True,
        selected_features=context.cell.feature_names,
        diagnostics={
            "regime": "A_within_cell_fully_unsupervised",
            "new_unrun_ablation": True,
            "present_families": list(families),
            "n_present_families": family_count,
            "dropped_degenerate_families": list(dropped_families),
            "graph_health": graph_health,
            "trace_a0": trace_a0,
            "trace_scale": trace_scale,
            "cross_gradient_norm": direction_norm,
            "correction_sd_before_scaling": correction_sd,
            "correction_sd_after_scaling": float(np.std(correction)),
            "base_iu_orientation_correlation": coordinates.orientation_correlation,
            "donor_cells_used": 0,
            "targets_used": False,
        },
        artifacts={
            "graph": graph,
            "laplacian": laplacian,
            "baseline_standardized": baseline,
            "family_contributions": coordinates.space.contributions,
            "residuals": residuals,
            "usable_family_indices": usable_indices,
            "A0": a0,
            "c0": c0,
            "trace_scale": np.asarray(trace_scale),
            "c": c,
            "direction": direction,
            "correction": correction,
            "oriented_iu_weights": coordinates.oriented_weights,
        },
    )


_RUNNERS: Mapping[str, Callable[[_FitContext], _RawFit]] = MappingProxyType({
    "equal_feature_mean": _run_equal_feature_mean,
    "equal_family_mean": _run_equal_family_mean,
    "continuous_lsml": _run_continuous_lsml,
    "dufs_pf_lsml": _run_dufs_pf_lsml,
    "dufs_stability_lsml": _run_dufs_stability_lsml,
    "upcr": _run_upcr,
    "iu_pcr": _run_iu_pcr,
    "dufs_liu": _run_dufs_liu,
    "su_pcr": _run_su_pcr,
    "ca_specrage_atomic": _run_ca_specrage_atomic,
    "deem_b3": _run_deem_b3,
    "family_nrm_a": _run_family_nrm_a,
    "pgrd_a": _run_pgrd_a,
})

if tuple(_RUNNERS) != PRIMARY_METHOD_IDS:
    raise RuntimeError("primary method specs and runner order disagree")


def _failure_result(
    spec: MethodSpec,
    cell: PreparedCell,
    status: FitStatus,
    error: BaseException,
) -> ScoreResult:
    return ScoreResult(
        method_id=spec.method_id,
        method_version_id=spec.method_version_id,
        config_sha256=spec.config_sha256,
        status=status,
        score=None,
        population_id=cell.population_id,
        cell_id=cell.cell_id,
        feature_contract=cell.feature_contract,
        prepared_matrix_sha256=prepared_matrix_sha256(
            cell.matrix, cell.feature_names, cell.row_ids
        ),
        fallback_reason=None,
        diagnostics={
            "error_type": type(error).__name__,
            "error": str(error),
            "solver_coordinate_semantics": "fit did not complete",
            "development_status": spec.development_status,
        },
    )


def run_method(method_id: str, cell: PreparedCell) -> ScoreResult:
    """Fit one registered method without labels or evaluation.

    The returned score always has ``higher_is_incorrect`` semantics.  Failed
    fits are records with explicit status and no score; they never become an
    implicit mean-score fallback.
    """

    if method_id not in PRIMARY_METHOD_SPECS:
        raise KeyError(f"unregistered primary method: {method_id}")
    if not isinstance(cell, PreparedCell):
        raise TypeError("run_method accepts only a PreparedCell")
    spec = PRIMARY_METHOD_SPECS[method_id]
    observed_hash = prepared_matrix_sha256(
        cell.matrix, cell.feature_names, cell.row_ids
    )
    if observed_hash != cell.matrix_sha256:
        return _failure_result(
            spec,
            cell,
            FitStatus.INPUT_INVALID,
            ValueError("prepared matrix changed after its contract hash was frozen"),
        )
    try:
        context = _make_context(cell)
        raw = _RUNNERS[method_id](context)
        if prepared_matrix_sha256(
            cell.matrix, cell.feature_names, cell.row_ids
        ) != observed_hash:
            raise MethodFitError("method mutated the prepared feature matrix")
        values = np.asarray(raw.score, dtype=float)
        if values.shape != (len(cell.row_ids),) or not np.isfinite(values).all():
            raise MethodFitError("method returned an invalid score vector")

        diagnostics = dict(raw.diagnostics or {})
        diagnostics["development_status"] = spec.development_status
        if raw.native_semantics == "confidence":
            if raw.globally_aligned:
                confidence_score = values
                diagnostics["global_orientation"] = {
                    "anchor": "equal_family_mean",
                    "already_aligned_by_fixed_rule": True,
                    "multiplier": 1.0,
                }
            else:
                multiplier, correlation = _orientation_multiplier(
                    values, context.confidence_anchor
                )
                confidence_score = multiplier * values
                diagnostics["global_orientation"] = {
                    "anchor": "equal_family_mean",
                    "already_aligned_by_fixed_rule": False,
                    "correlation_before": correlation,
                    "multiplier": multiplier,
                }
            output_score = -confidence_score
            diagnostics["solver_coordinate_semantics"] = (
                "prepared confidence coordinates; final fused score globally negated"
            )
        elif raw.native_semantics == "risk":
            if not raw.globally_aligned:
                raise MethodFitError("native risk score was not globally aligned")
            output_score = values
            diagnostics["solver_coordinate_semantics"] = (
                "common whole-matrix risk view (-X_confidence); native risk output"
            )
        else:
            raise MethodFitError(f"unknown native score semantics: {raw.native_semantics}")

        return ScoreResult(
            method_id=spec.method_id,
            method_version_id=spec.method_version_id,
            config_sha256=spec.config_sha256,
            status=raw.status,
            score=output_score,
            population_id=cell.population_id,
            cell_id=cell.cell_id,
            feature_contract=cell.feature_contract,
            prepared_matrix_sha256=observed_hash,
            selected_features=tuple(raw.selected_features),
            fallback_reason=raw.fallback_reason,
            diagnostics=diagnostics,
            artifacts=dict(raw.artifacts or {}),
        )
    except (ImportError, ModuleNotFoundError) as error:
        return _failure_result(
            spec,
            cell,
            FitStatus.BLOCKED_DEPENDENCY,
            error,
        )
    except Exception as error:
        return _failure_result(
            spec,
            cell,
            FitStatus.FIT_FAILED,
            error,
        )


def run_all_methods(
    cell: PreparedCell,
    method_ids: tuple[str, ...] = PRIMARY_METHOD_IDS,
) -> OrderedDict[str, ScoreResult]:
    """Fit a requested roster in stable order; no evaluation is performed."""

    if len(set(method_ids)) != len(method_ids):
        raise ValueError("method_ids contains duplicates")
    unknown = sorted(set(method_ids) - set(PRIMARY_METHOD_SPECS))
    if unknown:
        raise KeyError("unregistered primary method(s): " + ", ".join(unknown))
    return OrderedDict((method_id, run_method(method_id, cell)) for method_id in method_ids)


__all__ = [
    "MethodFitError",
    "PRIMARY_METHOD_IDS",
    "PRIMARY_METHOD_SPECS",
    "run_all_methods",
    "run_method",
]
