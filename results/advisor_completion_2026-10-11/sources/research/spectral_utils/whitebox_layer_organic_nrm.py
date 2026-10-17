"""Layer-organic feature groups for the white-box NRM addendum.

The original white-box headline compresses the residual-stream measurements
into one scalar expert per layer and then groups layers into four broad depth
bands.  This module keeps the atomic residual measurements instead: one group
is one transformer layer, and its internal features are the measurements made
at that layer.

The primary contract deliberately uses the three genuinely layer-local
quantities: entropy, target-token NLL, and top-1 surprisal.  KL-to-final is
available only as a named sensitivity because it couples every layer to the
final layer and therefore weakens the organic-local interpretation.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Sequence

import numpy as np

from .whitebox_layer_fusion import FeatureMatrix


LOCAL_RESID_METRICS = (
    "lens_H",
    "lens_logp_tgt",
    "lens_logp_top1",
)
KL_SENSITIVITY_METRICS = LOCAL_RESID_METRICS + ("lens_kl_final",)

_LENS_NAME = re.compile(
    r"^(?P<module>attn|mlp|resid)\."
    r"(?P<metric>lens_H|lens_logp_tgt|lens_logp_top1|lens_kl_final)\."
    r"layer_(?P<layer>[0-9]+)$"
)


def layer_organic_residual_matrix(
    lens_grid: FeatureMatrix,
    *,
    metrics: Sequence[str] = LOCAL_RESID_METRICS,
) -> FeatureMatrix:
    """Regroup a frozen all-layer lens matrix as one family per layer.

    Parameters
    ----------
    lens_grid:
        A label-free matrix produced by ``extract_lens_grid``.  Only residual
        stream columns are retained; no raw cache or outcome field is opened.
    metrics:
        Ordered residual metrics inside every layer.  The registered primary
        is :data:`LOCAL_RESID_METRICS`; :data:`KL_SENSITIVITY_METRICS` is the
        only registered sensitivity.

    Notes
    -----
    The final residual KL-to-final column is mechanically zero and is already
    absent from the frozen lens grid.  It is the sole allowed missing column
    in the KL sensitivity contract.
    """

    metric_order = tuple(str(value) for value in metrics)
    if metric_order not in (LOCAL_RESID_METRICS, KL_SENSITIVITY_METRICS):
        raise ValueError("metrics must be the frozen local triad or KL sensitivity quartet")

    parsed: dict[tuple[int, str], int] = {}
    observed_layers: set[int] = set()
    for index, name in enumerate(lens_grid.feature_names):
        match = _LENS_NAME.fullmatch(name)
        if match is None or match.group("module") != "resid":
            continue
        metric = match.group("metric")
        if metric not in metric_order:
            continue
        layer = int(match.group("layer"))
        key = (layer, metric)
        if key in parsed:
            raise ValueError(f"duplicate residual lens column: {name}")
        parsed[key] = index
        observed_layers.add(layer)

    if not observed_layers:
        raise ValueError("lens grid has no residual columns for the requested metrics")
    layers = tuple(range(max(observed_layers) + 1))
    if observed_layers != set(layers):
        raise ValueError("residual lens layers are not contiguous from zero")

    selected_indices: list[int] = []
    selected_names: list[str] = []
    groups: list[str] = []
    missing: list[str] = []
    for layer in layers:
        for metric in metric_order:
            key = (layer, metric)
            if key not in parsed:
                name = f"resid.{metric}.layer_{layer:02d}"
                if metric == "lens_kl_final" and layer == layers[-1]:
                    missing.append(name)
                    continue
                raise ValueError(f"missing non-degenerate organic feature: {name}")
            selected_indices.append(parsed[key])
            selected_names.append(f"resid.{metric}.layer_{layer:02d}")
            groups.append(f"layer_{layer:02d}")

    values = np.asarray(lens_grid.values[:, selected_indices], dtype=float)
    signature_payload = {
        "parent_protocol_signature": lens_grid.protocol_signature,
        "contract": "layer-organic-residual",
        "metrics": metric_order,
        "layers": layers,
        "feature_names": selected_names,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return FeatureMatrix(
        values=values,
        feature_names=tuple(selected_names),
        risk_anchor=np.asarray(lens_grid.risk_anchor, dtype=float),
        groups=tuple(groups),
        protocol_signature=signature,
        metadata={
            "contract": (
                "resid-layer-organic-triad"
                if metric_order == LOCAL_RESID_METRICS
                else "resid-layer-organic-kl-sensitivity"
            ),
            "parent_contract": lens_grid.metadata.get("contract"),
            "parent_protocol_signature": lens_grid.protocol_signature,
            "n_layers": len(layers),
            "layers": list(layers),
            "metrics": list(metric_order),
            "grouping": "one residual transformer layer per group",
            "within_group_features": "one oriented token-mean scalar per metric",
            "missing_mechanical_features": missing,
            "kl_is_nonlocal_sensitivity": "lens_kl_final" in metric_order,
        },
    )


def assert_layer_organic_contract(matrix: FeatureMatrix, *, n_layers: int) -> None:
    """Fail closed unless the matrix has the exact registered layer grouping."""

    expected_groups = tuple(f"layer_{layer:02d}" for layer in range(int(n_layers)))
    observed_groups = tuple(dict.fromkeys(matrix.groups))
    if observed_groups != expected_groups:
        raise AssertionError(
            f"organic group order mismatch: expected {expected_groups}, got {observed_groups}"
        )
    counts = {group: matrix.groups.count(group) for group in observed_groups}
    expected_per_layer = 4 if matrix.metadata.get("kl_is_nonlocal_sensitivity") else 3
    for layer, group in enumerate(expected_groups):
        expected = expected_per_layer - int(
            bool(matrix.metadata.get("kl_is_nonlocal_sensitivity"))
            and layer == int(n_layers) - 1
        )
        if counts[group] != expected:
            raise AssertionError(f"{group}: expected {expected} features, got {counts[group]}")
