"""Graph-coupled local family relevance for the U-PCR score.

This module is label-free.  It tests a specific extension of the project's
manual feature families: related feature families may be locally reliable or
locally noisy, and a small prior graph can share reliability evidence between
families.  The graph regularizes a relevance estimate; it does not define or
observe correctness.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.stats import rankdata

from .atomic_operator_audit import iu_state
from .specrage_views import FEATURE_TO_VIEW, VIEW_ORDER


EPS = 1e-12

# Prior relations are about measurement lineage, not empirical AUROC.
# Entropy level, entropy dynamics, and answer structure describe the entropy
# trajectory.  Sampled-token energy, partition energy, and top-k summaries
# describe related token-probability/energy channels.
FAMILY_PRIOR_EDGES = (
    ("entropy_level", "entropy_dynamics"),
    ("entropy_dynamics", "structural"),
    ("sampled_token_energy", "partition_energy"),
    ("partition_energy", "topk_distribution"),
)


def stable_seed(namespace: str) -> int:
    return int(hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:8], 16)


def family_members(feature_names):
    """Return present families and their feature indices in frozen order."""
    names = tuple(str(name) for name in feature_names)
    unknown = sorted(set(names) - set(FEATURE_TO_VIEW))
    if unknown:
        raise KeyError("unregistered feature(s): " + ", ".join(unknown))
    families = tuple(
        family for family in VIEW_ORDER
        if any(FEATURE_TO_VIEW[name] == family for name in names)
    )
    members = {
        family: np.asarray(
            [index for index, name in enumerate(names)
             if FEATURE_TO_VIEW[name] == family],
            dtype=int,
        )
        for family in families
    }
    return families, members


def rank_rows(F):
    """Convert every feature to deterministic fractional ranks in [0, 1]."""
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or min(F.shape) < 3 or not np.isfinite(F).all():
        raise ValueError("F must be a finite feature-by-sample matrix")
    denominator = max(F.shape[1] - 1, 1)
    return np.asarray([
        (rankdata(row, method="average") - 1.0) / denominator for row in F
    ])


def family_prior_laplacian(families, *, permutation=None):
    """Build the dense normalized Laplacian of the registered family prior."""
    families = tuple(families)
    index = {name: position for position, name in enumerate(families)}
    adjacency = np.zeros((len(families), len(families)), dtype=float)
    for left, right in FAMILY_PRIOR_EDGES:
        if left in index and right in index:
            adjacency[index[left], index[right]] = 1.0
            adjacency[index[right], index[left]] = 1.0
    if permutation is not None:
        permutation = np.asarray(permutation, dtype=int)
        if sorted(permutation.tolist()) != list(range(len(families))):
            raise ValueError("permutation must contain every family once")
        adjacency = adjacency[permutation][:, permutation]
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    positive = degree > EPS
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    laplacian = np.eye(len(families)) - normalized
    laplacian[~positive, ~positive] = 0.0
    return laplacian, adjacency


def local_family_evidence(F, feature_names):
    """Estimate local reliability from within-family rank agreement.

    A multi-feature family receives high evidence on samples where its members
    have similar oriented ranks.  Singleton families are left unobserved and
    obtain their estimate only from the unit prior and graph neighbours.
    """
    F = np.asarray(F, dtype=float)
    ranks = rank_rows(F)
    families, members = family_members(feature_names)
    n = F.shape[1]
    raw = np.ones((n, len(families)), dtype=float)
    observed = np.zeros(len(families), dtype=float)
    centers = np.empty_like(raw)
    dispersions = np.full_like(raw, np.nan)
    for family_index, family in enumerate(families):
        indices = members[family]
        local = ranks[indices]
        center = np.median(local, axis=0)
        centers[:, family_index] = center
        if len(indices) < 2:
            continue
        dispersion = np.median(np.abs(local - center[None, :]), axis=0)
        scale = float(np.median(dispersion[dispersion > EPS])) if np.any(
            dispersion > EPS
        ) else 1.0
        raw[:, family_index] = np.exp(-dispersion / (scale + EPS))
        dispersions[:, family_index] = dispersion
        observed[family_index] = 1.0
    return {
        "families": families,
        "members": members,
        "ranks": ranks,
        "centers": centers,
        "dispersions": dispersions,
        "raw_evidence": raw,
        "observed_family": observed,
    }


def smooth_family_evidence(
    raw_evidence,
    observed_family,
    laplacian,
    *,
    beta: float,
    prior_strength: float = 0.10,
):
    """Solve a graph-regularized reliability problem for every sample."""
    raw = np.asarray(raw_evidence, dtype=float)
    observed = np.asarray(observed_family, dtype=float)
    L = np.asarray(laplacian, dtype=float)
    beta = float(beta)
    prior_strength = float(prior_strength)
    if raw.ndim != 2 or observed.shape != (raw.shape[1],):
        raise ValueError("raw evidence and observed mask disagree")
    if L.shape != (raw.shape[1], raw.shape[1]):
        raise ValueError("family Laplacian has the wrong shape")
    if beta < 0 or prior_strength <= 0:
        raise ValueError("beta must be nonnegative and prior_strength positive")
    system = np.diag(observed) + beta * L + prior_strength * np.eye(raw.shape[1])
    rhs = raw * observed[None, :] + prior_strength
    gates = np.linalg.solve(system, rhs.T).T
    gates = np.clip(gates, 0.05, 1.5)
    # Keep mean leverage fixed per sample. The gate may redistribute influence,
    # but cannot improve merely by changing the total score scale.
    gates /= np.mean(gates, axis=1, keepdims=True) + EPS
    return gates


def gated_score(F, weights, families, members, gates, *, blend: float):
    """Apply sample-local family gates to fixed IU-PCR feature weights."""
    F = np.asarray(F, dtype=float)
    weights = np.asarray(weights, dtype=float)
    gates = np.asarray(gates, dtype=float)
    blend = float(blend)
    if weights.shape != (F.shape[0],) or gates.shape != (F.shape[1], len(families)):
        raise ValueError("weights/gates do not match F")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0, 1]")
    feature_gates = np.ones((F.shape[1], F.shape[0]), dtype=float)
    for family_index, family in enumerate(families):
        feature_gates[:, members[family]] = gates[:, family_index, None]
    contributions = F.T * weights[None, :] * feature_gates
    denominator = np.sum(np.abs(weights)[None, :] * feature_gates, axis=1)
    reference = float(np.sum(np.abs(weights)))
    local = np.sum(contributions, axis=1) * reference / (denominator + EPS)
    baseline = weights @ F
    return (1.0 - blend) * baseline + blend * local


def family_expert_scores(F, weights, families, members):
    """Return one fixed IU-weighted score per feature family."""
    F = np.asarray(F, dtype=float)
    weights = np.asarray(weights, dtype=float)
    scores = np.empty((len(families), F.shape[1]), dtype=float)
    for family_index, family in enumerate(families):
        indices = members[family]
        local_weights = weights[indices]
        if np.sum(np.abs(local_weights)) <= EPS:
            local_weights = np.ones(len(indices), dtype=float)
        scores[family_index] = local_weights @ F[indices]
        if np.std(scores[family_index]) <= EPS:
            scores[family_index] = np.mean(F[indices], axis=0)
    return scores


def fit_family_relevance_paths(
    F,
    feature_names,
    *,
    cell: str,
    betas=(0.0, 0.3, 1.0, 3.0),
    blends=(0.25, 0.5, 1.0),
    prior_strength=0.10,
):
    """Fit all registered label-free family-relevance score paths."""
    F = np.asarray(F, dtype=float)
    evidence = local_family_evidence(F, feature_names)
    families = evidence["families"]
    members = evidence["members"]
    state = iu_state(F)
    baseline = state.baseline_weights @ F
    correct_L, adjacency = family_prior_laplacian(families)
    rng = np.random.default_rng(stable_seed(f"family-prior:{cell}"))
    permutation = rng.permutation(len(families))
    wrong_L, wrong_adjacency = family_prior_laplacian(
        families, permutation=permutation
    )
    sample_permutation = rng.permutation(F.shape[1])

    outputs = {
        "sample_index": np.arange(F.shape[1], dtype=np.int64),
        "feature_names": np.asarray(tuple(map(str, feature_names)), dtype=str),
        "family_names": np.asarray(families, dtype=str),
        "iu_pcr": np.asarray(baseline, dtype=np.float64),
        "family_experts": np.asarray(
            family_expert_scores(F, state.baseline_weights, families, members),
            dtype=np.float64,
        ),
        "raw_family_evidence": np.asarray(
            evidence["raw_evidence"], dtype=np.float64
        ),
        "context_trace_length": np.asarray(
            evidence["ranks"][tuple(map(str, feature_names)).index("trace_length")]
            if "trace_length" in tuple(map(str, feature_names))
            else np.full(F.shape[1], 0.5),
            dtype=np.float64,
        ),
        "context_family_disagreement": np.asarray(
            np.std(evidence["centers"], axis=1), dtype=np.float64
        ),
        "context_iu_rank": np.asarray(
            (rankdata(baseline, method="average") - 1.0) / max(F.shape[1] - 1, 1),
            dtype=np.float64,
        ),
    }
    diagnostics = {
        "cell": cell,
        "families": list(families),
        "family_members": {
            family: [str(feature_names[index]) for index in members[family]]
            for family in families
        },
        "family_prior_adjacency": adjacency.tolist(),
        "permuted_family_prior_adjacency": wrong_adjacency.tolist(),
        "family_permutation": permutation.tolist(),
        "observed_family": evidence["observed_family"].tolist(),
        "raw_evidence_mean": np.mean(evidence["raw_evidence"], axis=0).tolist(),
        "raw_evidence_std": np.std(evidence["raw_evidence"], axis=0).tolist(),
        "gate_paths": [],
    }

    for beta in tuple(map(float, betas)):
        correct = smooth_family_evidence(
            evidence["raw_evidence"], evidence["observed_family"], correct_L,
            beta=beta, prior_strength=prior_strength,
        )
        wrong = smooth_family_evidence(
            evidence["raw_evidence"], evidence["observed_family"], wrong_L,
            beta=beta, prior_strength=prior_strength,
        )
        global_gate = np.repeat(correct.mean(axis=0, keepdims=True), F.shape[1], axis=0)
        shuffled = correct[sample_permutation]
        outputs[f"family_gates__beta_{beta:g}"] = np.asarray(
            correct, dtype=np.float64
        )
        outputs[f"permuted_family_gates__beta_{beta:g}"] = np.asarray(
            wrong, dtype=np.float64
        )
        for blend in tuple(map(float, blends)):
            token = f"beta_{beta:g}__blend_{blend:g}"
            outputs[f"manual_graph__{token}"] = np.asarray(
                gated_score(
                    F, state.baseline_weights, families, members, correct, blend=blend
                ), dtype=np.float64
            )
            outputs[f"permuted_graph__{token}"] = np.asarray(
                gated_score(
                    F, state.baseline_weights, families, members, wrong, blend=blend
                ), dtype=np.float64
            )
            outputs[f"global_gate__{token}"] = np.asarray(
                gated_score(
                    F, state.baseline_weights, families, members, global_gate,
                    blend=blend,
                ), dtype=np.float64
            )
            outputs[f"sample_permuted_gate__{token}"] = np.asarray(
                gated_score(
                    F, state.baseline_weights, families, members, shuffled,
                    blend=blend,
                ), dtype=np.float64
            )
        diagnostics["gate_paths"].append({
            "beta": beta,
            "mean_gate": np.mean(correct, axis=0).tolist(),
            "mean_sample_gate_std": float(np.mean(np.std(correct, axis=1))),
            "mean_family_gate_std": float(np.mean(np.std(correct, axis=0))),
            "min_gate": float(np.min(correct)),
            "max_gate": float(np.max(correct)),
        })
    return outputs, diagnostics


def generate_switching_family_world(
    *,
    seed: int,
    n=1200,
    family_sizes=(4, 4, 4, 4, 3, 3),
    correlated_nuisance=False,
    active_noise=0.55,
    inactive_noise=1.2,
):
    """Synthetic world with sample-local family relevance switches.

    Families 0/1 are active in regime 0, families 2/3 in regime 1, family 4 is
    weakly active everywhere, and family 5 is noise.  In the failure world,
    inactive family members share a correlated nuisance and therefore look
    internally consistent even though they do not measure the target.
    """
    rng = np.random.default_rng(int(seed))
    y = rng.standard_normal(int(n))
    labels = (y > np.median(y)).astype(int)
    regime = rng.integers(0, 2, size=int(n))
    rows, names = [], []
    synthetic_families = tuple(VIEW_ORDER[:len(family_sizes)])
    active_sets = ({0, 1}, {2, 3})
    for family_index, (family, size) in enumerate(zip(synthetic_families, family_sizes)):
        nuisance = rng.standard_normal(int(n))
        for member in range(int(size)):
            active = np.asarray([
                family_index in active_sets[int(value)] for value in regime
            ], dtype=bool)
            if family_index == 4:
                signal = 0.45 * y + active_noise * rng.standard_normal(int(n))
            elif family_index == 5:
                signal = inactive_noise * rng.standard_normal(int(n))
            else:
                active_value = y + active_noise * rng.standard_normal(int(n))
                inactive_value = (
                    nuisance + 0.15 * rng.standard_normal(int(n))
                    if correlated_nuisance
                    else inactive_noise * rng.standard_normal(int(n))
                )
                signal = np.where(active, active_value, inactive_value)
            rows.append(signal)
            # Use registered feature names while preserving the requested
            # synthetic family assignment.
            candidates = [name for name, assigned in FEATURE_TO_VIEW.items()
                          if assigned == family]
            names.append(candidates[member % len(candidates)])
    F = np.asarray(rows, dtype=float)
    F = (F - F.mean(axis=1, keepdims=True)) / np.maximum(
        F.std(axis=1, keepdims=True), EPS
    )
    return F, tuple(names), labels, regime


__all__ = [
    "FAMILY_PRIOR_EDGES",
    "family_members",
    "family_prior_laplacian",
    "fit_family_relevance_paths",
    "generate_switching_family_world",
    "gated_score",
    "local_family_evidence",
    "rank_rows",
    "smooth_family_evidence",
]
