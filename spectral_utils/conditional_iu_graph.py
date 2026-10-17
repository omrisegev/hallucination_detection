"""Sparse token graphs, population moments, and a convex coefficient penalty.

The +/-16 whole-answer token window and eighth *in-window* neighbor are our
application design. Only the locally scaled affinity exp(-d_ij^2/(s_i s_j))
is borrowed from Zelnik-Manor and Perona, Self-Tuning Spectral Clustering
(NIPS 2004), equations (1)--(2):
https://proceedings.neurips.cc/paper_files/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf
Their experiments used K=7, and their full clustering algorithm is not used.

The coefficient objective uses Hallac, Leskovec and Boyd's Network Lasso
(KDD 2015) edge L2 norm, not its square:
https://web.stanford.edu/~boyd/papers/network_lasso.html
Our solver is the primal-dual algorithm of Chambolle and Pock (2011),
doi:10.1007/s10851-010-0251-1, rather than that paper's distributed ADMM.

These primitives never accept labels, step boundaries, or benchmark folds.
The caller supplies already standardized features and any scientific edge
normalization. No token-by-token dense adjacency or distance matrix is built.
"""

from __future__ import annotations

import hashlib
from time import perf_counter

import numpy as np
from scipy import sparse


def _finite_array(value, name: str) -> np.ndarray:
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"{name} contains missing/masked values")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real array") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains missing or nonfinite values")
    return result


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _features(value) -> np.ndarray:
    x = _finite_array(value, "features")
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("features must have shape (tokens >= 1, features >= 1)")
    return x


def _adjacency(value, n_tokens: int | None = None) -> sparse.csr_matrix:
    if not sparse.issparse(value):
        raise ValueError("adjacency must be sparse; dense token-square arrays are not accepted")
    _finite_array(value.data, "adjacency weights")
    a = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    if a.shape[0] != a.shape[1] or a.shape[0] < 1:
        raise ValueError("adjacency must be a nonempty square sparse matrix")
    if n_tokens is not None and a.shape != (n_tokens, n_tokens):
        raise ValueError("adjacency size does not match the token count")
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    if not np.all(np.isfinite(a.data)) or np.any(a.data < 0):
        raise ValueError("adjacency weights must be finite and nonnegative")
    if np.any(a.diagonal() != 0):
        raise ValueError("adjacency must have zero diagonal; moment self-weight is added separately")
    if (a - a.T).nnz:
        raise ValueError("adjacency must be symmetric")
    return a


def build_token_graph(
    standardized_features,
    *,
    mode: str = "affinity",
    window: int = 16,
    bandwidth_neighbor: int = 8,
    bandwidth_floor: float = 1e-8,
) -> sparse.csr_matrix:
    """Return a symmetric CSR graph over one complete answer's token rows.

    Candidates are all distinct tokens within +/- ``window`` positions, even
    across step boundaries. Affinity bandwidth is the distance to the eighth
    closest candidate (the furthest available candidate on short answers),
    floored at ``bandwidth_floor``. Duplicates have affinity one. ``uniform``
    uses weight one on precisely the same candidate window. Neither mode
    adds self-loops or standardizes the caller's feature array again.

    Memory is O(T * window + T * feature_count), including local distances;
    the bandwidth sort only sees a single token's candidate list at a time.
    """
    x = _features(standardized_features)
    window = _positive_integer(window, "window")
    bandwidth_neighbor = _positive_integer(bandwidth_neighbor, "bandwidth_neighbor")
    if not np.isfinite(bandwidth_floor) or bandwidth_floor <= 0:
        raise ValueError("bandwidth_floor must be finite and positive")
    if mode not in {"affinity", "uniform"}:
        raise ValueError("mode must be 'affinity' or 'uniform'")
    n = x.shape[0]
    offsets = range(1, min(window, n - 1) + 1)
    left = [np.arange(n - offset, dtype=np.int64) for offset in offsets]
    if not left:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    i = np.concatenate(left)
    j = np.concatenate([row + offset for row, offset in zip(left, offsets)])
    if mode == "uniform":
        weight = np.ones(i.size, dtype=np.float64)
    else:
        # Work one offset at a time to avoid an (edge_count, feature_count)
        # temporary, which can substantially exceed the sparse graph itself.
        distances = np.empty(i.size, dtype=np.float64)
        start = 0
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for offset in offsets:
                difference = x[:-offset] - x[offset:]
                size = n - offset
                distances[start:start + size] = np.sqrt(
                    np.einsum("ij,ij->i", difference, difference)
                )
                start += size
        distance_graph = sparse.csr_matrix(
            (np.concatenate([distances, distances]),
             (np.concatenate([i, j]), np.concatenate([j, i]))),
            shape=(n, n),
        )
        # Do not eliminate zeros: a duplicate token is a real neighbor.
        bandwidth = np.full(n, float(bandwidth_floor))
        for token in range(n):
            row = distance_graph.data[distance_graph.indptr[token]:distance_graph.indptr[token + 1]]
            rank = min(bandwidth_neighbor, row.size) - 1
            bandwidth[token] = max(float(np.partition(row, rank)[rank]), bandwidth_floor)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            scaled = distances / np.sqrt(bandwidth[i]) / np.sqrt(bandwidth[j])
            weight = np.exp(-(scaled * scaled))
    graph = sparse.csr_matrix(
        (np.concatenate([weight, weight]),
         (np.concatenate([i, j]), np.concatenate([j, i]))),
        shape=(n, n),
    )
    # Numerical exponential underflow gives legitimate zero-weight edges.
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def permute_token_graph(adjacency, uid: str) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return ``A[p][:, p]`` with one SHA256 UID-seeded node permutation.

    This is a graph isomorphism, preserving edge weights and degree multiset.
    It changes which graph vertex is assigned to each *unchanged* feature row.
    The caller must not apply ``p`` to features, scores, labels, or readouts.
    """
    a = _adjacency(adjacency)
    if not isinstance(uid, str) or not uid:
        raise ValueError("uid must be a nonempty stable string")
    digest = hashlib.sha256(("conditional-iu-graph-v1\0" + uid).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    permutation = np.random.default_rng(seed).permutation(a.shape[0])
    shuffled = a[permutation][:, permutation].tocsr()
    shuffled.sort_indices()
    return shuffled, permutation


def graph_edges(adjacency) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract each positive undirected edge once, with ``i < j``."""
    upper = sparse.triu(_adjacency(adjacency), k=1, format="coo")
    return upper.row.astype(np.int64), upper.col.astype(np.int64), upper.data.copy()


def weighted_neighborhood_moments(features, adjacency) -> dict:
    """Population moments of each graph neighborhood, with self-weight one.

    Weights are the original edge affinities, without degree normalization.
    Covariance divides by sum(weights), without a degrees-of-freedom factor.
    ``effective_weight_count = sum(w)**2/sum(w**2)`` describes concentration
    of weights; correlated neighboring tokens are NOT independent samples.

    Output covariance costs O(T*D**2); each sparse product uses a T-vector,
    not T*D**2 product columns or a T*T matrix. A common reference translation
    reduces cancellation when input coordinates have large constant offsets.
    """
    x = _features(features)
    n, d = x.shape
    a = _adjacency(adjacency, n)
    weights = a + sparse.eye(n, format="csr")
    weight_sum = np.asarray(weights.sum(axis=1)).ravel()
    squared_sum = np.asarray(weights.multiply(weights).sum(axis=1)).ravel()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        shifted = x - x[0]
        centered_mean = (weights @ shifted) / weight_sum[:, None]
        mean = centered_mean + x[0]
        covariance = np.empty((n, d, d), dtype=np.float64)
        for p in range(d):
            for q in range(p, d):
                second = (weights @ (shifted[:, p] * shifted[:, q])) / weight_sum
                entry = second - centered_mean[:, p] * centered_mean[:, q]
                if p == q:
                    # Only remove floating-point roundoff, never a material
                    # negative variance or a failed numerical calculation.
                    bound = 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(second))
                    if np.any(entry < -bound):
                        raise FloatingPointError("materially negative neighborhood variance")
                    entry = np.maximum(entry, 0.0)
                covariance[:, p, q] = entry
                covariance[:, q, p] = entry
        effective_weight_count = weight_sum * weight_sum / squared_sum
    for name, value in (("mean", mean), ("covariance", covariance),
                        ("weight_sum", weight_sum), ("effective_weight_count", effective_weight_count)):
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"nonfinite neighborhood {name}")
    return {"mean": mean, "covariance": covariance, "weight_sum": weight_sum,
            "effective_weight_count": effective_weight_count}


def _edge_arrays(edge_i, edge_j, edge_weight, n: int):
    index_arrays = []
    for value, name in ((edge_i, "edge_i"), (edge_j, "edge_j")):
        values = _finite_array(value, name)
        if values.ndim != 1 or np.any(values != np.floor(values)):
            raise ValueError(f"{name} must be a vector of integer token indices")
        if np.any(values < 0) or np.any(values >= n):
            raise ValueError(f"{name} contains an out-of-range token index")
        index_arrays.append(values.astype(np.int64))
    i, j = index_arrays
    a = _finite_array(edge_weight, "edge_weight")
    if a.ndim != 1 or i.shape != j.shape or i.shape != a.shape:
        raise ValueError("edge_i, edge_j and edge_weight must be equal-length vectors")
    if np.any(i >= j) or np.any(a < 0):
        raise ValueError("edges require i < j and nonnegative weights")
    if i.size:
        order = np.lexsort((j, i))
        if np.any((i[order][1:] == i[order][:-1]) & (j[order][1:] == j[order][:-1])):
            raise ValueError("duplicate edges are not allowed")
    return i, j, a


def solve_network_lasso(
    G, r, edge_i, edge_j, edge_weight, eta: float, *, max_iter: int = 3000, tol: float = 1e-6,
) -> dict:
    """Minimize sum(.5 theta'G theta-r'theta)+eta*sum(a*||theta_i-theta_j||2).

    ``G`` is (T,2,2), symmetric positive definite; ``r`` is (T,2). ``a``
    is never normalized here. The application caller may scale sum(a) to T/2.
    Eta zero returns an exact batched local solve with zero iterations.

    For nonzero penalties use Chambolle-Pock with extrapolation one. An
    unweighted incidence K has ||K||**2 <= 2*max_degree. We fix
    tau=sigma=.99/sqrt(2*max_degree), so tau*sigma*||K||**2 < 1.
    Primal prox is (I+tau*G)^(-1)(v+tau*r); each dual row is projected on
    the Euclidean ball of radius eta*a. Stop when gap/max(1,|P|,|D|)<=tol,
    checked every 25 iterations and at the iteration cap. The registered
    application settings are max_iter=3000, tol=1e-6; overrides aid tests.

    The unrestricted dual bound is D=-.5*sum((r-K'p)'G^-1(r-K'p)).
    Gap is evaluated by its nonnegative Fenchel-residual decomposition to
    avoid subtraction of nearly equal objectives. Only roundoff is clipped.
    Iteration exhaustion returns a finite iterate and converged=False;
    invalid inputs or numerical failure raise, with no substitute scores.
    """
    start = perf_counter()
    g = _finite_array(G, "G")
    rhs = _finite_array(r, "r")
    if g.ndim != 3 or g.shape[1:] != (2, 2) or g.shape[0] < 1:
        raise ValueError("G must have shape (tokens >= 1, 2, 2)")
    n = g.shape[0]
    if rhs.shape != (n, 2):
        raise ValueError("r must have shape (tokens, 2)")
    if not np.allclose(g, g.swapaxes(1, 2), rtol=1e-12, atol=1e-12):
        raise ValueError("G must be symmetric")
    # Accept only roundoff asymmetry, explicitly removing it for the SPD solve.
    g = 0.5 * g + 0.5 * g.swapaxes(1, 2)
    try:
        np.linalg.cholesky(g)
    except np.linalg.LinAlgError as exc:
        raise ValueError("every G must be positive definite") from exc
    i, j, a = _edge_arrays(edge_i, edge_j, edge_weight, n)
    max_iter = _positive_integer(max_iter, "max_iter")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        theta = np.linalg.solve(g, rhs[..., None])[..., 0]
        inverse_g = np.linalg.inv(g)
        radius = float(eta) * a
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(inverse_g)):
        raise FloatingPointError("nonfinite local quadratic solve")
    # Zero-weight edges do not affect the objective or incidence norm bound.
    active = radius > 0
    i_active, j_active, radius_active = i[active], j[active], radius[active]
    m = i_active.size
    rows = np.repeat(np.arange(m), 2)
    cols = np.column_stack([i_active, j_active]).ravel()
    incidence = sparse.csr_matrix((np.tile([1.0, -1.0], m), (rows, cols)), shape=(m, n))
    transpose = incidence.T.tocsr()
    dual = np.zeros((m, 2), dtype=np.float64)

    def diagnostics(current, dual_value):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            differences = incidence @ current
            difference_norm = np.hypot(differences[:, 0], differences[:, 1])
            kt_dual = transpose @ dual_value
            quadratic_gradient = np.einsum("tij,tj->ti", g, current) - rhs
            residual = quadratic_gradient + kt_dual
            shifted_rhs = rhs - kt_dual
            local_penalty = radius_active * difference_norm
            edge_gap = local_penalty - np.einsum("ei,ei->e", dual_value, differences)
            roundoff = 64 * np.finfo(float).eps * np.maximum(1.0, local_penalty)
            if np.any(edge_gap < -roundoff):
                raise FloatingPointError("dual iterate violates the edge L2 constraint")
            primal = float(0.5 * np.einsum("ti,tij,tj->", current, g, current)
                           - np.einsum("ti,ti->", rhs, current) + local_penalty.sum())
            dual_objective = float(-0.5 * np.einsum("ti,tij,tj->", shifted_rhs, inverse_g, shifted_rhs))
            stationarity_gap = float(0.5 * np.einsum("ti,tij,tj->", residual, inverse_g, residual))
            if stationarity_gap < 0:
                raise FloatingPointError("negative SPD stationarity residual")
            gap = stationarity_gap + float(np.maximum(edge_gap, 0.0).sum())
            relative = gap / max(1.0, abs(primal), abs(dual_objective))
        if not np.all(np.isfinite([primal, dual_objective, gap, relative])):
            raise FloatingPointError("nonfinite Network Lasso diagnostics")
        return {"primal_objective": primal, "dual_objective": dual_objective,
                "primal_dual_gap": gap, "relative_gap": relative}

    report = diagnostics(theta, dual)
    if not m:
        # The mathematical optimum is exactly the uncoupled solve. Keep
        # diagnostics' tiny roundoff residual honest instead of replacing it.
        return {"theta": theta, **report, "iterations": 0, "converged": True,
                "elapsed_seconds": perf_counter() - start, "status": "exact_unpenalized",
                "tau": 0.0, "sigma": 0.0, "incidence_norm_squared_bound": 0.0}
    degree = np.bincount(np.concatenate([i_active, j_active]), minlength=n)
    norm_bound = float(2 * degree.max())
    tau = sigma = 0.99 / np.sqrt(norm_bound)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        inverse_prox = np.linalg.inv(np.eye(2)[None, :, :] + tau * g)
    extrapolated = theta.copy()
    converged = report["relative_gap"] <= tol
    iterations = 0
    for iteration in range(1, max_iter + 1):
        if converged:
            break
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            dual += sigma * (incidence @ extrapolated)
            norms = np.hypot(dual[:, 0], dual[:, 1])
            outside = norms > radius_active
            dual[outside] *= (radius_active[outside] / norms[outside])[:, None]
            previous = theta
            prox_rhs = theta - tau * (transpose @ dual) + tau * rhs
            theta = np.einsum("tij,tj->ti", inverse_prox, prox_rhs)
            extrapolated = 2.0 * theta - previous
        iterations = iteration
        if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(dual)):
            raise FloatingPointError("nonfinite Network Lasso iterate")
        if iteration % 25 == 0 or iteration == max_iter:
            report = diagnostics(theta, dual)
            converged = report["relative_gap"] <= tol
    return {"theta": theta, **report, "iterations": iterations, "converged": bool(converged),
            "elapsed_seconds": perf_counter() - start,
            "status": "converged" if converged else "max_iter",
            "tau": float(tau), "sigma": float(sigma),
            "incidence_norm_squared_bound": norm_bound}
