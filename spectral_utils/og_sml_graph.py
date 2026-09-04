"""Graph-identifiability primitives for the OG-SML Agent B study.

The module is deliberately independent of benchmark loaders and outcome code.
Groups are hard vertex sets (cliques); soft co-assignment matrices are outside
this model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


Array = np.ndarray


def _canonical_groups(groups: Sequence[Iterable[int]], p: int | None) -> tuple[tuple[int, ...], ...]:
    canonical = tuple(tuple(sorted({int(i) for i in group})) for group in groups)
    if p is None:
        if not canonical or not any(canonical):
            raise ValueError("p is required when groups do not contain vertices")
        p = 1 + max(i for group in canonical for i in group)
    if p < 1:
        raise ValueError("p must be positive")
    for group in canonical:
        if any(i < 0 or i >= p for i in group):
            raise ValueError("group vertex outside [0, p)")
    return canonical


def groups_from_partition(labels: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Convert integer partition labels to deterministically ordered vertex sets."""
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1 or labels_arr.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    return tuple(tuple(np.flatnonzero(labels_arr == label).tolist()) for label in sorted(set(labels_arr.tolist())))


def _clique_adjacency(group: Iterable[int], p: int) -> Array:
    adjacency = np.zeros((p, p), dtype=bool)
    vertices = np.asarray(tuple(group), dtype=int)
    if vertices.size >= 2:
        adjacency[np.ix_(vertices, vertices)] = True
        adjacency[vertices, vertices] = False
    return adjacency


def free_graph(groups: Sequence[Iterable[int]], p: int | None = None) -> Array:
    """Return H = K_p minus the union of all within-group clique edges."""
    canonical = _canonical_groups(groups, p)
    if p is None:
        p = 1 + max(i for group in canonical for i in group)
    union = np.zeros((p, p), dtype=bool)
    for group in canonical:
        union |= _clique_adjacency(group, p)
    graph = ~union
    np.fill_diagonal(graph, False)
    return graph


def exclusive_graphs(groups: Sequence[Iterable[int]], p: int | None = None) -> tuple[Array, ...]:
    """Return the edges unique to each group clique."""
    canonical = _canonical_groups(groups, p)
    if p is None:
        p = 1 + max(i for group in canonical for i in group)
    cliques = tuple(_clique_adjacency(group, p) for group in canonical)
    if not cliques:
        return tuple()
    counts = np.sum(np.stack(cliques, axis=0), axis=0)
    return tuple(clique & (counts == 1) for clique in cliques)


def connected_components(adjacency: Array, vertices: Sequence[int] | None = None) -> tuple[tuple[int, ...], ...]:
    adjacency = np.asarray(adjacency, dtype=bool)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric")
    selected = tuple(range(adjacency.shape[0])) if vertices is None else tuple(sorted({int(v) for v in vertices}))
    allowed = set(selected)
    unseen = set(selected)
    components: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        queue: deque[int] = deque([root])
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            neighbours = [int(v) for v in np.flatnonzero(adjacency[node]) if int(v) in allowed and int(v) in unseen]
            for neighbour in neighbours:
                unseen.remove(neighbour)
                queue.append(neighbour)
        components.append(tuple(sorted(component)))
    return tuple(components)


def is_bipartite(adjacency: Array, vertices: Sequence[int] | None = None) -> bool:
    adjacency = np.asarray(adjacency, dtype=bool)
    selected = tuple(range(adjacency.shape[0])) if vertices is None else tuple(sorted({int(v) for v in vertices}))
    allowed = set(selected)
    colours: dict[int, int] = {}
    for root in selected:
        if root in colours:
            continue
        colours[root] = 0
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            for neighbour_raw in np.flatnonzero(adjacency[node]):
                neighbour = int(neighbour_raw)
                if neighbour not in allowed:
                    continue
                if neighbour not in colours:
                    colours[neighbour] = 1 - colours[node]
                    queue.append(neighbour)
                elif colours[neighbour] == colours[node]:
                    return False
    return True


def fiedler(adjacency: Array, vertices: Sequence[int] | None = None) -> float:
    """Return lambda_2 of the combinatorial Laplacian of a weighted graph."""
    weights = np.asarray(adjacency, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.allclose(weights, weights.T, rtol=0.0, atol=1e-12):
        raise ValueError("adjacency must be symmetric")
    if np.any(weights < -1e-15):
        raise ValueError("edge weights must be non-negative")
    selected = np.arange(weights.shape[0], dtype=int) if vertices is None else np.asarray(sorted({int(v) for v in vertices}), dtype=int)
    if selected.size < 2:
        return 0.0
    sub = weights[np.ix_(selected, selected)]
    laplacian = np.diag(np.sum(sub, axis=1)) - sub
    eigenvalues = np.linalg.eigvalsh(laplacian)
    value = float(eigenvalues[1])
    return 0.0 if abs(value) < 1e-12 else value


@dataclass(frozen=True)
class GraphIdentifiabilityReport:
    admissible: bool
    p: int
    group_count: int
    group_sizes: tuple[int, ...]
    free_edge_count: int
    free_component_count: int
    free_component_sizes: tuple[int, ...]
    free_connected: bool
    free_bipartite: bool
    free_non_bipartite: bool
    free_fiedler_weighted: float
    exclusive: tuple[dict[str, object], ...]
    j_raw: float
    j_selection: float
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def graph_identifiability_report(
    groups: Sequence[Iterable[int]],
    *,
    p: int,
    global_loading: Sequence[float] | None = None,
) -> GraphIdentifiabilityReport:
    canonical = _canonical_groups(groups, p)
    h = free_graph(canonical, p)
    h_components = connected_components(h)
    h_bipartite = is_bipartite(h)
    if global_loading is None:
        h_weights = h.astype(float)
    else:
        loading = np.asarray(global_loading, dtype=float)
        if loading.shape != (p,) or not np.all(np.isfinite(loading)):
            raise ValueError("global_loading must contain p finite values")
        h_weights = h.astype(float) * np.abs(np.outer(loading, loading))
    h_fiedler = fiedler(h_weights)

    blockers: list[str] = []
    if len(h_components) != 1:
        blockers.append("FREE_GRAPH_DISCONNECTED")
    if h_bipartite:
        blockers.append("FREE_GRAPH_BIPARTITE")

    exclusive_reports: list[dict[str, object]] = []
    exclusive_fiedlers: list[float] = []
    for group_index, graph in enumerate(exclusive_graphs(canonical, p)):
        vertices = tuple(int(i) for i in np.flatnonzero(np.any(graph, axis=1)))
        components = connected_components(graph, vertices) if vertices else tuple()
        bipartite = is_bipartite(graph, vertices) if vertices else True
        value = fiedler(graph.astype(float), vertices) if vertices else 0.0
        edge_count = int(np.count_nonzero(np.triu(graph, 1)))
        triangle_condition = len(vertices) >= 3 and not bipartite
        connected = len(components) == 1
        if len(vertices) < 3:
            blockers.append(f"GROUP_{group_index}_EXCLUSIVE_SUPPORT_LT3")
        if not connected:
            blockers.append(f"GROUP_{group_index}_EXCLUSIVE_DISCONNECTED")
        if bipartite:
            blockers.append(f"GROUP_{group_index}_EXCLUSIVE_BIPARTITE")
        exclusive_fiedlers.append(value)
        exclusive_reports.append(
            {
                "group_index": group_index,
                "group_size": len(canonical[group_index]),
                "exclusive_vertex_count": len(vertices),
                "exclusive_edge_count": edge_count,
                "component_count": len(components),
                "component_sizes": [len(component) for component in components],
                "connected_on_nonisolated": connected,
                "bipartite_on_nonisolated": bipartite,
                "non_bipartite_on_nonisolated": not bipartite,
                "triangle_condition": triangle_condition,
                "fiedler_unweighted": value,
            }
        )

    score_terms = [h_fiedler, *exclusive_fiedlers]
    j_raw = float(min(score_terms)) if score_terms else h_fiedler
    admissible = not blockers
    return GraphIdentifiabilityReport(
        admissible=admissible,
        p=p,
        group_count=len(canonical),
        group_sizes=tuple(len(group) for group in canonical),
        free_edge_count=int(np.count_nonzero(np.triu(h, 1))),
        free_component_count=len(h_components),
        free_component_sizes=tuple(len(component) for component in h_components),
        free_connected=len(h_components) == 1,
        free_bipartite=h_bipartite,
        free_non_bipartite=not h_bipartite,
        free_fiedler_weighted=h_fiedler,
        exclusive=tuple(exclusive_reports),
        j_raw=j_raw,
        j_selection=j_raw if admissible else 0.0,
        blockers=tuple(blockers),
    )


def is_admissible(groups: Sequence[Iterable[int]], p: int | None = None) -> bool:
    canonical = _canonical_groups(groups, p)
    if p is None:
        p = 1 + max(i for group in canonical for i in group)
    return graph_identifiability_report(canonical, p=p).admissible


def identifiability_score(
    groups: Sequence[Iterable[int]],
    *,
    p: int,
    global_loading: Sequence[float] | None = None,
) -> float:
    """Return J for admissible families and zero for inadmissible families."""
    return graph_identifiability_report(groups, p=p, global_loading=global_loading).j_selection

