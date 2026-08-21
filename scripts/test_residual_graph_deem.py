#!/usr/bin/env python3
"""Dataset-free known-answer tests for Residual-Graph DEEM v1."""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_adapter import (  # noqa: E402
    continuous_to_deem_soft,
    hard_adapter020_config,
    repaired_soft_adapter020_config,
    risk_consensus_align,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ContinuousDeemConfig,
    DufsConfig,
    GraphDeemConfig,
    _FamilyAdditiveEnergy,
    apply_standardization,
    assign_grouped_length_folds,
    atomic_save_npz,
    build_inventory_graph,
    cross_view_dufs,
    donor_risk_matrix,
    fit_continuous_deem,
    fit_standardization,
    graph_health,
    metric_weights,
    persistent_mala,
    present_family_laplacian,
    symmetric_normalized_laplacian,
    unique_edge_loss,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    TargetFreeCellBundle,
    assert_no_target_fields,
    load_registry,
    load_target_free_bundle,
    write_target_free_bundle,
)
from spectral_utils.residual_graph_deem_labels import (  # noqa: E402
    LabelSidecar,
    join_labels_by_id,
)


FAILED = []


def check(name, condition, detail=""):
    state = "PASS" if condition else "FAIL"
    print(f"  [{state}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILED.append(name)


NAMES = (
    "epr", "spectral_entropy", "low_band_power", "epr_spilled",
    "epr_energy", "mean_top1_logprob", "trace_length",
)


def test_registry():
    print("\n1. frozen registry")
    registry = load_registry(ROOT / "configs/residual_graph_deem_24cell_v1_registry.json")
    check("exact 24-cell roster", len(registry["cells"]) == 24)
    check("exact 48,607 rows", sum(cell["n_rows"] for cell in registry["cells"]) == 48_607)
    check("seven real schemas", len(registry["schemas"]) == 7)
    check("38 structural omissions", sum(registry["missing_by_feature"].values()) == 38)
    spilled = next(cell for cell in registry["cells"] if cell["cell_id"] == "spilled_triviaqa_llama8b")
    check("spilled source registered", spilled["source"]["source_size"] == 7_808_360)
    seiclr = next(cell for cell in registry["cells"] if cell["cell_id"] == "seiclr_triviaqa_opt30b")
    check("SE-ICLR stays 19D", seiclr["n_features"] == 19)


def test_firewall_and_join():
    print("\n2. physical target firewall and ID join")
    rng = np.random.default_rng(1)
    rows = tuple(f"cell::q{i}::candidate0" for i in range(12))
    groups = tuple(f"cell::q{i}" for i in range(12))
    bundle = TargetFreeCellBundle(
        cell_id="cell", X_raw=rng.normal(size=(12, len(NAMES))), feature_names=NAMES,
        confidence_signs=np.array([-1, -1, -1, -1, 1, 1, -1], dtype=np.int8),
        row_ids=rows, group_ids=groups, raw_trace_length=np.arange(12) + 3,
        dataset_family="sciq", task_type="QA", source_sha256="a" * 64,
        manifest_sha256="b" * 64, admission_sha256="c" * 64,
        inventory_sha256="d" * 64,
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "bundle.npz"
        write_target_free_bundle(path, bundle)
        loaded = load_target_free_bundle(path)
        with np.load(path, allow_pickle=False) as data:
            check("allow_pickle=False bundle", all(not data[key].dtype.hasobject for key in data.files))
            check("no target-like field", not any("label" in key.lower() for key in data.files))
        check("bundle IDs round trip", loaded.row_ids == rows)
    sidecar = LabelSidecar("cell", tuple(reversed(rows)), np.arange(12, dtype=np.int8)[::-1] % 2)
    joined = join_labels_by_id(bundle, sidecar)
    expected = np.asarray([dict(zip(sidecar.row_ids, sidecar.y_h))[row] for row in rows])
    check("sidecar join is order-independent", np.array_equal(joined, expected))
    try:
        assert_no_target_fields({"X_raw": None, "labels": None})
    except Exception:
        check("target-like payload fails closed", True)
    else:
        check("target-like payload fails closed", False)
    runner = (ROOT / "scripts/run_residual_graph_deem_24cell_v1.py").read_text(encoding="utf-8")
    imports = [node for node in ast.walk(ast.parse(runner)) if isinstance(node, (ast.Import, ast.ImportFrom))]
    check("Stage-A runner imports no evaluation-label module",
          not any("residual_graph_deem_labels" in ast.unparse(node) for node in imports))


def test_transforms_and_folds():
    print("\n3. donor-only transforms and grouped folds")
    donor = np.array([[1., 2., 3., 4., 5., 6., 9.], [1., 3., 2., 5., 4., 7., 9.],
                      [1., 4., 1., 6., 3., 8., 9.]])
    held = np.array([[100., 5., 0., 7., 2., 9., 9.],
                     [200., 6., -1., 8., 1., 10., 9.],
                     [300., 7., -2., 9., 0., 11., 9.]])
    risk_donor, risk_held, transform = donor_risk_matrix(donor, held, NAMES)
    check("constant donor gets scale one", transform.scale[0] == 1 and transform.constant_mask[0])
    check("held row does not alter donor mean", transform.mean[1] == 3.0)
    expected = -(held[0, 1] - donor[:, 1].mean()) / donor[:, 1].std() * -1
    check("orientation applied exactly once", np.isclose(risk_held[0, 1], expected))
    group_ids = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])
    folds = assign_grouped_length_folds(group_ids, np.arange(10) + 1)
    check("siblings never split", all(len(set(folds[group_ids == group])) == 1 for group in set(group_ids)))
    check("folding deterministic", np.array_equal(folds, assign_grouped_length_folds(group_ids, np.arange(10) + 1)))


def test_continuous_model():
    print("\n4. continuous energy, contributions, class swap, replay")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, len(NAMES)))
    config = ContinuousDeemConfig(epochs=4, anchor_tolerance=1e-12, posterior_sd_min=0.0)
    first = fit_continuous_deem(X, NAMES, seed=2, config=config)
    second = fit_continuous_deem(X, NAMES, seed=2, config=config)
    reconstruction = np.max(np.abs(first.aligned_bias + first.contributions.sum(1) - first.logit))
    check("atomic contributions reconstruct", reconstruction <= 1e-8, f"{reconstruction:.3e}")
    check("deterministic replay", np.array_equal(first.score, second.score))
    check("larger class is risk aligned", first.risk_anchor_difference > 0)
    check("posterior is binary normalized", np.allclose(first.posterior.sum(1), 1.0))
    check("finite normalizable-energy training", all(np.isfinite(row["loss"]) for row in first.objective_history))
    alias = fit_continuous_deem(
        X, NAMES, seed=2, config=config,
        graph_config=GraphDeemConfig(lambda_=0.0, mechanism="target"),
        baseline_result=first,
    )
    check("lambda-zero direct alias is exact",
          alias.alias_of == "B3" and np.array_equal(alias.score, first.score)
          and alias.objective_history == first.objective_history)

    # Standard-Gaussian known answer: zero every logit parameter so the marginal
    # free energy is 0.5||x||^2 plus a constant.
    model = _FamilyAdditiveEnergy(NAMES, replace(config, epochs=1), seed=0)
    import torch
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    generator = torch.Generator().manual_seed(9)
    start = torch.randn((512, len(NAMES)), generator=generator, dtype=torch.float64)
    sampled, acceptance = persistent_mala(model, start, delta=0.10, steps=20, generator=generator)
    check("MALA Gaussian mean", abs(float(sampled.mean())) < 0.08)
    check("MALA Gaussian variance", abs(float(sampled.var()) - 1.0) < 0.15)
    check("MALA uses nontrivial valid MH rate", 0.0 < acceptance <= 1.0)


def test_sparse_graph_and_dufs():
    print("\n5. sparse graph, duplicates, DUFS, family Laplacian")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(64, len(NAMES)))
    X[1] = X[0]
    ids = [f"row{i:03d}" for i in range(len(X))]
    W = build_inventory_graph(X, NAMES, ids, k=7)
    health = graph_health(W)
    check("graph is CSR sparse", sparse.isspmatrix_csr(W))
    check("duplicate-safe no self edges", np.allclose(W.diagonal(), 0.0))
    check("symmetric to tolerance", (W - W.T).nnz == 0 or np.max(np.abs((W - W.T).data)) <= 1e-10)
    check("mixed duplicate graph connected", health["largest_component_fraction"] >= 0.90)
    values = rng.normal(size=(64, 2))
    edge = unique_edge_loss(values, W)
    dense = 0.5 * np.sum(W.toarray()[:, :, None] * (values[:, None, :] - values[None, :, :]) ** 2)
    check("sparse edge loss matches dense fixture", np.isclose(edge, dense, atol=1e-10))
    folds = np.arange(64) % 5
    gates, diagnostics = cross_view_dufs(
        X, NAMES, folds, ids,
        config=DufsConfig(epochs=3, seeds=(0, 1), median_cosine_min=-1.0),
    )
    expected_mass = 1 / len(diagnostics["family_mass"])
    check("DUFS exact equal family mass",
          all(np.isclose(value, expected_mass, atol=1e-10)
              for value in diagnostics["family_mass"].values()))
    check("DUFS target family excluded", diagnostics["target_family_excluded_from_reference"])
    check("DUFS effective count finite", np.isfinite(diagnostics["effective_feature_count"]))
    family_L, order, affinity = present_family_laplacian(X, NAMES)
    check("present-family graph dimensions", family_L.shape == (len(order), len(order)))
    check("family affinity symmetric", np.allclose(affinity, affinity.T))


def test_adapter_contract():
    print("\n6. packaged adapter controls")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(51, len(NAMES)))
    X[:, 0] = np.round(X[:, 0], 1)
    soft = continuous_to_deem_soft(X)
    check("average-rank pseudo-probabilities normalize", np.allclose(soft.sum(1), 1.0))
    check("hard/soft frozen learning rates",
          hard_adapter020_config().learning_rate == 1e-3
          and repaired_soft_adapter020_config().learning_rate == 1e-4)
    raw = np.column_stack([np.linspace(.9, .1, len(X)), np.linspace(.1, .9, len(X))])
    aligned, mapping, difference = risk_consensus_align(raw, X, feature_names=NAMES, tolerance=1e-12)
    swapped, swapped_mapping, _ = risk_consensus_align(raw[:, ::-1], X, feature_names=NAMES, tolerance=1e-12)
    check("class permutation leaves semantic score unchanged", np.allclose(aligned[:, 1], swapped[:, 1]))
    check("risk mapping is explicit", set(mapping) == {0, 1} and set(swapped_mapping) == {0, 1} and difference > 0)
    from scripts.deem_soft_collapse_probe import one_fit
    check("winner-validation regression signature fixed",
          list(inspect.signature(one_fit).parameters) ==
          ["F_a", "cell_key", "stage", "seed", "config", "artifact_dir"])


def main():
    test_registry()
    test_firewall_and_join()
    test_transforms_and_folds()
    test_continuous_model()
    test_sparse_graph_and_dufs()
    test_adapter_contract()
    if FAILED:
        raise SystemExit(f"{len(FAILED)} test(s) failed: {', '.join(FAILED)}")
    print("\nALL RESIDUAL-GRAPH DEEM TESTS PASSED")


if __name__ == "__main__":
    main()
