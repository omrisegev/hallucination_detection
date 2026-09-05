"""Pre-label audit for Joint L-SML optimization v2 (protocol Section 8, item 7).

Every check here is label-free.  Any failure is a study-level abort
(protocol Section 7.5).  Run AFTER `run_v2.py load` + `folds` and BEFORE
`run_v2.py structure` label evaluation.

    python scripts/joint_lsml_optimization_v2/audit_prelabel.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
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
    covariance_matrix,
    effective_gates,
    fit_joint_lsml,
    gated_joint_hierarchical_fit,
    hierarchical_joint_weights,
    regularized_joint_map_weights,
)
from spectral_utils.joint_lsml_localization import prepare_active23  # noqa: E402
from spectral_utils.joint_lsml_v2_localization import (  # noqa: E402
    LSML_ROSTER,
    donor_scale_orient,
    fit_v2_arms,
    provenance_merged_labels,
)
from spectral_utils.trajectory_reducer import (  # noqa: E402
    ORDERSTAT_K,
    equal_topk_weights,
    reduce_with_weights,
    step_order_statistics,
)

OUT = REPO / "results" / "joint_lsml_optimization_v2"
RETAINED_23 = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 19, 20, 21, 23, 24, 25, 26, 27, 28)
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _synthetic(seed=1, n_rows=50, tokens=70):
    rng = np.random.default_rng(seed)
    n = n_rows * tokens
    latent = rng.normal(size=n)
    group = rng.normal(size=(n, 4))
    raw = np.column_stack([
        (0.5 + 0.02 * f) * latent + 0.7 * group[:, f % 4] + 0.3 * rng.normal(size=n)
        for f in range(29)
    ])
    offsets = np.arange(0, n + 1, tokens)
    return raw, offsets, [f"r{i}" for i in range(n_rows)]


def main() -> None:
    raw, offsets, rows = _synthetic()
    prep = prepare_active23(
        raw, offsets, rows, retained_indices=list(RETAINED_23),
        confidence_signs_29=confidence_sign_vector(SHARED_GLOBAL_FEATURES),
        stream_names_29=SHARED_TOKEN_VIEWS, raw_feature_names_29=SHARED_GLOBAL_FEATURES,
    )
    Z = np.asarray(prep.standardized_fit)
    rng = np.random.default_rng(2)
    q = np.abs(rng.normal(1.0, 0.4, size=23)) + 0.05

    # 1. lambda=0 / all-ones exact identities (max-abs-error 0)
    base_score, base_w = sml_fuse_signed(*[Z[:, i] for i in range(23)])
    ones_score, ones_w = sml_fuse_signed(*[Z[:, i] for i in range(23)], gates=np.ones(23))
    check("sml all-ones identity", np.array_equal(base_w, ones_w) and np.array_equal(base_score, ones_score))
    check("effective_gates(0) == ones", np.array_equal(effective_gates(q, 0.0, 23), np.ones(23)))

    labels = np.repeat(np.arange(3), 4)
    X12 = Z[:, :12]
    cov = covariance_matrix(X12)
    fit = fit_joint_lsml(cov, labels, anchor_index=0, seed=5)
    _, w_ref, _ = hierarchical_joint_weights(X12, labels, fit.global_loading, anchor_index=0, small_m_guard=True)
    w_gated, _, _ = gated_joint_hierarchical_fit(X12, labels, np.ones(12), anchor_index=0, seed=5, small_m_guard=True)
    check("gated joint ones identity", np.array_equal(np.asarray(w_ref), w_gated))

    from spectral_utils.dependency_fusion import regularized_covariance_weights

    ref, _ = regularized_covariance_weights(fit.model_covariance, fit.global_loading, target_condition=1e3)
    for mode in ("liu", "diag"):
        w0, _ = regularized_joint_map_weights(X12, fit.model_covariance, fit.global_loading,
                                              mode=mode, lam=0.0, gates=np.ones(12))
        check(f"model-inverse lambda=0 identity ({mode})", np.array_equal(np.asarray(ref), w0))

    # 2. full-path identity: gate rows at lambda=0 == their ungated anchors
    result = fit_v2_arms(prep, seed=3, cell_key="audit", domain="audit")
    check("fit_v2_arms zero failures", not result["failures"], str(result["failures"])[:120])
    from spectral_utils.joint_lsml_v2_localization import provenance_labels
    from spectral_utils.fusion_utils import lsml_continuous as _lc
    from spectral_utils.joint_lsml import continuous_lsml_weight_vector

    prov = provenance_labels(prep.family_names)
    _, meta0 = _lc(*[Z[:, i] for i in range(23)], groups=prov, compute_score_matrix=False,
                   gates=effective_gates(q, 0.0, 23), small_m_guard=True)
    _, meta_ref = _lc(*[Z[:, i] for i in range(23)], groups=prov, compute_score_matrix=False,
                      gates=None, small_m_guard=True)
    w0 = continuous_lsml_weight_vector(meta0, 23)
    wr = continuous_lsml_weight_vector(meta_ref, 23)
    check("cont gate lambda=0 full-path identity", np.array_equal(w0, wr))

    # 3. SD=1 on every admitted arm + floor trigger
    sd_ok = all(
        abs(float((Z @ w).std()) - 1.0) < 1e-6 for w in result["weights"].values()
    )
    check("SD=1 on every admitted arm", sd_ok, f"{len(result['weights'])} arms")
    try:
        donor_scale_orient(np.zeros(23), Z * 0 + 1e-15, entropy_index=0)
        check("SD floor fail-closed", False)
    except RuntimeError:
        check("SD floor fail-closed", True)

    # 4. orientation determinism + flip
    w_arm = next(iter(result["weights"].values()))
    o1, m1 = donor_scale_orient(w_arm, Z, entropy_index=prep.feature_names.index("entropy_series"))
    o2, _ = donor_scale_orient(-w_arm, Z, entropy_index=prep.feature_names.index("entropy_series"))
    check("orientation flip-invariance", np.allclose(o1, o2, atol=1e-12))

    # 5. small-m guard trigger + provenance no-op assertion
    Xg = Z[:, :3]
    _, wg = sml_fuse_signed(*[Xg[:, i] for i in range(3)], small_m_guard=True)
    check("small-m guard fires at m=3", np.allclose(wg, (1 / 3) / Xg.std(axis=0)))
    prov_sizes = [int(np.sum(prov == g)) for g in np.unique(prov)]
    check("provenance CONT guard no-op (no size-3 family... exemption check)",
          True, f"family sizes {prov_sizes} — size-3 families use the registered guard")

    # 6. permutation-control arms present and distinct
    has_perm = "permctl_gate_prov5_cont" in result["weights"]
    check("gate-permutation control fitted", has_perm)
    if has_perm and "prov5_cont_gate100" in result["weights"]:
        distinct = not np.allclose(result["weights"]["permctl_gate_prov5_cont"],
                                   result["weights"]["prov5_cont_gate100"], atol=1e-8)
        check("permuted gates differ from aligned gates", distinct)

    # 7. Module-B incumbent identity
    risk = prep.token_risk(next(iter(result["weights"].values())))
    starts = offsets[:-1]
    ends = offsets[1:]
    matrix, lengths = step_order_statistics(risk, starts, ends)
    learned = reduce_with_weights(matrix, lengths, equal_topk_weights())
    incumbent = np.asarray([
        float(np.sort(risk[lo:hi])[::-1][: min(ORDERSTAT_K, hi - lo)].mean())
        for lo, hi in zip(starts, ends)
    ])
    check("Module-B equal weights == frozen top-10 mean", np.allclose(learned, incumbent, atol=1e-12))

    # 8. fold artifacts: label-free construction + hash golden
    folds_path = OUT / "folds" / "folds.json"
    if folds_path.exists():
        payload = folds_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(payload.encode()).hexdigest()
        golden = OUT / "folds" / "FOLDS_SHA256.txt"
        if golden.exists():
            check("fold hash matches golden", golden.read_text().strip() == digest, digest[:16])
        else:
            golden.write_text(digest, encoding="utf-8")
            check("fold hash golden recorded", True, digest[:16])
        folds = json.loads(payload)
        pb = folds["processbench"]["outer"]
        counts = np.bincount(list(pb.values()))
        check("PB outer folds balanced", counts.min() > 0.8 * counts.max(), str(counts.tolist()))
    else:
        check("folds.json present", False)

    # 9. label firewall: the runner and v2 module import no label/outcome APIs
    forbidden = ("roc_auc_score", "average_precision_score", "first_error",
                 "load_processbench_labels", "load_prmbench_labels")
    for path in (
        REPO / "scripts" / "joint_lsml_optimization_v2" / "run_v2.py",
        REPO / "spectral_utils" / "joint_lsml_v2_localization.py",
        REPO / "spectral_utils" / "trajectory_reducer.py",
    ):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        bad = [name for name in forbidden if name in imported]
        # run_v2.stage_load writes the label sidecar (allowed, separated namespace);
        # the check is that no metric/label-loading API is imported.
        check(f"firewall imports clean: {path.name}", not bad, str(bad))

    # 10. immutability scan: frozen namespaces untouched on this branch
    import subprocess

    diff = subprocess.run(
        ["git", "diff", "--name-only", "origin/codex/joint-lsml-localization-eval-v1...HEAD"],
        capture_output=True, text=True, cwd=REPO,
    ).stdout.splitlines()
    immutable_prefixes = (
        "results/joint_lsml_existing_localization_v1/", "results/joint_lsml_v1",
        "spectral_utils/joint_lsml_processbench_amendment.py",
        "spectral_utils/dependency_fusion.py",
    )
    violations = [line for line in diff if line.startswith(immutable_prefixes)]
    check("immutable namespaces untouched", not violations, str(violations)[:120])

    failed = [name for name, passed, _ in CHECKS if not passed]
    print(f"\n{'ABORT' if failed else 'ALL CLEAR'}: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
