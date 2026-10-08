"""
Load, freeze and enforce the prefix-detection claim registry.

Codex review addendum §8 (Q5): "create the versioned registry ... and freeze [it] ... Emit a
frozen JSON rendering plus hashes before opening comparison results." Also: machine-check
**all ten** preregistered nulls rather than a favourable subset, and map Codex's ten
structural checks onto the canonical list rather than silently creating an eleventh protocol.

What this module guarantees
--------------------------
1. The registry has a content hash. A results file cites that hash; if the registry is edited
   afterwards, the hashes disagree and the mismatch is visible. That is the whole audit trail —
   there is no way to quietly move a goalpost.
2. All ten canonical nulls are present, and every structural gate declares which null it
   protects (or is marked `structural_only`). `verify_registry` fails on a missing null, so a
   registry that dropped an inconvenient one cannot be frozen.
3. The structural gates are executable, not prose: `run_structural_gates` runs the ones that
   are checkable without labels, and refuses to report a pass it did not actually establish.

Nothing here computes a comparison result. Freezing happens first, on purpose.
"""
import hashlib
import json
import os

import numpy as np

REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs", "experiments", "PAPER_EXACT_CLAIM_REGISTRY_V1.yaml")

#: The canonical ten, from the phase-1 checkpoint §5.6. A registry missing any of these
#: cannot be frozen — that is the point of hard-coding the list here rather than trusting
#: whatever the YAML happens to contain.
CANONICAL_NULLS = ("N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10")

#: Codex §8's ten structural checks, by registry key.
REQUIRED_STRUCTURAL_GATES = (
    "suffix_invariance",
    "tokenwise_vs_chunked_replay",
    "endpoint_handling",
    "label_permutation",
    "feature_order_perturbation",
    "sign_orientation",
    "length_only_leakage",
    "split_isolation",
    "alarm_horizon_calibration",
    "grouped_resampling_validity",
)


def load_registry(path: str = None) -> dict:
    import yaml
    path = path or REGISTRY_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonical_json(registry: dict) -> str:
    """Deterministic JSON rendering — sorted keys, fixed separators.

    The hash must depend on the registry's *content*, not on YAML key order or whitespace,
    or a cosmetic reformat would look like a tampered registry and a reordered edit would
    look clean.
    """
    return json.dumps(registry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def registry_hash(registry: dict) -> str:
    return hashlib.sha256(canonical_json(registry).encode("utf-8")).hexdigest()


def verify_registry(registry: dict) -> list:
    """Return a list of problems; empty means the registry may be frozen."""
    problems = []
    if registry.get("lane") != "prefix-detection":
        problems.append(f"lane is {registry.get('lane')!r}, expected 'prefix-detection'")

    nulls = registry.get("nulls") or {}
    missing = [n for n in CANONICAL_NULLS if n not in nulls]
    if missing:
        problems.append(f"missing canonical nulls {missing} — all ten must be registered")
    extra = [n for n in nulls if n not in CANONICAL_NULLS]
    if extra:
        problems.append(f"unrecognised nulls {extra}; map new ones in structural_gates "
                        f"rather than extending the canonical list")
    for key, spec in nulls.items():
        if not (spec or {}).get("check"):
            problems.append(f"null {key} has no `check` naming its machine-checkable test")

    gates = registry.get("structural_gates") or {}
    missing_g = [g for g in REQUIRED_STRUCTURAL_GATES if g not in gates]
    if missing_g:
        problems.append(f"missing structural gates {missing_g} (Codex §8 names all ten)")
    for key, spec in gates.items():
        spec = spec or {}
        if not spec.get("assertion"):
            problems.append(f"structural gate {key} has no assertion")
        protects = spec.get("protects")
        if protects not in set(CANONICAL_NULLS) | {"structural_only"}:
            problems.append(f"structural gate {key} protects {protects!r}, which is neither "
                            f"a canonical null nor 'structural_only'")

    impl = registry.get("implementation") or {}
    if impl.get("primary_method") != "iu28_no_length":
        problems.append(f"primary_method is {impl.get('primary_method')!r}; Codex §8 fixes it "
                        f"to iu28_no_length (elapsed length is an ablation only)")
    if not impl.get("label_free_fit"):
        problems.append("implementation.label_free_fit must be true for the primary arm")

    b = (registry.get("budgets") or {}).get("primary_absolute_tokens")
    if b != [16, 32, 64, 128, 256, 512]:
        problems.append(f"primary budgets are {b}, expected [16,32,64,128,256,512]")

    alarm = registry.get("alarm") or {}
    if "max_t" not in str(alarm.get("calibrated_on", "")):
        problems.append("alarm must be calibrated on max_t score(t) over the full horizon")
    if alarm.get("fpr_definition") != "ever_alarmed_before_natural_completion":
        problems.append("trace-level FPR must be 'ever alarmed before natural completion'")

    if (registry.get("splits") or {}).get("unit") != "question_id":
        problems.append("splits.unit must be question_id — trace-level splits leak")
    if (registry.get("inference") or {}).get("bootstrap_unit") != "question_id":
        problems.append("inference.bootstrap_unit must be question_id")
    return problems


def freeze(registry: dict, out_dir: str) -> dict:
    """Write the frozen JSON rendering plus its hash. Refuses an invalid registry."""
    problems = verify_registry(registry)
    if problems:
        raise ValueError("registry cannot be frozen:\n  - " + "\n  - ".join(problems))
    os.makedirs(out_dir, exist_ok=True)
    h = registry_hash(registry)
    payload = {"registry_hash": h, "registry_version": registry.get("registry_version"),
               "registry": registry}
    path = os.path.join(out_dir, "CLAIM_REGISTRY_FROZEN.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    return {"path": path, "registry_hash": h}


# ── executable structural gates ────────────────────────────────────────────────
#
# Each returns (passed, reason). They take callables/data rather than importing the analysis
# so they can be exercised on synthetic input in the offline test suite, which is what makes
# them a gate rather than a comment.

def gate_suffix_invariance(score_prefix, channels_a, channels_b, budgets) -> tuple:
    """Identical prefixes, arbitrary different suffixes -> identical prefix scores."""
    worst = 0.0
    for t in budgets:
        d = abs(float(score_prefix(channels_a, t)) - float(score_prefix(channels_b, t)))
        worst = max(worst, d)
    return worst == 0.0, f"max |delta| over budgets {list(budgets)} = {worst:.3e}"


def gate_tokenwise_vs_chunked(score_prefix, channels, n: int, chunk: int = 32) -> tuple:
    tokenwise = [float(score_prefix(channels, t)) for t in range(1, n + 1)]
    chunked = []
    for start in range(0, n, chunk):
        for t in range(start + 1, min(start + chunk + 1, n + 1)):
            chunked.append(float(score_prefix(channels, t)))
    ok = np.array_equal(np.asarray(tokenwise), np.asarray(chunked))
    return ok, f"{n} decision points compared, exact match={ok}"


def gate_label_permutation(labels, scores, auroc_fn, seed: int = 0,
                           n_perm: int = 200, tol: float = 0.06) -> tuple:
    """Permuting labels must drive AUROC to chance."""
    rng = np.random.default_rng(seed)
    y = np.asarray(labels, dtype=float)
    perm = [auroc_fn(rng.permutation(y), scores) for _ in range(n_perm)]
    perm = np.asarray([p for p in perm if np.isfinite(p)], dtype=float)
    if perm.size == 0:
        return False, "no finite permuted AUROCs"
    m = float(perm.mean())
    return abs(m - 0.5) <= tol, f"mean permuted AUROC = {m:.4f} (tol {tol})"


def gate_feature_order(score_matrix_fn, matrix, names, seed: int = 0) -> tuple:
    """Permuting the input stream order must leave the score unchanged."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    base = np.asarray(score_matrix_fn(matrix, list(names)), dtype=float)
    perm = np.asarray(score_matrix_fn(matrix[:, order], [names[i] for i in order]),
                      dtype=float)
    ok = np.allclose(base, perm, atol=1e-9, rtol=0)
    d = float(np.max(np.abs(base - perm))) if base.shape == perm.shape else float("nan")
    return ok, f"max |delta| under stream permutation = {d:.3e}"


def gate_split_isolation(dev_groups, cal_groups, eval_groups) -> tuple:
    d, c, e = set(map(str, dev_groups)), set(map(str, cal_groups)), set(map(str, eval_groups))
    overlaps = {"dev∩cal": d & c, "dev∩eval": d & e, "cal∩eval": c & e}
    bad = {k: sorted(v)[:5] for k, v in overlaps.items() if v}
    return not bad, ("disjoint" if not bad else f"overlapping groups {bad}")


def gate_alarm_horizon(threshold: float, held_out_correct_paths, target_fpr: float,
                       n_sigma: float = 3.0) -> tuple:
    """The frozen threshold must realize its target ever-alarm FPR on held-out correct traces.

    The tolerance is **binomial, not a magic constant**: with n held-out correct traces the
    realized rate has standard error `sqrt(p(1-p)/n)`, which at the target 5% and n=200 is
    1.5 percentage points. A fixed +/-0.03 band is therefore about 2 sigma and rejects
    perfectly calibrated thresholds one time in twenty for no reason, while at large n the same
    band would wave through a genuinely miscalibrated one.

    This is the same point the registry makes about the real data: with ~60 MATH calibration
    questions a 5% target is coarse, and what can honestly be claimed is the interval, not the
    point. So the gate reports the band it actually used.
    """
    maxes = np.array([np.nanmax(p) for p in held_out_correct_paths if len(p)], dtype=float)
    maxes = maxes[np.isfinite(maxes)]
    if maxes.size == 0:
        return False, "no held-out correct traces"
    n = int(maxes.size)
    realized = float(np.mean(maxes >= threshold))
    se = float(np.sqrt(max(target_fpr * (1.0 - target_fpr), 1e-12) / n))
    tol = n_sigma * se
    ok = abs(realized - target_fpr) <= tol
    return ok, (f"realized ever-alarm FPR = {realized:.4f} vs target {target_fpr} "
                f"(n={n}, binomial SE {se:.4f}, {n_sigma:g}-sigma band +/-{tol:.4f}, "
                f"granularity {1.0 / n:.4f})")


def gate_grouped_resampling(groups, fn, grouped_ci_fn) -> tuple:
    """The question-grouped interval must be wider than the trace-level illusion."""
    g = grouped_ci_fn(groups, fn)
    flat = np.concatenate([np.asarray(v, dtype=float) for v in groups.values()])
    trace_se = float(flat.std(ddof=1) / np.sqrt(flat.size))
    grouped_se = (g["hi"] - g["lo"]) / (2 * 1.96)
    ok = grouped_se > trace_se
    return ok, f"grouped SE {grouped_se:.5f} vs trace-level {trace_se:.5f}"


def gate_length_only_leakage(lengths, labels, scores, auroc_fn) -> tuple:
    """Report the length-only baseline, and require the primary to beat it.

    A score that merely re-encodes observed prefix length is not detection. This is the
    weakest form of the check (unmatched); N3's residualized comparison is the strong form
    and needs the full analysis, so it stays a null rather than a structural gate.
    """
    a_len = auroc_fn(labels, np.asarray(lengths, dtype=float))
    a_pri = auroc_fn(labels, np.asarray(scores, dtype=float))
    if not (np.isfinite(a_len) and np.isfinite(a_pri)):
        return False, f"non-finite AUROC (length {a_len}, primary {a_pri})"
    return a_pri > a_len, f"primary AUROC {a_pri:.4f} vs length-only {a_len:.4f}"


#: Gates that can only be established with the real analysis in hand. Declared explicitly so
#: `run_structural_gates` reports them as NOT ESTABLISHED rather than silently omitting them —
#: an omitted check reads as a pass.
DEFERRED_GATES = {
    "endpoint_handling": "needs the declared already-finished / cap-pinned rule applied to a "
                         "real acquisition; assert during the lane run, not on synthetic data",
    "sign_orientation": "needs development resamples of the real fit to show orientation "
                        "stability; assert during the lane run",
}
