#!/usr/bin/env python3
"""Run the registered sparse-dependency fusion experiment from cached views.

The data machine should need one command only:

    python3 scripts/run_dependency_fusion_experiment.py \
        --data-dir local_cache --device auto

The script is transductive and label-free in the same sense as the project's
incumbent U-PCR/L-SML evaluations: every method is fitted using only the feature
matrix.  Correctness labels enter only after a score has been frozen, inside
``evaluate_score`` and the final reporting functions.

Outputs are checkpointed after every arm/seed so an interrupted DEEM sweep can
resume safely.  See ``SPEC_DEPENDENCY_FUSION_EXPERIMENT.md`` for the registered
hypotheses, paper provenance, primary contrasts, and interpretation rules.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, replace

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (REPO, os.path.join(REPO, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from inscope_cells import GROUP, INSCOPE                              # noqa: E402
from inscope_bench_common import assert_good6                         # noqa: E402
from spectral_utils.deem_adapter import DeemConfig, fit_deem_score    # noqa: E402
from spectral_utils.dependency_fusion import (                        # noqa: E402
    regularized_covariance_weights,
    sparse_upcr_fit,
)
from spectral_utils.fusion_utils import lsml_continuous               # noqa: E402
from spectral_utils.selector_bench import UnlabeledCell               # noqa: E402
from spectral_utils.streaming_utils import anchor_orient              # noqa: E402
from spectral_utils.subset_sweep import (                             # noqa: E402
    ALL_SIGNS, CANONICAL_POOL, iter_cells, prepare_cell,
)
from spectral_utils.upcr import upcr_fit                              # noqa: E402


EXPERIMENT_VERSION = "dependency-fusion-v1"
DEFAULT_OUT = os.path.join(REPO, "results", "dependency_fusion_study")

# Verbatim incumbent configuration from labelfree_standing_report.py.  It is
# used both to reproduce the deployed reference and to derive relative polarity.
INCUMBENT_FIT = dict(
    loss="l2",
    exclusion=True,
    difficulty_gate=False,
    simple_avg_fallback=True,
    recompute_after_exclusion=True,
    g2_projection_k=1,
    scale_ratio=0.25,
)

# IU-PCR comparison inside the fixed-input factorial.  K=2 is fixed because
# Eq. (15) of Tenzer et al. is explicitly a two-component PCR rule.
IU_PAPER_FIT = dict(
    loss="l2",
    exclusion=False,
    difficulty_gate=False,
    simple_avg_fallback=False,
    recompute_after_exclusion=False,
    g2_projection_k=1,
    scale_ratio=0.25,
    n_components=2,
    auto_components=False,
)

SPARSE_FIT = dict(
    scale_ratio=0.25,
    rank=2,
    n_components=2,
    g2_projection_components=1,
    threshold_multiplier=1.0,       # Appendix-B registered primary
    max_iter=100,
    inner_completion_iter=40,
    decomposition_tol=1e-8,
    max_sparse_fraction=None,       # never tune support with correctness labels
    target_condition=100.0,
)

DEEM_BASE = DeemConfig(
    epochs=100,
    batch_size=1024,
    learning_rate=0.001,
    momentum=0.9,
    cd_k=10,
    sampler_steps=5,
    use_weighted=True,
    init_method="mv_rand",
    device="auto",
    strict_version=True,
)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_hash(value):
    payload = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def source_hashes():
    """Hash the scientific implementation so resumed runs cannot mix code."""
    relpaths = (
        "spectral_utils/upcr.py",
        "spectral_utils/dependency_fusion.py",
        "spectral_utils/deem_adapter.py",
        "scripts/run_dependency_fusion_experiment.py",
    )
    out = {}
    for relpath in relpaths:
        with open(os.path.join(REPO, relpath), "rb") as handle:
            out[relpath] = hashlib.sha256(handle.read()).hexdigest()
    return out


def dataset_family(cell_key):
    """Dataset cluster used for a repeated-dataset sensitivity analysis."""
    known = (
        "triviaqa", "coqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
        "truthfulqa", "gsm8k", "math500", "gpqa", "webq", "humaneval",
    )
    for name in known:
        if name in cell_key:
            return name
    return cell_key


class JsonlStore:
    """Append-only, arm-level checkpoint store."""

    def __init__(self, path, config_hash):
        self.path = path
        self.config_hash = config_hash
        self.latest = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"corrupt checkpoint line {line_no}: {exc}")
                    if rec.get("config_hash") != config_hash:
                        continue
                    self.latest[self.key(rec)] = rec

    @staticmethod
    def key(rec):
        return (rec.get("cell"), rec.get("arena"), rec.get("arm"), rec.get("seed"))

    def successful(self, cell, arena, arm, seed=None):
        rec = self.latest.get((cell, arena, arm, seed))
        return rec is not None and rec.get("status") == "ok"

    def append(self, record):
        record = dict(record)
        record["config_hash"] = self.config_hash
        record["experiment_version"] = EXPERIMENT_VERSION
        record["written_at_unix"] = time.time()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(record), sort_keys=True, allow_nan=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.latest[self.key(record)] = record


def load_cells(data_dir):
    """Canonical loader with an explicit data directory for the second machine."""
    derived = os.path.join(data_dir, "derived_views.pkl")
    traces = os.path.join(data_dir, "trace_cells.pkl")
    loaded = {}
    for domain, cell_key, fd, labels in iter_cells(
        data_dir=data_dir,
        derived_views_pkl=derived,
        trace_cells_pkl=traces,
    ):
        if cell_key not in INSCOPE:
            continue
        cell_obj = prepare_cell(
            domain, cell_key, fd, labels, feature_pool=CANONICAL_POOL,
        )
        if cell_obj is None:
            continue
        rho = np.abs(np.corrcoef(cell_obj.V, rowvar=False))
        unlabeled = UnlabeledCell(
            domain=cell_obj.domain,
            cell_key=cell_obj.cell_key,
            pool=cell_obj.pool,
            pool_bits=cell_obj.pool_bits,
            V=cell_obj.V,
            anchor=cell_obj.anchor,
            anchor_name=cell_obj.anchor_name,
            rho=rho,
        )
        loaded[cell_key] = {
            "domain": domain,
            "V": np.asarray(cell_obj.V, dtype=float),
            "anchor": np.asarray(cell_obj.anchor, dtype=float),
            "pool": list(cell_obj.pool),
            "labels": np.asarray(cell_obj.labels, dtype=int),
            "unlabeled": unlabeled,
        }
    return loaded


def derive_oriented_matrix(cell):
    """Mirror the incumbent's two-pass sign(rho) orientation exactly."""
    V = np.asarray(cell["V"], dtype=float)
    hand = np.array([ALL_SIGNS.get(name, +1) for name in cell["pool"]], dtype=float)
    V_unoriented = V * hand
    probe = upcr_fit(V_unoriented.T, **INCUMBENT_FIT)
    polarity = np.sign(probe.rho_hat_full)
    polarity[polarity == 0] = 1.0
    return (V_unoriented * polarity).T, polarity, probe


def orient_score(score, anchor):
    score = np.asarray(score, dtype=float)
    if score.ndim != 1 or not np.isfinite(score).all() or np.std(score) < 1e-12:
        raise ValueError("method returned a non-finite or constant score")
    oriented, flipped = anchor_orient(score, np.asarray(anchor, dtype=float))
    return np.asarray(oriented, dtype=float), bool(flipped)


def evaluate_score(labels, frozen_oriented_score):
    """The only per-arm function that receives correctness labels."""
    return float(roc_auc_score(np.asarray(labels, dtype=int), frozen_oriented_score))


def sparse_diagnostics(fit, feature_names):
    dec = fit.decomposition
    iu = np.triu_indices(len(feature_names), 1)
    entries = []
    for i, j, value in zip(iu[0], iu[1], dec.sparse[iu]):
        if value != 0:
            entries.append({"i": int(i), "j": int(j),
                            "feature_i": feature_names[i],
                            "feature_j": feature_names[j],
                            "value": float(value)})
    entries.sort(key=lambda item: -abs(item["value"]))
    return {
        "g2_hat": fit.g2_hat,
        "var_y": fit.var_y,
        "projection_residual": fit.projection_residual,
        "rho_hat": fit.rho_hat,
        "pcr_eigenvalues": fit.pcr_eigenvalues,
        "decomposition": {
            "converged": dec.converged,
            "n_iter": dec.n_iter,
            "mu": dec.mu,
            "threshold": dec.threshold,
            "sparse_fraction": dec.sparse_fraction,
            "relative_residual": dec.relative_residual,
            "theorem_support_ok": dec.theorem_support_ok,
            **dec.meta,
            "top_sparse_edges": entries[:30],
        },
        "ridge": fit.ridge_diagnostics,
        **fit.meta,
    }


def load_dufs_choices():
    path = os.path.join(REPO, "results", "selector_bench", "a2_groupfs__c46.csv")
    if not os.path.exists(path):
        return {}
    choices = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == "a2.dufs_pf" and row.get("chosen"):
                choices[row["cell"]] = row["chosen"].split("|")
    return choices


def should_run(only_arms, arm):
    return not only_arms or arm in only_arms


def save_method_record(store, *, cell_key, cell, arena, arm, score_fn,
                       seed=None, diagnostics=None):
    if store.successful(cell_key, arena, arm, seed):
        return store.latest[(cell_key, arena, arm, seed)]
    started = time.time()
    try:
        raw_score, dynamic_diag = score_fn()
        oriented, flipped = orient_score(raw_score, cell["anchor"])
        auc = evaluate_score(cell["labels"], oriented)
        rec = {
            "cell": cell_key,
            "domain": GROUP.get(cell_key, cell.get("domain", "?")),
            "arena": arena,
            "arm": arm,
            "seed": seed,
            "status": "ok",
            "n": int(len(cell["labels"])),
            "m": int(len(cell["pool"])),
            "auc": auc,
            "score": oriented,
            "global_anchor_flip": flipped,
            "runtime_seconds": time.time() - started,
            "diagnostics": {**(diagnostics or {}), **(dynamic_diag or {})},
        }
    except Exception as exc:
        rec = {
            "cell": cell_key,
            "domain": GROUP.get(cell_key, cell.get("domain", "?")),
            "arena": arena,
            "arm": arm,
            "seed": seed,
            "status": "failed",
            "runtime_seconds": time.time() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"    FAILED {arena}/{arm}/seed={seed}: {exc}", flush=True)
    store.append(rec)
    return rec


def run_cell(cell_key, cell, store, *, arenas, seeds, device, skip_deem,
             only_arms, dufs_choices, verbose_deem=False):
    print(f"\n[{cell_key}] n={len(cell['labels'])} m={len(cell['pool'])}", flush=True)
    F, polarity, _ = derive_oriented_matrix(cell)

    # Deployed incumbent: exact maintained path, not the old fusion_utils entry point.
    deployed = upcr_fit(F, **INCUMBENT_FIT)
    if should_run(only_arms, "upcr_signrho"):
        save_method_record(
            store, cell_key=cell_key, cell=cell, arena="deployed",
            arm="upcr_signrho", seed=None,
            score_fn=lambda: (
                deployed.w @ F,
                {"keep": deployed.keep, "rho_hat": deployed.rho_hat_full,
                 "weights": deployed.w, "g2_hat": deployed.g2_hat,
                 "polarity": polarity, **deployed.meta},
            ),
        )

    # Deployed DUFS+L-SML is reconstructed from the stored label-free chosen set.
    if should_run(only_arms, "dufs_lsml") and cell_key in dufs_choices:
        chosen = dufs_choices[cell_key]
        cols = [cell["pool"].index(name) for name in chosen if name in cell["pool"]]
        if len(cols) >= 3:
            save_method_record(
                store, cell_key=cell_key, cell=cell, arena="deployed",
                arm="dufs_lsml", seed=None,
                score_fn=lambda cols=cols, chosen=chosen: (
                    lsml_continuous(*[cell["V"][:, j] for j in cols],
                                    compute_score_matrix=False)[0],
                    {"chosen": chosen, "n_chosen": len(cols)},
                ),
            )

    arena_indices = {
        "full": np.arange(F.shape[0]),
        "keep": np.where(deployed.keep)[0],
    }
    for arena in arenas:
        indices = arena_indices[arena]
        if len(indices) < 3:
            print(f"  {arena}: skipped, only {len(indices)} features", flush=True)
            continue
        F_a = F[indices]
        names = [cell["pool"][int(j)] for j in indices]
        print(f"  {arena}: {len(indices)} fixed input features", flush=True)

        iu = upcr_fit(F_a, **IU_PAPER_FIT)
        C_a = (F_a @ F_a.T) / F_a.shape[1]
        iu_ridge_w, iu_ridge_diag = regularized_covariance_weights(
            C_a, iu.rho_hat_full, target_condition=SPARSE_FIT["target_condition"],
        )
        sparse = sparse_upcr_fit(F_a, **SPARSE_FIT)
        common = {"feature_names": names, "polarity": polarity[indices]}

        deterministic = {
            "iu_pcr": (
                iu.w @ F_a,
                {**common, "rho_hat": iu.rho_hat_full, "weights": iu.w,
                 "g2_hat": iu.g2_hat, **iu.meta},
            ),
            "iu_ridge": (
                iu_ridge_w @ F_a,
                {**common, "rho_hat": iu.rho_hat_full, "weights": iu_ridge_w,
                 "g2_hat": iu.g2_hat, "ridge": iu_ridge_diag},
            ),
            "su_pcr_reproduction": (
                sparse.w_pcr @ F_a,
                {**common, "weights": sparse.w_pcr,
                 **sparse_diagnostics(sparse, names)},
            ),
            "sdsf": (
                sparse.w_structured @ F_a,
                {**common, "weights": sparse.w_structured,
                 **sparse_diagnostics(sparse, names)},
            ),
        }
        for arm, (frozen, diag) in deterministic.items():
            if not should_run(only_arms, arm):
                continue
            save_method_record(
                store, cell_key=cell_key, cell=cell, arena=arena, arm=arm,
                seed=None, score_fn=lambda frozen=frozen, diag=diag: (frozen, diag),
            )

        if skip_deem:
            continue
        deem_configs = {
            "deem_irbm_hard": replace(
                DEEM_BASE, input_mode="hard", use_preprocessing=False, device=device,
            ),
            "deem_deep_hard": replace(
                DEEM_BASE, input_mode="hard", use_preprocessing=True, device=device,
            ),
            "deem_deep_soft": replace(
                DEEM_BASE, input_mode="soft", use_preprocessing=True, device=device,
            ),
        }
        for arm, deem_config in deem_configs.items():
            if not should_run(only_arms, arm):
                continue
            for seed in seeds:
                print(f"    {arm} seed={seed}", flush=True)

                def _fit(config=deem_config, this_seed=seed):
                    result = fit_deem_score(
                        F_a.T, seed=this_seed, config=config, verbose=verbose_deem,
                    )
                    return result.score, {
                        "feature_names": names,
                        "class_map": result.class_map,
                        "package_version": result.package_version,
                        "config": result.config,
                        "history": result.history,
                    }

                save_method_record(
                    store, cell_key=cell_key, cell=cell, arena=arena, arm=arm,
                    seed=seed, score_fn=_fit,
                )


def bootstrap_mean_ci(delta, label, n_boot=20000):
    delta = np.asarray(delta, dtype=float)
    if len(delta) == 0:
        return float("nan"), float("nan")
    seed = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot), dtype=float)
    chunk = 1000
    for start in range(0, int(n_boot), chunk):
        size = min(chunk, int(n_boot) - start)
        idx = rng.integers(0, len(delta), size=(size, len(delta)))
        means[start:start + size] = delta[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def paired_stats(reference, candidate, label):
    cells = sorted(set(reference) & set(candidate))
    d = np.array([candidate[c] - reference[c] for c in cells], dtype=float)
    p = float("nan")
    if len(d) >= 5 and np.any(d != 0):
        try:
            p = float(wilcoxon(d).pvalue)
        except ValueError:
            pass
    lo, hi = bootstrap_mean_ci(d, label)
    row = {
        "n": len(d),
        "mean_delta": float(np.mean(d)) if len(d) else float("nan"),
        "median_delta": float(np.median(d)) if len(d) else float("nan"),
        "ci95_low": lo,
        "ci95_high": hi,
        "wins": int(np.sum(d > 0)),
        "losses": int(np.sum(d < 0)),
        "ties": int(np.sum(d == 0)),
        "p_wilcoxon": p,
        "cells": cells,
        "deltas": d.tolist(),
    }
    for group_name in ("QA", "math"):
        group_delta = np.array(
            [candidate[c] - reference[c] for c in cells
             if GROUP.get(c) == group_name], dtype=float,
        )
        row[f"mean_delta_{group_name.lower()}"] = (
            float(group_delta.mean()) if len(group_delta) else float("nan")
        )

    # Several cells share a source dataset (notably GSM8K, MATH500, and
    # TriviaQA).  The primary estimand remains the registered cell macro, but
    # this equal-dataset sensitivity prevents repeated datasets from masquerading
    # as independent replications.
    family_values = {}
    for cell, value in zip(cells, d):
        family_values.setdefault(dataset_family(cell), []).append(float(value))
    family_delta = np.array(
        [np.mean(values) for _, values in sorted(family_values.items())], dtype=float,
    )
    family_lo, family_hi = bootstrap_mean_ci(family_delta, label + "_dataset")
    row.update({
        "n_dataset_families": len(family_delta),
        "dataset_macro_delta": (float(family_delta.mean()) if len(family_delta)
                                else float("nan")),
        "dataset_ci95_low": family_lo,
        "dataset_ci95_high": family_hi,
        "dataset_wins": int(np.sum(family_delta > 0)),
        "dataset_losses": int(np.sum(family_delta < 0)),
    })
    return row


def holm_fixed_family(rows, primary_names, family_size):
    """Holm correction without shrinking the registered family on failures."""
    finite = [(name, rows[name]["p_wilcoxon"]) for name in primary_names
              if name in rows and np.isfinite(rows[name]["p_wilcoxon"])]
    finite.sort(key=lambda item: item[1])
    running = 0.0
    for rank, (name, p) in enumerate(finite, 1):
        adjusted = min(1.0, (family_size - rank + 1) * p)
        running = max(running, adjusted)
        rows[name]["holm_adjusted_p"] = running
    for name in primary_names:
        if name in rows and "holm_adjusted_p" not in rows[name]:
            rows[name]["holm_adjusted_p"] = float("nan")


def collect_scores(cells, store, seeds):
    """Return per-cell AUCs, including fixed-seed and seed-ensemble DEEM arms."""
    by_arm = {}
    seed_rows = []
    for rec in store.latest.values():
        if rec.get("status") != "ok":
            continue
        key = f"{rec['arena']}.{rec['arm']}"
        if rec.get("seed") is None:
            by_arm.setdefault(key, {})[rec["cell"]] = float(rec["auc"])
        else:
            seed_rows.append({
                "cell": rec["cell"], "domain": rec.get("domain"),
                "arena": rec["arena"], "arm": rec["arm"],
                "seed": rec["seed"], "auc": rec["auc"],
                "runtime_seconds": rec.get("runtime_seconds"),
            })
            if int(rec["seed"]) == int(seeds[0]):
                by_arm.setdefault(key + "_seed0", {})[rec["cell"]] = float(rec["auc"])

    for arena in ("full", "keep"):
        for arm in ("deem_irbm_hard", "deem_deep_hard", "deem_deep_soft"):
            for cell_key, cell in cells.items():
                records = [store.latest.get((cell_key, arena, arm, int(seed)))
                           for seed in seeds]
                if not records or any(r is None or r.get("status") != "ok" for r in records):
                    continue
                score = np.mean([np.asarray(r["score"], dtype=float) for r in records], axis=0)
                auc = evaluate_score(cell["labels"], score)
                by_arm.setdefault(f"{arena}.{arm}_ensemble", {})[cell_key] = auc
    return by_arm, seed_rows


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_reports(cells, store, seeds, out_dir, config_hash):
    by_arm, seed_rows = collect_scores(cells, store, seeds)
    successful_records = [rec for rec in store.latest.values()
                          if rec.get("status") == "ok"]
    failures = [rec for rec in store.latest.values()
                if rec.get("status") == "failed"]
    arm_summary = []
    per_cell = []
    for arm, scores in sorted(by_arm.items()):
        values = np.array(list(scores.values()), dtype=float)
        qa = [v for c, v in scores.items() if GROUP.get(c) == "QA"]
        math = [v for c, v in scores.items() if GROUP.get(c) == "math"]
        arena, method = arm.split(".", 1)
        source_method = method.removesuffix("_ensemble").removesuffix("_seed0")
        matching_ok = [rec for rec in successful_records
                       if rec.get("arena") == arena and rec.get("arm") == source_method]
        matching_failed = [rec for rec in failures
                           if rec.get("arena") == arena and rec.get("arm") == source_method]
        runtimes = [float(rec["runtime_seconds"]) for rec in matching_ok
                    if rec.get("runtime_seconds") is not None]
        arm_summary.append({
            "arm": arm,
            "n_cells": len(values),
            "macro_auc": float(np.mean(values)) if len(values) else float("nan"),
            "macro_qa": float(np.mean(qa)) if qa else float("nan"),
            "macro_math": float(np.mean(math)) if math else float("nan"),
            "median_fit_seconds": (float(np.median(runtimes)) if runtimes
                                   else float("nan")),
            "n_failed_records": len(matching_failed),
        })
        for cell_key, auc in scores.items():
            per_cell.append({"cell": cell_key, "domain": GROUP.get(cell_key),
                             "arm": arm, "auc": auc})

    contrast_specs = [
        ("H1_sparse_reliability", "full.iu_pcr", "full.su_pcr_reproduction", True),
        ("H2_dependency_weights", "full.su_pcr_reproduction", "full.sdsf", True),
        ("A1_ridge_without_sparse", "full.iu_pcr", "full.iu_ridge", False),
        ("H3_deem_soft", "full.iu_pcr", "full.deem_deep_soft_ensemble", True),
        ("A2_deem_hard_to_soft", "full.deem_deep_hard_ensemble",
         "full.deem_deep_soft_ensemble", False),
        ("A3_deem_depth_hard", "full.deem_irbm_hard_ensemble",
         "full.deem_deep_hard_ensemble", False),
        ("P1_sdsf_vs_deployed", "deployed.upcr_signrho", "full.sdsf", False),
        ("P2_deem_vs_deployed", "deployed.upcr_signrho",
         "full.deem_deep_soft_ensemble", False),
        ("P3_sdsf_vs_dufs", "deployed.dufs_lsml", "full.sdsf", False),
        ("K1_keep_sparse_reliability", "keep.iu_pcr",
         "keep.su_pcr_reproduction", False),
        ("K2_keep_dependency_weights", "keep.su_pcr_reproduction", "keep.sdsf", False),
    ]
    contrasts = {}
    primary_names = []
    for name, ref, cand, primary in contrast_specs:
        if ref not in by_arm or cand not in by_arm:
            continue
        row = paired_stats(by_arm[ref], by_arm[cand], name)
        row.update({"contrast": name, "reference": ref, "candidate": cand,
                    "primary": primary})
        contrasts[name] = row
        if primary:
            primary_names.append(name)
    # Registered family is three even if DEEM fails or an arm is missing.
    holm_fixed_family(contrasts, ["H1_sparse_reliability", "H2_dependency_weights",
                                  "H3_deem_soft"], family_size=3)

    # Factorial interaction: extra benefit of ridge weights after sparse rho.
    required = ("full.iu_pcr", "full.iu_ridge", "full.su_pcr_reproduction", "full.sdsf")
    if all(key in by_arm for key in required):
        shared = sorted(set.intersection(*(set(by_arm[key]) for key in required)))
        interaction = np.array([
            (by_arm["full.sdsf"][c] - by_arm["full.su_pcr_reproduction"][c])
            - (by_arm["full.iu_ridge"][c] - by_arm["full.iu_pcr"][c])
            for c in shared
        ])
        lo, hi = bootstrap_mean_ci(interaction, "factorial_interaction")
        p = float(wilcoxon(interaction).pvalue) if len(interaction) >= 5 and np.any(interaction) else float("nan")
        contrasts["A4_factorial_interaction"] = {
            "contrast": "A4_factorial_interaction",
            "reference": "(iu_ridge-iu_pcr)",
            "candidate": "(sdsf-su_pcr)",
            "primary": False,
            "n": len(interaction),
            "mean_delta": float(interaction.mean()) if len(interaction) else float("nan"),
            "median_delta": float(np.median(interaction)) if len(interaction) else float("nan"),
            "ci95_low": lo, "ci95_high": hi,
            "wins": int(np.sum(interaction > 0)),
            "losses": int(np.sum(interaction < 0)),
            "ties": int(np.sum(interaction == 0)),
            "p_wilcoxon": p,
            "holm_adjusted_p": float("nan"),
            "cells": shared, "deltas": interaction.tolist(),
        }

        # Add the same domain and repeated-dataset diagnostics as ordinary
        # paired contrasts by treating the two factorial effects as score maps.
        interaction_map = {c: float(v) for c, v in zip(shared, interaction)}
        interaction_diag = paired_stats(
            {c: 0.0 for c in shared}, interaction_map, "factorial_interaction_diag",
        )
        for key in ("mean_delta_qa", "mean_delta_math", "n_dataset_families",
                    "dataset_macro_delta", "dataset_ci95_low", "dataset_ci95_high",
                    "dataset_wins", "dataset_losses"):
            contrasts["A4_factorial_interaction"][key] = interaction_diag[key]

    seed_summary = []
    seed_groups = {}
    for row in seed_rows:
        key = (row["cell"], row["domain"], row["arena"], row["arm"])
        seed_groups.setdefault(key, []).append(float(row["auc"]))
    for (cell_key, domain, arena, arm), values in sorted(seed_groups.items()):
        seed_summary.append({
            "cell": cell_key, "domain": domain, "arena": arena, "arm": arm,
            "n_seeds": len(values), "mean_auc": float(np.mean(values)),
            "std_auc": float(np.std(values)), "min_auc": float(np.min(values)),
            "max_auc": float(np.max(values)),
            "range_auc": float(np.max(values) - np.min(values)),
        })

    sparse_rows = []
    for rec in successful_records:
        if rec.get("arm") != "su_pcr_reproduction":
            continue
        diag = rec.get("diagnostics", {}).get("decomposition", {})
        sparse_rows.append({
            "cell": rec["cell"], "domain": rec.get("domain"),
            "arena": rec["arena"], "converged": diag.get("converged"),
            "n_iter": diag.get("n_iter"), "nnz_pairs": diag.get("nnz_pairs"),
            "n_pairs": diag.get("n_pairs"),
            "sparse_fraction": diag.get("sparse_fraction"),
            "relative_residual": diag.get("relative_residual"),
            "theorem_support_ok": diag.get("theorem_support_ok"),
            "threshold": diag.get("threshold"), "mu": diag.get("mu"),
        })

    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, "per_cell.csv"), per_cell)
    write_csv(os.path.join(out_dir, "arm_summary.csv"), arm_summary)
    write_csv(os.path.join(out_dir, "deem_seeds.csv"), seed_rows)
    write_csv(os.path.join(out_dir, "deem_seed_summary.csv"), seed_summary)
    write_csv(os.path.join(out_dir, "sparse_diagnostics.csv"), sparse_rows)
    contrast_rows = list(contrasts.values())
    write_csv(
        os.path.join(out_dir, "contrasts.csv"),
        contrast_rows,
        fieldnames=["contrast", "reference", "candidate", "primary", "n",
                    "mean_delta", "median_delta", "ci95_low", "ci95_high",
                    "mean_delta_qa", "mean_delta_math",
                    "n_dataset_families", "dataset_macro_delta",
                    "dataset_ci95_low", "dataset_ci95_high",
                    "dataset_wins", "dataset_losses",
                    "wins", "losses", "ties", "p_wilcoxon", "holm_adjusted_p"],
    )
    summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "config_hash": config_hash,
        "n_loaded_cells": len(cells),
        "arms": arm_summary,
        "contrasts": contrasts,
        "deem_seed_summary": seed_summary,
        "sparse_diagnostics": sparse_rows,
        "failures": failures,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, indent=2, sort_keys=True, allow_nan=True)

    lines = [
        "# Dependency-aware fusion experiment — results",
        "",
        f"Configuration hash: `{config_hash}`. Loaded cells: **{len(cells)}**. ",
        f"Failed arm/seed records: **{len(failures)}**.",
        "",
        "Labels were used only to compute the metrics below, after every method score was frozen.",
        "",
        "## Arm summary",
        "",
        "| arm | n | macro AUROC | QA | math | median fit s | failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(arm_summary, key=lambda r: -r["macro_auc"]):
        lines.append(f"| `{row['arm']}` | {row['n_cells']} | {row['macro_auc']:.4f} | "
                     f"{row['macro_qa']:.4f} | {row['macro_math']:.4f} | "
                     f"{row['median_fit_seconds']:.3f} | {row['n_failed_records']} |")
    lines.extend([
        "", "## Paired contrasts", "",
        "Positive means the candidate is better. Deltas are AUROC percentage points.",
        "",
        "| contrast | reference → candidate | mean [95% CI] | QA/math | dataset macro [CI] | W/L | p | Holm |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in contrast_rows:
        holm = row.get("holm_adjusted_p", float("nan"))
        lines.append(
            f"| `{row['contrast']}` | `{row['reference']}` → `{row['candidate']}` | "
            f"{100*row['mean_delta']:+.2f} [{100*row['ci95_low']:+.2f}, "
            f"{100*row['ci95_high']:+.2f}] | "
            f"{100*row.get('mean_delta_qa', float('nan')):+.2f}/"
            f"{100*row.get('mean_delta_math', float('nan')):+.2f} | "
            f"{100*row.get('dataset_macro_delta', float('nan')):+.2f} "
            f"[{100*row.get('dataset_ci95_low', float('nan')):+.2f}, "
            f"{100*row.get('dataset_ci95_high', float('nan')):+.2f}] | "
            f"{row['wins']}/{row['losses']} | {row['p_wilcoxon']:.4g} | "
            f"{holm:.4g} |"
        )
    if seed_summary:
        lines.extend([
            "", "## DEEM seed stability", "",
            "AUROC standard deviation and range are computed across the registered seeds per cell.",
            "", "| arena.arm | complete cells | mean cell std | max cell range |",
            "|---|---:|---:|---:|",
        ])
        for key in sorted({(r["arena"], r["arm"]) for r in seed_summary}):
            rows = [r for r in seed_summary if (r["arena"], r["arm"]) == key
                    and r["n_seeds"] == len(seeds)]
            if rows:
                lines.append(
                    f"| `{key[0]}.{key[1]}` | {len(rows)} | "
                    f"{100*np.mean([r['std_auc'] for r in rows]):.2f}pp | "
                    f"{100*np.max([r['range_auc'] for r in rows]):.2f}pp |"
                )
    if sparse_rows:
        lines.extend([
            "", "## Sparse decomposition diagnostics", "",
            "| arena | cells | converged | theorem support | median sparse fraction | median residual |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for arena in sorted({row["arena"] for row in sparse_rows}):
            rows = [row for row in sparse_rows if row["arena"] == arena]
            lines.append(
                f"| `{arena}` | {len(rows)} | "
                f"{sum(bool(r['converged']) for r in rows)}/{len(rows)} | "
                f"{sum(bool(r['theorem_support_ok']) for r in rows)}/{len(rows)} | "
                f"{np.median([r['sparse_fraction'] for r in rows]):.4f} | "
                f"{np.median([r['relative_residual'] for r in rows]):.4f} |"
            )
    lines.extend([
        "", "## Registered interpretation gate", "",
        "SDSF advances as a contribution only if `H2_dependency_weights` has mean "
        "gain at least +1.0pp, a positive cell-bootstrap lower bound, Holm-adjusted "
        "p<0.05, QA and math deltas each at least -0.5pp, a positive equal-dataset "
        "macro, and at least 90% primary-arena decomposition convergence. `H1` "
        "alone is evidence for the published sparse-error correction, not for our "
        "new weighting method.",
        "",
    ])
    if failures:
        lines.extend(["## Failures", ""])
        for rec in failures:
            lines.append(f"- `{rec.get('cell')}/{rec.get('arena')}/{rec.get('arm')}` "
                         f"seed={rec.get('seed')}: {rec.get('error_type')}: {rec.get('error')}")
        lines.append("")
    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(REPO, "local_cache"))
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--deem-seeds", default="0,1,2,3,4",
                        help="comma-separated; probabilities are averaged without labels")
    parser.add_argument("--arenas", default="full,keep", choices=("full", "keep", "full,keep"))
    parser.add_argument("--skip-deem", action="store_true")
    parser.add_argument("--only-arms", default="",
                        help="comma-separated exact arm names; empty runs all")
    parser.add_argument("--only-cells", default="",
                        help="comma-separated cell keys for an operational dry run")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--verbose-deem", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(item) for item in args.deem_seeds.split(",") if item.strip()]
    if not seeds:
        raise SystemExit("--deem-seeds must contain at least one integer")
    arenas = args.arenas.split(",")
    only_arms = {item for item in args.only_arms.split(",") if item}

    registered_config = {
        "experiment_version": EXPERIMENT_VERSION,
        "incumbent_fit": INCUMBENT_FIT,
        "iu_paper_fit": IU_PAPER_FIT,
        "sparse_fit": SPARSE_FIT,
        "deem_base": asdict(replace(DEEM_BASE, device=args.device)),
        "deem_seeds": seeds,
        "source_sha256": source_hashes(),
    }
    config_hash = stable_hash(registered_config)
    os.makedirs(args.out_dir, exist_ok=True)
    config_path = os.path.join(args.out_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as handle:
            old = json.load(handle)
        if old.get("config_hash") != config_hash:
            raise SystemExit(
                "The output directory contains a different registered configuration. "
                "Choose a new --out-dir; refusing to mix incomparable checkpoints."
            )
    else:
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump({"config_hash": config_hash, **jsonable(registered_config)},
                      handle, indent=2, sort_keys=True)

    print(f"Experiment {EXPERIMENT_VERSION} config={config_hash}")
    print(f"Loading canonical cells from {args.data_dir}", flush=True)
    cells = load_cells(os.path.abspath(args.data_dir))
    if not cells:
        raise SystemExit(
            "No in-scope cells loaded. Copy the complete local_cache directory, "
            "including derived_views.pkl and trace_cells.pkl, to the data machine."
        )
    validity_check = {key: {"V": value["V"], "anchor": value["anchor"],
                            "pool": value["pool"], "labels": value["labels"]}
                      for key, value in cells.items()}
    valid, macro = assert_good6(validity_check, verbose=True)
    if not valid:
        raise SystemExit(
            f"Canonical validity gate failed: GOOD_6={macro:.4f}. Refusing to "
            "run or report the dependency experiment on a mismatched cache."
        )

    requested = [key for key in INSCOPE if key in cells]
    if args.only_cells:
        wanted = set(args.only_cells.split(","))
        requested = [key for key in requested if key in wanted]
    if args.max_cells is not None:
        requested = requested[:max(0, args.max_cells)]
    print(f"Running {len(requested)}/{len(cells)} loaded cells", flush=True)

    store = JsonlStore(os.path.join(args.out_dir, "records.jsonl"), config_hash)
    dufs_choices = load_dufs_choices()
    for cell_key in requested:
        run_cell(
            cell_key, cells[cell_key], store,
            arenas=arenas, seeds=seeds, device=args.device,
            skip_deem=args.skip_deem, only_arms=only_arms,
            dufs_choices=dufs_choices, verbose_deem=args.verbose_deem,
        )

    summary = build_reports(cells, store, seeds, args.out_dir, config_hash)
    print(f"\nWrote {os.path.join(args.out_dir, 'REPORT.md')}")
    print(f"Failures: {len(summary['failures'])}")


if __name__ == "__main__":
    main()
