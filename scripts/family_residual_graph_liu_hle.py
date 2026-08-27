#!/usr/bin/env python3
"""Frozen external-to-development HLE audit for Family-residual graph LIU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.family_residual_graph_liu_fit import (  # noqa: E402
    DEFAULT_OUT as DEVELOPMENT_OUT,
    VERSION,
    canonical_hash,
    sha256_file,
    write_json,
)
from scripts.family_residual_graph_liu_prmbench import (  # noqa: E402
    candidate_score,
    verify_development,
)
from scripts.family_residual_graph_liu_report import DEFAULT_KEY  # noqa: E402
from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DUFS_EPOCHS,
    DUFS_SEEDS,
)
from scripts.leverage_balanced_processbench_transfer import mixed_v2_matrix  # noqa: E402
from scripts.neutral_residual_mode_hle_confirmation import (  # noqa: E402
    DEFAULT_JUDGE_MANIFEST,
    DEFAULT_LABELS,
    DEFAULT_RAW,
    read_jsonl,
    telemetry_payload,
)
from spectral_utils.contribution_subspace import (  # noqa: E402
    cardinality_balanced_contribution_score,
)
from spectral_utils.family_residual_graph import fit_family_residual_state  # noqa: E402
from spectral_utils.laplacian_upcr import dufs_soft_gates  # noqa: E402


TRANSFER_VERSION = "family-residual-graph-liu-hle-v3-2026-08-23"
DEFAULT_OUT = REPO / "results" / "family_residual_graph_liu_hle_v3"
DEFAULT_NRM = (
    REPO / "results" / "neutral_residual_mode_hle_v1" / "FROZEN_SCORES.npz"
)
BOOTSTRAPS = 10000


def score_phase(args):
    args.out.mkdir(parents=True, exist_ok=False)
    selection, selection_path = verify_development(args.development)
    config_path = args.development / "CONFIG_INDEX.json"
    configs = json.loads(config_path.read_text())
    with args.raw.open("rb") as handle:
        cache = pickle.load(handle)
    row_keys = sorted(cache, key=lambda value: int(value))
    telemetry = []
    for key in row_keys:
        candidates = cache[key].get("candidates")
        if not candidates:
            raise RuntimeError(f"missing HLE candidate for {key}")
        telemetry.append(telemetry_payload(candidates[0]))
    F, names, availability, contract = mixed_v2_matrix(telemetry)
    state = fit_family_residual_state(F, names)
    gates, _ = dufs_soft_gates(F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS)
    finalist = candidate_score(
        F, names, state, gates, selection["selected_config"]
    )
    fixed_default = candidate_score(F, names, state, gates, configs[DEFAULT_KEY])
    cardinality = cardinality_balanced_contribution_score(
        state.contribution_space, state.baseline_fit.w
    ).score
    score_path = args.out / "FROZEN_SCORES.npz"
    np.savez_compressed(
        score_path,
        row_keys=np.asarray([int(value) for value in row_keys]),
        iu=state.baseline,
        finalist=finalist,
        fixed_default=fixed_default,
        cardinality=cardinality,
    )
    manifest = {
        "version": TRANSFER_VERSION,
        "development_version": VERSION,
        "phase": "telemetry_only_frozen_transfer_fit",
        "selection_hash": selection["selection_hash"],
        "selected_key": selection["selected_key"],
        "n": len(row_keys),
        "feature_names": list(names),
        "availability": availability,
        "contract": contract,
        "labels_used": False,
        "target_fields_received_by_fusion": [],
        "hashes": {
            "raw": sha256_file(args.raw),
            "selection": sha256_file(selection_path),
            "config_index": sha256_file(config_path),
            "scores": sha256_file(score_path),
            "nrm_comparator": sha256_file(args.nrm),
            "transfer_script": sha256_file(Path(__file__)),
        },
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(args.out / "FIT_MANIFEST.json", manifest)
    print(json.dumps({
        "phase": manifest["phase"], "n": manifest["n"],
        "labels_used": False, "manifest_hash": manifest["manifest_hash"],
    }, indent=2))


def bootstrap(y, scores):
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(20260823)
    names = [name for name in scores if name != "iu"]
    draws = {name: np.empty(BOOTSTRAPS) for name in names}
    for draw in range(BOOTSTRAPS):
        index = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        base = roc_auc_score(y[index], scores["iu"][index])
        for name in names:
            draws[name][draw] = (
                roc_auc_score(y[index], scores[name][index]) - base
            )
    intervals = {
        name: {
            "low_pp": 100 * float(np.quantile(values, .025)),
            "high_pp": 100 * float(np.quantile(values, .975)),
            "probability_positive": float(np.mean(values > 0)),
        }
        for name, values in draws.items()
    }
    return intervals, draws


def report_phase(args):
    manifest = json.loads((args.out / "FIT_MANIFEST.json").read_text())
    payload = dict(manifest)
    recorded_manifest_hash = payload.pop("manifest_hash")
    if canonical_hash(payload) != recorded_manifest_hash:
        raise RuntimeError("HLE transfer manifest is not self-consistent")
    selection, selection_path = verify_development(args.development)
    score_path = args.out / "FROZEN_SCORES.npz"
    current_hashes = {
        "raw": sha256_file(args.raw),
        "selection": sha256_file(selection_path),
        "config_index": sha256_file(args.development / "CONFIG_INDEX.json"),
        "scores": sha256_file(score_path),
        "nrm_comparator": sha256_file(args.nrm),
        "transfer_script": sha256_file(Path(__file__)),
    }
    if current_hashes != manifest["hashes"]:
        raise RuntimeError("HLE transfer input/source hash changed")
    if selection["selection_hash"] != manifest["selection_hash"]:
        raise RuntimeError("HLE development selection changed")
    with np.load(score_path) as stored:
        row_keys = stored["row_keys"].astype(int)
        scores = {
            name: stored[name].astype(float)
            for name in ("iu", "finalist", "fixed_default", "cardinality")
        }
    with np.load(args.nrm) as stored:
        if not np.array_equal(stored["row_keys"].astype(int), row_keys):
            raise RuntimeError("HLE NRM/graph-LIU row mismatch")
        scores["family_nrm"] = stored["nrm_correctness_score"].astype(float)
    labels = read_jsonl(args.labels)
    judge_manifest = json.loads(args.judge_manifest.read_text())
    if judge_manifest["hashes"]["output_judgments_sha256"] != sha256_file(
        args.labels
    ):
        raise RuntimeError("HLE judge manifest does not authenticate labels")
    if [int(row["row_key"]) for row in labels] != row_keys.tolist():
        raise RuntimeError("HLE labels do not align")
    y = np.asarray([row["correct"] == "yes" for row in labels], dtype=int)
    metrics = {
        name: {
            "auroc": float(roc_auc_score(y, value)),
            "auprc": float(average_precision_score(y, value)),
        }
        for name, value in scores.items()
    }
    intervals, draws = bootstrap(y, scores)
    delta = metrics["finalist"]["auroc"] - metrics["iu"]["auroc"]
    nrm_delta = metrics["family_nrm"]["auroc"] - metrics["iu"]["auroc"]
    d50 = draws["finalist"] - .5 * draws["family_nrm"]
    result = {
        "version": TRANSFER_VERSION,
        "status": "PASS" if intervals["finalist"]["low_pp"] > 0 else "FAIL",
        "scope": "post-audit retrospective bug-repair sensitivity under interim judge",
        "n": len(y), "n_correct": int(y.sum()),
        "metrics": metrics,
        "delta_vs_iu_pp": 100 * delta,
        "family_nrm_delta_pp": 100 * nrm_delta,
        "nrm_recovery_fraction": delta / nrm_delta,
        "bootstrap": intervals,
        "d50_pp": 100 * float(np.mean(d50)),
        "d50_ci_pp": [
            100 * float(np.quantile(d50, .025)),
            100 * float(np.quantile(d50, .975)),
        ],
        "label_hashes": {
            "labels": sha256_file(args.labels),
            "judge_manifest": sha256_file(args.judge_manifest),
        },
    }
    write_json(args.out / "RESULT.json", result)
    interval = intervals["finalist"]
    lines = [
        "# Family-residual graph LIU v3 bug-repair sensitivity — HLE", "",
        f"**{result['status']}**: finalist vs IU {result['delta_vs_iu_pp']:+.3f}pp "
        f"(stratified 95% CI [{interval['low_pp']:+.3f}, "
        f"{interval['high_pp']:+.3f}]pp).", "",
        f"Family-NRM: {result['family_nrm_delta_pp']:+.3f}pp; recovery "
        f"{100 * result['nrm_recovery_fraction']:.1f}%; "
        f"`D_0.5`={result['d50_pp']:+.3f}pp.", "",
        "| method | AUROC | AUPRC | delta vs IU |", "|---|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        change = 100 * (row["auroc"] - metrics["iu"]["auroc"])
        lines.append(
            f"| `{name}` | {row['auroc']:.6f} | {row['auprc']:.6f} "
            f"| {change:+.3f}pp |"
        )
    lines += ["", "HLE contains only 68 judged-correct answers, uses the "
              "interim Codex judge, and its outcome was known before v3. This "
              "is a retrospective bug-repair sensitivity only.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("score", "report", "both"))
    parser.add_argument("--development", type=Path, default=DEVELOPMENT_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--judge-manifest", type=Path, default=DEFAULT_JUDGE_MANIFEST)
    parser.add_argument("--nrm", type=Path, default=DEFAULT_NRM)
    args = parser.parse_args()
    if args.phase in {"score", "both"}:
        score_phase(args)
    if args.phase in {"report", "both"}:
        report_phase(args)


if __name__ == "__main__":
    main()
