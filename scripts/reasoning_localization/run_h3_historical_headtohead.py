#!/usr/bin/env python3
"""Matched historical-regime H0/H2/H3 head-to-head.

Stage ``freeze`` imports only identity and already frozen candidate scores.
Stage ``evaluate`` verifies the freeze, uses historical calibration labels for
the H0 detector threshold, and only then opens the exact historical audit
records.  No score-side parameter is fit in this executable.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_global_local_online_architecture_v2 import _best_threshold, _processbench  # noqa: E402
from scripts.run_local_online_comprehensive_stage1 import SEED, _stage_partition  # noqa: E402
from spectral_utils.reconstruction_benchmark.external_final_answer import (  # noqa: E402
    apply_external_id_contract,
    load_external_registry,
    load_identity_key,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    load_npz_no_pickle,
    sha256_file,
)


EXPERIMENT = "P4_H3_HISTORICAL_HEADTOHEAD"
MODELS = ("qwen3_8b", "llama31_8b")
FAMILIES = ("gsm8k", "math", "olympiadbench", "omnimath")
CELLS = tuple((model, family) for model in MODELS for family in FAMILIES)
BOOTSTRAP_DRAWS = 2_000
FINALIST = "HISTORICAL_FINALIST"
ENTROPY = "HISTORICAL_ENTROPY_TOP5"
H0 = "H0_CURRENT_FAMILY_TOP10"
H2 = "H2_CLEAN_C7"
H3 = "H3_EQUAL_C8"
END_TO_END = (ENTROPY, FINALIST, H0, H2, H3)
METRICS = ("f1", "exact_error", "clean_abstention", "within_one")

PROGRAM = REPO / "results/reasoning_localization_03662_v1"
ROOT = PROGRAM / "phase_4/h3_historical_headtohead_v1"
REGISTRY = PROGRAM / "phase_4/H3_HISTORICAL_HEADTOHEAD_EXECUTION_REGISTRY.json"
S0_ROWS = PROGRAM / "phase_0/p0_s0_historical_replay/P0_S0_LOCAL_PER_QUESTION.csv"
S0_VERIFY = PROGRAM / "phase_0/p0_s0_historical_replay/P0_S0_VERIFICATION.json"
QWEN_SCORE_ROOT = PROGRAM / "phase_2/diagnostic/h3_reliability_fusion_v1/score_freeze/cells"
LLAMA_SCORE_ROOT = PROGRAM / "phase_2/transfer/h3_llama4/score_freeze/cells"


class HeadToHeadError(RuntimeError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise HeadToHeadError(f"refusing to write empty table: {path}")
    fields = list(dict.fromkeys(key for row in values for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in values])


def registered_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def load_contract() -> dict[str, Any]:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    expected = {
        "schema": "reasoning-localization-h3-historical-headtohead-execution-v1",
        "status": "FROZEN_BEFORE_RUN",
        "experiment_id": EXPERIMENT,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": SEED,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise HeadToHeadError(f"execution registry mismatch: {key}")
    for source in payload["sources"]:
        path = registered_path(source["path"])
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise HeadToHeadError(f"frozen source hash mismatch: {path}")
    return payload


def verify_historical_anchor(contract: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(S0_VERIFY.read_text(encoding="utf-8"))
    checks = value.get("checks", {})
    if value.get("status") != "CHECKSUM_EQUIVALENT" or not all(
        checks.get(key) is True
        for key in (
            "per_question_byte_exact", "cell_metrics_semantic_exact",
            "aggregate_semantic_exact", "intervals_semantic_exact",
        )
    ):
        raise HeadToHeadError("historical S0 replay is not checksum-equivalent")
    expected = contract["expected"]
    if abs(float(value["reference_macro_f1"]) - expected["entropy_macro_f1"]) > 1e-15:
        raise HeadToHeadError("historical entropy alias failed")
    if abs(float(value["finalist_macro_f1"]) - expected["finalist_macro_f1"]) > 1e-15:
        raise HeadToHeadError("historical finalist alias failed")
    if value["population_sha256"] != expected["population_sha256"]:
        raise HeadToHeadError("historical population alias failed")
    return value


def raw_units(path: Path) -> tuple[str, ...]:
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    units = []
    for key in sorted(cache, key=lambda value: (str(type(value)), str(value))):
        row = cache[key]
        unit = str(row.get("id", ""))
        if not unit:
            raise HeadToHeadError(f"raw row lacks official ID: {path}")
        units.append(unit)
    if len(units) != len(set(units)):
        raise HeadToHeadError(f"duplicate raw IDs: {path}")
    return tuple(units)


def candidate_source(model: str, family: str) -> Path:
    root = QWEN_SCORE_ROOT if model == "qwen3_8b" else LLAMA_SCORE_ROOT
    return root / f"processbench_{family}_{model}/scores.npz"


def freeze_scores(contract: Mapping[str, Any]) -> None:
    if ROOT.exists():
        raise FileExistsError(ROOT)
    score_root = ROOT / "score_freeze/cells"
    score_root.mkdir(parents=True)
    registry = load_external_registry(
        repo=contract["external_repo"],
        registry_path=contract["external_registry"],
        population_registry_path=contract["population_registry"],
    )
    key = load_identity_key(contract["identity_key"])
    source_by_family = {row["family"]: registered_path(row["path"]) for row in contract["raw_unit_sources"]}
    unit_rosters = {family: raw_units(source_by_family[family]) for family in FAMILIES}
    records = []
    for model, family in CELLS:
        cell_id = f"processbench_{family}_{model}"
        source = candidate_source(model, family)
        frozen = load_npz_no_pickle(source)
        row_ids = tuple(map(str, frozen["row_ids"].tolist()))
        offsets = np.asarray(frozen["segment_offsets"], dtype=np.int64)
        units = unit_rosters[family]
        spec = registry.by_cell[cell_id]
        identity = apply_external_id_contract(
            registry, spec, units, units, identity_key=key
        )
        opaque_by_unit = dict(zip(units, identity.row_ids))
        index_by_opaque = {row_id: index for index, row_id in enumerate(row_ids)}
        if set(opaque_by_unit.values()) != set(row_ids):
            raise HeadToHeadError(f"opaque identity mapping failed: {cell_id}")
        output: dict[str, list[Any]] = defaultdict(list)
        for unit in sorted(units):
            role = _stage_partition(family, unit)
            if role not in {"calibration", "audit"}:
                continue
            row_index = index_by_opaque[opaque_by_unit[unit]]
            lo, hi = map(int, offsets[row_index:row_index + 2])
            if hi <= lo:
                raise HeadToHeadError(f"empty step roster: {cell_id}/{unit}")
            h0_combined = np.asarray(frozen["h0_combined"][lo:hi], dtype=np.float64)
            h0_local = np.asarray(frozen["h0_local"][lo:hi], dtype=np.float64)
            h2_local = np.asarray(frozen["h2_local"][lo:hi], dtype=np.float64)
            h3_local = np.asarray(frozen["h3_equal_local"][lo:hi], dtype=np.float64)
            if not all(np.isfinite(x).all() for x in (h0_combined, h0_local, h2_local, h3_local)):
                raise HeadToHeadError(f"nonfinite candidate score: {cell_id}/{unit}")
            output["unit"].append(unit)
            output["role"].append(role)
            output["h0_detector"].append(float(np.max(h0_combined)))
            output["h0_locator"].append(int(np.argmax(h0_combined)))
            output["h2_locator"].append(int(np.argmax(h2_local)))
            output["h3_locator"].append(int(np.argmax(h3_local)))
        target = score_root / cell_id
        target.mkdir()
        arrays = {
            "unit": np.asarray(output["unit"], dtype="<U64"),
            "role": np.asarray(output["role"], dtype="<U16"),
            "h0_detector": np.asarray(output["h0_detector"], dtype="<f8"),
            "h0_locator": np.asarray(output["h0_locator"], dtype="<i8"),
            "h2_locator": np.asarray(output["h2_locator"], dtype="<i8"),
            "h3_locator": np.asarray(output["h3_locator"], dtype="<i8"),
        }
        score_sha = atomic_write_npz(target / "scores.npz", arrays)
        counts = {role: output["role"].count(role) for role in ("calibration", "audit")}
        record = {
            "cell_id": cell_id, "model": model, "family": family,
            "source_score_path": str(source.relative_to(REPO)),
            "source_score_sha256": sha256_file(source),
            "score_sha256": score_sha, "n_rows": len(output["unit"]),
            "role_counts": counts, "labels_selected": False,
        }
        atomic_write_json(target / "CELL_MANIFEST.json", record)
        records.append({
            "cell_id": cell_id, "score_sha256": score_sha,
            "cell_manifest_sha256": sha256_file(target / "CELL_MANIFEST.json"),
            "role_counts": counts,
        })
    if sum(row["role_counts"]["audit"] for row in records) != 1270:
        raise HeadToHeadError("historical audit scorer-row count changed")
    manifest = {
        "schema": "reasoning-localization-h3-historical-score-freeze-v1",
        "status": "FROZEN_BEFORE_AUDIT_LABEL_OPEN", "experiment_id": EXPERIMENT,
        "labels_selected": False, "historical_anchor_verified": True,
        "execution_registry_sha256": sha256_file(REGISTRY), "cells": records,
    }
    atomic_write_json(ROOT / "SCORE_FREEZE_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2))


def labels_by_family(contract: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    output = {}
    for source in contract["raw_unit_sources"]:
        path = registered_path(source["path"])
        with path.open("rb") as handle:
            cache = pickle.load(handle)
        labels = {}
        for row in cache.values():
            unit = str(row["id"])
            labels[unit] = int(row["label"])
        output[source["family"]] = labels
    return output


def verified_freeze() -> dict[tuple[str, str], Mapping[str, np.ndarray]]:
    manifest_path = ROOT / "SCORE_FREEZE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_AUDIT_LABEL_OPEN" or manifest.get("labels_selected") is not False:
        raise HeadToHeadError("invalid score freeze manifest")
    output = {}
    for record in manifest["cells"]:
        path = ROOT / "score_freeze/cells" / record["cell_id"] / "scores.npz"
        if sha256_file(path) != record["score_sha256"]:
            raise HeadToHeadError(f"candidate score hash changed: {path}")
        arrays = load_npz_no_pickle(path)
        model = "qwen3_8b" if record["cell_id"].endswith("_qwen3_8b") else "llama31_8b"
        family = record["cell_id"].removeprefix("processbench_").removesuffix("_" + model)
        output[(model, family)] = arrays
    return output


def historical_rows() -> dict[tuple[str, str, str, str], dict[str, Any]]:
    rows = read_csv(S0_ROWS)
    selected = {}
    aliases = {
        "finalist_global_detector_local_locator": FINALIST,
        "max_entropy__step_top5mean": ENTROPY,
    }
    for row in rows:
        if row["candidate"] not in aliases:
            continue
        key = (aliases[row["candidate"]], row["model"], row["family"], row["unit"])
        selected[key] = {
            "candidate": aliases[row["candidate"]], "model": row["model"],
            "family": row["family"], "unit": row["unit"], "task": "local",
            "target": int(row["target"]), "prediction": int(row["prediction"]),
            "locator": int(row["locator"]),
        }
    if len(selected) != 2540:
        raise HeadToHeadError(f"historical candidate roster changed: {len(selected)}")
    return selected


def metric_value(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    result = _processbench(
        [int(row["prediction"]) for row in rows],
        [int(row["target"]) for row in rows],
    )
    return float(result[metric])


def panels(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_rows, panel_rows = [], []
    candidates = sorted({str(row["candidate"]) for row in records})
    for candidate in candidates:
        values = []
        for model, family in CELLS:
            selected = [row for row in records if row["candidate"] == candidate and row["model"] == model and row["family"] == family]
            if not selected:
                continue
            result = _processbench([row["prediction"] for row in selected], [row["target"] for row in selected])
            cell_rows.append({"candidate": candidate, "model": model, "family": family, "cell_id": f"processbench_{family}_{model}", "n": len(selected), **result})
            values.append(result)
        if len(values) != 8:
            raise HeadToHeadError(f"candidate does not cover eight cells: {candidate}")
        panel_rows.append({"candidate": candidate, **{metric: float(np.mean([row[metric] for row in values])) for metric in METRICS}, "n_scorer_rows": 1270, "n_groups": 635})
    return cell_rows, panel_rows


def paired_contrast(
    records: Sequence[Mapping[str, Any]], cell_rows: Sequence[Mapping[str, Any]],
    left: str, right: str, metric: str,
) -> dict[str, Any]:
    relevant = [row for row in records if row["candidate"] in {left, right}]
    prepared = []
    for family in FAMILIES:
        rows = [row for row in relevant if row["family"] == family]
        lookup = {(row["candidate"], row["model"], row["unit"]): row for row in rows}
        units = sorted(set.intersection(*[
            {row["unit"] for row in rows if row["candidate"] == method and row["model"] == model}
            for method in (left, right) for model in MODELS
        ]))
        prepared.append((units, lookup))

    def aggregate(method: str, samples: Sequence[tuple[list[str], Mapping[Any, Any]]]) -> float:
        values = []
        for units, lookup in samples:
            for model in MODELS:
                rows = [lookup[(method, model, unit)] for unit in units]
                values.append(metric_value(rows, metric))
        return float(np.mean(values))

    point = aggregate(left, prepared) - aggregate(right, prepared)
    seed_suffix = "local" if metric == "f1" else metric
    rng = np.random.default_rng(SEED + sum(ord(ch) for ch in left + right + seed_suffix))
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = []
        for units, lookup in prepared:
            chosen = [units[index] for index in rng.integers(0, len(units), len(units))]
            sampled.append((chosen, lookup))
        draws[draw] = aggregate(left, sampled) - aggregate(right, sampled)
    by_cell = {(row["candidate"], row["cell_id"]): float(row[metric]) for row in cell_rows}
    deltas = {cell: by_cell[(left, cell)] - by_cell[(right, cell)] for cell in sorted({row["cell_id"] for row in cell_rows if row["candidate"] == left})}
    return {
        "contrast_id": f"{left}__vs__{right}__{metric}", "left": left, "right": right,
        "metric": metric, "delta": point,
        "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
        "wins": sum(value > 1e-12 for value in deltas.values()),
        "ties": sum(abs(value) <= 1e-12 for value in deltas.values()),
        "losses": sum(value < -1e-12 for value in deltas.values()),
        "worst_cell": min(deltas, key=deltas.get), "worst_cell_delta": min(deltas.values()),
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": int(SEED),
        "interval": "paired historical source-question grouped bootstrap",
    }


def interaction_contrast(
    records: Sequence[Mapping[str, Any]], cell_rows: Sequence[Mapping[str, Any]], metric: str,
) -> dict[str, Any]:
    """Paired 2x2 detector-by-localizer difference in differences.

    The estimand is the H3-localizer gain under the current H0 detector minus
    the H3-localizer gain under the historical detector:

        (H0DET_H3LOC - H0DET_HISTLOC)
        - (HISTDET_H3LOC - HISTDET_HISTLOC).
    """
    arms = (
        "HISTDET_HISTLOC", "HISTDET_H3LOC",
        "H0DET_HISTLOC", "H0DET_H3LOC",
    )
    relevant = [row for row in records if row["candidate"] in arms]
    prepared = []
    for family in FAMILIES:
        rows = [row for row in relevant if row["family"] == family]
        lookup = {(row["candidate"], row["model"], row["unit"]): row for row in rows}
        units = sorted(set.intersection(*[
            {row["unit"] for row in rows if row["candidate"] == arm and row["model"] == model}
            for arm in arms for model in MODELS
        ]))
        prepared.append((units, lookup))

    def aggregate(arm: str, samples: Sequence[tuple[list[str], Mapping[Any, Any]]]) -> float:
        values = []
        for units, lookup in samples:
            for model in MODELS:
                values.append(metric_value([lookup[(arm, model, unit)] for unit in units], metric))
        return float(np.mean(values))

    def did(samples: Sequence[tuple[list[str], Mapping[Any, Any]]]) -> float:
        return (
            aggregate("H0DET_H3LOC", samples) - aggregate("H0DET_HISTLOC", samples)
            - aggregate("HISTDET_H3LOC", samples) + aggregate("HISTDET_HISTLOC", samples)
        )

    point = did(prepared)
    rng = np.random.default_rng(SEED + sum(ord(ch) for ch in "detector_localizer_interaction" + metric))
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = []
        for units, lookup in prepared:
            chosen = [units[index] for index in rng.integers(0, len(units), len(units))]
            sampled.append((chosen, lookup))
        draws[draw] = did(sampled)

    by_cell = {(row["candidate"], row["cell_id"]): float(row[metric]) for row in cell_rows}
    cells = sorted({row["cell_id"] for row in cell_rows if row["candidate"] == arms[0]})
    deltas = {
        cell: (
            by_cell[("H0DET_H3LOC", cell)] - by_cell[("H0DET_HISTLOC", cell)]
            - by_cell[("HISTDET_H3LOC", cell)] + by_cell[("HISTDET_HISTLOC", cell)]
        )
        for cell in cells
    }
    return {
        "contrast_id": f"DETECTOR_X_LOCALIZER__{metric}",
        "metric": metric, "delta": point,
        "ci_low": float(np.quantile(draws, .025)),
        "ci_high": float(np.quantile(draws, .975)),
        "wins": sum(value > 1e-12 for value in deltas.values()),
        "ties": sum(abs(value) <= 1e-12 for value in deltas.values()),
        "losses": sum(value < -1e-12 for value in deltas.values()),
        "worst_cell": min(deltas, key=deltas.get),
        "worst_cell_delta": min(deltas.values()),
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": int(SEED),
        "interval": "paired historical source-question grouped bootstrap",
        "estimand": "(H0DET_H3LOC-H0DET_HISTLOC)-(HISTDET_H3LOC-HISTDET_HISTLOC)",
    }


def make_svg(panel_rows: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]], path: Path) -> None:
    absolute = {row["candidate"]: float(row["f1"]) for row in panel_rows if row["candidate"] in END_TO_END}
    versus = {row["left"]: row for row in contrasts if row["right"] == FINALIST and row["metric"] == "f1"}
    labels = {ENTROPY: "Entropy", FINALIST: "Historical finalist", H0: "H0", H2: "H2", H3: "H3 equal"}
    colors = {ENTROPY: "#8795a8", FINALIST: "#172f55", H0: "#6b7b91", H2: "#147d79", H3: "#3156b8"}
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="520" viewBox="0 0 1080 520" role="img" aria-labelledby="title desc">',
        '<title id="title">Historical-regime H3 head-to-head</title>',
        '<desc id="desc">Absolute macro F1 and paired deltas versus the historical 0.3662 finalist.</desc>',
        '<rect width="1080" height="520" fill="#fbfcfe"/>',
        '<style>text{font-family:Inter,system-ui,sans-serif;fill:#172f55}.t{font-size:24px;font-weight:800}.s{font-size:13px;fill:#5b6b80}.l{font-size:13px;font-weight:700}.v{font-size:12px;font-weight:700}.a{stroke:#a3afbf}.z{stroke:#7b8798;stroke-dasharray:5 4}.ci{stroke:#3156b8;stroke-width:4}.p{fill:#3156b8}</style>',
        '<text class="t" x="38" y="38">Historical-regime head-to-head</text>',
        '<text class="s" x="38" y="62">Exact Stage-4 audit rows · 635 grouped questions · retrospective</text>',
        '<text class="l" x="38" y="102">Absolute macro F1</text>',
    ]
    x0, x1 = 190.0, 500.0
    def ax(v: float) -> float: return x0 + (v - .30) / .10 * (x1 - x0)
    for tick in (.30, .32, .34, .36, .38, .40):
        x = ax(tick); lines += [f'<line class="a" x1="{x:.1f}" y1="116" x2="{x:.1f}" y2="382" opacity=".25"/>', f'<text class="s" x="{x:.1f}" y="402" text-anchor="middle">{tick:.2f}</text>']
    for index, candidate in enumerate(END_TO_END):
        y = 142 + index * 48; value = absolute[candidate]; x = ax(value)
        lines += [f'<text class="l" x="38" y="{y+5}">{labels[candidate]}</text>', f'<rect x="{x0}" y="{y-13}" width="{max(x-x0,1):.1f}" height="24" rx="5" fill="{colors[candidate]}"/>', f'<text class="v" x="{x+8:.1f}" y="{y+5}">{value:.4f}</text>']
    dx0, dx1 = 720.0, 1010.0
    all_bounds = [0.0] + [float(row[key]) for row in versus.values() for key in ("ci_low", "ci_high")]
    low, high = min(all_bounds) - .004, max(all_bounds) + .004
    def dx(v: float) -> float: return dx0 + (v-low)/(high-low)*(dx1-dx0)
    lines += ['<text class="l" x="610" y="102">Paired delta vs historical finalist</text>', f'<line class="z" x1="{dx(0):.1f}" y1="116" x2="{dx(0):.1f}" y2="382"/>']
    for index, candidate in enumerate((ENTROPY, H0, H2, H3)):
        y = 142 + index * 60; row = versus[candidate]; point, lo, hi = map(float, (row["delta"], row["ci_low"], row["ci_high"]))
        lines += [f'<text class="l" x="610" y="{y+5}">{labels[candidate]}</text>', f'<line class="ci" x1="{dx(lo):.1f}" y1="{y}" x2="{dx(hi):.1f}" y2="{y}"/>', f'<circle class="p" cx="{dx(point):.1f}" cy="{y}" r="6"/>', f'<text class="v" x="{dx1}" y="{y-12}" text-anchor="end">{point:+.4f} [{lo:+.4f}, {hi:+.4f}]</text>']
    lines += ['<text class="s" x="38" y="472">Primary: H3 equal − historical finalist. Zero is the inferential boundary; +0.003 is practical context only.</text>', '</svg>']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_tradeoff_svg(contrasts: Sequence[Mapping[str, Any]], path: Path) -> None:
    lookup = {(row["left"], row["metric"]): float(row["delta"]) for row in contrasts if row["right"] == FINALIST}
    labels = {H0: "H0", H2: "H2", H3: "H3"}; colors = {H0: "#6b7b91", H2: "#147d79", H3: "#3156b8"}
    points = [(lookup[(arm,"clean_abstention")], lookup[(arm,"exact_error")]) for arm in (H0,H2,H3)]
    lim = max(.01, max(abs(v) for point in points for v in point) + .01)
    def x(v: float) -> float: return 320 + v/lim*250
    def y(v: float) -> float: return 260 - v/lim*190
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="650" height="440" viewBox="0 0 650 440" role="img" aria-labelledby="title desc"><title id="title">Exact-error versus clean-abstention deltas</title><desc id="desc">H0 H2 and H3 relative to the historical finalist.</desc><rect width="650" height="440" fill="#fbfcfe"/><style>text{font-family:Inter,system-ui,sans-serif;fill:#172f55}.t{font-size:21px;font-weight:800}.s{font-size:12px;fill:#5b6b80}.l{font-size:13px;font-weight:700}.a{stroke:#8b98aa}.p{stroke:white;stroke-width:2}</style><text class="t" x="32" y="38">Localization vs abstention trade-off</text><text class="s" x="32" y="60">Deltas relative to historical finalist</text>',f'<line class="a" x1="70" y1="{y(0):.1f}" x2="600" y2="{y(0):.1f}"/><line class="a" x1="{x(0):.1f}" y1="80" x2="{x(0):.1f}" y2="390"/>']
    for arm,(clean,exact) in zip((H0,H2,H3),points):
        lines += [f'<circle class="p" cx="{x(clean):.1f}" cy="{y(exact):.1f}" r="9" fill="{colors[arm]}"/>',f'<text class="l" x="{x(clean)+12:.1f}" y="{y(exact)+5:.1f}">{labels[arm]} ({clean:+.3f}, {exact:+.3f})</text>']
    lines += ['<text class="s" x="275" y="425">Δ clean abstention →</text><text class="s" transform="translate(18 275) rotate(-90)">Δ exact first-error →</text></svg>']
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def evaluate(contract: Mapping[str, Any]) -> None:
    anchor = verify_historical_anchor(contract)
    frozen = verified_freeze()
    labels = labels_by_family(contract)
    historical = historical_rows()
    records = list(historical.values())
    h0_nonabstain: dict[tuple[str, str, str], bool] = {}
    hist_nonabstain: dict[tuple[str, str, str], bool] = {}
    thresholds = []
    for model, family in CELLS:
        arrays = frozen[(model, family)]
        units = arrays["unit"].astype(str)
        roles = arrays["role"].astype(str)
        detector = np.asarray(arrays["h0_detector"], dtype=np.float64)
        h0_locator = np.asarray(arrays["h0_locator"], dtype=np.int64)
        h2_locator = np.asarray(arrays["h2_locator"], dtype=np.int64)
        h3_locator = np.asarray(arrays["h3_locator"], dtype=np.int64)
        cal = roles == "calibration"; audit = roles == "audit"
        cal_target = np.asarray([labels[family][unit] for unit in units[cal]], dtype=np.int64)
        threshold, calibration_f1 = _best_threshold(detector[cal], h0_locator[cal], cal_target)
        thresholds.append({"model": model, "family": family, "threshold": threshold, "calibration_f1": calibration_f1, "n_calibration": int(cal.sum()), "n_audit": int(audit.sum())})
        for index in np.flatnonzero(audit):
            unit = units[index]; target = labels[family][unit]
            hist = historical[(FINALIST, model, family, unit)]
            if int(hist["target"]) != target:
                raise HeadToHeadError(f"historical target alias failed: {model}/{family}/{unit}")
            current_error = bool(detector[index] > threshold)
            historical_error = int(hist["prediction"]) != -1
            h0_nonabstain[(model,family,unit)] = current_error
            hist_nonabstain[(model,family,unit)] = historical_error
            for candidate, locator in ((H0,h0_locator[index]),(H2,h2_locator[index]),(H3,h3_locator[index])):
                records.append({"candidate": candidate, "model": model, "family": family, "unit": unit, "task": "local", "target": target, "locator": int(locator), "prediction": int(locator) if current_error else -1})

    # Shared historical-detector diagnostic and the 2x2 cross.
    diagnostic = []
    localizers = (("HISTLOC", None),("H0LOC",H0),("H2LOC",H2),("H3LOC",H3))
    by_candidate = {(row["candidate"],row["model"],row["family"],row["unit"]):row for row in records}
    for model, family in CELLS:
        units = sorted({row["unit"] for row in records if row["candidate"] == FINALIST and row["model"] == model and row["family"] == family})
        for unit in units:
            hist = by_candidate[(FINALIST,model,family,unit)]
            for label, source in localizers:
                locator = int(hist["locator"]) if source is None else int(by_candidate[(source,model,family,unit)]["locator"])
                diagnostic.append({"candidate": f"HISTDET_{label}", "model": model, "family": family, "unit": unit, "task": "local", "target": int(hist["target"]), "locator": locator, "prediction": locator if hist_nonabstain[(model,family,unit)] else -1})
            for label, source in (("HISTLOC",None),("H3LOC",H3)):
                locator = int(hist["locator"]) if source is None else int(by_candidate[(source,model,family,unit)]["locator"])
                diagnostic.append({"candidate": f"H0DET_{label}", "model": model, "family": family, "unit": unit, "task": "local", "target": int(hist["target"]), "locator": locator, "prediction": locator if h0_nonabstain[(model,family,unit)] else -1})

    end_cells, end_panels = panels(records)
    diag_cells, diag_panels = panels(diagnostic)
    panel_map = {row["candidate"]: row for row in end_panels}
    if abs(panel_map[ENTROPY]["f1"] - contract["expected"]["entropy_macro_f1"]) > 1e-15 or abs(panel_map[FINALIST]["f1"] - contract["expected"]["finalist_macro_f1"]) > 1e-15:
        raise HeadToHeadError("post-label historical macro aliases failed")
    pairs = ((H3,FINALIST),(H2,FINALIST),(H0,FINALIST),(H3,H0),(H3,H2),(H3,ENTROPY),(FINALIST,ENTROPY),(ENTROPY,FINALIST))
    contrasts = [paired_contrast(records,end_cells,left,right,metric) for left,right in pairs for metric in METRICS]
    diagnostic_pairs = (("HISTDET_H3LOC","HISTDET_HISTLOC"),("H0DET_HISTLOC","HISTDET_HISTLOC"),("H0DET_H3LOC","H0DET_HISTLOC"),("H0DET_H3LOC","HISTDET_H3LOC"))
    diagnostic_contrasts = [paired_contrast(diagnostic,diag_cells,left,right,metric) for left,right in diagnostic_pairs for metric in METRICS]
    interactions = [interaction_contrast(diagnostic, diag_cells, metric) for metric in METRICS]
    primary = next(row for row in contrasts if row["left"] == H3 and row["right"] == FINALIST and row["metric"] == "f1")
    if primary["ci_low"] > 0:
        verdict = "SUPPORTED_IMPROVEMENT"
    elif primary["delta"] > 0:
        verdict = "NUMERICALLY_BETTER_UNRESOLVED"
    else:
        verdict = "NO_EVIDENCE_OR_WORSE"

    evaluation = ROOT / "evaluation"; evaluation.mkdir()
    write_csv(evaluation / "END_TO_END_DECISIONS.csv", records)
    write_csv(evaluation / "END_TO_END_BY_CELL.csv", end_cells)
    write_csv(evaluation / "END_TO_END_PANELS.csv", end_panels)
    write_csv(evaluation / "END_TO_END_CONTRASTS.csv", contrasts)
    write_csv(evaluation / "DIAGNOSTIC_DECISIONS.csv", diagnostic)
    write_csv(evaluation / "DIAGNOSTIC_BY_CELL.csv", diag_cells)
    write_csv(evaluation / "DIAGNOSTIC_PANELS.csv", diag_panels)
    write_csv(evaluation / "DIAGNOSTIC_CONTRASTS.csv", diagnostic_contrasts)
    write_csv(evaluation / "INTERACTIONS.csv", interactions)
    write_csv(evaluation / "THRESHOLDS.csv", thresholds)
    make_svg(end_panels, contrasts, evaluation / "H3_HISTORICAL_HEADTOHEAD.svg")
    make_tradeoff_svg(contrasts, evaluation / "H3_HISTORICAL_TRADEOFF.svg")
    summary = {
        "schema": "reasoning-localization-h3-historical-headtohead-result-v1",
        "status": "COMPLETE", "evidence_status": "RETROSPECTIVE",
        "primary_hypothesis": "H3_EQUAL_C8 > HISTORICAL_FINALIST",
        "primary_contrast": primary, "verdict": verdict,
        "detector_localizer_interaction": {row["metric"]: row for row in interactions},
        "historical_aliases": {"entropy": anchor["reference_macro_f1"], "finalist": anchor["finalist_macro_f1"]},
        "panels": {candidate: {metric: panel_map[candidate][metric] for metric in METRICS} for candidate in END_TO_END},
        "n_groups": 635, "n_scorer_rows": 1270,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": SEED,
        "practical_context_bound": 0.003,
        "fresh_confirmation_required": True,
    }
    atomic_write_json(evaluation / "SUMMARY.json", summary)
    artifacts = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"):
        artifacts.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)})
    atomic_write_json(ROOT / "ARTIFACT_MANIFEST.json", {"schema": "reasoning-localization-h3-historical-artifacts-v1", "status": "COMPLETE", "experiment_id": EXPERIMENT, "artifacts": artifacts})
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    args = parser.parse_args()
    contract = load_contract()
    verify_historical_anchor(contract)
    if args.stage == "freeze":
        freeze_scores(contract)
    else:
        evaluate(contract)


if __name__ == "__main__":
    main()
