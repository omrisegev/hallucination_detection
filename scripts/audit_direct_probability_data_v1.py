"""Audit cached probability-rank coverage for the two gray-box experiments.

This reads existing artifacts only.  It checks the full localization caches and
the canonical frozen 24-cell roster for a finite, descending top-K log-probability
matrix.  The experiment contract uses K=15 so that the direct-probability method
and the existing token entropy see the same number of vocabulary ranks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inscope_cells import INSCOPE  # noqa: E402


DEFAULT_K = 15
SOURCE_ROOT = ROOT
PB_DIRS: tuple[Path, Path]
PRMB_PATH: Path
PRMB_LABEL_PATH: Path
REPGRID: Path
OUT = ROOT / "results" / "direct_probability_fusion_v1"


def configure_source_root(source_root: Path) -> None:
    """Point the read-only inputs at the frozen source checkout."""

    global SOURCE_ROOT, PB_DIRS, PRMB_PATH, PRMB_LABEL_PATH, REPGRID
    SOURCE_ROOT = source_root.resolve()
    PB_DIRS = (
        SOURCE_ROOT / "dataset_cache" / "repgrid" / "pb_qwen3_4b",
        SOURCE_ROOT / "dataset_cache" / "repgrid" / "pb_qwen3_8b",
    )
    PRMB_PATH = (
        SOURCE_ROOT
        / "dataset_cache"
        / "four_localization"
        / "prmbench_qwen3_8b_telemetry_full"
        / "prmbench_telemetry.pkl"
    )
    PRMB_LABEL_PATH = (
        SOURCE_ROOT
        / "dataset_cache"
        / "four_localization"
        / "prmbench_qwen25math7b_full"
        / "prmbench_prm.pkl"
    )
    REPGRID = SOURCE_ROOT / "dataset_cache" / "repgrid"


configure_source_root(ROOT)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _rows(payload: Any) -> Iterable[dict[str, Any]]:
    values = payload.values() if isinstance(payload, dict) else payload
    for value in values:
        if not isinstance(value, dict):
            continue
        candidates = value.get("candidates")
        if isinstance(candidates, (list, tuple)):
            for candidate in candidates:
                if isinstance(candidate, dict):
                    yield candidate
        else:
            yield value


def _matrix(row: dict[str, Any]) -> np.ndarray | None:
    for key in ("top_k_logprobs", "top_k_logprobs_raw"):
        value = row.get(key)
        if isinstance(value, dict) and value.get("logprobs") is not None:
            matrix = np.asarray(value["logprobs"])
            return matrix
    return None


def audit_artifact(
    path: Path, required_k: int, *, require_inline_label: bool = True
) -> dict[str, Any]:
    payload = _load(path)
    counts = {
        "rows": 0,
        "with_topk": 0,
        "with_required_k": 0,
        "finite": 0,
        "descending": 0,
        "length_matches_entropy": 0,
        "with_label": 0,
    }
    ranks: list[int] = []
    token_lengths: list[int] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(_rows(payload)):
        counts["rows"] += 1
        if "label" in row or "labels" in row:
            counts["with_label"] += 1
        matrix = _matrix(row)
        if matrix is None or matrix.ndim != 2 or matrix.shape[0] == 0:
            if len(failures) < 10:
                failures.append({"row": index, "reason": "missing_or_invalid_topk"})
            continue
        counts["with_topk"] += 1
        token_lengths.append(int(matrix.shape[0]))
        ranks.append(int(matrix.shape[1]))
        if matrix.shape[1] >= required_k:
            counts["with_required_k"] += 1
        if np.isfinite(matrix).all():
            counts["finite"] += 1
        elif len(failures) < 10:
            failures.append({"row": index, "reason": "nonfinite_topk"})
        if matrix.shape[1] < 2 or np.all(np.diff(matrix, axis=1) <= 1e-7):
            counts["descending"] += 1
        elif len(failures) < 10:
            failures.append({"row": index, "reason": "not_descending"})
        entropy = row.get("token_entropies")
        if entropy is not None and len(entropy) == matrix.shape[0]:
            counts["length_matches_entropy"] += 1
    n_rows = counts["rows"]
    ready = bool(
        n_rows
        and counts["with_required_k"] == n_rows
        and counts["finite"] == n_rows
        and counts["descending"] == n_rows
        and counts["length_matches_entropy"] == n_rows
        and (not require_inline_label or counts["with_label"] == n_rows)
    )
    return {
        "artifact": str(path.relative_to(SOURCE_ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "required_k": required_k,
        "ready": ready,
        "counts": counts,
        "min_saved_k": min(ranks) if ranks else None,
        "max_saved_k": max(ranks) if ranks else None,
        "min_tokens": min(token_lengths) if token_lengths else None,
        "median_tokens": float(np.median(token_lengths)) if token_lengths else None,
        "max_tokens": max(token_lengths) if token_lengths else None,
        "example_failures": failures,
        "inline_label_required": require_inline_label,
    }


def audit_prm_label_join() -> dict[str, Any]:
    telemetry = _load(PRMB_PATH)
    labels = _load(PRMB_LABEL_PATH)
    telemetry_ids = {
        str(row.get("idx")) for row in _rows(telemetry) if row.get("idx") is not None
    }
    label_rows = list(_rows(labels))
    label_ids = {
        str(row.get("idx")) for row in label_rows if row.get("idx") is not None
    }
    with_step_labels = sum(
        isinstance(row.get("labels"), (list, tuple, np.ndarray))
        and len(row["labels"]) > 0
        for row in label_rows
    )
    return {
        "artifact": str(PRMB_LABEL_PATH.relative_to(SOURCE_ROOT)).replace("\\", "/"),
        "bytes": PRMB_LABEL_PATH.stat().st_size,
        "sha256": _sha256(PRMB_LABEL_PATH),
        "telemetry_ids": len(telemetry_ids),
        "label_ids": len(label_ids),
        "matched_ids": len(telemetry_ids & label_ids),
        "telemetry_only_ids": sorted(telemetry_ids - label_ids)[:10],
        "label_only_ids": sorted(label_ids - telemetry_ids)[:10],
        "rows_with_step_labels": with_step_labels,
        "ready": bool(
            telemetry_ids
            and telemetry_ids == label_ids
            and with_step_labels == len(label_rows)
        ),
    }


def _historical_artifact(cell: str) -> Path:
    matches = sorted((REPGRID / cell).glob("raw_*.pkl"))
    if len(matches) != 1:
        raise RuntimeError(f"{cell}: expected exactly one raw_*.pkl, found {len(matches)}")
    return matches[0]


def build_audit(required_k: int) -> dict[str, Any]:
    localization_paths = [
        path
        for directory in PB_DIRS
        for path in sorted(directory.glob("processbench_*.pkl"))
    ] + [PRMB_PATH]
    localization = [
        audit_artifact(
            path,
            required_k,
            require_inline_label=path != PRMB_PATH,
        )
        for path in localization_paths
    ]
    prm_label_join = audit_prm_label_join()
    for row in localization:
        if row["artifact"].endswith("prmbench_telemetry.pkl"):
            row["external_label_join"] = prm_label_join
            row["ready"] = bool(row["ready"] and prm_label_join["ready"])
    historical = []
    for cell in INSCOPE:
        row = audit_artifact(_historical_artifact(cell), required_k)
        row["cell"] = cell
        historical.append(row)
        print(
            f"[historical] {len(historical):02d}/{len(INSCOPE)} {cell}: "
            f"{'READY' if row['ready'] else 'NOT READY'} "
            f"({row['counts']['with_required_k']}/{row['counts']['rows']} rows)",
            flush=True,
        )
    return {
        "schema": "direct-probability-data-audit-v1",
        "method_scope": "gray-box cached logits only; white-box excluded",
        "source_root": str(SOURCE_ROOT),
        "required_k": required_k,
        "entropy_reference": {
            "saved_token_entropies_support": 15,
            "definition": "Shannon entropy after renormalizing the top-15 probabilities",
        },
        "localization": localization,
        "prmbench_label_join": prm_label_join,
        "historical_24": historical,
        "summary": {
            "localization_artifacts_ready": sum(row["ready"] for row in localization),
            "localization_artifacts_total": len(localization),
            "historical_cells_ready": sum(row["ready"] for row in historical),
            "historical_cells_total": len(historical),
        },
    }


def render_report(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Direct probability fusion: data audit",
        "",
        f"Required vocabulary ranks: **K={audit['required_k']}**.",
        "The existing `token_entropies` use top-15 probabilities, renormalized before Shannon entropy.",
        "The token top-10 readout is a separate aggregation over token positions inside a step.",
        "",
        "## Readiness",
        "",
        f"- Localization: {summary['localization_artifacts_ready']}/{summary['localization_artifacts_total']} artifacts ready.",
        f"- Historical complete-answer benchmark: {summary['historical_cells_ready']}/{summary['historical_cells_total']} cells ready.",
        "- This audit uses cached gray-box data only. It does not use or alter white-box captures.",
        "",
        "## Historical cells",
        "",
        "| cell | rows | min K | token range | status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in audit["historical_24"]:
        lines.append(
            f"| `{row['cell']}` | {row['counts']['rows']} | {row['min_saved_k']} | "
            f"{row['min_tokens']}–{row['max_tokens']} | {'READY' if row['ready'] else 'NOT READY'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-k", type=int, default=DEFAULT_K)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.required_k < 2:
        raise ValueError("required K must be at least 2")
    configure_source_root(args.source_root)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args.required_k)
    (out / "DATA_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "DATA_AUDIT.md").write_text(render_report(audit), encoding="utf-8")
    print(json.dumps(audit["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
