"""Preflight the selected-token and residual-tail probability fusion inputs."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_direct_probability_data_v1 as base  # noqa: E402


DEFAULT_K = 15
MATCH_ATOL = 2e-5
# Frozen float32 corpus maximum measured before scoring: 3.35e-7.
MASS_TOL = 5e-7
OUT = ROOT / "results" / "direct_probability_fusion_v2_selected_tail"


def _load(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _topk(row: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("top_k_logprobs", "top_k_logprobs_raw"):
        value = row.get(key)
        if isinstance(value, dict) and value.get("logprobs") is not None:
            return value
    return None


def audit_selected_tail(path: Path, k: int) -> dict[str, Any]:
    """Audit alignment and semantics of ATP/tail fields in one artifact."""

    counts = {
        "rows": 0,
        "selected_aligned": 0,
        "selected_finite_nonnegative": 0,
        "generated_ids_aligned": 0,
        "top_ids_aligned": 0,
        "top15_mass_valid": 0,
        "selected_top1_rows": 0,
        "selected_top15_rows": 0,
        "selected_top50_rows": 0,
        "selected_outside_saved_rows": 0,
        "matched_logprob_rows": 0,
        "matched_logprob_consistent_rows": 0,
    }
    token_total = 0
    tail_sum = 0.0
    tail_sum_of_squares = 0.0
    tail_threshold_counts = {"gt_0": 0, "gt_1e_6": 0, "gt_1e_3": 0, "gt_1e_2": 0}
    within_row_tail_std: list[float] = []
    answer_top10_tail: list[float] = []
    max_head_mass = 0.0
    min_tail_mass = 1.0
    max_tail_mass = 0.0
    max_matched_abs_error = 0.0
    failures: list[dict[str, Any]] = []
    for row_index, row in enumerate(base._rows(_load(path))):
        counts["rows"] += 1
        payload = _topk(row)
        if payload is None:
            failures.append({"row": row_index, "reason": "missing_topk"})
            continue
        logprobs = np.asarray(payload["logprobs"], dtype=float)
        if logprobs.ndim != 2 or logprobs.shape[1] < k:
            failures.append({"row": row_index, "reason": "invalid_topk_shape"})
            continue
        n_tokens = len(logprobs)
        token_total += n_tokens
        selected = np.asarray(row.get("token_spilled_energies"), dtype=float)
        generated = np.asarray(row.get("gen_token_ids"))
        ids_value = payload.get("ids")
        ids = np.asarray(ids_value) if ids_value is not None else np.empty((0, 0))
        selected_ok = selected.shape == (n_tokens,)
        generated_ok = generated.shape == (n_tokens,)
        ids_ok = ids.ndim == 2 and ids.shape == logprobs.shape
        counts["selected_aligned"] += int(selected_ok)
        counts["generated_ids_aligned"] += int(generated_ok)
        counts["top_ids_aligned"] += int(ids_ok)
        if selected_ok and np.isfinite(selected).all() and (selected >= -MASS_TOL).all():
            counts["selected_finite_nonnegative"] += 1
        else:
            failures.append({"row": row_index, "reason": "invalid_selected_surprisal"})

        head_mass = np.exp(logprobs[:, :k]).sum(axis=1)
        tail = np.clip(1.0 - head_mass, 0.0, 1.0)
        tail_sum += float(tail.sum())
        tail_sum_of_squares += float(tail @ tail)
        tail_threshold_counts["gt_0"] += int(np.sum(tail > 0.0))
        tail_threshold_counts["gt_1e_6"] += int(np.sum(tail > 1e-6))
        tail_threshold_counts["gt_1e_3"] += int(np.sum(tail > 1e-3))
        tail_threshold_counts["gt_1e_2"] += int(np.sum(tail > 1e-2))
        within_row_tail_std.append(float(np.std(tail)))
        n_top = min(10, len(tail))
        answer_top10_tail.append(float(np.partition(tail, len(tail) - n_top)[-n_top:].mean()))
        max_head_mass = max(max_head_mass, float(np.max(head_mass)))
        min_tail_mass = min(min_tail_mass, float(np.min(tail)))
        max_tail_mass = max(max_tail_mass, float(np.max(tail)))
        mass_ok = np.isfinite(head_mass).all() and (head_mass <= 1.0 + MASS_TOL).all()
        counts["top15_mass_valid"] += int(mass_ok)
        if not mass_ok:
            failures.append({"row": row_index, "reason": "top15_mass_exceeds_one"})

        if not (selected_ok and generated_ok and ids_ok):
            continue
        match = ids == generated[:, None]
        any_saved = match.any(axis=1)
        counts["selected_top1_rows"] += int(np.sum(match[:, :1].any(axis=1)))
        counts["selected_top15_rows"] += int(np.sum(match[:, :k].any(axis=1)))
        counts["selected_top50_rows"] += int(np.sum(any_saved))
        counts["selected_outside_saved_rows"] += int(np.sum(~any_saved))
        if any_saved.any():
            row_ids = np.flatnonzero(any_saved)
            positions = np.argmax(match[any_saved], axis=1)
            recovered = logprobs[row_ids, positions]
            error = np.abs(recovered + selected[any_saved])
            max_matched_abs_error = max(max_matched_abs_error, float(np.max(error)))
            counts["matched_logprob_rows"] += int(len(error))
            counts["matched_logprob_consistent_rows"] += int(np.sum(error <= MATCH_ATOL))
            if (error > MATCH_ATOL).any():
                failures.append({"row": row_index, "reason": "selected_logprob_mismatch"})

    n_rows = counts["rows"]
    ready = bool(
        n_rows
        and counts["selected_aligned"] == n_rows
        and counts["selected_finite_nonnegative"] == n_rows
        and counts["generated_ids_aligned"] == n_rows
        and counts["top_ids_aligned"] == n_rows
        and counts["top15_mass_valid"] == n_rows
        and counts["matched_logprob_rows"] == counts["matched_logprob_consistent_rows"]
    )
    denominator = max(token_total, 1)
    tail_mean = tail_sum / denominator
    tail_sd = max(0.0, tail_sum_of_squares / denominator - tail_mean**2) ** 0.5
    row_std = np.asarray(within_row_tail_std, dtype=float)
    answer_tail = np.asarray(answer_top10_tail, dtype=float)
    return {
        "ready": ready,
        "counts": counts,
        "tokens": token_total,
        "membership_rates": {
            "top1": counts["selected_top1_rows"] / denominator,
            "top15": counts["selected_top15_rows"] / denominator,
            "top50": counts["selected_top50_rows"] / denominator,
            "outside_saved_topk": counts["selected_outside_saved_rows"] / denominator,
        },
        "top15_mass": {
            "maximum": max_head_mass,
            "minimum_residual_tail": min_tail_mass,
            "maximum_residual_tail": max_tail_mass,
        },
        "tail_signal": {
            "mean": tail_mean,
            "standard_deviation": tail_sd,
            "token_rates": {
                key: value / denominator for key, value in tail_threshold_counts.items()
            },
            "within_row_standard_deviation": {
                "minimum": float(np.min(row_std)),
                "p01": float(np.quantile(row_std, 0.01)),
                "p05": float(np.quantile(row_std, 0.05)),
                "median": float(np.median(row_std)),
            },
            "top10_answer_tail_standard_deviation_across_rows": float(np.std(answer_tail)),
            "float_noise_guard": MASS_TOL,
        },
        "max_selected_logprob_abs_error_when_saved": max_matched_abs_error,
        "match_tolerance": MATCH_ATOL,
        "example_failures": failures[:10],
    }


def build_audit(source_root: Path, k: int) -> dict[str, Any]:
    base.configure_source_root(source_root)
    inherited = base.build_audit(k)
    rows = [*inherited["localization"], *inherited["historical_24"]]
    for row in rows:
        extension = audit_selected_tail(source_root / row["artifact"], k)
        row["selected_tail"] = extension
        row["ready"] = bool(row["ready"] and extension["ready"])
    # Localization fits each answer separately, so its tail must vary within
    # every answer by more than the measured float32 clipping guard. Historical
    # fusion first aggregates each answer, so the aggregated tail must vary
    # across answers in every cell.
    for row in inherited["localization"]:
        signal = row["selected_tail"]["tail_signal"]
        signal["fit_scale_ready"] = bool(
            signal["within_row_standard_deviation"]["minimum"] > MASS_TOL
        )
        row["ready"] = bool(row["ready"] and signal["fit_scale_ready"])
    for row in inherited["historical_24"]:
        signal = row["selected_tail"]["tail_signal"]
        signal["fit_scale_ready"] = bool(
            signal["top10_answer_tail_standard_deviation_across_rows"] > MASS_TOL
        )
        row["ready"] = bool(row["ready"] and signal["fit_scale_ready"])
    localization = inherited["localization"]
    historical = inherited["historical_24"]
    inherited.update(
        schema="direct-probability-data-audit-v2-selected-tail",
        method_scope="gray-box cached distributions plus cached actual-token probability",
        selected_token_semantics=(
            "token_spilled_energies is -log p(scored token); generated/sampled in the "
            "historical cells and teacher-forced answer-token scoring in PB/PRMB"
        ),
        residual_tail_semantics="1 - raw saved top-15 probability mass; no top-K renormalization",
        summary={
            "localization_artifacts_ready": sum(row["ready"] for row in localization),
            "localization_artifacts_total": len(localization),
            "historical_cells_ready": sum(row["ready"] for row in historical),
            "historical_cells_total": len(historical),
            "localization_tokens": sum(row["selected_tail"]["tokens"] for row in localization),
            "historical_tokens": sum(row["selected_tail"]["tokens"] for row in historical),
        },
    )
    return inherited


def render(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Selected-token and tail preflight",
        "",
        f"- Localization artifacts ready: {summary['localization_artifacts_ready']}/{summary['localization_artifacts_total']}.",
        f"- Historical cells ready: {summary['historical_cells_ready']}/{summary['historical_cells_total']}.",
        f"- Audited tokens: {summary['localization_tokens']:,} localization and {summary['historical_tokens']:,} historical.",
        "- PB/PRMB tokens are teacher-forced scored answer tokens; historical tokens are generated/sampled outputs.",
        "- A selected token outside saved Top-50 is valid because its probability is stored separately.",
        "",
        "| artifact/cell | tokens | selected Top-15 | tail >1e-6 | tail >1e-3 | tail >1e-2 | min within-row tail SD | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in [*audit["localization"], *audit["historical_24"]]:
        ext = row["selected_tail"]
        rates = ext["membership_rates"]
        tail = ext["tail_signal"]
        label = row.get("cell", row["artifact"])
        lines.append(
            f"| `{label}` | {ext['tokens']} | {rates['top15']:.1%} | "
            f"{tail['token_rates']['gt_1e_6']:.1%} | {tail['token_rates']['gt_1e_3']:.1%} | "
            f"{tail['token_rates']['gt_1e_2']:.1%} | "
            f"{tail['within_row_standard_deviation']['minimum']:.2e} | "
            f"{'READY' if row['ready'] else 'STOP'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()
    if args.k != DEFAULT_K:
        raise ValueError("v2 is frozen to K=15")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args.source_root.resolve(), args.k)
    (output / "DATA_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "DATA_AUDIT.md").write_text(render(audit), encoding="utf-8")
    print(json.dumps(audit["summary"], sort_keys=True), flush=True)
    if (
        audit["summary"]["localization_artifacts_ready"]
        != audit["summary"]["localization_artifacts_total"]
        or audit["summary"]["historical_cells_ready"]
        != audit["summary"]["historical_cells_total"]
    ):
        raise SystemExit("preflight failed")


if __name__ == "__main__":
    main()
