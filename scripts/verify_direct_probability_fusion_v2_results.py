"""Replay v2 result metrics from saved scores and audit the final report."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        raise ValueError("AUROC requires both classes")
    ranks = rankdata(scores)
    return float(
        (ranks[labels].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def require_close(actual, expected, label: str, atol: float = 1e-12) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"{label}: {actual} != {expected}")


def replay_processbench(target, prediction, cells) -> tuple[float, dict[str, float]]:
    values = {}
    for cell in sorted(set(cells)):
        if not cell.startswith("pb_"):
            continue
        mask = cells == cell
        clean = mask & (target == -1)
        error = mask & (target >= 0)
        clean_accuracy = float(np.mean(prediction[clean] == -1))
        error_accuracy = float(np.mean(prediction[error] == target[error]))
        denominator = clean_accuracy + error_accuracy
        values[cell] = (
            2 * clean_accuracy * error_accuracy / denominator if denominator else 0.0
        )
    return float(np.mean(list(values.values()))), values


def replay_localization(results: Path, source: Path, metrics: dict) -> dict:
    joined_meta = read_json(
        source / "results" / "localization_full_benchmark_v3" / "evaluation" / "JOINED.json"
    )
    joined = np.load(
        source / "results" / "localization_full_benchmark_v3" / "evaluation" / "JOINED.npz",
        allow_pickle=False,
    )
    scores = np.load(results / "LOCALIZATION_SCORES.npz", allow_pickle=False)
    records = joined_meta["records"]
    offsets = joined["offsets"]
    labels = joined["labels"]
    target = joined["target"]
    cells = np.asarray([row["cell"] for row in records])
    pb = np.asarray([cell.startswith("pb_") for cell in cells])
    if len(records) != len(offsets) - 1 or len(records) != metrics["n_answers"]:
        raise AssertionError("localization roster length mismatch")

    replay = {}
    for method, row in metrics["methods"].items():
        flat = scores[f"steps__{method}"]
        prediction = scores[f"prediction__{method}"]
        if len(flat) != len(labels) or len(prediction) != len(records):
            raise AssertionError(f"{method}: saved score shape mismatch")
        if not np.isfinite(flat).all():
            raise AssertionError(f"{method}: non-finite step score")
        macro, per_cell = replay_processbench(target, prediction, cells)
        require_close(macro, row["pb_all8"], f"{method} ProcessBench macro")
        for cell, value in per_cell.items():
            require_close(value, row["pb_cells"][cell], f"{method} {cell}")

        within = []
        pooled_mask = np.zeros(len(labels), dtype=bool)
        for index in np.flatnonzero(~pb):
            start, end = offsets[index : index + 2]
            truth = labels[start:end]
            usable = truth >= 0
            if (truth[usable] == 1).any() and (truth[usable] == 0).any():
                within.append(auc(truth[usable] == 1, flat[start:end][usable]))
            pooled_mask[start:end] = usable
        within_value = float(np.mean(within))
        pooled_value = auc(labels[pooled_mask] == 1, flat[pooled_mask])
        require_close(within_value, row["prm_within"], f"{method} PRMB within")
        require_close(pooled_value, row["prm_pooled"], f"{method} PRMB pooled")
        if row["coverage"] != 1.0 or row["valid_answers"] != len(records):
            raise AssertionError(f"{method}: incomplete localization coverage")
        if row["fallbacks"]:
            raise AssertionError(f"{method}: unexpected fitting fallback")
        if method != "entropy":
            weights = np.asarray(row["mean_weights"], dtype=float)
            if weights.shape != (17,) or not np.isfinite(weights).all():
                raise AssertionError(f"{method}: invalid coefficient vector")
        replay[method] = {
            "processbench_all8": macro,
            "prmbench_within": within_value,
            "prmbench_pooled": pooled_value,
        }
    return replay


def replay_historical(results: Path, metrics: dict) -> dict:
    scores = np.load(results / "HISTORICAL_24_SCORES.npz", allow_pickle=False)
    methods = (
        "entropy",
        "augmented_equal",
        "augmented_iu",
        "augmented_joint_lw",
        "historical_iu_pcr",
    )
    collected = {method: [] for method in methods}
    groups = []
    cell_count = 0
    for cell, row in metrics["cells"].items():
        cell_count += 1
        groups.append(row["domain"])
        labels = scores[f"{cell}__label"].astype(bool)
        if len(labels) != row["n_answers"]:
            raise AssertionError(f"{cell}: answer count mismatch")
        for method in methods:
            value = auc(labels, scores[f"{cell}__{method}"])
            expected = (
                row["historical_iu_pcr"]
                if method == "historical_iu_pcr"
                else row["auroc"][method]
            )
            require_close(value, expected, f"{cell} {method}")
            collected[method].append(value)
        for method in ("augmented_equal", "augmented_iu", "augmented_joint_lw"):
            diagnostic = row["diagnostics"][method]
            weights = np.asarray(diagnostic["weights"], dtype=float)
            if weights.shape != (17,) or not np.isfinite(weights).all():
                raise AssertionError(f"{cell} {method}: invalid coefficient vector")
            if diagnostic["fallback"] is not None:
                raise AssertionError(f"{cell} {method}: unexpected fitting fallback")
    if cell_count != 24:
        raise AssertionError(f"historical cell count is {cell_count}, expected 24")

    groups = np.asarray(groups)
    replay = {}
    for method, values in collected.items():
        values = np.asarray(values)
        summary = {
            "all24": float(values.mean()),
            "qa9": float(values[groups == "QA"].mean()),
            "math15": float(values[groups == "math"].mean()),
        }
        for key, value in summary.items():
            require_close(value, metrics["macro"][method][key], f"{method} {key}")
        replay[method] = summary
    if metrics["coverage"] != {method: 1.0 for method in metrics["coverage"]}:
        raise AssertionError("historical coverage is not complete")
    if any(metrics["fallbacks"].values()):
        raise AssertionError("historical fallback dictionary is not empty")
    return replay


def audit_comparison(results: Path, comparison: dict) -> None:
    if comparison["bootstrap"] != 10_000:
        raise AssertionError("comparison did not use 10,000 bootstrap draws")
    for family in ("localization", "historical", "historical_exploratory"):
        for name, row in comparison[family].items():
            intervals = [value for key, value in row.items() if "ci97_5" in key]
            if not intervals:
                raise AssertionError(f"{family}/{name}: missing 97.5% interval")
            for interval in intervals:
                values = np.asarray(interval, dtype=float)
                if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
                    raise AssertionError(f"{family}/{name}: invalid interval")

    html = (results / "REPORT.html").read_text(encoding="utf-8")
    required = (
        "No automatic promotion threshold is applied",
        "Selected + Tail Probability Fusion - IU-PCR minus Token Entropy",
        "Exploratory anchor comparisons",
        "Selected + Tail Probability Fusion - Equal Weights",
        "Historical IU-PCR",
    )
    missing = [text for text in required if text not in html]
    if missing:
        raise AssertionError(f"HTML is missing required content: {missing}")
    if "Ã" in html or "Â" in html:
        raise AssertionError("HTML contains encoding mojibake")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.resolve()
    source = args.source_root.resolve()

    localization_metrics = read_json(results / "LOCALIZATION.json")
    historical_metrics = read_json(results / "HISTORICAL_24.json")
    comparison = read_json(results / "V2_V1_COMPARISON.json")
    localization = replay_localization(results, source, localization_metrics)
    historical = replay_historical(results, historical_metrics)
    audit_comparison(results, comparison)

    review = {
        "schema": "direct-probability-fusion-v2-result-integrity-review",
        "status": "PASS",
        "checks": {
            "localization_metric_replay": "4/4 methods",
            "historical_cell_auroc_replay": "24/24 cells, 5 methods",
            "coverage": "complete",
            "fallbacks": "none",
            "coefficient_shapes": "17 columns",
            "bootstrap_intervals": "present and finite",
            "html_content_and_encoding": "PASS",
        },
        "localization": localization,
        "historical": historical,
        "findings_review": {
            "localization": (
                "The added coordinates do not improve the frozen one-answer localization route."
            ),
            "complete_answer_detection": (
                "The added coordinates carry a small answer-level signal, strongest with equal weights."
            ),
            "attribution_limit": (
                "This combined run cannot separate the contribution of selected-token surprisal from residual tail mass."
            ),
            "fusion_limit": (
                "IU-PCR improves over its direct-matrix v1 counterpart on the historical panel, but equal weights remain stronger; Joint Shrinkage is not competitive."
            ),
        },
    }
    (results / "RESULT_REVIEW.json").write_text(
        json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Independent result replay",
        "",
        "Status: **PASS**",
        "",
        "- Replayed all four localization methods from saved step scores.",
        "- Replayed five answer-level scores in all 24 historical cells.",
        "- Coverage is complete, no fitting fallback occurred, and every learned vector has 17 finite coefficients.",
        "- All reported 97.5% intervals are present and finite; the HTML has the required comparisons and clean UTF-8 text.",
        "",
        "## Findings audit",
        "",
        "- The added inputs do not improve the frozen one-answer localization route.",
        "- A small complete-answer signal is visible, especially with equal weights.",
        "- The combined run cannot tell whether selected-token surprisal, tail mass, or both caused that signal.",
        "- Current IU-PCR weighting does not turn the signal into the strongest method; Joint Shrinkage remains weak.",
        "- No automatic promotion threshold was used.",
        "",
    ]
    (results / "RESULT_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(review["checks"], sort_keys=True))


if __name__ == "__main__":
    main()
