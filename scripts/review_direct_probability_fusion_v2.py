"""Compare v2 with frozen v1 and render the self-contained result report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inscope_cells import GROUP, INSCOPE  # noqa: E402
from scripts.run_direct_probability_fusion_v2 import (  # noqa: E402
    BOOT_SEED,
    _historical_source,
    _matched_historical_candidates,
    _paired_group_auc_bootstrap,
    auc,
    configure_source_root,
    load_pickle,
)
from spectral_utils.historical_fusion_evaluation import pb_metrics  # noqa: E402


NAMES = {
    "entropy": "Token Entropy",
    "augmented_equal": "Selected + Tail Probability Fusion - Equal Weights",
    "augmented_iu": "Selected + Tail Probability Fusion - IU-PCR",
    "augmented_joint_lw": "Selected + Tail Probability Fusion - Joint Shrinkage",
    "v1_equal": "Top-15 Probability Fusion - Equal Weights (v1)",
    "v1_iu": "Top-15 Probability Fusion - IU-PCR (v1)",
    "v1_joint": "Top-15 Probability Fusion - Joint Shrinkage (v1)",
    "historical_iu_pcr": "Historical IU-PCR",
}
COLORS = {
    "entropy": "#64748b",
    "augmented_equal": "#06b6d4",
    "augmented_iu": "#2563eb",
    "augmented_joint_lw": "#7c3aed",
    "v1_equal": "#a5f3fc",
    "v1_iu": "#93c5fd",
    "v1_joint": "#c4b5fd",
    "historical_iu_pcr": "#dc2626",
}
V1_MAP = {"v1_equal": "rank_equal", "v1_iu": "rank_iu", "v1_joint": "rank_joint_lw"}
V2_MAP = {
    "augmented_equal": "augmented_equal",
    "augmented_iu": "augmented_iu",
    "augmented_joint_lw": "augmented_joint_lw",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _within_per_answer(records, joined, flat: np.ndarray) -> np.ndarray:
    offsets = joined["offsets"]
    labels = joined["labels"]
    output = np.full(len(records), np.nan)
    for index, record in enumerate(records):
        if record["cell"].startswith("pb_"):
            continue
        truth = labels[offsets[index] : offsets[index + 1]]
        score = flat[offsets[index] : offsets[index + 1]]
        usable = truth >= 0
        if (truth[usable] == 1).any() and (truth[usable] == 0).any():
            output[index] = auc(truth[usable] == 1, score[usable])
    return output


def localization_contrasts(
    out: Path,
    v1: Path,
    source_root: Path,
    draws: int,
) -> dict:
    current = read_json(out / "LOCALIZATION.json")
    previous = read_json(v1 / "LOCALIZATION.json")
    current_scores = np.load(out / "LOCALIZATION_SCORES.npz", allow_pickle=False)
    previous_scores = np.load(v1 / "LOCALIZATION_SCORES.npz", allow_pickle=False)
    bench = source_root / "results" / "localization_full_benchmark_v3" / "evaluation"
    records = read_json(bench / "JOINED.json")["records"]
    joined = np.load(bench / "JOINED.npz", allow_pickle=False)
    cells = np.asarray([record["cell"] for record in records])
    groups = np.asarray([record["group_id"] for record in records])
    pb = np.asarray([cell.startswith("pb_") for cell in cells])
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    pairs = [
        ("augmented_iu", "v1_iu"),
        ("augmented_equal", "v1_equal"),
        ("augmented_joint_lw", "v1_joint"),
    ]
    per = {}
    metrics = {}
    for left, right_alias in pairs:
        right = V1_MAP[right_alias]
        per[left] = {
            "prediction": current_scores[f"prediction__{left}"],
            "within": _within_per_answer(records, joined, current_scores[f"steps__{left}"]),
        }
        per[right_alias] = {
            "prediction": previous_scores[f"prediction__{right}"],
            "within": _within_per_answer(records, joined, previous_scores[f"steps__{right}"]),
        }
        metrics[left] = current["methods"][left]
        metrics[right_alias] = previous["methods"][right]

    rng = np.random.default_rng(BOOT_SEED + 33)
    sampled_groups = rng.integers(0, len(unique_groups), size=(draws, len(unique_groups)))
    output = {}
    for left, right in pairs:
        pb_values = []
        within_values = []
        common = np.isfinite(per[left]["within"]) & np.isfinite(per[right]["within"])
        for sampled in sampled_groups:
            weight = np.bincount(sampled, minlength=len(unique_groups))[inverse].astype(float)
            left_pb = pb_metrics(
                joined["target"][pb], per[left]["prediction"][pb], np.ones(pb.sum(), dtype=bool),
                cells[pb], weights=weight[pb]
            )["macros"]["all"]
            right_pb = pb_metrics(
                joined["target"][pb], per[right]["prediction"][pb], np.ones(pb.sum(), dtype=bool),
                cells[pb], weights=weight[pb]
            )["macros"]["all"]
            pb_values.append(float(left_pb - right_pb))
            within_values.append(
                float(np.average(per[left]["within"][common] - per[right]["within"][common], weights=weight[common]))
            )
        key = f"{left}_minus_{right}"
        output[key] = {
            "left": left,
            "right": right,
            "pb_delta": metrics[left]["pb_all8"] - metrics[right]["pb_all8"],
            "pb_ci97_5": np.percentile(pb_values, [1.25, 98.75]).tolist(),
            "prm_within_delta": metrics[left]["prm_within"] - metrics[right]["prm_within"],
            "prm_within_ci97_5": np.percentile(within_values, [1.25, 98.75]).tolist(),
            "bootstrap_unit": "canonical source group",
            "draws": draws,
        }
    return output


def historical_contrasts(
    out: Path,
    v1: Path,
    source_root: Path,
    draws: int,
) -> dict:
    current = read_json(out / "HISTORICAL_24.json")
    previous = read_json(v1 / "HISTORICAL_24.json")
    current_scores = np.load(out / "HISTORICAL_24_SCORES.npz", allow_pickle=False)
    previous_scores = np.load(v1 / "HISTORICAL_24_SCORES.npz", allow_pickle=False)
    pairs = [
        ("augmented_iu", "v1_iu"),
        ("augmented_equal", "v1_equal"),
        ("augmented_joint_lw", "v1_joint"),
    ]
    deltas = {f"{l}_minus_{r}": [] for l, r in pairs}
    primary_draws = []
    configure_source_root(source_root)
    for position, cell in enumerate(INSCOPE, start=1):
        safe = cell.replace("-", "_")
        labels = current_scores[f"{safe}__label"].astype(bool)
        if not np.array_equal(labels, previous_scores[f"{safe}__label"].astype(bool)):
            raise ValueError(f"{cell}: v1/v2 labels differ")
        for left, right_alias in pairs:
            right = V1_MAP[right_alias]
            left_score = current_scores[f"{safe}__{left}"]
            right_score = previous_scores[f"{safe}__{right}"]
            left_auc = auc(labels, left_score)
            right_auc = auc(labels, right_score)
            expected_left = current["cells"][cell]["auroc"][left]
            expected_right = previous["cells"][cell]["auroc"][right]
            if not np.isclose(left_auc, expected_left, atol=1e-12, rtol=0.0):
                raise AssertionError(f"{cell}: v2 score replay mismatch")
            if not np.isclose(right_auc, expected_right, atol=1e-12, rtol=0.0):
                raise AssertionError(f"{cell}: v1 score replay mismatch")
            deltas[f"{left}_minus_{right_alias}"].append(left_auc - right_auc)
        payload = load_pickle(_historical_source(cell))
        _, problem_ids = _matched_historical_candidates(payload, cell, len(labels))
        primary_draws.append(
            _paired_group_auc_bootstrap(
                labels,
                current_scores[f"{safe}__augmented_iu"],
                previous_scores[f"{safe}__rank_iu"],
                problem_ids,
                draws=draws,
                seed=BOOT_SEED + 500 + position,
            )
        )

    rng = np.random.default_rng(BOOT_SEED + 34)
    sampled_cells = rng.integers(0, len(INSCOPE), size=(draws, len(INSCOPE)))
    output = {}
    for left, right in pairs:
        key = f"{left}_minus_{right}"
        delta = np.asarray(deltas[key])
        cell_draw = delta[sampled_cells].mean(axis=1)
        output[key] = {
            "left": left,
            "right": right,
            "delta": float(delta.mean()),
            "paired_cell_ci97_5": np.percentile(cell_draw, [1.25, 98.75]).tolist(),
            "wins": int(np.sum(delta > 0)),
            "ties": int(np.sum(np.isclose(delta, 0.0, atol=1e-12))),
            "losses": int(np.sum(delta < 0)),
            "draws": draws,
        }
    matrix = np.stack(primary_draws)
    draw_index = np.arange(draws)[:, None]
    hierarchical = matrix[sampled_cells, draw_index].mean(axis=1)
    output["augmented_iu_minus_v1_iu"]["hierarchical_group_ci97_5"] = np.percentile(
        hierarchical, [1.25, 98.75]
    ).tolist()
    output["augmented_iu_minus_v1_iu"]["bootstrap_unit"] = (
        "canonical problems within cells and paired cells for the macro"
    )
    return output


def exploratory_historical_contrasts(current: dict, draws: int) -> dict:
    """Describe post-table anchor comparisons without changing the frozen run."""

    pairs = (
        ("augmented_equal", "historical_iu_pcr"),
        ("augmented_equal", "entropy"),
    )
    rng = np.random.default_rng(BOOT_SEED + 35)
    sampled_cells = rng.integers(0, len(INSCOPE), size=(draws, len(INSCOPE)))
    output = {}
    for left, right in pairs:
        delta = []
        for cell in INSCOPE:
            row = current["cells"][cell]
            left_value = row["auroc"][left]
            right_value = (
                row["historical_iu_pcr"]
                if right == "historical_iu_pcr"
                else row["auroc"][right]
            )
            delta.append(left_value - right_value)
        delta = np.asarray(delta, dtype=float)
        cell_draw = delta[sampled_cells].mean(axis=1)
        output[f"{left}_minus_{right}"] = {
            "left": left,
            "right": right,
            "delta": float(delta.mean()),
            "paired_cell_ci97_5": np.percentile(
                cell_draw, [1.25, 98.75]
            ).tolist(),
            "wins": int(np.sum(delta > 0)),
            "ties": int(np.sum(np.isclose(delta, 0.0, atol=1e-12))),
            "losses": int(np.sum(delta < 0)),
            "draws": draws,
            "status": "exploratory_after_table_review",
            "bootstrap_unit": "paired historical cell",
        }
    return output


def esc(value) -> str:
    return html.escape(str(value))


def f4(value) -> str:
    return f"{float(value):.4f}"


def pct(value) -> str:
    return f"{100 * float(value):.2f}%"


def bars(rows: list[tuple[str, float]], maximum: float, *, percent: bool) -> str:
    body = []
    for key, value in rows:
        width = max(0.0, min(100.0, 100.0 * float(value) / maximum))
        label = pct(value) if percent else f4(value)
        body.append(
            f'<div class="bar-row"><span>{esc(NAMES[key])}</span><div class="track"><i style="width:{width:.2f}%;background:{COLORS[key]}"></i></div><b>{label}</b></div>'
        )
    return "".join(body)


def render(
    current_loc: dict,
    current_hist: dict,
    previous_loc: dict,
    previous_hist: dict,
    comparison: dict,
    audit: dict,
) -> str:
    loc = {
        "entropy": current_loc["methods"]["entropy"],
        **{key: current_loc["methods"][value] for key, value in V2_MAP.items()},
        **{key: previous_loc["methods"][value] for key, value in V1_MAP.items()},
    }
    hist = {
        "historical_iu_pcr": current_hist["macro"]["historical_iu_pcr"],
        "entropy": current_hist["macro"]["entropy"],
        **{key: current_hist["macro"][value] for key, value in V2_MAP.items()},
        **{key: previous_hist["macro"][value] for key, value in V1_MAP.items()},
    }
    loc_order = ["entropy", "v1_equal", "augmented_equal", "v1_iu", "augmented_iu", "v1_joint", "augmented_joint_lw"]
    hist_order = ["historical_iu_pcr", "entropy", "v1_equal", "augmented_equal", "v1_iu", "augmented_iu", "v1_joint", "augmented_joint_lw"]
    loc_rows = "".join(
        f'<tr><th>{esc(NAMES[key])}</th><td>{pct(loc[key]["pb_all8"])}</td><td>{pct(loc[key]["pb_q4"])}</td><td>{pct(loc[key]["pb_q8"])}</td><td>{f4(loc[key]["prm_within"])}</td><td>{f4(loc[key]["prm_pooled"])}</td><td>{f4(loc[key]["prmscore_q08"])}</td><td>{pct(loc[key]["coverage"])}</td></tr>'
        for key in loc_order
    )
    hist_rows = "".join(
        f'<tr><th>{esc(NAMES[key])}</th><td>{f4(hist[key]["all24"])}</td><td>{f4(hist[key]["qa9"])}</td><td>{f4(hist[key]["math15"])}</td></tr>'
        for key in hist_order
    )
    pb_cells = sorted(current_loc["methods"]["entropy"]["pb_cells"])
    pb_cell_rows = "".join(
        f'<tr><th>{esc(cell)}</th><td>{pct(loc["entropy"]["pb_cells"][cell])}</td><td>{pct(loc["v1_iu"]["pb_cells"][cell])}</td><td>{pct(loc["augmented_iu"]["pb_cells"][cell])}</td><td>{100*(loc["augmented_iu"]["pb_cells"][cell]-loc["v1_iu"]["pb_cells"][cell]):+.2f} pp</td></tr>'
        for cell in pb_cells
    )
    cell_rows = []
    for cell in INSCOPE:
        old = previous_hist["cells"][cell]["auroc"]["rank_iu"]
        new = current_hist["cells"][cell]["auroc"]["augmented_iu"]
        anchor = current_hist["cells"][cell]["historical_iu_pcr"]
        cell_rows.append(
            f'<tr><th>{esc(cell)}</th><td>{esc(GROUP[cell])}</td><td>{f4(anchor)}</td><td>{f4(old)}</td><td>{f4(new)}</td><td class="{("gain" if new>=old else "loss")}">{new-old:+.4f}</td></tr>'
        )
    entropy_route = current_loc["contrasts"]["augmented_iu_minus_entropy"]
    loc_contrast_rows = (
        f'<tr><th>{esc(NAMES["augmented_iu"])} minus {esc(NAMES["entropy"])}</th>'
        f'<td>{100*entropy_route["pb_delta"]:+.2f} pp '
        f'[{100*entropy_route["pb_ci_97_5"][0]:+.2f}, {100*entropy_route["pb_ci_97_5"][1]:+.2f}]</td>'
        f'<td>{entropy_route["prm_within_delta"]:+.4f} '
        f'[{entropy_route["prm_within_ci_97_5"][0]:+.4f}, {entropy_route["prm_within_ci_97_5"][1]:+.4f}]</td></tr>'
    ) + "".join(
        f'<tr><th>{esc(NAMES[row["left"]])} minus {esc(NAMES[row["right"]])}</th><td>{100*row["pb_delta"]:+.2f} pp [{100*row["pb_ci97_5"][0]:+.2f}, {100*row["pb_ci97_5"][1]:+.2f}]</td><td>{row["prm_within_delta"]:+.4f} [{row["prm_within_ci97_5"][0]:+.4f}, {row["prm_within_ci97_5"][1]:+.4f}]</td></tr>'
        for row in comparison["localization"].values()
    )
    hist_contrast_rows = "".join(
        f'<tr><th>{esc(NAMES[row["left"]])} minus {esc(NAMES[row["right"]])}</th><td>{row["delta"]:+.4f}</td><td>[{row.get("hierarchical_group_ci97_5",row["paired_cell_ci97_5"])[0]:+.4f}, {row.get("hierarchical_group_ci97_5",row["paired_cell_ci97_5"])[1]:+.4f}]</td><td>{row["wins"]} / {row["ties"]} / {row["losses"]}</td></tr>'
        for row in comparison["historical"].values()
    )
    diagnostics = "".join(
        f'<tr><th>{esc(NAMES[key])}</th><td>{pct(loc[key]["pb_raw_exact"])}</td><td>{pct(loc[key]["pb_within_one"])}</td><td>{loc[key]["pb_early"]:,}</td><td>{loc[key]["pb_late"]:,}</td><td>{pct(loc[key]["pb_clean_accuracy"])}</td><td>{sum(loc[key].get("fallbacks",{}).values())}</td></tr>'
        for key in ("entropy", "v1_iu", "augmented_iu")
    )
    columns = current_loc["matrix_columns"]
    loc_weights = current_loc["methods"]["augmented_iu"]["mean_weights"]
    hist_weights = [
        float(np.mean([current_hist["cells"][cell]["diagnostics"]["augmented_iu"]["weights"][i] for cell in INSCOPE]))
        for i in range(len(columns))
    ]
    weight_rows = "".join(
        f'<tr><th>{esc(name)}</th><td>{loc_weights[i]:+.5f}</td><td>{hist_weights[i]:+.5f}</td></tr>'
        for i, name in enumerate(columns)
    )
    all_audit_rows = [*audit["localization"], *audit["historical_24"]]
    total_tokens = sum(row["selected_tail"]["tokens"] for row in all_audit_rows)
    top1 = sum(row["selected_tail"]["counts"]["selected_top1_rows"] for row in all_audit_rows) / total_tokens
    top15 = sum(row["selected_tail"]["counts"]["selected_top15_rows"] for row in all_audit_rows) / total_tokens
    top50 = sum(row["selected_tail"]["counts"]["selected_top50_rows"] for row in all_audit_rows) / total_tokens
    tail_mean = sum(
        row["selected_tail"]["tail_signal"]["mean"] * row["selected_tail"]["tokens"]
        for row in all_audit_rows
    ) / total_tokens
    tail_gt_1e6 = sum(
        row["selected_tail"]["tail_signal"]["token_rates"]["gt_1e_6"]
        * row["selected_tail"]["tokens"]
        for row in all_audit_rows
    ) / total_tokens
    tail_gt_1e3 = sum(
        row["selected_tail"]["tail_signal"]["token_rates"]["gt_1e_3"]
        * row["selected_tail"]["tokens"]
        for row in all_audit_rows
    ) / total_tokens
    tail_gt_1e2 = sum(
        row["selected_tail"]["tail_signal"]["token_rates"]["gt_1e_2"]
        * row["selected_tail"]["tokens"]
        for row in all_audit_rows
    ) / total_tokens
    minimum_local_tail_sd = min(
        row["selected_tail"]["tail_signal"]["within_row_standard_deviation"]["minimum"]
        for row in audit["localization"]
    )
    primary = comparison["localization"]["augmented_iu_minus_v1_iu"]
    historical_primary = current_hist["contrasts"]["augmented_iu_minus_historical_iu_pcr"]
    exploratory_rows = "".join(
        f'<tr><th>{esc(NAMES[row["left"]])} minus {esc(NAMES[row["right"]])}</th>'
        f'<td>{row["delta"]:+.4f}</td>'
        f'<td>[{row["paired_cell_ci97_5"][0]:+.4f}, {row["paired_cell_ci97_5"][1]:+.4f}]</td>'
        f'<td>{row["wins"]} / {row["ties"]} / {row["losses"]}</td></tr>'
        for row in comparison["historical_exploratory"].values()
    )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Selected + Tail Probability Fusion v2</title>
<style>:root{{--ink:#172033;--muted:#536176;--paper:#f6f8fc;--card:#fff;--line:#dbe3ee}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Segoe UI,Arial,sans-serif}}main{{max-width:1180px;margin:auto;padding:34px 22px 70px}}h1{{font-size:34px;margin:0}}h2{{margin-top:40px}}.sub{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:22px 0}}.card,.panel,.callout{{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:17px;margin:14px 0;overflow:auto}}.card b{{display:block;font-size:23px;margin-top:5px}}.callout{{border-left:6px solid #2563eb}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead th{{background:#edf3fa}}.bar-row{{display:grid;grid-template-columns:310px 1fr 75px;gap:10px;align-items:center;margin:9px 0}}.track{{height:17px;background:#e7edf5;border-radius:9px;overflow:hidden}}.track i{{display:block;height:100%}}.gain{{color:#047857;font-weight:700}}.loss{{color:#b91c1c;font-weight:700}}code{{background:#edf2f7;padding:2px 5px;border-radius:4px}}@media(max-width:800px){{.cards{{grid-template-columns:1fr}}.bar-row{{grid-template-columns:190px 1fr 65px}}}}</style></head><body><main>
<h1>Selected + Tail Probability Fusion v2</h1><p class="sub">Frozen gray-box experiment. Full ProcessBench, full PRMBench, and the historical 24 cells.</p>
<div class="cards"><div class="card">ProcessBench leader: Token Entropy<b>{pct(loc["entropy"]["pb_all8"])}</b></div><div class="card">Best v2 PRMBench within-answer AUC<b>{f4(loc["augmented_equal"]["prm_within"])}</b><span>Equal Weights</span></div><div class="card">Best v2 historical macro AUROC<b>{f4(hist["augmented_equal"]["all24"])}</b><span>Equal Weights</span></div></div>
<div class="callout"><b>How to use the result.</b> No automatic promotion threshold is applied. The paired differences below show whether there is useful signal; the next development step is chosen only after reviewing all three benchmark views.</div>
<h2>1. What changed?</h2><div class="panel"><p>v1 used a <code>T tokens x 15 probability ranks</code> matrix. v2 uses <code>T x 17</code>: the same 15 rank risks, selected-token surprisal, and residual Top-15 tail mass. K=15, top-10 step readout, annotations, folds and gates are unchanged.</p><p>Selected-token surprisal and tail summaries were already present elsewhere in the project. Within the direct probability matrix, v1 used only the sorted ranks; v2 appends the two raw token-level coordinates.</p><p>The data audit passed 9/9 localization artifacts and 24/24 historical cells. Across {total_tokens:,} audited tokens, the selected token was in Top-1 {top1:.1%}, Top-15 {top15:.1%}, and saved Top-50 {top50:.1%}. Tokens outside Top-50 are valid because their probability is stored separately.</p><p>The float32 mass excess is clipped only below 5e-7. The real tail has mean {tail_mean:.4f}; it is above 1e-6 in {tail_gt_1e6:.1%}, above 0.001 in {tail_gt_1e3:.1%}, and above 0.01 in {tail_gt_1e2:.1%} of tokens. The smallest within-answer tail standard deviation in localization is {minimum_local_tail_sd:.2e}, so answer-local standardization is not amplifying the clipping noise.</p></div>
<div class="panel"><h3>What the complete run says</h3><table><thead><tr><th>Question</th><th>Observed result</th><th>Meaning</th></tr></thead><tbody><tr><th>Does v2 improve one-answer localization?</th><td>IU-PCR is {100*entropy_route["pb_delta"]:+.2f} pp versus Token Entropy; PRMB within-answer is {entropy_route["prm_within_delta"]:+.4f}.</td><td>No reliable localization gain in this formulation.</td></tr><tr><th>Do the two added inputs carry answer-level signal?</th><td>Equal Weights reaches {f4(hist["augmented_equal"]["all24"])} on 24 cells. IU-PCR improves {comparison["historical"]["augmented_iu_minus_v1_iu"]["delta"]:+.4f} over its v1 matrix.</td><td>A small signal appears in complete-answer detection.</td></tr><tr><th>Did learned fusion beat the simple version?</th><td>Equal Weights is above IU-PCR and Joint Shrinkage on the 24-cell macro.</td><td>The current covariance-based weighting does not use the added signal reliably.</td></tr></tbody></table></div>
<h2>2. One-answer localization</h2><div class="panel"><h3>ProcessBench all-eight macro F1</h3>{bars([(k,loc[k]["pb_all8"]) for k in loc_order],0.50,percent=True)}</div>
<div class="panel"><table><thead><tr><th>Method</th><th>PB all 8</th><th>PB Q4</th><th>PB Q8</th><th>PRMB within</th><th>PRMB pooled</th><th>PRMScore</th><th>Coverage</th></tr></thead><tbody>{loc_rows}</tbody></table></div>
<div class="callout"><b>Primary v2-v1 change.</b> ProcessBench {100*primary["pb_delta"]:+.2f} pp, 97.5% CI [{100*primary["pb_ci97_5"][0]:+.2f}, {100*primary["pb_ci97_5"][1]:+.2f}]. PRMB within-answer {primary["prm_within_delta"]:+.4f}, CI [{primary["prm_within_ci97_5"][0]:+.4f}, {primary["prm_within_ci97_5"][1]:+.4f}].</div>
<div class="panel"><h3>All predeclared v2-v1 comparisons</h3><table><thead><tr><th>Comparison</th><th>ProcessBench difference</th><th>PRMB within difference</th></tr></thead><tbody>{loc_contrast_rows}</tbody></table></div>
<div class="panel"><h3>ProcessBench cells</h3><table><thead><tr><th>Cell</th><th>Token Entropy</th><th>Top-15 IU-PCR v1</th><th>Selected + Tail IU-PCR v2</th><th>v2-v1</th></tr></thead><tbody>{pb_cell_rows}</tbody></table></div>
<div class="panel"><h3>Location diagnostics</h3><table><thead><tr><th>Method</th><th>Raw exact peak</th><th>Within one</th><th>Early</th><th>Late</th><th>Clean accuracy</th><th>Fallbacks</th></tr></thead><tbody>{diagnostics}</tbody></table></div>
<h2>3. Historical complete-answer detection</h2><div class="panel"><h3>All-24 macro AUROC</h3>{bars([(k,hist[k]["all24"]) for k in hist_order],0.90,percent=False)}</div>
<div class="panel"><table><thead><tr><th>Method</th><th>All 24</th><th>QA 9</th><th>Math 15</th></tr></thead><tbody>{hist_rows}</tbody></table></div>
<div class="callout"><b>Against Historical IU-PCR.</b> Selected + Tail IU-PCR difference {historical_primary["delta"]:+.4f}; hierarchical 97.5% CI [{historical_primary["hierarchical_group_ci97_5"][0]:+.4f}, {historical_primary["hierarchical_group_ci97_5"][1]:+.4f}].</div>
<div class="panel"><h3>v2 versus v1</h3><table><thead><tr><th>Comparison</th><th>Macro difference</th><th>97.5% interval</th><th>Wins / ties / losses</th></tr></thead><tbody>{hist_contrast_rows}</tbody></table></div>
<div class="panel"><h3>Exploratory anchor comparisons</h3><p>These two comparisons were added after seeing the complete table. They describe the result and were not used to tune the method.</p><table><thead><tr><th>Comparison</th><th>Macro difference</th><th>Paired-cell 97.5% interval</th><th>Wins / ties / losses</th></tr></thead><tbody>{exploratory_rows}</tbody></table></div>
<div class="panel"><h3>All 24 cells</h3><table><thead><tr><th>Cell</th><th>Domain</th><th>Historical IU-PCR</th><th>Top-15 IU-PCR v1</th><th>Selected + Tail IU-PCR v2</th><th>v2-v1</th></tr></thead><tbody>{''.join(cell_rows)}</tbody></table></div>
<h2>4. Learned IU-PCR coefficients</h2><p>Columns are standardized before fitting, so coefficients compare standardized inputs. They are unconstrained and do not need to sum to one.</p><div class="panel"><table><thead><tr><th>Input</th><th>Mean inside one answer</th><th>Mean across 24 cells</th></tr></thead><tbody>{weight_rows}</tbody></table></div>
<h2>5. What LOS-Net teaches us</h2><div class="panel"><ul><li>LOS is the union of the sorted Token Distribution Sequence and Actual Token Probability. Within this direct-matrix test, v1 used the first part and v2 adds the probability of the actual token.</li><li>LOS-Net's ATP-only learned baselines were weaker than the full LOS model. Distribution shape carried extra information beyond selected-token probability and rank.</li><li>The paper used K=1000 by default, but K=10 already captured more than 91% probability mass in every reported HD model/dataset combination. Larger K gave small or diminishing improvements.</li><li>Zero-shot transfer for hallucination detection did not beat simple probability baselines. Fine-tuning was needed. This warns us not to infer localization gains from the paper's supervised response-level AUC.</li><li>LOS-Net is a supervised answer classifier with a Transformer across token positions. It does not test answer-local, label-free first-error localization.</li><li>The paper's function-approximation result says LOS-Net can represent a broad family of probability scoring rules. It does not prove that every direct-distribution fusion improves accuracy.</li><li>The reported low detector latency starts after probability distributions are available; it does not include obtaining model logits.</li></ul></div>
<h2>6. Interpretation limits</h2><p>For PB and PRMB, probabilities score the provided answer tokens by teacher forcing. Historical cells contain generated or sampled outputs, and their saved distributions can include the generation warpers used by each original cell. The exact full-vocabulary selected-token rank is unavailable, so this is not an exact LOS-Net reproduction. Residual tail mass is our feature, not a component claimed by LOS-Net.</p>
</main></body></html>'''


def write_csv(out: Path, loc: dict, hist: dict, previous_loc: dict, previous_hist: dict) -> None:
    with (out / "LOCALIZATION_COMPARISON.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "display_name", "pb_all8", "pb_q4", "pb_q8", "prm_within", "prm_pooled", "prmscore", "coverage"])
        for key, source, method in [
            ("entropy", loc, "entropy"),
            ("v1_equal", previous_loc, "rank_equal"),
            ("augmented_equal", loc, "augmented_equal"),
            ("v1_iu", previous_loc, "rank_iu"),
            ("augmented_iu", loc, "augmented_iu"),
            ("v1_joint", previous_loc, "rank_joint_lw"),
            ("augmented_joint_lw", loc, "augmented_joint_lw"),
        ]:
            row = source["methods"][method]
            writer.writerow([key, NAMES[key], row["pb_all8"], row["pb_q4"], row["pb_q8"], row["prm_within"], row["prm_pooled"], row["prmscore_q08"], row["coverage"]])
    with (out / "HISTORICAL_24_COMPARISON.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell", "domain", "historical_iu_pcr", "v1_iu", "augmented_iu", "v2_minus_v1"])
        for cell in INSCOPE:
            anchor = hist["cells"][cell]["historical_iu_pcr"]
            old = previous_hist["cells"][cell]["auroc"]["rank_iu"]
            new = hist["cells"][cell]["auroc"]["augmented_iu"]
            writer.writerow([cell, GROUP[cell], anchor, old, new, new-old])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--v1-results", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    out = args.results.resolve()
    v1 = args.v1_results.resolve()
    source = args.source_root.resolve()
    loc = read_json(out / "LOCALIZATION.json")
    hist = read_json(out / "HISTORICAL_24.json")
    v1_loc = read_json(v1 / "LOCALIZATION.json")
    v1_hist = read_json(v1 / "HISTORICAL_24.json")
    comparison = {
        "schema": "direct-probability-v2-versus-v1-comparison",
        "bootstrap": args.bootstrap,
        "localization": localization_contrasts(out, v1, source, args.bootstrap),
        "historical": historical_contrasts(out, v1, source, args.bootstrap),
        "historical_exploratory": exploratory_historical_contrasts(
            hist, args.bootstrap
        ),
    }
    write_json(out / "V2_V1_COMPARISON.json", comparison)
    audit = read_json(out / "DATA_AUDIT.json")
    write_csv(out, loc, hist, v1_loc, v1_hist)
    (out / "REPORT.html").write_text(
        render(loc, hist, v1_loc, v1_hist, comparison, audit), encoding="utf-8"
    )
    print(out / "REPORT.html")


if __name__ == "__main__":
    main()
