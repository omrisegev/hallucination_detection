#!/usr/bin/env python3
"""Build the self-contained final report and decision from frozen stage files."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_local_online_comprehensive_stage1 import (  # noqa: E402
    OUT,
    PROTOCOL_SHA256,
    _sha256,
)


def _rows(name: str):
    with (OUT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) not in {"", "nan"}]
    return sum(values) / len(values) if values else float("nan")


def _inline(value: str) -> str:
    value = html.escape(value)
    value = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    return value


def _markdown_html(markdown: str) -> str:
    lines = markdown.splitlines()
    output, index = [], 0
    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            output.append(f"<h{level}>{_inline(line[level:].strip())}</h{level}>")
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|---"):
            header = [cell.strip() for cell in line.strip("|").split("|")]
            index += 2
            body = []
            while index < len(lines) and lines[index].startswith("|"):
                body.append([cell.strip() for cell in lines[index].strip("|").split("|")])
                index += 1
            output.append("<div class=table-wrap><table><thead><tr>" + "".join(
                f"<th>{_inline(cell)}</th>" for cell in header
            ) + "</tr></thead><tbody>" + "".join(
                "<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in row) + "</tr>"
                for row in body
            ) + "</tbody></table></div>")
            continue
        if line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append(lines[index][2:])
                index += 1
            output.append("<ul>" + "".join(f"<li>{_inline(item)}</li>" for item in items) + "</ul>")
            continue
        output.append(f"<p>{_inline(line)}</p>")
        index += 1
    return "\n".join(output)


def main() -> None:
    s1 = json.loads((OUT / "STAGE_1_LOCAL_SELECTION.json").read_text())
    s2 = json.loads((OUT / "STAGE_2_ONLINE_SELECTION.json").read_text())
    s3 = json.loads((OUT / "STAGE_3_ARCHITECTURE_SELECTION.json").read_text())
    s4 = json.loads((OUT / "STAGE_4_DECISION.json").read_text())
    audit = json.loads((OUT / "AUDIT.json").read_text())
    warnings = _rows("STAGE_4_WARNING_METRICS.csv")
    residual = _rows("STAGE_4_LENGTH_RESIDUALIZED.csv")
    ablations = _rows("STAGE_4_ABLATIONS.csv")
    cell_metrics = _rows("STAGE_4_CELL_METRICS.csv")
    efficiency = _rows("STAGE_4_EFFICIENCY.csv")

    warning_summary = {}
    for alpha in (0.05, 0.10):
        for candidate in (
            "finalist_global_detector_local_locator",
            "iu28_registered", "step272_twohead",
        ):
            selected = [
                row for row in warnings
                if row["candidate"] == candidate
                and float(row["target_false_warning"]) == alpha
            ]
            warning_summary[(candidate, alpha)] = {
                "fpr": _mean(selected, "audit_false_warning"),
                "coverage": _mean(selected, "audit_error_coverage"),
                "precision": _mean(selected, "audit_warning_precision"),
                "budget": _mean(selected, "mean_first_warning_budget"),
                "remaining": _mean(selected, "mean_potential_tokens_remaining"),
            }
    residual_summary = {}
    for candidate in (
        "finalist_global_detector_local_locator", "iu28_registered",
        "step272_twohead",
    ):
        for budget in (64, 128):
            selected = [
                row for row in residual if row["candidate"] == candidate
                and int(row["budget"]) == budget
            ]
            residual_summary[(candidate, budget)] = {
                "raw": _mean(selected, "raw_auroc"),
                "residual": _mean(selected, "length_residualized_auroc"),
            }

    base = {
        (row["model"], row["family"]): float(row["primary"])
        for row in cell_metrics
        if row["candidate"] == "finalist_global_detector_local_locator"
        and row["task"] == "local"
    }
    ablation_summary = []
    for name in sorted({row["ablation"] for row in ablations}):
        deltas = [
            float(row["primary"]) - base[(row["model"], row["family"])]
            for row in ablations if row["ablation"] == name
        ]
        ablation_summary.append((name, sum(deltas) / len(deltas), min(deltas), max(deltas)))

    critic = [
        row for row in cell_metrics
        if row["candidate"] == "qwen72b_critic" and row["task"] == "local"
    ]
    critic_valid = sum(int(row["n"]) for row in critic)
    critic_requested = sum(int(row.get("requested_n") or row["n"]) for row in critic)
    critic_abstentions = critic_requested - critic_valid

    stage_rows = [
        ("S1 Local feature/locator", "Qwen3-4B GSM8K+MATH development", 0.3517, "Step-272 0.3503", "+0.0014", "PARITY"),
        ("S2 causal Online", "Qwen3-4B GSM8K+MATH development", 0.6020, "Step-272 0.5899", "+0.0121", "PARITY"),
        ("S3 Local architecture", "Qwen3-4B Olympiad+Omni architecture", 0.3515, "Max entropy 0.3407", "+0.0108", "PARITY"),
        ("S3 Online architecture", "Qwen3-4B Olympiad+Omni architecture", 0.5769, "Step-272 0.5746", "+0.0023", "PARITY"),
        ("S4 Local transfer", "Qwen3-8B+Llama, four-family audit", 0.3662, "Max entropy 0.3614", "+0.0048", "PARITY"),
        ("S4 Online transfer", "Qwen3-8B+Llama, four-family audit", 0.5882, "IU28 0.6104", "−0.0222", "REGRESSES"),
    ]

    report = [
        "# Comprehensive Local and Online hallucination-detection cycle",
        "",
        "**Final decision: do not promote the joint finalist.** Local transfer is numerically positive but uncertain; Online transfer breaches the frozen regression margin and loses to IU28 in three of four families. No new GPU/inference run is justified by this retrospective cycle.",
        "",
        "## Executive result",
        "",
        "| stage | frozen evidence | candidate | direct bar | delta | verdict |",
        "|---|---|---:|---|---:|---|",
    ]
    report.extend(
        f"| {name} | {scope} | {value:.4f} | {bar} | {delta} | {verdict} |"
        for name, scope, value, bar, delta, verdict in stage_rows
    )
    report.extend([
        "",
        "The apparent gains in S1-S3 did not become a stable two-task improvement. On the scorer-transfer audit, Local reaches 0.3662 ProcessBench F1 versus 0.3614 for maximum entropy plus the top-five step locator, delta +0.0048 with grouped 95% CI [−0.0264,+0.0375]. Online reaches 0.5882 AUROC@64/128 versus 0.6104 for IU28, delta −0.0222 [−0.0502,+0.0042]. The Online result triggers the preregistered regression verdict even though the interval still includes zero.",
        "",
        "## What was tested",
        "",
        "- Local representations: raw nine token risks; opened raw-seven drop; all 28 broad token views; provenance-balanced six-family compression; historical core-five.",
        "- Local dynamics: level, innovation, short/long contrast, and their frozen combinations. Locators: peak, first persistent calibration-q90 run, and step top-five mean.",
        "- Online dynamics: level/slow, fast/slow, slow/positive-area/persistence, short-long/innovation/recovery, and the five-state combination. Every score was recomputed from explicitly truncated telemetry at 16/32/64/128/256/512 tokens.",
        "- Same-matrix fusion: equal average, ordinary IU-PCR, historical U-PCR compatibility, uniform Laplacian, DUFS-gated Laplacian, temporal Laplacian for Local, and hierarchical U-PCR where identifiable.",
        "- Architecture: shared Local, independent Local/Online, Global+Local, Global+Online, and all quarter-grid three-signal simplex weights.",
        "",
        "## Feature and architecture findings",
        "",
        "The useful development mechanism was provenance balancing. `family6 + level + step_top5mean` was the S1 Local selection; `family6 + fast/slow` was the S2 Online selection. The raw-seven opened drop and uncompressed broad-28 candidates did not satisfy the family-stability guard. Innovation and short-long event coordinates were inconsistent for localization.",
        "",
        "No same-matrix fusion alternative had a wholly positive interval over ordinary IU. Local hierarchical fusion was +0.0098 numerically but uncertain; Online DUFS/uniform/compatibility changes were below +0.001. The S3 simplicity rule selected the registered Global signal for completed-trace detection and Online scoring, retaining the family-six Local head only for the step locator: two physical heads, 36 fitted coordinates, and six persistent Local state scalars.",
        "",
        "Transfer exposed the weakness of that choice: the Global-only prefix score generalized worse than IU28. The finalist beat the Local direct bar in three of four families but beat the Online direct bar in only one of four.",
        "",
        "## Direct and compute-heavy competitors",
        "",
        "On S4, same-access Tier-A Local methods rank: finalist 0.3662, max entropy/top-five locator 0.3614, GL-LIU v1 0.3364, Step-272 0.3078, and Mind the Gap 0.2646. Tier-B Qwen2.5-Math-PRM-7B reaches 0.7280 and the Qwen2.5-72B critic protocol 0.5895. The critic has "
        f"{critic_valid}/{critic_requested} valid scorer-row predictions ({critic_abstentions} abstentions, all in OmniMath); its score is therefore a partial-coverage ceiling, never a same-access reference.",
        "",
        "On S4 Online, IU28 is 0.6104, Step-272 0.6082, mean entropy 0.5926, DeepConf-w64 0.5922, max entropy 0.5921, finalist 0.5882, and DeepConf-w32 0.5853.",
        "",
        "Cross-protocol papers remain context only. uPRM uses next-token probabilities and reports gains over an LLM judge on ProcessBench, but it was not reproduced here. The supervised Streaming Hallucination Detection probe uses hidden states and a different annotated dataset. ProcessBench evaluates the first erroneous step or no-error outcome, while DeepConf motivates the black-box group-confidence baseline. [uPRM](https://arxiv.org/abs/2605.10158), [Streaming Hallucination Detection](https://arxiv.org/abs/2601.02170), [ProcessBench](https://arxiv.org/abs/2412.06559), [DeepConf](https://arxiv.org/abs/2508.15260).",
        "",
        "## Non-withdrawable warning behavior",
        "",
        "| calibration target | method | audit false warning | wrong-trace coverage | precision | mean first budget |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for alpha in (0.05, 0.10):
        for candidate, label in (
            ("finalist_global_detector_local_locator", "finalist"),
            ("step272_twohead", "Step-272"),
            ("iu28_registered", "IU28"),
        ):
            item = warning_summary[(candidate, alpha)]
            report.append(
                f"| {alpha:.0%} | {label} | {item['fpr']:.1%} | {item['coverage']:.1%} | {item['precision']:.1%} | {item['budget']:.1f} |"
            )
    report.extend([
        "",
        "Warnings are one-sided and never withdrawn. Potential remaining tokens are diagnostic only; no forced-stop inference was run, so they are not realized savings. At the 5% target, the finalist covers fewer wrong traces than Step-272 (11.8% versus 13.0%). At the 10% target it covers slightly more (22.3% versus 21.2%) but exceeds the transfer false-warning target (10.1% versus 7.6% for Step-272).",
        "",
        "## Length, ablations, and failure strata",
        "",
        "| method | raw AUROC@64 | residualized | raw AUROC@128 | residualized |",
        "|---|---:|---:|---:|---:|",
    ])
    for candidate, label in (
        ("finalist_global_detector_local_locator", "finalist"),
        ("step272_twohead", "Step-272"),
        ("iu28_registered", "IU28"),
    ):
        left, right = residual_summary[(candidate, 64)], residual_summary[(candidate, 128)]
        report.append(
            f"| {label} | {left['raw']:.4f} | {left['residual']:.4f} | {right['raw']:.4f} | {right['residual']:.4f} |"
        )
    report.extend([
        "",
        "Length residualization is a non-deployable diagnostic because it uses completed trace length. It does not reverse the finalist/IU28 ordering.",
        "",
        "The Local missing-family audit changes only the locator; the Global detector and its threshold stay fixed. Mean deltas after removal are:",
        "",
        "| removed source | mean delta F1 | minimum | maximum |",
        "|---|---:|---:|---:|",
    ])
    report.extend(
        f"| {name} | {mean:+.4f} | {minimum:+.4f} | {maximum:+.4f} |"
        for name, mean, minimum, maximum in ablation_summary
    )
    report.extend([
        "",
        "Entropy dynamics and the combined entropy primitive are the only removals with a material mean loss (about −0.0127/−0.0125). Removing structural or top-k families is mildly positive on average but heterogeneous; these are outcome-opened diagnostics and do not authorize post-hoc pruning. Error-position quartiles, calibration-defined short/medium/long strata, and the answer-correct/process-error versus answer-wrong/process-clean cases are retained in `STAGE_4_STRATA.csv` rather than pooled into one misleading score.",
        "",
        "## Robustness and cost",
        "",
        f"All eight cells pass repeated-fit identity, label-permutation identity, feature-order score equivalence, suffix replacement, and chunk-endpoint identity. The largest feature-order remapping discrepancy is {max(row['feature_order_max_abs_weight_difference'] for row in audit['robustness']):.2e}.",
        "",
        f"Median per-cell feature/head fit time is {statistics.median(float(row['fit_seconds']) for row in efficiency):.1f}s; median complete six-budget Online scoring time is {statistics.median(float(row['online_score_seconds']) for row in efficiency):.1f}s; median measured Python peak memory is {statistics.median(float(row['python_peak_bytes']) for row in efficiency) / 2**20:.1f} MiB. All work was CPU-only over existing caches: zero GPU hours, no new inference, and no Drive mutation.",
        "",
        "## Decision",
        "",
        "- Do not replace the Online incumbent with the S3 joint finalist. Retain IU28 as the strongest S4 direct Online bar; Step-272 remains statistically tied and has better 5% warning coverage than the finalist.",
        "- Treat the Local family-six/top-five mechanism as a promising retrospective candidate, not a confirmed replacement. Its transfer delta is positive but uncertain, and simple maximum entropy remains the strongest transparent direct bar.",
        "- Do not reopen graph fusion, event operators, raw-seven pruning, or a Global-only Online path on these opened cells.",
        "- Do not request a GPU run merely to rescue this result. A future cycle needs a materially new signal or fresh unopened evidence: for example, explicit process supervision/hidden-state access under a new authorization, or a genuinely external model/dataset transfer protocol.",
        "",
        "## Evidence boundary",
        "",
        "All twelve ProcessBench telemetry cells and competitor artifacts were historically opened before this cycle. S1/S2 selected on Qwen3-4B GSM8K/MATH; S3 selected architecture on Qwen3-4B OlympiadBench/OmniMath; S4 audited Qwen3-8B and Llama-3.1-8B scorer telemetry. These scorer copies repeat the same source questions and were resampled together. The result is rigorous retrospective development evidence, not independent confirmation or a SOTA claim.",
        "",
        f"Frozen protocol SHA-256: `{PROTOCOL_SHA256}`.",
    ])
    markdown = "\n".join(report) + "\n"
    (OUT / "REPORT.md").write_text(markdown)

    decision = {
        "verdict": "REGRESSES_DIRECT_COMPETITOR",
        "promote_joint_finalist": False,
        "finalist": s4["finalist"],
        "local": {
            "status": "PARITY_WITH_DIRECT_COMPETITOR",
            "primary": s4["local_primary"],
            "reference": s4["local_reference"],
            "reference_primary": s4["reference_values"]["local"],
            "delta": s4["intervals"]["local"]["delta"],
            "ci": [s4["intervals"]["local"]["ci_low"], s4["intervals"]["local"]["ci_high"]],
            "research_candidate": "family6 level with step-top-five locator and Global detector",
        },
        "online": {
            "status": "REGRESSES_DIRECT_COMPETITOR",
            "primary": s4["online_primary"],
            "reference": s4["online_reference"],
            "reference_primary": s4["reference_values"]["online"],
            "delta": s4["intervals"]["online"]["delta"],
            "ci": [s4["intervals"]["online"]["ci_low"], s4["intervals"]["online"]["ci_high"]],
            "retain_direct_bar": "iu28_registered",
        },
        "stage_verdicts": {
            "S1": s1["verdict"], "S2": s2["verdict"],
            "S3": s3["verdict"], "S4": s4["verdict"],
        },
        "new_gpu_or_inference_run_recommended": False,
        "all_results_retrospective": True,
        "protocol_sha256": PROTOCOL_SHA256,
    }
    (OUT / "DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    body = _markdown_html(markdown)
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Comprehensive Local and Online hallucination-detection cycle</title>
<style>
body{{font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1180px;margin:0 auto;padding:32px;color:#172033;background:#f7f8fb}}
h1,h2{{line-height:1.2;color:#111827}} h1{{font-size:2rem}} h2{{margin-top:2.2rem;border-bottom:1px solid #d8deea;padding-bottom:.35rem}}
p,li{{max-width:90ch}} code{{background:#e8edf5;padding:.12rem .3rem;border-radius:4px}} strong{{color:#0b3b66}}
.table-wrap{{overflow-x:auto;margin:1rem 0}} table{{border-collapse:collapse;width:100%;background:white;box-shadow:0 1px 3px #0001}}
th,td{{border:1px solid #d8deea;padding:.5rem .65rem;text-align:left;vertical-align:top}} th{{background:#eaf0f8;position:sticky;top:0}} tr:nth-child(even) td{{background:#fafcff}}
a{{color:#075ea8}} @media print{{body{{background:white;padding:0}} .table-wrap{{overflow:visible}}}}
</style></head><body>{body}</body></html>"""
    (OUT / "REPORT.html").write_text(html_doc)

    manifest_path = OUT / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "status": "COMPLETE",
        "report_md_sha256": _sha256(OUT / "REPORT.md"),
        "report_html_sha256": _sha256(OUT / "REPORT.html"),
        "decision_sha256": _sha256(OUT / "DECISION.json"),
        "protocol_sha256": PROTOCOL_SHA256,
        "new_inference": False,
        "gpu_hours": 0,
        "drive_mutation": False,
    })
    (manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
