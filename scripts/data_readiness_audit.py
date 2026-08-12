#!/usr/bin/env python3
"""Validate collected data and build the dataset-only readiness report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.data_readiness import (  # noqa: E402
    BLOCKED,
    INCOMPLETE,
    READY,
    READY_WITH_LIMITATIONS,
    SCHEMA_VERSION,
    Audit,
    audit_all,
    registry_payload,
    restricted_pickle,
    sha256_file,
    write_quality_csv,
)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _even_sample(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Choose deterministic, score-spread rows without random state."""

    rows = sorted(rows, key=lambda row: (float(row["bem_score"]), int(row["row_key"])))
    if len(rows) <= n:
        return rows
    indexes = np.linspace(0, len(rows) - 1, n, dtype=int)
    return [rows[int(index)] for index in indexes]


def build_semgrad_review_queue(repo: Path, local_out: Path, report_out: Path) -> None:
    sources = {
        "SciQ": repo / "local_cache/semgrad_bem_regraded/raw_semgrad_sciq_T0.0_bem.pkl",
        "TruthfulQA": repo / "local_cache/semgrad_bem_regraded/raw_semgrad_truthfulqa_T0.0_bem.pkl",
    }
    disagreements: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for dataset, source in sources.items():
        rows = restricted_pickle(source)
        for row_key, row in rows.items():
            candidate = row["candidates"][0]
            proxy = bool(candidate["label"])
            bem = bool(candidate["bem_correct"])
            if proxy == bem:
                continue
            direction = "proxy_correct_bem_incorrect" if proxy else "proxy_incorrect_bem_correct"
            counts[f"{dataset}:{direction}"] += 1
            disagreements.append({
                "dataset": dataset,
                "row_key": int(row_key),
                "direction": direction,
                "question": str(row.get("question", "")),
                "candidate_answer": str(candidate.get("full_text", "")),
                "accepted_answers_json": json.dumps(
                    row.get("gold_row", {}).get("truthful_answers", []), ensure_ascii=False
                ),
                "proxy_label": int(proxy),
                "bem_score": float(candidate["bem_score"]),
                "bem_label": int(bem),
                "human_equivalent": "",
                "reviewer": "",
                "notes": "",
            })
    selected: list[dict[str, Any]] = []
    for dataset in sources:
        for direction in ("proxy_correct_bem_incorrect", "proxy_incorrect_bem_correct"):
            stratum = [
                row for row in disagreements
                if row["dataset"] == dataset and row["direction"] == direction
            ]
            selected.extend(_even_sample(stratum, 20))
    selected.sort(key=lambda row: (row["dataset"], row["direction"], row["row_key"]))
    local_out.mkdir(parents=True, exist_ok=True)
    queue = local_out / "semgrad_bem_manual_audit.csv"
    with queue.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    _json(report_out / "semgrad_bem_manual_audit_manifest.json", {
        "queue_path": str(queue.relative_to(repo)),
        "queue_sha256": sha256_file(queue),
        "selection": "20 rows per dataset and disagreement direction, spread across BEM score",
        "selected_rows": len(selected),
        "all_disagreement_counts": dict(sorted(counts.items())),
        "human_fields_are_blank": ["human_equivalent", "reviewer", "notes"],
        "contains_dataset_text": True,
        "git_policy": "Keep the queue under ignored local_cache; do not publish benchmark text.",
    })


def build_hle_grading_queue(repo: Path, local_out: Path, report_out: Path) -> None:
    source = repo / "dataset_cache/four_localization/hle_full/raw_hle_T0.0.pkl"
    rows = restricted_pickle(source)
    local_out.mkdir(parents=True, exist_ok=True)
    queue = local_out / "hle_official_judge_queue.jsonl"
    with queue.open("w", encoding="utf-8") as handle:
        for row_key in sorted(rows):
            row = rows[row_key]
            gold = row.get("gold_row", {})
            candidate = row["candidates"][0]
            record = {
                "row_key": int(row_key),
                "id": str(gold.get("id", row_key)),
                "question": str(gold.get("question", row.get("question", ""))),
                "correct_answer": str(gold.get("answer", "")),
                "response": str(candidate.get("full_text", "")),
                "answer_type": str(gold.get("answer_type", "")),
                "category": str(gold.get("category", "")),
                "tier": str(gold.get("tier", "")),
                "provisional_rouge_label": int(bool(candidate.get("label", False))),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    _json(report_out / "hle_grading_queue_manifest.json", {
        "queue_path": str(queue.relative_to(repo)),
        "queue_sha256": sha256_file(queue),
        "rows": len(rows),
        "judge_protocol": "data/hle_protocol/PROVENANCE.md#grading--deferred-omri-2026-08-08",
        "required_output_fields": [
            "extracted_final_answer", "reasoning", "correct", "confidence",
        ],
        "contains_protected_benchmark_text": True,
        "git_policy": "Keep the queue under ignored local_cache; never commit or publish it.",
        "research_scores_computed": False,
    })


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if 0 <= value <= 1:
            return f"{100 * value:.1f}%"
        return f"{value:,.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _primary_count(audit: Audit) -> str:
    for key in ("rows", "responses", "claims", "conditions"):
        if key in audit.observed:
            return _fmt(audit.observed[key])
    return "—"


def _balance(audit: Audit) -> str:
    return ", ".join(f"{key}: {_fmt(value)}" for key, value in audit.balance.items()) or "—"


def _failed_checks(audit: Audit) -> list[str]:
    return [key for key, value in audit.checks.items() if not value]


def _md_table(audits: list[Audit]) -> str:
    rows = [
        "| Dataset package | Kind | Count | Balance | Status |",
        "|---|---|---:|---|---|",
    ]
    for audit in audits:
        rows.append(
            f"| `{audit.dataset_id}` | {audit.kind} | {_primary_count(audit)} | "
            f"{_balance(audit)} | **{audit.status}** |"
        )
    return "\n".join(rows)


def _evaluation_use(audit: Audit) -> str:
    """State whether a package can support evaluation without hiding scope limits."""

    if audit.status == READY:
        return "Yes — use under its frozen benchmark protocol."
    if audit.status in {INCOMPLETE, BLOCKED}:
        return "No — required data or integrity work remains."
    uses = {
        "frozen_24cell": "Yes — development/exploratory evaluation only; not an independent confirmation set.",
        "hle_qwen72b": "Yes — interim evaluation only; not a paper-faithful HLE score until the original GPT-4o judge is run.",
        "ragtruth_full_evidence": "Yes — exploratory evaluation only because these labels were already opened.",
        "gasp_ragtruth_400": "Yes — protocol-level comparison only; the paper's exact 400 IDs and splitter are unavailable.",
        "refchecker_claims": "Yes — fixed-claim verification only; do not claim claim-extraction performance.",
        "prmbench_qwen3_8b_telemetry": "Conditionally — first resolve or explicitly exclude the three identified alignment defects.",
    }
    return uses.get(
        audit.dataset_id,
        "Yes — provided every result carries the package's stated limitation.",
    )


def build_markdown(audits: list[Audit], registry: dict[str, Any]) -> str:
    counts = Counter(audit.status for audit in audits)
    ready_for_loading = counts[READY] + counts[READY_WITH_LIMITATIONS]
    hle = next(audit for audit in audits if audit.dataset_id == "hle_qwen72b")
    sections = [
        "# Data Readiness Report",
        "",
        "**Date:** 2026-08-11",
        "",
        f"**Schema:** `{SCHEMA_VERSION}`",
        "",
        f"**Registry fingerprint:** `{registry['registry_fingerprint']}`",
        "",
        "## Purpose",
        "",
        "This report validates the collected data before any hallucination method is run. "
        "It contains no U-PCR, DUFS, detection, or localization result. Raw artifacts were "
        "read but not changed.",
        "",
        "A package is **READY** when its expected files, rows, labels and structural alignment "
        "passed. **READY_WITH_LIMITATIONS** means evaluation is allowed only within the stated "
        "scope and with the caveat attached; it does not mean the package is unusable. "
        "**INCOMPLETE** means files are missing. **BLOCKED** means a required label or integrity "
        "condition is invalid.",
        "",
        "## Headline",
        "",
        f"- {len(audits)} packages were inspected.",
        f"- {ready_for_loading} are structurally loadable now ({counts[READY]} READY and "
        f"{counts[READY_WITH_LIMITATIONS]} READY_WITH_LIMITATIONS).",
        f"- {counts[INCOMPLETE]} package(s) are incomplete and {counts[BLOCKED]} package(s) are blocked.",
        "- Class balance was measured, not modified. No example was resampled or deleted.",
        "- Review queues were prepared under ignored `local_cache`; protected benchmark text is not copied into this report.",
        "",
        "## Readiness matrix",
        "",
        _md_table(audits),
        "",
        "## Can these packages be used for evaluation?",
        "",
        "Yes. Both READY and READY_WITH_LIMITATIONS packages can support evaluation. The latter "
        "must be used only for the claim stated below; INCOMPLETE and BLOCKED packages cannot be "
        "used yet.",
        "",
        "| Package | Evaluation use |",
        "|---|---|",
        *[f"| `{audit.dataset_id}` | {_evaluation_use(audit)} |" for audit in audits],
        "",
        "## Required data work",
        "",
    ]
    action_index = 1
    for audit in audits:
        if audit.blockers or _failed_checks(audit):
            sections.append(f"{action_index}. **{audit.title}:**")
            for item in audit.blockers:
                sections.append(f"   - {item}")
            failed = _failed_checks(audit)
            if failed:
                sections.append(f"   - Failed checks: {', '.join(failed)}.")
            action_index += 1
    if action_index == 1:
        sections.append("No blocking data work remains.")

    sections.extend([
        "",
        "## Dataset details",
        "",
    ])
    for audit in audits:
        sections.extend([
            f"### {audit.title}",
            "",
            f"- **Package ID:** `{audit.dataset_id}`",
            f"- **Status:** {audit.status}",
            f"- **Observed:** {_fmt(audit.observed)}",
            f"- **Balance:** {_balance(audit)}",
            f"- **Checks passed:** {sum(audit.checks.values())}/{len(audit.checks)}",
        ])
        if audit.limitations:
            sections.append("- **Limitations:** " + " ".join(audit.limitations))
        if audit.blockers:
            sections.append("- **Blockers:** " + " ".join(audit.blockers))
        if audit.warnings:
            sections.append("- **Warnings:** " + " ".join(audit.warnings))
        sections.append("")

    sections.extend([
        "## Canonical data contract",
        "",
        "Future consumers must address records by stable `dataset_id`, `record_id`, "
        "`source_id`, split, task, model, condition and parent ID. Large token arrays remain "
        "inside the immutable source artifact and are addressed by artifact path and row key.",
        "",
        "Labels use a separate sidecar contract containing `record_id`, label space, value and "
        "provenance. This prevents a future label-free fitting program from receiving labels by "
        "accident.",
        "",
        "## Decision",
        "",
        (
            "HLE remains blocked until a valid grader lands. "
            if hle.status == BLOCKED else
            "HLE now has a complete interim Codex-judge label sidecar, but it is not the paper-faithful GPT-4o label set. "
        )
        + "Resolve any incomplete package required by the intended experiment. Packages marked "
        "READY may be used unchanged once the benchmark protocol is frozen. Packages marked "
        "READY_WITH_LIMITATIONS may also be evaluated now, but only for the documented scope and "
        "with that limitation included in every result.",
        "",
    ])
    return "\n".join(sections)


def _status_svg(audits: list[Audit]) -> str:
    colors = {
        READY: "#1f9d68",
        READY_WITH_LIMITATIONS: "#d99a14",
        INCOMPLETE: "#d46a1f",
        BLOCKED: "#c43d4b",
    }
    counts = Counter(audit.status for audit in audits)
    total = max(1, len(audits))
    x = 0.0
    rects = []
    labels = []
    for status in (READY, READY_WITH_LIMITATIONS, INCOMPLETE, BLOCKED):
        width = 880 * counts[status] / total
        rects.append(
            f'<rect x="{x:.1f}" y="15" width="{width:.1f}" height="48" '
            f'fill="{colors[status]}" rx="4" />'
        )
        if width > 70:
            labels.append(
                f'<text x="{x + width / 2:.1f}" y="45" text-anchor="middle" '
                f'fill="white" font-size="15" font-weight="700">{counts[status]}</text>'
            )
        x += width
    return (
        '<svg viewBox="0 0 880 80" role="img" aria-label="Dataset readiness counts">'
        + "".join(rects + labels) + "</svg>"
    )


def _balance_svg(audits: list[Audit]) -> str:
    rows = []
    y = 18
    for audit in audits:
        rate = next((audit.balance[key] for key in (
            "positive_rate", "hallucinated_rate", "error_rate"
        ) if key in audit.balance), None)
        if not isinstance(rate, (int, float)):
            continue
        width = 520 * max(0.0, min(1.0, float(rate)))
        name = html.escape(audit.dataset_id)
        rows.append(f'<text x="0" y="{y + 14}" font-size="12">{name}</text>')
        rows.append(f'<rect x="250" y="{y}" width="520" height="18" rx="3" fill="#e9eef5"/>')
        rows.append(f'<rect x="250" y="{y}" width="{width:.1f}" height="18" rx="3" fill="#386cb0"/>')
        rows.append(f'<text x="780" y="{y + 14}" font-size="12">{100 * rate:.1f}%</text>')
        y += 31
    return (
        f'<svg viewBox="0 0 840 {max(55, y + 5)}" role="img" '
        'aria-label="Observed positive or error rates">' + "".join(rows) + "</svg>"
    )


def build_html(audits: list[Audit], registry: dict[str, Any], markdown: str) -> str:
    status_class = {
        READY: "ready",
        READY_WITH_LIMITATIONS: "limited",
        INCOMPLETE: "incomplete",
        BLOCKED: "blocked",
    }
    table_rows = []
    cards = []
    for audit in audits:
        table_rows.append(
            "<tr>"
            f"<td><code>{html.escape(audit.dataset_id)}</code></td>"
            f"<td>{html.escape(audit.kind)}</td>"
            f"<td>{html.escape(_primary_count(audit))}</td>"
            f"<td>{html.escape(_balance(audit))}</td>"
            f'<td><span class="pill {status_class[audit.status]}">{audit.status}</span></td>'
            "</tr>"
        )
        issues = audit.blockers + audit.limitations + audit.warnings
        issue_html = "".join(f"<li>{html.escape(item)}</li>" for item in issues) or "<li>None.</li>"
        checks = "".join(
            f'<li class="check {"pass" if value else "fail"}">'
            f'{"PASS" if value else "FAIL"}: {html.escape(key)}</li>'
            for key, value in audit.checks.items()
        )
        cards.append(
            f'<article class="card"><div class="card-head"><h3>{html.escape(audit.title)}</h3>'
            f'<span class="pill {status_class[audit.status]}">{audit.status}</span></div>'
            f'<p><code>{html.escape(audit.dataset_id)}</code> · {html.escape(audit.kind)}</p>'
            f'<details><summary>Observed structure</summary><pre>{html.escape(json.dumps(audit.observed, indent=2, sort_keys=True))}</pre></details>'
            f'<h4>Checks</h4><ul class="checks">{checks}</ul>'
            f'<h4>Limitations and blockers</h4><ul>{issue_html}</ul></article>'
        )
    counts = Counter(audit.status for audit in audits)
    hle = next(audit for audit in audits if audit.dataset_id == "hle_qwen72b")
    decision = (
        "HLE is blocked by its placeholder grader. "
        if hle.status == BLOCKED else
        "HLE has complete interim gpt-5.6-sol xhigh labels; they are not paper-faithful GPT-4o labels. "
    ) + (
        "READY and READY_WITH_LIMITATIONS packages may be evaluated now; limited packages must "
        "retain their stated scope in every result. Incomplete competitor caches remain unavailable "
        "until their missing files land."
    )
    use_rows = "".join(
        f"<tr><td><code>{html.escape(audit.dataset_id)}</code></td>"
        f"<td>{html.escape(_evaluation_use(audit))}</td></tr>"
        for audit in audits
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data Readiness Report</title>
<style>
:root{{--ink:#172033;--muted:#5d6878;--line:#dce3ec;--paper:#fff;--bg:#f4f7fb;--blue:#245ca6}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Arial,sans-serif;line-height:1.52}}
main{{max-width:1180px;margin:0 auto;padding:42px 24px 80px}}h1{{font-size:42px;margin:0 0 8px}}h2{{margin-top:42px;color:#174f91}}h3{{margin:0}}h4{{margin-bottom:6px}}
.lede{{font-size:18px;color:var(--muted);max-width:900px}}.panel,.card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:22px;box-shadow:0 4px 18px rgba(23,32,51,.05)}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}.stat{{background:white;border:1px solid var(--line);border-radius:12px;padding:16px}}.stat b{{display:block;font-size:28px}}.stat span{{color:var(--muted);font-size:13px}}
.table-wrap{{overflow:auto;background:white;border:1px solid var(--line);border-radius:14px}}table{{border-collapse:collapse;width:100%;min-width:900px}}th,td{{padding:12px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#eaf1fa;color:#174f91}}
.pill{{display:inline-block;border-radius:999px;padding:4px 9px;font-size:11px;font-weight:800;letter-spacing:.02em}}.ready{{background:#d9f4e7;color:#11613f}}.limited{{background:#fff0bf;color:#7b5700}}.incomplete{{background:#ffe2c9;color:#833900}}.blocked{{background:#ffdadd;color:#871f2d}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.card-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}}pre{{white-space:pre-wrap;background:#f5f7fa;padding:12px;border-radius:8px;max-height:330px;overflow:auto}}
.checks{{list-style:none;padding:0}}.check{{padding:4px 0}}.pass{{color:#176c49}}.fail{{color:#a52638;font-weight:700}}code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}details summary{{cursor:pointer;color:#245ca6}}svg{{width:100%;height:auto}}
.note{{border-left:4px solid #386cb0;padding:10px 14px;background:#edf4fc}}@media(max-width:800px){{.stats,.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:560px){{.stats,.grid{{grid-template-columns:1fr}}h1{{font-size:34px}}}}
</style></head><body><main>
<h1>Data Readiness Report</h1>
<p class="lede">A dataset-only audit completed before any detection, localization, U-PCR, or DUFS evaluation. Raw artifacts were read but not modified.</p>
<p><strong>Registry fingerprint:</strong> <code>{registry['registry_fingerprint']}</code></p>
<div class="stats"><div class="stat"><b>{len(audits)}</b><span>packages inspected</span></div><div class="stat"><b>{counts[READY]}</b><span>ready</span></div><div class="stat"><b>{counts[READY_WITH_LIMITATIONS]}</b><span>ready with limitations</span></div><div class="stat"><b>{counts[INCOMPLETE] + counts[BLOCKED]}</b><span>incomplete or blocked</span></div></div>
<section class="panel"><h2>Readiness distribution</h2>{_status_svg(audits)}</section>
<h2>Readiness matrix</h2><div class="table-wrap"><table><thead><tr><th>Package</th><th>Kind</th><th>Count</th><th>Observed balance</th><th>Status</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<h2>Evaluation use</h2><div class="panel"><p><strong>READY_WITH_LIMITATIONS is usable.</strong> It means the evaluation and its claim must stay inside the documented scope. INCOMPLETE or BLOCKED packages are not usable yet.</p><div class="table-wrap"><table><thead><tr><th>Package</th><th>Allowed evaluation use</th></tr></thead><tbody>{use_rows}</tbody></table></div></div>
<h2>Observed class balance</h2><div class="panel"><p>Bars show the recorded positive, hallucinated, or first-error rate. They describe the data; the audit did not rebalance it.</p>{_balance_svg(audits)}</div>
<h2>Package details</h2><div class="grid">{''.join(cards)}</div>
<h2>Canonical contract</h2><div class="panel"><p>Future consumers address data with stable dataset, record, source, split, task, model, condition, parent, artifact, and row-key fields. Labels remain in a separate sidecar with explicit provenance.</p></div>
<h2>Decision</h2><p class="note">{html.escape(decision)}</p>
<details><summary>Plain-text report source</summary><pre>{html.escape(markdown)}</pre></details>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument(
        "--out", type=Path, default=REPO / "results" / "data_readiness_2026_08_11"
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "validations").mkdir(exist_ok=True)
    local_out = repo / "local_cache" / "data_readiness"

    build_semgrad_review_queue(repo, local_out, out)
    build_hle_grading_queue(repo, local_out, out)
    audits = audit_all(repo)
    registry = registry_payload(repo, audits)
    _json(out / "dataset_registry.json", registry)
    for audit in audits:
        _json(out / "validations" / f"{audit.dataset_id}.json", audit.as_json())
    write_quality_csv(out / "data_quality_summary.csv", audits)
    schema = {
        "schema_version": SCHEMA_VERSION,
        "label_free_unit": [
            "dataset_id", "record_id", "source_id", "split", "task", "model_id",
            "artifact", "row_key", "condition", "parent_id",
        ],
        "label_sidecar": ["dataset_id", "record_id", "label_space", "value", "provenance"],
        "invariant": "Large arrays stay in immutable source artifacts; labels are separate.",
    }
    _json(out / "canonical_data_contract.json", schema)
    markdown = build_markdown(audits, registry)
    (out / "REPORT.md").write_text(markdown, encoding="utf-8")
    (out / "REPORT.html").write_text(
        build_html(audits, registry, markdown), encoding="utf-8"
    )
    print(out / "REPORT.md")
    print(out / "REPORT.html")
    print(json.dumps(registry["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
