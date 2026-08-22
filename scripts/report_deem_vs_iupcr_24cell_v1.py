#!/usr/bin/env python3
"""Build the compact reviewer-facing report for the B0-B3 benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import atomic_write_json, canonical_sha256, sha256_file  # noqa: E402


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    decision = json.loads((args.evaluation_dir / "DECISION.json").read_text(encoding="utf-8"))
    summaries = json.loads((args.evaluation_dir / "FAMILY_SUMMARY.json").read_text(encoding="utf-8"))
    comparisons = rows(args.evaluation_dir / "PAIRWISE_COMPARISONS.csv")
    macro = {(row["arm_id"], row["metric"]): row for row in summaries}
    lines = [
        "# DEEM vs IU-PCR — 24-cell frozen benchmark", "",
        f"**Decision:** `{decision['decision']}`", "",
        "The previous residual-graph hypothesis was closed because Phase 0 failed the frozen specificity gate. "
        "That result does not falsify graph-free continuous B3. This report evaluates only B0–B3.", "",
        "## Equal-family results", "",
        "| arm | AUROC | AUPRC | QA AUROC | math AUROC |", "|---|---:|---:|---:|---:|",
    ]
    names = {"B0": "IU-PCR", "B1": "hard DEEM adapter", "B2": "repaired soft/rank DEEM", "B3": "continuous additive DEEM"}
    for arm in ("B0", "B1", "B2", "B3"):
        auroc, auprc = macro[(arm, "auroc")], macro[(arm, "auprc")]
        lines.append(f"| {arm} — {names[arm]} | {auroc['equal_family_macro']:.6f} | {auprc['equal_family_macro']:.6f} | {auroc['qa_macro']:.6f} | {auroc['math_macro']:.6f} |")
    lines.extend(["", "## Preregistered B3 contrasts", "",
                  "| contrast | Δ AUROC | 95% interval | Holm p | W/T/L |", "|---|---:|---:|---:|---:|"])
    for row in comparisons:
        lines.append(f"| B3−{row['reference']} | {float(row['equal_family_auroc_delta']):+.6f} | [{float(row['lower']):+.6f}, {float(row['upper']):+.6f}] | {float(row['holm_p']):.4f} | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines.extend(["", "## Interpretation boundary", "",
                  "B1/B2 are pinned 0.2.0 adapter controls, not paper-exact DEEM. B3 is a continuous-visible adaptation. "
                  "No graph arm, graph hyperparameter, Localization result, or Early Detection result is part of this benchmark."])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = args.out_dir / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    guide = args.out_dir / "REVIEWER_GUIDE.md"
    guide.write_text(
        "# Reviewer guide\n\nStart with `DECISION.json`, then `PAIRWISE_COMPARISONS.csv`, "
        "`PER_CELL_METRICS.csv`, `SEED_STABILITY.csv`, and `WHOLE_SEARCH_NULL.json`. "
        "Verify `SCORE_FREEZE_MANIFEST.json` predates the label sidecars. The active arm roster is exactly B0/B1/B2/B3.\n",
        encoding="utf-8",
    )
    manifest = {"schema": "deem_vs_iupcr_report_manifest_v1",
                "evaluation_complete_sha256": sha256_file(args.evaluation_dir / "EVALUATION_COMPLETE.json"),
                "files": {path.name: sha256_file(path) for path in (report, guide)}}
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_write_json(args.out_dir / "REPORT_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
