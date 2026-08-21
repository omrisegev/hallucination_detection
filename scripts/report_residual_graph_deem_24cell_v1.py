#!/usr/bin/env python3
"""Render the compact scientific report and reviewer recomputation guide."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import atomic_write_json, canonical_sha256, sha256_file


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value, digits=4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--phase0-complete", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluation = args.evaluation_dir.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    decision = json.loads((evaluation / "DECISION.json").read_text(encoding="utf-8"))
    rebuild_path = args.run_dir.resolve().parent / "rebuild" / "REBUILD_VERIFICATION.json"
    rebuild = json.loads(rebuild_path.read_text(encoding="utf-8")) if rebuild_path.is_file() else None
    primary_decision = (
        rebuild.get("primary_decision", decision["primary_decision"])
        if rebuild is not None else decision["primary_decision"]
    )
    summaries = read_csv(evaluation / "FAMILY_SUMMARY.csv")
    comparisons = read_csv(evaluation / "PAIRWISE_COMPARISONS.csv")
    null = json.loads((evaluation / "WHOLE_SEARCH_NULL.json").read_text(encoding="utf-8"))
    lookup = {(row["method"], row["metric"]): row for row in summaries}
    lines = [
        "# Residual-Graph DEEM — 24-cell Phase 1 report", "",
        f"**Primary decision:** `{primary_decision}`", "",
        f"- `ADVANCE_CORE={str(decision['advance_core']).lower()}`",
        f"- `ADVANCE_GRAPH={str(decision['advance_graph']).lower()}`",
        f"- whole-search null draws: `{null['B']}`", "",
        "## Headline equal-family results", "",
        "| arm | AUROC | AUPRC | QA AUROC | math AUROC | worst cell |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("B0", "B1", "B2", "B3", "G0", "G1", "G2", "G3", "G4", "G5"):
        auc = lookup[(method, "auroc")]
        ap = lookup[(method, "auprc")]
        lines.append(
            f"| {method} | {fmt(auc['equal_family_macro'])} | "
            f"{fmt(ap['equal_family_macro'])} | {fmt(auc['qa_macro'])} | "
            f"{fmt(auc['math_macro'])} | {fmt(auc['worst_cell'])} |"
        )
    lines.extend([
        "", "## Frozen gates", "",
        "| gate | pass |", "|---|---:|",
        f"| A — stable continuous DEEM | {decision['gate_a']} |",
        f"| B — G3 utility | {decision['gate_b_g3']} |",
        f"| B — G4 utility | {decision['gate_b_g4']} |",
        f"| C — residual/DUFS specificity | {decision['gate_c']} |",
        f"| D — nuisance separation | {decision['gate_d']} |",
        "", "## Multiplicity-controlled conditional null", "",
    ])
    for name, value in null["p_values"].items():
        lines.append(f"- `{name}` max-statistic p-value: {fmt(value, 5)}")
    if rebuild is not None:
        lines.extend([
            "", "## Rebuild verification", "",
            f"- status: `{rebuild['status']}`",
            f"- evidence: `{rebuild_path}`", "",
        ])
    lines.extend([
        "", "## Historical context", "",
        "The historical variable-inventory study reported full IU-PCR AUROC "
        "`0.7541776` and hard-DEEM AUROC `0.7543620`. Those results use a "
        "different input contract and are context only; no score was recycled here. "
        "The historical soft-DEEM lane collapsed, while its repair pilot covered "
        "only three cells (15/15 healthy fits).", "",
        "## Claim boundary", "",
        "This report covers only Global Phase 1 on the original 24 cells. It does "
        "not validate Localization, Early Detection, a universal hallucination "
        "manifold, or the categorical DEEM theorem for the continuous-visible adaptation.", "",
        "## Figures", "",
    ])
    for index, title in enumerate((
        "Architecture", "Per-cell score map", "Paired-change forest",
        "Residual graph atlas", "Neighbor composition", "Raw versus residual",
        "DUFS gate heatmap", "Target versus nuisance actuation",
        "Linear versus graph", "Control dashboard", "Lambda paths", "Seed stability",
    ), 1):
        filename = next(out.glob(f"{index:02d}_*.png"), None)
        lines.append(f"{index}. **{title}:** `{filename.name if filename else 'missing'}`")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    guide = f"""# Reviewer guide — Residual-Graph DEEM 24-cell v1

The immutable Stage-A score manifest is `{args.run_dir.resolve() / 'SCORE_FREEZE_MANIFEST.json'}`.
Natural labels exist only in `{args.sidecar_dir.resolve()}` and are opened by the evaluator/plotter.

## Recompute the evaluation

```bash
{sys.executable} {ROOT / 'scripts/evaluate_residual_graph_deem_24cell_v1.py'} \\
  --run-dir {args.run_dir.resolve()} --bundle-dir {args.bundle_dir.resolve()} \\
  --sidecar-dir {args.sidecar_dir.resolve()} --phase0-complete {args.phase0_complete.resolve()} \\
  --out-dir {evaluation} --B {null['B']}
```

## Recreate figures and report

```bash
{sys.executable} {ROOT / 'scripts/plot_residual_graph_deem_24cell_v1.py'} \\
  --run-dir {args.run_dir.resolve()} --evaluation-dir {evaluation} \\
  --bundle-dir {args.bundle_dir.resolve()} --sidecar-dir {args.sidecar_dir.resolve()} \\
  --phase0-dir {args.phase0_complete.resolve().parent} --out-dir {out}
{sys.executable} {ROOT / 'scripts/report_residual_graph_deem_24cell_v1.py'} \\
  --run-dir {args.run_dir.resolve()} --evaluation-dir {evaluation} \\
  --bundle-dir {args.bundle_dir.resolve()} --sidecar-dir {args.sidecar_dir.resolve()} \\
  --phase0-complete {args.phase0_complete.resolve()} --out-dir {out}
```

## Boundaries

- Phase 2/3, Localization, and Early Detection are out of scope.
- B1/B2 are packaged `deem==0.2.0` adapter controls, not paper-exact claims.
- B3/G0–G5 are continuous-visible adaptations with no DEEM theorem claim.
- No pooled row-level AUROC is reported; inference is family-blocked.
"""
    (out / "REVIEWER_GUIDE.md").write_text(guide, encoding="utf-8")
    manifest = {
        "schema": "residual_graph_deem_report_complete_v1",
        "decision_sha256": sha256_file(evaluation / "DECISION.json"),
        "rebuild_verification_sha256": (
            sha256_file(rebuild_path) if rebuild_path.is_file() else None
        ),
        "report_sha256": sha256_file(out / "REPORT.md"),
        "reviewer_guide_sha256": sha256_file(out / "REVIEWER_GUIDE.md"),
        "pairwise_rows": len(comparisons),
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_write_json(out / "REPORT_COMPLETE.json", manifest)


if __name__ == "__main__":
    main()
