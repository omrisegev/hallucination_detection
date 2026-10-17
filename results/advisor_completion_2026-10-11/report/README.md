# Advisor v11 report builder

`build_v11.py` integrates the bank matrices, nine-category leaderboard/radar,
all-bank top1, exact decisions, nuisance controls, the source-group LR diagnostic, conditional diagnostics and
internal validation into one standalone HTML build. It does not patch a previous
HTML file. Historical sources and values remain in `legacy/` for comparison.

The original category illustration is `prmbench_overview.svg`. All radar axes
start at zero; the series are frozen N16 Continuous L-SML and IU-PCR under Q80,
and native Qwen2.5-Math-PRM-7B at reward < 0.5. The MS axis is valid-step F1,
and Sensitivity includes PS, DR and MS throughout. Scientific bank plots use
Matplotlib and are saved as both SVG and PNG.

`audit_literature.py` parses paper v1 Table 3 and the archived official current
leaderboard JSON. Published category means are preserved. The universal claim
about all roughly forty rows is withdrawn: the primary sources contain both
counterexamples and different populations. Current leaderboard values are not
substituted for the fixed paper reference table.

`audit_provenance.py` separates the protocol's run-directory date from versioned
committer timestamps. The earliest protocol commit is dated before the
corrected-input export commit. Original public push time, private registration
and absence of historical selection cannot be independently established. The repository is public; the new local
HTML has not been pushed or published to Claude.

Rebuild with the repository environment, without installing packages:

```sh
.venv/bin/python results/advisor_completion_2026-10-11/report/audit_literature.py
.venv/bin/python results/advisor_completion_2026-10-11/report/audit_provenance.py
.venv/bin/python results/advisor_completion_2026-10-11/report/build_v11.py --final
.venv/bin/python results/advisor_completion_2026-10-11/report/validate_html.py
```

`--final` requires `report_data.json` in the package root, containing every
Claude task ID in `task_status`. The parent analysis can add extra sections:

```json
{
  "sections": [{
    "id": "lr-diagnostics",
    "title": "Feature selection versus weighting",
    "summary": "A finding grounded in the saved outputs.",
    "source": "lr/SUMMARY.json",
    "n": "Coverage and population",
    "tables": [{"columns": ["Method", "Value"], "rows": [["Example", "0.123"]]}],
    "figures": [{"path": "lr/weights.svg", "caption": "Coverage and metric."}]
  }],
  "task_status": [{"id": "7", "previous_state": "Historical globally selected restrictions.", "action": "Training-only selection rerun.", "status": "completed", "finding": "Finding and limits.", "evidence": ["lr/SUMMARY.json"]}]
}
```

Native adapters read `controls/`, `statistics/`, `validation/` and `lr/` output schemas.
A supplied section with the same ID intentionally overrides a native adapter.
Every figure is embedded, so no network or JavaScript library is required to
read the HTML. Raw evidence links remain relative to the complete output folder.
`BUILD_RECEIPT.json` pins source hashes and distinguishes DRAFT from FINAL.
`HTML_VALIDATION.json` checks duplicate SVG IDs, local evidence links, section
anchors, embedded runtime dependencies and required visible disclosures. This
validates the presentation interface; separate analysis audits verify arithmetic.
The N11 failed L-SML / LR-half arm remains undefined in the page. Synthetic
clean controls (758) and all no-in-range-error answers (934) have separate
false-flag denominators; these populations are not interchanged.
