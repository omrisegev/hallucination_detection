---
name: gemini-deep-research
description: Perform iterative, multi-step web research to synthesize information from multiple sources. Use when a query requires deep investigation, comparison of different viewpoints, or aggregating data from various websites.
---

# Gemini Deep Research Skill

This skill provides a systematic workflow for performing deep research using Gemini CLI's native web tools.

## Core Workflow

1.  **Breakdown**: Deconstruct the research query into 3-5 specific sub-questions or search targets.
2.  **Iterative Search**:
    - Use `google_web_search` for broad discovery.
    - Use `web_fetch` on high-signal URLs (especially GitHub, arXiv, or documentation sites).
3.  **Synthesis**: Aggregate findings, noting contradictions or consensus across sources.
4.  **Reporting**: Produce a structured report (Markdown or JSON) with citations.

## Guidelines

- **Source Diversity**: Ensure you fetch from at least 3 distinct domains.
- **Depth**: If a search result is a GitHub repository, always `web_fetch` the `README.md`.
- **Truth Discovery**: If sources conflict, highlight the discrepancy and look for a third-party tie-breaker.

## Resources

- [WORKFLOW.md](references/WORKFLOW.md): Detailed multi-turn research patterns.
- [TEMPLATES.md](references/TEMPLATES.md): Output formats for research reports.
