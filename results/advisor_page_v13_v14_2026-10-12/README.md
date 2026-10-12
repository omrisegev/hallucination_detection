# Advisor page v13 and v14 (2026-10-12)

Artifact: https://claude.ai/artifact/NQNTYpFqVyx6xRnRS4NENV

| File | Content |
|---|---|
| `published_v13_2026-10-12.html` | Live page v13 (artifact version 1791822103-9f6b). Built from outline v2. |
| `published_v14_2026-10-12.html` | Live page v14 (artifact version 1791824234-383c). Section 3 names Qwen3-8B and adds Qwen3-4B. Section 5 compares plain average, DS-filter, L-SML, IU-PCR, LR and PRM-7B. |
| `published_v15_2026-10-12.html` | Live page v15 (artifact version 1791842634-85c9). Section 5 adds LR (8B, 4B) and Qwen2.5-Math-PRM-7B (0.5 and Q80) on the same external answers. |
| `build_v13.py`, `build_v14.py`, `build_v15.py` | Build scripts. Every number comes from saved result files; asserts tie the text to the data. |
| `FLOW_OUTLINE_2026-10-15.md` | Outline v1. |
| `FLOW_OUTLINE_v2_2026-10-15.md` | Outline v2, approved by Omri on 2026-10-12. |
| `FLOW_OUTLINE_DROPPED_ITEMS_2026-10-12.md` | Items dropped from the outline. |
| `PUBLISHED_CONTEXT_external.json` | Published PRM-7B values for Socratic-PRMBench and Hard2Verify (input of `build_v14.py`). |

## Rebuild

`PF` is a folder with the `qwen3_4b_vs_8b__*.csv` files (see `results/qwen3_4b_vs_8b_2026-10-12/`) and `advisor_page/PUBLISHED_CONTEXT_external.json`.
`PKG` is `results/advisor_completion_2026-10-11` from branch `codex/advisor-v11-completion` @ 07cb7a4.

```
python3 build_v13.py ../advisor_page_v12_2026-10-12/published_v12_2026-10-12.html PKG PF out_v13.html
python3 build_v14.py ../advisor_page_v12_2026-10-12/published_v12_2026-10-12.html PKG PF out_v14.html [external_lr_prm.csv]
python3 build_v15.py ../advisor_page_v12_2026-10-12/published_v12_2026-10-12.html PKG PF out_v15.html external_lr_prm_v1/RESULTS_relayed.csv
```

The optional 5th argument of `build_v14.py` is a CSV with columns
`benchmark, model, method, rule, official_metric` (LR and PRM-7B on the external benchmarks).
Without it, section 5 shows the LR and PRM-7B bars as "running".

## External LR and PRM-7B

`external_lr_prm_v1/RESULTS_relayed.csv` is a 3-decimal copy of the LR and PRM-7B rows in `results/external_lr_prm_v1/RESULTS.csv`
(full results, code and logs; pushed separately from the local checkout).
