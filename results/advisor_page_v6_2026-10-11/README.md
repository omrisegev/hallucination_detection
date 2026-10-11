# advisor_page_v6_2026-10-11: official PRMScore and advisor page after the think-block fix

This folder holds the official PRMScore run and the generator for version 6 of the advisor results page
(https://claude.ai/artifact/NQNTYpFqVyx6xRnRS4NENV, meeting 2026-10-15). All inputs come from
`results/thinkfix_rescore_v1` on this branch (LR from commit 8810e49), column / variant `thinkfix_mask1`.

## Files

| file | what it is |
|---|---|
| `official_prmscore_v1.py` | Official-format PRMScore. Label-free flag rules R0, R1, R2, R1F, TOP1 and TOP2 feed the repository port of the PRMBench evaluator (`spectral_utils/prmbench.py::prmbench_evaluate`). The docstring defines each rule. |
| `official_prmscore_thinkfix_mask1/OFFICIAL_PRMSCORE_<bank>.csv/.json` | Output for the 10 banks. There is one row per method × rule, with the overall, 3 category and 9 subcategory scores, plus flag statistics. A PRM-7B reference row (native 0.5) is at the end. |
| `gen_prmb_v2.py`, `template_prmb_v2.html` | Page generator and page template. |
| `prmbench_page_v2.html`, `.numbers.json` | The published page (version 6) and the headline numbers in it. |

## Reproduce

```
python official_prmscore_v1.py --repo <repo> --smoke
for B in N11 N13 N14 N16 N22 N25 N40 N43 N53 N56; do
  python official_prmscore_v1.py --repo <repo> --bank $B \
    --scores <repo>/results/thinkfix_rescore_v1/prmscore_inputs/bank_$B.npz \
    --meta <repo>/results/thinkfix_rescore_v1/prmscore_inputs/prmbench_meta.json \
    --prm-manifest <repo>/dataset_cache/four_localization/prmbench_qwen25math7b_full/manifest.json \
    --out official_prmscore_thinkfix_mask1
done
python gen_prmb_v2.py <repo> official_prmscore_thinkfix_mask1 prmbench_page_v2.html
```

## Checks

- On N16, TOP1 with MEAN_MOM_ga gives 0.6481. This equals `prmscore_top1` in `PRMBENCH_COMPARISON.csv`.
- On N16, R0 with MEAN_MOM_ga gives 0.6523. This equals `prmscore_own`.

## Key result (N16)

| rule | official PRMScore |
|---|---|
| R0 | 0.647–0.654 |
| R1 | 0.653 (method-independent) |
| R2 | about R0, no gain |
| TOP2 | 0.675–0.680 (a construction-prior control) |
| PRM-7B at 0.5 | 0.655 |

By category, MEAN_MOM_ga with R0 against PRM-7B:

| category | MEAN_MOM_ga, R0 | PRM-7B |
|---|---|---|
| Simplicity | 0.689 | 0.520 |
| Soundness | 0.666 | 0.710 |
| Sensitivity | 0.599 | 0.633 |

## Limits

- `G` is re-built from its written definition, so R1 and R2 are not a replay of `decision_rule_v1`.
- R1F is not run, because the export has no `G__<method>` arrays.
- redundency and circular use the validity fallback.
- multi_solutions is not in the Sensitivity mean.
- The page includes the supervised LR re-score (8810e49). The external benchmarks are not re-scored yet.
