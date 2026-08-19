# S0 competitor baseline

**Verdict: `MECHANICS_ONLY_NO_PERFORMANCE_CLAIM`.**

The numbers below establish reporting bars; incompatible scopes are not subtracted.

| task | method | value | tier | scope |
|---|---|---:|---|---|
| local | Mind the Gap | 0.2496 | A | ProcessBench full four-subset macro |
| local | GL-LIU v1 | 0.3125 | A | Qwen ProcessBench macro |
| local | maximum token entropy | 0.3150 | A | Llama ProcessBench macro |
| local | Step-272 two-head | 0.3136 | A | 12 scorer/family-cell macro |
| local | broad-28 DUFS | 0.2903 | A | 8-cell historical macro |
| local | Qwen2.5-72B critic | 0.5940 | B | same 3,400 ProcessBench rows |
| local | Qwen2.5-Math-PRM-7B | 0.7294 | B | same 3,400 ProcessBench rows |
| local | Qwen3-8B judge control | 0.0964 | B | same 3,400 ProcessBench rows |
| online | IU28 | 0.6534 | A | historical 11-cell equal-family AUROC@64/128 |
| online | DeepConf-w64 | 0.6343 | A | historical 11-cell equal-family AUROC@64/128 |
| online | Step-272 two-head | 0.6075 | A | 12 scorer/family-cell macro AUROC@64/128 |
| online | Streaming supervised probe | 0.8110 | C | published Qwen, different data/labels/hidden-state access |

Tier A is the only same-access improvement tier. Tier B uses the same ProcessBench rows but substantially different compute. Tier C is cross-protocol context only.
