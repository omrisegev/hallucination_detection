# Unified Causal versus classic Global IU-PCR v1

This is the access-matched comparison that was missing from the feature-subset cycle. The classic registered mixed-v2 contract is refit without final response length, so both methods are causal with respect to length. The primary comparison here is Global; companion live Local/Early incumbents are reported below.

## Repeated Qwen3-8B development splits

- base7_full28: 0.6914
- classic_mixed_v2_no_length: 0.6543
- delta: +0.0371

Both methods use the exact 3 x 3 source-question-grouped split schedule. The classic head is refit separately inside each fit family and sees no labels.

## Frozen Qwen-to-Llama scorer transfer

- base7_full28: 0.6629
- classic_mixed_v2_no_length: 0.6870
- delta: -0.0241 [95% CI -0.0407, -0.0070]

After excluding the 128 source questions used to fit the Qwen heads, the question-disjoint panel has 3272 questions. Its delta is -0.0276 [-0.0452, -0.0109].

The classic head is fit on the same 32 selected Qwen questions per family and then frozen before all 3,400 Llama questions are scored. Validation labels affect only the metric and paired bootstrap.

## Taskwise live incumbents on the same frozen transfer

| task | base7_full28 | incumbent | incumbent score | delta [95% CI] |
|---|---:|---|---:|---:|
| Global | 0.6629 | classic mixed-v2, no length | 0.6870 | -0.0241 [-0.0407, -0.0070] |
| Localization | 0.2880 | max entropy + top-5 step | 0.2419 | +0.0461 [+0.0228, +0.0691] |
| Early | 0.5587 | max entropy | 0.5777 | -0.0189 [-0.0366, -0.0005] |

Against the historical IU28-without-length Early control alone, base7 is +0.0213 [+0.0131, +0.0289]; max entropy is nevertheless the stronger Early incumbent in this exact transfer protocol.

## Interpretation

A compact unified causal head improves Localization relative to the live baseline, but it does not clear the frozen noninferiority margins against the strongest Global and Early incumbents. It must not be promoted as one replacement for all three heads. The exact historical 30-coordinate Global method also contains final length and therefore has strictly greater end-of-trace access.

Source subset run: `/private/tmp/hallucination-unified-causal-iu-v1/results/unified_causal_subset_search_compact_v1`.
Retrospective opened-data comparison; not untouched confirmation.
