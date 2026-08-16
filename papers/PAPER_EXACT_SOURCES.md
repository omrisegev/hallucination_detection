# Exact paper sources for the cluster comparison

This registry pins the primary papers that define the localization and early/online
comparison protocols. SHA-256 values are for the PDFs committed under `papers/`.

The detailed per-paper protocol cards live under `papers/digests/`. Cluster manifests must
record the corresponding PDF hash and, where available, the official-code commit used by the
run. A paper PDF does not by itself resolve parameters omitted by the authors; such runs must
be labelled `paper-specified` rather than `official-exact`.

| Lane | Committed PDF | Primary source | Version | SHA-256 |
|---|---|---|---|---|
| Localization | `ProcessBench Identifying Process Errors in Mathematical Reasoning (arXiv 2412.06559v4).pdf` | https://arxiv.org/pdf/2412.06559 | arXiv v4 | `30a74dc4ed897e077a243b82326df8529c037db0352b0ac35817dd29262ca653` |
| Localization | `Mind the Gap -  Catching Hallucinations via Evidence Drop on the Reasoning.pdf` | https://github.com/QJ0114/evidence-drop | ICML 2026 / PMLR 306 | `7b59671030f87cb61460e76df9ac996df9b004ea8384c2880d1b6d5eee0cc19b` |
| Localization | `Unsupervised Process Reward Models.pdf` | https://arxiv.org/pdf/2605.10158 | arXiv v1 | `8668b273b8f91984c786c46ce907b52534f8c4ad1b28dae360d6ca6bbe9900a8` |
| Multi-trace adaptive compute | `DEEP THINK WITH CONFIDENCE.pdf` | https://arxiv.org/pdf/2508.15260 | arXiv v1 | `ae5eaa9f32263be120ae9e7e4569884b29f97b9a0aaa7f5a0292b108e6946528` |
| Streaming detection | `Streaming Hallucination Detection in Long Chain-of-Thought Reasoning (arXiv 2601.02170v1).pdf` | https://arxiv.org/pdf/2601.02170 | arXiv v1 | `1c4869a9b7ba8dac9ced600fccab3da30139667506e3f80efc78c1ab4afd8b7a` |
| Single-trace stopping | `Stop When Enough Adaptive Early-Stopping for Chain-of-Thought Reasoning (ACL 2026).pdf` | https://aclanthology.org/2026.acl-long.1256.pdf | ACL 2026 | `e29b9a4e453e4b4561e6d278cd03bb1b361a7f6dc158609021f8920e2f775a0b` |
| Single-trace stopping | `LEASH Logit-Entropy Adaptive Stopping Heuristic for Efficient Chain-of-Thought Reasoning (arXiv 2511.04654v1).pdf` | https://arxiv.org/pdf/2511.04654 | arXiv v1 | `36f7d96bec68a33f25849c08f9846ef6157bc6697607277a048ddcbf2596a0f1` |

## Official-code status at handoff

- ProcessBench: https://github.com/QwenLM/ProcessBench
- Mind the Gap: https://github.com/QJ0114/evidence-drop
- DeepConf: https://github.com/facebookresearch/deepconf
- REFRAIN: https://github.com/RLSNLP/Adaptive-Reasoning (release placeholder when audited)
- Streaming Hallucination Detection: the anonymous code URL printed by the paper was
  unreachable when audited; do not substitute a different dataset or labeler silently.
- uPRM and LEASH: no audited official runnable implementation is pinned in this repository.

## Integrity

Run `shasum -a 256` over the seven PDFs before syncing code to AIRCC and copy the resulting
hashes into the acquisition manifest. Never fetch a floating PDF or repository branch from
inside a full experiment job.
