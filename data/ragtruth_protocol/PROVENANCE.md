# RAGTruth corpus — provenance

Vendored on 2026-08-09 for the Evidence-Contrast U-PCR/DUFS-LIU campaign (see
`cluster/manifests/ragtruth_ec_v1.json`).

## Source

- Paper: "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented
  Language Models", Niu, Wu, Zhu, Xu, Shum, Zhong, Song, Zhang. ACL 2024.
  [aclanthology.org/2024.acl-long.585](https://aclanthology.org/2024.acl-long.585/)
- Official code + data: [github.com/ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth),
  MIT license (`LICENSE`, copied verbatim from the repo root).
- **Pinned commit**: `c103204b9ce28d6bbad859304bf30de72b8ed8fe` (`main` branch HEAD at fetch
  time; the repo has been unchanged since 2024-12-02). Files fetched via
  `raw.githubusercontent.com` at this commit.

## Files

| File | Source path | Rows | sha256 |
|---|---|---:|---|
| `response.jsonl` | `dataset/response.jsonl` | 17,790 | `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073` |
| `source_info.jsonl` | `dataset/source_info.jsonl` | 2,965 | `0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b` |
| `LICENSE` | repo root | — | MIT |

Re-fetch at the pinned commit to reproduce byte-identical files.

## Schema

**`response.jsonl`** (one row per generated response): `id` (response id, str),
`source_id` (str, joins to `source_info.jsonl`), `model`, `temperature`, `split`
(`"train"`/`"test"`), `quality` (`"good"` / `"incorrect_refusal"` / `"truncated"`),
`response` (str), `labels` (list of `{start, end, text, label_type, meta,
implicit_true, due_to_null}` — `start`/`end` are **character offsets into `response`**,
verified exact: `response[start:end] == text` holds for every label in the corpus).
`label_type` ∈ `{Evident Baseless Info, Evident Conflict, Subtle Baseless Info, Subtle
Conflict}`. A response is hallucinated iff `labels` is non-empty.

**`source_info.jsonl`** (one row per source item, joined by `source_id`): `task_type`
(`"QA"` / `"Summary"` / `"Data2txt"`), `source` (`MARCO` / `Yelp` / `CNN/DM` / `Recent
News`), `source_info` (task-type-dependent: a `{question, passages}` dict for QA — the
`passages` string contains three `"passage N:"`-delimited blocks; a plain document string
for Summary; a business-record dict for Data2txt), and **`prompt`** — the exact original
generation prompt, which always contains the evidence text (`source_info["passages"]` /
`source_info` / `str(source_info)`) as a verbatim substring. `spectral_utils/ragtruth.py`
builds every evidence condition by editing this substring in place inside `prompt`, never
by reconstructing the surrounding instruction text.

## Row counts (measured directly from the vendored files)

| Split | Responses | Hallucinated |
|---|---:|---:|
| train | 15,090 | 6,721 |
| test | 2,700 | 943 |

Models (2,965 responses each, both splits): `gpt-4-0613`, `gpt-3.5-turbo-0613`,
`mistral-7B-instruct`, `llama-2-7b-chat`, `llama-2-13b-chat`, `llama-2-70b-chat`.
`task_type` by **source** (2,965 total): Data2txt 1,033 (Yelp), QA 989 (MARCO), Summary
943 (628 CNN/DM + 315 Recent News). `quality`: good 17,617 / incorrect_refusal 144 /
truncated 29.

## Chunking coverage (measured on the full test split, 2,700/2,700 rows load cleanly,
zero span-mismatch or missing-source drops)

| Task type | Chunk unit | Chunks/row | Distinct conditions (`full`+`noctx`+LOO) |
|---|---|---:|---:|
| QA | one of 3 `"passage N:"` blocks | 3 (always) | 5 |
| Data2txt | one of 9 top-level JSON fields | 9 (always) | 11 |
| Summary | paragraph split, ≥200 chars, ≤8 chunks | **1 (always)** | **2** |

**Summary documents have zero `"\n\n"` breaks in every single row of the corpus** — they
are single-block news-article prose, not multi-paragraph text. This is not a rare edge
case; it is the universal case for this task type in RAGTruth. The preregistered fallback
(`docs/research_notes/ragtruth_ec_preregistration_v1.md`) therefore applies to all 900
Summary rows in the test split: they get `{full, noctx}` only, no leave-one-chunk-out
condition — a real coverage limitation for the evidence-graph/LOO-based arms on 1 of the
3 task types, not a bug. `spectral_utils.ragtruth.distinct_conditions` detects and skips
the resulting `loo_0 == noctx` duplicate automatically so the driver never wastes a
teacher-forced pass reproducing an already-scored prompt.

Total teacher-forced passes for the full test split (all task types, all distinct
conditions): **16,200** (mean 6.0 passes/response).
