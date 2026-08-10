# SemGrad protocol assets — provenance

Vendored on 2026-08-07 for the SemGrad pilot (see `cluster/manifests/semgrad_pilot_v1.json`).

## Source

- Paper: "Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of
  Large Language Models" (SemGrad / HybridGrad), Mingda Li et al., ICML 2026.
  arXiv 2605.04638 ([HTML v2](https://arxiv.org/html/2605.04638v2)).
- Official code: [github.com/mingdali6717/SemGrad](https://github.com/mingdali6717/SemGrad), MIT
  license (`LICENSE`, copied verbatim below).
- **Pinned commit**: `118b6949f9641df3872caa7ad65a797f4ae28d63` (`main` branch HEAD at fetch
  time, 2026-08-07). All files in this directory were fetched from this exact commit via
  `raw.githubusercontent.com` — re-fetch at this hash to reproduce byte-identical files.

## Files

| File | Source path in SemGrad repo | Verified |
|---|---|---|
| `sciq_test.jsonl` | `data/datasets/sciq/test.jsonl` | 1000 lines, matches paper §4.1 / Appendix C.3 exactly |
| `truthfulqa_test.jsonl` | `data/datasets/truthfulqa/test.jsonl` | 817 lines, matches paper exactly |
| `vocab.txt` | `uncertainty/generation_evaluation/metrics/vocab.txt` | BERT wordpiece vocab used by the BEM tokenizer — must be used as-is, not substituted with a generic BERT vocab |
| `official_bem_reference.py` | `uncertainty/generation_evaluation/metrics/bem.py` | Kept verbatim as the ground-truth reference for `spectral_utils/bem_scorer.py`'s reimplementation |
| `LICENSE` | repo root | MIT |

Schema (both jsonl files): `{"query": str, "truthful answer": [str, ...]}`. SciQ answers are
single lowercase words/short phrases; TruthfulQA answers are full sentences (from HF
`truthful_qa`/`generation` config's `correct_answers`).

## Prompt template (verified from `uncertainty/response_generator/generator.py`)

Single template for every dataset, `template_id: 2` (used by both `sciq_config.yaml` and
`truthfulqa_config.yaml`), **no system message** (`system_id:` is empty in both configs →
`use_system_message = False`):

```
Please directly answer the following question with one or few words:
{query}
```

Rendered via `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`
with a single user turn — no system turn.

## Model

`Qwen/Qwen3-4B-Instruct-2507`. The official repo's own config
(`uncertainty/utils/llm.py`, key `qwen3-4b-instruct`) has a literal bug —
`"model_path": "/Qwen/Qwen3-4B-Instruct-2507"` (leading slash) — which will not resolve as an HF
Hub id. The intended id has no leading slash; confirmed by the sibling `qwen3-30b-instruct` entry
in the same file, which has no such typo. No upstream revision/commit pin exists for this model
in the SemGrad repo or paper — record whatever revision resolves at download time in the
manifest.

## Decoding (verified from `config/{sciq,truthfulqa}_config.yaml`, byte-identical generation
block in both)

```yaml
generate_kwargs:
  do_sample: False        # greedy — temperature/top_p/top_k below are inert under do_sample=False
  temperature: 1.0
  top_p: 0.9
  top_k: 50
  max_new_tokens: 150
  num_responses_per_prompt: 1
```
Seed `42` (`run_generation.py`). `do_sample: False` is enforced unconditionally by
`run_generation.py`'s CLI wrapper regardless of the YAML value.

In this repo: `temps=[0.0]` in a `cluster/presets.py` preset reproduces this exactly —
`spectral_utils/model_utils.py:429`'s `sampling = temp > 1e-4` already gives `do_sample=False`
at `temp=0.0`, no code change needed.

## Correctness grading: BEM (Bulian et al. 2022, "Answer Equivalence")

Paper, Appendix D.1: *"We choose BEM (Bulian et al., 2022) as the primary correctness evaluation
metric..."* Verified from `official_bem_reference.py` + `uncertainty/generation_evaluation/utils.py`
+ `uncertainty/generation_evaluation/__init__.py` (not vendored, source read during research —
logic reproduced here in full):

- **Checkpoint**: originally `https://tfhub.dev/google/answer_equivalence/bem/1` (TF Hub) — this
  URL is now dead (TF Hub sunset). Live copy: Kaggle Models,
  `kagglehub.model_download('google/bert/tensorFlow2/answer-equivalence-bem')`.
- **Tokenization**: `text.BertTokenizer(vocab_lookup_table=<this vocab.txt>, token_out_type=tf.int64, preserve_unused_token=True, lower_case=True)`.
- **Segment order**: `(candidate, reference, question)` — candidate first, then reference, then
  question — combined via `text.combine_segments(..., cls_id, sep_id)`. This order is load-bearing;
  do not reorder.
- **Truncation**: `MAX_LENGTH = 512`, `max_len = MAX_LENGTH - 4 = 508` (reserves 4 for
  `[CLS]`+3×`[SEP]`... actually 1×`[CLS]`+2×`[SEP]` structurally, but the code reserves 4 flatly).
  If `len(q) + len(r) + len(c) > max_len`, the **candidate is truncated from the end** by the
  overflow amount (`c[:-(overflow)]`) — question and reference are never truncated.
- **Empty candidate**: if the generated candidate string is `""`, it is replaced with the literal
  string `"None"` before scoring (`uncertainty/generation_evaluation/__init__.py`).
- **Score**: raw model output logits → `softmax(logits, axis=1)[:, 1]` — the positive-class
  ("equivalent") probability.
- **Multi-reference aggregation**: **max** over all reference answers for a given question — a
  candidate is scored against every reference in the answer list, and the sample's BEM score is
  the maximum across references.
- **Threshold = 0.8**, not 0.5. This value exists **only** in
  `uncertainty/generation_evaluation/utils.py`'s `compute_score(threshold=0.8)` default argument —
  it is never stated anywhere in the paper text. Using the more common 0.5 threshold will silently
  produce different label base rates and non-reproducible AUROC.
- **GPU requirement**: `BemCalculator.__init__` raises `RuntimeError("No GPU devices are
  available.")` if TF sees no GPU. BEM only ever consumes already-generated
  `(question, reference, candidate)` text triples — it never touches the LLM — so there is no need
  to run it on the B200/NGC cluster container. Run it locally instead (this sidesteps getting
  `tensorflow` + `tensorflow_hub` + `tensorflow_text` working against sm_100 inside the PyTorch
  NGC image, a real and unnecessary risk). `spectral_utils/bem_scorer.py` reimplements the
  tokenization/segment-packing in pure Python + `transformers.BertTokenizer` and calls the
  downloaded SavedModel directly via `tf.saved_model.load()` — it never instantiates the
  official `BemCalculator` class, so its hard GPU-required check is never hit at all.

## Baseline sampling note (for the pre-registered LN-PE / G-NLL competitor pair)

SemGrad's own published LN-PE / Semantic-Entropy / SAR baselines use 10 (or 5) stochastic
generations at T=1.0 — not the greedy output. Our LN-PE/G-NLL, computed from the single greedy
generation's saved logprobs (`token_logsumexp`, `top_k_logprobs`), is a **sampling-free
adaptation**, not a paired reproduction of SemGrad's cited baseline numbers. Label this
Comparison Level 3 ("same dataset only") per `external_data_collection_plan_2026.md`'s comparison
levels — never present it as reproducing SemGrad's own LN-PE score.
