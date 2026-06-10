# baselines/

This directory holds external baseline repositories cloned locally for code inspection.
The repos are **not committed** to this repo (see `.gitignore`).

## External repos

| Repo | Paper | Purpose |
|------|-------|---------|
| `lapeigvals/` | Binkowski et al., EMNLP 2025 | White-box spectral method (attn. maps). Phase 7 already ran their unsupervised baseline on GSM8K. |
| `losnet/` | Bar-Shira et al., AAAI 2026 | Supervised GNN on Token Distribution Sequences. Reference comparison for RAG domain (paper number used; different task). |

## Implemented baselines (in `spectral_utils/baselines.py`)

These are production-ready implementations used in the Phase 12 benchmarking notebook:

| Method | Access | Paper | Function |
|--------|--------|-------|----------|
| Official Semantic Entropy | Gray-box + sampling | Farquhar et al., Nature 2024 | `official_semantic_entropy()` |
| Self-Consistency | Black-box + sampling | Wang et al., ICLR 2023 | `self_consistency_score()` |
| SelfCheckGPT (NLI) | Black-box + sampling | Manakul et al., EMNLP 2023 | `selfcheck_nli_score()` |
| Verbalized Confidence | Black-box, 1-pass | AUQ / Zhang et al. 2026 | `parse_verbalized_confidence()` |
| Lite Semantic Entropy | Gray-box + sampling | Kuhn et al., 2023 (Jaccard approx) | `lite_semantic_entropy_for_statement()` |

## NLI model

All NLI-based methods use `cross-encoder/nli-deberta-v3-base` by default.
Load with: `nli_model, nli_tok, device = nli_load_model(device='cuda')`

## On Colab

The external repos are NOT cloned on Colab. The `spectral_utils/baselines.py`
implementations are self-contained and require only:
```
pip install transformers accelerate
```
(DeBERTa model weights are ~180 MB, downloaded from HuggingFace Hub.)
