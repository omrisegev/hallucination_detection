# Hallucination Detection — Spectral Entropy Pipeline

Thesis project on hallucination detection in LLMs using the spectral structure of
token-level entropy traces H(n).

**Author:** Omri Segev  
**Advisors:** Ofir Lindenbaum, Bracha Laufer-Goldshtein

---

## Core idea

When a language model generates a response, each token is produced with a probability
distribution over the vocabulary. The entropy H(n) of that distribution varies token
by token — forming a trajectory. Correct (grounded) responses and hallucinated responses
have systematically different spectral structures in this trajectory. We extract 12
spectral features (FFT, STFT, sliding-window variance, tail ratio) and fuse them using
Nadler's combinatorial spectral algorithm to produce a single hallucination score per
response. The method is **fully unsupervised and gray-box** (requires only token-level
logits, not attention maps or hidden states).

---

## Results summary

| Dataset / Model | Our AUC | Method | Competitor | Their AUC |
|---|---|---|---|---|
| MATH-500 / Qwen2.5-7B T=1.5 | **96.6%** | Spectral Nadler | RENT | TBD |
| MATH-500 / Qwen2.5-7B T=1.0 | **90.0%** | Spectral Nadler | RENT | TBD |
| MATH-500 / Qwen2.5-1.5B T=1.5 | **88.3%** | Spectral Nadler | RENT | TBD |
| GSM8K / Llama-3.1-8B T=1.0 | TBD (Phase 7 running) | Spectral Nadler | LapEigvals | 87.2% (supervised) |
| GPQA Diamond / Mistral-7B T=1.0 | 65.4% | Spectral Nadler | — | — |
| HotpotQA / Mistral-7B T=1.0 | 59.5% | Spectral Nadler | LOS-Net | 72.9% (supervised) |

---

## Repository structure

```
hallucination_detection/
├── spectral_utils/               # Shared importable package
│   ├── __init__.py
│   ├── io_utils.py               # load_cache, save_cache
│   ├── model_utils.py            # load_model, generate_full, token_entropies_from_scores
│   ├── feature_utils.py          # 12 spectral features, extract_all_features, FEAT_NAMES
│   ├── fusion_utils.py           # zscore, boot_auc, nadler_fuse, simple_average_fusion, best_nadler_on
│   └── data_loaders.py           # GSM8K, MATH-500, GPQA Diamond, HotpotQA
│
├── Spectral_Analysis_Phase4.ipynb     # MATH-500 + GPQA, T=1.5
├── Spectral_Analysis_Phase5.ipynb     # T=1.0 + cross-temperature fusion
├── Spectral_Analysis_Phase6.ipynb     # HotpotQA generalization test
├── Spectral_Analysis_GSM8K_vs_LapEigvals.ipynb   # Phase 7: GSM8K vs LapEigvals
├── Meeting_Presentation_Plots.ipynb  # Figures for Apr 27 meeting
│
├── HISTORY.md                    # Step-by-step experiment log
├── ROADMAP.md                    # Planned next steps with rationale
├── Advisor_Feedback_May2026.md   # Action items from May 2026 meeting
├── Research_Directions.md        # Longer-term research questions
│
├── setup.py                      # pip install -e . or pip install git+...
└── .gitignore
```

---

## Using `spectral_utils` in a Colab notebook

**Option A — install from GitHub (recommended, always gets latest):**
```python
!pip install git+https://github.com/omrisegev/hallucination_detection.git -q
from spectral_utils import load_model, extract_all_features, best_nadler_on, FEAT_NAMES
from spectral_utils.data_loaders import load_gsm8k, gsm8k_prompt, is_correct_gsm8k
```

**Option B — mount Drive and add to path (useful during active development):**
```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/hallucination_detection')
from spectral_utils import load_model, extract_all_features, best_nadler_on, FEAT_NAMES
```

**Option C — local development:**
```bash
git clone https://github.com/omrisegev/hallucination_detection.git
cd hallucination_detection
pip install -e ".[inference]"
```

---

## Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Trace splitting | Full response only | Marker phrase appears <2% of the time; fallback split is arbitrary |
| Window ablation | w ∈ {3,5,7,9,16}, sw_step=1 | Token-by-token sensitivity; optimal w is dataset-dependent |
| Fusion | Nadler spectral, max_size=4 | Exploits covariance structure between complementary features |
| Normalization | Z-score before fusion | Prevents high-scale features (trace_length ~300) dominating the covariance |
| Evaluation | Bootstrap AUROC, 95% CI, n=1000 | Unsupervised; no train/test split needed |
| Supervision | None | Our primary advantage over LapEigvals (supervised, requires 80% labels) |

---

## Running a new experiment

1. Add any new dataset loader to `spectral_utils/data_loaders.py`
2. Create a new notebook — minimal boilerplate:

```python
!pip install git+https://github.com/omrisegev/hallucination_detection.git -q
from google.colab import drive, userdata
drive.mount('/content/drive')

from spectral_utils import (
    load_model, generate_full, free_memory,
    extract_all_features, sw_var_peak_with_window, FEAT_NAMES,
    load_cache, save_cache,
    boot_auc, best_nadler_on,
)
import numpy as np, os

# --- config, inference loop, feature extraction, window ablation,
#     individual AUCs, nadler fusion, decision gates ---
# Everything else is dataset-specific logic.
```

---

## Dependencies

Core (always needed): `numpy scipy scikit-learn`  
Inference (Colab runs): `torch transformers>=4.40 accelerate datasets bitsandbytes`
