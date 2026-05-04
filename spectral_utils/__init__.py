"""
spectral_utils — shared utilities for the hallucination detection spectral pipeline.

Usage in Colab:
    !pip install git+https://github.com/omrisegev/hallucination_detection.git -q
    from spectral_utils import load_model, extract_all_features, best_nadler_on, FEAT_NAMES
    from spectral_utils.data_loaders import load_gsm8k, gsm8k_prompt, is_correct_gsm8k
    from spectral_utils.data_loaders import load_trivia_qa, trivia_qa_prompt, is_correct_trivia_qa
    from spectral_utils.data_loaders import load_webq, webq_prompt, is_correct_webq
"""

from .io_utils import load_cache, save_cache
from .model_utils import load_model, fmt_prompt, generate_full, token_entropies_from_scores, free_memory
from .feature_utils import (
    compute_spectral_features,
    compute_stft_features,
    compute_time_domain,
    extract_all_features,
    sw_var_peak_with_window,
    sw_var_peak_adaptive,
    FEAT_NAMES,
)
from .fusion_utils import zscore, boot_auc, nadler_fuse, simple_average_fusion, best_nadler_on

__version__ = "0.1.0"
__all__ = [
    "load_cache", "save_cache",
    "load_model", "fmt_prompt", "generate_full", "token_entropies_from_scores", "free_memory",
    "compute_spectral_features", "compute_stft_features", "compute_time_domain",
    "extract_all_features", "sw_var_peak_with_window", "sw_var_peak_adaptive", "FEAT_NAMES",
    "zscore", "boot_auc", "nadler_fuse", "simple_average_fusion", "best_nadler_on",
]
