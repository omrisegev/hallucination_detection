"""
spectral_utils — shared utilities for the hallucination detection spectral pipeline.

Usage in Colab (always use git clone, never pip install git+):
    import os, sys, shutil
    REPO_DIR = '/content/hallucination_detection'
    if os.path.exists(REPO_DIR) and not os.path.exists(os.path.join(REPO_DIR, 'spectral_utils')):
        shutil.rmtree(REPO_DIR)
    if not os.path.exists(REPO_DIR):
        os.system(f'git clone -b master https://github.com/omrisegev/hallucination_detection.git {REPO_DIR}')
    else:
        os.system(f'git -C {REPO_DIR} pull -q')
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

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
    segment_by_citations,
    FEAT_NAMES,
)
from .data_loaders import (
    load_gsm8k, gsm8k_prompt, is_correct_gsm8k,
    load_math500, math_prompt, is_correct_math,
    load_gpqa, gpqa_prompt_and_answer, is_correct_gpqa,
    load_hotpotqa, hotpotqa_prompt, is_correct_hotpotqa,
    load_hotpotqa_agentic, load_2wikimultihopqa, load_agentic_multihop_dataset,
    normalize_agentic_multihop_row,
    load_trivia_qa, trivia_qa_prompt, is_correct_trivia_qa,
    load_webq, webq_prompt, is_correct_webq,
    load_humaneval, humaneval_prompt, is_correct_humaneval,
    load_lciteeval, lciteeval_prompt, lciteeval_grounding_label,
)
from .fusion_utils import zscore, boot_auc, nadler_fuse, simple_average_fusion, best_nadler_on
from .baselines import lite_semantic_entropy_for_statement, mean_neg_logprob_baseline
from .agent_utils import (
    react_system_prompt, react_user_prompt,
    parse_thought, parse_action, parse_confidence, parse_concern,
    simulate_retrieve_tool, step_retrieved_supporting_fact,
    run_react_episode, aggregate_trajectory, categorize_failure_mode,
    query_support_overlap, first_incorrect_step_index, categorize_failure_mode_v2,
    branching_entropy, run_spiral_injection_replay,
    execute_python_solution, run_humaneval_episode,
)

__version__ = "0.1.0"
__all__ = [
    "load_cache", "save_cache",
    "load_model", "fmt_prompt", "generate_full", "token_entropies_from_scores", "free_memory",
    "compute_spectral_features", "compute_stft_features", "compute_time_domain",
    "extract_all_features", "sw_var_peak_with_window", "sw_var_peak_adaptive",
    "segment_by_citations", "FEAT_NAMES",
    "load_gsm8k", "gsm8k_prompt", "is_correct_gsm8k",
    "load_math500", "math_prompt", "is_correct_math",
    "load_gpqa", "gpqa_prompt_and_answer", "is_correct_gpqa",
    "load_hotpotqa", "hotpotqa_prompt", "is_correct_hotpotqa",
    "load_hotpotqa_agentic", "load_2wikimultihopqa", "load_agentic_multihop_dataset",
    "normalize_agentic_multihop_row",
    "load_trivia_qa", "trivia_qa_prompt", "is_correct_trivia_qa",
    "load_webq", "webq_prompt", "is_correct_webq",
    "load_humaneval", "humaneval_prompt", "is_correct_humaneval",
    "load_lciteeval", "lciteeval_prompt", "lciteeval_grounding_label",
    "zscore", "boot_auc", "nadler_fuse", "simple_average_fusion", "best_nadler_on",
    "lite_semantic_entropy_for_statement", "mean_neg_logprob_baseline",
    "react_system_prompt", "react_user_prompt",
    "parse_thought", "parse_action", "parse_confidence", "parse_concern",
    "simulate_retrieve_tool", "step_retrieved_supporting_fact",
    "run_react_episode", "aggregate_trajectory", "categorize_failure_mode",
    "query_support_overlap", "first_incorrect_step_index", "categorize_failure_mode_v2",
    "branching_entropy", "run_spiral_injection_replay",
    "execute_python_solution", "run_humaneval_episode",
]
