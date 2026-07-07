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

from .io_utils import load_cache, save_cache, save_cache_atomic
from .model_utils import load_model, fmt_prompt, generate_full, token_entropies_from_scores, token_entropies_and_spilled, extract_top_k_logprobs, free_memory
from .feature_utils import (
    compute_spectral_features,
    compute_stft_features,
    compute_time_domain,
    extract_all_features,
    sw_var_peak_with_window,
    sw_var_peak_adaptive,
    compute_edis,
    compute_spilled_energy_features,
    segment_by_citations,
    FEAT_NAMES,
)
from .data_loaders import (
    load_gsm8k, gsm8k_prompt, gsm8k_prompt_with_conf, is_correct_gsm8k, normalize_gsm8k,
    extract_gold_gsm8k, extract_model_answer_gsm8k,
    load_math500, math_prompt, is_correct_math,
    load_amc23, amc23_prompt, is_correct_amc23,
    load_aime24, aime24_prompt, is_correct_aime24,
    load_gpqa, gpqa_prompt_and_answer, is_correct_gpqa,
    load_hotpotqa, hotpotqa_prompt, is_correct_hotpotqa,
    load_hotpotqa_agentic, load_2wikimultihopqa, load_agentic_multihop_dataset,
    normalize_agentic_multihop_row,
    load_trivia_qa, trivia_qa_prompt, is_correct_trivia_qa,
    load_webq, webq_prompt, is_correct_webq,
    load_humaneval, humaneval_prompt, is_correct_humaneval,
    load_lciteeval, lciteeval_prompt, lciteeval_grounding_label,
)
from .fusion_utils import (
    zscore, boot_auc, paired_boot_delta_auc,
    binarize_classifiers, sml_fuse, nadler_fuse,
    simple_average_fusion, best_nadler_on, best_nadler_pseudo_label,
    sml_fuse_signed, detect_dependent_groups, lsml_fuse,
    lsml_continuous, lsml_continuous_pipeline, multipass_lsml_continuous,
    sml_unsupervised, sml_unsupervised_compare,
    upcr_fuse, upcr_pipeline,
)
from .diagnostics import (
    decompose_auroc, threshold_sensitivity, derive_consensus_signs,
    plot_decomposition, plot_per_feature_heatmap,
    plot_sign_agreement, plot_threshold_sweep,
    plot_correlation_with_groups,
)
from .baselines import (
    lite_semantic_entropy_for_statement, mean_neg_logprob_baseline,
    nli_load_model, nli_classify,
    official_semantic_entropy,
    discrete_semantic_entropy,
    likelihood_weighted_semantic_entropy,
    self_consistency_score,
    selfcheck_nli_score,
    selfcheck_nli_score_official,
    parse_verbalized_confidence, VERBALIZED_CONF_SUFFIX,
)
from .streaming_utils import (
    FEATURE_SIGNS, iter_entropy_traces, iter_trace_records, anchor_orient,
    prefix_features, prefix_feature_matrix,
    deepconf_lowest_group_conf, deepconf_tail_conf,
    causal_trajectories, earliness_index, online_flag_curve,
)
from .anomaly_utils import (
    build_feature_matrix, mahalanobis_scores, gmm_nll_scores,
    kde_nll_scores, iforest_scores, ae_scores, prae_scores,
    TRACKA_METHODS, AE_MIN_SAMPLES,
)
from .temporal_models import (
    fit_gaussian_hmm, hmm_posteriors, hmm_trace_scores,
    bocpd_gaussian, ar_innovation_scores, kalman_innovation_scores,
)
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
    "load_cache", "save_cache", "save_cache_atomic",
    "load_model", "fmt_prompt", "generate_full", "token_entropies_from_scores",
    "token_entropies_and_spilled", "extract_top_k_logprobs", "free_memory",
    "compute_spectral_features", "compute_stft_features", "compute_time_domain",
    "extract_all_features", "sw_var_peak_with_window", "sw_var_peak_adaptive",
    "compute_edis", "compute_spilled_energy_features", "segment_by_citations", "FEAT_NAMES",
    "load_gsm8k", "gsm8k_prompt", "gsm8k_prompt_with_conf", "is_correct_gsm8k", "normalize_gsm8k",
    "extract_gold_gsm8k", "extract_model_answer_gsm8k",
    "load_math500", "math_prompt", "is_correct_math",
    "load_amc23", "amc23_prompt", "is_correct_amc23",
    "load_aime24", "aime24_prompt", "is_correct_aime24",
    "load_gpqa", "gpqa_prompt_and_answer", "is_correct_gpqa",
    "load_hotpotqa", "hotpotqa_prompt", "is_correct_hotpotqa",
    "load_hotpotqa_agentic", "load_2wikimultihopqa", "load_agentic_multihop_dataset",
    "normalize_agentic_multihop_row",
    "load_trivia_qa", "trivia_qa_prompt", "is_correct_trivia_qa",
    "load_webq", "webq_prompt", "is_correct_webq",
    "load_humaneval", "humaneval_prompt", "is_correct_humaneval",
    "load_lciteeval", "lciteeval_prompt", "lciteeval_grounding_label",
    "zscore", "boot_auc", "paired_boot_delta_auc",
    "binarize_classifiers", "sml_fuse",
    "nadler_fuse", "simple_average_fusion", "best_nadler_on",
    "best_nadler_pseudo_label",
    "sml_fuse_signed", "detect_dependent_groups", "lsml_fuse",
    "lsml_continuous", "lsml_continuous_pipeline", "multipass_lsml_continuous",
    "sml_unsupervised", "sml_unsupervised_compare",
    "upcr_fuse", "upcr_pipeline",
    "decompose_auroc", "threshold_sensitivity", "derive_consensus_signs",
    "plot_decomposition", "plot_per_feature_heatmap",
    "plot_sign_agreement", "plot_threshold_sweep",
    "plot_correlation_with_groups",
    "lite_semantic_entropy_for_statement", "mean_neg_logprob_baseline",
    "nli_load_model", "nli_classify",
    "official_semantic_entropy",
    "discrete_semantic_entropy",
    "likelihood_weighted_semantic_entropy",
    "self_consistency_score",
    "selfcheck_nli_score",
    "selfcheck_nli_score_official",
    "parse_verbalized_confidence", "VERBALIZED_CONF_SUFFIX",
    "FEATURE_SIGNS", "iter_entropy_traces", "iter_trace_records", "anchor_orient",
    "prefix_features", "prefix_feature_matrix",
    "deepconf_lowest_group_conf", "deepconf_tail_conf",
    "causal_trajectories", "earliness_index", "online_flag_curve",
    "build_feature_matrix", "mahalanobis_scores", "gmm_nll_scores",
    "kde_nll_scores", "iforest_scores", "ae_scores", "prae_scores",
    "TRACKA_METHODS", "AE_MIN_SAMPLES",
    "fit_gaussian_hmm", "hmm_posteriors", "hmm_trace_scores",
    "bocpd_gaussian", "ar_innovation_scores", "kalman_innovation_scores",
    "react_system_prompt", "react_user_prompt",
    "parse_thought", "parse_action", "parse_confidence", "parse_concern",
    "simulate_retrieve_tool", "step_retrieved_supporting_fact",
    "run_react_episode", "aggregate_trajectory", "categorize_failure_mode",
    "query_support_overlap", "first_incorrect_step_index", "categorize_failure_mode_v2",
    "branching_entropy", "run_spiral_injection_replay",
    "execute_python_solution", "run_humaneval_episode",
]
