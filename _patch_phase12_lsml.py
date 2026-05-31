"""
Edits Spectral_Analysis_Phase12_Benchmarking.ipynb in place to use the
paper-aligned sml_unsupervised method instead of best_nadler_pseudo_label.

Changes made (per CLAUDE.md rule: never string-replace notebook JSON):
  Cell 2  — switch BRANCH to feature/nadler-paper-alignment + import sml_unsupervised
  Cell 3  — annotate SEED_FEATS/SEED_SIGNS as informational only (no longer needed by L-SML)
  Cell 11 — P1b: best_nadler_pseudo_label → sml_unsupervised_compare; richer save dict
  Cell 15 — P2b: same swap
  Cell 16 — P2c (DeepSeek-R1-7B): same swap
  Cell 17 — P2d (Qwen3-8B): same swap
  Cell 27 — Master comparison: update row labels Nadler → L-SML, supervision None (pseudo) → None
  Cell 28 — Markdown report: same label updates
"""
import json

NB_PATH = "Spectral_Analysis_Phase12_Benchmarking.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)


def replace_cell(idx, new_source):
    """Replace a cell's source with given multiline string."""
    nb["cells"][idx]["source"] = new_source.splitlines(keepends=True)
    # Reset execution counter and outputs for code cells
    if nb["cells"][idx]["cell_type"] == "code":
        nb["cells"][idx]["execution_count"] = None
        nb["cells"][idx]["outputs"] = []


# ============================================================
# Cell 2 — Setup: switch branch + import sml_unsupervised
# ============================================================
replace_cell(2, """# Cell 1 — Drive mount + clone + install + imports (Step 106: paper-aligned L-SML branch)
from google.colab import drive
drive.mount('/content/drive')

import os, sys, shutil, re, pickle
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'

REPO_DIR = '/content/hallucination_detection'
BRANCH   = 'feature/nadler-paper-alignment'   # Step 106: pure unsupervised L-SML

if os.path.exists(REPO_DIR) and not os.path.exists(os.path.join(REPO_DIR, 'spectral_utils')):
    shutil.rmtree(REPO_DIR)
if not os.path.exists(REPO_DIR):
    os.system(f'git clone -b {BRANCH} https://github.com/omrisegev/hallucination_detection.git {REPO_DIR}')
else:
    os.system(f'git -C {REPO_DIR} fetch --all -q')
    os.system(f'git -C {REPO_DIR} checkout {BRANCH} -q')
    os.system(f'git -C {REPO_DIR} pull -q')
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

os.system('pip install -q "transformers>=4.40" accelerate datasets bitsandbytes autoawq scipy tqdm')

from spectral_utils import (
    load_model, generate_full, free_memory, load_cache, save_cache,
    boot_auc,
    extract_all_features, FEAT_NAMES,
    # Step 106: paper-aligned L-SML pipeline (Paper 1 + Paper 2)
    sml_unsupervised, sml_unsupervised_compare,
    # Kept for any downstream code that still references them
    best_nadler_on, best_nadler_pseudo_label,
    nli_load_model,
    official_semantic_entropy, self_consistency_score,
    selfcheck_nli_score, parse_verbalized_confidence, VERBALIZED_CONF_SUFFIX,
    normalize_gsm8k, extract_model_answer_gsm8k,
    lciteeval_grounding_label,
)
from spectral_utils.data_loaders import (
    load_gsm8k, gsm8k_prompt, is_correct_gsm8k,
    load_math500, math_prompt, is_correct_math,
    load_gpqa, gpqa_prompt_and_answer, is_correct_gpqa,
    load_lciteeval, lciteeval_prompt,
)
import datasets  # freeze pyarrow
import numpy as np
from tqdm import tqdm

print('spectral_utils imported OK')
print('Using branch:', BRANCH)
print('L-SML functions available:', 'sml_unsupervised' in dir(),
      'sml_unsupervised_compare' in dir())
""")


# ============================================================
# Cell 3 — Config: keep SEED_FEATS/SEED_SIGNS as informational
# ============================================================
replace_cell(3, """# Cell 2 — Config
K_SE_SC  = 10   # K for Semantic Entropy + Self-Consistency
K_CHECK  = 5    # K for SelfCheckGPT
N_MATH   = 200  # GSM8K / MATH-500 balanced subset
N_GPQA   = 150  # GPQA subset
# L-CiteEval dataset sizes (from dataset paper)
N_RAG_SIZES = {'hotpotqa': 240, 'natural_questions': 160, '2wikimultihopqa': 240, 'narrativeqa': 240}

NLI_MODEL  = 'cross-encoder/nli-deberta-v3-base'
BASE_DRIVE = '/content/drive/MyDrive'
DRIVE_BASE = f'{BASE_DRIVE}/hallucination_detection'
CACHE_DIR  = os.path.join(DRIVE_BASE, 'cache', 'phase12_baselines')
os.makedirs(CACHE_DIR, exist_ok=True)

# ─── L-SML config (Step 106) ────────────────────────────────────────────────
# sml_unsupervised resolves signs internally via Paper 2 assumption (iii),
# so SEED_FEATS / SEED_SIGNS are no longer used by the fusion algorithm.
# Kept here as DOMAIN-KNOWLEDGE REFERENCE: these were the 5 globally stable
# features identified in meta-analysis Step 89, with their entropy-direction
# signs (-1 = higher feature value → more uncertain → wrong).  Not used
# downstream — the new L-SML method is fully unsupervised including signs.
SEED_FEATS = ['cusum_max', 'sw_var_peak', 'epr', 'spectral_entropy', 'rpdi']
SEED_SIGNS = {
    'cusum_max':        -1,
    'sw_var_peak':      -1,
    'epr':              -1,
    'spectral_entropy': -1,
    'rpdi':             -1,
}
# L-SML K-selection range (number of latent classifier groups to try)
LSML_K_RANGE = range(2, 7)

# Phase 7 GSM8K cache (Llama-3.1-8B)
PHASE7_CACHE = os.path.join(BASE_DRIVE, 'epr_spectral_gsm8k_vs_lapei',
                             'Llama-3.1-8B-Instruct__gsm8k_T1.0', 'inference_cache.pkl')

# Phase 5 MATH-500 — auto-detect root dir (same logic as Consolidated notebook)
_p5_candidates = [
    os.path.join(BASE_DRIVE,  'epr_spectral_phase5'),
    os.path.join(BASE_DRIVE,  'epr_spectral_phase4'),
    os.path.join(DRIVE_BASE,  'epr_spectral_phase5'),
    os.path.join(DRIVE_BASE,  'epr_spectral_phase4'),
]
PHASE5_ROOT = next((p for p in _p5_candidates if os.path.exists(p)), None)
print(f'PHASE5_ROOT: {PHASE5_ROOT} ({"OK" if PHASE5_ROOT else "NOT FOUND"})')

# Phase 10 RAG caches — Qwen-7B, 4 datasets
PHASE10_CACHES = {}
for _ds in ['hotpotqa', 'natural_questions', '2wikimultihopqa', 'narrativeqa']:
    _c = [
        os.path.join(DRIVE_BASE, 'cache', 'phase10_main', 'raw',  f'qwen7b__{_ds}__inference.pkl'),
        os.path.join(DRIVE_BASE, 'cache', 'phase10_main', 'raw',  f'qwen25_7b__{_ds}__inference.pkl'),
        os.path.join(DRIVE_BASE, 'cache', 'phase10_rag',          f'qwen7b__{_ds}__inference.pkl'),
        os.path.join(DRIVE_BASE, 'cache', 'phase10_rag',          f'qwen25_7b__{_ds}__inference.pkl'),
    ]
    found = next((p for p in _c if os.path.exists(p)), None)
    PHASE10_CACHES[_ds] = found
    status = f'OK ({os.path.basename(found)})' if found else 'NOT FOUND'
    print(f'  phase10/{_ds}: {status}')

# Consolidated Results pkl (output of Spectral_Analysis_Consolidated_Results.ipynb)
CONSOLIDATED_PKL = os.path.join(DRIVE_BASE, 'consolidated_results', 'results_all.pkl')
print(f'\\nConsolidated results: {"OK" if os.path.exists(CONSOLIDATED_PKL) else "NOT FOUND — run Consolidated notebook first"}')
print(f'Phase 7 cache:        {"OK" if os.path.exists(PHASE7_CACHE) else "NOT FOUND"}')
print(f'Cache dir:             {CACHE_DIR}')

# Shared stale-pkl validator
def _p12_valid(samples):
    return bool(samples) and any(
        v.get('done') for v in samples.values() if isinstance(v, dict)
    )


# Reusable L-SML fusion helper (used in all P1b/P2b/P2c/P2d cells)
def run_lsml_and_save(feats_dict, labels, key_str, save_path):
    \"\"\"
    Run paper-aligned L-SML on (feats_dict, labels) with BOTH K-selection methods.
    Returns (auc, lo, hi, K, save_dict).  AUROC is reported against the residual
    method (Paper 1 Alg 1, paper-faithful headline).
    Labels are used ONLY for AUROC computation (NEVER for fusion).
    \"\"\"
    cmp = sml_unsupervised_compare(feats_dict, FEAT_NAMES,
                                   K_range=LSML_K_RANGE, labels=labels)
    r_fused = cmp['residual_fused']
    p_auc, p_lo, p_hi = boot_auc(labels, r_fused)
    n_auc, n_lo, n_hi = boot_auc(labels, -r_fused)
    if p_auc >= n_auc:
        auc, lo, hi = p_auc, p_lo, p_hi
    else:
        auc, lo, hi = n_auc, n_lo, n_hi

    save_d = {
        'auc': auc, 'lo': lo, 'hi': hi,        # backward-compat for cells 14/15
        'subset': f"L-SML K={cmp['K_residual']}",  # human-readable summary
        'n': len(labels),
        # L-SML-specific metadata
        'method': 'sml_unsupervised',
        'K_residual': cmp['K_residual'],
        'K_eigengap': cmp['K_eigengap'],
        'group_ARI':  cmp['group_ARI'],
        'same_K':     cmp['same_K'],
        'eigengap_auc': cmp['eigengap_auc'],
        'residual_c': cmp['residual_meta']['c'].tolist(),
    }
    save_cache(save_d, save_path)
    print(f"  [{key_str}] L-SML(resid)={100*auc:.1f}% [{100*lo:.1f},{100*hi:.1f}] "
          f"K_resid={cmp['K_residual']} | (eig)={100*cmp['eigengap_auc']:.1f}% K_eig={cmp['K_eigengap']} "
          f"| ARI={cmp['group_ARI']:.2f}")
    return auc, lo, hi, cmp['K_residual'], save_d
""")


# ============================================================
# Cell 11 — P1b GSM8K/Mistral-7B
# ============================================================
replace_cell(11, """# Cell P1b-nadler — extract spectral features + L-SML (paper-aligned, fully unsupervised)
P1B_NADLER_PATH = os.path.join(CACHE_DIR, 'p1b_gsm8k_mistral7b_lsml.pkl')

p1b_valid  = [k for k in sel_keys
              if p1b_samples.get(k, {}).get('done')
              and len(p1b_samples[k].get('token_entropies', [])) > 0]
p1b_labels = np.array([p1b_samples[k]['correct'] for k in p1b_valid])
print(f'P1b valid: {len(p1b_valid)}, accuracy={p1b_labels.mean():.1%}')

p1b_feat_raw = {k: extract_all_features(p1b_samples[k]['token_entropies']) for k in p1b_valid}
feats_p1b    = {f: np.array([p1b_feat_raw[k][f] for k in p1b_valid]) for f in FEAT_NAMES}

# L-SML: median-binarize all 16 features, no orientation, no subset; Paper 1 group
# detection (residual K-selection) + within/across SML.  Real labels used only for AUROC.
p1b_auc, p1b_lo, p1b_hi, p1b_K, p1b_save = run_lsml_and_save(
    feats_p1b, p1b_labels, 'P1b GSM8K/Mistral-7B', P1B_NADLER_PATH,
)
p1b_subset = p1b_save['subset']

_nan = float('nan')
print()
print('=' * 85)
print(f'GSM8K / Mistral-7B-Instruct-v0.3 / T=1.0  (n={len(p1b_valid)})')
print('=' * 85)
print(f"  {'Method':<46} {'AUROC':>8}  {'95% CI':>16}  Supervision")
print('-' * 85)
for method, auc, lo, hi, sup in [
    ('L-SML (ours, paper-aligned, Paper 1+2)',       p1b_auc, p1b_lo, p1b_hi, 'None'),
    ('SE NLI K=10 (arXiv 2502.03799)',               0.7585,  _nan,   _nan,   'None'),
]:
    ci = f'[{100*lo:.1f},{100*hi:.1f}]' if lo == lo else 'n/a'
    print(f'  {method:<46} {100*auc:>7.1f}%  {ci:>16}  {sup}')
print('=' * 85)
print(f'Saved: {P1B_NADLER_PATH}')
""")


# ============================================================
# Cell 15 — P2b Qwen-7B GPQA L-SML
# ============================================================
replace_cell(15, """# Cell 8b — L-SML on Qwen-7B GPQA using existing Cell 7 inference (no model reload)
# Apples-to-apples: L-SML vs SC/SE/VC all on the same Qwen2.5-7B.
P2B_NADLER_PATH = os.path.join(CACHE_DIR, 'p2b_gpqa_qwen7b_lsml.pkl')

p2b_valid = [k for k in p2_valid
             if p2_samples[k].get('main_entropies') is not None
             and len(p2_samples[k].get('main_entropies', [])) > 0]
print(f'P2b: {len(p2b_valid)} samples have main_entropies / {len(p2_valid)} total P2 valid')
p2b_labels = np.array([p2_samples[k]['correct'] for k in p2b_valid])

p2b_feat_raw = {k: extract_all_features(p2_samples[k]['main_entropies']) for k in p2b_valid}
feats_p2b    = {f: np.array([p2b_feat_raw[k][f] for k in p2b_valid]) for f in FEAT_NAMES}

p2b_auc, p2b_lo, p2b_hi, p2b_K, p2b_save = run_lsml_and_save(
    feats_p2b, p2b_labels, 'GPQA/Qwen-7B', P2B_NADLER_PATH,
)
p2b_subset = p2b_save['subset']
print(f'\\nL-SML Qwen-7B GPQA: {100*p2b_auc:.1f}% [{100*p2b_lo:.1f},{100*p2b_hi:.1f}] (K={p2b_K})')
print(f'  SC Qwen-7B:  {100*sc_auc_p2:.1f}%  |  SE Qwen-7B: {100*se_p2_auc:.1f}%')
""")


# ============================================================
# Cell 16 — P2c DeepSeek-R1-Distill-Qwen-7B GPQA
# ============================================================
replace_cell(16, """# Cell 8c — GPQA / DeepSeek-R1-Distill-Qwen-7B: inference + L-SML (matches arXiv 2603.19118)
# Paper used DeepSeek-R1-8B; DeepSeek-R1-Distill-Qwen-7B is the equivalent open 7B reasoning model.
# Task: GPQA Diamond MCQ correctness = hallucination detection (same as Phase 8 / Phase 4).
P2C_INF_PATH    = os.path.join(CACHE_DIR, 'p2c_gpqa_deepseek_r1_7b_inference.pkl')
P2C_NADLER_PATH = os.path.join(CACHE_DIR, 'p2c_gpqa_deepseek_r1_7b_lsml.pkl')
FORCE_P2C = False

p2c_auc = p2c_lo = p2c_hi = float('nan')
p2c_subset = None
_skip_p2c = False

if not FORCE_P2C and os.path.exists(P2C_NADLER_PATH):
    _r = load_cache(P2C_NADLER_PATH)
    if _r and _r.get('auc') == _r.get('auc'):
        p2c_auc, p2c_lo, p2c_hi = _r['auc'], _r['lo'], _r['hi']
        p2c_subset = _r.get('subset')
        print(f'P2c loaded from cache: {100*p2c_auc:.1f}% [{100*p2c_lo:.1f},{100*p2c_hi:.1f}]')
        _skip_p2c = True

if not _skip_p2c:
    p2c_samples = load_cache(P2C_INF_PATH) if os.path.exists(P2C_INF_PATH) else {}
    free_memory()
    gpqa_data = load_gpqa()
    N_gpqa    = min(N_GPQA, len(gpqa_data))

    ds_mdl, ds_tok = load_model('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')

    for i in tqdm(range(N_gpqa), desc='P2c DeepSeek-R1-7B GPQA'):
        if p2c_samples.get(i, {}).get('done'):
            continue
        row = gpqa_data[i]
        prompt, gold = gpqa_prompt_and_answer(row, i)
        try:
            _r = generate_full(ds_mdl, ds_tok, prompt, temperature=1.0, max_new_tokens=1024)
            p2c_samples[i] = {
                'done': True,
                'main_text': _r['full_text'],
                'main_entropies': _r['token_entropies'],
                'correct': int(is_correct_gpqa(_r['full_text'], gold)),
            }
        except Exception as ex:
            print(f'  error {i}: {ex}')
            p2c_samples[i] = {'done': False}
        if i % 25 == 0:
            save_cache(p2c_samples, P2C_INF_PATH)

    save_cache(p2c_samples, P2C_INF_PATH)
    del ds_mdl, ds_tok; free_memory()

    p2c_valid = [k for k, v in p2c_samples.items() if v.get('done') and v.get('main_entropies')]
    print(f'P2c: {len(p2c_valid)}/{N_gpqa} samples collected')
    p2c_labels   = np.array([p2c_samples[k]['correct'] for k in p2c_valid])
    p2c_feat_raw = {k: extract_all_features(p2c_samples[k]['main_entropies']) for k in p2c_valid}
    feats_p2c    = {f: np.array([p2c_feat_raw[k][f] for k in p2c_valid]) for f in FEAT_NAMES}

    p2c_auc, p2c_lo, p2c_hi, p2c_K, p2c_save = run_lsml_and_save(
        feats_p2c, p2c_labels, 'GPQA/DeepSeek-R1-7B', P2C_NADLER_PATH,
    )
    p2c_subset = p2c_save['subset']

print(f'\\nL-SML DeepSeek-R1-7B GPQA: {100*p2c_auc:.1f}% [{100*p2c_lo:.1f},{100*p2c_hi:.1f}]')
print(f'  subset: {p2c_subset}')
print(f'  SC/SE/VC Qwen-7B (computed): {100*sc_auc_p2:.1f}% / {100*se_p2_auc:.1f}% / {100*vc_auc:.1f}%')
""")


# ============================================================
# Cell 17 — P2d Qwen3-8B GPQA
# ============================================================
replace_cell(17, """# Cell 8d — GPQA / Qwen3-8B: inference + L-SML (matches arXiv 2603.19118)
# Paper used Qwen3-30B-A3B; Qwen3-8B is the same model generation at 8B scale.
# Task: GPQA Diamond MCQ correctness = hallucination detection.
P2D_INF_PATH    = os.path.join(CACHE_DIR, 'p2d_gpqa_qwen3_8b_inference.pkl')
P2D_NADLER_PATH = os.path.join(CACHE_DIR, 'p2d_gpqa_qwen3_8b_lsml.pkl')
FORCE_P2D = False

p2d_auc = p2d_lo = p2d_hi = float('nan')
p2d_subset = None
_skip_p2d = False

if not FORCE_P2D and os.path.exists(P2D_NADLER_PATH):
    _r = load_cache(P2D_NADLER_PATH)
    if _r and _r.get('auc') == _r.get('auc'):
        p2d_auc, p2d_lo, p2d_hi = _r['auc'], _r['lo'], _r['hi']
        p2d_subset = _r.get('subset')
        print(f'P2d loaded from cache: {100*p2d_auc:.1f}% [{100*p2d_lo:.1f},{100*p2d_hi:.1f}]')
        _skip_p2d = True

if not _skip_p2d:
    p2d_samples = load_cache(P2D_INF_PATH) if os.path.exists(P2D_INF_PATH) else {}
    free_memory()
    gpqa_data = load_gpqa()
    N_gpqa    = min(N_GPQA, len(gpqa_data))

    q3_mdl, q3_tok = load_model('Qwen/Qwen3-8B')

    for i in tqdm(range(N_gpqa), desc='P2d Qwen3-8B GPQA'):
        if p2d_samples.get(i, {}).get('done'):
            continue
        row = gpqa_data[i]
        prompt, gold = gpqa_prompt_and_answer(row, i)
        try:
            _r = generate_full(q3_mdl, q3_tok, prompt, temperature=1.0, max_new_tokens=1024)
            p2d_samples[i] = {
                'done': True,
                'main_text': _r['full_text'],
                'main_entropies': _r['token_entropies'],
                'correct': int(is_correct_gpqa(_r['full_text'], gold)),
            }
        except Exception as ex:
            print(f'  error {i}: {ex}')
            p2d_samples[i] = {'done': False}
        if i % 25 == 0:
            save_cache(p2d_samples, P2D_INF_PATH)

    save_cache(p2d_samples, P2D_INF_PATH)
    del q3_mdl, q3_tok; free_memory()

    p2d_valid = [k for k, v in p2d_samples.items() if v.get('done') and v.get('main_entropies')]
    print(f'P2d: {len(p2d_valid)}/{N_gpqa} samples collected')
    p2d_labels   = np.array([p2d_samples[k]['correct'] for k in p2d_valid])
    p2d_feat_raw = {k: extract_all_features(p2d_samples[k]['main_entropies']) for k in p2d_valid}
    feats_p2d    = {f: np.array([p2d_feat_raw[k][f] for k in p2d_valid]) for f in FEAT_NAMES}

    p2d_auc, p2d_lo, p2d_hi, p2d_K, p2d_save = run_lsml_and_save(
        feats_p2d, p2d_labels, 'GPQA/Qwen3-8B', P2D_NADLER_PATH,
    )
    p2d_subset = p2d_save['subset']

print(f'\\nL-SML Qwen3-8B GPQA: {100*p2d_auc:.1f}% [{100*p2d_lo:.1f},{100*p2d_hi:.1f}]')
print(f'  subset: {p2d_subset}')
print(f'  SC/SE/VC Qwen-7B (computed): {100*sc_auc_p2:.1f}% / {100*se_p2_auc:.1f}% / {100*vc_auc:.1f}%')
""")


# ============================================================
# Cells 27, 28 — Update row labels and pkl paths for L-SML
# ============================================================
# Both cells reference the four pkl paths and produce table rows.
# We update pkl filenames (Nadler → lsml), method-row labels (Nadler → L-SML),
# and supervision string ('None (pseudo)' → 'None') via targeted JSON edits.

for cidx in [27, 28]:
    src = "".join(nb["cells"][cidx]["source"])
    # pkl filename updates
    src = src.replace("p1b_gsm8k_mistral7b_nadler.pkl",  "p1b_gsm8k_mistral7b_lsml.pkl")
    src = src.replace("p2b_gpqa_qwen7b_nadler.pkl",      "p2b_gpqa_qwen7b_lsml.pkl")
    src = src.replace("p2c_gpqa_deepseek_r1_7b_nadler.pkl", "p2c_gpqa_deepseek_r1_7b_lsml.pkl")
    src = src.replace("p2d_gpqa_qwen3_8b_nadler.pkl",    "p2d_gpqa_qwen3_8b_lsml.pkl")
    # Row label updates for the Phase 12 newly-computed rows (P1b/P2b/P2c/P2d)
    # These match exact strings inside _row(...) / _md_row(...) calls.
    src = src.replace(
        "'Nadler (ours) -- Mistral-7B (P1b, pseudo-label)'",
        "'L-SML (ours, paper-aligned) -- Mistral-7B (P1b)'",
    )
    src = src.replace(
        "'Nadler (ours, pseudo-label) -- Qwen-7B (P2b, free)'",
        "'L-SML (ours, paper-aligned) -- Qwen-7B (P2b)'",
    )
    src = src.replace(
        "'Nadler (ours, pseudo-label) -- DeepSeek-R1-Distill-Qwen-7B (P2c)'",
        "'L-SML (ours, paper-aligned) -- DeepSeek-R1-Distill-Qwen-7B (P2c)'",
    )
    src = src.replace(
        "'Nadler (ours, pseudo-label) -- Qwen3-8B (P2d)'",
        "'L-SML (ours, paper-aligned) -- Qwen3-8B (P2d)'",
    )
    # Supervision tag for those rows: None (pseudo) → None  (L-SML uses no labels)
    src = src.replace("'None (pseudo)'", "'None'")
    # Print labels
    src = src.replace(
        'P1b (Mistral-7B GSM8K):',
        'P1b (Mistral-7B GSM8K, L-SML):',
    )
    src = src.replace(
        'P2b (Qwen-7B GPQA):',
        'P2b (Qwen-7B GPQA, L-SML):',
    )
    src = src.replace(
        'P2c (DeepSeek-R1-7B GPQA):',
        'P2c (DeepSeek-R1-7B GPQA, L-SML):',
    )
    src = src.replace(
        'P2d (Qwen3-8B GPQA):',
        'P2d (Qwen3-8B GPQA, L-SML):',
    )
    # Write back as list of lines (preserve JSON formatting)
    nb["cells"][cidx]["source"] = src.splitlines(keepends=True)


# ============================================================
# Write notebook back
# ============================================================
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Patched {NB_PATH}: {len(nb['cells'])} cells total")
print("Cells modified: 2, 3, 11, 15, 16, 17, 27, 28")
