"""
Builder for Spectral_Analysis_Phase12_Benchmarking.ipynb.

Scope: ONLY compute the cells marked [COMPUTE] in Research_Phase12_Comparison_Tables.md.
Everything else is already published or available from previous phases.

Priority:
  P1 - GSM8K / Llama-3.1-8B: SE (NLI, K=10) + SC (K=10). Phase 7 cache exists → no model reload.
  P2 - GPQA / Qwen2.5-7B: VC + SC (K=10) + SE (NLI, K=10). Fresh inference.
  P3 - RAG HotpotQA / Llama-3.1-8B or Qwen2.5-7B: SelfCheckGPT (K=5). Phase 10 cache exists.
  P4 - MATH-500 / Qwen2.5-Math-7B: SC + SE (K=10). Phase 5 cache exists.

Run locally: python _build_phase12_notebook.py
"""
import json, os

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'Spectral_Analysis_Phase12_Benchmarking.ipynb')


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': text.splitlines(keepends=True)}


CELLS = []

# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md("""# Phase 12 -- Baseline Computation for Gaps in Published Literature

See `Research_Phase12_Comparison_Tables.md` for the full comparison tables.
This notebook ONLY computes the cells marked `[COMPUTE]` -- where no published number exists
for our exact (dataset, model) pair.

| Priority | Task | Method(s) | Reason |
|----------|------|-----------|--------|
| P1 | GSM8K / Llama-3.1-8B | SE (NLI, K=10) + SC (K=10) | Published SE exists for Mistral-7B but not Llama-3.1-8B |
| P2 | GPQA / Qwen2.5-7B | VC (K=1) + SC (K=10) + SE (NLI, K=10) | Published numbers only for reasoning models (gpt-oss, DeepSeek-R1) |
| P3 | RAG HotpotQA / qwen7b | SelfCheckGPT NLI (K=5) | No published competitor on L-CiteEval task |
| P4 | MATH-500 / Qwen2.5-Math-7B | SC (K=10) + SE (NLI, K=10) | No competitor has published per-dataset MATH-500 AUROC |

Drive cache dir: `/content/drive/MyDrive/hallucination_detection/cache/phase12_baselines/`
"""))

# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md("## Section 1 -- Setup"))

CELLS.append(code('''# Cell 1 -- Drive mount + clone + install + imports
from google.colab import drive
drive.mount('/content/drive')

import os, sys, shutil, re, pickle
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'

REPO_DIR = '/content/hallucination_detection'
if os.path.exists(REPO_DIR) and not os.path.exists(os.path.join(REPO_DIR, 'spectral_utils')):
    shutil.rmtree(REPO_DIR)
if not os.path.exists(REPO_DIR):
    os.system(f'git clone -b master https://github.com/omrisegev/hallucination_detection.git {REPO_DIR}')
else:
    os.system(f'git -C {REPO_DIR} pull -q')
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

os.system('pip install -q "transformers>=4.40" accelerate datasets bitsandbytes autoawq scipy tqdm')

from spectral_utils import (
    load_model, generate_full, free_memory, load_cache, save_cache,
    boot_auc,
    nli_load_model,
    official_semantic_entropy, self_consistency_score,
    selfcheck_nli_score, parse_verbalized_confidence, VERBALIZED_CONF_SUFFIX,
    normalize_gsm8k, extract_model_answer_gsm8k,
)
from spectral_utils.data_loaders import (
    load_gsm8k, gsm8k_prompt, is_correct_gsm8k,
    load_math500, math_prompt, is_correct_math,
    load_gpqa, gpqa_prompt_and_answer, is_correct_gpqa,
    load_lciteeval, lciteeval_prompt, lciteeval_grounding_label,
)
import datasets  # freeze pyarrow
import numpy as np
from tqdm import tqdm

print('spectral_utils imported OK')'''))

CELLS.append(code('''# Cell 2 -- Config
K_SE_SC   = 10   # K for Semantic Entropy + Self-Consistency
K_CHECK   = 5    # K for SelfCheckGPT
N_MATH    = 200  # GSM8K subset (from Phase 7 cache; balanced 100 correct + 100 incorrect)
N_GPQA    = 150  # GPQA subset
N_RAG     = 200  # L-CiteEval HotpotQA subset

NLI_MODEL = 'cross-encoder/nli-deberta-v3-base'

DRIVE_BASE  = '/content/drive/MyDrive/hallucination_detection'
CACHE_DIR   = os.path.join(DRIVE_BASE, 'cache', 'phase12_baselines')
os.makedirs(CACHE_DIR, exist_ok=True)

# Paths to existing inference caches from previous phases
PHASE7_CACHE  = os.path.join(DRIVE_BASE, 'cache', 'epr_spectral_gsm8k_vs_lapei',
                              'Llama-3.1-8B-Instruct__gsm8k_T1.0', 'inference_cache.pkl')
PHASE5_CACHE  = os.path.join(DRIVE_BASE, 'cache', 'phase5_math500', 'inference_cache.pkl')  # adjust if needed
PHASE10_CACHE = os.path.join(DRIVE_BASE, 'cache', 'phase10_rag',
                              'qwen7b__hotpotqa__inference.pkl')

print(f'K_SE_SC={K_SE_SC}, K_CHECK={K_CHECK}')
print(f'Cache dir: {CACHE_DIR}')'''))

CELLS.append(code('''# Cell 3 -- Load NLI model (shared across all priorities)
NLI_MDL, NLI_TOK, NLI_DEV = nli_load_model(model_name=NLI_MODEL, device='cuda')
print(f'NLI model on {NLI_DEV}: {NLI_MODEL}')
print(f'Labels: {NLI_MDL.config.id2label}')'''))

# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md("""## Priority 1 -- GSM8K / Llama-3.1-8B: SE + SC

Phase 7 already ran our Nadler (76.0%) and LapEigvals (72.0% unsup / 87.2% sup) on this exact setup.
We now add Semantic Entropy (NLI, K=10) and Self-Consistency (K=10) for the same N=200 subset.

**Published reference for this gap**: SE on Mistral-7B-Instruct = 75.85% (arXiv 2502.03799 Table 3).
"""))

CELLS.append(code('''# Cell 4 -- Build balanced N=200 subset from Phase 7 cache
p7 = load_cache(PHASE7_CACHE)
valid_keys = sorted(k for k, v in p7.items() if v.get('done') and v.get('all_entropies'))
labels_all = np.array([int(p7[k]['correct']) for k in valid_keys])
# Select balanced subset
idx_c = [k for k, l in zip(valid_keys, labels_all) if l == 1][:N_MATH // 2]
idx_i = [k for k, l in zip(valid_keys, labels_all) if l == 0][:N_MATH - len(idx_c)]
sel_keys = sorted(idx_c + idx_i)
gsm8k_data = load_gsm8k('test')
print(f'GSM8K subset: {len(sel_keys)} samples '
      f'({sum(int(p7[k]["correct"]) for k in sel_keys)} correct, '
      f'{sum(1-int(p7[k]["correct"]) for k in sel_keys)} incorrect)')'''))

CELLS.append(code('''# Cell 5 -- P1: Generate K=10 samples for SE + SC (Llama-3.1-8B)
P1_SAMPLES_PATH = os.path.join(CACHE_DIR, 'p1_gsm8k_llama8b_k10.pkl')
FORCE_P1 = False

if not FORCE_P1 and os.path.exists(P1_SAMPLES_PATH):
    p1_samples = load_cache(P1_SAMPLES_PATH)
    print(f'P1 cache: {sum(v.get("done") for v in p1_samples.values())} / {len(sel_keys)} done')
else:
    p1_samples = load_cache(P1_SAMPLES_PATH) if os.path.exists(P1_SAMPLES_PATH) else {}
    llama_mdl, llama_tok = load_model('meta-llama/Llama-3.1-8B-Instruct')

    for k in tqdm(sel_keys, desc='P1 GSM8K K-sampling'):
        if p1_samples.get(k, {}).get('done'):
            continue
        row = gsm8k_data[k]
        prompt = gsm8k_prompt(row['question'])
        try:
            texts, answers = [], []
            for _ in range(K_SE_SC):
                t, _ = generate_full(llama_mdl, llama_tok, prompt, temperature=1.0, max_new_tokens=512)
                texts.append(t)
                answers.append(extract_model_answer_gsm8k(t))
            p1_samples[k] = {'done': True, 'texts': texts, 'answers': answers,
                              'correct': p7[k]['correct']}
        except Exception as ex:
            print(f'  error {k}: {ex}')
            p1_samples[k] = {'done': False}
        if k % 25 == 0:
            save_cache(p1_samples, P1_SAMPLES_PATH)
        free_memory()

    save_cache(p1_samples, P1_SAMPLES_PATH)
    del llama_mdl, llama_tok; free_memory()
    print(f'P1 sampling done: {sum(v.get("done") for v in p1_samples.values())} / {len(sel_keys)}')'''))

CELLS.append(code('''# Cell 6 -- P1: Compute SE + SC AUCs and print table
p1_valid = [k for k in sel_keys if p1_samples.get(k, {}).get('done')]
p1_labels = np.array([int(p1_samples[k]['correct']) for k in p1_valid])

# Self-Consistency
sc_p1 = np.array([self_consistency_score(p1_samples[k]['answers'], normalize_fn=normalize_gsm8k)
                  for k in p1_valid])
sc_auc, sc_lo, sc_hi = boot_auc(p1_labels, sc_p1)

# Semantic Entropy (NLI) -- cache intermediate
SE_P1_PATH = os.path.join(CACHE_DIR, 'p1_se_scores.pkl')
if os.path.exists(SE_P1_PATH):
    se_p1 = np.array(load_cache(SE_P1_PATH))
else:
    se_p1 = []
    for k in tqdm(p1_valid, desc='P1 SE'):
        ans_extracts = [str(a) if a is not None else t[-80:]
                        for a, t in zip(p1_samples[k]['answers'], p1_samples[k]['texts'])]
        se_p1.append(official_semantic_entropy(ans_extracts, NLI_MDL, NLI_TOK, NLI_DEV, NLI_MODEL))
    se_p1 = np.array(se_p1)
    save_cache(se_p1.tolist(), SE_P1_PATH)

# orient SE (higher entropy = more uncertain = more likely WRONG)
se_auc_p, *_ = boot_auc(p1_labels,  se_p1)
se_auc_n, *_ = boot_auc(p1_labels, -se_p1)
se_auc = max(se_auc_p, se_auc_n)
_, se_lo, se_hi = boot_auc(p1_labels, se_p1 if se_auc_p >= se_auc_n else -se_p1)

n = len(p1_valid)
print()
print("=" * 85)
print(f"MATH -- GSM8K / Llama-3.1-8B / T=1.0  (n={n})")
print("=" * 85)
print(f"  {'Method':<40} {'AUROC':>8}  {'95% CI':>15}  {'Access':>12}  {'Compute':>8}")
print("-" * 85)
rows = [
    ("Nadler Spectral Fusion (ours, Phase 7)", 0.760, 0.725, 0.793, "Gray-box", "1-pass"),
    (f"Self-Consistency (K={K_SE_SC}) [computed]", sc_auc, sc_lo, sc_hi, "Black-box", f"K={K_SE_SC}"),
    (f"Semantic Entropy NLI (K={K_SE_SC}) [computed]", se_auc, se_lo, se_hi, "Black-box", f"K={K_SE_SC}"),
    ("SE reference: Mistral-7B (arXiv 2502.03799)", 0.7585, float('nan'), float('nan'), "Black-box", "K=10"),
    ("LapEigvals unsupervised (Phase 7)", 0.720, float('nan'), float('nan'), "White-box", "1-pass"),
    ("LapEigvals supervised (Phase 7)", 0.872, float('nan'), float('nan'), "White-box", "80% labeled"),
]
for method, auc, lo, hi, access, compute in rows:
    ci = f"[{100*lo:.1f},{100*hi:.1f}]" if lo == lo else "n/a"
    mark = " <--" if "ours" in method else ""
    print(f"  {method:<40} {100*auc:>7.1f}%  {ci:>15}  {access:>12}  {compute:>8}{mark}")
print("=" * 85)'''))

# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md("""## Priority 2 -- GPQA Diamond / Qwen2.5-7B: VC + SC + SE

**Published reference**: VC (K=1) = 74.6%, SC (K=8) = 75.4% averaged over reasoning models
(gpt-oss-20b, Qwen3-30B, DeepSeek-R1-8B) -- arXiv 2603.19118.
These are 4-10x stronger models; our Qwen-7B result is expected to be lower.
"""))

CELLS.append(code('''# Cell 7 -- GPQA: inference + K=10 sampling + VC
P2_SAMPLES_PATH = os.path.join(CACHE_DIR, 'p2_gpqa_qwen7b_k10.pkl')
P2_VC_PATH      = os.path.join(CACHE_DIR, 'p2_gpqa_qwen7b_vc.pkl')
FORCE_P2 = False

if not FORCE_P2 and os.path.exists(P2_SAMPLES_PATH) and os.path.exists(P2_VC_PATH):
    p2_samples = load_cache(P2_SAMPLES_PATH)
    p2_vc      = load_cache(P2_VC_PATH)
    print(f'P2 cache: {sum(v.get("done") for v in p2_samples.values())} done')
else:
    free_memory()
    gpqa_data = load_gpqa(split='test')
    N_gpqa    = min(N_GPQA, len(gpqa_data))
    p2_samples = load_cache(P2_SAMPLES_PATH) if os.path.exists(P2_SAMPLES_PATH) else {}
    p2_vc      = load_cache(P2_VC_PATH)      if os.path.exists(P2_VC_PATH)      else {}

    qwen_mdl, qwen_tok = load_model('Qwen/Qwen2.5-7B-Instruct')

    for i in tqdm(range(N_gpqa), desc='P2 GPQA K-sampling'):
        row      = gpqa_data[i]
        prompt, gold = gpqa_prompt_and_answer(row)

        if not p2_samples.get(i, {}).get('done'):
            try:
                texts, answers = [], []
                for _ in range(K_SE_SC):
                    t, _ = generate_full(qwen_mdl, qwen_tok, prompt, temperature=1.0, max_new_tokens=512)
                    texts.append(t)
                    m = re.search(r'\b([A-D])\b', t[-200:])
                    answers.append(m.group(1) if m else None)
                main_t, main_e = generate_full(qwen_mdl, qwen_tok, prompt, temperature=1.0, max_new_tokens=512)
                p2_samples[i] = {
                    'done': True, 'texts': texts, 'answers': answers,
                    'main_text': main_t, 'main_entropies': main_e,
                    'correct': int(is_correct_gpqa(main_t, gold)),
                }
            except Exception as ex:
                print(f'  error {i}: {ex}')
                p2_samples[i] = {'done': False}

        # VC: one extra prompt
        if not p2_vc.get(i, {}).get('done') and p2_samples.get(i, {}).get('done'):
            try:
                vc_p = prompt + '\\n\\n' + p2_samples[i]['main_text'] + VERBALIZED_CONF_SUFFIX
                vc_t, _ = generate_full(qwen_mdl, qwen_tok, vc_p, temperature=0.0, max_new_tokens=20)
                p2_vc[i] = {'done': True, 'conf': parse_verbalized_confidence(vc_t)}
            except Exception as ex:
                p2_vc[i] = {'done': False}

        if i % 25 == 0:
            save_cache(p2_samples, P2_SAMPLES_PATH)
            save_cache(p2_vc,      P2_VC_PATH)
        free_memory()

    save_cache(p2_samples, P2_SAMPLES_PATH)
    save_cache(p2_vc,      P2_VC_PATH)
    del qwen_mdl, qwen_tok; free_memory()
    print(f'P2 done: {sum(v.get("done") for v in p2_samples.values())}')'''))

CELLS.append(code('''# Cell 8 -- P2: Compute VC + SC + SE AUCs
p2_valid  = [k for k, v in p2_samples.items() if v.get('done')]
p2_labels = np.array([p2_samples[k]['correct'] for k in p2_valid])
print(f'GPQA valid: {len(p2_valid)}, accuracy={p2_labels.mean():.1%}')

# Verbalized Confidence
vc_p2_keys  = [k for k in p2_valid if p2_vc.get(k, {}).get('done')]
vc_labels   = np.array([p2_samples[k]['correct'] for k in vc_p2_keys])
vc_scores   = np.array([p2_vc[k]['conf'] for k in vc_p2_keys])
valid_vc    = ~np.isnan(vc_scores)
vc_auc, vc_lo, vc_hi = boot_auc(vc_labels[valid_vc], vc_scores[valid_vc]) if valid_vc.sum() >= 10 else (float('nan'),)*3

# Self-Consistency (letter answers A/B/C/D -- normalize to uppercase)
sc_p2 = np.array([
    self_consistency_score(p2_samples[k]['answers'], normalize_fn=lambda x: x.upper() if x else None)
    for k in p2_valid
])
sc_auc_p2, sc_lo_p2, sc_hi_p2 = boot_auc(p2_labels, sc_p2)

# Semantic Entropy
SE_P2_PATH = os.path.join(CACHE_DIR, 'p2_se_scores.pkl')
if os.path.exists(SE_P2_PATH):
    se_p2 = np.array(load_cache(SE_P2_PATH))
else:
    se_p2 = []
    for k in tqdm(p2_valid, desc='P2 SE'):
        ans = [a if a else p2_samples[k]['texts'][j][-60:]
               for j, a in enumerate(p2_samples[k]['answers'])]
        se_p2.append(official_semantic_entropy(ans, NLI_MDL, NLI_TOK, NLI_DEV, NLI_MODEL))
    se_p2 = np.array(se_p2)
    save_cache(se_p2.tolist(), SE_P2_PATH)
se_p2_auc_p, *_ = boot_auc(p2_labels,  se_p2)
se_p2_auc_n, *_ = boot_auc(p2_labels, -se_p2)
se_p2_auc = max(se_p2_auc_p, se_p2_auc_n)
_, se_p2_lo, se_p2_hi = boot_auc(p2_labels, se_p2 if se_p2_auc_p >= se_p2_auc_n else -se_p2)

print()
print("=" * 90)
print(f"SCIENCE -- GPQA Diamond / Qwen2.5-7B / T=1.0  (n={len(p2_valid)})")
print("=" * 90)
print(f"  {'Method':<48} {'AUROC':>8}  {'95% CI':>15}  Access  Compute")
print("-" * 90)
gpqa_rows = [
    ("Nadler Spectral Fusion (ours, Phase 4, Mistral-7B)", 0.654, float('nan'), float('nan'), "Gray-box", "1-pass"),
    ("Nadler Spectral Fusion (ours, Phase 8, Qwen2.5-72B)", 0.690, float('nan'), float('nan'), "Gray-box", "1-pass"),
    (f"Verbalized Confidence (K=1) [computed, Qwen-7B]", vc_auc, vc_lo, vc_hi, "Black-box", "1-pass"),
    (f"Self-Consistency (K={K_SE_SC}) [computed, Qwen-7B]", sc_auc_p2, sc_lo_p2, sc_hi_p2, "Black-box", f"K={K_SE_SC}"),
    (f"Semantic Entropy NLI (K={K_SE_SC}) [computed, Qwen-7B]", se_p2_auc, se_p2_lo, se_p2_hi, "Black-box", f"K={K_SE_SC}"),
    ("VC ref: reasoning models avg. (arXiv 2603.19118)", 0.746, float('nan'), float('nan'), "Black-box", "1-pass"),
    ("SC ref: reasoning models avg. (arXiv 2603.19118)", 0.754, float('nan'), float('nan'), "Black-box", "K=8"),
]
for method, auc, lo, hi, access, compute in gpqa_rows:
    ci  = f"[{100*lo:.1f},{100*hi:.1f}]" if lo == lo else "n/a"
    mark = " <--" if "ours" in method else ""
    print(f"  {method:<48} {100*auc:>7.1f}%  {ci:>15}  {access:<10}  {compute}{mark}")
print("=" * 90)'''))

# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md("""## Priority 3 -- RAG L-CiteEval HotpotQA / Qwen2.5-7B: SelfCheckGPT (K=5)

No published competitor uses L-CiteEval's citation grounding framing with AUROC.
LOS-Net (72.9%) is for standard HotpotQA -- different task.
We compute SelfCheckGPT as the closest available black-box baseline on our exact task.
"""))

CELLS.append(code('''# Cell 9 -- RAG: K=5 SelfCheckGPT samples (reuse Phase 10 inference cache)
P3_SAMPLES_PATH = os.path.join(CACHE_DIR, 'p3_rag_hotpotqa_qwen7b_k5.pkl')
FORCE_P3 = False

# Load Phase 10 cache to get existing labels
if os.path.exists(PHASE10_CACHE):
    p10 = load_cache(PHASE10_CACHE)
    p10_valid = sorted(k for k, v in p10.items() if v.get('done'))
    sel_rag   = p10_valid[:N_RAG]
    print(f'Phase 10 cache: {len(p10_valid)} valid entries, using {len(sel_rag)}')
    # determine label key
    sample = p10[sel_rag[0]]
    label_key = 'label' if 'label' in sample else 'grounding_label' if 'grounding_label' in sample else 'correct'
    print(f'Label key: {label_key}')
else:
    print('Phase 10 cache not found -- will run fresh inference')
    p10 = {}; sel_rag = list(range(N_RAG)); label_key = 'label'

if not FORCE_P3 and os.path.exists(P3_SAMPLES_PATH):
    p3_samples = load_cache(P3_SAMPLES_PATH)
    print(f'P3 cache: {sum(v.get("done") for v in p3_samples.values())} done')
else:
    free_memory()
    hotpot_data = load_lciteeval('hotpotqa', split='test', n=N_RAG)
    p3_samples  = load_cache(P3_SAMPLES_PATH) if os.path.exists(P3_SAMPLES_PATH) else {}

    qwen_mdl, qwen_tok = load_model('Qwen/Qwen2.5-7B-Instruct')

    for local_i, global_k in enumerate(tqdm(sel_rag, desc='P3 RAG K-sampling')):
        if p3_samples.get(global_k, {}).get('done'):
            continue
        row    = hotpot_data[local_i]
        prompt = lciteeval_prompt(row)
        label  = lciteeval_grounding_label(row)
        try:
            main_t, _ = generate_full(qwen_mdl, qwen_tok, prompt, temperature=1.0, max_new_tokens=512)
            sample_txts = []
            for _ in range(K_CHECK):
                t, _ = generate_full(qwen_mdl, qwen_tok, prompt, temperature=1.0, max_new_tokens=512)
                sample_txts.append(t)
            p3_samples[global_k] = {
                'done': True, 'main_text': main_t,
                'sample_texts': sample_txts, 'label': int(label),
            }
        except Exception as ex:
            print(f'  error {global_k}: {ex}')
            p3_samples[global_k] = {'done': False}
        if local_i % 25 == 0:
            save_cache(p3_samples, P3_SAMPLES_PATH)
        free_memory()

    save_cache(p3_samples, P3_SAMPLES_PATH)
    del qwen_mdl, qwen_tok; free_memory()
    print(f'P3 done: {sum(v.get("done") for v in p3_samples.values())}')'''))

CELLS.append(code('''# Cell 10 -- P3: SelfCheckGPT AUC + comparison table
p3_valid  = [k for k, v in p3_samples.items() if v.get('done')]
p3_labels = np.array([p3_samples[k]['label'] for k in p3_valid])

SC_P3_PATH = os.path.join(CACHE_DIR, 'p3_selfcheck_scores.pkl')
if os.path.exists(SC_P3_PATH):
    sc_p3 = np.array(load_cache(SC_P3_PATH))
else:
    sc_p3 = []
    for k in tqdm(p3_valid, desc='P3 SelfCheck'):
        s = selfcheck_nli_score(p3_samples[k]['main_text'], p3_samples[k]['sample_texts'],
                                NLI_MDL, NLI_TOK, NLI_DEV, NLI_MODEL)
        sc_p3.append(s)
    sc_p3 = np.array(sc_p3)
    save_cache(sc_p3.tolist(), SC_P3_PATH)

sc_p3_auc_p, *_ = boot_auc(p3_labels,  sc_p3)
sc_p3_auc_n, *_ = boot_auc(p3_labels, -sc_p3)
sc_p3_auc = max(sc_p3_auc_p, sc_p3_auc_n)
_, sc_p3_lo, sc_p3_hi = boot_auc(p3_labels, sc_p3 if sc_p3_auc_p >= sc_p3_auc_n else -sc_p3)

print()
print("=" * 90)
print(f"RAG -- L-CiteEval HotpotQA / Qwen2.5-7B / T=1.0  (n={len(p3_valid)})")
print("=" * 90)
print(f"  {'Method':<45} {'AUROC':>8}  {'95% CI':>15}  Access  Compute  Task")
print("-" * 90)
rag_rows = [
    ("Nadler Spectral Fusion (ours, Phase 10)", 0.877, float('nan'), float('nan'), "Gray-box", "1-pass", "L-CiteEval"),
    (f"SelfCheckGPT NLI (K={K_CHECK}) [computed, Qwen-7B]", sc_p3_auc, sc_p3_lo, sc_p3_hi, "Black-box", f"K={K_CHECK}", "L-CiteEval"),
    ("LOS-Net supervised (arXiv 2503.14043)", 0.7292, float('nan'), float('nan'), "Gray-box", "supervised", "std HotpotQA*"),
    ("Nadler Phase 10 / qwen7b / hotpotqa", 0.795, float('nan'), float('nan'), "Gray-box", "1-pass", "L-CiteEval"),
]
for method, auc, lo, hi, access, compute, task in rag_rows:
    ci   = f"[{100*lo:.1f},{100*hi:.1f}]" if lo == lo else "n/a"
    mark = " <--" if "ours" in method or "Phase 10" in method else ""
    print(f"  {method:<45} {100*auc:>7.1f}%  {ci:>15}  {access:<10}  {compute:<9}  {task}{mark}")
print("=" * 90)
print("  * LOS-Net uses standard HotpotQA (no citation markers) -- different task from L-CiteEval")'''))

# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md("""## Priority 4 -- MATH-500 / Qwen2.5-Math-7B: SC + SE (optional)

Only run if P1-P3 completed successfully. Phase 5 cache stores inference results for all 500 MATH-500 problems.
"""))

CELLS.append(code('''# Cell 11 -- MATH-500: K=10 sampling (uses Phase 5 cache for labels; needs fresh model run)
P4_SAMPLES_PATH = os.path.join(CACHE_DIR, 'p4_math500_qwen7b_k10.pkl')
FORCE_P4 = False

# Try to load Phase 5 label cache
if os.path.exists(PHASE5_CACHE):
    p5 = load_cache(PHASE5_CACHE)
    p5_valid = sorted(k for k, v in p5.items() if v.get('done'))
    # balanced subset: N_MATH from Phase 5
    l_all = np.array([int(p5[k]['correct']) for k in p5_valid])
    idx_c = [k for k, l in zip(p5_valid, l_all) if l == 1][:N_MATH // 2]
    idx_i = [k for k, l in zip(p5_valid, l_all) if l == 0][:N_MATH - len(idx_c)]
    sel_math = sorted(idx_c + idx_i)
    math500_data = load_math500(split='test')
    print(f'MATH-500 subset: {len(sel_math)} samples from Phase 5 cache')
else:
    print('Phase 5 MATH-500 cache not found -- adjust PHASE5_CACHE path in Cell 2')
    print('Skipping P4.')
    sel_math = []

if sel_math and not FORCE_P4 and os.path.exists(P4_SAMPLES_PATH):
    p4_samples = load_cache(P4_SAMPLES_PATH)
    print(f'P4 cache: {sum(v.get("done") for v in p4_samples.values())} done')
elif sel_math:
    free_memory()
    p4_samples = load_cache(P4_SAMPLES_PATH) if os.path.exists(P4_SAMPLES_PATH) else {}
    math_mdl, math_tok = load_model('Qwen/Qwen2.5-Math-7B-Instruct')

    for k in tqdm(sel_math, desc='P4 MATH-500 K-sampling'):
        if p4_samples.get(k, {}).get('done'):
            continue
        row    = math500_data[k]
        prompt = math_prompt(row['problem'])
        try:
            texts, answers = [], []
            for _ in range(K_SE_SC):
                t, _ = generate_full(math_mdl, math_tok, prompt, temperature=1.0, max_new_tokens=512)
                texts.append(t)
                # extract final boxed answer for MATH-500
                m = re.search(r'\\\\boxed\\{([^}]+)\\}', t)
                answers.append(m.group(1).strip() if m else None)
            p4_samples[k] = {'done': True, 'texts': texts, 'answers': answers,
                              'correct': int(p5[k]['correct'])}
        except Exception as ex:
            print(f'  error {k}: {ex}')
            p4_samples[k] = {'done': False}
        if k % 25 == 0:
            save_cache(p4_samples, P4_SAMPLES_PATH)
        free_memory()

    save_cache(p4_samples, P4_SAMPLES_PATH)
    del math_mdl, math_tok; free_memory()'''))

CELLS.append(code('''# Cell 12 -- P4: Compute SC + SE AUCs for MATH-500
if not sel_math:
    print('P4 skipped (no Phase 5 cache). Set PHASE5_CACHE path in Cell 2 and rerun.')
else:
    p4_valid  = [k for k in sel_math if p4_samples.get(k, {}).get('done')]
    p4_labels = np.array([p4_samples[k]['correct'] for k in p4_valid])

    # Self-Consistency (exact boxed answer match)
    sc_p4 = np.array([
        self_consistency_score(p4_samples[k]['answers'], normalize_fn=lambda x: x.strip() if x else None)
        for k in p4_valid
    ])
    sc_p4_auc, sc_p4_lo, sc_p4_hi = boot_auc(p4_labels, sc_p4)

    # Semantic Entropy
    SE_P4_PATH = os.path.join(CACHE_DIR, 'p4_se_scores.pkl')
    if os.path.exists(SE_P4_PATH):
        se_p4 = np.array(load_cache(SE_P4_PATH))
    else:
        se_p4 = []
        for k in tqdm(p4_valid, desc='P4 SE'):
            ans = [a if a else p4_samples[k]['texts'][j][-80:]
                   for j, a in enumerate(p4_samples[k]['answers'])]
            se_p4.append(official_semantic_entropy(ans, NLI_MDL, NLI_TOK, NLI_DEV, NLI_MODEL))
        se_p4 = np.array(se_p4)
        save_cache(se_p4.tolist(), SE_P4_PATH)
    se_p4_auc_p, *_ = boot_auc(p4_labels,  se_p4)
    se_p4_auc_n, *_ = boot_auc(p4_labels, -se_p4)
    se_p4_auc = max(se_p4_auc_p, se_p4_auc_n)
    _, se_p4_lo, se_p4_hi = boot_auc(p4_labels, se_p4 if se_p4_auc_p >= se_p4_auc_n else -se_p4)

    print()
    print("=" * 80)
    print(f"MATH -- MATH-500 / Qwen2.5-Math-7B / T=1.0  (n={len(p4_valid)})")
    print("=" * 80)
    rows = [
        ("Nadler Spectral Fusion (ours, Phase 5)", 0.900, float('nan'), float('nan'), "Gray-box", "1-pass"),
        (f"Self-Consistency (K={K_SE_SC}) [computed]", sc_p4_auc, sc_p4_lo, sc_p4_hi, "Black-box", f"K={K_SE_SC}"),
        (f"Semantic Entropy NLI (K={K_SE_SC}) [computed]", se_p4_auc, se_p4_lo, se_p4_hi, "Black-box", f"K={K_SE_SC}"),
        ("EDIS (agg. 4 math datasets, Qwen2.5-Math-1.5B)", 0.804, float('nan'), float('nan'), "Gray-box", "1-pass"),
    ]
    print(f"  {'Method':<45} {'AUROC':>8}  {'95% CI':>15}  Access  Compute")
    print("-" * 80)
    for method, auc, lo, hi, access, compute in rows:
        ci   = f"[{100*lo:.1f},{100*hi:.1f}]" if lo == lo else "n/a"
        mark = " <--" if "ours" in method else ""
        print(f"  {method:<45} {100*auc:>7.1f}%  {ci:>15}  {access:<10}  {compute}{mark}")
    print("=" * 80)'''))

# ─────────────────────────────────────────────────────────────────────────────
CELLS.append(md("## Summary -- Update Research_Phase12_Comparison_Tables.md"))

CELLS.append(code('''# Cell 13 -- Print fill-in values for Research_Phase12_Comparison_Tables.md
print("Values to fill into Research_Phase12_Comparison_Tables.md:")
print()
print("P1 -- GSM8K / Llama-3.1-8B:")
print(f"  SE NLI (K={K_SE_SC}): {100*se_auc:.1f}% [{100*se_lo:.1f},{100*se_hi:.1f}]")
print(f"  SC (K={K_SE_SC}):     {100*sc_auc:.1f}% [{100*sc_lo:.1f},{100*sc_hi:.1f}]")
print()
print("P2 -- GPQA / Qwen2.5-7B:")
print(f"  VC:                   {100*vc_auc:.1f}%")
print(f"  SC (K={K_SE_SC}):     {100*sc_auc_p2:.1f}% [{100*sc_lo_p2:.1f},{100*sc_hi_p2:.1f}]")
print(f"  SE NLI (K={K_SE_SC}): {100*se_p2_auc:.1f}% [{100*se_p2_lo:.1f},{100*se_p2_hi:.1f}]")
print()
print("P3 -- RAG HotpotQA / Qwen2.5-7B:")
print(f"  SelfCheckGPT NLI (K={K_CHECK}): {100*sc_p3_auc:.1f}% [{100*sc_p3_lo:.1f},{100*sc_p3_hi:.1f}]")'''))

# ─────────────────────────────────────────────────────────────────────────────
NB = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12.0'},
        'colab': {'provenance': []},
    },
    'cells': CELLS,
}

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)

print(f'Written: {NB_PATH}')
print(f'Cells: {len(CELLS)}')
