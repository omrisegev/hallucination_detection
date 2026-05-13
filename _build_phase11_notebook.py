"""
One-shot builder for Spectral_Analysis_Phase11_Agentic.ipynb.

Why a builder script: Phase 10 Main RAG taught us that 40-cell notebooks are
too large for NotebookEdit and awkward to write as one literal JSON blob.
Building cells as Python lists is faster to read, diff, and re-run.

Idempotent: rerunning produces the same notebook.
"""
import json
import os

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'Spectral_Analysis_Phase11_Agentic.ipynb')


def md(text: str) -> dict:
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': text.splitlines(keepends=True),
    }


CELLS = []


# ═══════════════════════════════════════════════════════════════════════
# Title + Section 1: Setup
# ═══════════════════════════════════════════════════════════════════════

CELLS.append(md("""# Phase 11 — Spectral Hallucination Detection for ReAct Agent Trajectories

**Goal**: Apply the spectral pipeline (per-token entropy → 13 spectral features → Nadler fusion) to 3-step ReAct agent trajectories on HotpotQA. Compare against AUQ verbalized confidence (prior art, Zhang et al. 2026, arXiv:2601.15703) and LOS-Net (supervised, arXiv:2503.14043).

**Models**: Qwen2.5-7B-Instruct, Mistral-Small-24B-Instruct (no AWQ/BNB pain — runnable in one Colab session)
**Dataset**: HotpotQA fullwiki validation, N=200 samples per model
**Tool**: simulated retriever over each question's 10-passage gold context
**Aggregations**: Φ_min, Φ_avg, Φ_last (AUQ convention)

See `Phase11_Agentic_Plan.md` for the full design rationale and gate definitions.
"""))

CELLS.append(md("## Section 1 — Setup"))

CELLS.append(code('''# Cell 1 — Clone repo + install deps + imports + freeze pyarrow
import os, sys, shutil

# Set BEFORE any torch import. Expandable segments help when the allocator must
# reclaim physical pages after a model is freed (critical for the second model).
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Persist HuggingFace cache to Drive — saves the re-download cost across runtime restarts.
os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'

REPO_DIR = '/content/hallucination_detection'

# Remove stale clone if spectral_utils is missing
if os.path.exists(REPO_DIR) and not os.path.exists(os.path.join(REPO_DIR, 'spectral_utils')):
    shutil.rmtree(REPO_DIR)

if not os.path.exists(REPO_DIR):
    os.system(f'git clone -b master https://github.com/omrisegev/hallucination_detection.git {REPO_DIR}')
else:
    os.system(f'git -C {REPO_DIR} pull -q')

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# autoawq is safe; we are NOT loading any AWQ model in Phase 11, so gptqmodel is unneeded.
os.system('pip install -q "transformers>=4.40" accelerate datasets bitsandbytes scipy scikit-learn')

from spectral_utils import (
    load_model, generate_full, free_memory,
    extract_all_features, sw_var_peak_with_window, sw_var_peak_adaptive,
    FEAT_NAMES, load_cache, save_cache,
    zscore, boot_auc, nadler_fuse, simple_average_fusion, best_nadler_on,
    load_hotpotqa,
    react_system_prompt, parse_action, parse_confidence,
    simulate_retrieve_tool, step_retrieved_supporting_fact,
    run_react_episode, aggregate_trajectory, categorize_failure_mode,
    branching_entropy, run_spiral_injection_replay,
    lite_semantic_entropy_for_statement,
)

# Force-load datasets BEFORE any later C-extension install. Phase 10's lesson:
# Colab can't unload pyarrow once it's been imported.
import datasets  # noqa: F401

print('spectral_utils imported OK')
print(f'FEAT_NAMES ({len(FEAT_NAMES)}): {FEAT_NAMES}')
'''))

CELLS.append(code('''# Cell 2 — Master config
import torch, numpy as np, pickle

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

TEMP             = 1.0
N_SAMPLES        = 200       # trajectories per model
MAX_STEPS        = 3         # ReAct steps per trajectory
MAX_NEW_PER_STEP = 256       # generation budget per step
AGGS             = ['min', 'avg', 'last']   # AUQ-style trajectory aggregations
SE_K             = 10        # SE MC samples per trajectory
SE_SUBSET        = 50        # SE is expensive — eval on a 50-trajectory subset

# Two models. Standard `load_model` defaults work for both.
MODELS = [
    ('qwen7b',     'Qwen/Qwen2.5-7B-Instruct',                  {'quantize_4bit': False}),
    ('mistral24b', 'mistralai/Mistral-Small-24B-Instruct-2501', {'quantize_4bit': False}),
]

DATASET = 'hotpotqa'

BASE_DIR  = '/content/drive/MyDrive/hallucination_detection/cache/phase11_agentic'
RAW_DIR   = f'{BASE_DIR}/raw'         # one .pkl per model — raw trajectories
FEAT_DIR  = f'{BASE_DIR}/features'    # one .pkl per model — extracted features + labels
RES_DIR   = f'{BASE_DIR}/results'     # persisted analysis result dicts
PLOT_DIR  = f'{BASE_DIR}/plots'

print(f'Models:   {[m[0] for m in MODELS]}')
print(f'Dataset:  {DATASET}, N={N_SAMPLES} trajectories per model')
print(f'ReAct:    max_steps={MAX_STEPS}, max_new_per_step={MAX_NEW_PER_STEP}')
'''))

CELLS.append(code('''# Cell 3 — Mount Drive + create cache dirs
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

for d in (RAW_DIR, FEAT_DIR, RES_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)
print('Cache dirs ready:')
for d in (RAW_DIR, FEAT_DIR, RES_DIR, PLOT_DIR):
    print(f'  {d}')
'''))

CELLS.append(code('''# Cell 3c — Flat-dir downloader for large models (Drive symlink workaround).
# Kept identical to Phase 10 even though Phase 11 only uses non-AWQ models, so the
# loader path is uniform across notebooks in this project.
from huggingface_hub import snapshot_download

FLAT_CACHE = '/content/drive/MyDrive/hf_cache_flat'
os.makedirs(FLAT_CACHE, exist_ok=True)

def ensure_flat_dir(repo_id, token=None):
    """Download repo to flat dir on Drive (real files, no symlinks). Idempotent."""
    local_dir = os.path.join(FLAT_CACHE, repo_id.replace('/', '__'))
    sentinel = os.path.join(local_dir, 'config.json')
    if os.path.exists(sentinel):
        return local_dir
    kwargs = dict(repo_id=repo_id, local_dir=local_dir, token=token)
    try:
        snapshot_download(**kwargs, local_dir_use_symlinks=False)
    except TypeError:
        snapshot_download(**kwargs)
    return local_dir

print('ensure_flat_dir() ready.')
'''))


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Data + tool env
# ═══════════════════════════════════════════════════════════════════════

CELLS.append(md("## Section 2 — Data + tool environment"))

CELLS.append(code('''# Cell 4 — Load HotpotQA fullwiki validation
ROWS = load_hotpotqa(n_samples=N_SAMPLES)
print(f'Loaded {len(ROWS)} HotpotQA rows.')
print(f'Sample question: {ROWS[0]["question"][:140]}')
print(f'Sample gold:     {ROWS[0]["answer"]}')
print(f'Supporting:      {ROWS[0]["supporting_facts"]["title"][:3]}')
print(f'Context paragraphs: {len(ROWS[0]["context"]["title"])}')
'''))

CELLS.append(code('''# Cell 5 — Spot-check the tool simulator on one sample (no model needed)
row = ROWS[0]
title, passage = simulate_retrieve_tool(row['question'], row['context'])
print(f'Query: {row["question"][:140]}')
print(f'Retrieved title: {title}')
print(f'Is supporting? {step_retrieved_supporting_fact(title, row["supporting_facts"]["title"])}')
print(f'Passage (first 200): {passage[:200]}')
'''))


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Inference
# ═══════════════════════════════════════════════════════════════════════

CELLS.append(md("## Section 3 — Agentic inference (per-model drivers)"))

CELLS.append(code('''# Cell 6 — Workhorse inference function
from tqdm.auto import tqdm

def raw_path(model_key):
    return os.path.join(RAW_DIR, f'{model_key}__{DATASET}.pkl')

def run_agentic_inference(mdl, tok, model_key, n_samples=N_SAMPLES,
                          checkpoint_every=10):
    """Run ReAct episodes for one model. Resumes from .pkl checkpoint if present."""
    path = raw_path(model_key)
    results = []
    if os.path.exists(path):
        with open(path, 'rb') as f:
            results = pickle.load(f)
        print(f'    [{model_key}] resumed from {len(results)} trajectories')

    start = len(results)
    if start >= n_samples:
        print(f'    [{model_key}] already complete ({len(results)}/{n_samples})')
        return

    for i in tqdm(range(start, n_samples), desc=f'{model_key}/{DATASET}'):
        row = ROWS[i]
        try:
            traj = run_react_episode(
                mdl, tok,
                question=row['question'],
                context=row['context'],
                supporting_titles=row['supporting_facts']['title'],
                gold_answer=row['answer'],
                T=TEMP, max_steps=MAX_STEPS,
                max_new_per_step=MAX_NEW_PER_STEP,
            )
            traj['idx'] = i
            results.append(traj)
        except Exception as ex:
            print(f'    error idx={i}: {ex}')
            continue
        if (i + 1) % checkpoint_every == 0:
            with open(path, 'wb') as f:
                pickle.dump(results, f)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print(f'    [{model_key}] complete: {len(results)}/{n_samples} saved to {path}')


def status_for_model(model_key):
    p = raw_path(model_key)
    n = len(pickle.load(open(p, 'rb'))) if os.path.exists(p) else 0
    print(f'  status [{model_key}]: {n}/{N_SAMPLES} trajectories')

print('Helpers ready.')
'''))

CELLS.append(md("### Driver — Qwen2.5-7B-Instruct"))

CELLS.append(code('''# Cell 7 — Qwen-7B
MODEL_KEY, MODEL_ID, KW = MODELS[0]
mdl, tok = load_model(MODEL_ID, **KW)
print(f'GPU: {torch.cuda.memory_allocated()/1e9:.1f} / '
      f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

run_agentic_inference(mdl, tok, MODEL_KEY)
del mdl, tok; free_memory()
status_for_model(MODEL_KEY)
'''))

CELLS.append(md("### Driver — Mistral-Small-24B-Instruct"))

CELLS.append(code('''# Cell 8 — Mistral-24B
MODEL_KEY, MODEL_ID, KW = MODELS[1]
mdl, tok = load_model(MODEL_ID, **KW)
print(f'GPU: {torch.cuda.memory_allocated()/1e9:.1f} / '
      f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

run_agentic_inference(mdl, tok, MODEL_KEY)
del mdl, tok; free_memory()
status_for_model(MODEL_KEY)
'''))


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Feature extraction
# ═══════════════════════════════════════════════════════════════════════

CELLS.append(md("## Section 4 — Feature extraction (per-step + per-trajectory aggregations)"))

CELLS.append(code('''# Cell 9 — Build ALL_CELLS: per-step spectral features + per-trajectory aggregations
#
# Per (model) cell, we store:
#   step_features:  list of list of dict — step_features[traj_i][step_k] = {feat: val, ...}
#   step_verb_conf: list of list of float — verbalized confidence per step
#   step_labels:    list of list of bool  — step_correct per step
#   traj_features:  dict of {agg: {feat: array(n_traj)}} — Φ_min/avg/last aggregations
#   traj_labels:    array of bool — trajectory_correct
#   failure_modes:  list of str  — per-trajectory failure category
#
# Persisted as feat_path so re-runs are fast.

FEAT_KEYS = None  # populated from the first non-empty feature dict

def feat_path(model_key):
    return os.path.join(FEAT_DIR, f'{model_key}__{DATASET}.pkl')

def extract_for_model(model_key, force=False):
    path = feat_path(model_key)
    if not force and os.path.exists(path):
        with open(path, 'rb') as f:
            cell = pickle.load(f)
        print(f'  [{model_key}] loaded from {path}: {len(cell["traj_labels"])} trajectories')
        return cell

    raw = pickle.load(open(raw_path(model_key), 'rb'))
    step_features  = []
    step_verb_conf = []
    step_labels    = []
    traj_labels    = []
    failure_modes  = []

    for traj in raw:
        per_step_f, per_step_c, per_step_l = [], [], []
        for s in traj['steps']:
            ents = s.get('token_entropies') or []
            f = extract_all_features(ents) if len(ents) >= 8 else None
            # Add Step-Boundary Uncertainty Trajectory (SBUT) feature — survey-recommended
            # 14th view. Robust to short token sequences (needs only ≥1 token).
            if f is not None and len(ents) >= 1:
                f['branching_entropy'] = branching_entropy(ents, window=3)
            per_step_f.append(f)
            per_step_c.append(s['confidence'])
            per_step_l.append(s['step_correct'])
        step_features.append(per_step_f)
        step_verb_conf.append(per_step_c)
        step_labels.append(per_step_l)
        traj_labels.append(traj['trajectory_correct'])
        failure_modes.append(categorize_failure_mode(traj))

    # Discover the feature key set from the first non-None step
    global FEAT_KEYS
    for traj_f in step_features:
        for s in traj_f:
            if s is not None:
                FEAT_KEYS = list(s.keys())
                break
        if FEAT_KEYS is not None:
            break
    if FEAT_KEYS is None:
        raise RuntimeError(f'No valid step features extracted for {model_key}')

    # Trajectory-level aggregations: Φ_min, Φ_avg, Φ_last per feature.
    # Also add verbalized confidence as a "feature" for symmetric handling.
    traj_features = {agg: {} for agg in AGGS}
    n = len(traj_labels)

    for agg in AGGS:
        for k in FEAT_KEYS:
            arr = np.full(n, np.nan)
            for i, traj_f in enumerate(step_features):
                vals = [s[k] for s in traj_f if s is not None and k in s]
                if vals:
                    arr[i] = aggregate_trajectory(vals, agg=agg)
            traj_features[agg][k] = arr

        # verbalized confidence
        arr_c = np.full(n, np.nan)
        for i, vc in enumerate(step_verb_conf):
            if vc:
                arr_c[i] = aggregate_trajectory(vc, agg=agg)
        traj_features[agg]['verb_conf'] = arr_c

    cell = {
        'step_features':  step_features,
        'step_verb_conf': step_verb_conf,
        'step_labels':    step_labels,
        'traj_features':  traj_features,
        'traj_labels':    np.asarray(traj_labels, dtype=bool),
        'failure_modes':  failure_modes,
        'n_traj':         n,
    }
    with open(path, 'wb') as f:
        pickle.dump(cell, f)
    print(f'  [{model_key}] saved {n} trajectories to {path}')
    return cell

ALL_CELLS = {}
for mk, _, _ in MODELS:
    ALL_CELLS[mk] = extract_for_model(mk)

# Cache the feature-key list at top level for downstream cells
for mk, c in ALL_CELLS.items():
    for traj_f in c['step_features']:
        for s in traj_f:
            if s is not None:
                FEAT_KEYS = list(s.keys()); break
        if FEAT_KEYS: break
    if FEAT_KEYS: break

print(f'\\nDiscovered {len(FEAT_KEYS)} spectral features per step: {FEAT_KEYS}')
print(f'Plus verb_conf → {len(FEAT_KEYS)+1} features per trajectory aggregation.')
'''))


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Per-cell analysis
# ═══════════════════════════════════════════════════════════════════════

CELLS.append(md("## Section 5 — Per-cell analysis"))

CELLS.append(code('''# Cell 10 — Pre-conditions G0 per cell
import pandas as pd

g0_rows = []
for mk, c in ALL_CELLS.items():
    labels = c['traj_labels']
    n_pos = int(labels.sum()); n_neg = int(len(labels) - n_pos)
    # mean step success rate (over all search-steps)
    step_succ = []
    for traj_l, traj_f in zip(c['step_labels'], c['step_features']):
        if traj_f:
            step_succ.extend(traj_l)
    sr = (sum(step_succ) / len(step_succ)) if step_succ else 0.0
    g0_rows.append({
        'model': mk,
        'n_traj': len(labels),
        'n_correct':   n_pos,
        'n_incorrect': n_neg,
        'mean_step_success': f'{100*sr:.1f}%',
        'G0-A (n≥150)':       '✓' if len(labels) >= 150 else '✗',
        'G0-B (both≥10)':     '✓' if (n_pos >= 10 and n_neg >= 10) else '✗',
    })
df_g0 = pd.DataFrame(g0_rows)
print('=== G0 Pre-conditions per cell ===')
print(df_g0.to_string(index=False))
'''))

CELLS.append(code('''# Cell 11 — Individual feature AUC table per (model, aggregation)
# Rows: (model, aggregation). Cols: 13 spectral features + verb_conf.

auc_records = []
for mk, c in ALL_CELLS.items():
    labels = c['traj_labels']
    for agg in AGGS:
        row = {'model': mk, 'agg': agg}
        for k in FEAT_KEYS + ['verb_conf']:
            scores = c['traj_features'][agg][k]
            mask = ~np.isnan(scores)
            if mask.sum() < 20:
                row[k] = float('nan')
                continue
            a, lo, hi = boot_auc(labels[mask], scores[mask])
            if not np.isnan(a) and a < 0.5:
                a, lo, hi = boot_auc(labels[mask], -scores[mask])
            row[k] = a
        auc_records.append(row)
df_auc = pd.DataFrame(auc_records)
print('=== Individual Feature AUCs (2 models × 3 aggs × 14 features) ===')
with pd.option_context('display.max_columns', None, 'display.width', 220,
                       'display.float_format', lambda x: f'{x:.3f}' if isinstance(x, float) else str(x)):
    print(df_auc.to_string(index=False))
'''))

CELLS.append(code('''# Cell 12 — Best Nadler subset + weights per (model, aggregation)
#
# Persistence-first pattern: in-memory → on-disk → recompute. Set FORCE_RECOMPUTE_NADLER=True to refresh.
NADLER_PATH = os.path.join(RES_DIR, 'nadler_res.pkl')
FORCE_RECOMPUTE_NADLER = False

if not FORCE_RECOMPUTE_NADLER and 'NADLER_RES' in globals() and NADLER_RES:
    print(f'NADLER_RES already in memory ({len(NADLER_RES)} cells); skipping.')
elif not FORCE_RECOMPUTE_NADLER and os.path.exists(NADLER_PATH):
    with open(NADLER_PATH, 'rb') as _f:
        NADLER_RES = pickle.load(_f)
    print(f'NADLER_RES loaded from {NADLER_PATH} ({len(NADLER_RES)} cells).')
else:
    NADLER_RES = {}  # (model, agg) -> {auc, lo, hi, subset, weights, sign_map}
    for mk, c in ALL_CELLS.items():
        labels = c['traj_labels']
        for agg in AGGS:
            # Spectral features only — exclude verb_conf and trace_length from this run;
            # we'll add verb_conf as a separate Nadler fusion in Cell 14, and check the
            # length confound in Cell 15.
            keys = [k for k in FEAT_KEYS if k != 'trace_length']
            feat_dict = {k: c['traj_features'][agg][k] for k in keys}
            mask = np.all([~np.isnan(feat_dict[k]) for k in keys], axis=0)
            if mask.sum() < 30:
                NADLER_RES[(mk, agg)] = {'auc': float('nan'), 'lo': float('nan'),
                                          'hi': float('nan'), 'subset': [], 'weights': [],
                                          'sign_map': {}}
                continue

            feat_dict_m = {k: v[mask] for k, v in feat_dict.items()}
            labels_m    = labels[mask]

            sign_map = {}
            for k in keys:
                ap, *_ = boot_auc(labels_m,  feat_dict_m[k])
                an, *_ = boot_auc(labels_m, -feat_dict_m[k])
                sign_map[k] = +1 if (not np.isnan(ap) and ap >= an) else -1

            auc, lo, hi, subset, weights = best_nadler_on(
                feat_dict_m, keys, labels_m,
                max_size=4, label=f'{mk}/{agg}', compare_mean=False,
            )
            NADLER_RES[(mk, agg)] = {
                'auc': auc, 'lo': lo, 'hi': hi,
                'subset': list(subset) if subset else [],
                'weights': list(weights) if weights is not None else [],
                'sign_map': sign_map,
                'n_used': int(mask.sum()),
            }
    with open(NADLER_PATH, 'wb') as _f:
        pickle.dump(NADLER_RES, _f)
    print(f'NADLER_RES saved to {NADLER_PATH}')

print('\\n=== Best spectral Nadler per (model, aggregation) ===')
for (mk, agg), r in NADLER_RES.items():
    sub_str = ' + '.join(r['subset']) if r['subset'] else '(none)'
    if np.isnan(r['auc']):
        print(f'  [{mk:<10s}/{agg:<5s}] AUC=NaN  (too few valid trajectories)')
    else:
        print(f'  [{mk:<10s}/{agg:<5s}] AUC={100*r["auc"]:5.1f}% [{100*r["lo"]:.1f},{100*r["hi"]:.1f}]  subset: {sub_str}')
'''))

CELLS.append(code('''# Cell 13 — AUQ baseline: verbalized confidence Φ_min / Φ_avg / Φ_last
#
# This is the prior-art SOTA from Zhang et al. 2026 (arXiv:2601.15703). On
# ALFWorld, AUQ paper reports Φ_min AUROC = 0.791 (vs ReAct baseline 0.667).
AUQ_PATH = os.path.join(RES_DIR, 'auq_res.pkl')
FORCE_RECOMPUTE_AUQ = False

if not FORCE_RECOMPUTE_AUQ and 'AUQ_RES' in globals() and AUQ_RES:
    print(f'AUQ_RES already in memory ({len(AUQ_RES)} cells); skipping.')
elif not FORCE_RECOMPUTE_AUQ and os.path.exists(AUQ_PATH):
    with open(AUQ_PATH, 'rb') as _f:
        AUQ_RES = pickle.load(_f)
    print(f'AUQ_RES loaded from {AUQ_PATH} ({len(AUQ_RES)} cells).')
else:
    AUQ_RES = {}
    for mk, c in ALL_CELLS.items():
        labels = c['traj_labels']
        for agg in AGGS:
            scores = c['traj_features'][agg]['verb_conf']
            mask = ~np.isnan(scores)
            if mask.sum() < 20:
                AUQ_RES[(mk, agg)] = {'auc': float('nan'), 'lo': float('nan'), 'hi': float('nan'),
                                       'n_used': int(mask.sum())}
                continue
            a, lo, hi = boot_auc(labels[mask], scores[mask])
            if not np.isnan(a) and a < 0.5:
                a, lo, hi = boot_auc(labels[mask], -scores[mask])
            AUQ_RES[(mk, agg)] = {'auc': a, 'lo': lo, 'hi': hi, 'n_used': int(mask.sum())}
    with open(AUQ_PATH, 'wb') as _f:
        pickle.dump(AUQ_RES, _f)
    print(f'AUQ_RES saved to {AUQ_PATH}')

print('\\n=== AUQ verbalized confidence AUC per (model, aggregation) ===')
for (mk, agg), r in AUQ_RES.items():
    if np.isnan(r["auc"]):
        print(f'  [{mk:<10s}/{agg:<5s}] AUC=NaN')
    else:
        print(f'  [{mk:<10s}/{agg:<5s}] AUC={100*r["auc"]:5.1f}% [{100*r["lo"]:.1f},{100*r["hi"]:.1f}]  (AUQ paper ALFWorld Φ_min: 79.1%)')
'''))

CELLS.append(code('''# Cell 14 — Spectral + AUQ Nadler fusion
#
# Adds verb_conf as a candidate Nadler view alongside the spectral features.
# If the Spearman ρ check (Cell 17) shows verb_conf is orthogonal to spectral,
# fusion should beat either signal alone — that's G3.
FUSION_PATH = os.path.join(RES_DIR, 'fusion_res.pkl')
FORCE_RECOMPUTE_FUSION = False

if not FORCE_RECOMPUTE_FUSION and 'FUSION_RES' in globals() and FUSION_RES:
    print(f'FUSION_RES already in memory ({len(FUSION_RES)} cells); skipping.')
elif not FORCE_RECOMPUTE_FUSION and os.path.exists(FUSION_PATH):
    with open(FUSION_PATH, 'rb') as _f:
        FUSION_RES = pickle.load(_f)
    print(f'FUSION_RES loaded from {FUSION_PATH} ({len(FUSION_RES)} cells).')
else:
    FUSION_RES = {}
    for mk, c in ALL_CELLS.items():
        labels = c['traj_labels']
        for agg in AGGS:
            keys = [k for k in FEAT_KEYS if k != 'trace_length'] + ['verb_conf']
            feat_dict = {k: c['traj_features'][agg][k] for k in keys}
            mask = np.all([~np.isnan(feat_dict[k]) for k in keys], axis=0)
            if mask.sum() < 30:
                FUSION_RES[(mk, agg)] = {'auc': float('nan'), 'subset': [], 'weights': []}
                continue
            feat_dict_m = {k: v[mask] for k, v in feat_dict.items()}
            labels_m    = labels[mask]

            auc, lo, hi, subset, weights = best_nadler_on(
                feat_dict_m, keys, labels_m,
                max_size=4, label=f'{mk}/{agg}-fused', compare_mean=False,
            )
            FUSION_RES[(mk, agg)] = {
                'auc': auc, 'lo': lo, 'hi': hi,
                'subset': list(subset) if subset else [],
                'weights': list(weights) if weights is not None else [],
                'verb_conf_in_subset': bool(subset and 'verb_conf' in subset),
                'n_used': int(mask.sum()),
            }
    with open(FUSION_PATH, 'wb') as _f:
        pickle.dump(FUSION_RES, _f)
    print(f'FUSION_RES saved to {FUSION_PATH}')

print('\\n=== Spectral + AUQ Nadler fusion per (model, aggregation) ===')
for (mk, agg), r in FUSION_RES.items():
    sub = ' + '.join(r['subset']) if r['subset'] else '(none)'
    vc  = '✓ verb_conf used' if r.get('verb_conf_in_subset') else 'verb_conf NOT in subset'
    if np.isnan(r['auc']):
        print(f'  [{mk:<10s}/{agg:<5s}] AUC=NaN')
    else:
        print(f'  [{mk:<10s}/{agg:<5s}] AUC={100*r["auc"]:5.1f}%  {vc:<25s} subset: {sub}')
'''))

CELLS.append(code('''# Cell 15 — Length-controlled analysis per cell (trace_length vs spectral-only)
#
# G4 check: does spectral fusion beat naive trace_length alone? Same control as
# Phase 10's Cell 15. Without this, a high AUC could just be "longer trajectories
# correlate with success" — a confound, not a signal.
LEN_PATH = os.path.join(RES_DIR, 'len_res.pkl')
FORCE_RECOMPUTE_LEN = False

if not FORCE_RECOMPUTE_LEN and 'LEN_RES' in globals() and LEN_RES:
    print(f'LEN_RES already in memory ({len(LEN_RES)} cells); skipping.')
elif not FORCE_RECOMPUTE_LEN and os.path.exists(LEN_PATH):
    with open(LEN_PATH, 'rb') as _f:
        LEN_RES = pickle.load(_f)
    print(f'LEN_RES loaded from {LEN_PATH} ({len(LEN_RES)} cells).')
else:
    LEN_RES = {}
    for mk, c in ALL_CELLS.items():
        labels = c['traj_labels']
        for agg in AGGS:
            tl = c['traj_features'][agg]['trace_length']
            mask_tl = ~np.isnan(tl)
            if mask_tl.sum() < 20:
                LEN_RES[(mk, agg)] = {}
                continue
            a_tl, lo_tl, hi_tl = boot_auc(labels[mask_tl], tl[mask_tl])
            if not np.isnan(a_tl) and a_tl < 0.5:
                a_tl, lo_tl, hi_tl = boot_auc(labels[mask_tl], -tl[mask_tl])

            spectral_keys = [k for k in FEAT_KEYS if k != 'trace_length']
            feat_dict = {k: c['traj_features'][agg][k] for k in spectral_keys}
            mask = np.all([~np.isnan(feat_dict[k]) for k in spectral_keys], axis=0)
            feat_dict_m = {k: v[mask] for k, v in feat_dict.items()}
            labels_m    = labels[mask]
            a_sp, lo_sp, hi_sp, sub_sp, w_sp = best_nadler_on(
                feat_dict_m, spectral_keys, labels_m,
                max_size=4, label=f'{mk}/{agg}-spectralonly', compare_mean=False,
            )
            LEN_RES[(mk, agg)] = {
                'trace_length_auc':   a_tl, 'trace_length_ci': (lo_tl, hi_tl),
                'spectral_only_auc':  a_sp, 'spectral_only_ci': (lo_sp, hi_sp),
                'spectral_only_subset':  list(sub_sp) if sub_sp else [],
                'lift_over_length_pp':   100 * (a_sp - a_tl),
            }
    with open(LEN_PATH, 'wb') as _f:
        pickle.dump(LEN_RES, _f)
    print(f'LEN_RES saved to {LEN_PATH}')

print('\\n=== Length-controlled summary ===')
print(f'{"cell":<22s}  {"len-only":>10s}  {"spec-only":>10s}  {"Δ (pp)":>10s}')
for (mk, agg), r in LEN_RES.items():
    if not r:
        print(f'  {mk+"/"+agg:<20s}  (insufficient data)'); continue
    print(f'  {mk+"/"+agg:<20s}  {100*r["trace_length_auc"]:>9.1f}%  '
          f'{100*r["spectral_only_auc"]:>9.1f}%  {r["lift_over_length_pp"]:>+9.1f}')
'''))

CELLS.append(code('''# Cell 16 — PCA diagnostic per cell (PC1 AUC vs Nadler AUC)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

PCA_PATH = os.path.join(RES_DIR, 'pca_res.pkl')
FORCE_RECOMPUTE_PCA = False

if not FORCE_RECOMPUTE_PCA and 'PCA_RES' in globals() and PCA_RES:
    print(f'PCA_RES already in memory ({len(PCA_RES)} cells); skipping.')
elif not FORCE_RECOMPUTE_PCA and os.path.exists(PCA_PATH):
    with open(PCA_PATH, 'rb') as _f:
        PCA_RES = pickle.load(_f)
    print(f'PCA_RES loaded from {PCA_PATH} ({len(PCA_RES)} cells).')
else:
    PCA_RES = {}
    for mk, c in ALL_CELLS.items():
        labels = c['traj_labels']
        for agg in AGGS:
            keys = [k for k in FEAT_KEYS if k != 'trace_length']
            X = np.column_stack([c['traj_features'][agg][k] for k in keys])
            mask = ~np.any(np.isnan(X), axis=1)
            if mask.sum() < 30:
                PCA_RES[(mk, agg)] = {}
                continue
            Xm = X[mask]; ym = labels[mask]
            Xs = StandardScaler().fit_transform(Xm)
            pca = PCA(n_components=min(5, Xm.shape[1])).fit(Xs)
            pc1 = pca.transform(Xs)[:, 0]
            a_pc1 = roc_auc_score(ym, pc1)
            if a_pc1 < 0.5:
                a_pc1 = roc_auc_score(ym, -pc1)
            loadings = pd.Series(np.abs(pca.components_[0]), index=keys).sort_values(ascending=False)
            nadler_auc = NADLER_RES.get((mk, agg), {}).get('auc', float('nan'))
            PCA_RES[(mk, agg)] = {
                'pc1_auc':         a_pc1,
                'pc1_var_ratio':   float(pca.explained_variance_ratio_[0]),
                'top3_loadings':   loadings.head(3).to_dict(),
                'nadler_lift_over_pc1_pp': 100 * (nadler_auc - a_pc1) if not np.isnan(nadler_auc) else float('nan'),
            }
    with open(PCA_PATH, 'wb') as _f:
        pickle.dump(PCA_RES, _f)
    print(f'PCA_RES saved to {PCA_PATH}')

print('\\n=== PCA vs Nadler ===')
print(f'{"cell":<22s}  {"PC1 AUC":>9s}  {"Nadler":>9s}  {"lift":>7s}  PC1 top3')
for (mk, agg), r in PCA_RES.items():
    if not r:
        print(f'  {mk+"/"+agg:<20s}  (insufficient data)'); continue
    nadler_auc = NADLER_RES.get((mk, agg), {}).get('auc', float('nan'))
    top3 = ', '.join(f'{k}({v:.2f})' for k, v in r['top3_loadings'].items())
    print(f'  {mk+"/"+agg:<20s}  {100*r["pc1_auc"]:>8.1f}%  '
          f'{100*nadler_auc:>8.1f}%  {r["nadler_lift_over_pc1_pp"]:>+6.1f}  {top3}')
'''))

CELLS.append(code('''# Cell 17 — Spearman ρ(EPR_step, verb_conf_step) — Direction 4's G1 critical gate
#
# Nadler fusion of spectral + verbalized confidence is only worthwhile if these
# two signals are decorrelated. ρ < 0.5 is the threshold Direction 4 specifies.
from scipy.stats import spearmanr

print('=== Spearman ρ(EPR_step, verb_conf_step) per model ===')
print('(pooled over all steps of all trajectories; |ρ| < 0.5 satisfies G1)\\n')

RHO_RES = {}
for mk, c in ALL_CELLS.items():
    epr_steps  = []
    conf_steps = []
    for traj_f, traj_c in zip(c['step_features'], c['step_verb_conf']):
        for s, cf in zip(traj_f, traj_c):
            if s is not None and 'epr' in s:
                epr_steps.append(s['epr'])
                conf_steps.append(cf)
    if len(epr_steps) >= 30:
        rho, p = spearmanr(epr_steps, conf_steps)
    else:
        rho, p = float('nan'), float('nan')
    RHO_RES[mk] = {'rho': rho, 'p': p, 'n': len(epr_steps)}
    flag = '✓ G1 pass' if (not np.isnan(rho) and abs(rho) < 0.5) else '✗ G1 fail'
    print(f'  [{mk:<10s}] n={len(epr_steps):4d}   ρ = {rho:+.3f}   p = {p:.2e}   {flag}')
'''))


CELLS.append(code('''# Cell 17b — Step Localization Accuracy (Φ_LSA) — AgentHallu's headline metric
#
# For each incorrect trajectory: does the detector's argmin-score step match
# the first actually-incorrect step? Compare against AgentHallu's 41.1% SOTA
# (Gemini 2.5 Pro on ALFWorld trajectories).
LSA_PATH = os.path.join(RES_DIR, 'lsa_res.pkl')
FORCE_RECOMPUTE_LSA = False

if not FORCE_RECOMPUTE_LSA and 'LSA_RES' in globals() and LSA_RES:
    print(f'LSA_RES already in memory; skipping.')
elif not FORCE_RECOMPUTE_LSA and os.path.exists(LSA_PATH):
    with open(LSA_PATH, 'rb') as _f:
        LSA_RES = pickle.load(_f)
    print(f'LSA_RES loaded from {LSA_PATH}.')
else:
    LSA_RES = {}

    # Per-model z-score params for each spectral feature (population = all steps)
    z_params = {}
    vc_params = {}
    for mk, c in ALL_CELLS.items():
        z_params[mk] = {}
        all_steps = [s for traj_f in c['step_features'] for s in traj_f if s is not None]
        for k in (FEAT_KEYS or []):
            vals = np.asarray([s[k] for s in all_steps if k in s], dtype=float)
            if vals.size:
                z_params[mk][k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
        vc_flat = [v for vc in c['step_verb_conf'] for v in vc if v is not None and not np.isnan(v)]
        vc_params[mk] = (float(np.mean(vc_flat)) if vc_flat else 0.0,
                          float(np.std(vc_flat))  if vc_flat else 1.0)

    def _per_step_score(mk, traj_i, step_k, detector):
        c = ALL_CELLS[mk]
        if step_k >= len(c['step_features'][traj_i]):
            return float('nan')
        f = c['step_features'][traj_i][step_k]
        # Score orientation: higher = MORE correct (so argmin = responsible step)
        if detector == 'EPR':
            return -f['epr'] if f is not None and 'epr' in f else float('nan')
        if detector == 'AUQ':
            v = c['step_verb_conf'][traj_i][step_k]
            return v if v is not None else float('nan')
        if detector in ('SpectralNadler', 'Fused'):
            res = (NADLER_RES if detector == 'SpectralNadler'
                   else FUSION_RES).get((mk, 'min'), {})
            sub = res.get('subset') or []
            wts = res.get('weights') or []
            signs = NADLER_RES.get((mk, 'min'), {}).get('sign_map', {})
            if not sub or f is None:
                return float('nan')
            score = 0.0
            for j, k in enumerate(sub):
                if k == 'verb_conf':
                    v = c['step_verb_conf'][traj_i][step_k]
                    if v is None: return float('nan')
                    mu, sd = vc_params[mk]
                    score += wts[j] * ((v - mu) / (sd + 1e-9))   # verb_conf already higher=correct
                else:
                    if k not in f: return float('nan')
                    mu, sd = z_params[mk].get(k, (0.0, 1.0))
                    z = (f[k] - mu) / (sd + 1e-9)
                    score += wts[j] * signs.get(k, 1) * z
            return score
        return float('nan')

    for mk, c in ALL_CELLS.items():
        per_det = {'EPR': [], 'AUQ': [], 'SpectralNadler': [], 'Fused': []}
        n_evaluable = 0
        n_random   = 0
        for traj_i, (traj_correct, step_labs, traj_f) in enumerate(
                zip(c['traj_labels'], c['step_labels'], c['step_features'])):
            if traj_correct:
                continue
            try:
                gt = next(k for k, ok in enumerate(step_labs) if not ok)
            except StopIteration:
                continue   # all steps individually correct but trajectory wrong → no GT step
            n_steps_valid = sum(1 for s in traj_f if s is not None)
            if n_steps_valid < 2:
                continue
            n_evaluable += 1
            # Random baseline: pick uniformly among valid step indices
            n_random += 1.0 / n_steps_valid
            for det in per_det:
                scores = [_per_step_score(mk, traj_i, k, det) for k in range(len(traj_f))]
                valid = [(k, s) for k, s in enumerate(scores) if not np.isnan(s)]
                if len(valid) < 2:
                    continue
                pred = min(valid, key=lambda x: x[1])[0]   # argmin = least correct
                per_det[det].append(pred == gt)
        LSA_RES[mk] = {
            det: {'phi_lsa': (float(np.mean(v)) if v else float('nan')),
                  'n': len(v)}
            for det, v in per_det.items()
        }
        LSA_RES[mk]['n_evaluable'] = n_evaluable
        LSA_RES[mk]['random_baseline'] = (n_random / n_evaluable) if n_evaluable else float('nan')

    with open(LSA_PATH, 'wb') as _f:
        pickle.dump(LSA_RES, _f)
    print(f'LSA_RES saved to {LSA_PATH}')

print('=== Step Localization Accuracy (Φ_LSA) — AgentHallu metric ===')
print('AgentHallu SOTA on ALFWorld trajectories: 41.1% (Gemini 2.5 Pro)')
print('Open-source models on AgentHallu: ~10.9% average\\n')
for mk in [m[0] for m in MODELS]:
    res = LSA_RES.get(mk, {})
    print(f'[{mk}] n_incorrect_evaluable = {res.get("n_evaluable", 0)}   '
          f'random baseline = {100*res.get("random_baseline", float("nan")):.1f}%')
    for det in ('EPR', 'AUQ', 'SpectralNadler', 'Fused'):
        v = res.get(det, {}).get('phi_lsa', float('nan'))
        n = res.get(det, {}).get('n', 0)
        bar = '' if np.isnan(v) else (' ✓' if v >= 0.411 else '')
        if np.isnan(v):
            print(f'    {det:<16s}  Φ_LSA = NaN          (n={n})')
        else:
            print(f'    {det:<16s}  Φ_LSA = {100*v:5.1f}%  (n={n}){bar}')
    print()
'''))


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Domain insights
# ═══════════════════════════════════════════════════════════════════════

CELLS.append(md("## Section 6 — Domain insights"))

CELLS.append(code('''# Cell 18 — Step-position analysis: which step's signal is most informative?
#
# For each model, restrict trajectories to those that ran exactly MAX_STEPS=3,
# then compute single-step AUC of EPR(step_k) and verb_conf(step_k) for k=1..3.
# Insight for the advisor presentation: does the agent's "weakest link" tend to
# be step 1 (planning), step 2 (intermediate), or step 3 (synthesis)?
STEP_RES = {}
print('=== Per-step single-feature AUC (trajectories with all 3 steps) ===')
print(f'{"model":<12s} {"feature":<12s} {"step1":>7s} {"step2":>7s} {"step3":>7s}')
for mk, c in ALL_CELLS.items():
    labels = c['traj_labels']
    # mask: trajectories that completed all 3 steps with valid features
    full_idx = [i for i, traj_f in enumerate(c['step_features'])
                if len(traj_f) == MAX_STEPS and all(s is not None for s in traj_f)]
    if len(full_idx) < 30:
        continue
    full_idx = np.asarray(full_idx)
    labels_full = labels[full_idx]

    for feat in ('epr', 'verb_conf'):
        aucs = []
        for k in range(MAX_STEPS):
            if feat == 'verb_conf':
                vals = np.asarray([c['step_verb_conf'][i][k] for i in full_idx])
            else:
                vals = np.asarray([c['step_features'][i][k][feat] for i in full_idx])
            a, *_ = boot_auc(labels_full, vals)
            if not np.isnan(a) and a < 0.5:
                a, *_ = boot_auc(labels_full, -vals)
            aucs.append(a)
        STEP_RES[(mk, feat)] = aucs
        print(f'{mk:<12s} {feat:<12s} ' + ' '.join(f'{100*a:6.1f}%' for a in aucs))
'''))

CELLS.append(code('''# Cell 19 — Failure-mode breakdown
# Categorize each incorrect trajectory: planning / tool / invalid / no_finish.
# Then evaluate detector recall conditional on failure mode.
print('=== Failure-mode breakdown per model ===\\n')
FAILURE_RES = {}
for mk, c in ALL_CELLS.items():
    fm = c['failure_modes']
    counts = pd.Series(fm).value_counts().to_dict()
    n = len(fm)
    print(f'[{mk}]  N={n}')
    for k in ('correct', 'planning', 'tool', 'invalid', 'no_finish'):
        v = counts.get(k, 0)
        print(f'    {k:<10s}: {v:4d} ({100*v/n:5.1f}%)')

    # Detector recall at FPR=20% conditional on each non-correct mode
    print(f'    {"":<10s}  recall@FPR=20%  for incorrect trajectories of this type')
    for agg in ('min',):  # focus on Φ_min, the AUQ headline aggregation
        for det_name, det_key, source in [
            ('AUQ',      None, AUQ_RES),
            ('Nadler',   None, NADLER_RES),
            ('Fused',    None, FUSION_RES),
        ]:
            r = source.get((mk, agg), {})
            auc = r.get('auc', float('nan'))
            print(f'    {det_name:<10s}  Φ_{agg} AUC = {100*auc:5.1f}%' if not np.isnan(auc) else f'    {det_name:<10s}  Φ_{agg} AUC = NaN')
    FAILURE_RES[mk] = counts
    print()
'''))

CELLS.append(code('''# Cell 20 — Lite Semantic Entropy baseline (per-trajectory, k=10 MC samples on the final answer)
#
# Expensive — only run on SE_SUBSET=50 random trajectories per model.
# Compares against the established sampling-based baseline (Farquhar et al. Nature 2024).
SE_PATH = os.path.join(RES_DIR, 'se_res.pkl')
FORCE_RECOMPUTE_SE = False

if not FORCE_RECOMPUTE_SE and 'SE_RES' in globals() and SE_RES:
    print(f'SE_RES already in memory ({len(SE_RES)} cells); skipping.')
elif not FORCE_RECOMPUTE_SE and os.path.exists(SE_PATH):
    with open(SE_PATH, 'rb') as _f:
        SE_RES = pickle.load(_f)
    print(f'SE_RES loaded from {SE_PATH} ({len(SE_RES)} cells).')
else:
    SE_RES = {}
    rng = np.random.default_rng(SEED)
    # SE requires a loaded model; if not running this section, leave SE_RES empty
    # and the gates / headline table will report SE as missing.
    print('NOTE: SE requires loading each model again. This cell is the only one in')
    print('Section 6 that needs a model — skip it (leave SE_RES empty) for analysis-only re-runs.')
    print('To compute: uncomment the loader block below.')

    # ── Uncomment to compute SE ───────────────────────────────────────────
    # for mk, MODEL_ID, KW in MODELS:
    #     raw = pickle.load(open(raw_path(mk), 'rb'))
    #     subset_idx = rng.choice(len(raw), size=min(SE_SUBSET, len(raw)), replace=False)
    #     mdl, tok = load_model(MODEL_ID, **KW)
    #     se_scores = []
    #     se_labels = []
    #     for i in tqdm(subset_idx, desc=f'SE {mk}'):
    #         traj = raw[i]
    #         q = traj['question']
    #         samples = []
    #         for _ in range(SE_K):
    #             r = generate_full(mdl, tok, q + '\\n\\nAnswer in one short phrase.',
    #                               T=1.0, max_new_tokens=64)
    #             samples.append(r['full_text'])
    #         se = lite_semantic_entropy_for_statement(samples)
    #         se_scores.append(se)
    #         se_labels.append(traj['trajectory_correct'])
    #     del mdl, tok; free_memory()
    #     se_scores = np.asarray(se_scores); se_labels = np.asarray(se_labels, dtype=bool)
    #     mask = ~np.isnan(se_scores)
    #     if mask.sum() >= 20 and se_labels[mask].sum() not in (0, mask.sum()):
    #         a, lo, hi = boot_auc(se_labels[mask], se_scores[mask])
    #         if not np.isnan(a) and a < 0.5:
    #             a, lo, hi = boot_auc(se_labels[mask], -se_scores[mask])
    #     else:
    #         a, lo, hi = float('nan'), float('nan'), float('nan')
    #     SE_RES[mk] = {'auc': a, 'lo': lo, 'hi': hi, 'n_used': int(mask.sum())}
    # with open(SE_PATH, 'wb') as _f:
    #     pickle.dump(SE_RES, _f)
    # print(f'SE_RES saved to {SE_PATH}')

if SE_RES:
    print('\\n=== Lite Semantic Entropy AUC (subset of trajectories) ===')
    for mk, r in SE_RES.items():
        if np.isnan(r['auc']):
            print(f'  [{mk}] AUC=NaN')
        else:
            print(f'  [{mk}] AUC={100*r["auc"]:.1f}% [{100*r["lo"]:.1f},{100*r["hi"]:.1f}]  n={r["n_used"]}')
'''))


CELLS.append(code('''# Cell 20c — Spiral-of-Hallucination injection diagnostic (Direction 4D)
#
# Replay SPIRAL_SUBSET originally-CORRECT trajectories with the step-1
# observation forcibly replaced by a distractor passage. Compare per-step
# EPR and verb_conf curves vs the originals. Hypothesis: spectral spikes
# at step 2 (right after the poisoned observation enters context) while
# verbalized confidence only spikes at step 3 (synthesis failure).
SPIRAL_PATH   = os.path.join(RES_DIR, 'spiral_res.pkl')
SPIRAL_SUBSET = 30
FORCE_RECOMPUTE_SPIRAL = False

if not FORCE_RECOMPUTE_SPIRAL and 'SPIRAL_RES' in globals() and SPIRAL_RES:
    print(f'SPIRAL_RES already in memory; skipping.')
elif not FORCE_RECOMPUTE_SPIRAL and os.path.exists(SPIRAL_PATH):
    with open(SPIRAL_PATH, 'rb') as _f:
        SPIRAL_RES = pickle.load(_f)
    print(f'SPIRAL_RES loaded from {SPIRAL_PATH}.')
else:
    SPIRAL_RES = {}
    print('NOTE: this cell reloads each model for SPIRAL_SUBSET replays.')
    print('Uncomment the block below to compute. Adds ~5–10 min per model.')

    # ── Uncomment to compute ────────────────────────────────────────────────
    # rng = np.random.default_rng(SEED)
    # for mk, MODEL_ID, KW in MODELS:
    #     raw = pickle.load(open(raw_path(mk), 'rb'))
    #     # Pick originally-correct trajectories with all MAX_STEPS steps recorded
    #     correct_idx = [i for i, t in enumerate(raw)
    #                    if t['trajectory_correct'] and len(t['steps']) >= MAX_STEPS]
    #     if len(correct_idx) < 5:
    #         print(f'  [{mk}] too few correct trajectories for spiral test'); continue
    #     subset = rng.choice(correct_idx, size=min(SPIRAL_SUBSET, len(correct_idx)), replace=False)
    #     mdl, tok = load_model(MODEL_ID, **KW)
    #     records = []
    #     for i in tqdm(subset, desc=f'spiral {mk}'):
    #         orig = raw[i]
    #         row  = ROWS[orig.get('idx', i)]
    #         supp = set(row['supporting_facts']['title'])
    #         distractor_idx = [j for j, t in enumerate(row['context']['title']) if t not in supp]
    #         if not distractor_idx: continue
    #         dj = int(rng.choice(distractor_idx))
    #         distractor = f"[{row['context']['title'][dj]}] " + ' '.join(row['context']['sentences'][dj])
    #         distractor = distractor[:700]
    #         injected = run_spiral_injection_replay(
    #             mdl, tok,
    #             question=row['question'], context=row['context'],
    #             supporting_titles=row['supporting_facts']['title'],
    #             gold_answer=row['answer'],
    #             original_step1=orig['steps'][0],
    #             distractor_passage=distractor,
    #             T=TEMP, max_steps=MAX_STEPS, max_new_per_step=MAX_NEW_PER_STEP,
    #         )
    #         records.append({'orig': orig, 'injected': injected})
    #     del mdl, tok; free_memory()
    #     SPIRAL_RES[mk] = records
    # with open(SPIRAL_PATH, 'wb') as _f:
    #     pickle.dump(SPIRAL_RES, _f)
    # print(f'SPIRAL_RES saved to {SPIRAL_PATH}')

if SPIRAL_RES:
    print('\\n=== Spiral injection — per-step response curves ===')
    print(f'{"model":<12s} {"step":<7s} {"EPR-orig":>10s} {"EPR-inj":>10s} {"Δ-EPR":>9s}  '
          f'{"VC-orig":>9s} {"VC-inj":>9s} {"Δ-VC":>8s}  '
          f'{"BE-orig":>9s} {"BE-inj":>9s} {"Δ-BE":>8s}')
    for mk, recs in SPIRAL_RES.items():
        for step_k in range(MAX_STEPS):
            e_o, e_i, v_o, v_i, be_o, be_i = [], [], [], [], [], []
            for r in recs:
                if len(r['orig']['steps']) > step_k:
                    s = r['orig']['steps'][step_k]
                    ents = s.get('token_entropies') or []
                    if ents:
                        e_o.append(float(np.mean(ents)))
                        be_o.append(float(np.mean(ents[:3])))
                    v_o.append(s.get('confidence', np.nan))
                if len(r['injected']['steps']) > step_k:
                    s = r['injected']['steps'][step_k]
                    ents = s.get('token_entropies') or []
                    if ents:
                        e_i.append(float(np.mean(ents)))
                        be_i.append(float(np.mean(ents[:3])))
                    v_i.append(s.get('confidence', np.nan))
            mean_e_o = np.mean(e_o) if e_o else np.nan
            mean_e_i = np.mean(e_i) if e_i else np.nan
            mean_v_o = np.nanmean(v_o) if v_o else np.nan
            mean_v_i = np.nanmean(v_i) if v_i else np.nan
            mean_be_o = np.mean(be_o) if be_o else np.nan
            mean_be_i = np.mean(be_i) if be_i else np.nan
            print(f'{mk:<12s} step{step_k+1:<3d}   {mean_e_o:>10.3f} {mean_e_i:>10.3f} '
                  f'{mean_e_i-mean_e_o:>+9.3f}  '
                  f'{mean_v_o:>9.3f} {mean_v_i:>9.3f} {mean_v_i-mean_v_o:>+8.3f}  '
                  f'{mean_be_o:>9.3f} {mean_be_i:>9.3f} {mean_be_i-mean_be_o:>+8.3f}')

    # Cascade-onset test: at which step does the signal first detect the injection?
    print('\\n=== Cascade-onset: which step shows the largest Δ for each signal? ===')
    print('Hypothesis: spectral (EPR/branching) peaks at step 2; verb_conf peaks at step 3.')
    for mk, recs in SPIRAL_RES.items():
        deltas = {'EPR': [], 'verb_conf': [], 'branching': []}
        for step_k in range(MAX_STEPS):
            for sig in deltas:
                d_vals = []
                for r in recs:
                    if (len(r['orig']['steps']) > step_k
                        and len(r['injected']['steps']) > step_k):
                        so = r['orig']['steps'][step_k]
                        si = r['injected']['steps'][step_k]
                        if sig == 'EPR' and so.get('token_entropies') and si.get('token_entropies'):
                            d_vals.append(np.mean(si['token_entropies']) - np.mean(so['token_entropies']))
                        elif sig == 'verb_conf':
                            d_vals.append(so['confidence'] - si['confidence'])  # confidence DROP → positive
                        elif sig == 'branching' and so.get('token_entropies') and si.get('token_entropies'):
                            d_vals.append(np.mean(si['token_entropies'][:3]) - np.mean(so['token_entropies'][:3]))
                deltas[sig].append(np.mean(d_vals) if d_vals else np.nan)
        print(f'[{mk}]')
        for sig, ds in deltas.items():
            if all(np.isnan(d) for d in ds):
                continue
            peak_step = int(np.nanargmax(np.abs(ds))) + 1
            print(f'    {sig:<12s} per-step Δ = ' +
                  ', '.join(f"step{k+1}:{d:+.3f}" for k, d in enumerate(ds)) +
                  f'   →  peak at step {peak_step}')
'''))


# ═══════════════════════════════════════════════════════════════════════
# Section 7: SOTA comparisons
# ═══════════════════════════════════════════════════════════════════════

CELLS.append(md("## Section 7 — Headline comparison vs SOTA"))

CELLS.append(code('''# Cell 21 — Detector AUC headline table
#
# Rows: (model, aggregation)
# Cols: EPR-only, AUQ verbalized, Spectral Nadler, Spectral+AUQ Nadler, SE (if computed)
# Reference rows at the bottom: AUQ paper SOTA, LOS-Net.

headline_rows = []
for mk, c in ALL_CELLS.items():
    labels = c['traj_labels']
    for agg in AGGS:
        # EPR-only (single-view baseline)
        epr_scores = c['traj_features'][agg]['epr']
        mask = ~np.isnan(epr_scores)
        if mask.sum() >= 20:
            a, *_ = boot_auc(labels[mask], epr_scores[mask])
            if not np.isnan(a) and a < 0.5:
                a, *_ = boot_auc(labels[mask], -epr_scores[mask])
            epr_only = a
        else:
            epr_only = float('nan')

        auq    = AUQ_RES.get((mk, agg), {}).get('auc', float('nan'))
        nadler = NADLER_RES.get((mk, agg), {}).get('auc', float('nan'))
        fused  = FUSION_RES.get((mk, agg), {}).get('auc', float('nan'))
        se     = SE_RES.get(mk, {}).get('auc', float('nan')) if agg == 'min' else float('nan')

        headline_rows.append({
            'model':           mk,
            'agg':             f'Φ_{agg}',
            'EPR-only':        epr_only,
            'AUQ verb_conf':   auq,
            'Spectral Nadler': nadler,
            'Spectral+AUQ':    fused,
            'SE (trajectory)': se,
        })

df_headline = pd.DataFrame(headline_rows)
print('=== Detector AUC headline (this notebook) ===')
with pd.option_context('display.float_format', lambda x: f'{100*x:.1f}%' if isinstance(x, float) and not np.isnan(x) else '  -  '):
    print(df_headline.to_string(index=False))

print('\\n=== SOTA reference (different datasets, different setups) ===')
print(f'  AUQ paper (ALFWorld, 7B+, Φ_min):    79.1%      Zhang et al. 2026 (arXiv:2601.15703)')
print(f'  AUQ paper (WebShop,  7B+, Φ_min):    75.5%      ibid.')
print(f'  AUQ paper (ReAct baseline ALFWorld):  66.7%      ← we want EPR-only to beat this')
print(f'  LOS-Net (HotpotQA, Mistral-7B, sup.): 72.9%      arXiv:2503.14043')
print(f'  Phase 10 RAG (HotpotQA, qwen7b):      79.5%      same-paper reference point')
print()
print('=== Φ_LSA (Step Localization Accuracy) — AgentHallu metric, see Cell 17b ===')
print(f'  AgentHallu SOTA (Gemini 2.5 Pro):       41.1%      Wang et al. 2026 (arXiv:2601.06818)')
print(f'  AgentHallu open-source avg:             10.9%      ibid.')
print('  Our Φ_LSA per detector printed in Cell 17b output above.')
'''))


# ═══════════════════════════════════════════════════════════════════════
# Section 8: Gates + save + plots
# ═══════════════════════════════════════════════════════════════════════

CELLS.append(md("## Section 8 — Gates + final save"))

CELLS.append(code('''# Cell 22 — Gate evaluation (G0–G5)
print('=== Gate evaluation ===\\n')

# G0 — already evaluated in Cell 10; here we collect into the verdict
g0_pass = all(len(c['traj_labels']) >= 150
              and c['traj_labels'].sum() >= 10
              and (len(c['traj_labels']) - c['traj_labels'].sum()) >= 10
              for c in ALL_CELLS.values())

# G1 — ρ(EPR_step, verb_conf_step) < 0.5 on ≥1 model
g1_models = [mk for mk, r in RHO_RES.items()
             if not np.isnan(r['rho']) and abs(r['rho']) < 0.5]
g1_pass = len(g1_models) >= 1

# G2 — Best spectral Nadler Φ_min ≥ 0.70 on ≥1 model
g2_aucs = [r['auc'] for (mk, agg), r in NADLER_RES.items()
           if agg == 'min' and not np.isnan(r['auc'])]
g2_pass = any(a >= 0.70 for a in g2_aucs)
g2_best = max(g2_aucs) if g2_aucs else float('nan')

# G3 — Spectral+AUQ beats AUQ-alone by ≥3pp on ≥1 cell
g3_diffs = []
for key in FUSION_RES:
    f = FUSION_RES[key].get('auc', float('nan'))
    a = AUQ_RES.get(key, {}).get('auc', float('nan'))
    if not (np.isnan(f) or np.isnan(a)):
        g3_diffs.append((key, f - a))
g3_pass = any(d >= 0.03 for _, d in g3_diffs)
g3_best = max((d for _, d in g3_diffs), default=float('nan'))

# G4 — Spectral-only beats trace_length by ≥3pp on ≥1 cell
g4_diffs = []
for key, r in LEN_RES.items():
    if r:
        g4_diffs.append((key, r['spectral_only_auc'] - r['trace_length_auc']))
g4_pass = any(d >= 0.03 for _, d in g4_diffs)
g4_best = max((d for _, d in g4_diffs), default=float('nan'))

# G5 — Spectral+AUQ Φ_min ≥ 0.791 on ≥1 model
g5_pass = any(FUSION_RES.get((mk, 'min'), {}).get('auc', 0) >= 0.791
              for mk, _, _ in MODELS)
g5_best = max((FUSION_RES.get((mk, 'min'), {}).get('auc', 0) for mk, _, _ in MODELS), default=0)

print(f'G0 — n≥150 per model, both classes ≥10            {"✓" if g0_pass else "✗"}')
print(f'G1 — ρ(EPR_step, verb_conf_step) < 0.5 on ≥1 model  {"✓" if g1_pass else "✗"}  '
      f'(models passing: {g1_models or "[]"})')
print(f'G2 — Best spectral Nadler Φ_min ≥ 70%               {"✓" if g2_pass else "✗"}  '
      f'(best = {100*g2_best:.1f}% if not NaN)')
print(f'G3 — Spectral+AUQ beats AUQ-alone by ≥3pp             {"✓" if g3_pass else "✗"}  '
      f'(best lift = {100*g3_best:+.1f}pp)')
print(f'G4 — Spectral-only beats trace_length by ≥3pp         {"✓" if g4_pass else "✗"}  '
      f'(best lift = {100*g4_best:+.1f}pp)')
print(f'G5 — Spectral+AUQ Φ_min ≥ 79.1% (AUQ paper SOTA)     {"✓" if g5_pass else "✗"}  '
      f'(best = {100*g5_best:.1f}%)')
'''))

CELLS.append(code('''# Cell 23 — Save comprehensive results pickle
results = {
    'config': {
        'models': [(mk, mid, kw) for mk, mid, kw in MODELS],
        'dataset': DATASET,
        'n_samples': N_SAMPLES,
        'max_steps': MAX_STEPS,
        'temperature': TEMP,
        'aggs': AGGS,
    },
    'NADLER_RES':  NADLER_RES,
    'AUQ_RES':     AUQ_RES,
    'FUSION_RES':  FUSION_RES,
    'LEN_RES':     LEN_RES,
    'PCA_RES':     PCA_RES,
    'RHO_RES':     RHO_RES,
    'STEP_RES':    STEP_RES,
    'FAILURE_RES': FAILURE_RES,
    'SE_RES':      SE_RES,
    'LSA_RES':     LSA_RES,
    'SPIRAL_RES':  SPIRAL_RES,
    'headline_table': df_headline.to_dict(orient='records'),
    'gates': {'g0': g0_pass, 'g1': g1_pass, 'g2': g2_pass, 'g3': g3_pass, 'g4': g4_pass, 'g5': g5_pass},
}
out_path = os.path.join(RES_DIR, 'phase11_results.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
print(f'Saved comprehensive results to {out_path}')
'''))

CELLS.append(code('''# Cell 24 — Plots
import matplotlib.pyplot as plt

# Plot 1 — Detector AUC heatmap
fig, ax = plt.subplots(figsize=(9, 4.5))
det_cols = ['EPR-only', 'AUQ verb_conf', 'Spectral Nadler', 'Spectral+AUQ']
hm = df_headline.set_index(['model', 'agg'])[det_cols].values
ax.imshow(hm, aspect='auto', cmap='viridis', vmin=0.5, vmax=0.85)
ax.set_xticks(range(len(det_cols))); ax.set_xticklabels(det_cols, rotation=20, ha='right')
ax.set_yticks(range(len(df_headline)))
ax.set_yticklabels([f'{r["model"]}/{r["agg"]}' for _, r in df_headline.iterrows()])
for i, row in df_headline.iterrows():
    for j, col in enumerate(det_cols):
        v = row[col]
        if not np.isnan(v):
            ax.text(j, i, f'{100*v:.1f}', ha='center', va='center',
                    color='white' if v < 0.72 else 'black', fontsize=9)
ax.set_title('Phase 11 — Detector AUC heatmap (HotpotQA agent trajectories)')
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, 'detector_heatmap.png'), dpi=120); plt.show()

# Plot 2 — ρ(EPR_step, verb_conf_step) bar
fig, ax = plt.subplots(figsize=(6, 3))
mks = list(RHO_RES.keys())
rhos = [RHO_RES[m]['rho'] for m in mks]
ax.bar(mks, rhos, color=['steelblue' if abs(r) < 0.5 else 'crimson' for r in rhos])
ax.axhline(0.5, color='red', ls='--', lw=1, label='G1 threshold')
ax.axhline(-0.5, color='red', ls='--', lw=1)
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Spearman ρ'); ax.set_title('ρ(EPR_step, verb_conf_step) per model')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'rho_bar.png'), dpi=120); plt.show()

# Plot 3 — Per-step AUC bars
if STEP_RES:
    fig, axes = plt.subplots(1, len(ALL_CELLS), figsize=(4*len(ALL_CELLS), 3), sharey=True)
    if len(ALL_CELLS) == 1: axes = [axes]
    for ax, (mk, _) in zip(axes, ALL_CELLS.items()):
        epr_a = STEP_RES.get((mk, 'epr'), [np.nan]*MAX_STEPS)
        conf_a = STEP_RES.get((mk, 'verb_conf'), [np.nan]*MAX_STEPS)
        x = np.arange(MAX_STEPS)
        w = 0.35
        ax.bar(x - w/2, [100*a for a in epr_a], w, label='EPR', color='steelblue')
        ax.bar(x + w/2, [100*a for a in conf_a], w, label='verb_conf', color='orange')
        ax.set_xticks(x); ax.set_xticklabels([f'step {k+1}' for k in range(MAX_STEPS)])
        ax.set_ylim(45, 80); ax.axhline(50, color='black', lw=0.5)
        ax.set_title(f'{mk} — per-step AUC')
        ax.set_ylabel('AUC (%)'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'step_position.png'), dpi=120); plt.show()

# Plot 4 — Failure mode counts
fig, ax = plt.subplots(figsize=(8, 3))
labels = ['correct', 'planning', 'tool', 'invalid', 'no_finish']
x = np.arange(len(labels))
w = 0.35
for i, (mk, counts) in enumerate(FAILURE_RES.items()):
    vals = [counts.get(l, 0) for l in labels]
    ax.bar(x + (i-0.5)*w, vals, w, label=mk)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Trajectories'); ax.set_title('Failure-mode breakdown')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'failure_modes.png'), dpi=120); plt.show()

# Plot 5 — Φ_LSA per detector per model
if LSA_RES:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    dets = ['EPR', 'AUQ', 'SpectralNadler', 'Fused']
    mks = [m[0] for m in MODELS]
    x = np.arange(len(dets))
    w = 0.35
    for i, mk in enumerate(mks):
        vals = [LSA_RES.get(mk, {}).get(d, {}).get('phi_lsa', np.nan) for d in dets]
        ax.bar(x + (i - 0.5) * w, [100*v if not np.isnan(v) else 0 for v in vals], w, label=mk)
    ax.axhline(41.1, color='red', ls='--', lw=1, label='AgentHallu SOTA (41.1%)')
    rb = np.nanmean([LSA_RES.get(mk, {}).get('random_baseline', np.nan) for mk in mks])
    if not np.isnan(rb):
        ax.axhline(100 * rb, color='gray', ls=':', lw=1, label=f'Random ({100*rb:.0f}%)')
    ax.set_xticks(x); ax.set_xticklabels(dets)
    ax.set_ylabel('Φ_LSA (%)'); ax.set_title('Step Localization Accuracy — AgentHallu metric')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'phi_lsa.png'), dpi=120); plt.show()

# Plot 6 — Spiral injection response curves (only if SPIRAL_RES was computed)
if SPIRAL_RES:
    fig, axes = plt.subplots(1, len(SPIRAL_RES), figsize=(4.5*len(SPIRAL_RES), 3.5), sharey=False)
    if len(SPIRAL_RES) == 1: axes = [axes]
    for ax, (mk, recs) in zip(axes, SPIRAL_RES.items()):
        e_o_curve, e_i_curve, v_o_curve, v_i_curve = [], [], [], []
        for step_k in range(MAX_STEPS):
            e_o = [np.mean(r['orig']['steps'][step_k]['token_entropies'])
                   for r in recs if len(r['orig']['steps']) > step_k
                   and r['orig']['steps'][step_k].get('token_entropies')]
            e_i = [np.mean(r['injected']['steps'][step_k]['token_entropies'])
                   for r in recs if len(r['injected']['steps']) > step_k
                   and r['injected']['steps'][step_k].get('token_entropies')]
            v_o = [r['orig']['steps'][step_k]['confidence']
                   for r in recs if len(r['orig']['steps']) > step_k]
            v_i = [r['injected']['steps'][step_k]['confidence']
                   for r in recs if len(r['injected']['steps']) > step_k]
            e_o_curve.append(np.mean(e_o) if e_o else np.nan)
            e_i_curve.append(np.mean(e_i) if e_i else np.nan)
            v_o_curve.append(np.nanmean(v_o) if v_o else np.nan)
            v_i_curve.append(np.nanmean(v_i) if v_i else np.nan)
        xs = np.arange(1, MAX_STEPS + 1)
        ax2 = ax.twinx()
        l1, = ax.plot(xs, e_o_curve, 'o-', color='steelblue', label='EPR orig')
        l2, = ax.plot(xs, e_i_curve, 's--', color='crimson', label='EPR injected')
        l3, = ax2.plot(xs, v_o_curve, '^-', color='lightblue', label='verb_conf orig')
        l4, = ax2.plot(xs, v_i_curve, 'v--', color='lightcoral', label='verb_conf injected')
        ax.set_xticks(xs); ax.set_xlabel('Step')
        ax.set_ylabel('Mean EPR'); ax2.set_ylabel('Mean verb_conf')
        ax.set_title(f'{mk} — spiral response (injection at step 1)')
        ax.axvline(1.0, color='gray', ls=':', alpha=0.4)
        ax.legend([l1, l2, l3, l4], ['EPR orig', 'EPR injected', 'VC orig', 'VC injected'],
                  loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'spiral_response.png'), dpi=120); plt.show()

print('All plots saved to', PLOT_DIR)
'''))

CELLS.append(md("""## Section 9 — Notes for advisor presentation

**For Ofir (spectral / multiview)**
- **Cell 17 + plot 2** answer the orthogonality question: is the spectral signal genuinely complementary to verbalized confidence, or do they collapse into one view? If |ρ| < 0.5, Nadler is appropriate by construction.
- **Cell 12 Nadler subset table** shows which spectral features survive subset search on agent trajectories — compare against Phase 10 RAG's per-statement subsets to argue domain-generality of the core spectral feature set.

**For Bracha (calibration / baselines)**
- **Cell 21 headline table** sits the spectral approach next to AUQ verbalized confidence (prior-art SOTA) under the same evaluation harness. The Spectral+AUQ row is the key number.
- **Cells 13 + 14** are the AUQ-only vs Fused contrast — the lift (or lack of it) is the direct argument for adding spectral on top of an existing AUQ-style deployment.
- **Cell 22 gate evaluation** explicit pass/fail thresholds set up Direction 5 (LTT conformal calibration) as the next thesis step on whichever score beat the gates.

**Open questions to surface in the meeting**
1. Is ρ(EPR, verb_conf) low enough on both models, or only on one? Domain-conditional answer.
2. Does spectral lift load more on Φ_min (catastrophic-step detection) or Φ_avg (overall track-record)? Implications for deployment.
3. Failure-mode breakdown (Cell 19) — do tool-fabrication and planning errors have different uncertainty signatures?
4. Φ_LSA (Cell 17b) — for the responsible-step-localisation task, does spectral fusion beat verbalized confidence? This is AgentHallu's headline metric — Gemini 2.5 Pro SOTA at 41.1%, open-source average 10.9%.
5. Spiral diagnostic (Cell 20c, optional) — does EPR / branching_entropy show the injected error at step 2 while verbalized confidence only catches up at step 3? Direct evidence for the cascade-detection claim.
6. Branching entropy (14th feature, added in Cell 9) — does it survive Cell 12's Nadler subset search? Survey-recommended SBUT signal — its inclusion or exclusion is itself a thesis-worthy result.
"""))


# ═══════════════════════════════════════════════════════════════════════
# Assemble notebook + write
# ═══════════════════════════════════════════════════════════════════════

NB = {
    'cells': CELLS,
    'metadata': {
        'colab': {'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU',
    },
    'nbformat': 4,
    'nbformat_minor': 0,
}


def main():
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(NB, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f'Wrote {len(CELLS)} cells to {NB_PATH}')


if __name__ == '__main__':
    main()
