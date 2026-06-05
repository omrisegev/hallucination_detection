"""
Build Spectral_Analysis_Consolidated_Results_LSML_v2.ipynb
Step 120: oriented L-SML v2 — binarize_classifiers(FEATURE_SIGNS) + lsml_fuse
CPU-only notebook; re-uses cached feature pkls from consolidated_results/.
"""
import json, uuid

def uid():
    return str(uuid.uuid4())[:8]

def md(source):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": source}

def code(source, background_save=False):
    meta = {"colab": {"background_save": background_save}} if background_save else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uid(),
        "metadata": meta,
        "outputs": [],
        "source": source,
    }

# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------

TITLE = """\
# Spectral Analysis — Consolidated Results (L-SML v2, Oriented)

**Step 120 notebook.** Re-runs all Consolidated Results fusion using the oriented
L-SML v2 pipeline derived from Jaffé-Fetaya-Nadler (2016):

1. Pre-orient each feature with consensus `FEATURE_SIGNS` (Step 110 cross-dataset analysis)
2. Binarize at empirical median via `binarize_classifiers`
3. Fuse binary classifiers with `lsml_fuse` (Algorithm 2)

**Difference from Step 107 notebook (`sml_unsupervised`):**

| Aspect | Step 107 | v2 (this notebook) |
|---|---|---|
| Sign orientation | Eigenvector majority vote (assumption iii) | **FEATURE_SIGNS — Step 110 consensus** |
| Binarization | Median split (no pre-orientation) | **Median split after orientation** |
| Labels used for | AUROC only | AUROC only |

**Re-uses cached feature pkls from `consolidated_results/` on Drive — no GPU needed.**

**Outputs (all in `consolidated_results/`):**
- `lsml_v2_math500_res.pkl`, `lsml_v2_gsm8k_res.pkl`, `lsml_v2_gpqa_res.pkl`
- `lsml_v2_rag_res.pkl`, `lsml_v2_qa_res.pkl`
- `lsml_v2_results_all.pkl` — combined
- `lsml_v2_summary.csv` — per-cell AUROC table

**Step 100 `results_all.pkl` and Step 107 `lsml_results_all.pkl` are untouched.**\
"""

SETUP_MD = "## Section 1 — Setup"

SETUP = """\
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

os.system('pip install -q "transformers>=4.40" accelerate datasets scipy scikit-learn')

import numpy as np
import pickle

from spectral_utils import (
    FEAT_NAMES, boot_auc, binarize_classifiers, lsml_fuse,
)

print('spectral_utils imported OK')
print(f'FEAT_NAMES ({len(FEAT_NAMES)}): {FEAT_NAMES}')\
"""

DRIVE_CONFIG = """\
from google.colab import drive
drive.mount('/content/drive')

BASE     = '/content/drive/MyDrive'
HALL_DIR = f'{BASE}/hallucination_detection'
OUT_DIR  = f'{HALL_DIR}/consolidated_results'
os.makedirs(OUT_DIR, exist_ok=True)

print(f'OUT_DIR = {OUT_DIR}  (exists: {os.path.exists(OUT_DIR)})')

# FEATURE_SIGNS: Step 110 consensus across 29 cells
# +1 = higher raw value -> more likely correct
# -1 = higher raw value -> hallucination / wrong answer
FEATURE_SIGNS = {
    'epr': -1, 'trace_length': 1, 'spectral_entropy': -1,
    'low_band_power': -1, 'high_band_power': -1, 'hl_ratio': -1,
    'dominant_freq': -1, 'spectral_centroid': -1,
    'stft_max_high_power': -1, 'stft_spectral_entropy': -1,
    'rpdi': -1, 'sw_var_peak': -1,
    'pe_mean': -1, 'hurst_exponent': 1,
    'cusum_max': -1, 'cusum_shift_idx': 1,
}

CACHED_FEAT_PKLS = {
    'math500': os.path.join(OUT_DIR, 'math500_res.pkl'),
    'gsm8k':   os.path.join(OUT_DIR, 'gsm8k_res.pkl'),
    'gpqa':    os.path.join(OUT_DIR, 'gpqa_res.pkl'),
    'rag':     os.path.join(OUT_DIR, 'rag_feats_all.pkl'),
    'qa':      os.path.join(OUT_DIR, 'qa_res.pkl'),
}

for name, path in CACHED_FEAT_PKLS.items():
    exists = os.path.exists(path)
    print(f'  {name}: {\"OK\" if exists else \"MISSING\"} -- {path}')\
"""

HELPERS = """\
def load_cached_feats(pkl_path):
    \"\"\"
    Load cached (feats_dict, labels) pairs from a pkl produced by the
    Consolidated Results notebook.  Handles two on-disk formats:
      1. {'feats': {key: (fd, lbl)}, 'results': ...}
      2. {key: (fd, lbl)} directly (e.g. rag_feats_all.pkl)
    Returns dict {key: (fd, lbl)} or None if the file is missing.
    \"\"\"
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'feats' in obj:
        return obj['feats']
    return obj


def run_lsml_v2(fd, lbl, key_str, verbose=True):
    \"\"\"
    Oriented L-SML v2 pipeline.

    Steps:
      1. binarize_classifiers(fd, FEATURE_SIGNS) -- orients then median-splits
      2. lsml_fuse(*binary.values())             -- Algorithm 2 (Jaffe et al. 2016)
      3. boot_auc(lbl, fused)                    -- AUROC, labels for eval only

    Returns a result dict or None if the cell is degenerate (one class).
    \"\"\"
    lbl = np.asarray(lbl, dtype=int)
    if len(set(lbl.tolist())) < 2:
        if verbose:
            print(f'  [{key_str}] only one class -- skip')
        return None
    n_pos = int(lbl.sum())
    n_neg = int(len(lbl) - n_pos)

    # Individual feature AUROCs oriented by FEATURE_SIGNS (no label leakage)
    ind_aucs = {}
    for fn in FEAT_NAMES:
        try:
            sign = FEATURE_SIGNS.get(fn, +1)
            a, *_ = boot_auc(lbl, np.array(fd[fn], dtype=float) * sign)
            ind_aucs[fn] = float(a) if not (a != a) else float('nan')  # nan check
        except Exception:
            ind_aucs[fn] = float('nan')

    try:
        binary = binarize_classifiers(fd, FEATURE_SIGNS)
        fused, meta = lsml_fuse(*binary.values())
    except Exception as e:
        if verbose:
            print(f'  [{key_str}] L-SML v2 error: {e}')
        return None

    # Safety: take max(AUC, 1-AUC) in case the fused sign is flipped
    p_auc, p_lo, p_hi = boot_auc(lbl,  fused)
    n_auc, n_lo, n_hi = boot_auc(lbl, -fused)
    if p_auc >= n_auc:
        v2_auc, v2_lo, v2_hi = p_auc, p_lo, p_hi
    else:
        v2_auc, v2_lo, v2_hi = n_auc, n_lo, n_hi

    if verbose:
        print(f'  [{key_str}] N={len(lbl)} (+{n_pos}/-{n_neg}) | '
              f'v2 L-SML={v2_auc:.3f} [{v2_lo:.3f},{v2_hi:.3f}] K={meta[\"K\"]}')

    return {
        'n': len(lbl), 'n_pos': n_pos, 'n_neg': n_neg,
        'ind_aucs': ind_aucs,
        'v2_auc': v2_auc, 'v2_lo': v2_lo, 'v2_hi': v2_hi,
        'K': meta['K'],
        'c': meta['c'].tolist(),
        'group_weights': [(idx.tolist(), w.tolist()) for idx, w in meta['group_weights']],
        'cross_weights': meta['cross_weights'].tolist(),
        'residual': float(meta['residual']),
        'method': meta['method'],
    }


print('Helpers defined.')\
"""

def domain_cell(domain_label, var_name, pkl_key, v2_pkl_name, bg_save=True):
    return """\
V2_PATH = os.path.join(OUT_DIR, '{v2_pkl_name}')
FORCE = False

{var_name}_FEATS = load_cached_feats(CACHED_FEAT_PKLS['{pkl_key}'])
if {var_name}_FEATS is None:
    print('ERROR: {pkl_key} pkl not found. Run Spectral_Analysis_Consolidated_Results.ipynb first.')
    {var_name}_V2 = {{}}
else:
    # Load partial progress if available (incremental save pattern)
    if not FORCE and os.path.exists(V2_PATH):
        with open(V2_PATH, 'rb') as f:
            {var_name}_V2 = pickle.load(f)
        remaining = [k for k in {var_name}_FEATS if k not in {var_name}_V2]
        print(f'Loaded {{len({var_name}_V2)}} cached {domain_label} keys; {{len(remaining)}} remaining')
    else:
        {var_name}_V2 = {{}}
        remaining = list({var_name}_FEATS.keys())
        print(f'Running L-SML v2 on {domain_label} ({{len(remaining)}} keys)...')

    for key in remaining:
        fd, lbl = {var_name}_FEATS[key]
        {var_name}_V2[key] = run_lsml_v2(fd, lbl, f'{domain_label}/{{key}}')
        with open(V2_PATH, 'wb') as f:
            pickle.dump({var_name}_V2, f)

    if remaining:
        print(f'Saved {{V2_PATH}} ({{len({var_name}_V2)}} total keys)')
    else:
        print(f'All {{len({var_name}_V2)}} {domain_label} keys already computed')\
""".format(
        v2_pkl_name=v2_pkl_name,
        var_name=var_name,
        pkl_key=pkl_key,
        domain_label=domain_label,
    )

SUMMARY_MD = "## Section 7 — Summary Table"

SUMMARY = """\
# (1) Save combined pkl
all_pkl = os.path.join(OUT_DIR, 'lsml_v2_results_all.pkl')
with open(all_pkl, 'wb') as f:
    pickle.dump({
        'math500': MATH500_V2, 'gsm8k': GSM8K_V2, 'gpqa': GPQA_V2,
        'rag': RAG_V2, 'qa': QA_V2,
    }, f)
print(f'Saved combined pkl -> {all_pkl}')

# (2) Build summary dataframe
import pandas as pd

domain_map = [
    ('MATH500', MATH500_V2), ('GSM8K', GSM8K_V2), ('GPQA', GPQA_V2),
    ('RAG', RAG_V2), ('QA', QA_V2),
]
all_v2 = {}
for domain, dres in domain_map:
    for k, v in dres.items():
        if v:
            all_v2[f'{domain}/{k}'] = v

rows = []
for full_key, res in sorted(all_v2.items(), key=lambda x: -x[1]['v2_auc']):
    rows.append({
        'domain_model': full_key,
        'v2_auc':       round(res['v2_auc'], 4),
        'v2_lo':        round(res['v2_lo'], 4),
        'v2_hi':        round(res['v2_hi'], 4),
        'v2_ci':        f"[{res['v2_lo']:.3f},{res['v2_hi']:.3f}]",
        'K':            res['K'],
        'n':            res['n'],
        'n_pos':        res['n_pos'],
        'n_neg':        res['n_neg'],
    })

summary_df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, 'lsml_v2_summary.csv')
summary_df.to_csv(csv_path, index=False)
print(f'Saved summary CSV -> {csv_path}')

# (3) Print results table
print()
print('=' * 90)
print(f'{"Domain/Model":<45} {"v2 L-SML AUROC":>16} {"95% CI":>20} {"K":>4} {"N":>6}')
print('=' * 90)
for _, row in summary_df.iterrows():
    print(f'{row["domain_model"]:<45} {100*row["v2_auc"]:>15.1f}% {row["v2_ci"]:>20} {row["K"]:>4} {row["n"]:>6}')
print('=' * 90)
print(f'Total cells with results: {len(summary_df)}')
valid = summary_df[summary_df['v2_auc'] >= 0.5]
print(f'Cells beating chance (AUROC >= 0.50): {len(valid)}/{len(summary_df)}')\
"""

DONE_MD = """\
## Done

**Saved to Drive `consolidated_results/`:**
- `lsml_v2_math500_res.pkl`
- `lsml_v2_gsm8k_res.pkl`
- `lsml_v2_gpqa_res.pkl`
- `lsml_v2_rag_res.pkl`
- `lsml_v2_qa_res.pkl`
- `lsml_v2_results_all.pkl` — combined; keys: `math500`, `gsm8k`, `gpqa`, `rag`, `qa`
- `lsml_v2_summary.csv` — per-cell AUROC table (sorted by AUROC desc)

**Per-key result dict fields:**
- `v2_auc`, `v2_lo`, `v2_hi` — AUROC with 95% bootstrap CI
- `K` — number of dependent groups detected by L-SML
- `c` — group assignment array
- `group_weights` — per-group SML weights `[(idx_list, weights_list)]`
- `cross_weights` — cross-group SML weights
- `residual` — Algorithm 1 residual at best K
- `ind_aucs` — individual feature AUROCs (oriented by FEATURE_SIGNS)
- `n`, `n_pos`, `n_neg` — sample counts\
"""

# ---------------------------------------------------------------------------
# Assemble cells
# ---------------------------------------------------------------------------

cells = [
    md(TITLE),
    md(SETUP_MD),
    code(SETUP),
    code(DRIVE_CONFIG),
    code(HELPERS),
    md("## Section 2 — MATH-500"),
    code(domain_cell("MATH-500", "MATH500", "math500", "lsml_v2_math500_res.pkl"), background_save=True),
    md("## Section 3 — GSM8K"),
    code(domain_cell("GSM8K", "GSM8K", "gsm8k", "lsml_v2_gsm8k_res.pkl"), background_save=True),
    md("## Section 4 — GPQA"),
    code(domain_cell("GPQA", "GPQA", "gpqa", "lsml_v2_gpqa_res.pkl"), background_save=True),
    md("## Section 5 — RAG (L-CiteEval)"),
    code(domain_cell("RAG", "RAG", "rag", "lsml_v2_rag_res.pkl"), background_save=True),
    md("## Section 6 — Factual QA"),
    code(domain_cell("QA", "QA", "qa", "lsml_v2_qa_res.pkl"), background_save=True),
    md(SUMMARY_MD),
    code(SUMMARY),
    md(DONE_MD),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {
            "provenance": [],
            "name": "Spectral_Analysis_Consolidated_Results_LSML_v2.ipynb",
            "toc_visible": True,
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "cells": cells,
}

out = "Spectral_Analysis_Consolidated_Results_LSML_v2.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written {out}  ({len(cells)} cells)")
