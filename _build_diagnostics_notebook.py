"""
Generator for LSML_Diagnostics.ipynb.

Per CLAUDE.md: never string-edit notebook JSON. This script builds the
notebook programmatically from cell sources, then json.dumps it.

The notebook reads cached features produced by Spectral_Analysis_Consolidated_Results_LSML
and runs the diagnostic decomposition (5-row table + per-feature heatmap +
sign agreement + threshold sweep + correlation/group structure) on every
cached (domain, model) cell, saving plots to consolidated_results/plots/diagnostics/.
"""
import json
import uuid

CELLS = []


def add_md(text):
    CELLS.append({
        'cell_type': 'markdown',
        'id': uuid.uuid4().hex[:12],
        'metadata': {},
        'source': text.splitlines(keepends=True),
    })


def add_code(text):
    CELLS.append({
        'cell_type': 'code',
        'id': uuid.uuid4().hex[:12],
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': text.splitlines(keepends=True),
    })


# ── 1. Title ────────────────────────────────────────────────────────────────
add_md("""# L-SML Diagnostics

Decompose the Step 107 L-SML AUROC into its constituent transformations to identify which step costs the most accuracy vs the old supervised continuous Nadler (Step 100).

**Five pipeline stages** (each swaps one variable from the row above):

| # | Inputs | Sign source | Fusion |
|---|--------|-------------|--------|
| 1 | continuous | supervised (labels) | simple average |
| 2 | continuous | supervised | SML weights |
| 3 | **binary** | supervised | SML weights |
| 4 | binary | **L-SML (unsupervised)** | SML weights (1 group) |
| 5 | binary | L-SML | **L-SML (K groups)** ← official Step 107 |

Plus per-feature AUROC at each stage, supervised-vs-L-SML sign agreement, threshold sensitivity (0.25/0.5/0.75 quantiles), and correlation structure with group overlay.

Runs on the cached features that `Spectral_Analysis_Consolidated_Results_LSML.ipynb` already produced — no GPU, no inference, CPU-only.
""")

# ── 2. Setup ────────────────────────────────────────────────────────────────
add_code("""# Cell 1 — Drive mount + clone + install + imports
import os, sys, shutil
os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    from google.colab import drive
    drive.mount('/content/drive')
    REPO_DIR = '/content/hallucination_detection'
    if os.path.exists(REPO_DIR) and not os.path.exists(os.path.join(REPO_DIR, 'spectral_utils')):
        shutil.rmtree(REPO_DIR)
    BRANCH = 'feature/nadler-paper-alignment'
    if not os.path.exists(REPO_DIR):
        os.system(f'git clone -b {BRANCH} https://github.com/omrisegev/hallucination_detection.git {REPO_DIR}')
    else:
        os.system(f'git -C {REPO_DIR} fetch -q && git -C {REPO_DIR} checkout -q {BRANCH} && git -C {REPO_DIR} pull -q')
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    os.system('pip install -q scikit-learn scipy')
except ImportError:
    # Local run: assume repo is the cwd
    REPO_DIR = os.getcwd()
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

import pickle, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from spectral_utils import (
    FEAT_NAMES, load_cache, save_cache,
    decompose_auroc, threshold_sensitivity, derive_consensus_signs,
    plot_decomposition, plot_per_feature_heatmap,
    plot_sign_agreement, plot_threshold_sweep,
    plot_correlation_with_groups,
)
print('spectral_utils.diagnostics loaded OK')
print(f'  FEAT_NAMES ({len(FEAT_NAMES)}): {FEAT_NAMES}')
""")

# ── 3. Config ───────────────────────────────────────────────────────────────
add_code("""# Cell 2 — Config
BASE_DRIVE   = '/content/drive/MyDrive'
DRIVE_BASE   = f'{BASE_DRIVE}/hallucination_detection'
OUT_DIR      = os.path.join(DRIVE_BASE, 'consolidated_results')
DIAG_DIR     = os.path.join(OUT_DIR, 'plots', 'diagnostics')
DIAG_PKL     = os.path.join(OUT_DIR, 'diagnostics_all.pkl')
os.makedirs(DIAG_DIR, exist_ok=True)

# Cached feature pkls produced by Spectral_Analysis_Consolidated_Results_LSML
CACHED_FEAT_PKLS = {
    'math500': os.path.join(OUT_DIR, 'math500_res.pkl'),
    'gsm8k':   os.path.join(OUT_DIR, 'gsm8k_res.pkl'),
    'gpqa':    os.path.join(OUT_DIR, 'gpqa_res.pkl'),
    'rag':     os.path.join(OUT_DIR, 'rag_feats_all.pkl'),
    'qa':      os.path.join(OUT_DIR, 'qa_res.pkl'),
}

K_RANGE      = range(2, 7)
THRESHOLDS   = (0.25, 0.5, 0.75)
BOOT_N       = 500
FORCE_RECOMPUTE = False

print(f'OUT_DIR   : {OUT_DIR}')
print(f'DIAG_DIR  : {DIAG_DIR}')
print(f'DIAG_PKL  : {DIAG_PKL}')
""")

# ── 4. Helper: load any cached (feats_dict, labels) cell ───────────────────
add_code("""# Cell 3 — Helper: load cached features into uniform (feats_dict, labels, key) list
def iter_domain_cells(domain, path):
    \"\"\"Yield (key_str, feats_dict, labels) for each (model, dataset) cell in the pkl.

    Handles both the math500/gsm8k/gpqa/qa schema ({'results': ..., 'feats': {k: (fd, lbl)}})
    and the rag schema ({k: (fd, lbl)} directly).
    \"\"\"
    if not os.path.exists(path):
        print(f'  [{domain}] MISSING: {path}')
        return
    blob = load_cache(path)
    feats_block = blob.get('feats', blob) if isinstance(blob, dict) else blob
    if not isinstance(feats_block, dict):
        print(f'  [{domain}] unrecognised schema'); return
    for k, val in feats_block.items():
        if not isinstance(val, tuple) or len(val) != 2:
            continue
        fd, lbl = val
        if not isinstance(fd, dict) or not isinstance(lbl, (list, tuple, np.ndarray)):
            continue
        lbl = np.asarray(lbl).astype(int)
        if lbl.size < 30 or len(set(lbl.tolist())) < 2:
            continue
        # Ensure every FEAT_NAMES entry is present
        if not all(f in fd for f in FEAT_NAMES):
            missing = [f for f in FEAT_NAMES if f not in fd]
            print(f'  [{domain}/{k}] missing features: {missing}'); continue
        yield (f'{domain}/{k}', fd, lbl)

# Inventory
print('=== Inventory ===')
all_cells = []
for dom, path in CACHED_FEAT_PKLS.items():
    for key, fd, lbl in iter_domain_cells(dom, path):
        all_cells.append((key, fd, lbl))
        print(f'  {key:<60s}  n={len(lbl):>5d}  acc={lbl.mean():.1%}')
print(f'Total cells: {len(all_cells)}')
""")

# ── 5. Run decomposition on every cell ──────────────────────────────────────
add_code("""# Cell 4 — Decompose + threshold sweep for every cell (saves per-cell pkl + figure)
if not FORCE_RECOMPUTE and os.path.exists(DIAG_PKL):
    DIAG_RES = load_cache(DIAG_PKL)
    print(f'Loaded {len(DIAG_RES)} previous diagnostics from {DIAG_PKL}')
else:
    DIAG_RES = {}

for key, fd, lbl in all_cells:
    if not FORCE_RECOMPUTE and key in DIAG_RES:
        continue
    print(f'\\n→ {key}  (n={len(lbl)}, acc={lbl.mean():.1%})')
    try:
        decomp = decompose_auroc(fd, FEAT_NAMES, lbl, K_range=K_RANGE, boot_n=BOOT_N)
        sweep  = threshold_sensitivity(fd, FEAT_NAMES, lbl, thresholds=THRESHOLDS,
                                       K_range=K_RANGE, boot_n=BOOT_N)
    except Exception as e:
        print(f'  ERROR: {e}'); continue
    for r in decomp['rows']:
        print(f"    {r['name']:<50s} AUROC={100*r['auc']:.1f}  CI=[{100*r['lo']:.1f},{100*r['hi']:.1f}]")
    print(f"    K_residual={decomp['groups']['K']}  K_eigengap={decomp['groups']['K_eigengap']}  "
          f"sign-agree={sum(decomp['signs']['agree'].values())}/{len(FEAT_NAMES)}")
    DIAG_RES[key] = {'decomp': decomp, 'sweep': sweep, 'n': int(len(lbl)), 'acc': float(lbl.mean())}
    save_cache(DIAG_RES, DIAG_PKL)  # checkpoint every cell

print(f'\\nSaved {len(DIAG_RES)} cells to {DIAG_PKL}')
""")

# ── 6. Plot grid per cell ───────────────────────────────────────────────────
add_code("""# Cell 5 — Per-cell 5-panel diagnostic figure
matplotlib.rcParams['figure.dpi'] = 100

for key, data in DIAG_RES.items():
    decomp = data['decomp']
    sweep  = data['sweep']
    safe_key = key.replace('/', '__')
    fig_path = os.path.join(DIAG_DIR, f'{safe_key}.png')
    if not FORCE_RECOMPUTE and os.path.exists(fig_path):
        continue

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs  = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_decomposition(decomp['rows'], ax1, title='5-stage AUROC decomposition')

    ax2 = fig.add_subplot(gs[0, 1])
    plot_per_feature_heatmap(decomp['per_feat'], FEAT_NAMES, ax2,
                             title='Per-feature AUROC per stage')

    ax3 = fig.add_subplot(gs[0, 2])
    plot_sign_agreement(decomp['signs'], FEAT_NAMES, ax3,
                        title='Sign agreement (sup. vs L-SML)')

    ax4 = fig.add_subplot(gs[1, 0])
    plot_threshold_sweep(sweep, ax4,
                         title='Threshold sensitivity (quantile)')

    ax5 = fig.add_subplot(gs[1, 1:])
    plot_correlation_with_groups(
        {f: decomp['binary']['unsupervised'][f] for f in FEAT_NAMES},
        FEAT_NAMES, decomp['groups']['assignment'], ax5,
        title='Binary-feature correlation, ordered by L-SML group',
    )

    fig.suptitle(f'{key}   (n={data["n"]}, acc={data["acc"]:.1%})', fontsize=13, fontweight='bold')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {fig_path}')

print(f'\\n{len(os.listdir(DIAG_DIR))} figures in {DIAG_DIR}')
""")

# ── 7. Summary table across all cells ──────────────────────────────────────
add_code("""# Cell 6 — Summary table: AUROC at each stage for every cell
# Save pkl first so summary survives kernel disconnect, then build dataframe.
SUMMARY = []
for key, data in DIAG_RES.items():
    row = {'cell': key, 'n': data['n'], 'acc': data['acc']}
    for i, r in enumerate(data['decomp']['rows'], 1):
        row[f'AUC_{i}'] = r['auc']
    row['delta_3_minus_2'] = row['AUC_3'] - row['AUC_2']   # binarization cost
    row['delta_4_minus_3'] = row['AUC_4'] - row['AUC_3']   # unsupervised sign cost
    row['delta_5_minus_4'] = row['AUC_5'] - row['AUC_4']   # group detection cost
    row['K']               = data['decomp']['groups']['K']
    row['signs_agree']     = sum(data['decomp']['signs']['agree'].values())
    SUMMARY.append(row)

SUMMARY_PATH = os.path.join(OUT_DIR, 'diagnostics_summary.pkl')
save_cache(SUMMARY, SUMMARY_PATH)
print(f'Saved {len(SUMMARY)} rows to {SUMMARY_PATH}')

# Render as dataframe if pandas is available
try:
    import pandas as pd
    df = pd.DataFrame(SUMMARY)
    cols = ['cell', 'n', 'acc', 'AUC_1', 'AUC_2', 'AUC_3', 'AUC_4', 'AUC_5',
            'delta_3_minus_2', 'delta_4_minus_3', 'delta_5_minus_4',
            'K', 'signs_agree']
    df = df[cols].sort_values('cell')
    pct = lambda x: f'{100*x:.1f}'
    fmt = df.copy()
    for c in ['acc', 'AUC_1', 'AUC_2', 'AUC_3', 'AUC_4', 'AUC_5',
              'delta_3_minus_2', 'delta_4_minus_3', 'delta_5_minus_4']:
        fmt[c] = df[c].map(pct)
    print(fmt.to_string(index=False))
    df.to_csv(os.path.join(OUT_DIR, 'diagnostics_summary.csv'), index=False)
    print(f'\\nCSV: {OUT_DIR}/diagnostics_summary.csv')
except ImportError:
    print('pandas unavailable; pkl saved.')
""")

# ── 8. Aggregate landscape plot ─────────────────────────────────────────────
add_code("""# Cell 7 — Landscape plot: stage AUROCs across all cells
fig, ax = plt.subplots(figsize=(14, 6))
keys = [s['cell'] for s in SUMMARY]
xpos = np.arange(len(keys))
width = 0.16
colors = ['#2c7fb8', '#41b6c4', '#a1dab4', '#fdae61', '#d7191c']
for i in range(5):
    ax.bar(xpos + (i - 2)*width, [100*s[f'AUC_{i+1}'] for s in SUMMARY],
           width, label=f'Stage {i+1}', color=colors[i])
ax.set_xticks(xpos)
ax.set_xticklabels(keys, rotation=70, ha='right', fontsize=8)
ax.set_ylabel('AUROC (%)')
ax.axhline(50, color='#888', lw=0.8, ls='--')
ax.set_ylim(40, 100)
ax.legend(loc='lower right', fontsize=9)
ax.set_title('AUROC across pipeline stages, all cached cells', fontsize=12)
fig.tight_layout()
landscape_path = os.path.join(DIAG_DIR, '_landscape.png')
fig.savefig(landscape_path, bbox_inches='tight')
plt.show()
print(f'Saved {landscape_path}')
""")

# ── 9. Consensus sign derivation + re-run ───────────────────────────────────
add_md("""## Consensus orientation

The Step 107 / 109 diagnostics show that L-SML's pure Paper 2 (iii) sign rule collapses (AUROC → 1−AUROC) on cells where most features point the same direction. Paper 2 assumption (iii) is *systematically* violated by our entropy-heavy feature set.

Solution: derive each feature's direction **once, offline** by majority vote across the 29 cells we just ran, then pre-orient every feature by that fixed direction before binarization. Still unsupervised at inference time (no per-cell labels used), but uses prior empirical knowledge of feature semantics.
""")

add_code("""# Cell 8 — Derive consensus signs from the 29-cell diagnostics, print + save
decomps = [data['decomp'] for data in DIAG_RES.values()]
consensus = derive_consensus_signs(DIAG_RES, agreement_threshold=0.6,
                                   use_auroc_weight=True)

print('=== Consensus per-feature direction ===')
print(f"{'feature':<22} {'sign':>5} {'confidence':>11} {'votes (+/−)':>14}")
print('-' * 56)
for f in FEAT_NAMES:
    s    = consensus['signs'][f]
    c    = consensus['confidence'][f]
    p, m = consensus['votes'][f]['plus'], consensus['votes'][f]['minus']
    flag = '  ⚠ low' if f in consensus['low_confidence'] else ''
    print(f'{f:<22} {s:>+5d} {100*c:>10.1f}%  {p:>5.1f} / {m:>5.1f}{flag}')

print(f"\\nlow-confidence features: {consensus['low_confidence']}")
CONSENSUS_PATH = os.path.join(OUT_DIR, 'consensus_signs.pkl')
save_cache(consensus, CONSENSUS_PATH)
print(f'\\nSaved {CONSENSUS_PATH}')

# Constant we will reuse downstream:
FEATURE_SIGNS = consensus['signs']
""")

add_code("""# Cell 9 — Re-run decomposition for every cell with consensus orientation
CONSENSUS_DIAG_PKL = os.path.join(OUT_DIR, 'diagnostics_consensus_all.pkl')
if not FORCE_RECOMPUTE and os.path.exists(CONSENSUS_DIAG_PKL):
    DIAG_RES_CONS = load_cache(CONSENSUS_DIAG_PKL)
    print(f'Loaded {len(DIAG_RES_CONS)} previous consensus diagnostics')
else:
    DIAG_RES_CONS = {}

for key, fd, lbl in all_cells:
    if not FORCE_RECOMPUTE and key in DIAG_RES_CONS:
        continue
    print(f'\\n→ {key}  (consensus orientation)')
    try:
        decomp = decompose_auroc(fd, FEAT_NAMES, lbl,
                                 K_range=K_RANGE, boot_n=BOOT_N,
                                 feature_signs=FEATURE_SIGNS)
    except Exception as e:
        print(f'  ERROR: {e}'); continue
    for r in decomp['rows']:
        print(f"    {r['name']:<50s} AUROC={100*r['auc']:.1f}  CI=[{100*r['lo']:.1f},{100*r['hi']:.1f}]")
    DIAG_RES_CONS[key] = {
        'decomp': decomp, 'n': int(len(lbl)), 'acc': float(lbl.mean()),
    }
    save_cache(DIAG_RES_CONS, CONSENSUS_DIAG_PKL)
print(f'\\nSaved {len(DIAG_RES_CONS)} consensus cells to {CONSENSUS_DIAG_PKL}')
""")

add_code("""# Cell 10 — Side-by-side: Paper-2 (iii) vs Consensus, AUROC delta per cell
DELTA_ROWS = []
for key in DIAG_RES:
    if key not in DIAG_RES_CONS:
        continue
    p2  = DIAG_RES[key]['decomp']
    con = DIAG_RES_CONS[key]['decomp']
    DELTA_ROWS.append({
        'cell':           key,
        'n':              DIAG_RES[key]['n'],
        'acc':            DIAG_RES[key]['acc'],
        'AUC_p2_st4':     p2['rows'][3]['auc'],
        'AUC_con_st4':    con['rows'][3]['auc'],
        'delta_st4':      con['rows'][3]['auc'] - p2['rows'][3]['auc'],
        'AUC_p2_st5':     p2['rows'][4]['auc'],
        'AUC_con_st5':    con['rows'][4]['auc'],
        'delta_st5':      con['rows'][4]['auc'] - p2['rows'][4]['auc'],
        'AUC_sup_st3':    p2['rows'][2]['auc'],
        'sign_agree_p2':  sum(p2['signs']['agree'].values()),
    })

# Save pkl first (background_save safety)
DELTA_PATH = os.path.join(OUT_DIR, 'consensus_vs_paper2_summary.pkl')
save_cache(DELTA_ROWS, DELTA_PATH)
print(f'Saved {len(DELTA_ROWS)} rows to {DELTA_PATH}')

try:
    import pandas as pd
    df = pd.DataFrame(DELTA_ROWS).sort_values('delta_st5', ascending=False)
    pct = lambda x: f'{100*x:+.1f}' if isinstance(x, float) else str(x)
    fmt = df.copy()
    for c in ['acc', 'AUC_p2_st4', 'AUC_con_st4', 'delta_st4',
              'AUC_p2_st5', 'AUC_con_st5', 'delta_st5', 'AUC_sup_st3']:
        fmt[c] = df[c].map(lambda x: f'{100*x:.1f}')
    print(fmt.to_string(index=False))
    csv_path = os.path.join(OUT_DIR, 'consensus_vs_paper2_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f'\\nCSV: {csv_path}')
except ImportError:
    print('pandas unavailable; pkl saved.')

# Landscape comparison plot: Stage 5 AUROC, Paper-2 vs Consensus, per cell
fig, ax = plt.subplots(figsize=(14, 6))
keys = [r['cell'] for r in DELTA_ROWS]
xpos = np.arange(len(keys))
width = 0.35
ax.bar(xpos - width/2, [100*r['AUC_p2_st5']  for r in DELTA_ROWS], width,
       label='Stage 5: Paper-2 (iii)',   color='#d7191c')
ax.bar(xpos + width/2, [100*r['AUC_con_st5'] for r in DELTA_ROWS], width,
       label='Stage 5: Consensus orient',color='#1a9641')
ax.axhline(50, color='#888', lw=0.8, ls='--')
ax.set_xticks(xpos)
ax.set_xticklabels(keys, rotation=70, ha='right', fontsize=8)
ax.set_ylabel('AUROC (%)')
ax.set_ylim(0, 100)
ax.legend(loc='lower right')
ax.set_title('L-SML Stage 5: Paper-2 sign rule vs offline-consensus orientation',
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(DIAG_DIR, '_consensus_vs_paper2.png'), bbox_inches='tight')
plt.show()
""")

# ── Final assembly ──────────────────────────────────────────────────────────
nb = {
    'cells': CELLS,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language':     'python',
            'name':         'python3',
        },
        'language_info': {'name': 'python'},
        'colab': {'provenance': []},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

out_path = 'LSML_Diagnostics.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print(f'Wrote {out_path} with {len(CELLS)} cells')
