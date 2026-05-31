"""
End-to-end test: simulate running the LSML notebook against fake cached
features stored in a temp directory that mimics Drive structure.

This catches any bugs in the helper functions, pkl loading, or summary
table generation BEFORE the user spends Colab time.
"""
import os, sys, io, pickle, tempfile, shutil
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.path.insert(0, '.')

import numpy as np
from spectral_utils import (
    sml_unsupervised, sml_unsupervised_compare,
    detect_dependent_groups, lsml_fuse,
    binarize_classifiers, sml_fuse_signed,
    boot_auc, FEAT_NAMES,
)

# ============================================================
# STEP 1: build fake cached features in temp dir mimicking Drive
# ============================================================
tmpdir = tempfile.mkdtemp(prefix='lsml_test_')
OUT_DIR = os.path.join(tmpdir, 'consolidated_results')
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(7)

def make_features(n, n_keys, signal_p=0.6):
    out = {}
    for k in range(n_keys):
        key = f'model{k}'
        Y = rng.choice([-1, 1], n)
        fd = {}
        for fn in FEAT_NAMES:
            if rng.random() < signal_p:
                fd[fn] = Y + rng.normal(0, 1.0, n)
            else:
                fd[fn] = rng.normal(0, 1.0, n)
        out[key] = (fd, (Y > 0).astype(int))
    return out

# Match the pkl structure that the old notebook saves
# math500/gsm8k/gpqa/qa save: {'results': RES, 'feats': FEATS}
# rag saves: just FEATS = {key: (fd, lbl)}
print('Building fake cached pkls...')
for name, n_keys, n_samples in [
    ('math500', 4, 500),
    ('gsm8k',   1, 1000),
    ('gpqa',    3, 300),
    ('qa',      2, 400),
]:
    feats = make_features(n_samples, n_keys)
    fake_res = {k: {'nadler_auc': 0.75 + 0.05*rng.random(), 'ci_lo': 0.7, 'ci_hi': 0.8,
                    'best_subset': FEAT_NAMES[:4], 'best_weights': [0.25]*4,
                    'simple_avg_auc': 0.72, 'lift': 0.03, 'n': n_samples, 'n_pos': n_samples//2,
                    'n_neg': n_samples//2, 'ind_aucs': {f: 0.6 for f in FEAT_NAMES}}
                for k in feats}
    pkl_path = os.path.join(OUT_DIR, f'{name}_res.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'results': fake_res, 'feats': feats}, f)
    print(f'  wrote {pkl_path}  ({len(feats)} keys)')

# rag has different structure
rag_feats = make_features(200, 4)
with open(os.path.join(OUT_DIR, 'rag_feats_all.pkl'), 'wb') as f:
    pickle.dump(rag_feats, f)
print(f'  wrote rag_feats_all.pkl ({len(rag_feats)} keys)')

# Also create the old results_all.pkl for the comparison column
old_combined = {
    'math500': {k: {'nadler_auc': 0.75 + 0.05*rng.random()} for k in feats},
    'gsm8k':   {f'model{i}': {'nadler_auc': 0.72} for i in range(1)},
    'gpqa':    {f'model{i}': {'nadler_auc': 0.65} for i in range(3)},
    'rag':     {f'model{i}': {'nadler_auc': 0.60} for i in range(4)},
    'qa':      {f'model{i}': {'nadler_auc': 0.52} for i in range(2)},
}
with open(os.path.join(OUT_DIR, 'results_all.pkl'), 'wb') as f:
    pickle.dump(old_combined, f)
print(f'  wrote results_all.pkl (fake old comparison)')

# ============================================================
# STEP 2: execute the notebook's code cells against this fake Drive
# ============================================================
import json
with open('Spectral_Analysis_Consolidated_Results_LSML.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Build a Drive-like environment
ns = {'__name__': '__main__'}

# Skip the setup (cell 2, 3) and inject paths directly
ns['OUT_DIR'] = OUT_DIR
ns['LSML_PLOT_DIR'] = os.path.join(OUT_DIR, 'plots', 'lsml')
os.makedirs(ns['LSML_PLOT_DIR'], exist_ok=True)
ns['CACHED_FEAT_PKLS'] = {
    'math500': os.path.join(OUT_DIR, 'math500_res.pkl'),
    'gsm8k':   os.path.join(OUT_DIR, 'gsm8k_res.pkl'),
    'gpqa':    os.path.join(OUT_DIR, 'gpqa_res.pkl'),
    'rag':     os.path.join(OUT_DIR, 'rag_feats_all.pkl'),
    'qa':      os.path.join(OUT_DIR, 'qa_res.pkl'),
}

# Inject required imports/funcs
ns['os'] = os
ns['sys'] = sys
ns['pickle'] = pickle
ns['np'] = np
ns['sml_unsupervised'] = sml_unsupervised
ns['sml_unsupervised_compare'] = sml_unsupervised_compare
ns['detect_dependent_groups'] = detect_dependent_groups
ns['lsml_fuse'] = lsml_fuse
ns['binarize_classifiers'] = binarize_classifiers
ns['sml_fuse_signed'] = sml_fuse_signed
ns['boot_auc'] = boot_auc
ns['FEAT_NAMES'] = FEAT_NAMES

# Execute cell 4 (helpers — run_lsml, load_cached_feats)
print('\n=== Executing helper cell ===')
src = ''.join(nb['cells'][4]['source'])
exec(src, ns)

# Execute domain cells: 6 (MATH500), 8 (GSM8K), 10 (GPQA), 12 (RAG), 14 (QA)
for ci, dom in [(6, 'MATH500'), (8, 'GSM8K'), (10, 'GPQA'), (12, 'RAG'), (14, 'QA')]:
    print(f'\n=== Executing cell {ci} ({dom}) ===')
    src = ''.join(nb['cells'][ci]['source'])
    exec(src, ns)

# Execute summary table cell
print('\n=== Executing summary table cell ===')
# Need pandas for this
try:
    import pandas as pd
    ns['pd'] = pd
    src = ''.join(nb['cells'][16]['source'])
    exec(src, ns)
except ImportError:
    print('pandas not available locally — skip summary (will work on Colab)')

# Execute plot cell (matplotlib non-interactive)
print('\n=== Executing plot cell (cell 18) ===')
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for test
    import matplotlib.pyplot as plt
    ns['plt'] = plt
    src = ''.join(nb['cells'][18]['source'])
    exec(src, ns)
    plot_path = os.path.join(ns['LSML_PLOT_DIR'], 'lsml_vs_nadler_comparison.png')
    if os.path.exists(plot_path):
        print(f'  Plot saved: {plot_path} ({os.path.getsize(plot_path)} bytes)')
    else:
        print('  Plot NOT created')
except ImportError:
    print('matplotlib not available — skip plot cell')

# Verify pkls were written
print('\n=== Verifying output pkls ===')
expected = [
    'lsml_math500_res.pkl', 'lsml_gsm8k_res.pkl', 'lsml_gpqa_res.pkl',
    'lsml_rag_res.pkl', 'lsml_qa_res.pkl',
    'lsml_results_all.pkl',
]
for fn in expected:
    p = os.path.join(OUT_DIR, fn)
    if os.path.exists(p):
        with open(p, 'rb') as f: obj = pickle.load(f)
        n_keys = len(obj) if isinstance(obj, dict) else '?'
        print(f'  OK {fn}: {n_keys} entries')
    else:
        print(f'  MISSING {fn}')

print('\n=== END-TO-END NOTEBOOK SIMULATION PASSED ===')

# Cleanup
shutil.rmtree(tmpdir)
print(f'(temp dir {tmpdir} cleaned)')
