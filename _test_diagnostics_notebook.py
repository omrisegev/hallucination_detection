"""End-to-end test of LSML_Diagnostics.ipynb against fake cached pkls."""
import os, sys, io, json, pickle, tempfile, shutil

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spectral_utils import (
    FEAT_NAMES, load_cache, save_cache,
    decompose_auroc, threshold_sensitivity,
    plot_decomposition, plot_per_feature_heatmap,
    plot_sign_agreement, plot_threshold_sweep,
    plot_correlation_with_groups,
)

tmpdir = tempfile.mkdtemp(prefix='diag_test_')
OUT_DIR = os.path.join(tmpdir, 'consolidated_results')
DIAG_DIR = os.path.join(OUT_DIR, 'plots', 'diagnostics')
os.makedirs(DIAG_DIR, exist_ok=True)

rng = np.random.default_rng(11)


def make_features(n, n_keys, signal_p=0.55):
    out = {}
    for k in range(n_keys):
        Y = rng.choice([-1, 1], n)
        fd = {}
        for fn in FEAT_NAMES:
            if rng.random() < signal_p:
                fd[fn] = Y + rng.normal(0, 1.5, n)
            else:
                fd[fn] = rng.normal(0, 1.0, n)
        out[f'model{k}'] = (fd, (Y > 0).astype(int))
    return out


print('Building fake cached pkls...')
for name, n_keys, n in [
    ('math500', 2, 400),
    ('gsm8k',   1, 600),
    ('gpqa',    2, 200),
    ('qa',      2, 250),
]:
    feats = make_features(n, n_keys)
    pkl = os.path.join(OUT_DIR, f'{name}_res.pkl')
    fake_res = {k: {'nadler_auc': 0.7} for k in feats}
    save_cache({'results': fake_res, 'feats': feats}, pkl)
    print(f'  {pkl}  ({n_keys} keys, n={n})')

rag_feats = make_features(180, 3)
save_cache(rag_feats, os.path.join(OUT_DIR, 'rag_feats_all.pkl'))
print(f'  rag_feats_all.pkl ({len(rag_feats)} keys, n=180)')

# Execute the diagnostics notebook cells
with open('LSML_Diagnostics.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Build a namespace mimicking Colab
ns = {'__name__': '__main__'}
# Inject paths so the Drive bootstrap cell can be skipped
ns['OUT_DIR']  = OUT_DIR
ns['DIAG_DIR'] = DIAG_DIR
ns['DIAG_PKL'] = os.path.join(OUT_DIR, 'diagnostics_all.pkl')
ns['CACHED_FEAT_PKLS'] = {
    'math500': os.path.join(OUT_DIR, 'math500_res.pkl'),
    'gsm8k':   os.path.join(OUT_DIR, 'gsm8k_res.pkl'),
    'gpqa':    os.path.join(OUT_DIR, 'gpqa_res.pkl'),
    'rag':     os.path.join(OUT_DIR, 'rag_feats_all.pkl'),
    'qa':      os.path.join(OUT_DIR, 'qa_res.pkl'),
}
ns['K_RANGE']    = range(2, 7)
ns['THRESHOLDS'] = (0.25, 0.5, 0.75)
ns['BOOT_N']     = 200
ns['FORCE_RECOMPUTE'] = False

# Imports normally done by the setup cells
ns.update({
    'os': os, 'sys': sys, 'pickle': pickle, 'np': np,
    'matplotlib': matplotlib, 'plt': plt, 'json': json,
    'FEAT_NAMES': FEAT_NAMES,
    'load_cache': load_cache, 'save_cache': save_cache,
    'decompose_auroc': decompose_auroc, 'threshold_sensitivity': threshold_sensitivity,
    'plot_decomposition': plot_decomposition,
    'plot_per_feature_heatmap': plot_per_feature_heatmap,
    'plot_sign_agreement': plot_sign_agreement,
    'plot_threshold_sweep': plot_threshold_sweep,
    'plot_correlation_with_groups': plot_correlation_with_groups,
})

# Run cells 3..7 (skipping the title md cell 0, setup cell 1 = Colab-only, config cell 2)
for idx in [3, 4, 5, 6, 7]:
    cell = nb['cells'][idx]
    src = ''.join(cell['source'])
    print(f'\n=== Cell {idx} ===')
    print(src.splitlines()[0])
    exec(src, ns)

# Verify outputs
expected_pkls = [
    os.path.join(OUT_DIR, 'diagnostics_all.pkl'),
    os.path.join(OUT_DIR, 'diagnostics_summary.pkl'),
]
print('\n=== Verification ===')
for p in expected_pkls:
    if os.path.exists(p):
        with open(p, 'rb') as f: obj = pickle.load(f)
        n_entries = len(obj) if isinstance(obj, (dict, list)) else '?'
        print(f'  OK {os.path.basename(p)} ({n_entries} entries)')
    else:
        print(f'  MISSING {p}')

figs = sorted(os.listdir(DIAG_DIR))
print(f'  {len(figs)} figures in {DIAG_DIR}:')
for fn in figs[:10]:
    p = os.path.join(DIAG_DIR, fn)
    print(f'    {fn}  ({os.path.getsize(p)} bytes)')

shutil.rmtree(tmpdir)
print(f'\nALL DIAGNOSTICS-NOTEBOOK CELLS PASSED. (temp dir cleaned)')
