"""Declared POST-EVALUATION contrasts for Renyi-view fusion v2 (fast pass).

These pairs were not in the pre-registered contrast list (`run_renyi_view_fusion_v2.contrast_pairs`),
which compared fused arms with single views and references, and single views with H1 only.  After
seeing that the single tail-order view H0.1 carries the highest within-answer AUC, the following
pairs are added with the same 10,000-draw paired source-group bootstrap (95 %), on the saved step
scores, and are labelled post hoc: they inform the reading, they do not promote a candidate.
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os
os.environ.setdefault('RENYI_V2_ROSTER', 'fast')
from scripts import run_renyi_view_fusion_v2 as run

base = run.base
PAIRS = [('view__H0.1', 'ref__varentropy15'), ('view__H0.1', 'ref__varentropy15_iu'), ('view__H0.25', 'ref__varentropy15'),
         ('view__H0.1', 'view__H0.25'), ('view__H0.25', 'view__H0.5'), ('view__H0.1', 'direct_iu'),
         ('R6_sel__iu', 'view__H0.25'), ('view__H0.1', 'view__sel1')]


def main():
    source = Path(sys.argv[1]).resolve(); base.old.configure_source_root(source)
    out = run.OUT
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    metrics = json.loads((out / 'METRICS.json').read_text(encoding='utf8'))['metrics']
    with np.load(out / 'SCORES.npz') as z:
        scores = {k[len('steps__'):]: z[k] for k in z.files if k.startswith('steps__')}
    m2, per = base.evaluate_arrays(records, joined, scores)
    for k in metrics:
        for key in ('pb_all8', 'prm_within', 'prm_pooled'):
            np.testing.assert_allclose(m2[k][key], metrics[k][key], atol=1e-12, rtol=0)
    contrasts = base.paired_bootstrap(records, joined, per, pairs=PAIRS, primary_pairs=set(), draws=10000)
    pb = np.array([r['cell'].startswith('pb_') for r in records]); target = joined['target']
    for a, b in PAIRS:
        c = contrasts[a + '_minus_' + b]; c['pb_delta'] = metrics[a]['pb_all8'] - metrics[b]['pb_all8']
        oldhit = pb & per[b]['decision_valid'] & (per[b]['prediction'] == target)
        newhit = pb & per[a]['decision_valid'] & (per[a]['prediction'] == target)
        c.update(gained=int((newhit & ~oldhit).sum()), lost=int((oldhit & ~newhit).sum()))
        print(f"{a} - {b}: PB {100 * c['pb_delta']:+.2f} pp [{100 * c['pb_ci'][0]:+.2f}, {100 * c['pb_ci'][1]:+.2f}]; "
              f"within {c['prm_within_delta_common']:+.4f} [{c['prm_within_ci'][0]:+.4f}, {c['prm_within_ci'][1]:+.4f}]; "
              f"pooled {metrics[a]['prm_pooled'] - metrics[b]['prm_pooled']:+.4f}; PRMScore {metrics[a]['prmscore_q08'] - metrics[b]['prmscore_q08']:+.4f}", flush=True)
    base.atomic_json(out / 'POSTHOC_CONTRASTS.json', run.clean(dict(
        scope='POST_EVALUATION; pairs chosen after seeing the fast-pass table; same bootstrap, 95 %; not pre-registered',
        pairs=[list(p) for p in PAIRS], contrasts=contrasts)))


if __name__ == '__main__':
    main()
