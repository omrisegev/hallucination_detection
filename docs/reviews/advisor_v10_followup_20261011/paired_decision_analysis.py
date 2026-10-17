"""Paired development diagnostics on authenticated PRM rewards and source groups.

Fixed exported Q80 rules are compared with PRM's native reward>=0.5 validity rule.
Oracle cuts/counts use evaluation labels; they are not held-out method results.
Bootstrap resamples whole canonical groups; it does not correct method selection.
"""
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from decision_analysis import OUT, INPUT, OFFICIAL, METHODS, metrics, f1, write_csv

EXTRA = OUT / 'sources'
PRM_SHA = '95be425c66d4dadc6a8e03567ad66ef2933aee27eb7967bf7200139c56518232'
GROUP_SHA = '63c767ffd373b37fdf08ff9ca7711260daa8ed527266ad5e619b7275015a5d94'
CONTRASTS = [('LSML_CONT', 'ALL_EQ'), ('IU_PCR', 'ALL_EQ'),
             ('LSML_CONT', 'DS_FILTER_EQ'), ('LSML_CONT', 'PRM_NATIVE'),
             ('IU_PCR', 'PRM_NATIVE')]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def bootstrap_score(confusions, weights):
    tp, fp, fn, tn = (weights @ confusions[:, j] for j in range(4))
    return (f1(tp, fp, fn) + f1(tn, fp, fn)) / 2

def main():
    pp = EXTRA / 'prmbench_prm.pkl'; gp = EXTRA / 'CANONICAL_GROUPS.json'
    assert sha(pp) == PRM_SHA and sha(gp) == GROUP_SHA
    # This is the dedicated PRM runner cache, not a rich-save replication-grid cell.
    # The repository's inspect_cell.py requires candidates/token fields and is inapplicable.
    records = pickle.loads(pp.read_bytes())
    records = list(records.values()) if isinstance(records, dict) else records
    prm = {r['idx']: r for r in records}
    rows = json.loads(gp.read_text())['rows']
    groups = {r['row_id']: r['canonical_group_id'] for r in rows}
    assert len(groups) == len(rows)
    z = np.load(INPUT / 'bank_N16.npz', allow_pickle=False)
    ids = z['idx'].astype(str); off = z['offsets']; n = np.diff(off)
    meta = {r['idx']: r for r in json.loads((INPUT / 'prmbench_meta.json').read_text())['rows']}
    assert len(prm) == len(records) == len(ids) == len(set(ids)) and set(prm) == set(ids)
    assert set(ids) <= set(groups) and np.all(n > 0)
    classes = np.array([meta[u]['classification'] for u in ids])
    answer = np.repeat(np.arange(len(ids)), n)
    pool = classes[answer] != 'correct'
    y = np.zeros(int(off[-1]), bool); reward = np.empty(len(y)); native = np.empty(len(y), bool)
    for i, u in enumerate(ids):
        r, m = prm[u], meta[u]; a, b = off[i:i+2]
        assert r['classification'] == m['classification'] and r['category'] == m['category']
        assert r['n_steps'] == m['n_steps'] == n[i]
        assert r['error_steps'] == m['error_steps'] and len(r['rewards']) == n[i]
        valid = np.asarray(r['rewards'], float)
        assert np.isfinite(valid).all() and np.all((valid >= 0) & (valid <= 1))
        assert np.array_equal(valid >= .5, np.asarray(r['labels'], bool))
        reward[a:b] = valid; native[a:b] = valid < .5
        y[a:b] = np.isin(np.arange(1, n[i]+1), m['error_steps'])
    unique_groups, ag = np.unique([groups[u] for u in ids], return_inverse=True)
    assert len(unique_groups) == 707
    sg = ag[answer]; ng = len(unique_groups)
    flags = {m: z['S__'+m] >= np.quantile(z['S__'+m], .8) for m in METHODS}
    flags['PRM_NATIVE'] = native
    truth_count = np.add.reduceat(y.astype(int), off[:-1])
    summary = {'n_answers': len(ids), 'n_steps': len(y), 'n_groups': ng,
               'bootstrap': {'n_resamples': 5000, 'seed': 20261011,
                             'unit': 'canonical source group; whole-group multiplicity, original answer/step weights',
                             'interval': 'pointwise percentile 95%, no multiplicity or selection adjustment'},
               'inputs_sha256': {str(p): sha(p) for p in [pp, gp, INPUT/'bank_N16.npz', INPUT/'prmbench_meta.json']},
               'script_sha256': sha(Path(__file__)),
               'arithmetic_module_sha256': sha(OUT/'decision_analysis.py'),
               'prm_schema': {'exact_half_rewards': int(np.sum(reward == .5)),
                              'native_error_flag': 'validity reward < 0.5; ties at 0.5 are VALID',
                              'alignment': 'exact idx, classification, n_steps, error_steps and cached labels'},
               'rules': {}, 'oracle_diagnostics': {}, 'contrasts': []}
    rng = np.random.default_rng(20261011)
    weights = rng.multinomial(ng, np.full(ng, 1/ng), size=5000).astype(float)
    for subset in ['ALL_NONCONTROL', 'redundency', 'circular']:
        selected = pool if subset == 'ALL_NONCONTROL' else classes[answer] == subset
        point = {}; boot = {}
        for m, flag in flags.items():
            masks = [y & flag, ~y & flag, y & ~flag, ~y & ~flag]
            confusion = np.column_stack([np.bincount(sg[selected & x], minlength=ng) for x in masks])
            point[m] = metrics(y[selected], flag[selected])['prm_score']
            boot[m] = bootstrap_score(confusion, weights)
            assert np.all(weights @ confusion.sum(axis=1) > 0)
            # Independent first-draw check: explicitly repeat selected step rows.
            repeats = weights[0, sg[selected]].astype(int)
            explicit = metrics(np.repeat(y[selected], repeats), np.repeat(flag[selected], repeats))
            assert abs(explicit['prm_score'] - boot[m][0]) < 1e-12
        for left, right in CONTRASTS:
            lo, hi = np.quantile(boot[left]-boot[right], [.025, .975])
            summary['contrasts'].append(dict(subset=subset, left=left, right=right,
                metric='PRMScore', difference=point[left]-point[right], ci_low=float(lo), ci_high=float(hi)))
    for m, flag in flags.items():
        counts = np.add.reduceat(flag.astype(int), off[:-1])
        summary['rules'][m] = dict(**metrics(y[pool], flag[pool]), flag_rate_all=float(flag.mean()),
            synthetic_control_false_flag_rate=float((counts[classes=='correct']>0).mean()),
            all_error_free_false_flag_rate=float((counts[truth_count==0]>0).mean()))
        # Perfect-ranking oracle preserves this method's actual per-answer Q80/native counts.
        perfect = np.zeros(len(y), bool)
        for a, b, count in zip(off[:-1], off[1:], counts):
            order = np.argsort(-y[a:b].astype(int), kind='stable')
            perfect[a+order[:count]] = True
        summary['oracle_diagnostics'][m] = {'PERFECT_RANKING_AT_FIXED_COUNTS': metrics(y[pool], perfect[pool]),
            'warning': 'Uses true step labels for placement; counts are frozen. Diagnostic, not attainable method performance.'}
        actual_risk = 1-reward if m == 'PRM_NATIVE' else z['S__'+m]
        true_quota = np.zeros(len(y), bool)
        for a, b, count in zip(off[:-1], off[1:], truth_count):
            order = np.argsort(-actual_risk[a:b], kind='stable')
            true_quota[a+order[:count]] = True
        summary['oracle_diagnostics'][m]['TRUE_COUNT_ORACLE'] = metrics(y[pool], true_quota[pool])
    # Fair threshold oracle for PRM: all tie-preserving cuts, same noncontrol scoring population.
    risk = 1-reward; order = np.argsort(-risk, kind='stable')
    selected = pool[order]; tp = np.cumsum(y[order] & selected); fp = np.cumsum(~y[order] & selected)
    stops = np.r_[np.flatnonzero(risk[order][:-1] != risk[order][1:]), len(y)-1]
    p = int(y[pool].sum()); v = int((~y[pool]).sum())
    values = (f1(tp[stops], fp[stops], p-tp[stops]) + f1(v-fp[stops], fp[stops], p-tp[stops]))/2
    none = metrics(y[pool], np.zeros(pool.sum(), bool))['prm_score']
    oi = int(np.argmax(values)); cut = float(risk[order][stops[oi]]) if values[oi] >= none else float('inf')
    oflag = risk >= cut; om = metrics(y[pool], oflag[pool])
    assert abs(om['prm_score'] - max(none, float(values.max()))) < 1e-12
    summary['oracle_diagnostics']['PRM_NATIVE']['GLOBAL_THRESHOLD_ORACLE'] = dict(**om,
        risk_cut=None if not np.isfinite(cut) else cut, flag_rate_all=float(oflag.mean()),
        gain_over_native=om['prm_score']-summary['rules']['PRM_NATIVE']['prm_score'])
    references = list(csv.DictReader(OFFICIAL.open()))
    ref = next(r for r in references if r['rule'] == 'native 0.5')
    assert abs(summary['rules']['PRM_NATIVE']['prm_score']-float(ref['overall'])) < 1e-12
    curves = []
    for rate in np.linspace(0,.5,501):
        threshold = float('inf') if rate == 0 else float(np.quantile(risk, 1-rate))
        flag = risk >= threshold
        curves.append(dict(method='PRM_NATIVE', nominal_rate=float(rate), actual_flag_rate=float(flag.mean()),
                           threshold=threshold, **metrics(y[pool], flag[pool])))
    summary['limits'] = ['All data are previously inspected development data, not untouched confirmation.',
        'Pointwise paired intervals account for source dependence only; not selection or multiple testing.',
        'Native PRM and Q80 have different flag budgets and supervised versus label-free training.',
        'Oracle cuts, perfect placement and true counts use labels; no new rule is promoted.',
        'Dedicated PRM cache lacks a redundancy head; all methods use the same declared validity fallback.',
        'No raw token lengths/repetition/features, corrected entropy gate, new model inference, or method refitting.']
    summary['checks'] = {'first_draw_explicit_step_expansion': 'PASS for all 15 method/subset cases',
                         'all_bootstrap_step_denominators_positive': True,
                         'native_prm_saved_overall_parity': True}
    (OUT/'PAIRED_DECISION_ANALYSIS.json').write_text(json.dumps(summary, indent=2)+'\n')
    write_csv(OUT/'paired_prmscore_contrasts.csv', summary['contrasts'])
    write_csv(OUT/'prm_threshold_curve.csv', curves)
    own = json.loads((OUT/'DECISION_ANALYSIS.json').read_text())
    own_curves = list(csv.DictReader((OUT/'threshold_curves.csv').open()))
    fig, ax = plt.subplots(figsize=(9,4.7))
    colors = {'LSML_CONT':'#246c9c', 'IU_PCR':'#278577', 'PRM_NATIVE':'#a75e70'}
    labels = {'LSML_CONT':'Continuous L-SML', 'IU_PCR':'IU-PCR', 'PRM_NATIVE':'PRM-7B'}
    for m in colors:
        rs = curves if m == 'PRM_NATIVE' else [r for r in own_curves if r['method']==m]
        ax.plot([100*float(r['actual_flag_rate']) for r in rs], [float(r['prm_score']) for r in rs], color=colors[m], label=labels[m])
        base = summary['rules'][m]
        ax.scatter(100*base['flag_rate_all'], base['prm_score'], color=colors[m], s=45, zorder=3)
        oracle = summary['oracle_diagnostics'][m]['GLOBAL_THRESHOLD_ORACLE'] if m=='PRM_NATIVE' else own['methods'][m]['GLOBAL_THRESHOLD_ORACLE']
        rate = oracle['flag_rate_all'] if m=='PRM_NATIVE' else oracle['step_flag_rate_all']
        ax.scatter(100*rate, oracle['prm_score'], marker='*', s=100, color=colors[m], zorder=3)
    ax.set(xlabel='Flagged steps (% of all steps, including synthetic controls)', ylabel='Pooled PRMScore',
           title='Fair scalar-threshold diagnosis: PRM also has substantial headroom', xlim=(0,50), ylim=(.58,.72))
    ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.text(.02,.01,'Same 6,211 noncontrol answers. Circles: frozen rules; stars: label-selected in-sample oracles. Not held-out gains.', fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1))
    for ext in ['png','svg']: fig.savefig(OUT/f'paired_threshold_curves.{ext}', dpi=180)
    plt.close(fig)
    print(json.dumps({'prm': summary['rules']['PRM_NATIVE'], 'prm_oracle':summary['oracle_diagnostics']['PRM_NATIVE'],
                      'contrasts':summary['contrasts'][:5]}, indent=2))

if __name__ == '__main__':
    main()
