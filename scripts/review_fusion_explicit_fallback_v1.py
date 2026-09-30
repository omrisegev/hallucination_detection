"""Independent routing, source inheritance, label, metric and bootstrap review.

Does not import the composite implementation or evaluator. Previously audited
source fits are reused, not refitted. Bootstrap explicitly materializes rows.
"""
import os
for option in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[option] = '1'
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/fusion_explicit_fallback_pilot_v1'
PARENT = ROOT / 'results/fusion_context_bank_pilot_v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def auc_value(y, x):
    y = np.asarray(y); x = np.asarray(x, float)
    pos, neg = x[y == 1], x[y == 0]
    if not len(pos) or not len(neg): return None
    return float(np.mean((pos[:, None] > neg) + .5 * (pos[:, None] == neg)))


def prm(rows, arm):
    records = [r for r in rows if r['cell'].startswith('prm') and r['valid'][arm]]
    if not records: return {'answers': 0, 'auroc': None, 'within_answer_auc': None, 'mixed_answers': 0}
    within = [auc_value(r['target'], r['scores'][arm]) for r in records]
    within = [v for v in within if v is not None]
    return {'answers': len(records), 'auroc': auc_value(np.concatenate([r['target'] for r in records]),
            np.concatenate([r['scores'][arm] for r in records])),
            'within_answer_auc': float(np.mean(within)) if within else None, 'mixed_answers': len(within)}


def pb(rows, arm, fixed=False):
    result = {}
    for cell in sorted({r['cell'] for r in rows if r['cell'].startswith('pb_')}):
        records = [r for r in rows if r['cell'] == cell]
        category = {True: [], False: []}; valid_count = 0
        for r in records:
            valid = r['fixed_iu_valid'][arm] if fixed else r['decision_valid'][arm]
            pred = r['fixed_iu_predictions'][arm] if fixed else r['predictions'][arm]
            valid_count += bool(valid)
            category[r['target'] == -1].append(int(valid and pred == r['target']))
        ca = sum(category[True]) / len(category[True]) if category[True] else None
        ea = sum(category[False]) / len(category[False]) if category[False] else None
        f1 = None if ca is None or ea is None else (2 * ca * ea / (ca + ea) if ca + ea else 0.)
        result[cell] = {'answers': len(records), 'clean': len(category[True]), 'erroneous': len(category[False]),
            'clean_accuracy': ca, 'error_exact_accuracy': ea, 'f1': f1, 'valid_decisions': valid_count}
    fs = [v['f1'] for v in result.values()]
    return {'cells': result, 'macro_f1': float(np.mean(fs)) if fs and None not in fs else None}


def check_equal(actual, expected, path='root'):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected: check_equal(actual[key], expected[key], path + '/' + key)
    elif expected is None: assert actual is None, path
    elif isinstance(expected, (float, list)):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, err_msg=path)
    else: assert actual == expected, (path, actual, expected)


def explicit_bootstrap(rows, left, right):
    pp = [r for r in rows if r['cell'].startswith('prm') and r['valid'][left] and r['valid'][right]]
    bb = [r for r in rows if r['cell'].startswith('pb_')]
    def strata(records):
        result = {}
        for r in records: result.setdefault(r['cell'], {}).setdefault(r['group_id'], []).append(r)
        return result
    def sample(groups, rng):
        chosen = []
        for cell in groups.values():
            ids = sorted(cell)
            for i in rng.integers(len(ids), size=len(ids)): chosen.extend(cell[ids[i]])
        return chosen
    ps, bs = strata(pp), strata(bb); rng = np.random.default_rng(2026090706)
    series = {key: [] for key in ('prm_common_valid', 'prm_within_answer_common_valid',
                                 'pb_all_population', 'pb_common_iu_gate_all_population')}
    for _ in range(1000):
        if pp:
            chosen = sample(ps, rng); a, b = prm(chosen, left), prm(chosen, right)
            for metric, name in (('auroc', 'prm_common_valid'), ('within_answer_auc', 'prm_within_answer_common_valid')):
                if a[metric] is not None and b[metric] is not None: series[name].append(a[metric] - b[metric])
        chosen = sample(bs, rng)
        for fixed, name in ((False, 'pb_all_population'), (True, 'pb_common_iu_gate_all_population')):
            a, b = [pb(chosen, arm, fixed)['macro_f1'] for arm in (left, right)]
            if a is not None and b is not None: series[name].append(a - b)
    result = {}
    for key, values in series.items():
        result[key + '_ci95'] = np.quantile(values, [.025, .975]).tolist() if values else None
        result[key + '_valid_draws'] = len(values)
    return result


def review():
    started = time.monotonic(); counts = Counter()
    manifest, frozen, evaluation, contrasts = [load(OUT / n) for n in (
        'MANIFEST.json', 'SCORES_FROZEN.json', 'EVALUATION.json', 'CONTRASTS.json')]
    assert frozen['manifest_sha256'] == sha(OUT / 'MANIFEST.json')
    assert evaluation['scores_sha256'] == sha(OUT / 'SCORES_FROZEN.json')
    assert contrasts['state'] == 'COMPLETE' and len(contrasts['pairs']) == 30
    assert contrasts['evaluation_sha256'] == sha(OUT / 'EVALUATION.json')
    for path, expected in {**manifest['hashes'], **frozen['files']}.items(): assert sha(path) == expected, path
    assert manifest['labels_decoded_for_composition'] is False and frozen['labels_decoded'] is False
    parent_review = load(PARENT / 'REVIEW.json'); assert parent_review['status'] == 'PASS'
    assert parent_review['review_script_sha256'] == sha(ROOT / 'scripts/review_fusion_context_bank_v1.py')
    release = load(ROOT / 'results/answer_localization_representation_pilot_v1/RELEASE.json')
    labels = {}
    for cell in {r['cell'] for r in manifest['selected']}:
        info = release['cells'][cell]; assert sha(info['label_path']) == info['label_opaque_sha256']
        with np.load(info['label_path'], allow_pickle=False) as z: labels[cell] = {k: z[k].copy() for k in z.files}
    rows = evaluation['rows']; by_uid = {r['uid']: r for r in rows}
    assert len(by_uid) == len(rows) == len(manifest['selected']) == 58
    routes = {task: {p: Counter() for p in ('single', 'dual')} for task in ('prm', 'pb')}
    transitions = []
    for rec in manifest['selected']:
        uid = rec['uid']; row = by_uid[uid]; bundle = labels[rec['cell']]
        indices = np.flatnonzero(bundle['row_ids'].astype(str) == rec['row_id']); assert len(indices) == 1
        i = indices.item()
        if rec['cell'].startswith('prm'):
            a, b = bundle['step_flag_offsets'][i:i+2]; target = bundle['step_error_flags'][a:b]
        else: target = bundle['first_error'][i]
        np.testing.assert_array_equal(row['target'], target); counts['direct_label_joins'] += 1
        meta = load(OUT / 'scores' / (uid + '.json')); parent = load(PARENT / 'scores' / (uid + '.json'))
        assert meta['labels_decoded'] is False
        old_valid = parent['methods']['moment__joint0']['valid']
        new_valid = parent['methods']['context__joint0']['valid']
        # A separate truth table, not the production route function.
        single = 'moment_joint' if old_valid else 'moment_iu'
        dual = {(True, True): 'moment_joint', (True, False): 'moment_joint',
                (False, True): 'context_joint', (False, False): 'moment_iu'}[(old_valid, new_valid)]
        assert meta['routing'] == {'eligibility': {'moment': old_valid, 'context': new_valid},
                                    'routes': {'single': single, 'dual': dual}}
        assert row['routing'] == meta['routing']; counts['independent_route_pairs'] += 1
        task = 'prm' if rec['cell'].startswith('prm') else 'pb'
        routes[task]['single'][single] += 1; routes[task]['dual'][dual] += 1
        with np.load(OUT / 'scores' / (uid + '.npz'), allow_pickle=False) as out, np.load(
            PARENT / 'scores' / (uid + '.npz'), allow_pickle=False) as source:
            for arm in manifest['arms']:
                if arm.startswith(('single__', 'dual__')):
                    policy, core = arm.split('__'); route = single if policy == 'single' else dual
                    if core in ('equal', 'iu'):
                        expected = ('context' if route == 'context_joint' else 'moment') + '__' + core
                    elif route == 'moment_iu': expected = 'moment__iu'
                    else: expected = ('moment' if route == 'moment_joint' else 'context') + '__' + core
                else: expected, route = arm, 'unchanged_reference'
                detail = meta['methods'][arm]; original = parent['methods'][expected]
                assert detail['source_arm'] == row['sources'][arm] == expected and detail['route'] == route
                assert {k: v for k, v in detail.items() if k not in ('source_arm', 'route')} == original
                for field in ('valid', 'decision_valid', 'fixed_iu_valid'): assert row[field][arm] == original[field]
                assert row['predictions'][arm] == original.get('prediction')
                assert row['fixed_iu_predictions'][arm] == original.get('fixed_iu_prediction')
                assert row['peaks'][arm] == original.get('peak')
                counts['exact_source_metadata'] += 1
                if original['valid']:
                    for suffix in ('window', 'risk'):
                        np.testing.assert_array_equal(out[arm + '__' + suffix], source[expected + '__' + suffix])
                        counts['exact_score_array_replays'] += 1
                    np.testing.assert_array_equal(row['scores'][arm], source[expected + '__risk'])
                    peak = int(np.argmax(row['scores'][arm])); assert peak == row['peaks'][arm]
                    assert row['fixed_iu_predictions'][arm] == (peak if row['predictions']['moment__iu'] != -1 else -1)
                    counts['peak_and_common_gate_checks'] += 1
                else: assert arm not in row['scores']
        if task == 'pb':
            transitions.append({'uid': uid, 'cell': rec['cell'], 'target': int(target), 'dual_route': dual,
                'changes': {core: {'prediction_changed': row['predictions']['single__' + core] != row['predictions']['dual__' + core],
                    'peak_changed': row['peaks']['single__' + core] != row['peaks']['dual__' + core],
                    'native_single': row['predictions']['single__' + core], 'native_dual': row['predictions']['dual__' + core]}
                    for core in ('joint0', 'graph010', 'graph_perm')}})
    assert set(by_uid) == {r['uid'] for r in manifest['selected']}
    parent_eval = load(PARENT / 'EVALUATION.json')
    for arm in manifest['arms']:
        actual = {'prm': prm(rows, arm), 'pb': pb(rows, arm), 'pb_common_iu_gate': pb(rows, arm, True)}
        check_equal(evaluation['metrics'][arm], actual, arm); counts['independent_metric_bundles'] += 1
        if arm in parent_eval['metrics']:
            assert evaluation['metrics'][arm] == parent_eval['metrics'][arm]; counts['unchanged_parent_bundles'] += 1
    for left, right in manifest['contrasts']:
        pair = contrasts['pairs'][left + ' minus ' + right]
        common = [r for r in rows if r['valid'][left] and r['valid'][right]]
        for side, arm in (('left', left), ('right', right)):
            check_equal(pair[side + '_prm'], prm(common, arm))
            check_equal(pair[side + '_pb'], pb(rows, arm))
            check_equal(pair[side + '_pb_common_iu_gate'], pb(rows, arm, True))
        counts['paired_point_bundles'] += 1
    checked = []
    for left, right in (('single__joint0', 'moment__iu'), ('dual__joint0', 'single__joint0'),
                        ('dual__graph010', 'dual__graph_perm')):
        t = time.monotonic(); explicit = explicit_bootstrap(rows, left, right)
        computed = contrasts['pairs'][left + ' minus ' + right]['uncertainty']
        for key, value in explicit.items(): check_equal(computed[key], value, key)
        checked.append({'left': left, 'right': right, 'draws': 1000, 'four_endpoint_intervals': 'MATCH',
                        'defined_draw_counts': 'MATCH', 'seconds': time.monotonic() - t})
        print('Explicit 1000-draw / four-endpoint bootstrap matches:', left, 'minus', right, flush=True)
    # Review identified an incumbent omitted from the registered paired roster.
    # Keep the original 30 contrasts frozen; label these as post-evaluation.
    additional = {'status': 'POST_EVALUATION_EXPLORATORY_REVIEW',
                  'reason': 'Always-context equal fusion is an existing full-coverage comparator; do not hide it behind routed controls.',
                  'evaluation_sha256': sha(OUT / 'EVALUATION.json'), 'pairs': {}}
    for left in ('single__joint0', 'dual__joint0'):
        right = 'context__equal'
        additional['pairs'][left + ' minus ' + right] = {
            'left': left, 'right': right, 'left_prm': prm(rows, left), 'right_prm': prm(rows, right),
            'left_pb': pb(rows, left), 'right_pb': pb(rows, right),
            'uncertainty': explicit_bootstrap(rows, left, right)}
    save(OUT / 'ADDITIONAL_COMPARISONS.json', additional)
    decision_changes = {}
    for core in ('joint0', 'graph010', 'graph_perm'):
        rr = [r for r in rows if r['cell'].startswith('pb_')]
        a, b = 'single__' + core, 'dual__' + core
        decision_changes[core] = {
            'prediction_changed': sum(r['predictions'][a] != r['predictions'][b] for r in rr),
            'peak_changed': sum(r['peaks'][a] != r['peaks'][b] for r in rr),
            'exact_success_changed': sum((r['decision_valid'][a] and r['predictions'][a] == r['target']) !=
                (r['decision_valid'][b] and r['predictions'][b] == r['target']) for r in rr)}
    subgroup = {}
    for route in ('moment_joint', 'context_joint', 'moment_iu'):
        rr = [r for r in rows if r['routing']['routes']['dual'] == route]
        subgroup[route] = {'answers': len(rr), 'prm': {arm: prm(rr, arm) for arm in (
            'moment__iu', 'dual__iu', 'dual__equal', 'single__joint0', 'dual__joint0', 'dual__graph010')},
            'pb_answers': sum(r['cell'].startswith('pb_') for r in rr)}
    hit_counts = {}
    for arm in manifest['arms']:
        rr = [r for r in rows if r['cell'].startswith('pb_')]
        hit_counts[arm] = {'fit_valid': sum(r['valid'][arm] for r in rows),
            'clean_hits': sum(r['decision_valid'][arm] and r['predictions'][arm] == -1 for r in rr if r['target'] == -1),
            'exact_error_hits': sum(r['decision_valid'][arm] and r['predictions'][arm] == r['target'] for r in rr if r['target'] != -1),
            'error_peak_hits': sum(r['valid'][arm] and r['peaks'][arm] == r['target'] for r in rr if r['target'] != -1)}
    result = {'status': 'PASS', 'counts': dict(counts), 'route_counts': routes, 'pb_transitions': transitions,
              'pb_decision_changes': decision_changes, 'additional_comparisons_sha256': sha(OUT / 'ADDITIONAL_COMPARISONS.json'),
              'subgroup_metrics_descriptive': subgroup, 'hit_counts': hit_counts, 'bootstrap_reference_checks': checked,
              'scope': 'Independent composition and metrics; audited parent fits reused, not refitted.',
              'source_hashes': {str(OUT / n): sha(OUT / n) for n in ('MANIFEST.json', 'SCORES_FROZEN.json', 'EVALUATION.json', 'CONTRASTS.json')},
              'parent_review_sha256': sha(PARENT / 'REVIEW.json'), 'review_script_sha256': sha(__file__),
              'seconds': time.monotonic() - started}
    save(OUT / 'REVIEW.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'counts', 'route_counts', 'seconds')}, indent=2), flush=True)


if __name__ == '__main__': review()
