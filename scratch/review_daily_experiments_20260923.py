"""Read-only checks of September 23 artifacts; write a separate review JSON.

No fitting, inference, source-result changes, or new confidence intervals.
Native-window comparisons reconstruct common-population points using the
original runner's exact fallback-to-equal identity, not saved raw predictions.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SSL = ROOT / '.worktrees/ssl-pseudolabel-residual-v1/results/ssl_pseudolabel_residual_v1'
LEVERS = ROOT / '.worktrees/lsml-ct7-levers-run'
OUT = ROOT / 'scratch/daily_experiment_review_20260923/AUDIT.json'
hashes = {}


def source(path):
    hashes[str(path.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(source(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with source(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def count(value):
    number = float(value)
    assert number.is_integer()
    return int(number)


audit = {'scope': 'Saved-artifact review, not a complete raw-input or training audit', 'ssl': {}}
for stage in ('S1', 'S2', 'S3', 'S5'):
    directory = SSL / stage / 'run_20260923'
    rows = read_csv(directory / 'OOF_ANSWERS.csv')
    pb = [r for r in rows if r['cell'].startswith('pb_')]
    prm = [r for r in rows if not r['cell'].startswith('pb_')]
    assert len(rows) == len({r['uid'] for r in rows}) == 13769
    assert len(pb) == 6800 and len(prm) == 6969
    groups = {}
    for row in rows:
        groups.setdefault(row['source_group'], set()).add(row['fold'])
    assert all(len(folds) == 1 for folds in groups.values())
    splits = read_json(directory / 'SPLITS.json')
    assert len(splits) == 15
    assert all(v['pairwise_group_overlap'] == 0 and v['covers_task'] for v in splits.values())
    cells = sorted({r['cell'] for r in pb})
    errors = {cell: [r for r in pb if r['cell'] == cell and int(r['target']) >= 0] for cell in cells}
    replay = {}
    for column in rows[0]:
        if not column.endswith('__pred'):
            continue
        method = column[:-6]
        sla = mean(mean(float(r[column]) == int(r['target']) for r in errors[cell]) for cell in cells)
        aucs = [float(r[method + '__within_auc']) for r in prm if r[method + '__within_auc']]
        assert len(aucs) == 6030
        replay[method] = {'pb_sla_macro8': sla, 'prm_within_auc_from_saved_per_answer_metrics': mean(aucs)}
    contrasts = read_csv(directory / 'CONTRASTS.csv')
    checks = []
    for contrast in contrasts:
        a, b = contrast['contrast_id'].split(' - ')
        key = ('pb_sla_macro8' if contrast['endpoint'] == 'pb_sla_macro8'
               else 'prm_within_auc_from_saved_per_answer_metrics')
        checks.append(abs(replay[a][key] - replay[b][key] - float(contrast['delta'])))
    assert max(checks) < 1e-12
    audit['ssl'][stage] = {
        'answers': len(rows), 'pb_answers': len(pb), 'pb_erroneous': sum(map(len, errors.values())),
        'prm_answers': len(prm), 'prm_auc_eligible': 6030, 'source_fold_crossings': 0,
        'split_records_checked': len(splits), 'recorded_split_overlaps': 0,
        'methods_checked': len(replay), 'contrasts_checked': len(checks), 'max_delta_error': max(checks),
        'paired_N_field': sorted({int(c['paired_N']) for c in contrasts}),
        'bootstrap_B_field': sorted({int(c['B']) for c in contrasts}),
        'paired_N_issue': 'Field contains bootstrap draws, not the paired sample size.',
        'metrics': replay,
    }

coverage = read_csv(SSL / 'S1/run_20260923/TRAINING_COVERAGE.csv')
agreement = [r for r in coverage if r['arm'] == 'P_AGREE']
audit['agreement_coverage'] = {}
for task, part in (('pb', [r for r in agreement if r['task'].startswith('pb')]),
                   ('prm', [r for r in agreement if r['task'].startswith('prm')])):
    totals = {k: (sum(count(r[k]) for r in part) if all(r[k] for r in part) else None)
              for k in ('B_answers', 'trainable_answers', 'selected_steps', 'total_steps')}
    totals['answer_fraction'] = totals['trainable_answers'] / totals['B_answers']
    totals['step_fraction'] = (totals['selected_steps'] / totals['total_steps']
                              if totals['selected_steps'] is not None else None)
    audit['agreement_coverage'][task] = totals

window = read_json(LEVERS / 'results/window_representation_b3_v1/fusion/RESULTS.json')
source(LEVERS / 'scripts/experiments/window_answer_local_fusion_v1.py')
assert window['fits']['uncovered_answers_below_width'] == 0
audit['matched_native_window'] = {}
for arm in ('iu', 'shrink_iu', 'lsml'):
    assert window['fits'][arm]['failed_fits'] == 0
    name = 'window_' + arm + '_top10'
    full, native, equal = (window['pb'][key]['cells'] for key in (name, name + '_native', 'window_equal_top10'))
    by_cell = {}
    for cell, n in native.items():
        f, e = full[cell], equal[cell]
        assert f['errors'] == e['errors']
        hits = [v['sla'] * v['errors'] for v in (f, n, e)]
        assert all(math.isclose(h, round(h), abs_tol=1e-8) for h in hits)
        full_hits, native_hits, equal_hits = map(round, hits)
        native_equal_hits = equal_hits - (full_hits - native_hits)
        assert 0 <= native_equal_hits <= n['errors']
        by_cell[cell] = {'native_answers': n['n'], 'native_errors': n['errors'],
                         'native_learned_hits': native_hits, 'native_equal_hits': native_equal_hits,
                         'learned_sla': n['sla'], 'equal_sla': native_equal_hits / n['errors']}
    l = mean(v['learned_sla'] for v in by_cell.values())
    e = mean(v['equal_sla'] for v in by_cell.values())
    pf, pn, pe = (window['prm'][key] for key in (name, name + '_native', 'window_equal_top10'))
    assert pf['eligible'] == pe['eligible']
    native_equal_auc = pn['within_auc'] - (pf['within_auc'] - pe['within_auc']) * pf['eligible'] / pn['eligible']
    audit['matched_native_window'][arm] = {
        'derivation': 'Fallback scores equal; subtract fallback hit counts / AUC sums from full-population totals.',
        'pb_cells': by_cell, 'pb_native_answers': sum(v['native_answers'] for v in by_cell.values()),
        'pb_native_errors': sum(v['native_errors'] for v in by_cell.values()),
        'pb_native_equal_sla': e, 'pb_native_learned_sla': l, 'pb_native_delta_pp': 100 * (l - e),
        'prm_native_answers': pn['answers'], 'prm_native_eligible': pn['eligible'],
        'prm_native_equal_auc': native_equal_auc, 'prm_native_learned_auc': pn['within_auc'],
        'prm_native_delta_auc': pn['within_auc'] - native_equal_auc,
        'uncertainty': 'No valid matched-native interval calculated; points do not establish improvement.',
    }

previous = read_json(ROOT / 'scratch/step_evidence_plan_review_20260923/AUDIT.json')
step = ROOT / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
audit['step432_previous_replay_still_applies'] = {}
for name, key in (('SUMMARY.csv', 'summary_sha256'), ('REPORT_MANIFEST.json', 'manifest_sha256')):
    digest = hashlib.sha256(source(step / name).read_bytes()).hexdigest()
    matches = digest == previous['manifest_replay'][key]
    assert matches
    audit['step432_previous_replay_still_applies'][name] = matches
audit['source_sha256'] = hashes
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf8')
print(json.dumps({'output': str(OUT), 'ssl_stages': len(audit['ssl']),
                  'methods_replayed': sum(v['methods_checked'] for v in audit['ssl'].values()),
                  'contrasts_replayed': sum(v['contrasts_checked'] for v in audit['ssl'].values()),
                  'agreement_coverage': audit['agreement_coverage'],
                  'lsml_native_delta_pp': audit['matched_native_window']['lsml']['pb_native_delta_pp']}, indent=2))
