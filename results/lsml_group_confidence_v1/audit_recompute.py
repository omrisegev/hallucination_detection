"""Independent raw-prediction metric replay; imports no experiment evaluators."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent
reads = {}

def read_bytes(path):
    path = Path(path)
    data = path.read_bytes()
    reads[str(path)] = hashlib.sha256(data).hexdigest()
    return data

freeze = json.loads(read_bytes(OUT / 'FREEZE.json'))
seal = json.loads(read_bytes(OUT / 'SEAL.json'))
assert reads[str(OUT / 'FREEZE.json')] == seal['freeze_sha256']
read_bytes(OUT / 'PREDICTIONS.npz')
assert reads[str(OUT / 'PREDICTIONS.npz')] == seal['predictions_sha256']
paths = {k: Path(v['path']) for k, v in freeze['inputs'].items()}
for key in ['arrays', 'joined', 'metadata', 'folds', 'oof_answers']:
    read_bytes(paths[key])
    assert reads[str(paths[key])] == freeze['inputs'][key]['sha256']
truth = dict(np.load(paths['arrays']))
predictions = np.load(OUT / 'PREDICTIONS.npz')
records = json.loads(paths['joined'].read_text())['records']
answers = pd.read_csv(paths['oof_answers'])
metadata = {v['idx']: v for v in pickle.loads(paths['metadata'].read_bytes()).values()}
foldmap = json.loads(paths['folds'].read_text())['outer']
offsets = truth['offsets']
error = truth['labels'].astype(bool)
np.testing.assert_array_equal(offsets, predictions['offsets'])
np.testing.assert_array_equal(predictions['writes'], np.ones(len(error), dtype=np.int8))
np.testing.assert_array_equal(answers.target, truth['target'])
np.testing.assert_array_equal(answers.fold, predictions['folds'])
assert len(answers) == len(records) == 13769
assert len(error) == offsets[-1] == 145597
prm = answers.cell.str.startswith('prm').to_numpy()
selected = np.zeros(len(answers), bool)
auc_eligible = np.zeros(len(answers), bool)
pb_eligible = np.zeros(len(answers), bool)
for i, rec in enumerate(records):
    assert (rec['uid'], rec['row_id'], rec['group_id'], rec['cell']) == (
        answers.uid[i], answers.id[i], answers.source_group[i], answers.cell[i])
    assert answers.fold[i] == foldmap[rec['group_id']]
    a, b = offsets[i:i+2]
    if prm[i]:
        official = metadata[answers.id[i]]
        expected = np.isin(np.arange(1, b-a+1), official['error_steps'])
        np.testing.assert_array_equal(error[a:b], expected)
        selected[i] = official['classification'] != 'correct'
        auc_eligible[i] = error[a:b].any() and not error[a:b].all()
    else:
        # PB exposes first-error target only; generic step labels are sentinel -2.
        np.testing.assert_array_equal(truth['labels'][a:b], -2)
        assert -1 <= truth['target'][i] < b-a
        pb_eligible[i] = truth['target'][i] >= 0
assert (prm.sum(), selected.sum(), auc_eligible.sum(), pb_eligible.sum()) == (6969, 6211, 6030, 4442)
stepmask = np.repeat(selected, np.diff(offsets))
assert stepmask.sum() == 83371
pb_cells = sorted(set(answers.cell[pb_eligible]))
assert len(pb_cells) == 8
results = {}
for arm in freeze['arms']:
    scores = predictions[arm+'_score']
    correct_predictions = predictions[arm+'_pred']
    assert scores.shape == error.shape and np.isfinite(scores).all()
    assert np.isin(correct_predictions, [0, 1]).all()
    true_correct = ~error[stepmask]
    predicted_correct = correct_predictions[stepmask].astype(bool)
    tp = int((true_correct & predicted_correct).sum())
    tn = int((~true_correct & ~predicted_correct).sum())
    fp = int((~true_correct & predicted_correct).sum())
    fn = int((true_correct & ~predicted_correct).sum())
    positive_f1 = 2*tp/(2*tp+fp+fn)
    negative_f1 = 2*tn/(2*tn+fp+fn)
    aucs = []
    cell_hits = {c: [] for c in pb_cells}
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        if auc_eligible[i]:
            y = error[a:b]
            npos, nneg = int(y.sum()), int((~y).sum())
            rank_sum = rankdata(scores[a:b], method='average')[y].sum()
            aucs.append((rank_sum-npos*(npos+1)/2)/(npos*nneg))
        if pb_eligible[i]:
            tied = np.flatnonzero(scores[a:b] >= scores[a:b].max()-8*np.finfo(float).eps)
            cell_hits[answers.cell[i]].append(int(tied[0] == truth['target'][i]))
    results[arm] = {
        'prmscore': (positive_f1+negative_f1)/2,
        'prm_within_auc': float(np.mean(aucs)),
        'pb_sla_macro8': float(np.mean([np.mean(v) for v in cell_hits.values()])),
        'confusion_correct_positive': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
        'pb_cells': {c: {'n': len(v), 'hit': int(sum(v)), 'mean': float(np.mean(v))} for c,v in cell_hits.items()},
    }
delta = {arm: {m: results['confidence_sum'][m]-results[arm][m]
               for m in ['prmscore', 'prm_within_auc', 'pb_sla_macro8']}
         for arm in freeze['arms'] if arm != 'confidence_sum'}
read_bytes(__file__)
report = {'status': 'PASS', 'independence': 'No METRICS, CONTRASTS, STATUS, reports, HISTORY or evaluator imports read.',
          'counts': {'answers': len(answers), 'steps': len(error), 'prm_noncontrol_answers': int(selected.sum()),
                     'prm_noncontrol_steps': int(stepmask.sum()), 'within_auc_answers': len(aucs),
                     'pb_error_answers': int(pb_eligible.sum()), 'pb_cells': len(pb_cells), 'arms': len(results)},
          'results': results, 'candidate_minus_comparator': delta,
          'input_sha256': reads, 'command': f'python "{Path(__file__).resolve()}"',
          'comparison_to_original': 'Parent must compare; original metric files deliberately unread.',
          'pb_label_contract': 'All 6800 PB answers have step-label sentinel -2; evaluate explicit first-error target, exactly matched between CSV and JOINED.npz. Initial extra audit assertion treating sentinel as bool was invalid and corrected before metric computation.'}
(OUT / 'AUDIT_RECOMPUTE.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'counts': report['counts'], 'metrics': {a: {m:r[m] for m in ['prmscore','prm_within_auc','pb_sla_macro8']} for a,r in results.items()}, 'deltas': delta}, indent=2))
