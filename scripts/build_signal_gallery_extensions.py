"""Frozen-feature PRMScore and conditional binary-error dependence diagnostics.

No feature/model/sign selection, threshold tuning, or new fusion is performed.
Predictions and their input identity are sealed before opening evaluation labels.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.advisor_controls import metrics
from spectral_utils.prmbench import prmbench_evaluate

SOURCE = ROOT / "results/advisor_completion_2026-10-11"
GALLERY = ROOT / "docs/reviews/signal_visualization_options_20261012"
OUT = ROOT / "results/signal_gallery_extensions_2026-10-12"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def dump(path, value, compact=False):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, allow_nan=False,
                                   indent=None if compact else 2) + '\n')

def csvout(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(clean(rows))

def pairwise_binary_errors(error):
    """Exact empirical 2x2 phi and MI bits; constant margins/diagonal undefined."""
    n, width = error.shape
    blank = np.full((width, width), np.nan)
    if n == 0: return blank.copy(), blank.copy(), np.full(width, np.nan)
    e = error.astype(np.int64)
    ones = e.sum(axis=0); n11 = e.T @ e
    cells = (n11, ones[:, None]-n11, ones[None, :]-n11,
             n-ones[:, None]-ones[None, :]+n11)
    denominator = np.sqrt((ones*(n-ones))[:, None].astype(float) *
                          (ones*(n-ones))[None, :].astype(float))
    phi = np.divide(n*n11-ones[:, None]*ones[None, :], denominator,
                    out=blank.copy(), where=denominator > 0)
    p = ones/n
    expected = (p[:, None]*p[None, :], p[:, None]*(1-p[None, :]),
                (1-p[:, None])*p[None, :], (1-p[:, None])*(1-p[None, :]))
    mi = np.zeros((width, width))
    for counts, independent in zip(cells, expected):
        joint = counts/n
        active = joint > 0
        term = np.zeros_like(joint)
        term[active] = joint[active] * np.log2(joint[active]/independent[active])
        mi += term
    mi[denominator == 0] = np.nan
    np.fill_diagonal(phi, np.nan); np.fill_diagonal(mi, np.nan)
    return phi, mi, p

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    gallery = json.loads((GALLERY / 'DATA.json').read_text())
    features = gallery['features']; names = [f['id'] for f in features]
    axes = gallery['axes'] + [{'code':'MS', 'key':'multi_solutions',
        'name':'Multi-solution consistency', 'he':'עקביות בין פתרונות'}]
    contexts = ['ALL_NONCONTROL'] + [a['key'] for a in axes]
    input_paths = [SOURCE / 'inputs' / name for name in ('population.npz',
        'bank_V_N56.npz', 'bank_V_N16.npz', 'FOLDS_V2.json', 'prmbench_meta.json')]
    input_paths += [GALLERY / 'DATA.json', ROOT / 'spectral_utils/prmbench.py',
                    ROOT / 'spectral_utils/advisor_controls.py', Path(__file__)]
    freeze = {str(p.relative_to(ROOT)): sha(p) for p in input_paths}
    dump(OUT / 'INPUT_FREEZE.json', freeze)
    protocol = {'version':1, 'status':'LOCKED_BEFORE_PREDICTION_AND_LABEL_EVALUATION',
        'purpose':'descriptive development visualization; no inferential independence certification or promotion',
        'features':'all 56 fixed signed answer-z V_N56 channels, unchanged label-free signs',
        'prediction':'per-feature global Q80 of donor V values in four source folds; heldout V >= threshold',
        'normalization':'V is already signed per-answer z; no second normalization or label-oriented sign repair',
        'split':'unchanged five FOLDS_V2 source-disjoint folds, including controls among donors',
        'evaluation':'labels/types opened only after prediction seal',
        'classification_error':'E_j = 1(predicted_error_j != true_error)',
        'conditional_dependence':'pairwise phi(E_i,E_j | Y=valid), separately phi(... | Y=error)',
        'mi':'empirical binary pair mutual information in bits, same class conditioning',
        'undefined':'zero class count or constant feature error margin => null, diagonal null; never imputed zero',
        'aggregates':'UI computes mean absolute phi over selected defined peers within each truth class; max of both class means requires both defined; display peer coverage',
        'aggregation_warning':'pairwise diagnostics are not mutual conditional independence; small estimates are not proof; no class/answer nuisance adjustment or confidence intervals',
        'prmscore':'pooled macro F1 valid/error excluding synthetic correct controls; per-type macro F1 except MS=valid-step F1',
        'adaptation':'NR/NCL official validity fallback; no separate redundancy head',
        'family_summary':'equal mean of member-feature PRMScores, not fused-family PRMScore',
        'MS':'all valid; error-conditioned dependence undefined; no AUC',
        'contexts':contexts, 'feature_order':names, 'input_hashes':freeze}
    dump(OUT / 'PROTOCOL.json', protocol)
    with np.load(SOURCE / 'inputs/population.npz', allow_pickle=False) as z:
        ids, off, groups, folds = (z[k] for k in ('idx', 'offsets', 'group', 'fold'))
    with np.load(SOURCE / 'inputs/bank_V_N56.npz', allow_pickle=False) as z:
        source_names = z['columns'].astype(str).tolist()
        assert np.array_equal(z['offsets'], off)
        V = z['V'][:, [source_names.index(n) for n in names]]
    assert V.shape == (94203, 56) and np.isfinite(V).all()
    with np.load(SOURCE / 'inputs/bank_V_N16.npz', allow_pickle=False) as z:
        small_names = z['columns'].astype(str).tolist()
        assert np.array_equal(z['offsets'], off)
        shared_diff = float(np.max(np.abs(z['V'] - V[:, [names.index(n) for n in small_names]])))
        assert shared_diff <= 1e-9
    map_folds = json.loads((SOURCE / 'inputs/FOLDS_V2.json').read_text())['prmbench']['outer']
    assert all(int(map_folds[str(g)]) == int(f) for g, f in zip(groups, folds))
    step_folds = np.repeat(folds, np.diff(off))
    thresholds = np.full((5, 56), np.nan); flags = np.zeros(V.shape, bool)
    fold_records = []
    for f in range(5):
        train_groups, test_groups = set(groups[folds != f]), set(groups[folds == f])
        assert not train_groups & test_groups
        train, test = step_folds != f, step_folds == f
        thresholds[f] = np.quantile(V[train], .8, axis=0)
        flags[test] = V[test] >= thresholds[f]
        fold_records.append({'fold':f, 'n_train_steps':int(train.sum()),
            'n_test_steps':int(test.sum()), 'train_groups':sorted(train_groups),
            'test_groups':sorted(test_groups)})
    np.savez_compressed(OUT / 'PREDICTIONS.npz', flags=flags, thresholds=thresholds,
        columns=np.asarray(names), idx=ids, offsets=off, group=groups, fold=folds)
    dump(OUT / 'FOLDS.json', fold_records)
    dump(OUT / 'PREDICTION_SEAL.json', {'status':'FROZEN_BEFORE_LABEL_EVALUATION',
        'files':{n:sha(OUT / n) for n in ('PROTOCOL.json','INPUT_FREEZE.json','PREDICTIONS.npz','FOLDS.json')}})
    print('Predictions sealed before label evaluation', flush=True)
    with np.load(SOURCE / 'inputs/population.npz', allow_pickle=False) as z:
        y, classifications = z['y'].astype(bool), z['classification'].astype(str)
    metadata = {str(r['idx']):r for r in json.loads((SOURCE / 'inputs/prmbench_meta.json').read_text())['rows']}
    metas = [metadata[str(i)] for i in ids]
    classes_steps = np.repeat(classifications, np.diff(off))
    errors = flags != y[:, None]
    payload = {'version':1, 'features':features, 'axes':axes, 'contexts':contexts,
               'featurePrm':{}, 'familyPrm':{'all':{},'n16':{}}, 'dependence':{},
               'protocol':protocol, 'protocolSha256':sha(OUT / 'PROTOCOL.json')}
    feature_rows, family_rows = [], []
    phi_stack, mi_stack, rate_stack, sample_stack = [], [], [], []
    for context in contexts:
        answers = classifications != 'correct' if context == 'ALL_NONCONTROL' else classifications == context
        selected = np.repeat(answers, np.diff(off))
        featuredata = payload['featurePrm'][context] = {}
        for j, feature in enumerate(features):
            m = metrics(y[selected], flags[selected, j])
            result = {'value':m['f1_valid'] if context == 'multi_solutions' else m['prm_score'],
                'nAnswers':int(answers.sum()), 'nSteps':int(selected.sum()),
                'metric':'valid-step F1' if context == 'multi_solutions' else 'pooled macro-F1',
                'flagRate':float(flags[selected,j].mean()), 'f1Valid':m['f1_valid'], 'f1Error':m['f1_error']}
            featuredata[feature['id']] = result
            feature_rows.append({'type':context,'feature':feature['id'],'family':feature['family'],
                'in_N16':feature['inN16'], **result, **m})
        for pool in ('all','n16'):
            data = payload['familyPrm'][pool][context] = {}
            for family in gallery['familyNames']:
                members = [f['id'] for f in features if f['family'] == family and (pool == 'all' or f['inN16'])]
                if not members: continue
                points = [featuredata[n]['value'] for n in members]
                result = {'value':float(np.mean(points)), 'members':members,
                    'q25':float(np.quantile(points,.25)), 'q75':float(np.quantile(points,.75)),
                    'nAnswers':int(answers.sum()), 'nSteps':int(selected.sum()),
                    'metric':'mean member valid-step F1' if context == 'multi_solutions' else 'mean member pooled macro-F1'}
                data[family] = result
                family_rows.append({'pool':pool,'type':context,'family':family,
                    'nChannels':len(members), **{k:v for k,v in result.items() if k != 'members'}})
        dep = payload['dependence'][context] = {}
        cp, cm, cr, cn = [], [], [], []
        for truth, label in ((False,'valid'),(True,'error')):
            take = selected & (y == truth)
            phi, mi, rate = pairwise_binary_errors(errors[take])
            dep[label] = {'n':int(take.sum()), 'phi':phi, 'miBits':mi, 'errorRate':rate}
            cp.append(phi); cm.append(mi); cr.append(rate); cn.append(int(take.sum()))
        phi_stack.append(cp); mi_stack.append(cm); rate_stack.append(cr); sample_stack.append(cn)
        print('Evaluated', context, flush=True)
    # Replay the maintained official port for every feature, overall and each subtype.
    checks = []
    for j, name in enumerate(names):
        predictions = [{'idx':str(i), 'labels':(~flags[a:b,j]).astype(int).tolist()}
                       for i,a,b in zip(ids, off[:-1], off[1:])]
        official = prmbench_evaluate(predictions, metas)
        value = .5*(official['total']['f1'] + official['total']['negative_f1'])
        difference = abs(value - payload['featurePrm']['ALL_NONCONTROL'][name]['value'])
        assert difference <= 1e-12
        for context in contexts[1:]:
            table = official['by_classification']
            value = table['f1'][context] if context == 'multi_solutions' else .5*(table['f1'][context]+table['negative_f1'][context])
            delta = abs(value - payload['featurePrm'][context][name]['value'])
            assert delta <= 1e-12
            difference = max(difference, delta)
        checks.append({'feature':name,'max_absolute_difference':difference})
    csvout(OUT / 'FEATURE_PRMSCORE.csv', feature_rows)
    csvout(OUT / 'FAMILY_PRMSCORE.csv', family_rows)
    np.savez_compressed(OUT / 'PAIRWISE_CONDITIONAL.npz', contexts=np.asarray(contexts),
        columns=np.asarray(names), truth_classes=np.asarray(['valid','error']),
        phi=np.asarray(phi_stack), mi_bits=np.asarray(mi_stack),
        error_rate=np.asarray(rate_stack), sample_counts=np.asarray(sample_stack))
    dump(OUT / 'extension.json', payload, compact=True)
    dump(OUT / 'AUDIT.json', {'labels_opened_after_prediction_seal':True,
        'source_disjoint':True, 'shared_N16_columns_max_abs_difference':shared_diff,
        'official_score_parity':checks, 'undefined_constants_and_diagonal':True,
        'frozen_inputs_unchanged':all(sha(ROOT / p)==h for p,h in freeze.items())})
    assert all(sha(ROOT / p)==h for p,h in freeze.items())
    (OUT / 'README.md').write_text('''# Gallery score and dependence extensions

Run with `.venv/bin/python scripts/build_signal_gallery_extensions.py` from the repository root.
The destination refuses reuse to preserve its locked protocol and prediction seal. To reproduce
in a fresh destination, change only the script OUT path in an isolated copy and retain its hash.

`extension.json` is the display payload. `FEATURE_PRMSCORE.csv` and `FAMILY_PRMSCORE.csv`
contain the score summaries. `PAIRWISE_CONDITIONAL.npz` stores empirical pair matrices,
class sample counts and per-feature error rates; columns match the original gallery order.
`PREDICTIONS.npz` retains all fixed source-OOF decisions and donor Q80 thresholds.
The official evaluator was replayed for every feature, overall and for all nine subtypes.

Higher feature values retain the previously frozen label-free error orientation. No evaluated
labels are used to choose signs, thresholds, features, models, or peer sets. The diagnostics
condition on valid/error truth separately, pool steps within each displayed context and do
not adjust for within-answer/source dependence, answer heterogeneity, position, or length.
No confidence interval, independence test, or external validation is claimed. Small pairwise
dependence does not establish joint conditional independence. Constant error margins and the
MS absent erroneous class remain undefined, rather than being represented as zero.

Family lines are equal mean member-feature scores, not the performance of a new fused family.
Overall PRMScore is pooled macro-F1 over noncontrols; eight error-type scores use pooled macro-F1
and MS uses valid-step F1. No AUC is defined for MS. NR/NCL use the declared validity-head
fallback, preserving the previous evaluator adaptation. This is an internal development
diagnostic. No GPU, package installation, Git publication, or frozen-package mutation.
''')
    dump(OUT / 'MANIFEST.json', {'protocol_sha256':sha(OUT / 'PROTOCOL.json'),
        'files':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()},
        'n_features':56, 'n_contexts':10, 'published':False})
    print('Completed', OUT, flush=True)

if __name__ == '__main__': main()
