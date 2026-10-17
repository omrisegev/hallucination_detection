#!/usr/bin/env python
"""One user-selected, post-hoc F1/F2/F7 PRMBench comparison; CPU only.

Fit never reads labels. No group-count forcing, sign repair or failed-fit fallback.
The output path refuses reuse; individual checkpoints preserve failed folds.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import ast, csv, hashlib, html, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.advisor_lr import fit_labelfree, as_dict, AffineFusion
from spectral_utils.advisor_controls import (answer_z, metrics, confusion_by_group,
    score_from_confusion, bootstrap_answer_mean)
from spectral_utils.advisor_statistics import answer_auc
from spectral_utils.prmbench import prmbench_evaluate

FROZEN = ROOT / 'results/advisor_completion_2026-10-11'
OUT = ROOT / 'results/family_127_prmbench_v1_2026-10-12'
BANKS = ('N56', 'N16')
METHODS = ('LSML_CONT', 'IU_PCR')
ARMS = ('FULL', 'F127')
CATEGORIES = {'simplicity': ['redundency', 'circular'],
    'soundness': ['counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence'],
    'sensitivity': ['missing_condition', 'deception', 'multi_solutions']}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def clean(v):
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    return v

def dump(path, value):
    p = Path(path); tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n'); tmp.replace(p)

def csvout(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(clean(rows))

def families():
    tree = ast.parse((ROOT / 'scripts/run_advisor_v11_lr.py').read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'FAMILIES' for t in n.targets))

def cut(off, answers):
    return (np.concatenate([np.arange(off[i], off[i+1]) for i in answers]),
            np.r_[0, np.cumsum(np.diff(off)[answers])])

def fit_all():
    OUT.mkdir(parents=True, exist_ok=False)
    for n in ('predictions', 'checkpoints', 'evaluation'): (OUT / n).mkdir()
    family = families(); allowed = set(sum([family[f] for f in ('F1', 'F2', 'F7')], []))
    protocol = {'version': 1, 'status': 'LOCKED_BEFORE_NEW_FITS',
        'origin': 'user chose F7/F2/F1 after inspecting gallery E; exploratory post-hoc development diagnostic',
        'primary_bank': 'N56 -> 27 raw channels', 'sensitivity_bank': 'N16 -> 11 raw channels',
        'methods': METHODS, 'arms': ARMS, 'selection': 'fixed family taxonomy F1/F2/F7; flat fusion of individual member channels',
        'not_three_views': 'family means are not supplied as three inputs; LSML automatically derives clusters',
        'failure_policy': 'LSML automatic K<3 or any invalid fit: failed fold, no fallback or forced regrouping',
        'inputs': 'corrected thinkfix mask1 signed answer-z V; inherited label-free signs; higher score = error',
        'split': 'unchanged five FOLDS_V2 source-disjoint outer folds',
        'normalization': 'LSML original answer-z channels; IU training-only mean/population SD',
        'decision': 'ANSWER_Z_Q80: donor global 0.8 quantile after per-answer z of fused score; holdout z >= threshold',
        'primary_endpoints': ['mean within-answer AUC', 'official evaluator Overall PRMScore'],
        'official_score': '0.5*(total.f1+total.negative_f1); correct controls excluded; NR/NCL validity fallback',
        'type_display': 'MS valid-step F1 only; other error types macro-F1; categories equal subtype means',
        'uncertainty': '5000 shared bootstrap draws of 707 source groups; paired subset minus full; pointwise 95% descriptive',
        'multiplicity': 'no promotion test; no variant search or external confirmation',
        'seed': 20261012, 'families': {f: family[f] for f in ('F1', 'F2', 'F7')}}
    dump(OUT / 'PROTOCOL.json', protocol)
    code = ['scripts/run_family_127_prmbench_v1.py', 'scripts/run_advisor_v11_lr.py',
        'spectral_utils/advisor_lr.py', 'spectral_utils/fusion_utils.py', 'spectral_utils/upcr.py',
        'spectral_utils/laplacian_upcr.py', 'spectral_utils/advisor_controls.py',
        'spectral_utils/advisor_statistics.py', 'spectral_utils/prmbench.py']
    inputs = [FROZEN / 'inputs' / n for n in ('population.npz', 'bank_V_N56.npz', 'bank_V_N16.npz',
        'FOLDS_V2.json', 'prmbench_meta.json')]
    identity = {'code': {p: sha(ROOT / p) for p in code},
        'inputs': {str(p.relative_to(ROOT)): sha(p) for p in inputs},
        'frozen_manifest': sha(FROZEN / 'PACKAGE_MANIFEST.json')}
    dump(OUT / 'INPUT_FREEZE.json', identity)
    with np.load(FROZEN / 'inputs/population.npz', allow_pickle=False) as z:
        pop = {k: z[k] for k in ('idx', 'offsets', 'group', 'fold')}
    off = pop['offsets']; folds = json.loads((FROZEN / 'inputs/FOLDS_V2.json').read_text())['prmbench']['outer']
    assert all(int(folds[str(g)]) == int(f) for g, f in zip(pop['group'], pop['fold']))
    assert len(set(pop['idx'])) == len(pop['idx']) == 6969 and off[-1] == 94203
    for bank in BANKS:
        with np.load(FROZEN / f'inputs/bank_V_{bank}.npz', allow_pickle=False) as z:
            X, names = z['V'], z['columns'].astype(str).tolist()
            assert np.array_equal(off, z['offsets'])
            if 'idx' in z.files: assert np.array_equal(pop['idx'], z['idx'])
        columns = np.array([j for j, n in enumerate(names) if n in allowed])
        assert len(columns) == (27 if bank == 'N56' else 11)
        pred = dict(pop)
        for arm in ARMS:
            for method in METHODS:
                key = arm + '__' + method; pred[key] = np.full(len(X), np.nan)
                pred[key + '__flags'] = np.zeros(len(X), bool)
        for f in range(5):
            began = time.time(); tr_ans = np.flatnonzero(pop['fold'] != f); te_ans = np.flatnonzero(pop['fold'] == f)
            train_groups = set(pop['group'][tr_ans]); test_groups = set(pop['group'][te_ans]); assert not train_groups & test_groups
            tr, toff = cut(off, tr_ans); te, eoff = cut(off, te_ans)
            cp = OUT / 'checkpoints' / f'{bank}_fold{f}'; cp.mkdir()
            diag = {'bank': bank, 'fold': f, 'all_columns': names, 'subset_columns': [names[j] for j in columns],
                'train_groups': sorted(train_groups), 'test_groups': sorted(test_groups), 'models': {}, 'failures': {}}
            print('FIT', bank, f, flush=True)
            old = np.load(FROZEN / f'lr/checkpoints/{bank}/outer_{f}/SCORES.npz', allow_pickle=False)
            assert np.array_equal(old['rows'], te)
            for arm in ARMS:
                cols = np.arange(len(names)) if arm == 'FULL' else columns
                for method in METHODS:
                    key = arm + '__' + method
                    try:
                        model = fit_labelfree(X[tr][:, cols], method)
                    except (ValueError, RuntimeError) as exc:
                        diag['failures'][key] = repr(exc); dump(cp / 'MODELS.json', diag)
                        print('FAILED', bank, f, key, repr(exc), flush=True); continue
                    train_score = model.predict(X[tr][:, cols]); test_score = model.predict(X[te][:, cols])
                    assert np.isfinite(train_score).all() and np.isfinite(test_score).all()
                    tau = float(np.quantile(answer_z(train_score, toff), .8))
                    flags = answer_z(test_score, eoff) >= tau
                    pred[key][te] = test_score; pred[key + '__flags'][te] = flags
                    diag['models'][key] = as_dict(model)
                    diag['models'][key].update({'columns': cols.tolist(), 'Q80_tau': tau})
                    if arm == 'FULL':
                        error = float(np.max(np.abs(test_score - old['LSML_FULL' if method == 'LSML_CONT' else 'IU_FULL'])))
                        diag['models'][key]['baseline_max_abs'] = error
                        assert error <= 1e-9, (key, error)
                    dump(cp / 'MODELS.json', diag)
            np.savez_compressed(cp / 'SCORES.npz', rows=te, **{k: v[te] for k, v in pred.items() if '__' in k})
            dump(cp / 'DONE.json', {'elapsed_seconds': time.time() - began, 'failures': diag['failures']})
            print('DONE', bank, f, round(time.time() - began, 2), flush=True)
        np.savez_compressed(OUT / 'predictions' / f'{bank}.npz', **pred)
    seal = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT / 'PREDICTION_SEAL.json', {'status': 'FROZEN_BEFORE_LABEL_EVALUATION', 'files': seal})
    assert all(sha(ROOT / p) == h for p, h in identity['inputs'].items())
    assert sha(FROZEN / 'PACKAGE_MANIFEST.json') == identity['frozen_manifest']

def official(y, cls, ids, off, flags, covered, meta):
    predictions = [{'idx': str(ids[i]), 'labels': (~flags[a:b]).astype(int).tolist()}
        for i, (a, b) in enumerate(zip(off[:-1], off[1:])) if covered[i]]
    metadata = [meta[str(ids[i])] for i in np.flatnonzero(covered)]
    out = prmbench_evaluate(predictions, metadata)
    value = .5 * (out['total']['f1'] + out['total']['negative_f1'])
    sm = np.repeat(covered & (cls != 'correct'), np.diff(off))
    assert abs(value - metrics(y[sm], flags[sm])['prm_score']) < 1e-12
    return value, out

def evaluate():
    seal = json.loads((OUT / 'PREDICTION_SEAL.json').read_text())
    assert all(sha(OUT / p) == h for p, h in seal['files'].items())
    with np.load(FROZEN / 'inputs/population.npz', allow_pickle=False) as z:
        y, cls = z['y'].astype(bool), z['classification'].astype(str)
    meta = {str(r['idx']): r for r in json.loads((FROZEN / 'inputs/prmbench_meta.json').read_text())['rows']}
    rows, types, categories, contrasts, telemetry = [], [], [], [], []
    audit = {'baseline_replays': [], 'affine_replays': [], 'official_equivalence': [],
        'labels_opened_after_prediction_seal': True, 'source_disjoint': True}
    for bank in BANKS:
        p = np.load(OUT / 'predictions' / f'{bank}.npz', allow_pickle=False)
        off, groups, ids = p['offsets'], p['group'], p['idx']
        unique, inv = np.unique(groups, return_inverse=True); sg = np.repeat(inv, np.diff(off))
        assert len(unique) == 707
        W = np.random.default_rng(20261012).multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=5000).astype(float)
        np.savez_compressed(OUT / 'evaluation' / f'{bank}_bootstrap.npz', groups=unique, multiplicity=W)
        aucs, cover = {}, {}; per_answer = dict(idx=ids, group=groups, offsets=off)
        for arm in ARMS:
            for method in METHODS:
                key = arm + '__' + method; score = p[key]; flag = p[key + '__flags']
                covered = np.array([np.isfinite(score[a:b]).all() for a, b in zip(off[:-1], off[1:])]); cover[key] = covered
                auc, _ = answer_auc(y, np.where(np.isfinite(score), score, 0.), off); auc[~covered] = np.nan
                aucs[key] = auc; per_answer[key + '__auc'] = auc; per_answer[key + '__covered'] = covered
                counts = np.add.reduceat(flag.astype(int), off[:-1]); per_answer[key + '__flag_count'] = counts
                mean_draw = bootstrap_answer_mean(auc, inv, covered, W)
                value, off_eval = official(y, cls, ids, off, flag, covered, meta)
                dump(OUT / 'evaluation' / f'{bank}_{key}_OFFICIAL.json', off_eval)
                audit['official_equivalence'].append({'bank': bank, 'arm': key, 'absolute_difference': 0.0})
                selected = np.repeat(covered & (cls != 'correct'), np.diff(off))
                conf = confusion_by_group(y, flag, sg, selected, len(unique)); prm_draw = score_from_confusion(conf, W)
                row = {'bank': bank, 'arm': arm, 'method': method, 'n_channels': len(families()['F1']) + len(families()['F2']) + len(families()['F7']) if arm == 'F127' and bank == 'N56' else 11 if arm == 'F127' else int(bank[1:]),
                    'complete': bool(covered.all()), 'n_answers_scored': int(covered.sum()), 'n_auc_answers': int(np.isfinite(auc).sum()),
                    'n_prmscore_answers': int((covered & (cls != 'correct')).sum()), 'n_prmscore_steps': int(selected.sum()),
                    'within_answer_auc': float(np.nanmean(auc)), 'auc_lo95': np.quantile(mean_draw, .025), 'auc_hi95': np.quantile(mean_draw, .975),
                    'official_prmscore': value, 'prmscore_lo95': np.quantile(prm_draw, .025), 'prmscore_hi95': np.quantile(prm_draw, .975),
                    'noncontrol_pooled_macro_f1': metrics(y[selected], flag[selected])['prm_score'],
                    'noncontrol_step_flag_rate': float(flag[selected].mean()),
                    'correct_control_answer_false_flag_rate': float((counts[covered & (cls == 'correct')] > 0).mean())}
                rows.append(row); subtype = {}
                for typ in np.unique(cls):
                    take = covered & (cls == typ); valid = take & np.isfinite(auc); sm = np.repeat(take, np.diff(off))
                    m = metrics(y[sm], flag[sm]); val = m['f1_valid'] if typ == 'multi_solutions' else m['prm_score'] if typ != 'correct' else None
                    subtype[typ] = val
                    types.append({'bank': bank, 'arm': arm, 'method': method, 'type': typ,
                        'n_answers': int(take.sum()), 'n_steps': int(sm.sum()), 'n_auc_answers': int(valid.sum()),
                        'auc': float(np.mean(auc[valid])) if valid.any() else None, 'prmscore_display': val,
                        'display_definition': 'valid-step F1' if typ == 'multi_solutions' else 'clean control, no PRMScore' if typ == 'correct' else 'macro-F1',
                        'flag_rate': float(flag[sm].mean()) if sm.any() else None, 'answer_flag_rate': float((counts[take] > 0).mean()) if take.any() else None, **m})
                for cat, members in CATEGORIES.items():
                    categories.append({'bank': bank, 'arm': arm, 'method': method, 'category': cat,
                        'score': float(np.mean([subtype[t] for t in members])), 'members': members,
                        'definition': 'equal mean subtype display scores; MS=valid-step F1'})
        for method in METHODS:
            sub, full = 'F127__' + method, 'FULL__' + method
            common = cover[sub] & cover[full]; delta = aucs[sub] - aucs[full]
            d = bootstrap_answer_mean(delta, inv, common, W)
            sm = np.repeat(common & (cls != 'correct'), np.diff(off))
            flags = [p[k + '__flags'] for k in (sub, full)]
            confs = [confusion_by_group(y, f, sg, sm, len(unique)) for f in flags]
            pd = score_from_confusion(confs[0], W) - score_from_confusion(confs[1], W)
            contrasts.append({'bank': bank, 'method': method, 'definition': 'F127 minus FULL, matched eligible source groups',
                'n_common_scored_answers': int(common.sum()), 'n_common_auc_answers': int((common & np.isfinite(delta)).sum()),
                'delta_auc': float(np.nanmean(delta[common])), 'auc_delta_lo95': np.quantile(d, .025), 'auc_delta_hi95': np.quantile(d, .975),
                'delta_prmscore': metrics(y[sm], flags[0][sm])['prm_score'] - metrics(y[sm], flags[1][sm])['prm_score'],
                'prmscore_delta_lo95': np.quantile(pd, .025), 'prmscore_delta_hi95': np.quantile(pd, .975),
                'fold_delta_auc': {str(f): float(np.nanmean(delta[p['fold'] == f])) for f in range(5)}})
        np.savez_compressed(OUT / 'evaluation' / f'{bank}_per_answer.npz', **per_answer)
        with np.load(FROZEN / f'inputs/bank_V_{bank}.npz', allow_pickle=False) as z: X = z['V']
        for f in range(5):
            cp = OUT / 'checkpoints' / f'{bank}_fold{f}'
            diag = json.loads((cp / 'MODELS.json').read_text()); saved = np.load(cp / 'SCORES.npz', allow_pickle=False)
            for key, model in diag['models'].items():
                cols = np.array(model['columns']); replay = X[saved['rows']][:, cols] @ model['weight'] + model['intercept']
                err = float(np.max(np.abs(replay - saved[key]))); assert err <= 1e-9
                audit['affine_replays'].append({'bank': bank, 'fold': f, 'arm': key, 'max_abs': err})
                if key.startswith('FULL'): audit['baseline_replays'].append({'bank': bank, 'fold': f, 'arm': key, 'max_abs': model['baseline_max_abs']})
                telemetry.append({'bank': bank, 'fold': f, 'arm': key, 'K': model['diagnostic'].get('K'),
                    'degenerate': model['diagnostic'].get('degenerate'), 'abstained': model['diagnostic'].get('abstained'),
                    'affine_max_abs': model['diagnostic']['affine_max_abs']})
    dump(OUT / 'RESULTS.json', {'metrics': rows, 'contrasts': contrasts, 'types': types, 'categories': categories, 'telemetry': telemetry})
    dump(OUT / 'AUDIT.json', audit)
    for name, values in [('SUMMARY', rows), ('CONTRASTS', contrasts), ('TYPES', types), ('CATEGORIES', categories), ('TELEMETRY', telemetry)]: csvout(OUT / (name + '.csv'), values)
    report(rows, contrasts, types, categories, telemetry)
    dump(OUT / 'PACKAGE_MANIFEST.json', {'files': {str(p.relative_to(OUT)): {'sha256': sha(p), 'bytes': p.stat().st_size, 'git_eligible': p.suffix != '.npz'} for p in OUT.rglob('*') if p.is_file()}})
    print(json.dumps(clean({'metrics': rows, 'contrasts': contrasts}), indent=2), flush=True)

def report(rows, contrasts, types, categories, telemetry):
    def table(data, cols):
        def fmt(v): return f'{v:.6f}' if isinstance(v, (float, np.floating)) else html.escape(str(v))
        return '<table><tr>' + ''.join('<th>' + html.escape(c) + '</th>' for c in cols) + '</tr>' + ''.join('<tr>' + ''.join('<td>' + fmt(r.get(c)) + '</td>' for c in cols) + '</tr>' for r in data) + '</table>'
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>F1 + F2 + F7 PRMBench</title><style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 20px;color:#223044}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef2f6}.scroll{overflow:auto}details{margin:25px 0}p{line-height:1.55}</style><h1>F1 + F2 + F7 only: continuous L-SML and IU-PCR</h1>'
    page += '<p>User-selected after inspecting family visualization E: exploratory, post-hoc internal development diagnostic. Corrected thinkfix mask1 PRMBench; 6,969 answers / 94,203 steps / 707 source groups. N56 → 27 channels is primary; N16 → 11 is sensitivity. Every member channel enters separately. Three families do not mean three inputs or three forced L-SML clusters.</p>'
    page += '<p>Both methods fit donor sources only, with frozen inherited signs. Predictions were sealed before opening labels. Full-bank scores reproduce saved advisor v11 to 1e-9. ANSWER_Z_Q80 is fitted on donor answer-z fused scores, without label calibration. No forced clustering or mean fallback: automatic L-SML K&lt;3 fails explicitly.</p>'
    page += '<div class="scroll">' + table(rows, ['bank', 'arm', 'method', 'n_channels', 'complete', 'n_auc_answers', 'within_answer_auc', 'official_prmscore', 'correct_control_answer_false_flag_rate']) + '</div>'
    page += '<h2>Subset minus full bank</h2><p>5,000 shared source-group bootstrap draws; pointwise descriptive 95% intervals. These data guided the family choice, so intervals are not independent confirmation or a promotion test.</p><div class="scroll">' + table(contrasts, ['bank', 'method', 'delta_auc', 'auc_delta_lo95', 'auc_delta_hi95', 'delta_prmscore', 'prmscore_delta_lo95', 'prmscore_delta_hi95']) + '</div>'
    page += '<p>Overall PRMScore = half the official evaluator total valid/error F1, excluding synthetic correct controls; independently checked against pooled noncontrol macro-F1. NR/NCL use the documented validity fallback (no separate redundancy head). Within-answer AUC averages answers containing both classes; MS has no AUC. Per-type display uses valid F1 for MS; Sensitivity averages PS/DR/MS. Categories are not averaged to obtain Overall.</p>'
    page += '<details><summary>Type results and eligible sample sizes</summary><div class="scroll">' + table(types, ['bank', 'arm', 'method', 'type', 'n_answers', 'n_auc_answers', 'auc', 'prmscore_display', 'display_definition', 'flag_rate']) + '</div></details>'
    page += '<details><summary>Categories</summary>' + table(categories, ['bank', 'arm', 'method', 'category', 'score']) + '</details>'
    page += '<details><summary>Fit telemetry</summary>' + table(telemetry, ['bank', 'fold', 'arm', 'K', 'degenerate', 'abstained', 'affine_max_abs']) + '</details>'
    page += '<p>Evidence: <a href="PROTOCOL.json">Protocol</a> · <a href="INPUT_FREEZE.json">Source hashes</a> · <a href="RESULTS.json">Results</a> · <a href="SUMMARY.csv">CSV</a> · <a href="AUDIT.json">Replay audit</a> · <a href="PREDICTION_SEAL.json">Prediction seal</a>. Raw scores/flags, per-answer AUCs, models, source splits and bootstrap draws are retained locally; NPZ arrays are excluded from Git.</p></html>'
    (OUT / 'index.html').write_text(page)
    (OUT / 'README.md').write_text('# F1/F2/F7 PRMBench diagnostic\n\nCPU-only, user-selected after visualization E; exploratory on previously inspected development data. No external confirmation or promoted feature contract.\n\nReproduction from repository root: `.venv/bin/python scripts/run_family_127_prmbench_v1.py`. Existing destination is refused, preserving evidence. Models and predictions precede evaluation; NPZ raw arrays remain local and are catalogued in PACKAGE_MANIFEST.json.\n\nStart with index.html, RESULTS.json, AUDIT.json, PROTOCOL.json. Official evaluator outputs are in evaluation/*OFFICIAL.json; checkpoints/*/MODELS.json includes automatic clustering, effective weights, donor thresholds, failed fits and baseline replay errors.\n')

if __name__ == '__main__':
    fit_all()
    evaluate()
