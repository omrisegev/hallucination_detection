"""Matched CT7 bank-replacement metrics, inference and persisted-model audit."""
import csv
import html
import json
import pickle
import time
from pathlib import Path

import numpy as np

from .core import ARMS, encode, location
from .data import digest, dump
from .readout import earliest_mode
from .scoring import collect as parent_collect, pb_metrics, prm_metrics, grid_prmscore, official
from .uncertainty import bootstrap
from .ct7 import NAMES


def new_method(d):
    return {'scores': np.full(int(d.off[-1]), np.nan), 'pred': np.full(d.n, -999, int),
            'median': np.full(d.n, -999, int), 'fallback': np.zeros(d.n, bool),
            'valid': np.zeros(d.n, bool)}


def collect(d, ps):
    methods = {}; jobs = []
    for path in sorted((d.out/'jobs').glob('*.json')):
        info = json.loads(path.read_text(encoding='utf8'))
        if info['inner_fold'] is not None:
            continue
        jobs.append(info)
        with np.load(path.with_suffix('.npz')) as z:
            idx = z['indices']; offsets = z['step_offsets']
            for key in z.files:
                if not key.endswith('__scores'):
                    continue
                arm = key[:-8]; name = 'fixed__'+info['population']+'__'+arm
                m = methods.setdefault(name, new_method(d))
                assert not m['valid'][idx].any(), name
                saved = z[key]
                for j, i in enumerate(idx):
                    m['scores'][d.off[i]:d.off[i+1]] = saved[offsets[j]:offsets[j+1]]
                m['valid'][idx] = True; m['fallback'][idx] = z[arm+'__fallback']
                if info['task'] != 'prm':
                    m['pred'][idx] = z[arm+'__mode']; m['median'][idx] = z[arm+'__median']
    assert len(jobs) == 50, len(jobs)
    for name, m in methods.items():
        assert np.array_equal(m['valid'], d.pb if '__errors__' in name else np.ones(d.n, bool)), name
        assert np.isfinite(m['scores'][np.repeat(m['valid'], np.diff(d.off))]).all()
    new_out = d.out
    try:
        d.out = d.parent_out
        old, _ = parent_collect(d)
    finally:
        d.out = new_out
    for name, m in old.items():
        if name.startswith(('top5__all__', 'selected__all__')):
            methods['eleven__'+name] = m
        elif name in d.references or name.startswith('control__'):
            methods[name] = m
    for j, name in enumerate(NAMES):
        m = new_method(d); m['scores'] = np.concatenate([p[:, j] for p in ps])
        m['pred'] = d.peaks(m['scores']); m['valid'][:] = True
        methods['ct7_single__'+name] = m
    return methods, jobs


def prmscore(d, methods):
    qgrid = np.linspace(*d.c['inner_threshold_quantiles'])
    old = json.loads((d.parent_out/'PRMSCORE.json').read_text(encoding='utf8'))
    table = {}; selections = {}; saved = {}
    for name in methods:
        oldname = name.removeprefix('eleven__')
        if oldname in old:
            table[name] = old[oldname]
    noncontrol = np.zeros(d.n, bool)
    for i in np.flatnonzero(d.prm):
        noncontrol[i] = d.meta_by_id[d.ids[i]]['classification'] != 'correct'
    step_count = np.diff(d.off)
    for name, m in methods.items():
        if not name.startswith('fixed__all__'):
            continue
        _, _, enc, kind = name.split('__'); arm = enc+'__'+kind
        stage = 'em' if kind in ['ds', 'hem'] else 'spectral'
        fixed = np.zeros(int(d.off[-1]), bool); tuned = fixed.copy(); choices = []
        for outer in range(5):
            train = d.prm & (d.fold != outer); test = d.prm & (d.fold == outer)
            trainsteps = np.flatnonzero(np.repeat(train, step_count))
            teststeps = np.flatnonzero(np.repeat(test, step_count))
            mapping = np.full(int(d.off[-1]), -1, int); mapping[trainsteps] = np.arange(len(trainsteps))
            decisions = np.zeros((len(qgrid), len(trainsteps)), bool); filled = np.zeros(len(trainsteps), bool)
            stem = f'prm__fold{outer}__fixed__all__{stage}'
            with np.load(d.out/'jobs'/f'{stem}.npz') as z:
                thresholds = z[arm+'__thresholds']; tau80 = float(z[arm+'__threshold_q80'])
            for inner in range(5):
                if inner == outer:
                    continue
                with np.load(d.out/'jobs'/f'{stem}__inner{inner}.npz') as z:
                    idx = z['indices']
                    steps = np.concatenate([np.arange(d.off[i], d.off[i+1]) for i in idx])
                    dest = mapping[steps]
                    assert (dest >= 0).all() and not filled[dest].any()
                    decisions[:, dest] = z[arm+'__grid_valid']; filled[dest] = True
            assert filled.all()
            eligible = np.repeat(noncontrol, step_count)[trainsteps]
            merits = grid_prmscore(decisions, d.labels[trainsteps], eligible)
            best = int(np.argmax(merits))
            fixed[teststeps] = m['scores'][teststeps] < tau80
            tuned[teststeps] = m['scores'][teststeps] < thresholds[best]
            choices.append({'fold': outer, 'quantile': float(qgrid[best]),
                            'threshold': float(thresholds[best]), 'q80_threshold': tau80,
                            'inner_prmscores': merits.tolist()})
        idx = np.flatnonzero(d.prm)
        table[name] = {'quantile_0.8': official(d, fixed, idx), 'inner_selected': official(d, tuned, idx)}
        selections[name] = choices; saved[name+'__q80'] = fixed; saved[name+'__inner'] = tuned
    dump(d.out/'PRMSCORE.json', table); dump(d.out/'THRESHOLD_SELECTION.json', selections)
    np.savez_compressed(d.out/'PRM_DECISIONS.npz', **saved)
    return table


def planned(names):
    pairs = []
    def add(a, b, reason):
        if a in names and b in names and a != b:
            pairs.append((a, b, reason))
    for pop in ['all', 'errors']:
        pre = f'fixed__{pop}__'
        for enc, kind in ARMS:
            name = pre+enc+'__'+kind
            for base in ['ct7', 'mindgap']:
                add(name, base, 'new_minus_'+base)
            if kind != 'equal':
                add(name, pre+enc+'__equal', 'learned_minus_equal')
            if pop == 'all':
                for roster in ['top5', 'selected']:
                    add(name, f'eleven__{roster}__all__{enc}__{kind}', 'ct7_bank_minus_eleven_'+roster)
            else:
                add(name, f'fixed__all__{enc}__{kind}', 'errors_minus_all_fit')
        for kind in ['equal', 'spectral', 'continuous_lsml']:
            add(pre+'soft__'+kind, pre+'hard__'+kind, 'soft_minus_hard')
        add(pre+'hard__continuous_lsml', pre+'hard__binary_lsml', 'continuous_core_bridge')
        for kind in ['ds', 'hem']:
            add(pre+'hard__'+kind, pre+'hard__spectral', 'em_minus_initialization')
        add(pre+'hard__hem', pre+'hard__ds', 'hem_minus_ds')
    return pairs


def inference(d, methods, within):
    intervals = bootstrap(d, methods, within)
    with np.load(d.out/'BOOTSTRAP_PRIMARY_DRAWS.npz') as z:
        names = list(z['names']); result = []
        for endpoint in ['pb_sla', 'pb_common_gate_f1', 'prm_within_auc']:
            draws = z[endpoint]; points = intervals['intervals'][endpoint]
            for a, b, reason in planned(points):
                delta = points[a]['point']-points[b]['point']
                boot = draws[:, names.index(a)]-draws[:, names.index(b)]
                assert np.isfinite(boot).all()
                p = (1+np.sum(abs(boot-delta) >= abs(delta)))/(len(boot)+1)
                result.append({'endpoint': endpoint, 'a': a, 'b': b, 'reason': reason,
                               'delta': delta, 'ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
                               'p_bootstrap': float(p)})
    order = np.argsort([r['p_bootstrap'] for r in result], kind='stable'); last = 0.
    for rank, i in enumerate(order):
        last = max(last, min(1., (len(order)-rank)*result[i]['p_bootstrap']))
        result[i]['p_holm'] = last
    out = {'draws': d.c['bootstrap_draws'], 'comparisons': len(result), 'unit': 'source question shared across scorers',
           'holm_family': 'all listed three-endpoint planned contrasts', 'contrasts': result}
    dump(d.out/'CT7_CONTRASTS.json', out)
    return intervals, out


def verify(d, ps):
    started = time.perf_counter(); outer = inner = replay = trajectories = 0
    bad = []; nonconverged = []; models_count = 0; small = []; likelihood_min = 0.
    for path in sorted((d.out/'jobs').glob('*.json')):
        info = json.loads(path.read_text(encoding='utf8'))
        with np.load(path.with_suffix('.npz')) as z:
            arrays = {k: z[k] for k in z.files}
        test = arrays['indices']; offsets = arrays['step_offsets']
        assert not set(info['train_source_groups']) & set(info['test_source_groups'])
        assert not set(info['train_source_groups']) & set(d.groups[d.fold == info['fold']])
        if info['inner_fold'] is None:
            outer += 1; assert np.all(d.fold[test] == info['fold'])
        else:
            inner += 1; assert np.all(d.fold[test] == info['inner_fold'])
            assert np.all(d.fold[test] != info['fold'])
        task = 'prm' if info['task'] == 'prm' else 'pb'
        with open(path.with_suffix('.pkl'), 'rb') as f:
            models = pickle.load(f)
        expected = [(e, k) for e, k in ARMS if (k in ['ds', 'hem']) == (info['stage'] == 'em')]
        assert set(models) == set(expected)
        for (enc, kind), model in models.items():
            arm = enc+'__'+kind; models_count += 1
            assert np.isfinite(arrays[arm+'__scores']).all()
            if model.status != 'ok': bad.append(path.stem+'__'+arm)
            if model.diagnostics.get('converged') is False: nonconverged.append(path.stem+'__'+arm)
            if model.diagnostics.get('small_groups'): small.append(path.stem+'__'+arm)
            for start in model.diagnostics.get('starts', []):
                ll = np.array(start['log_likelihood']); delta = np.diff(ll)
                assert np.all(delta >= -1e-9*np.maximum(1, abs(ll[:-1])))
                likelihood_min = min(likelihood_min, float(delta.min(initial=0)))
                if start['converged']:
                    assert len(delta) >= 3
                    assert np.all(delta[-3:]/np.maximum(1, abs(ll[-4:-1])) < d.c['em_relative_tolerance'])
                trajectories += 1
            if task == 'pb':
                for pos, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
                    v = arrays[arm+'__scores'][a:b]
                    assert (v >= 0).all() and abs(v.sum()-1) < 1e-12
                    assert arrays[arm+'__mode'][pos] == earliest_mode(v)
            for pos in sorted(set([0, len(test)//2, len(test)-1])):
                i = test[pos]; a, b = offsets[pos:pos+2]
                if task == 'pb':
                    score, failed = location(model, ps[i], enc)
                else:
                    x = encode(ps[i], enc, task); score = model.predict(x)
                    failed = model.status != 'ok' or not np.isfinite(score).all()
                    if failed: score = x.mean(1)
                np.testing.assert_allclose(score, arrays[arm+'__scores'][a:b], atol=1e-13, rtol=1e-13)
                assert failed == arrays[arm+'__fallback'][pos]; replay += 1
    assert (outer, inner) == (50, 40), (outer, inner)
    result = {'outer_jobs': outer, 'inner_jobs': inner, 'models': models_count,
              'likelihood_trajectories': trajectories, 'minimum_likelihood_increment': likelihood_min,
              'model_prediction_replays': replay, 'invalid_models': bad, 'nonconverged': nonconverged,
              'small_group_models': small, 'seconds': time.perf_counter()-started}
    dump(d.out/'VALIDATION.json', result)
    print('CT7 artifact verification PASS', {k: v for k, v in result.items() if k != 'small_group_models'}, flush=True)
    return result


def render(d, rows, contrasts, health):
    def table(head, body):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td dir="ltr">'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in body)+'</tbody></table>'
    numeric = lambda v: '—' if v is None else f'{v:.4f}'
    body = [[r['method'], f'{100*r["sla"]:.3f}', f'{100*r["f1"]:.3f}', numeric(r['within_auc']),
             numeric(r['prmscore_inner']), r['fallbacks']] for r in rows]
    main = [r for r in rows if r['method'].startswith('fixed__all__')]
    best = max(main, key=lambda r: r['sla'])
    pair = next(x for x in contrasts['contrasts'] if x['endpoint'] == 'pb_sla' and x['a'] == best['method'] and x['b'] == 'ct7')
    ci = [100*x for x in pair['ci95']]
    summary = (f'הזרוע עם אומדן ה־SLA הגבוה ביותר בין תשע הזרועות הראשיות היא {best["method"]}: '
               f'{100*best["sla"]:.3f}%. ההפרש מול CT7 המקורי הוא {100*pair["delta"]:+.3f} נקודות אחוז; '
               f'רווח סמך מזווג 95% [{ci[0]:+.3f}, {ci[1]:+.3f}], Holm p={pair["p_holm"]:.4f}. '
               'זהו תיאור בדיעבד של הטבלה; אין קידום אוטומטי של מועמד.')
    cards = [['מודלים', health['models']], ['התאמות outer', health['outer_jobs']], ['התאמות inner', health['inner_jobs']],
             ['מסלולי likelihood שנבדקו', health['likelihood_trajectories']], ['EM שנבחר ולא התכנס', len(health['nonconverged'])]]
    parts = ['<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><title>CT7: מיזוג בינארי, רך ו־EM</title>',
             '<style>body{font:17px Arial;max-width:1300px;margin:40px auto;padding:0 24px;line-height:1.65;color:#182334;background:#f7f9fc}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}td,th{padding:9px;border:1px solid #dde3ed}th{background:#e6edf7}td:first-child{text-align:left}p{max-width:1050px}a{color:#174fa0}</style>',
             '<h1>אותו ניסוי על שבעת פרופילי CT7</h1>',
             '<p>כל 6,800 תשובות ProcessBench ו־6,969 תשובות PRMBench. 4,442 תשובות PB שגויות; 6,030 תשובות PRMB זכאיות ל־within-answer AUROC. תוויות v3 וחמישה folds לפי שאלת מקור. נתוני development שכבר נבדקו.</p>',
             '<p>נשמרו readouts המקוריים של CT7: חמשת ערוצי Top10, שארית BOCPD ו־chosen-token z עם נטרול צעד 0. שונו רק קידוד ומיזוג. אין כאן רוסטר Top5 חדש, חיפוש readout או שיבוש טוקנים. המודלים הנלמדים מותאמים על תשובות האימון בלבד; gate של CT7 נשאר קפוא.</p>',
             '<p>'+html.escape(summary)+'</p>',
             '<p>ב־PRMB הקידוד הרך הוא אותו פרופיל מתוקנן של CT7 ולכן soft equal אמור לשחזר את הממוצע המקורי. ב־PB, softmax לכל ערוץ ואז מיצוע התפלגויות הוא שינוי ממשי לעומת מיצוע הציונים לפני argmax.</p>',
             table(['שיטה', 'PB SLA %', 'PB F1 gate משותף %', 'PRMB within AUROC', 'PRMScore inner', 'fallbacks'], body),
             '<p>fixed__all: התאמה על כל תשובות האימון. fixed__errors: PB בלבד, התאמה על תשובות שגויות. eleven: תוצאות הניסוי הקודם, בלי התאמה מחדש. טבלאות מפורטות כוללות כל תא, כל scorer, tolerance-one, הקדמה/איחור, מרחק, median, ו־AUROC/AUPRC בתוך fold.</p>',
             '<h2>בדיקות ואי־ודאות</h2>', table(['בדיקה', 'כמות'], cards),
             f'<p>10,000 דגימות bootstrap משותפות לפי שאלת מקור; {contrasts["comparisons"]} השוואות מתוכננות במשפחת Holm אחת. PRMScore ומדדי האבחון משניים. קבוצות תלויות קטנות ופרמטרים בגבול נשמרים באבחוני המודלים; אין לפרש את אמינות EM כאמת חיצונית.</p>',
             '<p>רכיב BOCPD שוחזר אלגברית ממיזוג היסטורי של ששת הערוצים ואומת מול פרופיל שחושב בנפרד. הפרשי העיגול המדויקים נשמרים ב־PROFILE_VALIDATION.json; אין טענה לשחזור בינארי זהה של הקובץ המקורי. הממוצע משחזר את ציוני CT7 עד 1e−12 ואת כל מיקומי השיא.</p>',
             '<p><a href="SUMMARY.csv">CSV</a> · <a href="PB_METRICS.json">ProcessBench</a> · <a href="PRM_METRICS.json">PRMBench</a> · <a href="PRMSCORE.json">PRMScore</a> · <a href="CT7_CONTRASTS.json">הפרשים מזווגים</a> · <a href="VALIDATION.json">בדיקות</a></p></html>']
    (d.out/'REPORT_HE.html').write_text('\n'.join(parts), encoding='utf8')


def evaluate(d, ps):
    methods, jobs = collect(d, ps)
    health = verify(d, ps)
    pb = {}; prm = {}; within = {}; rows = []
    for name, m in methods.items():
        pb[name] = pb_metrics(d, m)
        if (d.prm & m['valid']).any():
            prm[name], within[name] = prm_metrics(d, m)
    assert abs(prm['fixed__all__soft__equal']['within_auc']-prm['ct7']['within_auc']) < 1e-12
    assert abs(pb['ct7']['macro8']['f1']-.41188745848863717) < 1e-12
    dump(d.out/'PB_METRICS.json', pb); dump(d.out/'PRM_METRICS.json', prm)
    prs = prmscore(d, methods)
    intervals, contrasts = inference(d, methods, within)
    for name, m in methods.items():
        b = pb[name]['macro8']; r = prm.get(name, {}); pr = prs.get(name, {})
        rows.append({'method': name, 'sla': b['sla'], 'f1': b['f1'], 'within_auc': r.get('within_auc'),
                     'prmscore_q80': pr.get('quantile_0.8', {}).get('prmscore'),
                     'prmscore_inner': pr.get('inner_selected', {}).get('prmscore'),
                     'answers': int(m['valid'].sum()), 'fallbacks': int(m['fallback'].sum())})
    with open(d.out/'SUMMARY.csv', 'w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for filename, keys in [('OOF_SCORES.npz', ['scores']), ('OOF_ANSWERS.npz', ['pred', 'median', 'fallback', 'valid'])]:
        values = {name+'__'+key: m[key] for name, m in methods.items() if name.startswith('fixed__') for key in keys}
        if filename == 'OOF_ANSWERS.npz':
            values.update(ids=d.ids, cells=d.cells, groups=d.groups, folds=d.fold, gate=d.gate, target=d.target)
        else:
            values.update(offsets=d.off, labels=d.labels)
        np.savez_compressed(d.out/filename, **values)
    dump(d.out/'COVERAGE.json', {'answers': d.n, 'pb': int(d.pb.sum()), 'prm': int(d.prm.sum()),
         'prm_eligible': prm['ct7']['eligible'], 'outer_seconds': sum(j['seconds'] for j in jobs), 'methods': rows})
    render(d, rows, contrasts, health)
    artifacts = {str(p.relative_to(d.out)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                 for p in d.out.rglob('*') if p.is_file() and p.name != 'REPORT_MANIFEST.json'}
    dump(d.out/'REPORT_MANIFEST.json', {'candidate_id': d.settings['candidate_id'], 'artifacts': artifacts})
    print('CT7 COMPARISON COMPLETE', json.dumps([r for r in rows if r['method'] == 'ct7' or r['method'].startswith('fixed__all__')]), flush=True)
