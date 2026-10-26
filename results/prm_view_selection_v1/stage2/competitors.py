"""The outstanding subsets of stages 1-2 under the competitors' own protocols (CROP, ARES, PRISM, SLIDER).

Each subset enters twice: "+PRM" = the stage-1 fused score (PRM-anchored binary L-SML, PRM included) and "alone" = the
stage-2 fused score (telemetry only, no PRM). References: the PRM (same rule), CHAIN_N16, TELE_N16. The protocols are
those of results/prm_boost_collection_v1/collection_competitors.py (re-used function by function); parity: CHAIN_N16 and
the PRM must reproduce the existing COMP_* tables exactly (COMPETITORS_PARITY.json).

CAVEAT: the subsets were selected with PRMBench (and Socratic) labels, so every PRMBench number here is in-sample.

    python competitors.py      # -> COMPETITORS.csv, COMPETITORS_PARITY.json
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '4')
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'stage1'))
import stage2 as S2  # noqa: E402
import stage1 as S1  # noqa: E402
import analyze as AN  # noqa: E402
E = S1.E
sys.path.insert(0, str(E.HERE))
import collection_competitors as CC  # noqa: E402

PICK = ['S1 best (stage-1 goal, all 10)', 'S1 best 8 cells', 'S2 best (primary, all 10)', 'JOINT (max of min over 20 margins)',
        'JOINT 8 cells (no Math-Shepherd)']


def subsets():
    rows = {r['selection']: r['subset'] for r in E.read_csv(HERE / 'SUBSETS_NAMED.csv')}
    out = []
    for k in PICK:
        s = rows[k]
        if s not in [x[1] for x in out]: out.append((k, s))
    return out


def add_arms(bench, prm, subs):
    """Evaluator cell with the subset arms added (stage-1 and stage-2 fused scores, refitted on this cell)."""
    s1 = S1.CellData(bench, prm); c = s1.c; t = S2.TeleCell().build(bench, prm); names = []
    for i, (_, s) in enumerate(subs):
        S = [AN.NAMES.index(n) for n in s.split(' + ')]
        for tag, cell in (('+PRM', s1), ('alone', t)):
            o = cell.evaluate(S, want_detail=True)
            nm = f'SUB{i + 1} {tag}'
            if o['valid']: c.arms[nm] = o['fused']; names.append(nm)
    del t
    return c, names


def prism_rows(c, arms):
    out = []
    for arm in arms:
        rule = 'GAP'; f = c.flags(arm, rule); r = E.rates(CC._counts(c, f, c.keep))
        # the PRM's own score on its scale, as in COMP_PRISM_MATCHED_TPR (PRM_Q80_PROB raw; else PRM_native raw)
        v = c.arms[arm] if arm == 'PRM_Q80_PROB' else c.arms['PRM_native'] if arm == 'PRM_Q80' else CC.thr_score(c, arm)
        ts = np.quantile(v[c.keep], np.linspace(0.6, 0.95, 701))
        best = min((abs(E.rates(CC._counts(c, v >= t, c.keep))['TPR'] - 87.30), t) for t in ts)[1]
        fm = v >= best; rm = E.rates(CC._counts(c, fm, c.keep))
        out.append(dict(protocol='PRISM', prm=c.prm, arm=arm, PRMScore_GAP=E.official(c, f)[0], TPR_GAP=r['TPR'], FPR_GAP=r['FPR'],
                        matchedTPR_FPR=rm['FPR'], matchedTPR_PRMScore=E.official(c, fm)[0]))
    return out


def crop_rows(c, arms, s_c=None):
    sub, J, sl, labels, has_err = CC.crop_subset(c)
    per = lambda v: [v[s] for s in sl]; risk1 = 1 - c.extra['reward']; fixed = np.linspace(0, 1, 101)
    res = {a: dict(kept=[], risk=[], auroc=[]) for a in arms}
    for seed in CC.SEEDS:
        tr, ca, te = CC.split_traces(has_err, seed)
        rows_tr = np.concatenate([np.arange(sl[k].start, sl[k].stop) for k in tr]); yte = np.concatenate([labels[k] for k in te])
        for a in arms:
            if a == 'PRM_native':
                sc, grid = per(risk1), fixed                                        # 'Direct PRM (1-reward)', fixed 101 grid
            else:
                sc, grid = per(c.arms[a]), np.quantile(c.arms[a][rows_tr], np.linspace(0, 1, 201))
            res[a]['auroc'].append(roc_auc_score(yte, np.concatenate([sc[k] for k in te])))
            lam, _ = CC.crop_lambda(sc, labels, ca, grid, .05); M, Tn, cont = CC.crop_eval(sc, labels, te, lam)
            res[a]['kept'].append(100 * float(np.mean(M / np.maximum(Tn, 1)))); res[a]['risk'].append(100 * float(cont.mean()))
    out = [dict(protocol='CROP', prm=c.prm, arm=a, kept_alpha05=float(np.mean(d['kept'])), kept_sd=float(np.std(d['kept'], ddof=1)),
                risk_alpha05=float(np.mean(d['risk'])), auroc=float(np.mean(d['auroc']))) for a, d in res.items()]
    if s_c is not None:                                                             # cross-domain: lambda on PRMBench, test on Socratic
        s_lab = [s_c.err[a:b] & s_c.keep[a:b] for a, b in zip(s_c.off[:-1], s_c.off[1:])]
        for r in out:
            a = r['arm']; sa = {'PRM_native': 'PRM'}.get(a, a)
            if sa not in s_c.arms: continue
            pb = c.arms[a]; grid = np.quantile(np.concatenate([pb[s] for s in sl]), np.linspace(0, 1, 201))
            lam, _ = CC.crop_lambda(per(pb), labels, np.arange(len(sub)), grid, .05)
            sc = [s_c.arms[sa][x:y] for x, y in zip(s_c.off[:-1], s_c.off[1:])]
            M, Tn, cont = CC.crop_eval(sc, s_lab, np.arange(len(sc)), lam)
            r.update(socratic_kept=100 * float(np.mean(M / np.maximum(Tn, 1))), socratic_risk=100 * float(cont.mean()))
    return out


def ares_rows(c, arms):
    ans_ok = c.cls != 'correct'; groups = c.groups
    ug = np.unique(groups[ans_ok]); rng = np.random.default_rng(20261021); fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % 5))
    step_fold = np.repeat([fold_of.get(g, -1) for g in groups], np.diff(c.off)); out = []
    for arm in arms:
        v = CC.thr_score(c, arm); oof = np.zeros(len(v), bool); fs = []
        for k in range(5):
            trm = c.keep & (step_fold != k); tem = c.keep & (step_fold == k)
            ts = np.quantile(v[trm], np.linspace(0.01, 0.99, 99))
            best = max(ts, key=lambda t: E.official_from_counts(CC._counts(c, v >= t, trm), 'prmbench'))
            oof[tem] = v[tem] >= best; fs.append(float(E.official_from_counts(CC._counts(c, v >= best, tem), 'prmbench')))
        out.append(dict(protocol='ARES', prm=c.prm, arm=arm, macro_f1_oof=float(E.official_from_counts(CC._counts(c, oof, c.keep), 'prmbench')),
                        macro_f1_fold_sd=float(np.std(fs, ddof=1))))
    return out


def slider_rows(c, arms):
    R = np.flatnonzero(c.cls == 'redundency'); sl = [slice(c.off[j], c.off[j + 1]) for j in R]; out = []
    keep = [np.arange(x.start, x.stop) for x in sl]
    for arm in arms:
        s = c.arms[arm]; acc, bal = [], []
        for sd in range(10):
            perm = np.random.default_rng(sd).permutation(len(keep)); h = len(keep) // 2
            xh = np.concatenate([s[keep[k]] for k in perm[:h]])[:, None]; yh = np.concatenate([c.err[keep[k]] for k in perm[:h]])
            xt = np.concatenate([s[keep[k]] for k in perm[h:]])[:, None]; yt = np.concatenate([c.err[keep[k]] for k in perm[h:]])
            p = LogisticRegression(class_weight='balanced').fit(xh, yh).predict(xt).astype(bool)
            acc.append(100 * (p == yt).mean()); bal.append(50 * (p[yt].mean() + (~p[~yt]).mean()))
        out.append(dict(protocol='SLIDER', prm=c.prm, arm=arm, acc_balanced_threshold=float(np.mean(acc)), bal_acc=float(np.mean(bal))))
    return out


def main():
    subs = subsets(); rows, parity = [], {}
    print('subsets:', *[f'SUB{i + 1} = {k}: {s}' for i, (k, s) in enumerate(subs)], sep='\n  ')
    for prm in S1.PRMS:
        c, names = add_arms('prmbench', prm, subs)
        s_c, s_names = add_arms('socratic', prm, subs)
        scale = CC.best_scale(c)
        rows += prism_rows(c, [scale, 'CHAIN_N16', 'TELE_N16'] + names)
        rows += crop_rows(c, ['PRM_native', 'CHAIN_N16', 'TELE_N16'] + names, s_c)
        rows += ares_rows(c, ['PRM_native', 'CHAIN_N16', 'TELE_N16'] + names)
        rows += slider_rows(c, [scale, 'CHAIN_N16', 'TELE_N16'] + names)
        print(prm, 'done', flush=True); del c, s_c
    for r in rows:
        r['arm_label'] = r['arm']
        for i, (k, s) in enumerate(subs):
            r['arm_label'] = r['arm_label'].replace(f'SUB{i + 1} ', f'{k} :: ')
    E.write_csv(HERE / 'COMPETITORS.csv', rows)
    # parity with the existing competitor tables (CHAIN_N16 and the PRM)
    T = E.TABLES; get = lambda name: E.read_csv(T / name)
    P = [r for r in rows if r['arm'] in ('CHAIN_N16', 'TELE_N16', 'PRM_Q80', 'PRM_Q80_PROB', 'PRM_native')]
    mt = {(r['prm'], r['arm']): r for r in get('COMP_PRISM_MATCHED_TPR.csv')}
    t1 = {(r['prm'], r['arm']): r for r in get('COMP_PRISM_T1_T2.csv') if r['rule'] == 'GAP' and r['pool'].startswith('non')}
    t6 = {(r['prm'], r['proxy']): r for r in get('COMP_CROP_T6.csv')}
    ta = {(r['prm'], r['arm']): r for r in get('COMP_ARES.csv') if r['prm'] != 'paper'}
    ts = {(r['prm'], r['arm']): r for r in get('COMP_SLIDER.csv') if r['prm'] != 'paper' and r.get('rule') == 'GAP'}
    diffs = []
    for r in P:
        k = (r['prm'], r['arm'])
        if r['protocol'] == 'PRISM' and k in t1: diffs.append(abs(r['PRMScore_GAP'] - float(t1[k]['PRMScore'])))
        km = (r['prm'], {'PRM_Q80': 'PRM_native'}.get(r['arm'], r['arm']))
        if r['protocol'] == 'PRISM' and km in mt: diffs.append(abs(r['matchedTPR_FPR'] - float(mt[km]['FPR'])))
        if r['protocol'] == 'CROP' and r['arm'] == 'PRM_native' and (r['prm'], 'Direct PRM (1-reward)') in t6:
            diffs.append(abs(r['kept_alpha05'] - float(t6[(r['prm'], 'Direct PRM (1-reward)')]['kept_mean'])))
        if r['protocol'] == 'CROP' and k in t6: diffs.append(abs(r['kept_alpha05'] - float(t6[k]['kept_mean'])))
        if r['protocol'] == 'ARES' and k in ta: diffs.append(abs(r['macro_f1_oof'] - float(ta[k]['macro_f1_oof'])))
        if r['protocol'] == 'SLIDER' and k in ts: diffs.append(abs(r['acc_balanced_threshold'] - float(ts[k]['all_steps__LR_balanced__acc'])))
    parity = dict(n_checked=len(diffs), max_abs_diff=float(max(diffs)) if diffs else None)
    json.dump(dict(parity=parity, subsets=[dict(id=f'SUB{i + 1}', selection=k, subset=s) for i, (k, s) in enumerate(subs)]),
              open(HERE / 'COMPETITORS_PARITY.json', 'w'), indent=1)
    print('parity vs existing COMP tables:', parity)


if __name__ == '__main__':
    main()
