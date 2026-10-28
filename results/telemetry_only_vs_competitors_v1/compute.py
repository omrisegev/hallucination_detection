"""Telemetry only (standalone) vs the paired PRM, on every metric and benchmark that PRISM, SLIDER and CROP report.

Pairs (standalone base model -> the PRM built on it):
  Qwen2.5-Math-7B-Instruct -> Qwen-PRM-7B, Qwen2.5-Math-72B-Instruct -> Qwen-PRM-72B,
  Qwen2.5-Math-7B -> PURE-PRM-7B, Llama-3.1-8B-Instruct -> RLHFlow-PRM-8B.
Arms per pair: TELE_N16 (L-SML on the 16 standalone channels), MEAN_N16 (plain average of the same channels), the PRM
at its native threshold, and the PRM under the same label-free rule as the telemetry arms. All scores are the STORED
ones (no refit, no inference): standalone PRMBench cells via E.load_prmbench(..., 'prmbench_standalone'), the other
benchmarks via the CROP/ARES 'sa' cells (the PRM rewards are grafted onto the same answers in both).

Protocols re-used function by function:
  PRISM  PRMBench Table 2 rates + Table 7 categories: evaluate_collection.rates / category_scores / official and
         collection_competitors._counts (COMP_PRISM_T1_T2 / T7 code); matched TPR 87.30: stage2 competitors.prism_rows
         logic; ProcessBench Table 6 (TPR = error-sample accuracy, TNR = correct-sample accuracy, F1):
         eval_processbench.evaluate per subset.
  SLIDER PRMBench redundancy class: collection_competitors.slider (10 trace-level 50/50 splits, 1-D LR threshold,
         unweighted and balanced; all steps and steps >= 2; plus the label-free rule).
  CROP   Table 6 (alpha 0.05, 10 seeds, 60/20/20): stage2 competitors.crop_rows on PRMBench, crop_cab.crop elsewhere
         (q201 grid; the PRM also on CROP's fixed 101-point grid). Table 17 cross-domain CRV-Arith <-> CRV-GSM8K.

    python compute.py      # -> RESULTS.csv, PARITY.json
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'prm_sub5_all_benchmarks_v1'))
import sub5_all as A  # noqa: E402  (redirects E.load / E.channels to the CROP/ARES cells)
E, CMP, C6, EP, CAB = A.E, A.CMP, A.C6, A.EP, A.CAB
sys.path.insert(0, str(E.HERE))
import collection_competitors as CC  # noqa: E402

PAIRS = {'qwen_prm_7b': 'Qwen2.5-Math-7B-Instruct', 'qwen_prm_72b': 'Qwen2.5-Math-72B-Instruct',
         'pure_prm': 'Qwen2.5-Math-7B', 'rlhflow_prm': 'Llama-3.1-8B-Instruct'}
TELE = ['TELE_N16', 'MEAN_N16']
ROWS, PAR = [], []


def add(prm, bench, protocol, arm, metric, value, **kw):
    ROWS.append(dict(prm=prm, standalone_model=PAIRS[prm], benchmark=bench, protocol=protocol, arm=arm, metric=metric,
                     value=float(value), **kw))


def gate(name, got, ref, tol=1e-6):
    d = abs(float(got) - float(ref)); PAR.append(dict(check=name, got=float(got), ref=float(ref), abs_diff=d, tol=tol, ok=d <= tol))
    print(('PASS ' if d <= tol else 'FAIL ') + name, round(float(got), 6), round(float(ref), 6), flush=True)


def tables(name):
    return E.read_csv(E.TABLES / name)


# ============================================================================================== PRMBench
def prmbench(prm):
    c = E.load('prmbench', f'{prm}__standalone')
    scale = 'PRM_Q80_PROB' if A.META[f'prmbench/{prm}']['prm_reference_scale'] == '1-reward' else 'PRM_Q80'
    arms = [('PRM native', 'PRM_native', 'native'), ('PRM (same rule)', scale, 'GAP')] + [(a, a, 'GAP') for a in TELE]
    # ---- PRISM Table 2 / Table 7 at the decision rule
    for label, arm, rule in arms:
        f = c.flags(arm, rule); ps, _ = E.official(c, f, rule == 'native')
        for pool, m in (('non-control', c.keep), ('with control rows', np.ones_like(c.keep))):
            r = E.rates(CC._counts(c, f, m))
            for k in ('FPR', 'FNR', 'Precision', 'TPR', 'TNR'):
                add(prm, 'prmbench', 'PRISM T2', label, k, r[k], pool=pool, rule=rule)
        add(prm, 'prmbench', 'PRISM T2', label, 'PRMScore', ps, pool='non-control', rule=rule)
        for k, v in E.category_scores(c, f).items():
            add(prm, 'prmbench', 'PRISM T7', label, k, v, rule=rule)
        add(prm, 'prmbench', 'PRISM T7', label, 'Overall', ps, rule=rule)
    # ---- PRISM matched TPR 87.30 (global threshold chosen with labels), as in stage2 competitors.prism_rows
    for label, arm, _ in arms:
        if arm == scale: continue                     # same PRM score as PRM_native up to a monotone map
        v = c.arms[arm] if arm == 'PRM_native' else CC.thr_score(c, arm)
        ts = np.quantile(v[c.keep], np.linspace(0.6, 0.95, 701))
        best = min((abs(E.rates(CC._counts(c, v >= t, c.keep))['TPR'] - 87.30), t) for t in ts)[1]
        fm = v >= best; r = E.rates(CC._counts(c, fm, c.keep))
        for k in ('FPR', 'FNR', 'Precision', 'TPR', 'TNR'):
            add(prm, 'prmbench', 'PRISM matched TPR 87.3', label, k, r[k], rule='global threshold, labels')
        add(prm, 'prmbench', 'PRISM matched TPR 87.3', label, 'PRMScore', E.official(c, fm)[0], rule='global threshold, labels')
    # ---- SLIDER: redundancy class (collection_competitors.slider, unchanged logic)
    R = np.flatnonzero(c.cls == 'redundency'); sl = [slice(c.off[j], c.off[j + 1]) for j in R]
    for label, arm, rule in arms:
        s = c.arms[arm]; flags = c.flags(arm, rule)
        for scope in ('all_steps', 'steps_ge2'):
            keep = [np.arange(x.start, x.stop)[(slice(None) if scope == 'all_steps' else slice(1, None))] for x in sl]
            yy = np.concatenate([c.err[k] for k in keep]); ff = np.concatenate([flags[k] for k in keep])
            add(prm, 'prmbench', 'SLIDER', label, 'acc', 100 * float((ff == yy).mean()), scope=scope, rule=f'label-free {rule}')
            add(prm, 'prmbench', 'SLIDER', label, 'bal_acc', 50 * (ff[yy].mean() + (~ff[~yy]).mean()), scope=scope, rule=f'label-free {rule}')
            add(prm, 'prmbench', 'SLIDER', 'majority class', 'acc', 100 * float((~yy).mean()), scope=scope, rule='always not redundant')
            for cw in ('unweighted', 'balanced'):
                acc, bal = [], []
                for sd in range(10):
                    perm = np.random.default_rng(sd).permutation(len(keep)); h = len(keep) // 2
                    xh = np.concatenate([s[keep[k]] for k in perm[:h]])[:, None]; yh = np.concatenate([c.err[keep[k]] for k in perm[:h]])
                    xt = np.concatenate([s[keep[k]] for k in perm[h:]])[:, None]; yt = np.concatenate([c.err[keep[k]] for k in perm[h:]])
                    p = LogisticRegression(class_weight=None if cw == 'unweighted' else 'balanced').fit(xh, yh).predict(xt).astype(bool)
                    acc.append(100 * (p == yt).mean()); bal.append(50 * (p[yt].mean() + (~p[~yt]).mean()))
                add(prm, 'prmbench', 'SLIDER', label, 'acc', np.mean(acc), scope=scope, rule=f'LR threshold {cw}')
                add(prm, 'prmbench', 'SLIDER', label, 'bal_acc', np.mean(bal), scope=scope, rule=f'LR threshold {cw}')
    # ---- CROP Table 6 on PRMBench (stage2 competitors.crop_rows: PRM_native = 1-reward on the fixed 101 grid)
    assert np.allclose(c.arms['PRM_Q80_PROB'], 1 - c.extra['reward']) or scale == 'PRM_Q80'
    cr = {r['arm']: r for r in CMP.crop_rows(c, ['PRM_native', 'PRM_Q80_PROB'] + TELE)}
    for arm, label, grid in (('PRM_native', 'PRM (1-reward)', 'fixed101'), ('PRM_Q80_PROB', 'PRM (1-reward)', 'q201')) + tuple((a, a, 'q201') for a in TELE):
        for k, m in (('kept_alpha05', 'kept'), ('risk_alpha05', 'risk'), ('auroc', 'auroc')):
            add(prm, 'prmbench', 'CROP T6', label, m, cr[arm][k], grid=grid)
    # ---- parity
    t1 = {(r['prm'], r['arm'], r['rule']): r for r in tables('COMP_PRISM_T1_T2.csv') if r['pool'].startswith('non')}
    t7 = {(r['prm'], r['arm'], r['rule']): r for r in tables('COMP_PRISM_T7_CATEGORIES.csv')}
    mine = lambda proto, label, met, **kw: next(r['value'] for r in ROWS if r['prm'] == prm and r['benchmark'] == 'prmbench' and r['protocol'] == proto
                                                 and r['arm'] == label and r['metric'] == met and all(r.get(k) == v for k, v in kw.items()))
    for k in ('PRMScore', 'FPR', 'TPR', 'TNR', 'Precision'):
        gate(f'{prm} PRMBench PRM native {k} vs COMP_PRISM_T1_T2', mine('PRISM T2', 'PRM native', k, pool='non-control'),
             t1[(prm, 'PRM_native', 'native')][k])
    for k in ('NR', 'NCL', 'ES', 'SC', 'DC', 'CI', 'PS', 'DR', 'MS'):
        gate(f'{prm} PRMBench PRM native T7 {k} vs COMP_PRISM_T7', mine('PRISM T7', 'PRM native', k), t7[(prm, 'PRM_native', 'native')][k])
    mt = {(r['prm'], r['arm']): r for r in tables('COMP_PRISM_MATCHED_TPR.csv')}
    gate(f'{prm} PRMBench PRM matched-TPR FPR vs COMP_PRISM_MATCHED_TPR', mine('PRISM matched TPR 87.3', 'PRM native', 'FPR'), mt[(prm, 'PRM_native')]['FPR'])
    t6 = {(r['prm'], r['proxy']): r for r in tables('COMP_CROP_T6.csv')}
    gate(f'{prm} PRMBench CROP Direct PRM kept vs COMP_CROP_T6', cr['PRM_native']['kept_alpha05'], t6[(prm, 'Direct PRM (1-reward)')]['kept_mean'])
    ts = {(r['prm'], r['arm'], r['rule']): r for r in tables('COMP_SLIDER.csv') if r['prm'] != 'paper'}
    gate(f'{prm} PRMBench SLIDER PRM native LR balanced acc vs COMP_SLIDER',
         mine('SLIDER', 'PRM native', 'acc', scope='all_steps', rule='LR threshold balanced'), ts[(prm, 'PRM_native', 'native')]['all_steps__LR_balanced__acc'])
    sa = {(r['prm'], r['arm']): r for r in E.read_csv(A.ROOT / 'results/prm_standalone_telemetry_v1/tables/STANDALONE_MAIN.csv')}
    for a in TELE:
        gate(f'{prm} PRMBench standalone {a} PRMScore GAP vs STANDALONE_MAIN', mine('PRISM T2', a, 'PRMScore'), sa[(prm, f'SA_{a}')]['GAP'])
    s5 = {(r['arm'], r['metric']): float(r['value']) for r in E.read_csv(A.HERE / 'SUB5_ALL.csv')
          if r['benchmark'] == 'prmbench' and r['mode'] == 'standalone' and r['prm'] == prm}
    for a in TELE:
        gate(f'{prm} PRMBench standalone {a} matched-TPR PRMScore vs SUB5_ALL', mine('PRISM matched TPR 87.3', a, 'PRMScore'), s5[(a, 'PRMScore_sameTPR87.3')])
        gate(f'{prm} PRMBench standalone {a} CROP kept vs SUB5_ALL', cr[a]['kept_alpha05'], s5[(a, 'crop_kept_a05')])
    gate(f'{prm} PRMBench PRM (same rule) PRMScore GAP vs SUB5_ALL', mine('PRISM T2', 'PRM (same rule)', 'PRMScore'), s5[('PRM', 'PRMScore_GAP')])
    del c


# ============================================================================================== CAB cells (sa)
def cab_cell(bench, prm):
    key = f'cab|sa|{bench}'
    A.REG[key] = dict(mode='sa', bench=bench, common=A.common_uids(bench))
    c = E.load(key, prm)
    if bench == 'processbench': c.keep = c.keep_gold
    return c


def lam_of(per, labels, cal, grid):
    """crop_cab.crop's lambda (copied; kept/risk below are checked against crop_cab.crop)."""
    losses = np.zeros((len(grid), len(cal)))
    for k, i in enumerate(cal):
        fe = np.flatnonzero(labels[i])
        if not len(fe): continue
        losses[:, k] = grid >= np.max(per[i][:fe[0] + 1])
    risk = (1 + losses.sum(1)) / (len(cal) + 1); ok = np.flatnonzero(risk <= C6.ALPHA)
    return grid[ok[np.argmax(grid[ok])]] if len(ok) else grid.min()


def evalp(per, labels, test, lam):
    M = np.array([C6.prefix_len(per[i], lam) for i in test]); T = np.array([len(per[i]) for i in test])
    has = np.array([labels[i].any() for i in test]); cont = np.array([bool(m > 0 and labels[i][:m].any()) for m, i in zip(M, test)])
    acc = M == T
    return dict(kept=100 * float(np.mean(M / np.maximum(T, 1))), risk=100 * float(cont.mean()),
                FA=100 * float((acc & has).mean()), accept=100 * float(acc.mean()))


def crop_arms(c):
    return [('PRM (1-reward)', 'PRMPROB', 'fixed101'), ('PRM (1-reward)', 'PRMPROB', 'q201')] + [(a, a, 'q201') for a in TELE]


def grid_of(name, per, tr):
    return np.round(np.linspace(0, 1, 101), 2) if name == 'fixed101' else \
        np.unique(np.quantile(np.concatenate([per[i] for i in tr]), np.linspace(0, 1, 201)))


def crop_table6(c, bench, prm):
    labels = [c.err[a:b] for a, b in zip(c.off[:-1], c.off[1:])]; has_err = np.array([y.any() for y in labels])
    out = {}
    for label, arm, gname in crop_arms(c):
        per = [c.arms[arm][a:b] for a, b in zip(c.off[:-1], c.off[1:])]; res = []
        for seed in C6.SEEDS:
            tr, ca, te = C6.split_traces(has_err, seed); grid = grid_of(gname, per, tr)
            r = evalp(per, labels, te, lam_of(per, labels, ca, grid)); ref = C6.crop(per, labels, ca, te, grid)
            assert abs(r['kept'] - ref['kept']) < 1e-9 and abs(r['risk'] - ref['risk']) < 1e-9
            r['auroc'] = float(roc_auc_score(np.concatenate([labels[i] for i in te]), np.concatenate([per[i] for i in te])))
            res.append(r)
        for m in ('auroc', 'kept', 'risk'):
            v = np.array([r[m] for r in res])
            add(prm, bench, 'CROP T6', label, m, v.mean(), grid=gname, ci95=float(1.96 * v.std(ddof=1) / np.sqrt(len(v))))
        out[(arm, gname)] = float(np.mean([r['kept'] for r in res]))
    return out


def crop_cross(src, tgt, prm):
    """CROP Table 17: the threshold is calibrated on the source domain's calibration split (grid from its train split),
    then applied to the target domain's test split of the same seed. FA = accepted whole traces that contain an error;
    accept = whole traces accepted (our reading of the table's columns, both % of target test traces)."""
    cs, ct = cab_cell(src, prm), cab_cell(tgt, prm)
    ls = [cs.err[a:b] for a, b in zip(cs.off[:-1], cs.off[1:])]; lt = [ct.err[a:b] for a, b in zip(ct.off[:-1], ct.off[1:])]
    hs = np.array([y.any() for y in ls]); ht = np.array([y.any() for y in lt])
    for label, arm, gname in crop_arms(cs):
        ps = [cs.arms[arm][a:b] for a, b in zip(cs.off[:-1], cs.off[1:])]; pt = [ct.arms[arm][a:b] for a, b in zip(ct.off[:-1], ct.off[1:])]
        res = []
        for seed in C6.SEEDS:
            tr, ca, _ = C6.split_traces(hs, seed); _, _, te = C6.split_traces(ht, seed)
            res.append(evalp(pt, lt, te, lam_of(ps, ls, ca, grid_of(gname, ps, tr))))
        for m in ('risk', 'kept', 'FA', 'accept'):
            add(prm, f'{src}->{tgt}', 'CROP T17', label, m, np.mean([r[m] for r in res]), grid=gname)
    del cs, ct


def processbench(prm):
    c = cab_cell('processbench', prm)
    crop_table6(c, 'processbench', prm)
    # PRISM Table 6: per-subset error-sample accuracy (TPR), correct-sample accuracy (TNR), F1
    S = np.load(A.cab_dir('sa', prm, 'processbench') / 'scores' / f'SCORES_sa_{A.PFILE.get(prm, prm)}_processbench.npz')
    lay = {k: S[k] for k in ('uids', 'nonempty', 'nonempty_lengths', 'native_valid')}
    assert list(lay['uids'].astype(str)) == list(c.ids)
    R, _, _ = EP.evaluate(lay, {'PRM': c.arms['PRM'], **{a: c.arms[a] for a in TELE}}, A.PB_GOLD, 'scored', A.PB_GOLD, False)
    R = {r['arm']: r for r in R}
    for name, label, rule in [('PRM_native', 'PRM native', 'native')] + [(f'{a}_{r}', a, r) for a in TELE for r in ('Q80', 'NC')] + [('PRM_Q80', 'PRM (same rule)', 'Q80')]:
        r = R[name]
        for s in ('gsm8k', 'math', 'omnimath', 'olympiadbench'):
            add(prm, 'processbench', 'PRISM PB T6', label, f'{s} TPR', r[f'{s}_error_acc'], rule=rule)
            add(prm, 'processbench', 'PRISM PB T6', label, f'{s} TNR', r[f'{s}_correct_acc'], rule=rule)
            add(prm, 'processbench', 'PRISM PB T6', label, f'{s} F1', r[f'{s}_f1'], rule=rule)
        add(prm, 'processbench', 'PRISM PB T6', label, 'average F1', r['avg_f1'], rule=rule)
    s5 = {(r['arm'], r['metric']): float(r['value']) for r in E.read_csv(A.HERE / 'SUB5_ALL.csv')
          if r['benchmark'] == 'processbench' and r['mode'] == 'standalone' and r['prm'] == prm}
    gate(f'{prm} ProcessBench standalone TELE_N16 avg F1 NC vs SUB5_ALL', R['TELE_N16_NC']['avg_f1'], s5[('TELE_N16', 'avg_f1_NC')])
    gate(f'{prm} ProcessBench PRM native avg F1 vs SUB5_ALL', R['PRM_native']['avg_f1'], s5[('PRM native', 'avg_f1_native')])
    del c


def crop_parity(prm):
    """The CROP code path on a cell whose stored Table 6 exists (crop_cab's CROP_TABLE6_CAB, sa mode)."""
    ref = {(r['benchmark'], r['arm'], r['grid']): float(r['kept']) for r in E.read_csv(A.HERE / 'vendor/CROP_TABLE6_CAB.csv')
           if r['mode'] == 'sa' and r['prm'] == A.PFILE.get(prm, prm)}
    for bench in ('crv_arith', 'crv_gsm8k', 'prm800k'):
        c = cab_cell(bench, prm); got = crop_table6(c, bench, prm)
        for a in TELE:
            gate(f'{prm} {bench} standalone {a} CROP kept vs CROP_TABLE6_CAB', got[(a, 'q201')], ref[(bench, a, 'q201')])
        del c


def main():
    for prm in PAIRS:
        prmbench(prm); processbench(prm); crop_parity(prm)
        crop_cross('crv_arith', 'crv_gsm8k', prm); crop_cross('crv_gsm8k', 'crv_arith', prm)
        print(prm, 'done', flush=True)
    E.write_csv(HERE / 'RESULTS.csv', ROWS)
    fails = [p['check'] for p in PAR if not p['ok']]
    json.dump(dict(n_checks=len(PAR), failed=fails, max_abs_diff=max(p['abs_diff'] for p in PAR), checks=PAR),
              open(HERE / 'PARITY.json', 'w'), indent=1)
    print('rows', len(ROWS), 'parity', len(PAR), 'failed', fails)


if __name__ == '__main__':
    main()
