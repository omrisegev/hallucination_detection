"""Evaluation stage of dufs_liu_prmbench_v1 (PREREG.md). Called by `python run.py evaluate`.

Reads scores/SCORES_<prm>.npz and scores/FIT_<prm>.json, the exported reference arms and the labels through
evaluate_collection.load_prmbench. Writes RESULTS.csv, CONTRASTS.csv, DIAGNOSTICS.csv, GATES.csv, WEIGHTS.csv and
VERDICT.json. Labels are used here only to evaluate and for the declared label-using diagnostics.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

import evaluate_collection as E
from run import BUDGETS, HERE, LABEL_ARMS, REFS, SCORES, SINGLE_ARMS, sha256
from spectral_utils.anchored_liu import pooled_zscore_features

RULES = ('GAP', 'Q80')


def draws_of(c, arm):
    """Arm names in c.arms that make up one reported arm (one name, or one per calibration draw)."""
    return [arm] if arm in c.arms else sorted(k for k in c.arms if k.startswith(arm + '#d'))


def add_scores(c, prm):
    S = np.load(SCORES / f'SCORES_{prm}.npz')
    assert np.array_equal(S['offsets'], c.off) and np.array_equal(S['row_ids'].astype(str), c.ids)
    for k in S.files:
        if not k.startswith('score__'): continue
        v = S[k].astype(float); name = k[len('score__'):]
        if v.ndim == 1:
            c.arms[name] = v
        else:
            for d in range(v.shape[0]): c.arms[f'{name}#d{d}'] = v[d]
    return json.loads((SCORES / f'FIT_{prm}.json').read_text())


def boot_arm(c, arm, W, rule='GAP'):
    """Per-replicate (official, within) averaged over the arm's draws; replicate 0 is the full-sample point."""
    Wm, inv, ng = W; o, a = [], []
    for name in draws_of(c, arm):
        _, C, wa = E.measure(c, name, rule, fast=True)
        oo, aa = E.boot_stats(c, C, wa, Wm, inv, ng); o.append(oo); a.append(aa)
    return np.mean(o, 0), np.mean(a, 0)


def contrast(c, a, b, W, cache, rule='GAP'):
    for arm in (a, b):
        if (arm, rule) not in cache: cache[(arm, rule)] = boot_arm(c, arm, W, rule)
    (oa, aa), (ob, ab) = cache[(a, rule)], cache[(b, rule)]
    rows = []
    for metric, d in (('official', oa - ob), ('within_auc', aa - ab)):
        rows.append(dict(prm=c.prm, contrast=f'{a} - {b}', rule=rule, metric=metric, delta=float(d[0]),
                         ci_low=float(np.quantile(d[1:], .025)), ci_high=float(np.quantile(d[1:], .975)),
                         n_draws_a=len(draws_of(c, a)), n_groups=int(W[2]), bootstrap=E.DRAWS))
    return rows


def prm_best(c):
    """Better PRM scale under GAP, the collection's label-chosen reading."""
    g = {arm: E.measure(c, arm, 'GAP', fast=True)[0]['official'] for arm in ('PRM_Q80_PROB', 'PRM_Q80')}
    return max(g, key=g.get)


def pooled_corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def diagnostics(c, arm, pos_z, len_z, lr_z):
    """D1 nuisance alignment, D2 content-only within-AUC, D4 alignment with the supervised score."""
    rows = []
    for name in draws_of(c, arm):
        z = c.z(name)
        X = np.column_stack([np.ones(len(z)), pos_z, len_z])
        resid = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]
        rows.append(dict(corr_position=pooled_corr(z, pos_z), corr_log_length=pooled_corr(z, len_z),
                         r2_position_length=float(1 - resid.var() / z.var()),
                         within_auc_residualized=float(np.nanmean(E.per_answer_auc(c.err, resid, c.off))),
                         corr_supervised_score=pooled_corr(z, lr_z)))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def subspace_content(prm, F17, targets):
    """Post-hoc (not pre-registered): R^2 of each target on the top-k principal-component scores of the z-scored
    X17 covariance. k = 2 is the subspace every IU-PCR / DUFS-LIU / RHO arm is confined to."""
    C = F17 @ F17.T / F17.shape[1]; evals, vecs = np.linalg.eigh(C); order = np.argsort(evals)[::-1]
    rows = []
    for k in (1, 2, 3, 4, 6):
        P = (vecs[:, order[:k]].T @ F17).T
        X = np.column_stack([np.ones(len(P)), P])
        row = dict(prm=prm, k=k, variance_share=float(evals[order[:k]].sum() / evals.sum()))
        for name, t in targets.items():
            r = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]
            row[f'r2_{name}'] = float(1 - r.var() / t.var())
        rows.append(row)
    return rows


def prm_share(c, zp, ch, score):
    """|w_PRM| / sum|w| of the least-squares fit of the score on [zp, channels, 1] (collection definition)."""
    X = np.column_stack([zp, ch, np.ones(len(zp))])
    wf, *_ = np.linalg.lstsq(X, score, rcond=None); w = wf[:-1]
    return float(abs(w[0]) / np.abs(w).sum())


def main():
    results, contrasts, diags, gates_rows, weight_rows, subspace_rows = [], [], [], [], [], []
    for prm in E.PRMS:
        c = E.load_prmbench(prm); fit = add_scores(c, prm)
        best = prm_best(c); W = E.boot_weights(c); cache = {}
        ch = E.channels(c, 'N16'); zp = E.answer_z(c.arms['PRM_Q80'], c.off)
        pos_z = E.answer_z(c.extra['position'], c.off); len_z = E.answer_z(c.extra['log_step_tokens'], c.off)
        F17, k17 = pooled_zscore_features(np.column_stack([zp, ch]))
        lr = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000).fit(F17.T, c.err.astype(int))
        lr_z = E.answer_z(lr.decision_function(F17.T), c.off)              # in-sample supervised direction (diagnostic)
        arms = list(REFS) + list(SINGLE_ARMS) + [f'{a}@{b}' for a in LABEL_ARMS for b in BUDGETS]
        for arm in arms:
            names = draws_of(c, arm)
            per = {rule: [E.measure(c, n, rule, fast=True)[0] for n in names] for rule in RULES}
            row = dict(prm=prm, arm=arm, n_draws=len(names), access=access(arm),
                       within_auc=float(np.mean([r['within_auc'] for r in per['GAP']])),
                       within_auc_draw_sd=float(np.std([r['within_auc'] for r in per['GAP']])),
                       n_auc=per['GAP'][0]['n_auc'])
            for rule in RULES:
                row[rule] = float(np.mean([r['official'] for r in per[rule]]))
                row[f'{rule}_draw_sd'] = float(np.std([r['official'] for r in per[rule]]))
            row['prm_weight_share'] = float(np.mean([prm_share(c, zp, ch, c.arms[n]) for n in names]))
            row['is_prm_best'] = arm == best
            results.append(row)
            diags.append(dict(prm=prm, arm=arm, **diagnostics(c, arm, pos_z, len_z, lr_z)))
        pairs = [('DUFSLIU_PRM16', 'CHAIN_N16'), ('IU_PRM16', 'CHAIN_N16'), ('DUFSLIU_PRM16', 'IU_PRM16'),
                 ('DUFSLIU_N16', 'TELE_N16'), ('IU_N16', 'TELE_N16'), ('DUFSLIU_N16', 'IU_N16'),
                 ('DUFSLIU_PRM16', 'EQ_N16'), ('DUFSLIU_PRM16', best), ('CHAIN_N16', best),
                 ('PALIU_PRM16', 'DUFSLIU_PRM16'), ('PALIU_PRM16', 'CHAIN_N16'),
                 ('PRHOLIU_PRM16', 'DUFSLIU_PRM16'), ('PRHOLIU_PRM16', 'CHAIN_N16'),
                 ('PRHOPCR_PRM16', 'DUFSLIU_PRM16'), ('PRHOPCR_PRM16', 'CHAIN_N16')]
        for b in BUDGETS:
            pairs += [(f'TALIU_PRM16@{b}', 'DUFSLIU_PRM16'), (f'TALIU_PRM16@{b}', f'LR_PRM16@{b}'),
                      (f'RHOLIU_PRM16@{b}', 'DUFSLIU_PRM16'), (f'RHOLIU_PRM16@{b}', f'LR_PRM16@{b}'),
                      (f'RHOPCR_PRM16@{b}', 'DUFSLIU_PRM16'), (f'TALIU_PRM16@{b}', 'CHAIN_N16'),
                      (f'RHOLIU_PRM16@{b}', 'CHAIN_N16'), (f'LR_PRM16@{b}', 'CHAIN_N16')]
        for a, b in pairs:
            contrasts += contrast(c, a, b, W, cache)
        gates_rows += gate_table(c, prm, fit, ch, zp, pos_z, len_z)
        weight_rows += weight_table(prm, fit)
        subspace_rows += subspace_content(prm, F17, {'supervised_score': lr_z, 'prm_zp': zp, 'position': pos_z,
                                                     'log_length': len_z, 'error_label': c.err.astype(float)})
        print(prm, 'evaluated', flush=True)
    E.write_csv(HERE / 'RESULTS.csv', results); E.write_csv(HERE / 'CONTRASTS.csv', contrasts)
    E.write_csv(HERE / 'DIAGNOSTICS.csv', diags); E.write_csv(HERE / 'GATES.csv', gates_rows)
    E.write_csv(HERE / 'WEIGHTS.csv', weight_rows); E.write_csv(HERE / 'SUBSPACE_POSTHOC.csv', subspace_rows)
    verdict = decide(contrasts)
    verdict['inputs'] = {p.name: sha256(p) for p in sorted(SCORES.glob('*.npz'))}
    (HERE / 'VERDICT.json').write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items() if k != 'inputs'}, indent=1))


def access(arm):
    if arm in REFS: return E.access_of(arm)
    if arm.split('@')[0] in LABEL_ARMS: return 'labels (cross-fitted calibration)'
    if arm.startswith(('PALIU', 'PRHO')): return 'PRM pseudo-label (no PRMBench label)'
    return 'label-free fusion'


def gate_table(c, prm, fit, ch, zp, pos_z, len_z):
    names = ['PRM_zp'] + [str(n) for n in np.load(E.DATA / 'prmbench' / f'ABL5_CHANNELS_{prm}.npz')['channel_names__N16']]
    X = np.column_stack([zp, ch]); lf = fit['label_free']
    full = [v['TALIU']['gates'] for k, v in fit['label'].items() if k.startswith('FULL/')]
    rows = []
    for j, n in enumerate(names):
        rows.append(dict(prm=prm, channel=n, dufs_gate=lf['DUFSLIU_PRM16']['gates'].get(n, np.nan),
                         prm_pseudo_gate=lf['PALIU_PRM16']['gates'].get(n, np.nan),
                         prm_pseudo_corr=lf['PALIU_PRM16']['correlations'].get(n, np.nan),
                         label_gate_full_mean=float(np.mean([g.get(n, np.nan) for g in full])),
                         abs_corr_position=abs(pooled_corr(X[:, j], pos_z)), abs_corr_log_length=abs(pooled_corr(X[:, j], len_z)),
                         label_point_biserial=pooled_corr(X[:, j], c.err.astype(float))))
    return rows


def weight_table(prm, fit):
    rows = []
    lf = fit['label_free']
    for arm in SINGLE_ARMS:
        w = lf[arm]['weights']; tot = sum(abs(v) for v in w.values())
        rows += [dict(prm=prm, arm=arm, channel=k, weight=v, abs_share=abs(v) / tot) for k, v in w.items()]
    for arm, key in (('TALIU_PRM16@FULL', 'TALIU'), ('RHOLIU_PRM16@FULL', 'RHOLIU'), ('LR_PRM16@FULL', 'LR')):
        ws = [v[key]['weights'] for k, v in fit['label'].items() if k.startswith('FULL/')]
        names = list(ws[0])
        mean = {n: float(np.mean([w[n] for w in ws])) for n in names}; tot = sum(abs(v) for v in mean.values())
        rows += [dict(prm=prm, arm=arm, channel=n, weight=mean[n], abs_share=abs(mean[n]) / tot) for n in names]
    return rows


def decide(rows):
    def get(con, metric):
        return {r['prm']: r for r in rows if r['contrast'] == con and r['metric'] == metric and r['rule'] == 'GAP'}

    def replace_rule(arm):
        o, w = get(f'{arm} - CHAIN_N16', 'official'), get(f'{arm} - CHAIN_N16', 'within_auc')
        noninf = all(o[p]['ci_low'] > -0.5 and w[p]['ci_low'] > -0.005 for p in E.PRMS)
        better = sum(o[p]['ci_low'] > 0 for p in E.PRMS)
        return dict(non_inferior_all5=noninf, prmscore_ci_above0=better,
                    verdict=('replaces and improves' if noninf and better >= 3 else 'can replace' if noninf else 'cannot replace'))

    ta, ta_lr = get('TALIU_PRM16@512 - DUFSLIU_PRM16', 'within_auc'), get('TALIU_PRM16@512 - LR_PRM16@512', 'within_auc')
    rh, rh_lr = get('RHOLIU_PRM16@512 - DUFSLIU_PRM16', 'within_auc'), get('RHOLIU_PRM16@512 - LR_PRM16@512', 'within_auc')
    pa_w, pa_o = get('PALIU_PRM16 - DUFSLIU_PRM16', 'within_auc'), get('PALIU_PRM16 - DUFSLIU_PRM16', 'official')
    return {
        'A_replacement_DUFSLIU_PRM16': replace_rule('DUFSLIU_PRM16'),
        'B_label_calibration_TALIU_512': dict(
            labels_help_ci_above0=sum(ta[p]['ci_low'] > 0 for p in E.PRMS),
            efficient_vs_LR_lower_above_minus005=sum(ta_lr[p]['ci_low'] > -0.005 for p in E.PRMS),
            verdict_help=sum(ta[p]['ci_low'] > 0 for p in E.PRMS) >= 3,
            verdict_efficient=sum(ta_lr[p]['ci_low'] > -0.005 for p in E.PRMS) >= 3),
        'B_secondary_RHOLIU_512': dict(
            labels_help_ci_above0=sum(rh[p]['ci_low'] > 0 for p in E.PRMS),
            efficient_vs_LR_lower_above_minus005=sum(rh_lr[p]['ci_low'] > -0.005 for p in E.PRMS)),
        'C_pseudo_label_PALIU': dict(
            within_ci_above0=sum(pa_w[p]['ci_low'] > 0 for p in E.PRMS),
            prmscore_point_ge0=sum(pa_o[p]['delta'] >= 0 for p in E.PRMS),
            verdict_help=sum(pa_w[p]['ci_low'] > 0 for p in E.PRMS) >= 3 and sum(pa_o[p]['delta'] >= 0 for p in E.PRMS) >= 3,
            replacement=replace_rule('PALIU_PRM16')),
    }


if __name__ == '__main__':
    main()
