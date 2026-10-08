"""monday_ablation_v1: one-factor-at-a-time ablations of the Monday fusion pipeline (development data).
Protocol: PROTOCOL.json. Factors: channel filter, merge, tail quota, partition input; between-group estimator.
Labels are used only for evaluation and for the descriptive estimator-vs-truth and dependence diagnostics."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import hashlib, importlib.util, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

OUT = Path(__file__).resolve().parent; MAIN = OUT.parents[1]; EXT = MAIN / 'results/monday_external_panel_v1'
SE = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
spec = importlib.util.spec_from_file_location('ext_score', EXT / 'score.py'); ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
from spectral_utils.lsml_gate_locator_research import answer_standardize  # noqa: E402  (path set by ext)
from spectral_utils.monday_fusion_core import choose_anchor, between_weights  # noqa: E402
import er_stage_a as SA, er_stage_b as SB, tail_calib_common as TC, lsml_merge_step as MS, ds_group_weights as GW  # noqa: E402
from calfix_common import tail_marks  # noqa: E402

OB = json.loads((MAIN / 'results/estimator_input_v1/run_20261004_resume/INPUT_MANIFEST.json').read_text(encoding='utf8'))['banks']
CORE = OB['B13'][:11]; REAL = OB['B13'][11:]; DIG = OB['B16'][-3:]
E20 = [c for c in OB['B20'] if c not in CORE]; E51 = [c for c in OB['B51'] if c not in OB['B20']]
HIST18 = [c[4:] for c in OB['B32'] if c.startswith('lf__') and c[4:] not in E20]
NB = {'N13': CORE + REAL, 'N16': CORE + REAL + DIG, 'N25': CORE + REAL + E20 + DIG,
      'N43': CORE + REAL + E20 + HIST18 + DIG, 'N56': CORE + REAL + E20 + E51 + DIG}
ext.BN = NB
VARIANTS = {'default': dict(filt='DS', merge=True, q=.2, cont=False), 'filter_MOM': dict(filt='MOM', merge=True, q=.2, cont=False),
            'filter_SML': dict(filt='SML', merge=True, q=.2, cont=False), 'filter_NONE': dict(filt='NONE', merge=True, q=.2, cont=False),
            'filter_ORIENT': dict(filt='ORIENT', merge=True, q=.2, cont=False), 'merge_OFF': dict(filt='DS', merge=False, q=.2, cont=False),
            'quota_10': dict(filt='DS', merge=True, q=.1, cont=False), 'quota_30': dict(filt='DS', merge=True, q=.3, cont=False),
            'quota_40': dict(filt='DS', merge=True, q=.4, cont=False), 'partition_CONT': dict(filt='DS', merge=True, q=.2, cont=True)}
SCORERS = ['FILTER_EQ', 'MEAN_EQ', 'MEAN_DS', 'MEAN_SML', 'MEAN_MOM']
PRM = 'prmbench_qwen3_8b'; DRAWS, SEED = 2000, 20261007


class Fail(Exception):
    pass


def fit_variant(pop, bank, filt, merge, q, cont, y=None):
    V = bank['V'].copy(); off = bank['off']; key = bank['key']; names = bank['names']; rows = bank['rows']; diag = {}
    if filt == 'ORIENT':
        lev = pop.level[rows]
        sg = np.array([(np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1.) if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
        V = V * sg; diag['flipped'] = [names[j] for j in np.flatnonzero(sg < 0)]
    if q == .2 and filt != 'ORIENT':
        marks, TT = bank['marks'], bank['TT']
    else:
        marks = SB.random_tie_marks(V, off, q, key); TT = tail_marks(V, off, q, tie_aware=True, centred=True)[0]
    pi = np.ones(V.shape[1])
    if filt == 'DS':
        est = SA.em_estimate(marks, 'ds'); pi = est['pi']; surv = np.flatnonzero(pi > .5)
    elif filt == 'MOM':
        est = SA.tensor_mom_estimate(marks, 0); pi = est['pi']; surv = np.flatnonzero(pi > .5)
    elif filt == 'SML':
        est = SA.sml_estimate(marks, 0); pi = .5 + est['t']; surv = np.flatnonzero(est['t'] > 0)
    else:
        surv = np.arange(V.shape[1])
    diag['n_survivors'] = int(len(surv)); diag['dropped'] = [names[j] for j in range(len(names)) if j not in set(surv)]
    if len(surv) < 3: raise Fail('fewer than 3 survivors')
    x = V[:, surv]; scores = {'FILTER_EQ': x.mean(1)}
    anchor, _ = choose_anchor(names, surv, pi); ai = int(np.flatnonzero(surv == anchor)[0])
    part_in = x if cont else TT[:, surv]
    gb = MS.canon(TC.lsml_fit_scaled(part_in, ai, x, standardize=True, loading_scale='unit')['groups'])
    diag['K_before'] = int(len(set(gb)))
    if len(set(gb)) < 3: raise Fail(f'{len(set(gb))} groups before merge')
    if merge:
        groups, seq = MS.absorb_merge(np.corrcoef(part_in, rowvar=False), gb, min_groups=3); diag['merges'] = int(sum('a' in s for s in seq))
    else:
        groups = gb; diag['merges'] = 0
    groups = np.asarray(groups, int); K = int(len(set(groups))); diag['K_after'] = K
    if K < 3: raise Fail('fewer than 3 groups after merge')
    keyG = pop.KEYG[rows, :K] if K <= 13 else np.random.default_rng(20261007).random((pop.S, K))[rows]
    z = GW.group_matrix(x, off, groups, None, answer_standardize); votes = SB.random_tie_marks(z, off, q, keyG)
    diag['weights'] = {}
    for m in ('EQ', 'DS', 'SML', 'MOM'):
        w, _ = between_weights(m, z, votes, x[:, ai], int(groups[ai]))
        scores['MEAN_' + m] = GW.weighted_group_score(z, w, signed=m == 'SML')
        p = np.abs(w) / np.abs(w).sum()
        diag['weights'][m] = {'w': [float(v) for v in w], 'effective_groups': float(1 / (p @ p)), 'max_dev_from_equal': float(np.max(np.abs(p - 1 / K))),
                              'sml_equal_guard': bool(m == 'SML' and K == 3)}
    if y is not None:                                         # descriptive, PRMBench only (labels)
        def cond_dep(Z):
            out = []
            for mask in (y, ~y):
                C = np.corrcoef(Z[mask], rowvar=False); o = ~np.eye(len(C), dtype=bool); out.append(float(np.nanmean(np.abs(C[o]))))
            return out
        zb = GW.group_matrix(x, off, gb, None, answer_standardize)
        diag['dependence_before'] = cond_dep(zb); diag['dependence_after'] = cond_dep(z)
    diag['groups_after'] = [[names[surv[j]] for j in np.flatnonzero(groups == h)] for h in range(K)]
    diag['groups_before'] = [[names[surv[j]] for j in np.flatnonzero(gb == h)] for h in range(int(gb.max()) + 1)]
    return scores, diag, (part_in, gb, groups, [names[j] for j in surv])


def est_vs_truth(marks, y):
    tr = SA.truth(marks, y); out = {}
    for name, f in (('DS', lambda: SA.em_estimate(marks, 'ds')), ('MOM', lambda: SA.tensor_mom_estimate(marks, 0))):
        e = f(); out[name] = {'prevalence_est': float(e['prevalence']), 'prevalence_true': tr['prevalence'], 'prevalence_error': abs(float(e['prevalence']) - tr['prevalence']),
                              'psi_MAE': float(np.abs(e['psi'] - tr['psi']).mean()), 'eta_MAE': float(np.abs(e['eta'] - tr['eta']).mean()),
                              'spearman': float(spearmanr(e['pi'], tr['pi']).statistic)}
    s = SA.sml_estimate(marks, 0); out['SML'] = {'spearman': float(spearmanr(s['t'], tr['pi']).statistic)}
    out['marked_share'] = float((marks > 0).mean())
    return out


def main():
    t0 = time.time()
    P = json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8')); assert P['status'] == 'FROZEN'
    for a, b in (('N16', 'N25'), ('N25', 'N43'), ('N43', 'N56'), ('N13', 'N16')): assert set(NB[a]) < set(NB[b]), (a, b)
    (OUT / 'BANKS.json').write_text(json.dumps(NB, indent=1), encoding='utf8')
    A = pd.read_csv(SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
    with np.load(SE / 'OOF_STEP_SCORES.npz') as zz: OFF = zz['offsets'].astype(np.int64); LAB = zz['labels'].astype(bool)
    N = len(A); cells = A.cell.to_numpy(); target = A.target.to_numpy().astype(int); groups_q = A.source_group.to_numpy(); NS = np.diff(OFF)
    fr = json.loads((SE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
    meta = {m['idx']: m for m in pickle.load(open(fr['prm_metadata']['path'], 'rb')).values()}
    ids = A.id.astype(str).to_numpy(); pb = np.array([c.startswith('pb_') for c in cells])
    noncontrol = np.array([(not pb[i]) and meta[ids[i]]['classification'] != 'correct' for i in range(N)])
    F = np.load(ext.FE / 'source/FEATURES.npz'); pop = ext.Population(F, list(NB)); print('population', round(time.time() - t0), flush=True)
    # parity gate (default == validated adapter) on N16 / PRMBench
    ans = np.flatnonzero(cells == PRM); bank = pop.cell_bank('N16', ans)
    ref, _, _ = ext.fit_cell(pop, bank, 0)
    mine, _, _ = fit_variant(pop, bank, **VARIANTS['default'])
    gate = {k: float(np.max(np.abs(mine[a] - ref[k]))) for k, a in (('MEAN_MOM_ga', 'MEAN_MOM'), ('MEAN_EQ', 'MEAN_EQ'), ('DS_FILTER_EQ', 'FILTER_EQ'))}
    (OUT / 'PARITY_GATE.json').write_text(json.dumps(gate, indent=1), encoding='utf8')
    if max(gate.values()) > 1e-12: raise SystemExit(f'parity gate failed {gate}')
    print('parity', gate, flush=True)
    R, groups_rows, est_rows, fails = {}, [], [], []
    for bk in NB:
        for vname in VARIANTS: R[(bk, vname)] = {s: {'auc': np.full(N, np.nan), 't1': np.full(N, np.nan), 'c1': np.zeros((N, 4), int), 'am': np.full(N, -9, int), 'fail': np.zeros(N, bool)} for s in SCORERS}
        for ce in sorted(set(cells)):
            ans = np.flatnonzero(cells == ce); bank = pop.cell_bank(bk, ans); off = bank['off']
            y = LAB[bank['rows']] if ce == PRM else None
            for vname, kw in VARIANTS.items():
                try:
                    sc, d, ex = fit_variant(pop, bank, y=y, **kw)
                except Fail as exc:
                    fails.append({'bank': bk, 'cell': ce, 'variant': vname, 'reason': str(exc)})
                    for s in SCORERS: R[(bk, vname)][s]['fail'][ans] = True
                    continue
                if bk == 'N16' and ce == PRM and vname in ('default', 'merge_OFF'):
                    np.savez_compressed(OUT / f'MERGE_EXAMPLE_{vname}.npz', R=np.corrcoef(ex[0], rowvar=False), gb=ex[1], groups=ex[2], names=np.array(ex[3]))
                groups_rows.append({'bank': bk, 'cell': ce, 'variant': vname, 'K_before': d['K_before'], 'K_after': d['K_after'], 'merges': d['merges'],
                                    'n_survivors': d['n_survivors'], 'n_channels': len(NB[bk]),
                                    **{f'{m}_eff': d['weights'][m]['effective_groups'] for m in d['weights']},
                                    **{f'{m}_maxdev': d['weights'][m]['max_dev_from_equal'] for m in d['weights']},
                                    'sml_guard': d['weights']['SML']['sml_equal_guard'],
                                    'dep_err_before': d.get('dependence_before', [np.nan, np.nan])[0], 'dep_ok_before': d.get('dependence_before', [np.nan, np.nan])[1],
                                    'dep_err_after': d.get('dependence_after', [np.nan, np.nan])[0], 'dep_ok_after': d.get('dependence_after', [np.nan, np.nan])[1],
                                    'groups_after': json.dumps(d['groups_after'])})
                for s in SCORERS:
                    st = R[(bk, vname)][s]; v = sc[s]
                    for j, i in enumerate(ans):
                        ss = v[off[j]:off[j + 1]]; am = int(np.flatnonzero(ss >= ss.max() - 8 * np.finfo(float).eps)[0]); st['am'][i] = am
                        if ce == PRM:
                            yy = LAB[OFF[i]:OFF[i + 1]]; n1 = yy.sum(); n0 = len(yy) - n1
                            if n1 and n0: st['auc'][i] = (rankdata(ss)[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
                            if n1: st['t1'][i] = float(yy[am])
                            if noncontrol[i]:
                                f = np.zeros(len(ss), bool); f[am] = True; vv, g = ~f, ~yy
                                st['c1'][i] = [(vv & g).sum(), (vv & ~g).sum(), (~vv & ~g).sum(), (~vv & g).sum()]
            if ce == PRM:                                   # estimator vs truth at each quota (all channels' marks)
                for q in (.1, .2, .3, .4):
                    mk = bank['marks'] if q == .2 else SB.random_tie_marks(bank['V'], off, q, bank['key'])
                    e = est_vs_truth(mk, LAB[bank['rows']]); est_rows.append({'bank': bk, 'quota': q, **{f'{a}_{b}': c for a, dd in e.items() if isinstance(dd, dict) for b, c in dd.items()}, 'marked_share': e['marked_share']})
            print(bk, ce, round(time.time() - t0), flush=True)
    pd.DataFrame(groups_rows).to_csv(OUT / 'GROUPS.csv', index=False); pd.DataFrame(est_rows).to_csv(OUT / 'ESTIMATOR_VS_TRUTH.csv', index=False)
    (OUT / 'FAILURES.json').write_text(json.dumps(fails, indent=1), encoding='utf8')
    # ---- aggregation with paired source-group bootstrap (variant minus default, same scorer)
    prm_idx = np.flatnonzero(~pb); pb_idx = np.flatnonzero(pb)
    pg, pinv = np.unique(groups_q[prm_idx], return_inverse=True); bg, binv = np.unique(groups_q[pb_idx], return_inverse=True)
    rng = np.random.default_rng(SEED)
    Wp = np.vstack([np.ones(len(pg)), rng.multinomial(len(pg), np.full(len(pg), 1 / len(pg)), size=DRAWS)]).astype(float)
    Wb = np.vstack([np.ones(len(bg)), rng.multinomial(len(bg), np.full(len(bg), 1 / len(bg)), size=DRAWS)]).astype(float)
    pbc = sorted(set(cells[pb_idx])); cof = np.array([pbc.index(c) for c in cells[pb_idx]]); err = target[pb_idx] >= 0
    def gs(X, inv, G): o = np.zeros((G, X.shape[1])); np.add.at(o, inv, X); return o
    def stats(st):
        a = st['auc'][prm_idx]; t = st['t1'][prm_idx]
        S = Wp @ gs(np.column_stack([np.nan_to_num(a), np.isfinite(a), np.nan_to_num(t), np.isfinite(t)]).astype(float), pinv, len(pg))
        c = Wp @ gs(st['c1'][prm_idx].astype(float), pinv, len(pg)); tp, fp, tn, fn = c.T
        hit = st['am'][pb_idx] == target[pb_idx]; cols = []
        for k in range(len(pbc)): ck = cof == k; cols += [hit & err & ck, err & ck]
        H = (Wb @ gs(np.column_stack(cols).astype(float), binv, len(bg))).reshape(len(Wb), len(pbc), 2)
        with np.errstate(invalid='ignore', divide='ignore'):
            return {'prmb_auc': S[:, 0] / S[:, 1], 'prmb_top1': S[:, 2] / S[:, 3], 'prmb_ps1': .5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)),
                    'pb_hit': np.nanmean(H[:, :, 0] / H[:, :, 1], 1)}
    mrows, crows = [], []
    for (bk, vname), d in R.items():
        for s in SCORERS:
            st = d[s]; f_prm = bool(st['fail'][prm_idx].any()); f_pb = bool(st['fail'][pb_idx].any())
            if f_prm and f_pb: mrows.append({'bank': bk, 'variant': vname, 'scorer': s, 'status': 'FAILED'}); continue
            x = stats(st)
            row = {'bank': bk, 'variant': vname, 'scorer': s, 'status': 'COMPLETE' if not (f_prm or f_pb) else 'PARTIAL'}
            for k, vv in x.items():
                bad = (k.startswith('prmb') and f_prm) or (k == 'pb_hit' and f_pb); row[k] = np.nan if bad else float(vv[0])
            mrows.append(row)
            if vname != 'default' and not R[(bk, 'default')][s]['fail'].any():
                base = stats(R[(bk, 'default')][s])
                for k in x:
                    if (k.startswith('prmb') and f_prm) or (k == 'pb_hit' and f_pb): continue
                    dlt = x[k] - base[k]; crows.append({'bank': bk, 'variant': vname, 'scorer': s, 'metric': k, 'delta': float(dlt[0]),
                                                        'lo95': float(np.nanquantile(dlt[1:], .025)), 'hi95': float(np.nanquantile(dlt[1:], .975))})
    pd.DataFrame(mrows).to_csv(OUT / 'METRICS.csv', index=False); pd.DataFrame(crows).to_csv(OUT / 'CONTRASTS.csv', index=False)
    (OUT / 'STATUS.json').write_text(json.dumps({'status': 'DONE', 'runtime_s': round(time.time() - t0), 'failures': len(fails),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'protocol_sha256': hashlib.sha256((OUT / 'PROTOCOL.json').read_bytes()).hexdigest()}, indent=1), encoding='utf8')
    print('done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
