"""Diagnosis stages D1-D3 of the raw-null gate (plan: docs/experiments/RAW_NULL_GATE_DIAGNOSIS_PLAN_2026-10-31.md).

Diagnostic only. Labels are used ONLY to score label-free quantities; every label-using column is an oracle/diagnostic
number, never a method or a threshold for one. CPU only; no frozen file is edited.

Per ok cell (v1b `fix` CHECK.json; 68 cells):
  1. LABEL-FREE: recompute the v1b `fix` label-free stage (raw fused score, group scores u_g, null fits) with the frozen
     v1b code, and check it reproduces the saved fix/labelfree npz (raw_fused, p_err_step_G1/G2 to 1e-9).
  2. LABELS (only after 1): has_err per answer, evaluated answers, step labels and step mask (v1 labels_of = run_e1's).
  3. D1 answer-level AUC of each gate statistic + oracle closed_clean at kept_error = .85.
     D2 true correct-step distribution vs the fitted null, error share in the central band.
     D3 step-level AUC and Cohen's d of error vs correct steps, per channel, per group score, fused.
  4. Bundle (label-free + labels) per cell for re-analysis without raw data.

    python diag_d1_d3.py            # all cells -> D1.csv, D2.csv, D3.csv, bundle/
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import csv, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
_s = importlib.util.spec_from_file_location('run_v1b', ROOT / 'results/raw_null_gate_v1b/run_raw_null_gate_v1b.py')
B = importlib.util.module_from_spec(_s); _s.loader.exec_module(B)                # frozen v1b runner, used as a library
G, V1 = B.G, B.V1
FIX, MONO = ROOT / 'results/raw_null_gate_v1b/fix', ROOT / 'results/raw_null_gate_v1b/fix_mono'
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
KE_TARGET = .85


# ----------------------------------------------------------------------------------------------- label-free
def labelfree_parts(arm, c, X_az, Z_raw, pz, zr):
    """The same steps as v1b labelfree_cell, keeping the intermediate arrays D1-D3 need. No labels."""
    V, z_fused, p = V1.fit_arm(arm, X_az, pz)
    finite = np.isfinite(Z_raw)
    Zr = np.where(finite, Z_raw, 0.)
    names = list(V1.N16N if arm != 'SUB5' else V1.SUB5)
    if arm == 'AUTO':
        Zr = np.column_stack([zr, Zr]); finite = np.column_stack([np.ones(len(zr), bool), finite]); names = ['PRM'] + names
    defined = G.group_defined(finite, p)
    raw_fused = G.pooled_z(G.fused_linear(Zr, p))
    raw_groups = G.group_scores(Zr, p)
    orient = G.group_orientation(p)
    U = np.full(raw_groups.shape, np.nan); gfits = []
    for g in range(raw_groups.shape[1]):
        if orient[g] == 0:
            gfits.append(dict(ok=False, reason='orientation 0')); continue
        u, f = G._group_null(raw_groups[:, g], orient[g], defined[:, g])
        U[defined[:, g], g] = u[defined[:, g]]
        gfits.append({k: v for k, v in f.items() if k != 'lfdr'})
    g1 = G.empirical_null(raw_fused)
    g2 = G.gate_g2(raw_groups, orient, c.off, defined=defined)
    chan_sign = np.zeros(len(names))                                         # each channel's sign inside the fused score
    for g, (idx, w) in enumerate(p['gw']):
        for i, wi in zip(idx, w):
            chan_sign[int(i)] = np.sign(p['sign'] * (p['cw'][g] if len(p['gw']) > 1 else 1.) * (wi if len(idx) > 1 else 1.))
    return dict(z_fused=z_fused, raw_fused=raw_fused, U=U, defined=defined, orient=orient, p=p, finite=finite,
                Zr=np.where(finite, Zr, np.nan), names=names, g1fit={k: v for k, v in g1.items() if k != 'lfdr'},
                gfits=gfits, p_err_step_G1=G.p_error(raw_fused, g1),
                p_err_step_G2=g2['p_err_step'] if g2['ok'] else np.full(len(raw_fused), np.nan), chan_sign=chan_sign)


def answer_max_nan(v, off):
    out = np.full(len(off) - 1, np.nan)
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        x = v[a:b]; x = x[np.isfinite(x)]
        if len(x):
            out[j] = x.max()
    return out


# ----------------------------------------------------------------------------------------------- label-using (diagnostic)
def auc(y, s):
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if m.sum() < 2 or y.all() or not y.any() or np.all(s == s[0]):
        return np.nan, int(m.sum())
    return float(roc_auc_score(y, s)), int(m.sum())


def oracle_cc(y, s, ke=KE_TARGET):
    """ORACLE: call 'error' iff s >= t, with t the largest threshold keeping >= ke of erroneous answers; return the
    share of clean answers called clean (s < t). Higher s = more error. Uses labels to place t: diagnostic only."""
    m = np.isfinite(s); y, s = y[m], s[m]
    if not y.any() or y.all():
        return np.nan
    e = np.sort(s[y])
    t = e[int(np.floor((1 - ke) * len(e)))]                               # >= ke of errors have s >= t
    return float(np.mean(s[~y] < t))


def cohen_d(x, y):
    a, b = x[y], x[~y]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def d2_row(x, pool, err, labmask, stepmask, fit):
    """True correct-step distribution of score x vs the fitted null. 'Labelled' = keep_eval (c.keep; keep_gold on
    ProcessBench) WITHOUT the evaluated-answer filter, so PRMBench class-'correct' answers (in the null pool) count.
    The evaluated-only share (stepmask) is reported beside it."""
    lab = pool & labmask; cor = lab & ~err; labe = pool & stepmask
    xc = x[cor]
    q1, q2 = np.quantile(x[pool], [.25, .75])
    band = lab & (x >= q1) & (x <= q2)
    tm, ts = (float(xc.mean()), float(xc.std())) if len(xc) > 1 else (np.nan, np.nan)
    r = dict(n_pool=int(pool.sum()), n_pool_unlabelled=int((pool & ~labmask).sum()),
             share_pool_labelled=float(lab.sum() / max(pool.sum(), 1)),
             true_share_correct=float(cor.sum() / max(lab.sum(), 1)),
             true_share_correct_evaluated_only=float((labe & ~err).sum() / max(labe.sum(), 1)), true_mean=tm, true_sd=ts,
             true_skew=float(skew(xc)) if len(xc) > 2 else np.nan,
             true_exkurt=float(kurtosis(xc)) if len(xc) > 3 else np.nan,
             err_share_labelled=float(err[lab].mean()) if lab.any() else np.nan,
             err_share_central_band=float(err[band].mean()) if band.any() else np.nan,
             fit_ok=bool(fit.get('ok')), fit_reason=fit.get('reason', ''), mu0=fit.get('mu0'), sigma0=fit.get('sigma0'),
             pi0=fit.get('pi0'), pi0_raw=fit.get('pi0_raw'))
    if fit.get('ok') and ts and ts > 0:
        r.update(mu0_minus_true_mean_in_true_sd=(fit['mu0'] - tm) / ts, sigma0_over_true_sd=fit['sigma0'] / ts,
                 pi0_raw_minus_true_share=fit['pi0_raw'] - r['true_share_correct'])
    return r


# ----------------------------------------------------------------------------------------------- main
def main():
    chk = json.loads((FIX / 'CHECK.json').read_text())
    VE, R1 = V1.env()
    rows1, rows2, rows3, manifest = [], [], [], []
    (HERE / 'bundle').mkdir(exist_ok=True)
    for cell in chk['cells']:
        if cell['status'] != 'ok' or cell['prm'] not in PRMS:
            continue
        arm, mode, bench, prm = cell['arm'], cell['mode'], cell['bench'], cell['prm']
        tag = f'{arm}_{mode}_{bench}_{prm}'; t0 = time.time()
        c, X_az, Z_raw, pz, zr = V1.load_cell(VE, R1, arm, mode, bench, prm)
        off = c.off; n = np.diff(off)
        lf = labelfree_parts(arm, c, X_az, Z_raw, pz, zr)
        A = {v: np.load(d / 'labelfree' / f'{tag}.npz') for v, d in (('fix', FIX), ('mono', MONO))}
        for k in ('raw_fused', 'z_fused'):
            assert np.max(np.abs(lf[k] - A['fix'][k])) <= 1e-9, f'{tag}: {k} differs from fix npz'
        for k in ('p_err_step_G1', 'p_err_step_G2'):                         # the U / null path the gates used
            np.testing.assert_allclose(lf[k], A['fix'][k], atol=1e-9, rtol=0, equal_nan=True, err_msg=f'{tag}: {k}')
        w = G.worst_step(lf['z_fused'], off)
        K = lf['U'].shape[1]
        stats = {'G1_raw_at_worst': lf['raw_fused'][w], 'raw_fused_answer_max': G.answer_max(lf['raw_fused'], off),
                 'G1_p_worst': A['fix']['p_err_worst_G1'], 'G1_p_worst_mono': A['mono']['p_err_worst_G1'],
                 'G1b_1_minus_p': 1 - A['fix']['p_answer_G1b'],
                 'G2_p_worst': A['fix']['p_err_worst_G2'], 'G2_p_worst_mono': A['mono']['p_err_worst_G2'],
                 'G3_posterior': A['fix']['p_answer_G3'], 'G3_posterior_mono': A['mono']['p_answer_G3']}
        for g in range(K):
            stats[f'group{g}_answer_max'] = answer_max_nan(lf['U'][:, g], off)
        pos = VE.D.relative_position(off)
        # ---------------- labels from here on (all label-free arrays above are fixed)
        has_err, ev, step_err, stepmask, keep_gold = V1.labels_of(c, bench)
        labmask = (keep_gold if bench == 'processbench' else c.keep).astype(bool)
        err = step_err.astype(bool)
        y = has_err[ev]
        base = dict(arm=arm, mode=mode, bench=bench, prm=prm, K=K)
        for name, s in stats.items():
            a, m = auc(y, np.asarray(s, float)[ev])
            rows1.append(dict(base, statistic=name, n_answers=m, true_share_error=float(y.mean()), auc_DIAG=a,
                              oracle_closed_clean_at_ke85_DIAG=oracle_cc(y, np.asarray(s, float)[ev])))
        rows2.append(dict(base, score='fused', group='', size='', **d2_row(lf['raw_fused'], np.ones(len(err), bool), err,
                                                                         labmask, stepmask, lf['g1fit'])))
        for g in range(K):
            if lf['orient'][g] == 0:
                continue
            rows2.append(dict(base, score='group', group=g, size=len(lf['p']['gw'][g][0]),
                              **d2_row(np.nan_to_num(lf['U'][:, g]), lf['defined'][:, g], err, labmask, stepmask,
                                     lf['gfits'][g])))
        feats = [('fused', 'raw_fused', lf['raw_fused'], 1.)] + \
                [('group', f'group{g}', lf['U'][:, g], 1.) for g in range(K)] + \
                [('channel', nm, lf['Zr'][:, j], lf['chan_sign'][j]) for j, nm in enumerate(lf['names'])]
        for kind, nm, x, sgn in feats:
            m = stepmask & np.isfinite(x)
            a, _ = auc(err[m], x[m])
            rows3.append(dict(base, kind=kind, feature=nm, n_steps=int(m.sum()), err_share=float(err[m].mean()),
                              fused_sign=sgn, step_auc_DIAG=a, cohen_d_DIAG=cohen_d(x[m], err[m])))
        # ---------------- bundle (float32; labels included and named as such)
        np.savez_compressed(HERE / 'bundle' / f'{tag}.npz', offsets=off.astype(np.int64),
                            ans_evaluated=ev, ans_has_err_LABEL=has_err,
                            **{f'ans_{k}': np.asarray(v, np.float32) for k, v in stats.items()},
                            step_raw_fused=lf['raw_fused'].astype(np.float32), step_z_fused=lf['z_fused'].astype(np.float32),
                            step_U=lf['U'].astype(np.float32), step_defined=lf['defined'], step_position=pos.astype(np.float32),
                            step_err_LABEL=err, step_mask=stepmask, step_keep_eval=labmask, worst_step=w.astype(np.int64),
                            ans_class_LABEL=np.asarray(c.cls).astype(str) if hasattr(c, 'cls') else np.array([]),
                            fits_json=np.array(json.dumps(dict(g1=lf['g1fit'], groups=lf['gfits'],
                                                               orientation=lf['orient'].tolist(),
                                                               group_channels=[[int(i) for i in idx] for idx, _ in lf['p']['gw']],
                                                               channel_names=lf['names'], channel_sign_in_fused=lf['chan_sign'].tolist(),
                                                               sign=lf['p']['sign'], cross_weights=np.asarray(lf['p']['cw']).tolist(),
                                                               group_weights=[np.asarray(wt).tolist() for _, wt in lf['p']['gw']]),
                                                          default=float)))
        ch = HERE / 'bundle' / f'channels_{mode}_{bench}_{prm}.npz'
        if arm != 'SUB5':                                                     # the N16 channels (SUB5 = a subset)
            np.savez_compressed(ch, Z=(lf['Zr'][:, 1:] if arm == 'AUTO' else lf['Zr']).astype(np.float32),
                                names=np.array(V1.N16N), offsets=off.astype(np.int64))
        manifest.append(dict(cell=tag, n_answers=len(n), n_steps=int(off[-1]), K=K, seconds=round(time.time() - t0, 1)))
        print(time.strftime('%H:%M:%S'), tag, f'{time.time() - t0:.0f}s', flush=True)
    for name, rows in (('D1', rows1), ('D2', rows2), ('D3', rows3)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(HERE / f'{name}.csv', 'w', newline='') as fh:
            wr = csv.DictWriter(fh, keys); wr.writeheader(); wr.writerows(rows)
    (HERE / 'bundle' / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    print('DIAG DONE', len(manifest), 'cells')


if __name__ == '__main__':
    main()
