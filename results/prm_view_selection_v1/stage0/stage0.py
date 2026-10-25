"""Stage 0 of PRM-anchored view selection (descriptive; pre-registration
docs/experiments/PRM_VIEW_SELECTION_STAGE0_PREREG_2026-10-25.md, amendment 1).

Per benchmark x PRM: 17 views = the PRM error score (STATUS_META reference scale) + N16 backbone telemetry, oriented
by the sign of their pooled correlation with the PRM, answer-z. A: per-view GAP binarization. B: label-free binary
accuracy (spectral, Dawid-Skene) and CI diagnostics. C: anchored two-factor model on the continuous views. D: fused
scores, descriptive only. Label-free estimates are computed and stored before any VALIDATION quantity.

    COLLECTION_DATA=/Users/osegev/Desktop/hd_collection_data python stage0.py
"""
import csv, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ORIENT = os.environ.get('ORIENT', 'raw')     # 'raw' (as pre-registered) | 'az' (amendment 2 sensitivity: answer-z scale)
POOL = os.environ.get('POOL', 'keep')        # 'keep' (as pre-registered) | 'all' (amendment 2: control answers included)
OUT = HERE if (ORIENT, POOL) == ('raw', 'keep') else HERE / f'sensitivity_orient-{ORIENT}_pool-{POOL}'
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'results/prm_boost_collection_v1'))
import evaluate_collection as E  # noqa: E402
from spectral_utils import binary_view_estimation as B  # noqa: E402

PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd']
BENCHES = ['prmbench', 'socratic']
META = json.loads((ROOT / 'results/prm_boost_collection_v1/tables/STATUS_META.json').read_text())
STATUS = {(r['benchmark'], r['prm'], r['row']): r for r in E.read_csv(ROOT / 'results/prm_boost_collection_v1/tables/STATUS_MAIN.csv')}


def prm_arm(bench, prm):
    prob = META[f'{bench}/{prm}']['prm_reference_scale'] == '1-reward'
    return ('PRM_Q80_PROB' if prob else 'PRM_Q80') if bench == 'prmbench' else ('PRMPROB' if prob else 'PRM')


def official(c, f):
    if c.bench == 'prmbench':
        return float(E.official_from_counts(E.counts_by_answer(c, f).sum(0), 'prmbench'))
    return 100 * float(E.official(c, f)[0])                       # Socratic on the same 0-100 scale as STATUS_MAIN


def within(c, s):
    return float(np.nanmean(E.per_answer_auc(c.err, s, c.off, c.keep if c.bench != 'prmbench' else None)))


def gap_votes(z, off, const):
    f = E.topk_flags(z, off, E.gap_counts(z, off))
    f[np.repeat(const, np.diff(off))] = False                     # a view constant within an answer abstains
    return f


def run_cell(bench, prm):
    c = E.load(bench, prm); arm = prm_arm(bench, prm)
    T = E.channels(c, 'N16'); names = ['PRM'] + [str(x) for x in np.load(
        E.DATA / ('prmbench' if bench == 'prmbench' else 'socratic') /
        (f'ABL5_CHANNELS_{prm}.npz' if bench == 'prmbench' else f'SCORES_{E.EXT_NAME.get(prm, prm)}_CHANNELS.npz'))['channel_names__N16']]
    raw = np.column_stack([c.arms[arm], T])
    if bench == 'prmbench':
        assert spearmanr(c.arms[arm], -c.extra['reward'])[0] > .999, 'PRM arm is not an error score'
    lens = np.repeat(np.diff(c.off), np.diff(c.off))
    pool = (c.keep if POOL == 'keep' else np.ones(len(lens), bool)) & (lens > 1)
    O = raw if ORIENT == 'raw' else np.column_stack([E.answer_z(raw[:, k], c.off) for k in range(raw.shape[1])])
    sign = np.sign([1.] + [np.corrcoef(O[pool, 0], O[pool, k])[0, 1] for k in range(1, raw.shape[1])])
    sign[sign == 0] = 1
    X = raw * sign
    Z = np.column_stack([E.answer_z(X[:, k], c.off) for k in range(X.shape[1])])
    sd = np.array([[X[a:b, k].std() for k in range(X.shape[1])] for a, b in zip(c.off[:-1], c.off[1:])])
    const = sd < 1e-8
    F = np.column_stack([gap_votes(Z[:, k], c.off, const[:, k]) for k in range(X.shape[1])])
    V = np.where(F, 1., -1.)

    # ---------------- label-free estimates (written before any validation quantity)
    sp = B.spectral_binary(V[pool]); ds = B.dawid_skene_binary(V[pool])
    grp = ['P'] + ['T'] * (V.shape[1] - 1)
    mis = B.rank_one_misfit(sp['S'], sp['u'], grp); tci = B.tci_residuals(sp['S'], sp['T'])
    Cz = np.cov(Z[pool].T)
    u, lam, psi, fa = B.anchored_two_factor(Cz, 0)
    tri, tri_iqr = B.triad_lambda(Cz, 0)
    I_prm = u[0] ** 2 / psi[0]; I_all = B.factor_information(u, lam, psi)
    I_drop = np.array([B.factor_information(u, lam, psi, drop=k) for k in range(len(u))])
    mis = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in mis.items()}
    lf = dict(sign=sign.tolist(), abstain_answers=const.sum(0).tolist(), spectral_prev=sp['prevalence'],
              ds_prev=ds['prevalence'], misfit=mis, fa_misfit=fa['rel_misfit'], heywood=fa['heywood'],
              I_prm=I_prm, I_all=I_all, I_ratio=I_all / I_prm)

    # ---------------- validation (labels)
    y = c.err[pool]
    true_psi = (V[pool][y] > 0).mean(0); true_eta = (V[pool][~y] < 0).mean(0); true_bal = (true_psi + true_eta) / 2
    corr_err = np.corrcoef(V[pool][y].T); corr_ok = np.corrcoef(V[pool][~y].T); off = ~np.eye(len(u), dtype=bool)
    rows = []
    for k, nm in enumerate(names):
        rows.append(dict(benchmark=bench, prm=prm, view=nm, sign=int(sign[k]), abstain_answers=int(const[:, k].sum()),
                         flag_rate=float(F[pool, k].mean()),
                         est_spec_psi=sp['psi'][k], est_spec_eta=sp['eta'][k], est_spec_bal=sp['bal_acc'][k],
                         est_ds_psi=ds['psi'][k], est_ds_eta=ds['eta'][k], est_ds_bal=ds['bal_acc'][k],
                         misfit_view=mis['per_view'][k], tci_resid=tci[k],
                         fa_u=u[k], fa_lambda=lam[k], fa_psi=psi[k], triad_l2=tri[k], triad_l2_iqr=tri_iqr[k],
                         I_drop_ratio=I_drop[k] / I_all,
                         VAL_true_psi=true_psi[k], VAL_true_eta=true_eta[k], VAL_true_bal=true_bal[k],
                         VAL_prmscore_gap=official(c, F[:, k]), VAL_within_auc_cont=within(c, Z[:, k])))
    val = dict(true_prev=float(y.mean()),
               spearman_spec_vs_true=float(spearmanr(sp['bal_acc'], true_bal)[0]),
               spearman_ds_vs_true=float(spearmanr(ds['bal_acc'], true_bal)[0]),
               mae_spec_bal=float(np.abs(sp['bal_acc'] - true_bal).mean()), mae_ds_bal=float(np.abs(ds['bal_acc'] - true_bal).mean()),
               mean_abs_corr_err=float(np.abs(corr_err[off]).mean()), mean_abs_corr_ok=float(np.abs(corr_ok[off]).mean()))

    # ---------------- two-factor scores (validation) and descriptive fused scores
    S = B.bartlett_scores(Z, u, lam, psi)
    e_s, n_s = S[:, 0], S[:, 1]
    yv = c.err[pool].astype(float); A = np.column_stack([np.ones(pool.sum()), e_s[pool]])
    r_y = yv - A @ np.linalg.lstsq(A, yv, rcond=None)[0]; r_n = n_s[pool] - A @ np.linalg.lstsq(A, n_s[pool], rcond=None)[0]
    val.update(within_err_factor=within(c, e_s), within_nuis_factor=within(c, n_s), within_prm=within(c, Z[:, 0]),
               pooled_err_factor=float(E.auc(c.err[c.keep], e_s[c.keep])), pooled_nuis_factor=float(E.auc(c.err[c.keep], n_s[c.keep])),
               partial_r_nuis=float(np.corrcoef(r_y, r_n)[0, 1]))
    pos = V > 0
    ds_lo = np.where(pos, np.log(ds['psi'] / (1 - ds['eta'])), np.log((1 - ds['psi']) / ds['eta'])).sum(1)
    pe, ee = np.clip(sp['psi'], 1e-4, 1 - 1e-4), np.clip(sp['eta'], 1e-4, 1 - 1e-4)
    sp_lo = np.where(pos, np.log(pe / (1 - ee)), np.log((1 - pe) / ee)).sum(1)
    fused = []
    for nm, s in [('DS_BINARY_17', ds_lo), ('SPECTRAL_BINARY_17', sp_lo), ('TWOFACTOR_ERRSCORE_17', e_s)]:
        c.arms[nm] = s
    for nm in ['DS_BINARY_17', 'SPECTRAL_BINARY_17', 'TWOFACTOR_ERRSCORE_17', arm, 'CHAIN_N16', 'TELE_N16']:
        r = E.measure(c, nm, 'GAP', fast=True)[0]
        fused.append(dict(benchmark=bench, prm=prm, arm=nm if nm != arm else 'PRM (same rule)', official_gap=r['official'] * (1 if bench == 'prmbench' else 100),
                          within_auc=r['within_auc'], pooled_auroc=r['pooled_auroc'], flag_rate=r['flag_rate']))
    # parity: the PRM view's own GAP votes reproduce the frozen PRM-same-rule PRMScore
    ref = float(STATUS[(bench, prm, 'PRM same rule (better scale under GAP)')]['GAP']) if (bench, prm, 'PRM same rule (better scale under GAP)') in STATUS else None
    # (parity uses the frozen rule without the abstention of constant answers, which the frozen evaluator does not have)
    got = official(c, E.topk_flags(Z[:, 0], c.off, E.gap_counts(Z[:, 0], c.off)))
    parity = dict(prm_view_prmscore_no_abstain=got, prm_view_prmscore=rows[0]['VAL_prmscore_gap'], status_ref=ref,
                  ok=(ref is not None and abs(got - ref) < 1e-6))
    diag = dict(benchmark=bench, prm=prm, n_pool_steps=int(pool.sum()), n_answers=len(c.ids), label_free=lf, validation=val, parity=parity)
    return rows, fused, diag


def main():
    OUT.mkdir(exist_ok=True); t0 = time.time(); rows, fused, diags = [], [], []
    for bench in BENCHES:
        for prm in PRMS:
            t = time.time(); r, f, d = run_cell(bench, prm)
            rows += r; fused += f; diags.append(d)
            print(f'{bench:9s} {prm:14s} {time.time() - t:6.1f}s parity={d["parity"]["ok"]} I_ratio={d["label_free"]["I_ratio"]:.3f} '
                  f'rho_ds={d["validation"]["spearman_ds_vs_true"]:.2f} rho_spec={d["validation"]["spearman_spec_vs_true"]:.2f}', flush=True)
    E.write_csv(OUT / 'BINARY_VIEWS.csv', rows)
    E.write_csv(OUT / 'FUSED_DESCRIPTIVE.csv', fused)
    flat = []
    for d in diags:
        lf, v = d['label_free'], d['validation']
        flat.append(dict(benchmark=d['benchmark'], prm=d['prm'], n_pool_steps=d['n_pool_steps'],
                         spectral_prev=lf['spectral_prev'], ds_prev=lf['ds_prev'], true_prev=v['true_prev'],
                         misfit_overall=lf['misfit']['overall'], misfit_P_T=lf['misfit'].get('P-T'), misfit_T_T=lf['misfit'].get('T-T'),
                         fa_misfit=lf['fa_misfit'], heywood=lf['heywood'], I_prm=lf['I_prm'], I_all=lf['I_all'], I_ratio=lf['I_ratio'],
                         **{f'VAL_{k}': x for k, x in v.items() if k != 'true_prev'}, parity_ok=d['parity']['ok']))
    E.write_csv(OUT / 'CELL_SUMMARY.csv', flat)
    json.dump(dict(diags=diags, runtime_s=time.time() - t0), open(OUT / 'RUN_META.json', 'w'), indent=1, default=float)
    print(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
