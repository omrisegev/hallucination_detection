"""Independent verifier: parity inputs (rebuilt vs AIRCC bank_v3, rewards vs stored PRM) and Track T T-MEAN.

Own code. Imports nothing from results/answer_mixture_gate_v1. Optionally (flag --lsml) re-runs the frozen package
scorer spectral_utils.fusion_utils.lsml_continuous_guarded on an answer-z N16 bank built here, and compares with the
stored AIRCC SCORES_CHAIN_qwen_prm.npz (score__TELE_N16, score__CHAIN_N16).
    .venv/bin/python -I results/answer_mixture_gate_v1/verification/verify_t.py [--lsml]
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

REPO = Path('/Users/osegev/Desktop/hallucination_detection')
DATA = Path('/Users/osegev/Desktop/hd_collection_data')
OUT = REPO / 'results/answer_mixture_gate_v1/verification'
REBUILT = Path('/private/tmp/claude-501/-Users-osegev-Desktop-hallucination-detection/98d49308-5ca3-48f7-a452-63da0bb7d2a1/scratchpad/tt/bank_rebuilt')
V3 = DATA / 'processbench_raw/qwen_prm/bank_v3'
DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
SEED = 20261029


def zfull(v):
    return (v - v.mean()) / v.std()


def az(x, off, active=None):
    """answer-wise z (ddof 0), constants -> 0; optional active mask (inactive -> 0)."""
    x = np.asarray(x, float); out = np.zeros_like(x)
    act = np.ones(x.shape, bool) if active is None else active
    for a, b in zip(off[:-1], off[1:]):
        for j in range(x.shape[1]):
            m = act[a:b, j]; v = x[a:b, j][m]
            if len(v) and v.std() > 1e-12:
                blk = out[a:b, j]; blk[m] = (v - v.mean()) / v.std(); out[a:b, j] = blk
    return out


def main(do_lsml):
    rep = {}
    names16 = json.loads((REPO / 'results/monday_nested_banks_v2/BANKS.json').read_text())['N16']
    Fr, F3 = np.load(REBUILT / 'FEATURES.npz'), np.load(V3 / 'FEATURES.npz')
    Rr, R3 = np.load(REBUILT / 'REWARDS.npz'), np.load(V3 / 'REWARDS.npz')
    rep['rebuilt_vs_v3'] = {
        'values_max_abs_diff_all56': float(np.max(np.abs(Fr['values'] - F3['values']))),
        'names_equal': bool(np.array_equal(Fr['names'], F3['names'])),
        **{k + '_equal': bool(np.array_equal(Fr[k], F3[k])) for k in ('digit_active', 'offsets', 'uids', 'nonempty', 'nonempty_lengths')},
        'rewards_max_abs_diff': float(np.max(np.abs(Rr['rewards'] - R3['rewards']))),
        'rewards_uids_equal': bool(np.array_equal(Rr['uids'], R3['uids'])),
        'rewards_keys': list(Rr.files)}
    nm = [str(x) for x in Fr['names']]; cols = [nm.index(c) for c in names16]
    rep['rebuilt_vs_v3']['values_max_abs_diff_N16'] = float(np.max(np.abs(Fr['values'][:, cols] - F3['values'][:, cols])))
    rep['rebuilt_vs_v3']['values_max_rel_diff_N16'] = float(np.max(np.abs(Fr['values'][:, cols] - F3['values'][:, cols]) /
                                                              (np.abs(F3['values'][:, cols]) + 1e-12)))

    S = np.load(DATA / 'processbench_scores/SCORES_CHAIN_qwen_prm.npz')
    off = Fr['offsets'].astype(np.int64); uids = Fr['uids'].astype(str); nst = np.diff(off)
    assert np.array_equal(S['uids'].astype(str), uids) and np.array_equal(S['nonempty_lengths'], Fr['nonempty_lengths'])
    r = np.clip(1 - np.asarray(Rr['rewards'], float), 1e-6, 1 - 1e-6); prm = np.log(r / (1 - r))
    rep['stored_PRM_vs_rebuilt_reward_risk_max_abs'] = float(np.max(np.abs(prm - S['score__PRM'])))
    Sp = np.load(DATA / 'processbench_scores/SCORES_qwen_prm.npz')
    rep['stored_PRM_CHAINfile_vs_SCORESfile_max_abs'] = float(np.max(np.abs(Sp['score__PRM'] - S['score__PRM'])))

    raw = Fr['values'][:, cols].astype(float); act = np.ones(raw.shape, bool)
    da = Fr['digit_active'].astype(bool)
    for j, c in enumerate(names16):
        if c in DIG: act[:, j] = da[:, DIG.index(c)]
    rep['digit_active_counts'] = {c: int(da[:, DIG.index(c)].sum()) for c in DIG}
    rep['digit_inactive_values_all_zero'] = {c: bool(np.all(raw[~act[:, j], j] == 0)) for j, c in enumerate(names16) if c in DIG}

    # ---- optional independent L-SML parity (frozen package scorer, own bank construction)
    if do_lsml:
        sys.path.insert(0, str(REPO))
        from spectral_utils.fusion_utils import lsml_continuous_guarded as L
        V = az(raw, off, act)
        s, d = L(*[V[:, j] for j in range(V.shape[1])], groups=None); s = np.asarray(s, float)
        if np.corrcoef(s, V.mean(1))[0, 1] < 0: s = -s
        vg = np.asarray(d['virtual_classifiers'], float)
        VC = np.column_stack([az(prm[:, None], off)[:, 0]] + [az(vg[:, [g]], off)[:, 0] for g in range(vg.shape[1])])
        c, dc = L(*[VC[:, j] for j in range(VC.shape[1])], groups=np.arange(VC.shape[1])); c = np.asarray(c, float)
        if np.corrcoef(c, VC[:, 0])[0, 1] < 0: c = -c
        rep['independent_lsml'] = {'TELE_N16_max_abs': float(np.max(np.abs(s - S['score__TELE_N16']))),
                                   'CHAIN_N16_max_abs': float(np.max(np.abs(c - S['score__CHAIN_N16']))),
                                   'tele_groups': np.asarray(d['c']).tolist(), 'K': int(d['K'])}
        print('independent L-SML', rep['independent_lsml'], flush=True)

    # ---- T-MEAN (global z over all steps; digit channels over active steps, inactive 0), from both banks
    T = np.load(REPO / 'results/answer_mixture_gate_v1/trackT/PB_QWEN_ANSWER_TOKENS.npz')
    tk = dict(zip(T['uids'].astype(str), T['answer_tokens'].astype(float)))
    covs = [np.log(nst.astype(float)), np.log(np.array([tk[u] for u in uids]))]
    G = {g['uid']: g for g in json.loads((REPO / 'results/prm_boost_processbench_v1/local/processbench_gold.json').read_text())}
    ref = [r for r in csv.DictReader(open(REPO / 'results/answer_mixture_gate_v1/trackT/TRACK_T.csv'))
           if r['score'] == 'T-MEAN' and r['primary'] == 'True'][0]
    for tag, F in (('rebuilt', Fr), ('bank_v3', F3)):
        X = F['values'][:, cols].astype(float); Z = np.zeros_like(X)
        for j in range(16):
            m = act[:, j]; Z[m, j] = zfull(X[m, j])
        step = Z.mean(1)
        s_a = np.array([step[a:b].max() for a, b in zip(off[:-1], off[1:])])
        A = np.column_stack([np.ones(len(s_a))] + covs); beta = np.linalg.solve(A.T @ A, A.T @ s_a); res = s_a - A @ beta
        x = res[:, None]
        g2 = GaussianMixture(2, covariance_type='full', n_init=10, random_state=SEED).fit(x)
        g1 = GaussianMixture(1, covariance_type='full', n_init=10, random_state=SEED).fit(x)
        k = int(np.argmax(g2.means_.ravel())); clean = g2.predict_proba(x)[:, k] < .5
        y = np.array([int(G[u]['label']) != -1 for u in uids])      # labels only after the fit
        out = {'n': len(y), 'pi_true': float(y.mean()), 'pi_hat': float(g2.weights_[k]),
               'dbic': float(g2.bic(x) - g1.bic(x)), 'clean_closed': float(clean[~y].mean()),
               'err_kept': float((~clean[y]).mean()), 'auroc': float(roc_auc_score(y, s_a)),
               'per_answer_mean_sd': float(np.array([step[a:b].mean() for a, b in zip(off[:-1], off[1:])]).std())}
        out['ref'] = {'pi_true': float(ref['pi_true']), 'pi_hat': float(ref['pi_hat']), 'dbic': float(ref['bic2_minus_bic1']),
                      'clean_closed': float(ref['clean_closed']), 'err_kept': float(ref['err_kept']),
                      'auroc': float(ref['auroc_s_a']), 'per_answer_mean_sd': float(ref['per_answer_mean_sd'])}
        out['abs_diff'] = {k2: abs(out[k2] - out['ref'][k2]) for k2 in out['ref']}
        rep['T_MEAN_' + tag] = out
        print('T-MEAN', tag, json.dumps(out), flush=True)
    (OUT / 'verify_t_summary.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith('T_MEAN')}, indent=1))


if __name__ == '__main__':
    main('--lsml' in sys.argv)
