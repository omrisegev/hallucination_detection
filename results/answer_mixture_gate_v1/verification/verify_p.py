"""Independent verifier, Track P (answer-mixture gate pre-registration 2026-10-29).

Own code; imports nothing from results/answer_mixture_gate_v1 or results/prm_boost_collection_v1.
Reads raw score files and gold files directly. Labels are loaded in a separate function and used only
after the mixture fit of each cell (they are also used for the population checks).
    .venv/bin/python -I results/answer_mixture_gate_v1/verification/verify_p.py
"""
import csv, json
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

REPO = Path('/Users/osegev/Desktop/hallucination_detection')
DATA = Path('/Users/osegev/Desktop/hd_collection_data')
OUT = REPO / 'results/answer_mixture_gate_v1/verification'
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
FILE = {'qwen_prm_7b': 'qwen_prm'}
SEED = 20261029


def risk_from_reward(r):
    r = np.clip(np.asarray(r, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log1p(-r) - np.log(r)


def per_answer(values, lengths, fn):
    out, a = [], 0
    for L in lengths:
        out.append(fn(values[a:a + L])); a += L
    assert a == len(values)
    return np.array(out, dtype=np.float64)


def ols_resid(y, covs):
    X = np.column_stack([np.ones_like(y)] + covs)
    coef = np.linalg.solve(X.T @ X, X.T @ y)          # normal equations (different route than lstsq)
    return y - X @ coef, coef


def gmm(res):
    X = res.reshape(-1, 1)
    m2 = GaussianMixture(n_components=2, covariance_type='full', n_init=10, random_state=SEED).fit(X)
    m1 = GaussianMixture(n_components=1, covariance_type='full', n_init=10, random_state=SEED).fit(X)
    k = int(np.argmax(m2.means_.ravel()))
    post = m2.predict_proba(X)[:, k]
    return dict(pi_hat=float(m2.weights_[k]), post=post, dbic=float(m2.bic(X) - m1.bic(X)),
                means=m2.means_.ravel().tolist(), weights=m2.weights_.tolist())


# ------------------------------------------------------------------ label-free inputs
def prmbench_inputs(prm):
    S = np.load(DATA / 'prmbench' / f'ABL5_SCORES_{prm}.npz')
    ids = S['row_ids'].astype(str); lens = np.diff(S['offsets'])
    risk = risk_from_reward(S['reward'])
    tok = per_answer(np.expm1(S['log_step_tokens']), lens, np.sum)
    covs = [np.log(lens.astype(float)), np.log(tok)]
    return ids, lens, risk, covs


def h2v_common():
    sets = [set(np.load(DATA / 'h2v' / f'SCORES_{n}.npz')['uids'].astype(str))
            for n in ('qwen_prm', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd')]
    return set.intersection(*sets)


def h2v_inputs(prm, common):
    S = np.load(DATA / 'h2v' / f'SCORES_{FILE.get(prm, prm)}.npz')
    uids = S['uids'].astype(str); L = S['nonempty_lengths'].astype(int); ne = S['nonempty'].astype(bool)
    # arrays are over all listed steps; keep non-empty steps of the common answers
    assert len(ne) == L.sum() == len(S['score__PRM'])
    st = np.r_[0, np.cumsum(L)]
    ids, lens, vals = [], [], []
    for j, u in enumerate(uids):
        if u not in common: continue
        m = ne[st[j]:st[j + 1]]
        ids.append(u); lens.append(int(m.sum())); vals.append(S['score__PRM'][st[j]:st[j + 1]][m])
    lens = np.array(lens)
    assert (lens > 0).all()
    return np.array(ids), lens, np.concatenate(vals), [np.log(lens.astype(float))], int((~ne).sum())


def pb_inputs(prm):
    S = np.load(DATA / 'processbench_scores' / f'SCORES_{FILE.get(prm, prm)}.npz')
    uids = S['uids'].astype(str); L = S['nonempty_lengths'].astype(int); ne = S['nonempty'].astype(bool)
    assert ne.all() and len(S['score__PRM']) == L.sum()
    covs = [np.log(L.astype(float))]
    if prm == 'qwen_prm_7b':
        T = np.load(REPO / 'results/answer_mixture_gate_v1/trackT/PB_QWEN_ANSWER_TOKENS.npz')
        d = dict(zip(T['uids'].astype(str), T['answer_tokens'].astype(float)))
        dn = dict(zip(T['uids'].astype(str), T['n_nonempty'].astype(int)))
        assert all(dn[u] == l for u, l in zip(uids, L))
        covs.append(np.log(np.array([d[u] for u in uids])))
    return uids, L, S['score__PRM'].astype(np.float64), covs


# ------------------------------------------------------------------ labels (scoring only)
def labels(bench, ids, lens):
    if bench == 'prmbench':
        M = {r['idx']: r for r in json.loads((REPO / 'results/advisor_completion_2026-10-11/inputs/prmbench_meta.json').read_text())['rows']}
        y = []
        for i, l in zip(ids, lens):
            r = M[i]; assert int(r['n_steps']) == l
            y.append(any(1 <= int(e) <= l for e in r['error_steps']))
        return np.array(y)
    if bench == 'hard2verify':
        G = {g['uid']: g for g in json.loads((REPO / 'results/remote_claims_replay_2026-10-15/inputs/hard2verify_gold.json').read_text())}
        y = []
        for u, l in zip(ids, lens):
            c, inc = G[u]['correct'], G[u]['include']; assert len(c) == len(inc) == l
            y.append(any(ii and not cc for cc, ii in zip(c, inc)))
        return np.array(y)
    G = {g['uid']: g for g in json.loads((REPO / 'results/prm_boost_processbench_v1/local/processbench_gold.json').read_text())}
    assert all(int(G[u]['n_steps']) == l for u, l in zip(ids, lens))
    return np.array([int(G[u]['label']) != -1 for u in ids])


def main():
    ref = {(r['bench'], r['prm']): r for r in csv.DictReader(open(REPO / 'results/answer_mixture_gate_v1/TRACK_P.csv'))
           if r['primary'] == 'True'}
    common = h2v_common()
    rows, extra = [], {'h2v_common_n': len(common)}
    for bench in ('prmbench', 'hard2verify', 'processbench'):
        for prm in PRMS:
            if bench == 'prmbench': ids, lens, risk, covs = prmbench_inputs(prm)
            elif bench == 'hard2verify':
                ids, lens, risk, covs, n_empty = h2v_inputs(prm, common); extra[f'h2v_{prm}_empty_steps_in_file'] = n_empty
            else: ids, lens, risk, covs = pb_inputs(prm)
            s_a = per_answer(risk, lens, np.max)
            res, coef = ols_resid(s_a, covs)
            f = gmm(res)                                              # fit finished before labels are read
            y = labels(bench, ids, lens)
            clean = f['post'] < .5
            r0 = ref[(bench, prm)]
            row = dict(bench=bench, prm=prm, n=len(ids), n_clean=int((~y).sum()), pi_true=float(y.mean()),
                       auroc=float(roc_auc_score(y, s_a)), pi_hat=f['pi_hat'], dbic=f['dbic'],
                       clean_closed=float(clean[~y].mean()), err_kept=float((~clean[y]).mean()),
                       ref_n=int(r0['n']), ref_pi_true=float(r0['pi_true']), ref_auroc=float(r0['auroc_s_a']),
                       ref_pi_hat=float(r0['pi_hat']), ref_dbic=float(r0['bic2_minus_bic1']),
                       ref_clean_closed=float(r0['clean_closed']), ref_err_kept=float(r0['err_kept']),
                       ref_covariates=r0['covariates'], n_covs=len(covs))
            for k in ('pi_true', 'auroc', 'pi_hat', 'dbic', 'clean_closed', 'err_kept'):
                row['d_' + k] = row[k] - row['ref_' + k]
            rows.append(row)
            print(f"{bench:12s} {prm:12s} n={row['n']} pi={row['pi_true']:.4f} auc={row['auroc']:.4f} pi_hat={row['pi_hat']:.4f} "
                  f"(ref {row['ref_pi_hat']:.4f}) dBIC={row['dbic']:.2f} (ref {row['ref_dbic']:.2f}) cc={row['clean_closed']:.4f} "
                  f"ek={row['err_kept']:.4f} | d_auc={row['d_auroc']:.1e} d_pi={row['d_pi_true']:.1e} d_pihat={row['d_pi_hat']:.1e} "
                  f"d_cc={row['d_clean_closed']:.1e} d_ek={row['d_err_kept']:.1e}", flush=True)
    with open(OUT / 'VERIFY_TRACK_P.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {k: float(max(abs(r['d_' + k]) for r in rows)) for k in ('pi_true', 'auroc', 'pi_hat', 'dbic', 'clean_closed', 'err_kept')}
    summ['n_match'] = all(r['n'] == r['ref_n'] for r in rows)
    summ['bic_sign_match'] = all((r['dbic'] < 0) == (r['ref_dbic'] < 0) for r in rows)
    summ.update(extra)
    pops = {}
    for r in rows: pops.setdefault(r['bench'], set()).add((r['n'], round(r['pi_true'], 6)))
    summ['populations'] = {b: sorted(v) for b, v in pops.items()}
    (OUT / 'verify_p_summary.json').write_text(json.dumps(summ, indent=1, default=list))
    print(json.dumps(summ, indent=1, default=list))


if __name__ == '__main__':
    main()
