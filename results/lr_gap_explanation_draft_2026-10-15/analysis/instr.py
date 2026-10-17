"""Instrument-agreement test for within-cluster contrasts (label-free; labels only for evaluation).

Per outer fold, on the training fold, with the saved L-SML clusters:
  instruments  : every OTHER cluster's L-SML virtual score L_h (standardized)
  c_ih         : Cov(z_i, L_h)  ≈ mu_i * kappa_h  if cross-cluster noise were absent
  kappa_g      : per cluster g, leading right singular vector of C_g = [c_ih] (oriented to agree with L-SML)
  pair (a,b)   : d_h = (c_bh - r_ab c_ah) / kappa_h  estimates mu_b - r_ab mu_a (sign of b's partial weight vs a)
                 one estimate per instrument h; PASS if |mean_h d_h| > Z * SE_h(d_h)  (instruments agree)
  score        : within each cluster, w_g = (S_g^pass + lam I)^-1 mu_hat_g, where S_g^pass keeps only the
                 off-diagonal correlations of PASSING pairs (others treated as independent = voting);
                 cluster scores are rescaled to the L-SML virtual score's scale and fused with L-SML's cross weights.
Evaluation: sign accuracy of passed decisions vs the oracle sign of (mu_b - r mu_a); OOF within-answer AUC.
"""
import json, sys
import numpy as np
from numpy.linalg import solve, svd, eigvalsh
sys.path.insert(0, '/Users/osegev/Desktop/hallucination_detection/results/lr_gap_explanation_draft_2026-10-15/analysis')
from decomp import within, ROOT

Z_PASS = 2.0
p = np.load(f'{ROOT}/inputs/population.npz'); off = p['offsets'].astype(int); y = p['y'].astype(int)
sf = np.repeat(p['fold'].astype(int), np.diff(off))


def pd_solve(S, v, lam):
    lo = eigvalsh(S).min()
    extra = max(0., -lo + 1e-3)
    return solve(S + (lam + extra) * np.eye(len(v)), v)


def run(bank):
    X = np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); m = X.shape[1]
    arms = ['LSML', 'GATED_lam0.1', 'GATED_lam1', 'ALLPAIRS_lam0.1', 'ALLPAIRS_lam1', 'ORACLE_GATED', 'ORACLE_ALLPAIRS']
    sc = {k: np.full(len(y), np.nan) for k in arms}
    acc = {'all': [0, 0], 'pass': [0, 0], 'pass_neg': [0, 0], 'hi_r_all': [0, 0], 'hi_r_pass': [0, 0]}
    n_pairs = n_pass = 0
    for o in range(5):
        L = json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
        G = np.asarray(L['diagnostic']['group']); wl = np.asarray(L['weight']); tr, te = sf != o, sf == o
        mu0, sd = X[tr].mean(0), X[tr].std(0); Ztr, Zte = (X[tr] - mu0) / sd, (X[te] - mu0) / sd
        ws = wl * sd; groups = np.unique(G)
        sc['LSML'][te] = X[te] @ wl
        V = np.column_stack([Ztr[:, G == g] @ ws[G == g] for g in groups])
        Vs = (V - V.mean(0)) / V.std(0)
        mu_true = Ztr[y[tr] == 1].mean(0) - Ztr[y[tr] == 0].mean(0)
        R = np.corrcoef(Ztr, rowvar=False)
        fused = {k: np.zeros(te.sum()) for k in arms if k != 'LSML'}
        for gi, g in enumerate(groups):
            idx = np.flatnonzero(G == g); xi_old_te = Zte[:, idx] @ ws[idx]; xi_old_sd = (Ztr[:, idx] @ ws[idx]).std()
            others = [h for h in range(len(groups)) if h != gi]
            if len(idx) < 2 or len(others) < 3:
                for k in fused: fused[k] += xi_old_te  # singleton / too few instruments: keep L-SML
                continue
            C = Ztr[:, idx].T @ Vs[:, others] / tr.sum()                 # |g| x H
            U, s, Wt = svd(C, full_matrices=False); kap = Wt[0] * s[0]; mh = U[:, 0]
            if mh @ ws[idx] < 0: mh, kap = -mh, -kap                        # label-free orientation (agree with L-SML)
            mh = C @ kap / (kap @ kap)
            Sg = R[np.ix_(idx, idx)]; Mpass = np.eye(len(idx), dtype=bool)
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    r = Sg[a, b]; n_pairs += 1
                    ok = np.abs(kap) > 0.2 * np.abs(kap).max()
                    for (u, v) in ((a, b), (b, a)):
                        d = (C[v, ok] - r * C[u, ok]) / kap[ok]
                        mean, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.inf
                        passed = len(d) >= 3 and abs(mean) > Z_PASS * se
                        truth = mu_true[idx[v]] - r * mu_true[idx[u]]
                        hit = int(np.sign(mean) == np.sign(truth))
                        acc['all'][0] += hit; acc['all'][1] += 1
                        if abs(r) >= .8: acc['hi_r_all'][0] += hit; acc['hi_r_all'][1] += 1
                        if passed:
                            acc['pass'][0] += hit; acc['pass'][1] += 1
                            if abs(r) >= .8: acc['hi_r_pass'][0] += hit; acc['hi_r_pass'][1] += 1
                            if mean < 0: acc['pass_neg'][0] += hit; acc['pass_neg'][1] += 1
                            Mpass[a, b] = Mpass[b, a] = True
            n_pass += int((Mpass.sum() - len(idx)) / 2)
            Sp = np.where(Mpass, Sg, 0.)
            mt = mu_true[idx]
            for k, (S_, mu_, lam) in {'GATED_lam0.1': (Sp, mh, .1), 'GATED_lam1': (Sp, mh, 1.), 'ALLPAIRS_lam0.1': (Sg, mh, .1),
                                      'ALLPAIRS_lam1': (Sg, mh, 1.), 'ORACLE_GATED': (Sp, mt, .1), 'ORACLE_ALLPAIRS': (Sg, mt, .1)}.items():
                w = pd_solve(S_, mu_, lam); xi_tr = Ztr[:, idx] @ w
                sgn = 1. if np.corrcoef(xi_tr, Ztr[:, idx] @ ws[idx])[0, 1] >= 0 else -1.
                fused[k] += sgn * (Zte[:, idx] @ w) * xi_old_sd / xi_tr.std()
        for k, v in fused.items(): sc[k][te] = v
    res = {k: round(within(v, off, y), 4) for k, v in sc.items()}
    accs = {k: (f'{100 * a / b:.0f}% of {b}' if b else 'n/a') for k, (a, b) in acc.items()}
    print(bank, f'within-cluster pairs {n_pairs // 5}/fold, passing {n_pass / 5:.1f}/fold', flush=True)
    print('  sign accuracy (directed pairs):', accs, flush=True)
    print('  AUC:', res, flush=True)


if __name__ == '__main__':
    for b in sys.argv[1:]:
        run(b)
