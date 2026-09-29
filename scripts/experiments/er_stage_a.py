"""Stage A of expectation_realization_v1: are label-free SML estimates of each channel's sensitivity,
specificity and balanced accuracy accurate?  Frozen protocol: results/expectation_realization_v1/PROTOCOL.json.

Binary classifier i votes +1 (says "error step") or -1.  Parisi, Strino, Nadler & Kluger (PNAS 2014):
  psi_i = P(f_i = +1 | Y = +1)   sensitivity
  eta_i = P(f_i = -1 | Y = -1)   specificity
  pi_i  = (psi_i + eta_i) / 2    balanced accuracy,   b = P(Y=+1) - P(Y=-1)
Under conditional independence the off-diagonal covariance is q_ij = (1 - b^2)(2 pi_i - 1)(2 pi_j - 1),
so its rank-one completion gives t_i = sqrt(1 - b^2)(2 pi_i - 1) from the leading eigenvector.

Estimators: SML (rank-one completion; balanced accuracy only), the third-moment method of moments of Jaffe,
Nadler & Kluger (AISTATS 2015; `tensor_mom_estimate`), Dawid-Skene EM and the latent-group EM of the
cumulative-vote line (cvf_v2/em.py, imported unchanged).  Labels enter only `truth`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection')
CVF = MAIN / '.worktrees/cumulative-vote-fusion-v2/scripts/experiments'


def _cvf():
    if str(CVF) not in sys.path:
        sys.path.insert(0, str(CVF))
    import cvf_v2.core as core  # noqa: E402
    import cvf_v2.em as em  # noqa: E402
    return core, em


def binary_votes(V: np.ndarray, off: np.ndarray, frac: float = 0.2) -> tuple[np.ndarray, dict]:
    """+1 for the top ceil(frac n) steps of each answer and column (stable order: ties by position), else -1."""
    out = -np.ones(V.shape, np.int8); cells = ties = 0
    for a, b in zip(off[:-1], off[1:]):
        blk = V[a:b]; n = b - a; k = max(1, int(np.ceil(frac * n)))
        order = np.argsort(-blk, axis=0, kind='stable')
        m = np.zeros(blk.shape, bool); np.put_along_axis(m, order[:k], True, axis=0)
        out[a:b][m] = 1
        thr = -np.sort(-blk, axis=0)[k - 1]
        ties += int((((blk == thr).sum(0) > 1) & ((blk > thr).sum(0) + (blk == thr).sum(0) > k)).sum()); cells += blk.shape[1]
    return out, {'answer_columns': cells, 'boundary_tie_rate': ties / max(cells, 1)}


def truth(votes: np.ndarray, y: np.ndarray) -> dict:
    """y True = error step.  Evaluation only."""
    y = np.asarray(y, bool); pos = votes > 0
    psi = pos[y].mean(0); eta = (~pos[~y]).mean(0)
    return {'psi': psi, 'eta': eta, 'pi': (psi + eta) / 2, 'prevalence': float(y.mean())}


def rank_one_completion(Q: np.ndarray, iters: int = 2000, tol: float = 1e-13) -> tuple[float, np.ndarray]:
    """Leading eigenpair of Q with its diagonal replaced iteratively by lambda v^2 (off-diagonal kept)."""
    R = np.array(Q, float); off = ~np.eye(len(R), dtype=bool)
    np.fill_diagonal(R, np.abs(np.where(off, R, 0)).max(1))
    for _ in range(iters):
        w, U = np.linalg.eigh(R); lam, v = w[-1], U[:, -1]
        new = lam * v ** 2
        if np.max(np.abs(new - np.diag(R))) < tol:
            break
        np.fill_diagonal(R, new)
    return float(lam), v


def sml_estimate(votes: np.ndarray, anchor: int = 0, b_hat: float | None = None) -> dict:
    x = np.asarray(votes, float); Q = np.cov(x, rowvar=False, bias=True)
    lam, v = rank_one_completion(Q)
    if v[anchor] < 0:
        v = -v
    t = np.sqrt(max(lam, 0.0)) * v
    out = {'t': t, 'lambda': lam, 'eigvec': v}
    if b_hat is not None:
        out['pi'] = 0.5 + t / (2 * np.sqrt(max(1 - b_hat ** 2, 1e-12)))
    return out


def tensor_mom_estimate(votes: np.ndarray, anchor: int | None = None, eps: float = 1e-6) -> dict:
    """psi, eta, prevalence by the method of moments of Jaffe, Nadler & Kluger (AISTATS 2015): no EM, no b from elsewhere.

    Under conditional independence, with delta_i = 2 pi_i - 1, t_i = sqrt(1 - b^2) delta_i and z = f - mean(f):
      second moment (i != j):          E[z_i z_j]     = t_i t_j
      third moment (i, j, k distinct): E[z_i z_j z_k] = alpha t_i t_j t_k,   alpha = -2b / sqrt(1 - b^2)
    This is the paper's Algorithm 1 (Section 4.1; Lemma 4 = Eq. 13, Eq. 16-19): t is the paper's v, from the rank-one
    completion (as in `sml_estimate`); alpha is the least-squares fit of Eq. 19 over distinct index triples, summed in
    closed form by power sums (6 e_3 = p_1^3 - 3 p_1 p_2 + 2 p_3; ordered triples scale numerator and denominator by 6
    alike), so the m^3 tensor is never built.  Then b by Eq. 17 and, from E[f_i | Y] = mu_i + delta_i (Y - b),
      psi_i = (1 + mu_i + delta_i (1 - b)) / 2,   eta_i = (1 - mu_i + delta_i (1 + b)) / 2.
    The keep/drop side pi_i > 1/2 is sign(t_i), independent of b.  Orientation: t is signed so that t[anchor] > 0, or
    sum(t) > 0 without an anchor ('most classifiers beat random').  psi / eta outside [0, 1] are clipped and counted."""
    x = np.asarray(votes, float)
    if x.ndim != 2 or x.shape[1] < 3 or not np.isin(x, [-1, 1]).all():
        raise ValueError('votes must be n x m (m >= 3) in {-1, +1}')
    mu = x.mean(0); z = x - mu
    lam, v = rank_one_completion(np.cov(x, rowvar=False, bias=True))
    t = np.sqrt(max(lam, 0.0)) * v
    if (t[anchor] if anchor is not None else t.sum()) < 0:
        t = -t

    def e3(p1, p2, p3):  # sum over ordered distinct triples of a_i a_j a_k, from power sums
        return p1 ** 3 - 3 * p1 * p2 + 2 * p3
    s1, s2, s3 = z @ t, (z ** 2) @ (t ** 2), (z ** 3) @ (t ** 3)
    num = float(np.mean(e3(s1, s2, s3)))
    den = float(e3(np.sum(t ** 2), np.sum(t ** 4), np.sum(t ** 6)))
    if den <= 0:
        raise ArithmeticError('third-moment fit is degenerate (t has fewer than three non-zero entries)')
    alpha = num / den
    b = -alpha / np.sqrt(alpha ** 2 + 4)
    delta = t / np.sqrt(1 - b ** 2)
    psi_raw = (1 + mu + delta * (1 - b)) / 2; eta_raw = (1 - mu + delta * (1 + b)) / 2
    psi = np.clip(psi_raw, eps, 1 - eps); eta = np.clip(eta_raw, eps, 1 - eps)
    return {'psi': psi, 'eta': eta, 'pi': (psi + eta) / 2, 'prevalence': float((1 + b) / 2), 'b': float(b), 'alpha': float(alpha),
            't': t, 'lambda': lam, 'out_of_range': int(np.sum((psi_raw != psi) | (eta_raw != eta)))}


def em_estimate(votes: np.ndarray, kind: str, groups: np.ndarray | None = None, seed: int = 20260927) -> dict:
    """psi, eta, prevalence from cvf_v2 EM ('ds' or 'hem').  The error class is the latent class the
    model's orientation assigns to the all-(+1) vote pattern."""
    core, em = _cvf()
    x = np.asarray(votes, float); w = np.ones(len(x))
    spectral = core.fit_spectral(x, w, 'spectral')
    model = em.fit_em(x, w, kind, spectral, groups=None if kind == 'ds' else np.asarray(groups, int), seed=seed)
    if model.status != 'ok':
        raise ArithmeticError(f'EM status {model.status}')
    c = 1 if model.orientation > 0 else 0
    e = np.asarray(model.emissions, float)
    if kind == 'ds':
        psi = e[:, c]; fpr = e[:, 1 - c]
    else:
        g = np.asarray(model.groups, int); t = np.asarray(model.transition, float)
        psi = e[:, 0] * (1 - t[g, c]) + e[:, 1] * t[g, c]
        fpr = e[:, 0] * (1 - t[g, 1 - c]) + e[:, 1] * t[g, 1 - c]
    prev = model.prior if c == 1 else 1 - model.prior
    eta = 1 - fpr
    return {'psi': psi, 'eta': eta, 'pi': (psi + eta) / 2, 'prevalence': float(prev), 'converged': bool(model.diagnostics['converged']),
            'selected_start': int(model.diagnostics['selected_start']), 'boundary_emissions': int(model.diagnostics['boundary_emissions']),
            'orientation': float(model.orientation)}


def bar(est: dict, tru: dict, *, rho_min: float = 0.80, margin: float = 0.02, prev_tol: float | None = 0.05) -> dict:
    rho = float(spearmanr(est['pi'], tru['pi']).statistic)
    decided = np.abs(tru['pi'] - 0.5) > margin
    side_ok = np.sign(est['pi'][decided] - 0.5) == np.sign(tru['pi'][decided] - 0.5)
    res = {'spearman_pi': rho, 'rank_ok': bool(rho >= rho_min), 'side_checked': int(decided.sum()), 'side_wrong': int((~side_ok).sum()),
           'side_ok': bool(side_ok.all()), 'mae_pi': float(np.mean(np.abs(est['pi'] - tru['pi'])))}
    if 'psi' in est:
        res.update({'mae_psi': float(np.mean(np.abs(est['psi'] - tru['psi']))), 'mae_eta': float(np.mean(np.abs(est['eta'] - tru['eta'])))})
    if prev_tol is not None and 'prevalence' in est:
        res.update({'prevalence_error': float(est['prevalence'] - tru['prevalence']), 'prevalence_ok': bool(abs(est['prevalence'] - tru['prevalence']) <= prev_tol)})
    res['passes'] = bool(res['rank_ok'] and res['side_ok'] and res.get('prevalence_ok', True))
    return res
