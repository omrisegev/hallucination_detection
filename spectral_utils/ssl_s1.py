"""S1 of the SSL / pseudo-label / residual localization plan (v1.1, section 7).

Three ways to build a pseudo-target for ONE linear student on the frozen 11-channel top5
step profile, plus the planned controls.  No correctness label, error index, error type or
correct/incorrect filter is accepted by any fit function here (see `fit_head`).

Notation (plan section 5): P[a,s,c] = frozen top5 readout (profiles_full[:, :, 0]);
Z[a,s,c] = per-answer, per-channel standardization of P (population SD; SD<=1e-8 -> 1).
Teachers (5.2): PB  q_loc[a,s]  = mean_c softmax_s(Z[a,:,c]);
               PRMB q_step[a,s] = mean_c sigmoid(Z[a,s,c]).
"""
import hashlib
from collections import Counter
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, expit, log_softmax

CHANNELS = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50',
            'energy_level', 'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0']
FAMILIES = {'G1': [0, 1, 2, 3, 4], 'G2': [5, 6], 'G3': [7, 8], 'G4': [9, 10]}
TOL = 8 * np.finfo(float).eps
RIDGE = 0.01
AGREE_PB = dict(v_min=.75, c_min=.10)
AGREE_PRM = dict(lo=.25, hi=.75, min_views=3)
SEED = 20260923


# ------------------------------------------------------------------ representation
def answer_z(P):
    """Z for one answer: (S, C) -> standardize each channel over the answer's steps."""
    P = np.asarray(P, float)
    mu = P.mean(0); sd = P.std(0)
    return (P - mu) / np.where(sd > 1e-8, sd, 1.)


def first_argmax(v):
    v = np.asarray(v, float)
    return int(np.flatnonzero(v >= v.max() - TOL)[0])


def unique_peak(v):
    """Index of the unique maximum, or None when the top is tied (tolerance TOL)."""
    v = np.asarray(v, float); top = np.flatnonzero(v >= v.max() - TOL)
    return int(top[0]) if len(top) == 1 else None


# ------------------------------------------------------------------ teachers
def teacher_pb(Z, channels=None):
    """q_loc: mean over channels of the per-channel softmax over steps (T=1). Sums to 1."""
    Z = np.asarray(Z, float)
    if channels is not None: Z = Z[:, channels]
    return softmax(Z, axis=0).mean(1)


def teacher_prm(Z, channels=None):
    """q_step: mean over channels of sigmoid(Z). Per-step local support in (0,1)."""
    Z = np.asarray(Z, float)
    if channels is not None: Z = Z[:, channels]
    return expit(Z).mean(1)


def leave_family_out_channels():
    return {g: [c for c in range(len(CHANNELS)) if c not in idx] for g, idx in FAMILIES.items()}


# ------------------------------------------------------------------ targets
def entropy_conf(q):
    """c = 1 - H(q)/log S; S == 1 -> 0."""
    q = np.asarray(q, float); S = len(q)
    if S == 1: return 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.sum(np.where(q > 0, q * np.log(q), 0.))
    return float(max(0., 1 - h / np.log(S)))


def targets_pb(Z, arm, rng=None):
    """Return (target vector over steps, answer weight multiplier, selected flag, info)."""
    S = Z.shape[0]
    q = teacher_pb(Z)
    if arm == 'P_HARD':
        t = np.zeros(S); t[first_argmax(q)] = 1.
        return t, 1., True, {}
    if arm in ('P_SOFT', 'P_SOFT_COVERAGE_MATCH', 'P_POSITION_LENGTH'):
        return q, 1., True, {}
    if arm == 'P_RANDOM':
        shift = int(rng.integers(1, S)) if S > 1 else 0
        return np.roll(q, shift), 1., True, {'shift': shift}
    if arm == 'P_AGREE':
        views = [teacher_pb(Z, ch) for ch in leave_family_out_channels().values()]
        qA = np.mean(views, 0)
        peaks = [unique_peak(v) for v in views]
        counted = Counter(p for p in peaks if p is not None)
        v = (max(counted.values()) / len(views)) if counted else 0.
        c = entropy_conf(qA)
        keep = bool(v >= AGREE_PB['v_min'] and c >= AGREE_PB['c_min'])
        return qA, float(v * c), keep, {'v': float(v), 'c': float(c), 'modal_peak': (counted.most_common(1)[0][0] if counted else -1)}
    raise ValueError(arm)


def targets_prm(Z, arm, rng=None):
    """Return (per-step target, per-step loss weight m, selected-step mask, info)."""
    S = Z.shape[0]
    q = teacher_prm(Z)
    ones = np.ones(S); allsel = np.ones(S, bool)
    if arm == 'P_HARD':
        return (q >= .5).astype(float), ones, allsel, {}
    if arm in ('P_SOFT', 'P_SOFT_COVERAGE_MATCH', 'P_POSITION_LENGTH'):
        return q, ones, allsel, {}
    if arm == 'P_RANDOM':
        return q[rng.permutation(S)], ones, allsel, {}
    if arm == 'P_AGREE':
        views = np.stack([teacher_prm(Z, ch) for ch in leave_family_out_channels().values()], 1)   # S x 4
        qA = views.mean(1)
        side = views >= .5; qside = (qA >= .5)[:, None]
        agree = (side == qside).sum(1) >= AGREE_PRM['min_views']
        sel = ((qA <= AGREE_PRM['lo']) | (qA >= AGREE_PRM['hi'])) & agree
        return qA, np.abs(2 * qA - 1), sel, {'selected': int(sel.sum()), 'total': S}
    raise ValueError(arm)


# ------------------------------------------------------------------ sample weights (plan 5.3)
def answer_weights(cells, groups, task):
    """Equal mass per cell (PB) -> per source group in cell -> per answer in group. PRMB: group -> answer.
    Returns weights summing to 1 over the given answers."""
    cells = np.asarray(cells); groups = np.asarray(groups); n = len(cells); w = np.zeros(n)
    if task == 'prm': cells = np.array(['prm'] * n)
    ucells = np.unique(cells)
    for c in ucells:
        ic = np.flatnonzero(cells == c); ug = np.unique(groups[ic])
        for g in ug:
            ig = ic[groups[ic] == g]
            w[ig] = 1. / len(ucells) / len(ug) / len(ig)
    return w / w.sum()


# ------------------------------------------------------------------ student
def _check_no_labels(kwargs):
    banned = {'labels', 'label', 'target_step', 'error_steps', 'y', 'first_error', 'correct', 'classification'}
    bad = banned & set(kwargs)
    if bad: raise ValueError(f'fit_head does not accept label-like inputs: {sorted(bad)}')


def fit_head(X_list, T_list, w_answers, task, M_list=None, ridge=RIDGE, fit_bias=None, maxiter=1000, gtol=1e-6, **kwargs):
    """Linear head u = X w + b, fitted by L-BFGS from zero.

    X_list: per-answer (S_a, d) feature rows; T_list: per-answer targets (PB: distribution over steps;
    PRMB: per-step targets in [0,1]); w_answers: answer weights (renormalized here over answers that
    contribute); M_list: PRMB per-step loss weights (None -> all ones).  PB fixes b = 0.
    """
    _check_no_labels(kwargs)
    d = X_list[0].shape[1]
    if fit_bias is None: fit_bias = task == 'prm'
    w_ans = np.asarray(w_answers, float).copy()
    if task == 'prm':
        M_list = [np.ones(len(t)) if M_list is None else np.asarray(M_list[i], float) for i, t in enumerate(T_list)]
        contributes = np.array([m.sum() > 0 for m in M_list])
    else:
        contributes = np.array([len(t) > 1 for t in T_list])          # a one-step answer has no PB loss (softmax over 1 step)
    w_ans[~contributes] = 0.
    if w_ans.sum() <= 0: raise ValueError('no contributing answers')
    w_ans = w_ans / w_ans.sum()

    def f_and_g(theta):
        w = theta[:d]; b = theta[d] if fit_bias else 0.
        loss = ridge * float(w @ w); g = np.zeros(len(theta)); g[:d] = 2 * ridge * w
        for i, (X, T) in enumerate(zip(X_list, T_list)):
            if w_ans[i] == 0: continue
            u = X @ w + b
            if task == 'prm':
                m = M_list[i]; Mi = m.sum(); p = expit(u)
                # BCE with logits, numerically safe
                bce = np.maximum(u, 0) - u * T + np.log1p(np.exp(-np.abs(u)))
                loss += w_ans[i] * float((m * bce).sum() / Mi)
                r = w_ans[i] * m * (p - T) / Mi
                g[:d] += X.T @ r
                if fit_bias: g[d] += r.sum()
            else:
                ls = log_softmax(u); loss += -w_ans[i] * float(T @ ls)
                r = w_ans[i] * (np.exp(ls) - T)
                g[:d] += X.T @ r
        return loss, g

    theta0 = np.zeros(d + (1 if fit_bias else 0))
    res = minimize(f_and_g, theta0, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 0., 'maxfun': 20 * maxiter})
    w = res.x[:d]; b = float(res.x[d]) if fit_bias else 0.
    grad_inf = float(np.abs(res.jac).max())
    return {'w': w, 'b': b, 'converged': bool(res.success and grad_inf <= gtol * 10), 'status': int(res.status), 'message': str(res.message),
            'iterations': int(res.nit), 'loss': float(res.fun), 'grad_inf_norm': grad_inf, 'contributing_answers': int(contributes.sum()), 'answers': len(T_list)}


def predict_head(X, model):
    return np.asarray(X, float) @ model['w'] + model['b']


# ------------------------------------------------------------------ position/length control features
def position_length_features(S, step_tokens):
    r = np.arange(S) / max(S - 1, 1)
    return np.column_stack([r, r ** 2, np.full(S, np.log1p(S)), np.log1p(np.asarray(step_tokens, float))])


def weighted_scaler(rows, weights):
    mu = np.average(rows, axis=0, weights=weights)
    sd = np.sqrt(np.average((rows - mu) ** 2, axis=0, weights=weights))
    return mu, np.where(sd > 1e-8, sd, 1e-8)


# ------------------------------------------------------------------ coverage matching (plan 7.3)
def depth_bin(S): return '1' if S == 1 else '2-5' if S <= 5 else '6-10' if S <= 10 else '11+'
def rel_bin(s, S): return ['[0,.2)', '[.2,.4)', '[.4,.6)', '[.6,.8)', '[.8,1]'][min(int(5 * s / max(S - 1, 1)), 4)] if S > 1 else '[0,.2)'
def coverage_key(uid, step_id): return hashlib.sha256(f'coverage|{uid}|{step_id}|{SEED}'.encode()).hexdigest()


def coverage_match(bins_kept, candidates):
    """bins_kept: Counter of bin -> count kept by P_AGREE. candidates: list of (bin, key, id).
    Returns the set of ids selected: per bin, the `count` candidates with the smallest SHA256 keys."""
    chosen = set(); by_bin = {}
    for b, key, i in candidates: by_bin.setdefault(b, []).append((key, i))
    for b, cnt in bins_kept.items():
        pool = sorted(by_bin.get(b, []))
        chosen.update(i for _, i in pool[:cnt])
    return chosen


# ------------------------------------------------------------------ metrics
def within_auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0: return np.nan
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
