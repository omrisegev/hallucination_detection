"""Label-free latent-vote models and answer error-count decisions.

The models assume informative voters and conditional independence. Positive
loading orientation is an assumption, not something established by low
correlations. No fitting API accepts labels. Invalid moments are reported,
not repaired; clipping exists only for the explicitly named legacy replay.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
import math

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit


MOM_MAX_ITER = 2000
MOM_TOL = 1e-13
LEGACY_EPS = 1e-6
ZERO_BOUNDARY_EPS = 1e-6
ZERO_STARTS = tuple((rho, theta) for rho in (.1, .5, .9) for theta in (.25, .75))
ZERO_OPT_OPTIONS = dict(maxiter=500, maxfun=2000, ftol=1e-12, gtol=1e-7)
ZERO_NEAR_LL = 1e-8
ZERO_PARAM_SEPARATION = .05
ZERO_MAX_HESSIAN_CONDITION = 1e8


def _json(value):
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _votes(votes):
    x = np.asarray(votes, dtype=float)
    if x.ndim != 2 or len(x) == 0 or x.shape[1] < 3:
        raise ValueError('votes must be a nonempty n x m matrix, m >= 3')
    if not np.isfinite(x).all() or not np.isin(x, (-1., 1.)).all():
        raise ValueError('votes must contain only finite -1 and +1 values')
    return x


def _weights(weights, n):
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('weights must be finite, nonnegative, and have positive sum')
    return w / w.sum()


def _offsets(offsets, n, *, positive=False):
    raw = np.asarray(offsets)
    if raw.ndim != 1 or len(raw) < 2 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError('offsets must be a one-dimensional integer boundary array')
    off = raw.astype(np.int64)
    if off[0] != 0 or off[-1] != n or (np.diff(off) < 0).any():
        raise ValueError('offsets must partition the complete step array')
    if positive and (np.diff(off) <= 0).any():
        raise ValueError('answer model requires at least one step per answer')
    return off


def _failure(method, reason, **fields):
    return _json(dict(status='FAILED', promotable=False, method=method,
                      failure_reason=reason, **fields))


def _probability_health(psi, eta, prevalence):
    return bool(np.isfinite(prevalence) and 0 < prevalence < 1 and
                np.isfinite(psi).all() and np.isfinite(eta).all() and
                ((psi > 0) & (psi < 1) & (eta > 0) & (eta < 1)).all())


def _away_from_boundary(psi, eta, prevalence):
    return bool(ZERO_BOUNDARY_EPS < prevalence < 1 - ZERO_BOUNDARY_EPS and
                ((psi > ZERO_BOUNDARY_EPS) & (psi < 1 - ZERO_BOUNDARY_EPS) &
                 (eta > ZERO_BOUNDARY_EPS) & (eta < 1 - ZERO_BOUNDARY_EPS)).all())


def _rank_one_completion(Q):
    R = np.array(Q, dtype=float, copy=True)
    np.fill_diagonal(R, np.abs(np.where(~np.eye(len(R), dtype=bool), R, 0)).max(1))
    for iteration in range(1, MOM_MAX_ITER + 1):
        eigval, eigvec = np.linalg.eigh(R)
        lam, v = eigval[-1], eigvec[:, -1]
        new = lam * v ** 2
        change = float(np.max(np.abs(new - np.diag(R))))
        if change < MOM_TOL:
            return float(lam), v, dict(converged=True, iterations=iteration,
                                      diagonal_change=change, maxiter=MOM_MAX_ITER,
                                      tol=MOM_TOL)
        np.fill_diagonal(R, new)
    return float(lam), v, dict(converged=False, iterations=MOM_MAX_ITER,
                              diagonal_change=change, maxiter=MOM_MAX_ITER,
                              tol=MOM_TOL)


def fit_mom(votes, legacy_clip=False, weights=None):
    """Fit all-voter second/third-moment MoM; return a JSON-safe fit state.

    ``legacy_clip=True`` reproduces the documented incumbent's clipping and
    records both raw values and every clipped voter. New arms must leave it
    false. Source/answer diagnostic reweighting is supported explicitly.
    """
    x = _votes(votes)
    w = _weights(weights, len(x))
    method = 'LEGACY_CLIPPED_MOM' if legacy_clip else 'STRICT_ALL_GROUP_MOM'
    mu = x.mean(0) if w is None else w @ x
    z = x - mu
    Q = np.cov(x, rowvar=False, bias=True) if w is None else (z * w[:, None]).T @ z
    lam, v, completion = _rank_one_completion(Q)
    base = dict(n_training_steps=len(x), n_vote_columns=x.shape[1],
                columns=list(range(x.shape[1])), means=mu, covariance=Q,
                rank_one_completion=completion, legacy_clip=bool(legacy_clip),
                weighted=weights is not None,
                orientation='sum(loadings)>0; majority-better-than-chance assumption')
    if not completion['converged']:
        return _failure(method, 'RANK_ONE_NONCONVERGENCE', **base)
    if not np.isfinite(lam) or lam <= 0:
        return _failure(method, 'NONIDENTIFIABLE_SECOND_MOMENT', eigenvalue=lam, **base)
    t = np.sqrt(lam) * v
    if abs(t.sum()) <= MOM_TOL * max(float(np.linalg.norm(t)), 1.):
        return _failure(method, 'UNIDENTIFIED_ORIENTATION', loadings=t, **base)
    if t.sum() < 0:
        t = -t
    e3 = lambda a, b, c: a ** 3 - 3 * a * b + 2 * c
    s1, s2, s3 = z @ t, (z ** 2) @ (t ** 2), (z ** 3) @ (t ** 3)
    numer = np.mean(e3(s1, s2, s3)) if w is None else w @ e3(s1, s2, s3)
    denom = float(e3(np.sum(t ** 2), np.sum(t ** 4), np.sum(t ** 6)))
    base.update(loadings=t, eigenvalue=lam, third_numerator=float(numer),
                third_denominator=denom)
    if not np.isfinite(numer) or not np.isfinite(denom) or denom <= 0:
        return _failure(method, 'NONIDENTIFIABLE_THIRD_MOMENT', **base)
    alpha = float(numer / denom)
    b = -alpha / np.hypot(alpha, 2.)
    prevalence = float((1 + b) / 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = t / np.sqrt(1 - b ** 2)
        psi_raw = (1 + mu + delta * (1 - b)) / 2
        eta_raw = (1 - mu + delta * (1 + b)) / 2
    base.update(alpha=alpha, b=float(b), prevalence=prevalence,
                psi_raw=psi_raw, eta_raw=eta_raw)
    if legacy_clip:
        psi = np.clip(psi_raw, LEGACY_EPS, 1 - LEGACY_EPS)
        eta = np.clip(eta_raw, LEGACY_EPS, 1 - LEGACY_EPS)
    else:
        psi, eta = psi_raw, eta_raw
    clipped = (psi != psi_raw) | (eta != eta_raw)
    base.update(psi=psi, eta=eta, pi=(psi + eta) / 2,
                clipped_columns=np.flatnonzero(clipped), out_of_range=int(clipped.sum()),
                raw_out_of_range=int(((psi_raw <= 0) | (psi_raw >= 1) |
                                      (eta_raw <= 0) | (eta_raw >= 1)).sum()))
    if not _probability_health(psi, eta, prevalence):
        return _failure(method, 'INFEASIBLE_EMISSIONS', **base)
    if not legacy_clip and not _away_from_boundary(psi, eta, prevalence):
        return _failure(method, 'BOUNDARY_EMISSIONS_OR_PREVALENCE', boundary_eps=ZERO_BOUNDARY_EPS, **base)
    return _json(dict(status='PASS', promotable=True, method=method, **base))


def _selected_votes(votes, model):
    # Archived incumbent states predate the explicit status/column schema.
    # Only their complete, subsequently validated probability state is accepted.
    archived = 'status' not in model and all(key in model for key in ('psi', 'eta', 'prevalence'))
    if model.get('status') != 'PASS' and not archived:
        raise ValueError(f"model is not defined: {model.get('failure_reason', model.get('status'))}")
    x = _votes(votes)
    if x.shape[1] != model.get('n_vote_columns', len(model['psi'])):
        raise ValueError('vote columns differ from the fitted model')
    columns = np.asarray(model.get('columns', list(range(len(model['psi'])))), dtype=int)
    if len(columns) != len(model['psi']) or (columns < 0).any() or (columns >= x.shape[1]).any():
        raise ValueError('invalid selected model columns')
    return x[:, columns]


def _class_loglik(votes, model):
    x = _selected_votes(votes, model)
    psi, eta = np.asarray(model['psi'], float), np.asarray(model['eta'], float)
    if not _probability_health(psi, eta, .5):
        raise ValueError('emissions must be strict interior probabilities')
    pos = x > 0
    log0 = np.where(pos, np.log1p(-eta), np.log(eta)).sum(1)
    log1 = np.where(pos, np.log(psi), np.log1p(-psi)).sum(1)
    return log0, log1


def posterior(votes, model):
    """Return error probabilities from a valid model, selecting its columns."""
    log0, log1 = _class_loglik(votes, model)
    pi = float(model['prevalence'])
    if not np.isfinite(pi) or not 0 < pi < 1:
        raise ValueError('prevalence must be a strict interior probability')
    return expit(np.log(pi) - np.log1p(-pi) + log1 - log0)


def _triple_batch(mu, raw_pairs, raw_third):
    """Closed-form triple estimates for one or many sufficient-stat rows."""
    mu = np.atleast_2d(mu)
    raw_pairs = np.atleast_2d(raw_pairs)
    raw_third = np.atleast_1d(raw_third)
    # Pair order is (0,1), (0,2), (1,2).
    q = raw_pairs - mu[:, [0, 0, 1]] * mu[:, [1, 2, 2]]
    third = (raw_third - mu[:, 0] * raw_pairs[:, 2] - mu[:, 1] * raw_pairs[:, 1]
             - mu[:, 2] * raw_pairs[:, 0] + 2 * mu.prod(1))
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        t0 = np.sqrt(q[:, 0] * q[:, 1] / q[:, 2])
        t = np.column_stack((t0, q[:, 0] / t0, q[:, 1] / t0))
        alpha = third / t.prod(1)
        b = -alpha / np.hypot(alpha, 2.)
        pi = (1 + b) / 2
        delta = t / np.sqrt(1 - b[:, None] ** 2)
        psi = (1 + mu + delta * (1 - b[:, None])) / 2
        eta = (1 - mu + delta * (1 + b[:, None])) / 2
    valid = ((q > 0).all(1) & np.isfinite(t).all(1) & np.isfinite(pi) &
             (pi > 0) & (pi < 1) & np.isfinite(psi).all(1) & np.isfinite(eta).all(1) &
             (pi > ZERO_BOUNDARY_EPS) & (pi < 1 - ZERO_BOUNDARY_EPS) &
             ((psi > ZERO_BOUNDARY_EPS) & (psi < 1 - ZERO_BOUNDARY_EPS) &
              (eta > ZERO_BOUNDARY_EPS) & (eta < 1 - ZERO_BOUNDARY_EPS)).all(1))
    return dict(valid=valid, pair_covariance=q, third_moment=third, loadings=t,
                alpha=alpha, b=b, prevalence=pi, psi_raw=psi, eta_raw=eta)


def fit_fixed_triple(votes, columns, weights=None):
    """Refit a fixed selected triple; diagnostic weights do not reselect it."""
    x = _votes(votes)
    col = np.asarray(columns, dtype=int)
    if col.shape != (3,) or len(np.unique(col)) != 3 or (col < 0).any() or (col >= x.shape[1]).any():
        raise ValueError('columns must identify three distinct vote columns')
    v = x[:, col]
    w = _weights(weights, len(x))
    mean = lambda a: a.mean(0) if w is None else w @ a
    mu = mean(v)
    pairs = mean(v[:, [0, 0, 1]] * v[:, [1, 2, 2]])
    third = mean(v.prod(1))
    batch = _triple_batch(mu, pairs, third)
    fields = {key: value[0] for key, value in batch.items() if key != 'valid'}
    fields.update(n_vote_columns=x.shape[1], columns=col, n_training_steps=len(x),
                  means=mu, weighted=weights is not None,
                  boundary_eps=ZERO_BOUNDARY_EPS,
                  orientation='all-three-better-than-chance assumption; positive pair covariances')
    if not bool(batch['valid'][0]):
        return _failure('FIXED_TRIPLE_MOM', 'INFEASIBLE_OR_NONIDENTIFIABLE_TRIPLE', **fields)
    fields.update(psi=fields['psi_raw'], eta=fields['eta_raw'],
                  pi=(fields['psi_raw'] + fields['eta_raw']) / 2)
    return _json(dict(status='PASS', promotable=True, method='FIXED_TRIPLE_MOM', **fields))


def _source_sufficient_statistics(x, groups):
    source_ids, inverse = np.unique(groups, return_inverse=True)
    m = x.shape[1]
    pairs = list(combinations(range(m), 2))
    triples = list(combinations(range(m), 3))
    products = np.column_stack([x[:, j] for j in range(m)] +
                               [x[:, i] * x[:, j] for i, j in pairs] +
                               [x[:, i] * x[:, j] * x[:, k] for i, j, k in triples])
    counts = np.bincount(inverse, minlength=len(source_ids))
    sums = np.zeros((len(source_ids), products.shape[1]))
    np.add.at(sums, inverse, products)
    return source_ids, counts, sums, pairs, triples


def fit_stable_triple(votes, step_source_groups, group_member_indices,
                      seed=20261012, draws=200):
    """Choose one moment-stable triple using training votes and source blocks.

    Grouping and its weights are already fitted on training data and remain
    fixed during the bootstrap. These diagnostics do not certify conditional
    independence or unconditional grouping stability.
    """
    x = _votes(votes)
    groups = np.asarray(step_source_groups)
    if groups.ndim != 1 or len(groups) != len(x):
        raise ValueError('step_source_groups must provide a source for each step')
    members = [tuple(sorted(int(i) for i in member)) for member in group_member_indices]
    flat = [i for member in members for i in member]
    if (len(members) != x.shape[1] or any(not member for member in members) or
            any(i < 0 for i in flat) or len(flat) != len(set(flat))):
        raise ValueError('group_member_indices must be distinct nonempty feature groups')
    if not isinstance(draws, (int, np.integer)) or draws < 2:
        raise ValueError('draws must be an integer >= 2')
    ids, counts, sums, pairs, triples = _source_sufficient_statistics(x, groups)
    if len(ids) < 2:
        return _failure('STABLE_TRIPLE_MOM', 'INSUFFICIENT_SOURCE_GROUPS')
    multiplicity = np.random.default_rng(seed).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=draws)
    denominators = multiplicity @ counts
    boot = (multiplicity @ sums) / denominators[:, None]
    pair_index = {pair: x.shape[1] + i for i, pair in enumerate(pairs)}
    third_index = {triple: x.shape[1] + len(pairs) + i for i, triple in enumerate(triples)}
    table, eligible = [], []
    required = int(math.ceil(.9 * draws))
    for triple in triples:
        base = fit_fixed_triple(x, triple)
        pp = [pair_index[pair] for pair in combinations(triple, 2)]
        samples = _triple_batch(boot[:, triple], boot[:, pp], boot[:, third_index[triple]])
        valid = samples['valid']
        lower = np.quantile(samples['pair_covariance'], .025, axis=0)
        n_valid = int(valid.sum())
        ok = base['status'] == 'PASS' and n_valid >= required and bool((lower > 0).all())
        stability = None
        if ok:
            pi = np.asarray(samples['prevalence'])[valid]
            stability = float(np.subtract(*np.quantile(pi, [.75, .25])) /
                              np.sqrt(base['prevalence'] * (1 - base['prevalence'])))
            ok = np.isfinite(stability)
        canonical = tuple(sorted(members[j] for j in triple))
        row = dict(columns=list(triple), member_indices=canonical, full_fit=base,
                   bootstrap_valid=n_valid, bootstrap_required=required,
                   bootstrap_valid_mask=valid, bootstrap_prevalence=np.where(valid, samples['prevalence'], np.nan),
                   covariance_lower025=lower, normalized_prevalence_IQR=stability,
                   eligible=bool(ok))
        table.append(row)
        if ok:
            eligible.append((stability, canonical, triple, base))
    bootstrap = dict(unit='canonical_source_group', conditional_on_fitted_groups=True,
                     seed=int(seed), draws=int(draws), source_ids=ids, source_step_counts=counts,
                     first_draw_multiplicity=multiplicity[0], resampled_step_counts=denominators,
                     multiplicity_sha256=hashlib.sha256(np.asarray(multiplicity, dtype='<i8').tobytes()).hexdigest())
    if not eligible:
        return _failure('STABLE_TRIPLE_MOM', 'NO_ELIGIBLE_TRIPLE', triple_table=table,
                        bootstrap=bootstrap, n_vote_columns=x.shape[1])
    _, _, selected, model = min(eligible, key=lambda item: (item[0], item[1]))
    # The table retains each full fit, including the selected one. Copy before
    # attaching the table so the JSON state never contains a reference cycle.
    model = dict(model)
    model.update(method='STABLE_TRIPLE_MOM', selected_columns=list(selected),
                 triple_table=table, bootstrap=bootstrap,
                 selection='minimum bootstrap IQR(prevalence)/sqrt(pi*(1-pi)); canonical membership tie')
    return _json(model)


def _poisson_binomial(p):
    pmf = np.ones(1)
    for probability in p:
        updated = np.zeros(len(pmf) + 1)
        updated[:-1] = pmf * (1 - probability)
        updated[1:] += pmf * probability
        pmf = updated
    return pmf


def _distribution_result(pmfs, means):
    medians, rounded = [], []
    for pmf, mean in zip(pmfs, means):
        n = len(pmf) - 1
        if not np.isfinite(pmf).all() or (pmf < 0).any() or not np.isclose(pmf.sum(), 1., atol=1e-12, rtol=1e-12):
            raise ArithmeticError('invalid count distribution; no normalization or clipping applied')
        if not np.isclose(pmf @ np.arange(n + 1), mean, atol=1e-10, rtol=1e-12):
            raise ArithmeticError('count distribution does not reproduce its stated mean')
        median = int(np.searchsorted(np.cumsum(pmf), .5, side='left'))
        half_up = int(np.floor(mean + .5))
        if not 0 <= median <= n or not 0 <= half_up <= n:
            raise ArithmeticError('count decision lies outside the answer length')
        medians.append(median)
        rounded.append(half_up)
    boundaries = np.concatenate(([0], np.cumsum([len(pmf) for pmf in pmfs])))
    return _json(dict(pmf=np.concatenate(pmfs), pmf_offsets=boundaries, mean=means,
                      mean_half_up=rounded, median_lower=medians))


def count_distribution(p, offsets):
    """Exact Poisson-binomial readouts; assumes conditional step independence.

    The mean sum itself does not require independence. Its half-up integer
    decision minimizes squared count loss; the smallest median minimizes
    absolute count loss under this explicitly assumed joint distribution.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('p must contain probabilities in [0,1]')
    off = _offsets(offsets, len(p))
    pmfs = [_poisson_binomial(p[a:b]) for a, b in zip(off[:-1], off[1:])]
    means = [float(p[a:b].sum()) for a, b in zip(off[:-1], off[1:])]
    return _distribution_result(pmfs, means)


def _zero_objective(log0, log1, off):
    f0, f1 = np.exp(log0), np.exp(log1)
    clean = np.add.reduceat(log0, off[:-1])

    def evaluate(parameters):
        rho, theta = parameters
        if not 0 <= rho <= 1 or not 0 <= theta <= 1:
            return float('inf'), np.full(2, np.nan)
        mix = (1 - theta) * f0 + theta * f1
        active = np.add.reduceat(np.log(mix), off[:-1])
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            loglik = np.logaddexp(np.log(rho) + clean, np.log1p(-rho) + active)
            q_active = np.exp(np.log1p(-rho) + active - loglik)
            d_rho = np.exp(clean - loglik) - np.exp(active - loglik)
        d_theta = q_active * np.add.reduceat((f1 - f0) / mix, off[:-1])
        return -float(loglik.mean()), -np.array([d_rho.mean(), d_theta.mean()])

    return evaluate


def _zero_hessian(evaluate, parameters):
    parameters = np.asarray(parameters, float)
    distance = np.minimum(parameters, 1 - parameters)
    hessian = np.zeros((2, 2))
    for j in range(2):
        h = min(1e-4, float(distance[j]) / 10)
        if h <= np.finfo(float).eps:
            return dict(available=False, reason='boundary optimum')
        high, low = parameters.copy(), parameters.copy()
        high[j] += h
        low[j] -= h
        hessian[:, j] = (evaluate(high)[1] - evaluate(low)[1]) / (2 * h)
    hessian = (hessian + hessian.T) / 2
    if not np.isfinite(hessian).all():
        return dict(available=False, reason='nonfinite numerical Hessian')
    eigenvalues = np.linalg.eigvalsh(hessian)
    condition = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else float('inf')
    return dict(available=True, matrix=hessian, eigenvalues=eigenvalues, condition=condition,
                healthy=bool(eigenvalues[0] > 0 and condition <= ZERO_MAX_HESSIAN_CONDITION))


def fit_zero_mixture(votes, offsets, triple_model):
    """Freeze triple emissions and fit structural-clean mass/active prevalence.

    Active answers can themselves have zero errors. The structural-clean
    probability is therefore not the posterior probability of zero errors.
    No clean labels, pseudocounts, probability clipping, or rescue models enter.
    """
    if triple_model.get('status') != 'PASS':
        return _failure('ZERO_MIXTURE', 'FAILED_DEPENDENCY', dependency_status=triple_model.get('status'),
                        dependency_reason=triple_model.get('failure_reason'))
    if len(triple_model.get('columns', [])) != 3:
        return _failure('ZERO_MIXTURE', 'DEPENDENCY_IS_NOT_A_TRIPLE')
    log0, log1 = _class_loglik(votes, triple_model)
    off = _offsets(offsets, len(log0), positive=True)
    if not (np.diff(off) >= 2).any():
        return _failure('ZERO_MIXTURE', 'SINGLE_STEP_ANSWER_NONIDENTIFICATION')
    if np.max(np.abs(log1 - log0)) <= MOM_TOL:
        return _failure('ZERO_MIXTURE', 'UNINFORMATIVE_FROZEN_EMISSIONS')
    evaluate = _zero_objective(log0, log1, off)
    starts, candidates = [], []
    for start in ZERO_STARTS:
        try:
            result = minimize(evaluate, np.asarray(start), jac=True, method='L-BFGS-B',
                              bounds=((0., 1.), (0., 1.)), options=ZERO_OPT_OPTIONS)
            objective, gradient = evaluate(result.x)
            valid = bool(result.success and np.isfinite(objective) and np.isfinite(result.x).all()
                         and np.isfinite(gradient).all())
            row = dict(start=start, converged=valid, optimizer_success=bool(result.success),
                       parameters=result.x, average_log_likelihood=-objective,
                       gradient=gradient, nit=int(result.nit), nfev=int(result.nfev), message=str(result.message))
            if valid:
                candidates.append((objective, np.asarray(result.x), len(starts)))
        except (ValueError, ArithmeticError, FloatingPointError) as exc:
            row = dict(start=start, converged=False, message=repr(exc))
        starts.append(row)
    ordinary = minimize_scalar(lambda theta: evaluate((0., theta))[0], bounds=(0., 1.),
                               method='bounded', options=dict(maxiter=500, xatol=1e-10))
    ordinary_candidates = [(evaluate((0., theta))[0], float(theta)) for theta in (0., 1.)]
    if ordinary.success and np.isfinite(ordinary.fun):
        ordinary_candidates.append((float(ordinary.fun), float(ordinary.x)))
    ordinary_objective, ordinary_theta = min(ordinary_candidates)
    nested = dict(rho=0., theta=ordinary_theta, average_log_likelihood=-ordinary_objective,
                  converged=bool(ordinary.success), message=str(ordinary.message))
    boundary_model = dict(status='PASS', promotable=False, method='ZERO_MIXTURE',
                          rho=0., theta=ordinary_theta, columns=triple_model['columns'],
                          n_vote_columns=triple_model['n_vote_columns'],
                          psi=triple_model['psi'], eta=triple_model['eta'],
                          prevalence=ordinary_theta, boundary_status='NO_EXTRA_ZERO_COMPONENT',
                          average_log_likelihood=-ordinary_objective)
    if (not ordinary.success or ordinary_theta <= ZERO_BOUNDARY_EPS or
            ordinary_theta >= 1 - ZERO_BOUNDARY_EPS):
        boundary_model.update(status='FAILED', failure_reason='DEGENERATE_OR_NONCONVERGED_ORDINARY_MODEL')
    if ordinary.success and np.isfinite(ordinary_objective):
        candidates.append((ordinary_objective, np.array([0., ordinary_theta]), 'ordinary_rho_zero'))
    # Explicit all-clean boundary checks prevent an interior optimizer result
    # from hiding a higher-likelihood degenerate solution.
    all_clean_objective = evaluate((1., .5))[0]
    candidates.append((all_clean_objective, np.array([1., .5]), 'all_clean_boundary'))
    base = dict(starts=starts, start_grid=ZERO_STARTS, optimizer_options=ZERO_OPT_OPTIONS,
                nested_ordinary=nested, boundary_model=boundary_model,
                n_training_steps=len(log0), n_training_answers=len(off) - 1,
                n_vote_columns=triple_model['n_vote_columns'], columns=triple_model['columns'],
                psi=triple_model['psi'], eta=triple_model['eta'],
                step_model_prevalence=triple_model['prevalence'], boundary_eps=ZERO_BOUNDARY_EPS,
                emissions_frozen=True, fitting_unit='answer likelihood; source-related answers may remain dependent')
    if not any(row['converged'] for row in starts):
        return _failure('ZERO_MIXTURE', 'FAILED_CONVERGENCE', **base)
    finite = [candidate for candidate in candidates if np.isfinite(candidate[0])]
    best_objective, best, selected_start = min(finite, key=lambda candidate: candidate[0])
    rho, theta = map(float, best)
    near = [candidate for candidate in finite if candidate[0] - best_objective <= ZERO_NEAR_LL]
    competing = [dict(parameters=candidate[1], average_log_likelihood=-candidate[0], start=candidate[2])
                 for candidate in near if np.max(np.abs(candidate[1] - best)) > ZERO_PARAM_SEPARATION]
    base.update(rho=rho, theta=theta, prevalence=(1 - rho) * theta,
                selected_start=selected_start, average_log_likelihood=-best_objective,
                competing_near_optima=competing,
                identifiability_tolerances=dict(average_ll=ZERO_NEAR_LL,
                                               parameter_separation=ZERO_PARAM_SEPARATION,
                                               max_hessian_condition=ZERO_MAX_HESSIAN_CONDITION))
    if rho >= 1 - ZERO_BOUNDARY_EPS or theta <= ZERO_BOUNDARY_EPS:
        return _failure('ZERO_MIXTURE', 'DEGENERATE_ALL_CLEAN', **base)
    if theta >= 1 - ZERO_BOUNDARY_EPS:
        return _failure('ZERO_MIXTURE', 'DEGENERATE_ACTIVE_ALL_ERROR', **base)
    if competing:
        return _failure('ZERO_MIXTURE', 'COMPETING_NEAR_OPTIMA', **base)
    if rho <= ZERO_BOUNDARY_EPS:
        base.update(boundary_status='NO_EXTRA_ZERO_COMPONENT' if rho == 0 else 'NEAR_ZERO_EXTRA_COMPONENT',
                    hessian=dict(available=False, reason='clean-component boundary'))
        return _json(dict(status='PASS', promotable=False, method='ZERO_MIXTURE', **base))
    hessian = _zero_hessian(evaluate, best)
    base['hessian'] = hessian
    if not hessian.get('healthy', False):
        return _failure('ZERO_MIXTURE', 'WEAK_IDENTIFICATION', **base)
    return _json(dict(status='PASS', promotable=True, method='ZERO_MIXTURE',
                      boundary_status='INTERIOR', **base))


def predict_zero_mixture(votes, offsets, model, zero_model=None):
    """Return marginal/active posteriors and the actual zero-mixture count PMF."""
    if zero_model is not None:
        for key in ('columns', 'psi', 'eta'):
            if key in zero_model and not np.array_equal(np.asarray(zero_model[key]), np.asarray(model[key])):
                raise ValueError('zero-mixture state changes the frozen triple emissions')
        model = dict(model, **zero_model)
    log0, log1 = _class_loglik(votes, model)
    off = _offsets(offsets, len(log0), positive=True)
    rho, theta = float(model['rho']), float(model['theta'])
    if not 0 <= rho <= 1 or not 0 <= theta <= 1:
        raise ValueError('zero-mixture parameters must lie in [0,1]')
    f0, f1 = np.exp(log0), np.exp(log1)
    mix = (1 - theta) * f0 + theta * f1
    active_p = theta * f1 / mix
    clean = np.add.reduceat(log0, off[:-1])
    active = np.add.reduceat(np.log(mix), off[:-1])
    if rho == 0:
        q = np.zeros(len(off) - 1)
    elif rho == 1:
        q = np.ones(len(off) - 1)
    else:
        q = expit(np.log(rho) - np.log1p(-rho) + clean - active)
    marginal = active_p * np.repeat(1 - q, np.diff(off))
    pmfs, means = [], []
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        pmf = _poisson_binomial(active_p[a:b]) * (1 - q[i])
        pmf[0] += q[i]
        pmfs.append(pmf)
        means.append(float(marginal[a:b].sum()))
    result = _distribution_result(pmfs, means)
    result.update(marginal_p=marginal.tolist(), active_p=active_p.tolist(), structural_clean_p=q.tolist())
    return result
