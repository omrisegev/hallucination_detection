"""Supervised linear diagnostic on the answer-locally standardized expansion banks.

Access label: supervised, other-answer access (labels of other answers in the
same cell, source-group-disjoint folds).  A matched diagnostic for the label-free
arms, not a ceiling.  The step score is the SAME top-10 token mean used by the
label-free readout; supervision acts on aggregated steps only (no token-label
broadcasting).  This module owns neither the fold contract nor the metrics.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit

from .direct_probability_fusion import zscore_columns
from .varentropy_expansion_fusion import BANK_COLUMNS, WIDTH, expansion_columns

RIDGE = 0.01
BANKS = ('B2_sel', 'B2d_sel')          # B2d_sel is a column subset of the standardized B2_sel matrix
EXCLUDED = -1


def standardized_bank(logprobs, chosen):
    """Answer-local z-scored 138-column bank; constant columns (scale<=1e-10) become 0."""
    X, _, _ = expansion_columns(logprobs, chosen)
    Z, keep, _, _ = zscore_columns(X)
    full = np.zeros(X.shape, dtype=np.float64)
    full[:, keep] = Z
    return full, keep


def step_labels(kind, n_steps, *, target=None, labels=None):
    """PB: prefix steps 0, first-error step 1, later steps excluded; clean all 0. PRMB: given labels, unknown excluded."""
    if kind == 'pb':
        y = np.zeros(int(n_steps), dtype=int)
        t = int(target)
        if t >= 0:
            if t >= n_steps: raise ValueError('first-error step outside the answer')
            y[t] = 1; y[t + 1:] = EXCLUDED
        return y
    if kind == 'prm':
        y = np.asarray(labels, dtype=int).copy()
        if y.shape != (int(n_steps),): raise ValueError('PRMB label length mismatch')
        y[y < 0] = EXCLUDED
        return y
    raise ValueError(kind)


def class_balanced_weights(y):
    """Each known class carries total mass 0.5; excluded steps carry 0. Raises if a class is absent."""
    y = np.asarray(y, int); w = np.zeros(len(y), float)
    for label in (0, 1):
        mask = y == label
        if not mask.any(): raise ValueError(f'training set has no step of class {label} (declared fold failure)')
        w[mask] = 0.5 / mask.sum()
    return w


class StepTop10:
    """Top-10 token mean per step with its exact subgradient, over concatenated answers.

    Ties select earlier tokens (stable argsort of -token), matching
    ``rbm_matched_top10.top10_value_gradient``.  ``spans`` are global token index pairs.
    """

    def __init__(self, x, spans):
        self.x = np.asarray(x)
        self.spans = np.asarray(spans, int)
        if self.x.ndim != 2 or self.spans.ndim != 2 or self.spans.shape[1] != 2: raise ValueError('bad shapes')
        if np.any(self.spans[:, 0] < 0) or np.any(self.spans[:, 1] <= self.spans[:, 0]) or np.any(self.spans[:, 1] > len(self.x)):
            raise ValueError('invalid step span')

    def selection(self, token):
        rows, cols, vals = [], [], []
        for step, (a, b) in enumerate(self.spans):
            chosen = np.argsort(-token[a:b], kind='stable')[:min(10, b - a)] + a
            rows.extend([step] * len(chosen)); cols.extend(chosen.tolist()); vals.extend([1.0 / len(chosen)] * len(chosen))
        return csr_matrix((np.asarray(vals, dtype=self.x.dtype), (rows, cols)), shape=(len(self.spans), len(self.x)))

    def evaluate(self, w, b, derivative=False):
        w = np.asarray(w, dtype=self.x.dtype)
        token = self.x @ w + self.x.dtype.type(b)
        S = self.selection(token)
        value = np.asarray(S @ token, dtype=np.float64)
        if not derivative: return value, None
        jac = np.empty((len(value), self.x.shape[1] + 1)); jac[:, :-1] = S @ self.x; jac[:, -1] = 1.0
        return value, jac


def objective(theta, top, y, weight, ridge=RIDGE):
    theta = np.asarray(theta, float)
    s, jac = top.evaluate(theta[:-1], theta[-1], derivative=True)
    known = y != EXCLUDED
    if np.any(weight[~known] != 0): raise ValueError('excluded steps must carry zero weight')
    yy = np.where(known, y, 0)
    loss = float(np.sum(weight[known] * (np.logaddexp(0, s[known]) - yy[known] * s[known])))
    residual = np.zeros_like(s); residual[known] = weight[known] * (expit(s[known]) - yy[known])
    penalty = np.r_[theta[:-1], 0.0]
    return loss + 0.5 * ridge * float(penalty @ penalty), jac.T @ residual + ridge * penalty


def fit(x, spans, y, *, ridge=RIDGE, maxiter=200):
    """L-BFGS-B fit of the linear step-supervised model; returns (theta, info)."""
    top = StepTop10(x, spans); y = np.asarray(y, int); weight = class_balanced_weights(y)
    fun = lambda t: objective(t, top, y, weight, ridge)
    initial = np.zeros(x.shape[1] + 1); start = fun(initial)[0]
    result = minimize(fun, initial, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6, maxls=40))
    end, grad = fun(result.x)
    if not np.isfinite(result.x).all() or not np.isfinite(end) or end > start + 1e-8: raise RuntimeError('nonfinite or worsening fit')
    theta_norm = float(np.linalg.norm(result.x)); gradient_max = float(np.max(np.abs(grad)))
    # A fit that never left the origin or exits with a large gradient is a declared STALLED fit,
    # never reported as FIT/converged. No perturbed restart (it would be an undeclared second start).
    stalled = theta_norm < 1e-8 or gradient_max > 1e-2
    return result.x, dict(status='STALLED' if stalled else 'FIT', stalled=bool(stalled),
                          converged=bool(result.success and not stalled), message=str(result.message), iterations=int(result.nit),
                          initial_loss=float(start), final_loss=float(end), gradient_max=gradient_max,
                          theta_norm=theta_norm, n_train_steps=int(np.sum(y != EXCLUDED)),
                          n_positive_steps=int(np.sum(y == 1)), n_negative_steps=int(np.sum(y == 0)), ridge=float(ridge))


def score_steps(x, spans, theta):
    """Held-out step scores with the same top-10 readout (float64 token scores)."""
    x = np.asarray(x, np.float64); theta = np.asarray(theta, float)
    return StepTop10(x, spans).evaluate(theta[:-1], theta[-1])[0]


def bank_columns(bank):
    if bank not in BANKS: raise ValueError(bank)
    return BANK_COLUMNS[bank]


# ----------------------------------------------------------------------------
# Correction 2026-09-13 (supervised diagnostic audit): fit-outcome classification,
# smoke PASS rule, extraction verification and held-fold-blind calibration.
# None of this changes fit(), score_steps(), step_labels() or the bank layout.
# ----------------------------------------------------------------------------

FIT_OUTCOMES = ('FIT', 'FIT_ITERATION_LIMIT', 'EXPECTED_SMOKE_LIMITATION', 'STALLED', 'UNEXPECTED_FAILURE')
_BUDGET_MESSAGES = ('ITERATIONS REACHED LIMIT', 'EVALUATIONS EXCEEDS LIMIT')
_EXPECTED_LIMITATIONS = (re.compile(r'empty training or held fold'), re.compile(r'training set has no step of class'))


def classify_fit(info):
    """Map one saved fit-info record to a single outcome in ``FIT_OUTCOMES``.

    FIT: finite fit that the optimizer reported as converged.
    FIT_ITERATION_LIMIT: finite fit that stopped on the iteration/evaluation budget
    (explicitly never reported as converged).  EXPECTED_SMOKE_LIMITATION: declared
    fold failure whose reason is an empty train/held fold or a missing class (the
    expected consequence of a tiny subset; in a full run it is still a declared
    fold failure).  STALLED: declared stalled fit.  UNEXPECTED_FAILURE: anything
    else, including a finite non-converged fit with an unexplained message.
    """
    status = info.get('status')
    if status == 'STALLED': return 'STALLED'
    if status == 'FIT':
        if info.get('converged'): return 'FIT'
        message = str(info.get('message', ''))
        if any(m in message.upper() for m in _BUDGET_MESSAGES): return 'FIT_ITERATION_LIMIT'
        return 'UNEXPECTED_FAILURE'
    if status == 'FAILED':
        reason = str(info.get('reason', ''))
        if any(p.search(reason) for p in _EXPECTED_LIMITATIONS): return 'EXPECTED_SMOKE_LIMITATION'
        return 'UNEXPECTED_FAILURE'
    return 'UNEXPECTED_FAILURE'


def fit_health(fits):
    """Outcome counts for a list of fit-info records (each record keeps its own reason)."""
    outcomes = [classify_fit(f) for f in fits]
    counts = {o: int(sum(x == o for x in outcomes)) for o in FIT_OUTCOMES}
    return dict(n_fits=len(fits), n_converged=counts['FIT'], n_iteration_limit=counts['FIT_ITERATION_LIMIT'],
                n_stalled=counts['STALLED'], n_expected_limitations=counts['EXPECTED_SMOKE_LIMITATION'],
                n_unexpected_failures=counts['UNEXPECTED_FAILURE'], outcomes=counts,
                by_cell_bank_fold={f'{f.get("cell")}|{f.get("bank")}|{f.get("fold")}': o for f, o in zip(fits, outcomes)})


def smoke_status(fits, banks=BANKS):
    """PASS only if every bank has >=1 FIT/FIT_ITERATION_LIMIT and no UNEXPECTED_FAILURE/STALLED;
    INCONCLUSIVE if some bank has no successful fit but every non-success is an expected limitation;
    FAIL otherwise.  An empty fit list is FAIL.  Returns (status, health)."""
    fits = list(fits); health = fit_health(fits)
    if not fits: return 'FAIL', health
    outcomes = [classify_fit(f) for f in fits]
    if any(o in ('UNEXPECTED_FAILURE', 'STALLED') for o in outcomes): return 'FAIL', health
    successful = {f.get('bank') for f, o in zip(fits, outcomes) if o in ('FIT', 'FIT_ITERATION_LIMIT')}
    if all(b in successful for b in banks): return 'PASS', health
    return 'INCONCLUSIVE', health


def manifest_differences(existing, manifest):
    """Sorted list of top-level keys (and ``<key>/<sub-key>`` for dict values) whose values differ."""
    out = []
    for key in sorted(set(existing) | set(manifest)):
        a, b = existing.get(key), manifest.get(key)
        if a == b: continue
        if isinstance(a, dict) and isinstance(b, dict):
            out.extend(f'{key}/{sub}' for sub in sorted(set(a) | set(b)) if a.get(sub) != b.get(sub))
        else:
            out.append(key)
    return out


def spans_sha256(spans):
    """Content hash of a step-span array (int64 little-endian bytes, row-major)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(spans, np.int64)).astype('<i8').tobytes()).hexdigest()


def verify_answer(record, row, *, kind, dataset=None, bench_starts, bench_ends, detector=None):
    """Extraction-time contract for one cached answer; returns its provenance dict or raises.

    Checks the uid/row_id mapping, spans equal (values) to the frozen benchmark
    step boundaries, token count equal to len(token_entropies) and to the top-K
    logprob rows, alignment of the top-K matrix with token_spilled_energies (same
    length, selected surprisal finite), spans inside the answer, and PB gate
    detector equality (mean token entropy).
    """
    uid = record['uid']
    source_key = f"{dataset}::{row.get('id')}" if kind == 'pb' else str(row.get('idx'))
    if source_key != record['row_id']: raise ValueError(f'{uid}: source row key {source_key!r} != record row_id {record["row_id"]!r}')
    entropy = np.asarray(row['token_entropies'], float); chosen = np.asarray(row['token_spilled_energies'], float)
    payload = row.get('top_k_logprobs') if isinstance(row.get('top_k_logprobs'), dict) else row.get('top_k_logprobs_raw')
    if not isinstance(payload, dict) or payload.get('logprobs') is None: raise ValueError(f'{uid}: no saved top-K log-probability matrix')
    logprobs = np.asarray(payload['logprobs']); spans = np.asarray(row['step_token_spans'], int); T = int(len(entropy))
    if logprobs.ndim != 2 or logprobs.shape[0] != T: raise ValueError(f'{uid}: top-K rows {logprobs.shape} != token count {T}')
    if len(chosen) != T: raise ValueError(f'{uid}: token_spilled_energies length {len(chosen)} != token count {T}')
    if not np.isfinite(chosen).all(): raise ValueError(f'{uid}: non-finite selected surprisal')
    if not np.isfinite(entropy).all(): raise ValueError(f'{uid}: non-finite token entropy')
    if int(record['tokens']) != T: raise ValueError(f'{uid}: benchmark token count {record["tokens"]} != source {T}')
    if spans.shape != (int(record['steps']), 2): raise ValueError(f'{uid}: step count mismatch {spans.shape} vs {record["steps"]}')
    starts = np.asarray(bench_starts, int); ends = np.asarray(bench_ends, int)
    if starts.shape != (len(spans),) or ends.shape != (len(spans),): raise ValueError(f'{uid}: frozen boundary count mismatch')
    if not (np.array_equal(spans[:, 0], starts) and np.array_equal(spans[:, 1], ends)):
        raise ValueError(f'{uid}: step boundaries differ from frozen benchmark scores npz')
    if np.any(spans[:, 0] < 0) or np.any(spans[:, 1] <= spans[:, 0]) or np.any(spans[:, 1] > T): raise ValueError(f'{uid}: span outside answer')
    if kind == 'pb':
        if detector is None or not np.isclose(entropy.mean(), float(detector), atol=1e-12, rtol=0):
            raise ValueError(f'{uid}: PB gate detector mismatch {entropy.mean()} vs {detector}')
    return dict(row_id=record['row_id'], source_key=source_key, n_tokens=T, n_steps=int(len(spans)), spans_sha256=spans_sha256(spans),
                entropy_mean=float(entropy.mean()), topk_width=int(logprobs.shape[1]), topk_finite=bool(np.isfinite(logprobs).all()))


def training_matrix(answers, train, cols, *, kind, target, labels, offsets):
    """Concatenate the standardized banks of ``train`` answers (same convention as the outer driver)."""
    x = np.concatenate([answers[i][0][:, cols] for i in train]); spans = []; y = []; base_offset = 0
    for i in train:
        z, sp = answers[i]; spans.append(sp + base_offset); base_offset += len(z)
        y.append(step_labels(kind, len(sp), target=target[i], labels=labels[offsets[i]:offsets[i + 1]]))
    return x, np.concatenate(spans), np.concatenate(y)


def calibration_plan(indices, outer, held):
    """For held outer fold ``held``: [(h, train, score)] with train = outer not in {held, h}, score = outer == h."""
    plan = []
    for h in sorted({int(outer[i]) for i in indices} - {int(held)}):
        train = [i for i in indices if outer[i] not in (held, h)]; score = [i for i in indices if outer[i] == h]
        plan.append((h, train, score))
    return plan


def held_fold_blind_check(records, outer, held, train, score):
    """Index-level assertions that fold ``held`` contributes neither rows nor source groups to a calibration model."""
    if any(int(outer[i]) == int(held) for i in train): raise AssertionError(f'held fold {held} row inside calibration training set')
    if any(int(outer[i]) == int(held) for i in score): raise AssertionError(f'held fold {held} row inside calibration scoring set')
    held_groups = {records[i]['group_id'] for i in range(len(records)) if int(outer[i]) == int(held)}
    train_groups = {records[i]['group_id'] for i in train}
    if train_groups & held_groups: raise AssertionError(f'calibration training groups intersect held fold {held}')
    if {records[i]['group_id'] for i in score} & train_groups: raise AssertionError('calibration scoring groups intersect its training groups')
    return sorted(train_groups)


def calibration_quantile(flat, indices, offsets, q=.8):
    """Mirror of evaluate_arrays: concatenate the (all-finite) answer score blocks in index order, then np.quantile."""
    blocks = [flat[offsets[i]:offsets[i + 1]] for i in indices if offsets[i + 1] > offsets[i] and np.isfinite(flat[offsets[i]:offsets[i + 1]]).all()]
    if not blocks: raise ValueError('no valid calibration answers')
    return float(np.quantile(np.concatenate(blocks), q))


__all__ = ['BANKS', 'EXCLUDED', 'FIT_OUTCOMES', 'RIDGE', 'StepTop10', 'bank_columns', 'calibration_plan', 'calibration_quantile',
           'class_balanced_weights', 'classify_fit', 'fit', 'fit_health', 'held_fold_blind_check', 'manifest_differences', 'objective',
           'score_steps', 'smoke_status', 'spans_sha256', 'standardized_bank', 'step_labels', 'training_matrix', 'verify_answer', 'WIDTH']
