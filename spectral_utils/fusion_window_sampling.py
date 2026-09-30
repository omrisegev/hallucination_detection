"""Observation selection supporting the unchanged IU / Joint feature fusion.

No labels are accepted. Dense features are still needed: this is fit-row
selection, not inference saving. DUFS transposition gates windows while its
graph nodes are feature coordinates; diffusion sampling graphs windows.
"""
from __future__ import annotations
import hashlib
import time
import numpy as np
from .adapted_dufs import adapted_dufs_soft_gates
from .answer_localization_v2 import (PRIMITIVES, STREAM_NAMES, fit_local,
    moment_plan, moment_matrix, mixture_readout, json_safe)
from .window_localization import WindowPlan, windows_to_tokens

REP = 'moments27_local8'
METHODS = ('equal', 'iu', 'joint_lambda0', 'joint_graph010', 'joint_graph_permuted')
CORES = tuple(REP+'__'+m for m in METHODS)
SELECTORS = ('full', 'uniform', 'risk_top', 'dufs_transposed', 'dufs_permuted', 'window_diffusion')
NAMES = tuple(s+'__'+op for s in PRIMITIVES for op in ('level', 'sd', 'slope'))


def seed_for(identity):
    return int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)


def budget(n):
    if n < 1: raise ValueError('EMPTY_FIT_GRID')
    return min(n, max(32, (n+1)//2))


def top_indices(scores, count):
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 1 <= count <= len(scores):
        raise ValueError('INVALID_SELECTION_SCORES')
    return np.sort(np.lexsort((np.arange(len(scores)), -scores))[:count])


def selection_geometry(values):
    x = np.asarray(values, float)
    if x.ndim != 2 or not np.isfinite(x).all(): raise ValueError('INVALID_SELECTION_MATRIX')
    good = np.ptp(x, axis=0) > 1e-10 * np.maximum(1., np.max(np.abs(x), axis=0))
    z = x[:, good]
    z = (z-z.mean(axis=0))/z.std(axis=0)
    keep = []
    for j in range(z.shape[1]):
        if not any(abs(float(z[:, j] @ z[:, k] / len(z))) >= 1-1e-10 for k in keep): keep.append(j)
    if len(keep) < 3: raise ValueError('INSUFFICIENT_SELECTOR_COORDINATES')
    return z[:, keep]


def transposed_gates(z):
    """Input N windows x P moments; output N window probabilities."""
    dual = z-z.mean(axis=1, keepdims=True)
    dual /= np.maximum(np.linalg.norm(dual, axis=1, keepdims=True), 1e-12)
    _, detail = adapted_dufs_soft_gates(dual, seeds=(0, 1, 2), epochs=120)
    probabilities = np.asarray(detail['raw_probabilities'], float)
    if probabilities.shape != (len(z),): raise ValueError('DUFS_AXIS_MISMATCH')
    return probabilities, detail


def diffusion_indices(z, count):
    n = len(z)
    if not 1 <= count <= n: raise ValueError('INVALID_BUDGET')
    distances = np.maximum(0., ((z[:, None, :]-z[None, :, :])**2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind='stable')[:, :min(7, n-1)]
    sigma = np.sqrt(np.take_along_axis(distances, neighbors[:, -1:], axis=1).ravel())
    sigma = np.maximum(sigma, 1e-12)
    w = np.zeros((n, n))
    for i in range(n):
        js = neighbors[i]
        w[i, js] = np.exp(-distances[i, js]/np.maximum(sigma[i]*sigma[js], 1e-24))
    w = np.maximum(w, w.T)
    degree = w.sum(axis=1)
    if np.any(degree <= 0): raise ValueError('ISOLATED_DIFFUSION_NODE')
    walk = w/degree[:, None]
    embedding = (walk@walk)/np.sqrt(degree/degree.sum())[None, :]
    sqnorm = (embedding**2).sum(axis=1)
    d = np.maximum(0., sqnorm[:, None]+sqnorm[None, :]-2*embedding@embedding.T)
    selected = [int(np.argmin(d.sum(axis=1)))]
    nearest = d[:, selected[0]].copy()
    while len(selected) < count:
        nearest[selected] = -np.inf
        j = int(np.argmax(nearest)); selected.append(j)
        nearest = np.minimum(nearest, d[:, j])
    return np.sort(selected), {'nodes': n, 'coordinates': z.shape[1],
                               'edges': int(np.count_nonzero(w)//2)}


def choose_all(values, identity):
    n = len(values); m = budget(n); output = {}; details = {}
    for selector in SELECTORS:
        if selector == 'full' or m == n:
            output[selector] = np.arange(n)
            details[selector] = {'status': 'ALL_ROWS', 'seconds': 0.}
    if m == n: return output, details
    started = time.monotonic()
    output['uniform'] = np.rint(np.linspace(0, n-1, m)).astype(int)
    details['uniform'] = {'status': 'OK', 'seconds': time.monotonic()-started}
    started = time.monotonic()
    output['risk_top'] = top_indices(values[:, 0], m)
    details['risk_top'] = {'status': 'OK', 'seconds': time.monotonic()-started}
    started = time.monotonic()
    try:
        z = selection_geometry(values)
        p, diagnostic = transposed_gates(z)
        permutation = np.random.default_rng(seed_for(identity+'/sampling-permutation')).permutation(n)
        output['dufs_transposed'] = top_indices(p, m)
        output['dufs_permuted'] = top_indices(p[permutation], m)
        details['dufs_transposed'] = {'status': 'OK', 'seconds': time.monotonic()-started,
            'graph_nodes': z.shape[1], 'gated_windows': n, 'probabilities': p,
            'diagnostics': diagnostic}
        details['dufs_permuted'] = {'status': 'OK', 'seconds': 0., 'shared_cost': 'dufs_transposed',
            'permutation': permutation, 'probabilities': p[permutation]}
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        for selector in ('dufs_transposed', 'dufs_permuted'):
            details[selector] = {'status': 'FAILED', 'reason': str(exc), 'seconds': time.monotonic()-started}
    started = time.monotonic()
    try:
        output['window_diffusion'], diagnostic = diffusion_indices(selection_geometry(values), m)
        details['window_diffusion'] = {'status': 'OK', 'seconds': time.monotonic()-started, **diagnostic}
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        details['window_diffusion'] = {'status': 'FAILED', 'reason': str(exc), 'seconds': time.monotonic()-started}
    return output, details


def perturb_windows(raw, identity, replicate):
    """Within-window block bootstrap; never copies tokens between windows."""
    changed = np.array(raw, copy=True)
    rng = np.random.default_rng(seed_for(identity+'/block-perturb/'+str(replicate)))
    for lo in range(0, len(raw)-7, 8):
        blocks = rng.integers(0, 4, 4)
        positions = (2*blocks[:, None]+np.arange(2)).ravel()
        changed[lo:lo+8] = raw[lo+positions]
    return changed


def jaccard(left, right):
    a, b = set(map(int, left)), set(map(int, right))
    return len(a & b)/len(a | b)


def score_sampling(parent_arrays, parent_meta, raw, identity):
    plan = moment_plan(len(raw), 8)
    original = plan.fit_indices
    values = np.asarray(parent_arrays[REP+'__features'])
    assert np.array_equal(parent_arrays[REP+'__fit_indices'], original)
    selections, diagnostics = choose_all(values[original], identity)
    arrays = {'step_starts': parent_arrays['step_starts'], 'step_ends': parent_arrays['step_ends']}
    methods = {}; fit_details = {}
    for selector, chosen in selections.items():
        arrays[selector+'__selected'] = original[chosen]
        starts = plan.starts[original[chosen]]
        diagnostics[selector].update(n_original=len(original), n_selected=len(chosen),
            quartile_counts=np.bincount(np.minimum(3, starts*4//len(raw)), minlength=4),
            largest_start_gap_tokens=int(np.diff(np.r_[0, starts, len(raw)]).max()))
    perturb_started = time.monotonic()
    if budget(len(original)) < len(original):
        for replicate in (0, 1):
            changed = perturb_windows(raw, identity, replicate)
            perturbed, _ = moment_matrix(changed, plan)
            pert_selections, pert_detail = choose_all(perturbed[original], identity)
            for selector, chosen in selections.items():
                diagnostics[selector].setdefault('block_perturbation', []).append({
                    'replicate': replicate, 'status': pert_detail[selector]['status'],
                    'jaccard': jaccard(chosen, pert_selections[selector]) if selector in pert_selections else None})
        if 'dufs_transposed' in selections:
            per_seed = diagnostics['dufs_transposed']['diagnostics']['per_seed_probabilities']
            sets = [top_indices(p, budget(len(original))) for p in per_seed]
            diagnostics['dufs_transposed']['seed_pair_jaccard'] = [
                jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i+1, len(sets))]
    perturb_seconds = time.monotonic()-perturb_started
    for selector in SELECTORS:
        started = time.monotonic()
        if selector not in selections:
            for core in CORES: methods[core+'@@'+selector] = {'valid': False, 'reason': 'SELECTOR_FAILED'}
            continue
        chosen = selections[selector]
        replay = len(chosen) == len(original)
        scores, meta, shared = {}, {}, {}
        if not replay:
            selected_plan = WindowPlan(plan.token_count, plan.width, plan.stride, plan.starts, plan.ends, original[chosen])
            try:
                scores, meta, shared = fit_local(values, list(NAMES), selected_plan, identity+'/'+selector)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                meta = {m: {'valid': False, 'reason': str(exc)} for m in METHODS}
        for core, method in zip(CORES, METHODS):
            arm = core+'@@'+selector
            parent = parent_meta['report']['methods'][core]
            detail = dict(parent if replay else meta.get(method, {'valid': False, 'reason': 'MISSING_FIT'}))
            if replay and not parent.get('readout_valid'):
                detail.update(valid=False, reason='PARENT_READOUT_INVALID')
            detail['parent_replay'] = replay
            risk = parent_arrays[core+'__window'] if replay and core+'__window' in parent_arrays else scores.get(method)
            if risk is None or not detail.get('valid'):
                detail['valid'] = False; methods[arm] = detail; continue
            tokens = windows_to_tokens(plan, risk)
            steps = np.array([np.max(tokens[a:b]) for a, b in zip(arrays['step_starts'], arrays['step_ends'])])
            arrays[arm+'__window'], arrays[arm+'__risk'] = risk, steps
            try:
                gate = dict(parent['readout']) if replay and parent.get('readout_valid') else mixture_readout(risk[original], steps)
                detail['gate_readout'] = gate
                detail['prediction'] = int(np.argmax(steps)) if gate['prediction'] != -1 else -1
                detail['fixed_parent_gate_prediction'] = (
                    int(np.argmax(steps)) if parent['readout']['prediction'] != -1 else -1
                ) if parent.get('readout_valid') and parent.get('valid') else None
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                detail.update(valid=False, reason=str(exc))
            methods[arm] = detail
        fit_details[selector] = {'shared': shared, 'seconds': time.monotonic()-started, 'parent_replay': replay}
    return arrays, json_safe(methods), json_safe({'selectors': diagnostics, 'fits': fit_details,
        'perturbation_seconds': perturb_seconds, 'sampling_eligible': budget(len(original)) < len(original)})
