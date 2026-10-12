"""Fixed-bank observation selection serving the existing IU/Joint fusion.

All features and output windows remain dense. Only fusion fitting rows change;
the GMM still sees all original nonoverlapping windows, as in the first pilot.
"""
from copy import deepcopy
import time
import numpy as np
from .answer_localization_v2 import moment_plan, moment_matrix
from .fusion_context_bank import context_matrix
from .fusion_prediction_quality import CORES, JOINT_CORES, fit_bank, fixed_bank
from .fusion_token_gap import ANCHORS, apply_readout, chosen_core
from .fusion_window_sampling import (SELECTORS, budget, choose_all, perturb_windows,
                                    jaccard, top_indices)

NEW_ARMS = tuple(f'sample_{selector}__{core}' for selector in SELECTORS for core in CORES)


def arm_name(selector, core):
    return f'sample_{selector}__{core}'


def check_selection(chosen, original):
    chosen = np.asarray(chosen)
    if chosen.ndim != 1 or not np.issubdtype(chosen.dtype, np.integer):
        raise ValueError('INVALID_SELECTION_TYPE')
    if not len(chosen) or np.any(chosen < 0) or np.any(chosen >= len(original)) or np.any(np.diff(chosen) <= 0):
        raise ValueError('INVALID_SELECTION_ORDER_OR_RANGE')
    return original[chosen]


def selection_support(token_count, starts, ends, selected, step_starts, step_ends):
    """Target-free token coverage; evaluation later joins official error labels."""
    covered = np.zeros(token_count, dtype=bool)
    for a, b in zip(starts[selected], ends[selected]):
        covered[a:b] = True
    fractions = np.array([covered[a:b].mean() for a, b in zip(step_starts, step_ends)])
    return covered, fractions


def score_sampling(raw, ss, ee, original_arrays, original_meta, anchor_arrays, anchor_meta, identity):
    bank = fixed_bank(original_meta['routing'])
    plan = moment_plan(len(raw), 8)
    builder = moment_matrix if bank == 'moment' else context_matrix
    values, names = builder(raw, plan)
    np.testing.assert_array_equal(values, original_arrays[bank+'__features'])
    ss, ee = np.asarray(ss, int), np.asarray(ee, int)
    if ss.shape != ee.shape or not len(ss) or np.any(ss < 0) or np.any(ee > len(raw)) or np.any(ee <= ss):
        raise ValueError('INVALID_STEP_SPANS')
    original = plan.fit_indices
    choices, diagnostic = choose_all(values[original], identity)
    eligible = budget(len(original)) < len(original)
    joint_original = bool(original_meta['methods'][bank+'__joint0']['valid'])
    arrays = dict(features=values, window_starts=plan.starts, window_ends=plan.ends,
                  fit_indices=original, step_starts=ss, step_ends=ee)
    details = dict(bank=bank, names=names, eligible=eligible, original_joint_valid=joint_original,
                   original_rows=len(original), budget=budget(len(original)), selectors=diagnostic,
                   fits={}, labels_used=False, gate_support='all_original_fit_rows')
    for selector, chosen in choices.items():
        selected = check_selection(chosen, original)
        assert len(selected) == (len(original) if selector == 'full' else budget(len(original)))
        arrays[selector+'__selected'] = selected
        _, fraction = selection_support(len(raw), plan.starts, plan.ends, selected, ss, ee)
        arrays[selector+'__step_support_fraction'] = fraction
        starts = plan.starts[selected]
        diagnostic[selector].update(n_selected=len(selected),
            quartile_counts=np.bincount(np.minimum(3, starts*4//len(raw)), minlength=4),
            largest_start_gap_tokens=int(np.diff(np.r_[0, starts, len(raw)]).max()))
    perturb_start = time.monotonic()
    if eligible:
        for replicate in (0, 1):
            perturbed_raw = perturb_windows(raw, identity, replicate)
            perturbed, _ = builder(perturbed_raw, plan)
            pert_choices, pert_detail = choose_all(perturbed[original], identity)
            for selector, chosen in choices.items():
                diagnostic[selector].setdefault('block_perturbation', []).append(dict(
                    replicate=replicate, status=pert_detail[selector]['status'],
                    jaccard=jaccard(chosen, pert_choices[selector]) if selector in pert_choices else None))
        if 'dufs_transposed' in choices:
            per_seed = diagnostic['dufs_transposed']['diagnostics']['per_seed_probabilities']
            sets = [top_indices(p, budget(len(original))) for p in per_seed]
            diagnostic['dufs_transposed']['seed_pair_jaccard'] = [
                jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i+1, len(sets))]
    details['perturbation_seconds'] = time.monotonic()-perturb_start
    methods = {}
    reference = original_meta['methods']['moment__iu']
    for selector in SELECTORS:
        started = time.monotonic()
        if selector not in choices:
            details['fits'][selector] = dict(selector_failed=True, joint_valid=False)
            for core in CORES:
                methods[arm_name(selector, core)] = dict(valid=False, decision_valid=False,
                    fixed_iu_valid=False, reason='SELECTOR_FAILED', fallback_to_sample_iu=False)
            continue
        selected = arrays[selector+'__selected']
        replay = np.array_equal(selected, original)
        if replay:
            details['fits'][selector] = dict(replay=True, joint_valid=joint_original, seconds=0.)
            for core in CORES:
                arm, source = arm_name(selector, core), ANCHORS[core]
                detail = deepcopy(anchor_meta['methods'][source])
                detail.update(anchor_replay=True, source_arm=source, bank=bank,
                    joint_fit_valid=joint_original,
                    fallback_to_sample_iu=core in JOINT_CORES and not joint_original)
                methods[arm] = detail
                if detail['valid']:
                    for suffix in ('window', 'risk'):
                        arrays[arm+'__'+suffix] = anchor_arrays[source+'__'+suffix].copy()
            continue
        try:
            fitted, risks, raw_details, shared = fit_bank(values, names, selected, identity)
            arrays.update({selector+'__'+key: value for key, value in fitted.items()})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            risks = {}; raw_details = {c: dict(valid=False, reason=str(exc)) for c in CORES}
            shared = dict(joint_valid=False, preparation_failure=str(exc), joint_failure=str(exc))
        native = {}
        for core in CORES:
            # The ORIGINAL plan is intentional: GMM support stays dense.
            step, detail = apply_readout(risks.get(core), raw_details[core], plan, ss, ee, reference)
            native[core] = detail
            if detail['valid']:
                arrays[selector+'__native_'+core+'__window'] = risks[core]
                arrays[selector+'__native_'+core+'__risk'] = step
        for core in CORES:
            source = chosen_core(core, shared['joint_valid'])
            arm = arm_name(selector, core)
            detail = deepcopy(native[source])
            detail.update(anchor_replay=False, source_arm=selector+'__native_'+source, bank=bank,
                joint_fit_valid=shared['joint_valid'], fallback_to_sample_iu=source != core,
                route='fixed_original_bank')
            if source != core:
                detail['joint_failure'] = shared.get('joint_failure')
            methods[arm] = detail
            if detail['valid']:
                for suffix in ('window', 'risk'):
                    arrays[arm+'__'+suffix] = arrays[selector+'__native_'+source+'__'+suffix].copy()
        details['fits'][selector] = dict(shared=shared, native=native, replay=False,
            joint_valid=shared['joint_valid'], seconds=time.monotonic()-started)
    assert set(methods) == set(NEW_ARMS)
    return arrays, methods, details
