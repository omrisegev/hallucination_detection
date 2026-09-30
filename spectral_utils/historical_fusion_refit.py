"""Corrected group-fold refits of five historical active-23 controls.

This adapter has no label API. Historical functions are loaded read-only in a
separate namespace; their imported source files are bound by the run manifest.
The main answer-only scorer and Claude's checkout are never modified.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types

import numpy as np

ARMS = (
    'iu_c2_s25_l2_exoff', 'iu_c2_s25_l2_exon', 'equal_all23',
    'fixed_family_cont_unguarded', 'prov5_cont',
)
RETAINED = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16,
            19, 20, 21, 23, 24, 25, 26, 27, 28)


def reference_modules(checkout):
    """Avoid the inference-heavy package initializer; preserve relative imports."""
    source = Path(checkout).resolve() / 'spectral_utils'
    name = '_localization_historical_reference'
    sys.dont_write_bytecode = True
    for suffix, directory in (('', source), ('.reconstruction_benchmark', source / 'reconstruction_benchmark')):
        key = name + suffix
        if key not in sys.modules:
            package = types.ModuleType(key)
            package.__path__ = [str(directory)]
            package.__package__ = key
            sys.modules[key] = package
        else:
            assert sys.modules[key].__path__ == [str(directory)]
    modules = {key: importlib.import_module(name + '.' + key) for key in (
        'joint_lsml_localization', 'joint_lsml_v2_localization',
        'feature_contract', 'fixed_application_pipelines', 'upcr', 'fusion_utils', 'joint_lsml',
    )}
    files = sorted({str(Path(module.__file__).resolve())
                    for key, module in list(sys.modules.items())
                    if key.startswith(name + '.') and getattr(module, '__file__', None)})
    return modules, files


def prepare(cell, fit_mask, reference):
    definitions = reference['fixed_application_pipelines']
    return reference['joint_lsml_localization'].prepare_active23(
        cell['raw'], cell['token_offsets'], list(map(str, cell['row_ids'])),
        retained_indices=RETAINED,
        confidence_signs_29=reference['feature_contract'].confidence_sign_vector(definitions.SHARED_GLOBAL_FEATURES),
        stream_names_29=definitions.SHARED_TOKEN_VIEWS,
        raw_feature_names_29=definitions.SHARED_GLOBAL_FEATURES,
        fit_row_mask=fit_mask,
    )


def fit(preparation, reference):
    """Exact deployed configurations and historical donor orientation, no search."""
    z = preparation.standardized_fit
    v2 = reference['joint_lsml_v2_localization']
    labels = v2.provenance_labels(preparation.family_names)
    entropy_index = preparation.feature_names.index('entropy_series')
    weights, metadata, failures = {}, {}, {}
    configs = dict(v2.IU_ROSTER)
    for arm in ARMS:
        try:
            info = {}
            if arm.startswith('iu_'):
                model = reference['upcr'].upcr_fit(z.T, **configs[arm])
                weight = model.w
                # Historical behavior does not silently substitute another estimator.
                info = {'config': configs[arm], 'abstained': bool(model.abstained)}
            elif arm == 'equal_all23':
                weight = np.ones(z.shape[1]) / z.shape[1]
            elif arm == 'prov5_cont':
                weight, info = v2._cont_weight(z, labels, gates=None)
            else:
                _, details = reference['fusion_utils'].lsml_continuous(
                    *[z[:, i] for i in range(z.shape[1])], groups=labels,
                    compute_score_matrix=False, small_m_guard=False)
                weight = reference['joint_lsml'].continuous_lsml_weight_vector(details, z.shape[1])
                info = {'small_m_guard': False}
            weights[arm], orientation = v2.donor_scale_orient(weight, z, entropy_index=entropy_index)
            metadata[arm] = {**info, **orientation}
        except Exception as error:
            failures[arm] = type(error).__name__ + ': ' + str(error)
    return weights, metadata, failures


def summarize_risk(risk, cell):
    """Historical B0: top-10 mean locator, span-max ranker, whole-answer max gate."""
    starts, ends = cell['step_starts'], cell['step_ends']
    top, maximum = np.empty(len(starts)), np.empty(len(starts))
    for i, (lo, hi) in enumerate(zip(starts, ends)):
        values = risk[int(lo):int(hi)]
        values = values[np.isfinite(values)]
        if not len(values):
            top[i] = maximum[i] = np.nan
            continue
        k = min(10, len(values))
        top[i] = np.partition(values, -k)[-k:].mean()
        maximum[i] = values.max()
    offsets = cell['token_offsets']
    detector = np.array([np.nanmax(risk[int(lo):int(hi)]) for lo, hi in zip(offsets[:-1], offsets[1:])])
    return top.astype(np.float32), maximum.astype(np.float32), detector.astype(np.float32)


def score(preparation, cell, weights, evaluation_rows):
    """Save only the declared held-out rows; keep all steps and failed rows."""
    rows = np.asarray(evaluation_rows, dtype=np.int64)
    offsets = cell['step_row_offsets']
    steps = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in rows])
    output = dict(rows=rows, steps=steps)
    for arm, weight in weights.items():
        top, maximum, detector = summarize_risk(preparation.token_risk(weight), cell)
        output.update({arm+'__w': weight, arm+'__top10': top[steps],
                       arm+'__spanmax': maximum[steps], arm+'__detector': detector[rows]})
    return output


def fold_masks(groups, outer_map, inner_map, outer, inner=None):
    assignments = np.array([outer_map[g] for g in groups])
    excluded = assignments == outer
    if inner is None:
        train, evaluate = ~excluded, excluded
    else:
        assignments_inner = np.array([inner_map.get(g, -1) for g in groups])
        assert np.all(assignments_inner[~excluded] >= 0)
        train = ~excluded & (assignments_inner != inner)
        evaluate = ~excluded & (assignments_inner == inner)
    assert not (train & evaluate).any()
    assert not (train & excluded).any()
    assert set(np.asarray(groups)[train]).isdisjoint(set(np.asarray(groups)[evaluate]))
    assert train.any() and evaluate.any()
    return train, evaluate
