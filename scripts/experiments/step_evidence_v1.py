#!/usr/bin/env python
"""Per-channel evidence with a position-conditional null (Step 432).

Protocol: docs/experiments/STEP_EVIDENCE_V1_PROTOCOL.md.  Every job writes the cvf_v2 job
schema ({arm}__scores / __mode / __median / __fallback on ProcessBench; __scores /
__thresholds / __threshold_q80 / __grid_valid on PRMBench) so that cvf_v2.scoring,
cvf_v2.uncertainty and cvf_v2.report run unchanged on the result.

Stages: prepare (link the frozen readout-family profiles, replay the anchors), fit (outer
jobs), inner (PRMBench inner-fold threshold jobs), report.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))
from cvf_v2.core import CHANNELS, fit_spectral  # noqa: E402
from cvf_v2.data import Dataset, config, digest, dump, prepare  # noqa: E402
from cvf_v2.readout import earliest_mode  # noqa: E402
from spectral_utils.step_evidence_v1 import (position_bins, seed_mass_equal_softmax,  # noqa: E402
                                             fit_tables_flat, evidence_flat, position_prior_only, seed_agreement)

EV_ARMS = [('seed', 'equal'),
           ('plain', 'equal'), ('plain', 'equal_std'), ('plain', 'spectral'), ('plain', 'continuous_lsml'),
           ('position', 'equal'), ('position', 'equal_std'), ('position', 'spectral'), ('position', 'continuous_lsml'),
           ('plain2', 'equal'), ('position2', 'equal'),
           ('randomseed', 'equal'), ('randomseed_position', 'equal'),
           ('prioronly', 'equal'),
           ('ceiling', 'equal'), ('ceiling_position', 'equal')]
LABEL_ARMS = {'ceiling', 'ceiling_position'}
LINKED = ['profiles.npy', 'shuffled_top5.npy', 'profiles_full.npy', 'shuffled_full.npy', 'step_lengths.npy',
          'PROFILES_COMPLETE.json', 'PROFILES_EXT_COMPLETE.json', 'MINDGAP_ADAPTER_REPLAY.npz', 'MINDGAP_REPLAY_MANIFEST.json']
TASKS = ['pb_q4', 'pb_q8', 'prm']


def link_profiles(c):
    src = Path(c['profiles_source'])
    if not src.is_absolute():
        src = (Path(c['paths']['output']).parent / src.name).resolve()
    out = Path(c['paths']['output']); out.mkdir(parents=True, exist_ok=True)
    for name in LINKED:
        s = src / name; t = out / name
        if t.exists():
            continue
        if not s.exists():
            raise FileNotFoundError(s)
        try:
            os.link(s, t)
        except OSError:
            shutil.copy2(s, t)
    ext = json.loads((out / 'PROFILES_EXT_COMPLETE.json').read_text(encoding='utf8'))
    assert digest(out / 'profiles_full.npy') == ext['profiles_full_sha256'], 'profiles_full.npy differs from its manifest'
    dump(out / 'PROFILES_SOURCE.json', {'source': str(src), 'files': {n: digest(out / n) for n in LINKED}})
    print('profiles linked from ' + str(src), flush=True)


class Population:
    """Flat step-level arrays shared by every job of one roster."""
    def __init__(self, d, readout):
        self.d = d; self.readout = readout
        r = d.readouts.index(readout)
        p = np.load(d.out / d.profile_file, mmap_mode='r')
        self.X = np.ascontiguousarray(p[:, :, r], dtype=float)
        assert np.isfinite(self.X).all(), 'evidence needs a finite readout'
        B = d.c['evidence']['position_bins']
        self.pbin = np.concatenate([position_bins(b - a, B) for a, b in zip(d.off[:-1], d.off[1:])])
        self.seed = np.empty(int(d.off[-1])); self.seed_pred = np.empty(d.n, int)
        for i, (a, b) in enumerate(zip(d.off[:-1], d.off[1:])):
            m = seed_mass_equal_softmax(self.X[a:b]); self.seed[a:b] = m; self.seed_pred[i] = earliest_mode(m)
        self.steps_of = [np.arange(a, b) for a, b in zip(d.off[:-1], d.off[1:])]

    def weights_from_pred(self, answers, pred):
        """Pseudo-positive weights: one-hot at pred[i] for the listed answers, zero elsewhere."""
        w = np.zeros(int(self.d.off[-1]))
        for i in answers:
            w[self.d.off[i] + pred[i]] = 1.
        return w


def row_weights(d, answers, task):
    """cvf_v2.core.training_matrix weighting: cell-balanced per answer on ProcessBench,
    uniform on PRMBench, spread evenly over the answer's steps; one-step answers excluded on PB."""
    usable = [int(i) for i in answers if task != 'pb' or d.off[i + 1] - d.off[i] > 1]
    counts = {c: sum(d.cells[i] == c for i in usable) for c in set(d.cells[i] for i in usable)}
    idx = []; w = []
    for i in usable:
        a, b = d.off[i:i + 2]
        mass = 1 / (len(counts) * counts[d.cells[i]]) if task == 'pb' else 1 / len(usable)
        idx.append(np.arange(a, b)); w.append(np.full(b - a, mass / (b - a)))
    return np.concatenate(idx), np.concatenate(w)


def learned(E, train_steps, w, kind):
    """Fit cvf_v2 weights on the evidence columns (rows = training steps) and score all steps."""
    model = fit_spectral(E[train_steps], w, kind)
    scores = model.predict(E)
    fail = model.status != 'ok' or not np.isfinite(scores).all()
    if fail:
        scores = E.mean(1)
    return scores, fail, model


def one_job(d, pop, task_name, fold, roster, inner_fold=None):
    task = 'prm' if task_name == 'prm' else 'pb'
    selected = d.prm if task == 'prm' else (d.pb & np.char.endswith(d.cells, task_name[-2:]))
    train = np.flatnonzero(selected & (d.fold != fold)); test = np.flatnonzero(selected & (d.fold == fold))
    if inner_fold is not None:
        test = train[d.fold[train] == inner_fold]; train = train[d.fold[train] != inner_fold]
    assert not set(d.groups[train]) & set(d.groups[test])
    assert not (set(d.groups[train]) & set(d.groups[selected & (d.fold == fold)]))
    suffix = '' if inner_fold is None else f'__inner{inner_fold}'
    stem = f'{task_name}__fold{fold}__{roster}__all__spectral{suffix}'
    dest = d.out / 'jobs' / stem; dest.parent.mkdir(exist_ok=True)
    if dest.with_suffix('.json').exists():
        return
    started = time.perf_counter()
    ev = d.c['evidence']; kw = dict(B=ev['position_bins'], bins=ev['histogram_bins'], alpha=ev['alpha'], pseudo_count=ev['pseudo_count'])
    X, pbin = pop.X, pop.pbin
    train_steps = np.concatenate([pop.steps_of[i] for i in train]); test_steps = np.concatenate([pop.steps_of[i] for i in test])
    # Pseudo-positive answers.  ProcessBench: the training answers the frozen gate opens
    # (gate-closed answers are the clean reference).  PRMBench: the frozen CT7 gate never opens
    # on a PRMBench answer (0 of 6,969), so the declared rule produced no pseudo-positives at
    # all; protocol amendment A1 (2026-09-23) takes the seed argmax of EVERY PRMBench training
    # answer as its pseudo-positive step.
    gate_open = train[d.gate[train]] if task == 'pb' else train
    rng = np.random.default_rng(np.random.SeedSequence([d.c['seed'], fold, TASKS.index(task_name), 99 if inner_fold is None else inner_fold]))
    # --- tables -------------------------------------------------------------------------
    w_seed = pop.weights_from_pred(gate_open, pop.seed_pred)
    T_seed = fit_tables_flat(X, w_seed, pbin, train_steps, **kw)
    E_plain = evidence_flat(X, T_seed, pbin, False); E_pos = evidence_flat(X, T_seed, pbin, True)
    scores = {'seed__equal': pop.seed.copy(), 'plain__equal': E_plain.sum(1), 'position__equal': E_pos.sum(1)}
    # iteration 2: pseudo-labels from the plain evidence argmax on the same gate-open answers
    pred1 = np.array([earliest_mode(scores['plain__equal'][d.off[i]:d.off[i + 1]]) for i in range(d.n)])
    T2 = fit_tables_flat(X, pop.weights_from_pred(gate_open, pred1), pbin, train_steps, **kw)
    scores['plain2__equal'] = evidence_flat(X, T2, pbin, False).sum(1); scores['position2__equal'] = evidence_flat(X, T2, pbin, True).sum(1)
    # random-seed null: one uniformly random pseudo-positive step per gate-open training answer
    rand_pred = np.array([rng.integers(0, d.off[i + 1] - d.off[i]) for i in range(d.n)])
    T_rand = fit_tables_flat(X, pop.weights_from_pred(gate_open, rand_pred), pbin, train_steps, **kw)
    scores['randomseed__equal'] = evidence_flat(X, T_rand, pbin, False).sum(1); scores['randomseed_position__equal'] = evidence_flat(X, T_rand, pbin, True).sum(1)
    # position prior alone
    scores['prioronly__equal'] = np.concatenate([position_prior_only(b - a, T_seed) for a, b in zip(d.off[:-1], d.off[1:])])
    # label ceiling (labels of TRAINING answers only; never unsupervised)
    if task == 'pb':
        err_train = train[d.target[train] >= 0]; w_true = pop.weights_from_pred(err_train, d.target)
    else:
        w_true = np.zeros(int(d.off[-1])); w_true[train_steps] = d.labels[train_steps]
    T_true = fit_tables_flat(X, w_true, pbin, train_steps, **kw)
    scores['ceiling__equal'] = evidence_flat(X, T_true, pbin, False).sum(1); scores['ceiling_position__equal'] = evidence_flat(X, T_true, pbin, True).sum(1)
    # learned weights on the evidence columns (fusion ablation)
    fit_steps, fit_w = row_weights(d, train, task); models = {}; fallbacks = {}
    for enc, E in [('plain', E_plain), ('position', E_pos)]:
        for kind, name in [('equal', 'equal_std'), ('spectral', 'spectral'), ('continuous_lsml', 'continuous_lsml')]:
            s, fail, model = learned(E, fit_steps, fit_w, kind)
            scores[f'{enc}__{name}'] = s; fallbacks[f'{enc}__{name}'] = fail; models[f'{enc}__{name}'] = model
    # --- outputs in the cvf_v2 job schema ---------------------------------------------------
    result = {'indices': np.asarray(test), 'step_offsets': np.r_[0, np.cumsum([len(pop.steps_of[i]) for i in test])]}
    qgrid = np.linspace(*d.c['inner_threshold_quantiles']); agreement = {}; eff = {}
    for name, s in scores.items():
        result[name + '__scores'] = s[test_steps]
        result[name + '__fallback'] = np.full(len(test), fallbacks.get(name, False))
        if task == 'pb':
            modes = []; medians = []
            for i in test:
                v = s[d.off[i]:d.off[i + 1]]; modes.append(earliest_mode(v)); m = v - v.min()
                m = m / m.sum() if m.sum() > 1e-12 else np.full(len(v), 1 / len(v)); medians.append(int(np.searchsorted(np.cumsum(m), .5)))
            result[name + '__mode'] = np.array(modes); result[name + '__median'] = np.array(medians)
            agreement[name] = seed_agreement(pop.seed_pred[test], np.array(modes))
        else:
            tr = s[train_steps]; thresholds = np.quantile(tr, qgrid)
            result[name + '__thresholds'] = thresholds; result[name + '__threshold_q80'] = np.array(np.quantile(tr, .8))
            if inner_fold is not None:
                result[name + '__grid_valid'] = (s[test_steps][None, :] < thresholds[:, None])
    for enc, E in [('plain', E_plain), ('position', E_pos)]:
        eff[enc] = float(np.mean([int(np.sum(np.ptp(E[d.off[i]:d.off[i + 1]], axis=0) > 1e-12)) for i in test if d.off[i + 1] - d.off[i] > 1]))
    np.savez_compressed(dest.with_suffix('.npz'), **result)
    np.savez_compressed(dest.with_suffix('.tables.npz'), **{f'seed__{k}': v for k, v in T_seed.items() if isinstance(v, np.ndarray)},
                        **{f'true__{k}': v for k, v in T_true.items() if isinstance(v, np.ndarray)})
    model_info = {name: {'kind': m.kind, 'status': m.status, 'orientation': m.orientation, 'weights': None if m.weights is None else np.asarray(m.weights).tolist(),
                         'groups': None if m.groups is None else np.asarray(m.groups).tolist(), 'cross': None if m.cross is None else np.asarray(m.cross).tolist(),
                         'diagnostics': {k: v for k, v in m.diagnostics.items() if k != 'group_search'}} for name, m in models.items()}
    for name in scores:
        model_info.setdefault(name, {'kind': name, 'status': 'ok', 'orientation': 1., 'diagnostics': {}})
    info = {'task': task_name, 'fold': fold, 'inner_fold': inner_fold, 'roster': roster, 'population': 'all', 'stage': 'spectral',
            'readouts': [pop.readout] * len(CHANNELS), 'channels': list(CHANNELS), 'label_selected_readout': False, 'label_arms': sorted(LABEL_ARMS),
            'train_answers': len(train), 'test_answers': len(test), 'gate_open_train_answers': int(len(gate_open)),
            'train_source_groups': sorted(set(d.groups[train])), 'test_source_groups': sorted(set(d.groups[test])),
            'seed_agreement_test': agreement, 'effective_channels_test_mean': eff, 'n_bins': T_seed['n_bins'].tolist(),
            'pseudo_positive_total': T_seed['positive_weight_total'],
            'pseudo_positive_rule': 'gate-open training answers' if task == 'pb' else 'every training answer (amendment A1: the CT7 gate never opens on PRMBench)',
            'iteration2_changed_pseudo_labels': float(np.mean(pred1[gate_open] != pop.seed_pred[gate_open])) if len(gate_open) else None,
            'seconds': time.perf_counter() - started, 'models': model_info}
    dump(dest.with_suffix('.json'), info)
    print(stem + f' done in {info["seconds"]:.1f}s', flush=True)


def run(d, stage, task_filter='all', fold_filter=None):
    pops = {roster: Population(d, d.c['evidence']['readout'][roster]) for roster, _ in d.rosters}
    tasks = ['prm'] if stage == 'inner' else TASKS
    for task in tasks:
        if task_filter not in ['all', task]:
            continue
        for fold in range(5):
            if fold_filter is not None and fold_filter != fold:
                continue
            for roster, _ in d.rosters:
                if stage == 'inner':
                    for inner in range(5):
                        if inner != fold:
                            one_job(d, pops[roster], task, fold, roster, inner)
                else:
                    one_job(d, pops[roster], task, fold, roster)


def code_freeze(d):
    paths = [HERE, HERE.parents[2] / 'spectral_utils/step_evidence_v1.py'] + [HERE.parent / 'cvf_v2' / n for n in ['core.py', 'data.py', 'scoring.py', 'uncertainty.py', 'report.py', 'readout.py']]
    state = {p.name: digest(p) for p in paths}; state['inputs'] = digest(d.out / 'INPUT_FREEZE.json')
    f = d.out / 'RUN_FREEZE.json'
    if f.exists() and json.loads(f.read_text()) != state:
        raise ValueError('fit code changed; cannot resume mixed versions')
    if not f.exists():
        dump(f, state)
        for p in paths:
            dst = d.out / 'source_snapshot' / p.name; dst.parent.mkdir(exist_ok=True); dst.write_bytes(p.read_bytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(HERE.parents[2] / 'configs/step_evidence_v1.json'))
    p.add_argument('--stage', choices=['prepare', 'fit', 'inner', 'report', 'all'], default='all')
    p.add_argument('--task', choices=['pb_q4', 'pb_q8', 'prm', 'all'], default='all'); p.add_argument('--fold', type=int)
    args = p.parse_args(); c = config(args.config)
    if args.stage in ['prepare', 'all']:
        link_profiles(c)
    d = Dataset(c); d.fusion_arms = list(EV_ARMS)
    if args.stage in ['prepare', 'all']:
        prepare(d)
    if args.stage in ['fit', 'inner', 'all']:
        code_freeze(d)
        for stage in (['fit', 'inner'] if args.stage == 'all' else [args.stage]):
            run(d, stage, args.task, args.fold)
    if args.stage in ['report', 'all']:
        import cvf_v2.runner as runner
        runner.expected_outer_jobs = lambda dd: 3 * 5 * len(dd.rosters)   # one stage per (task, fold, roster); no EM stage here
        from cvf_v2.report import report
        # reference_run is written relative to the config directory, like the paths block.
        ref = Path(c['reference_run'])
        if not ref.is_absolute():
            d.c['reference_run'] = str((Path(args.config).resolve().parent / ref).resolve())
        report(d, html=False)


if __name__ == '__main__':
    main()
