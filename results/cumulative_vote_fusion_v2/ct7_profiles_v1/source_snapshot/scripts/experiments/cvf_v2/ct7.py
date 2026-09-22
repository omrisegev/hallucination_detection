"""Fixed CT7 profiles through the unchanged cumulative-vote-v2 experiment."""
import argparse
import csv
import html
import json
import pickle
import time
from pathlib import Path

import numpy as np

from .core import ARMS, encode, location, standardize
from .data import Dataset, config, digest, dump
from .readout import earliest_mode
from .runner import fit_models, predictions
from .scoring import pb_metrics, prm_metrics, grid_prmscore, official
from .uncertainty import bootstrap
from spectral_utils.digitfree_broad50 import masked_answer_standardize
from spectral_utils.frozen_locator_ct7 import despiked_chosen_token_z

NAMES = ['H0lim', 've0', 've0.75', 've1', 'H0lim_prefix_innovation',
         'bocpd_residual', 'chosen_token_z_despiked']


def load(path):
    path = Path(path).resolve()
    settings = json.loads(path.read_text(encoding='utf8'))
    d = Dataset(config(path.parent / settings['base_config']))
    d.parent_out = d.out
    d.out = (path.parent / settings['output']).resolve()
    d.out.mkdir(parents=True, exist_ok=True)
    d.settings = settings
    d.settings_path = path
    return d


def assemble(folder, key, d, width):
    result = np.full((int(d.off[-1]), width), np.nan)
    done = np.zeros(d.n, bool)
    files = sorted(folder.glob('*.npz'))
    if len(files) != 9:
        raise ValueError(f'expected nine cell caches: {folder}')
    for path in files:
        with np.load(path) as z:
            indices = z['indexes']; values = z[key]; cursor = 0
            assert values.ndim == 2 and values.shape[1] == width
            assert np.array_equal(np.sort(indices), np.flatnonzero(d.cells == path.stem))
            for i in indices:
                assert not done[i]
                a, b = d.off[i:i+2]
                result[a:b] = values[cursor:cursor+b-a]
                cursor += b-a; done[i] = True
            assert cursor == len(values)
    assert done.all()
    return result, files


def freeze(d, inputs):
    root = Path(__file__).resolve().parents[3]
    code = list(Path(__file__).parent.glob('*.py')) + [
        root/'spectral_utils/frozen_locator_ct7.py', root/'spectral_utils/digitfree_broad50.py',
        root/'spectral_utils/chosen_token_calibration.py', root/'spectral_utils/fusion_utils.py',
        root/'spectral_utils/prmbench.py', root/'scripts/experiments/run_ct7_vote_fusion_v1.py',
        root/'docs/experiments/CT7_VOTE_FUSION_V1_PROTOCOL.md']
    state = {'candidate_id': d.settings['candidate_id'], 'settings': d.settings,
             'base_config': d.c, 'code': {str(p.relative_to(root)): digest(p) for p in code},
             'inputs': {str(p): digest(p) for p in inputs}, 'channels': NAMES}
    dest = d.out/'RUN_FREEZE.json'
    if dest.exists():
        assert json.loads(dest.read_text(encoding='utf8')) == state, 'frozen inputs/code changed'
    else:
        dump(dest, state)
        for p in code:
            out = d.out/'source_snapshot'/p.relative_to(root)
            out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(p.read_bytes())


def prepare(d):
    source = Path(d.settings['profile_source'])
    top, bank_files = assemble(source/'length_explicit_ct7_v1/bank', 'top10', d, 5)
    suff, suff_files = assemble(source/'chosen_token_calibration_v1/extracted_sufficient', 'values', d, 6)
    assert np.isfinite(suff).all()
    bank = masked_answer_standardize(np.nan_to_num(top.astype(np.float32).astype(float)),
                                     np.isfinite(top), d.off)
    token = despiked_chosen_token_z(suff, d.off)
    with np.load(source/'chosen_token_calibration_v1/OOF.npz') as z:
        six = z['six_equal']
    residual = six*6-bank.sum(1)
    with np.load(source/'length_explicit_ct7_v1/bocpd.npz') as z:
        recomputed = masked_answer_standardize(z['top10'][:, None],
                        np.ones((len(bank), 1), bool), d.off)[:, 0]
    profiles = np.column_stack([bank, residual, token])
    delta = float(np.max(np.abs(profiles.mean(1)-d.references['ct7'])))
    residual_delta = float(np.max(np.abs(residual-recomputed)))
    assert delta < 1e-12 and residual_delta < 1e-8, (delta, residual_delta)
    assert np.array_equal(d.peaks(profiles.mean(1)), d.peaks(d.references['ct7']))
    assert np.isfinite(profiles).all() and profiles.shape == (int(d.off[-1]), 7)
    # Independent soft PRMB encoding must reproduce the answer-local CT7 mean.
    encoded_mean = np.concatenate([encode(profiles[a:b], 'soft', 'prm').mean(1)
                                   for a, b in zip(d.off[:-1], d.off[1:])])
    assert np.max(abs(encoded_mean-d.references['ct7'])) < 1e-10
    inputs = bank_files+suff_files+[
        source/'chosen_token_calibration_v1/OOF.npz', source/'length_explicit_ct7_v1/bocpd.npz',
        d.settings_path, d.parent_out/'INPUT_FREEZE.json', d.parent_out/'RAW_CACHE_AUDIT.json',
        d.parent_out/'MINDGAP_ADAPTER_REPLAY.npz', d.parent_out/'REPORT_MANIFEST.json']
    inputs += [Path(d.c['paths'][k]) for k in ['roster', 'joined', 'folds', 'ct7', 'ct7_manifest', 'prm_metadata']]
    freeze(d, inputs)
    if (d.out/'profiles.npy').exists():
        assert np.array_equal(np.load(d.out/'profiles.npy'), profiles)
    else:
        np.save(d.out/'profiles.npy', profiles)
    checks = {'answers': d.n, 'steps': len(bank), 'channels': NAMES,
              'ct7_mean_max_abs_difference': delta, 'identical_ct7_argmax': True,
              'bocpd_vs_independent_recompute_max_abs_difference': residual_delta,
              'bocpd_provenance': 'algebraic recovery from six_equal; not byte-exact original component',
              'soft_prm_mean_max_abs_difference': float(np.max(abs(encoded_mean-d.references['ct7']))),
              'profile_sha256': digest(d.out/'profiles.npy')}
    dump(d.out/'PROFILE_VALIDATION.json', checks)
    print('CT7 profile replay PASS', checks, flush=True)
    return [profiles[a:b] for a, b in zip(d.off[:-1], d.off[1:])]


def job(d, ps, task_name, fold, pop, stage, inner=None):
    task = 'prm' if task_name == 'prm' else 'pb'
    selected = d.prm if task == 'prm' else d.pb & np.char.endswith(d.cells, task_name[-2:])
    train = np.flatnonzero(selected & (d.fold != fold))
    test = np.flatnonzero(selected & (d.fold == fold))
    if inner is not None:
        test = train[d.fold[train] == inner]; train = train[d.fold[train] != inner]
    if pop == 'errors':
        train = train[d.target[train] >= 0]
    assert not set(d.groups[train]) & set(d.groups[test])
    assert not set(d.groups[train]) & set(d.groups[selected & (d.fold == fold)])
    stem = f'{task_name}__fold{fold}__fixed__{pop}__{stage}'
    if inner is not None:
        stem += f'__inner{inner}'
    dest = d.out/'jobs'/stem; dest.parent.mkdir(exist_ok=True)
    if dest.with_suffix('.json').exists():
        return
    started = time.perf_counter()
    models, matrices = fit_models(d, ps, train, task, stage)
    result = predictions(d, ps, test, task, models, matrices, inner is not None)
    np.savez_compressed(dest.with_suffix('.npz'), **result)
    with open(dest.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(models, f, protocol=5)
    info = {'task': task_name, 'fold': fold, 'inner_fold': inner, 'roster': 'fixed',
            'population': pop, 'stage': stage, 'train_answers': len(train), 'test_answers': len(test),
            'train_source_groups': sorted(set(d.groups[train])), 'test_source_groups': sorted(set(d.groups[test])),
            'seconds': time.perf_counter()-started,
            'models': {enc+'__'+kind: model for (enc, kind), model in models.items()}}
    dump(dest.with_suffix('.json'), info)
    print(stem, f'done {info["seconds"]:.1f}s', flush=True)


def run(d, ps, stage, task_filter='all', fold_filter=None):
    for task in (['prm'] if stage == 'inner' else ['pb_q4', 'pb_q8', 'prm']):
        if task_filter not in ['all', task]:
            continue
        for fold in range(5):
            if fold_filter is not None and fold != fold_filter:
                continue
            if stage == 'inner':
                for inner in range(5):
                    if inner != fold:
                        for sub in ['spectral', 'em']:
                            job(d, ps, task, fold, 'all', sub, inner)
            else:
                for pop in (['all'] if task == 'prm' else ['all', 'errors']):
                    job(d, ps, task, fold, pop, stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['prepare', 'spectral', 'em', 'inner', 'evaluate', 'verify'], required=True)
    parser.add_argument('--task', default='all')
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    d = load(args.config); ps = prepare(d)
    if args.stage in ['spectral', 'em', 'inner']:
        run(d, ps, args.stage, args.task, args.fold)
    elif args.stage == 'evaluate':
        from .ct7_evaluation import evaluate
        evaluate(d, ps)
    elif args.stage == 'verify':
        from .ct7_evaluation import verify
        verify(d, ps)


if __name__ == '__main__':
    main()
