"""Read-only source-definition audit; does NOT score external benchmarks.

Checks two formerly ambiguous step reducers on all materialized source answers.
This is cached-token -> step/pool parity, not raw-telemetry -> 48-channel parity.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pool-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root, pooldir = args.root.resolve(), args.pool_dir.resolve()
    paths = {
        'tokens': root/'.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz',
        'ct7': root/'results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz',
        'answers': root/'.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/OOF_STEP_SCORES.npz',
        'profiles': root/'.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/profiles_full.npy',
        'pool': pooldir/'pool_z.npy',
        'names': pooldir/'pool_names.json',
        'union': pooldir/'union_top10_profiles.npy',
        'union_builder': pooldir/'union_structure.py',
        'pool_builder': pooldir/'pool_structure.py',
        'hist_builder': pooldir/'hist29_align.py',
    }
    start = time.perf_counter()
    tok, ct = np.load(paths['tokens']), np.load(paths['ct7'])
    off = np.load(paths['answers'])['offsets']
    toff, spans, tokens = tok['token_offsets'], tok['step_spans'], tok['tokens']
    ctokens, valid = ct['tokens'], ct['valid']
    assert np.array_equal(ct['token_offsets'], toff)
    assert np.array_equal(ct['step_spans'], spans)
    assert len(off)-1 == 13769 and off[-1] == 145597
    hcol = list(tok['channels']).index('q15_H1')
    ccol = list(ct['channels']).index('chosen_std_excess')
    assert valid[:, ccol].all()
    pool = np.load(paths['pool'], mmap_mode='r')
    names = json.loads(paths['names'].read_text(encoding='utf8'))
    assert sha(paths['pool']) == 'd9abccfb561f459d2f3a6c5b4346a1f6766df7fdbaddb230338aa40349077a16'
    union = np.load(paths['union'], mmap_mode='r')
    profiles = np.load(paths['profiles'], mmap_mode='r')
    maxima = dict(chosen_raw_union=0., frac_raw_profiles=0., chosen_pool_z=0., frac_pool_z=0.)
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        ta, tb = toff[i:i+2]
        h = tokens[ta:tb, hcol].astype(float)
        c = ctokens[ta:tb, ccol].astype(float)
        lo, hi = np.percentile(h, [25, 75])
        scale = (hi-lo)/1.349
        if scale <= 1e-8:
            scale = h.std() if h.std() > 1e-8 else 1.
        hz = (h-np.median(h))/scale
        vals = []
        for l, r in spans[a:b]:
            v = c[l:r]
            k = min(10, len(v))
            vals.append([np.sort(v)[-k:].mean(), (hz[l:r] >= 1.).mean()])
        raw = np.asarray(vals)
        maxima['chosen_raw_union'] = max(maxima['chosen_raw_union'], float(np.max(np.abs(raw[:, 0]-union[a:b, 11+ccol]))))
        maxima['frac_raw_profiles'] = max(maxima['frac_raw_profiles'], float(np.max(np.abs(raw[:, 1]-profiles[a:b, 0, 10]))))
        sd = raw.std(axis=0)
        z = np.divide(raw-raw.mean(axis=0), sd, out=np.zeros_like(raw), where=sd>1e-12)
        for j, (channel, key) in enumerate((('ct7_chosen_std_excess', 'chosen_pool_z'), ('H1_frac_above_z', 'frac_pool_z'))):
            maxima[key] = max(maxima[key], float(np.max(np.abs(z[:, j]-pool[a:b, names.index(channel)]))))
        if i % 3000 == 0:
            print('checked', i, '/', len(off)-1, flush=True)
    passed = all(value <= 1e-6 for value in maxima.values())
    result = {
        'status': 'PASS' if passed else 'FAIL',
        'scope': 'FULL source cached-token to two step channels; NOT full raw-telemetry feature parity',
        'external_predictions_computed': False,
        'n_checked': len(off)-1, 'n_total': 13769, 'steps': int(off[-1]),
        'channels_checked': 2, 'locked_channels_total': 48,
        'maximum_absolute_errors': maxima,
        'seconds': time.perf_counter()-start,
        'definitions': {
            'ct7_chosen_std_excess': 'Top10 mean of cached float32 per-token standardized excess surprisal, cast to float64 before reduction. No first-step replacement, no pooled step-z-test.',
            'H1_frac_above_z': 'Fraction of step tokens >=1 after answer median/(IQR/1.349) normalization of cached q15_H1. Scale <=1e-8 falls back to std, then 1. Final step-channel answer-z.'
        },
        'inputs': {key: {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)} for key, path in paths.items()},
        'script_sha256': sha(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'}, indent=2))
    if not passed:
        raise AssertionError(maxima)


if __name__ == '__main__':
    main()
