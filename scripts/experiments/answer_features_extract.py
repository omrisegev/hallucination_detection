"""answer_gate_v1: label-free answer-level feature pool (27 views) for all 13,769 answers.
Protocol results/answer_gate_v1/PROTOCOL.json (frozen f745f1834). No label is read here.

    python -B scripts/experiments/answer_features_extract.py [--workers 4]
"""
import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spectral_utils.feature_utils import FEAT_NAMES, compute_spilled_energy_features, extract_all_features  # noqa: E402

TOK = MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz'
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
OUT = ROOT / 'results/answer_gate_v1'
ENERGY = ['epr_energy', 'sw_var_peak_energy', 'cusum_max_energy', 'min_energy']
MEANS = [('varentropy', 'q15_VE1'), ('logprob_margin', 'logprob_margin'), ('tail50_mass', 'true_tail50')]
NAMES = list(FEAT_NAMES) + ENERGY + [m for m, _ in MEANS]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def features(block, ch):
    H = block[:, ch.index('q15_H1')].astype(float); cs = block[:, ch.index('chosen_surprisal')].astype(float)
    en = block[:, ch.index('energy_level')].astype(float)
    f = extract_all_features(H, spilled_energies=cs) or {}
    e = compute_spilled_energy_features(en)
    f.update({'epr_energy': e['epr_spilled'], 'sw_var_peak_energy': e['sw_var_peak_spilled'],
              'cusum_max_energy': e['cusum_max_spilled'], 'min_energy': e['min_spilled']})
    for name, c in MEANS:
        f[name] = float(np.mean(block[:, ch.index(c)]))
    return [float(f.get(n, np.nan)) for n in NAMES]


def work(args):
    blocks, ch = args
    return [features(b, ch) for b in blocks]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=4); a = ap.parse_args()
    t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Zs = np.load(R / 'OOF_STEP_SCORES.npz')
    off = Zs['offsets']; ns = np.diff(off); n = len(ans)
    z = np.load(TOK, allow_pickle=True); ch = [str(c) for c in z['channels']]; to = z['token_offsets']; spans = z['step_spans']
    gates = {'answers_equal': len(to) - 1 == n, 'steps_equal': len(spans) == int(off[-1])}
    ntok = np.diff(to); aid = np.repeat(np.arange(n), ns)
    gates['spans_within_answer'] = bool(np.all(spans[:, 0] >= 0) and np.all(spans[:, 1] <= ntok[aid]) and np.all(spans[:, 1] > spans[:, 0]))
    first = off[:-1]; gates['first_span_starts_early'] = bool(np.all(spans[first, 0] <= 5))
    gates['spans_non_decreasing_within_answer'] = bool(np.all((np.diff(spans[:, 0]) >= 0) | (np.diff(aid) != 0)))
    if not all(gates.values()): raise SystemExit(f'STOP alignment: {gates}')
    tokens = z['tokens']
    idx = np.array_split(np.arange(n), a.workers * 16)
    jobs = [([tokens[to[i]:to[i + 1]] for i in chunk], ch) for chunk in idx]
    with Pool(a.workers) as pool:
        res = pool.map(work, jobs)
    X = np.array([row for part in res for row in part], dtype=float)
    assert X.shape == (n, len(NAMES))
    np.savez_compressed(OUT / 'ANSWER_FEATURES.npz', X=X, names=np.array(NAMES), ids=ans.id.to_numpy().astype(str),
                        cells=ans.cell.to_numpy().astype(str), folds=ans.fold.to_numpy())
    man = {'names': NAMES, 'n_answers': n, 'gates': gates, 'nonfinite_per_feature': {nm: int((~np.isfinite(X[:, j])).sum()) for j, nm in enumerate(NAMES)},
           'token_file': str(TOK), 'token_file_sha256': sha(TOK), 'oof_answers_sha256': sha(R / 'OOF_ANSWERS.csv'),
           'script_sha256': sha(Path(__file__)), 'feature_utils_sha256': sha(ROOT / 'spectral_utils/feature_utils.py'),
           'seconds': time.perf_counter() - t0, 'labels_read': False}
    (OUT / 'FEATURE_MANIFEST.json').write_text(json.dumps(man, indent=1), encoding='utf8')
    print(json.dumps({k: man[k] for k in ('gates', 'nonfinite_per_feature', 'seconds')}, indent=1))


if __name__ == '__main__':
    main()
