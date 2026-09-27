"""mtg_reproduction_v1, stage 1: per-token signals for every ProcessBench answer, straight from the
raw teacher-forced pickles. Frozen protocol: results/mtg_reproduction_v1/PROTOCOL.json.

Four evidence-oriented signals per token (higher = more confident):
  shannon       -H of the renormalized top-20            (paper Eq. 9-10)
  logtoku       sum of the top-20 log-probabilities      (paper Eq. 47)
  lns_top1      max over the top-20 log-probabilities    (released code, get_logprob_curve)
  lns_provided  log-probability of the provided token    (-token_spilled_energies)
No label enters a signal; the first-error label is stored beside them for evaluation only.
"""
from pathlib import Path
import gc, hashlib, json, sys, time
import numpy as np

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
TPF = MAIN / '.worktrees/token-probability-fusion-v1'; sys.path.insert(0, str(TPF))
import scripts.run_claude_feature_bank_v1 as runner  # noqa: E402
runner.ROOT = MAIN
OUT = ROOT / 'results/mtg_reproduction_v1'; OUT.mkdir(parents=True, exist_ok=True)
TOP_K = 20

records = json.load(open(TPF / 'results/localization_full_benchmark_v3/evaluation/JOINED.json', encoding='utf8'))['records']
pb_idx = [i for i, r in enumerate(records) if str(r['cell']).startswith('pb_')]
by_cell = {}
for i in pb_idx: by_cell.setdefault(str(records[i]['cell']), []).append(i)
tok_counts = np.array([int(records[i]['tokens']) for i in pb_idx]); step_counts = np.array([int(records[i]['steps']) for i in pb_idx])
toff = np.concatenate([[0], np.cumsum(tok_counts)]); soff = np.concatenate([[0], np.cumsum(step_counts)])
pos = {i: k for k, i in enumerate(pb_idx)}
S = {nm: np.zeros(int(toff[-1]), np.float64) for nm in ['shannon', 'logtoku', 'lns_top1', 'lns_provided']}
spans = np.zeros((int(soff[-1]), 2), np.int64); target = np.full(len(pb_idx), -9, np.int64); cells = np.empty(len(pb_idx), object); uids = np.empty(len(pb_idx), object)
t0 = time.time(); src_hash = {}
for cell, path, kind, dataset in runner.source_specs():
    if cell not in by_cell: continue
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    src_hash[cell] = {'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
    src = runner.source_row_map(runner.load_pickle(path), kind=kind, dataset=dataset)
    for i in by_cell[cell]:
        k = pos[i]; row = src[str(records[i]['row_id'])]; ta, tb = int(toff[k]), int(toff[k + 1])
        lp = np.asarray(row['top_k_logprobs']['logprobs'], float)
        if lp.shape != (tb - ta, lp.shape[1]) or lp.shape[1] < TOP_K: raise ValueError(f'{cell} row {i}: logprobs {lp.shape}, tokens {tb - ta}')
        if not np.all(np.diff(lp[:, :TOP_K], axis=1) <= 1e-7): raise ValueError(f'{cell} row {i}: top-k not sorted')
        l20 = lp[:, :TOP_K]; p = np.exp(l20); q = p / p.sum(1, keepdims=True)
        S['shannon'][ta:tb] = (q * np.log(np.maximum(q, 1e-300))).sum(1)             # = -H(renormalized top-20)
        S['logtoku'][ta:tb] = l20.sum(1)
        S['lns_top1'][ta:tb] = l20.max(1)
        S['lns_provided'][ta:tb] = -np.asarray(row['token_spilled_energies'], float)
        sp = np.asarray(row['step_token_spans'], np.int64)
        if sp.shape != (step_counts[k], 2): raise ValueError(f'{cell} row {i}: spans {sp.shape} vs {step_counts[k]} steps')
        spans[soff[k]:soff[k + 1]] = sp; target[k] = int(row['label']); cells[k] = cell; uids[k] = records[i]['uid']
    del src; gc.collect()
    print(f'{cell}: {len(by_cell[cell])} answers ({time.time() - t0:.0f}s)', flush=True)
assert (target != -9).all() and all(np.isfinite(v).all() for v in S.values())
np.savez_compressed(OUT / 'TOKEN_SIGNALS.npz', token_offsets=toff, step_offsets=soff, step_spans=spans, target=target,
                    cells=cells.astype(str), uids=uids.astype(str), **{k: v.astype(np.float64) for k, v in S.items()})
json.dump({'sources': src_hash, 'answers': len(pb_idx), 'tokens': int(toff[-1]), 'steps': int(soff[-1]),
           'erroneous': int((target >= 0).sum()), 'seconds': round(time.time() - t0, 1)},
          open(OUT / 'EXTRACT_MANIFEST.json', 'w'), indent=1)
print('done', len(pb_idx), 'answers', int(toff[-1]), 'tokens', int((target >= 0).sum()), 'erroneous', f'{time.time()-t0:.0f}s')
