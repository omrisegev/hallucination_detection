"""Evaluate Qwen2.5-Math-PRM-7B step rewards on Socratic-PRMBench and Hard2Verify.

Rules: NATIVE_0.5 (reward >= 0.5 -> valid, every step incl. empty ones) and ANSWER_Z_Q80 on the
risk 1 - reward over the non-empty steps (empty steps predicted incorrect, exactly as our methods).
Within-answer AUC on risk, include & non-empty mask, same evaluator code path as lr_transfer.py.
Usage: python prm_evaluate.py <dir with socratic/external_prm.pkl and hard2verify/external_prm.pkl>
"""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import lr_transfer as L  # noqa: E402

PRM_DIR = Path(sys.argv[1])
rows = []
for bench, cell in L.CELLS.items():
    cache = pickle.load(open(PRM_DIR / bench / 'external_prm.pkl', 'rb'))
    E = np.load(L.INP / f'{cell}_8b_FEATURES.npz', allow_pickle=True)   # uid order + non-empty mask only
    uids = [str(u) for u in E['uids']]; lens = E['nonempty_lengths']; ne = E['nonempty'].astype(bool)
    assert set(cache) == set(uids), (len(cache), len(uids))
    reward = np.concatenate([np.asarray(cache[u]['rewards'], float) for u in uids])
    assert all(cache[u]['n_steps'] == l for u, l in zip(uids, lens)) and len(reward) == lens.sum()
    risk = 1 - reward
    gold = {g['uid']: g for g in json.loads((L.GOLD / f'{bench}.json').read_text(encoding='utf8'))}
    starts = np.r_[0, np.cumsum(lens)]
    # NATIVE_0.5 on all steps
    counts, aucs = [], []
    for i, u in enumerate(uids):
        g = gold[u]; a, b = starts[i], starts[i + 1]; inc = np.asarray(g['include'], bool)
        counts.append(L.confusion(g['correct'], (reward[a:b] >= .5).astype(int), inc))
        v = L.within_auc(g['correct'], risk[a:b], inc & ne[a:b]); aucs.append(np.nan if v is None else v)
    rows.append(dict(benchmark=bench, model='-', method='Qwen2.5-Math-PRM-7B (supervised PRM)', rule='NATIVE_0.5',
                     official_metric=L.summary(np.array(counts), bench)['metric'], within_auc=float(np.nanmean(aucs)), n_answers=len(uids)))
    # ANSWER_Z_Q80 on non-empty steps
    s = risk[ne]; off = np.r_[0, np.cumsum(np.add.reduceat(ne.astype(int), starts[:-1]))]
    m, a, n = L.evaluate(bench, E, s, L.az_q80(s, off))
    rows.append(dict(benchmark=bench, model='-', method='Qwen2.5-Math-PRM-7B (supervised PRM)', rule='ANSWER_Z_Q80',
                     official_metric=m, within_auc=a, n_answers=n))
    over = sum(cache[u]['n_input_tokens'] > 4096 for u in uids)
    print(bench, rows[-2:], 'rows>4096 tokens:', over, 'max tokens', max(cache[u]['n_input_tokens'] for u in uids))
pd.DataFrame(rows).to_csv(OUT / 'PRM_EXTERNAL.csv', index=False)
res = pd.read_csv(OUT / 'RESULTS.csv'); res = res[~res.method.str.startswith('Qwen2.5-Math-PRM')]
pd.concat([res, pd.DataFrame(rows)]).sort_values(['benchmark', 'model', 'method', 'rule']).to_csv(OUT / 'RESULTS.csv', index=False)
