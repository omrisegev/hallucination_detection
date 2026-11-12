"""ProcessBench metrics for a vector of predicted first errors (0-based, -1 = none): per-subset F1 (official) and
Mind the Gap SLA (exact match on erroneous answers), macro over the 4 subsets."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
GOLD = {g['uid']: g for g in json.loads((ROOT / 'results/prm_boost_processbench_v1/local/processbench_gold.json').read_text())}
SUBSETS = ('gsm8k', 'math', 'olympiadbench', 'omnimath')


def metrics(uids, pred):
    lab = np.array([GOLD[u]['label'] for u in uids]); sub = np.array([GOLD[u]['subset'] for u in uids])
    pred = np.asarray(pred); out = {}
    for s in SUBSETS:
        m = sub == s; e = m & (lab >= 0); c = m & (lab < 0)
        ea = 100 * np.mean(pred[e] == lab[e]); ca = 100 * np.mean(pred[c] == -1)
        out[s] = dict(TPR=ea, TNR=ca, F1=2 * ea * ca / (ea + ca) if ea + ca else 0., SLA=ea,
                      early=100 * np.mean((pred[e] >= 0) & (pred[e] < lab[e])), late=100 * np.mean(pred[e] > lab[e]),
                      miss=100 * np.mean(pred[e] == -1))
    out['avg F1'] = float(np.mean([out[s]['F1'] for s in SUBSETS]))
    out['macro SLA'] = float(np.mean([out[s]['SLA'] for s in SUBSETS]))
    e = lab >= 0
    out['overall SLA'] = 100 * float(np.mean(pred[e] == lab[e]))
    return out


def labels(uids):
    return np.array([GOLD[u]['label'] for u in uids])
