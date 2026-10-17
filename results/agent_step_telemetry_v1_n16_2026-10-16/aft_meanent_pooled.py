"""Pooled injected/diagnosed AUROC for the best simple baseline (mean_ent, mean over steps), same definition
as agent_baselines.py (token_entropy_full, nonempty steps), so the N16 pooled split rows have a matching baseline row."""
import glob, json, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, sys.argv[3]); from score_n16_metrics import auroc  # noqa
def f(p):
    t = json.load(open(p))['payload']['telemetry']; e = np.asarray(t['token_entropy_full'], float)
    v = [e[s:x].mean() for s, x in t['step_token_spans'] if x > s]
    return p, (float(np.mean(v)) if v else None)
if __name__ == '__main__':
    files = sorted(glob.glob(sys.argv[1] + '/aft/*.record.json'))
    with Pool(8) as P: res = P.map(f, files, chunksize=8)
    uids = [json.load(open(p))['uid'] for p, _ in res]
    L = {d['uid']: d for d in map(json.loads, open(sys.argv[2] + '/aft.jsonl'))}
    v = np.array([s for _, s in res]); src = np.array([L[u]['unsafe_source'] or 'safe' if L[u]['label'] != 'safe' else 'safe' for u in uids])
    dom = np.array([L[u]['domain'] for u in uids]); out = {'all': auroc(src != 'safe', v)}
    for s in ('injected', 'diagnosed'):
        md = ((src == s) | (src == 'safe')) & np.isin(dom, np.unique(dom[src == s])); out[s] = auroc(src[md] == s, v[md])
    print(json.dumps(out)); open(sys.argv[3] + '/AFT_MEANENT_POOLED.json', 'w').write(json.dumps(out, indent=1))
