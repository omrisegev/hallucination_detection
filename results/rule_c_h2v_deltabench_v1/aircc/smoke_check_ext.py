"""smoke_check.py of results/ssl_one_experiment_v1 (4acda6f6) with ONE change: the benchmark loop runs over
hard2verify / deltabench instead of processbench / prmbench (S1, S2, S3 rules, tolerances and exit codes unchanged).
Imports the helper functions of the original file."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(os.environ["HD_ROOT"]) / "results/ssl_one_experiment_v1"))
from smoke_check import load_run, tok_values, prefix_cmp, RAWQ  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True); ap.add_argument('--variants', required=True)
    ap.add_argument('--stored-readouts', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--prefix-tol', type=float, default=1e-4)
    a = ap.parse_args(); R = {'prefix_tol': a.prefix_tol}; ok = True
    VS = ('V1', 'V2', 'V3', 'V4', 'V5')
    for bench in ('hard2verify', 'deltabench'):
        runs = {}
        for V in ('C',) + VS:
            d = Path(a.runs) / f'{bench}_{V}'
            if not d.exists(): continue
            man, rows = load_run(d); runs[V] = rows
            r = dict(status=man.get('status'), n_rows=man.get('n_rows'), failed=man.get('n_failed_rows'),
                     s_per_row=round(man['elapsed_sec'] / max(1, man['n_rows']), 2), max_input_tokens=man.get('max_input_tokens'))
            r['pass'] = r['status'] == 'COMPLETE' and r['failed'] == 0; ok &= r['pass']; R[f'S1 {bench} {V}'] = r
        side = {}
        for V in VS:
            if V not in runs: continue          # this job has no V run (a C job, or another variant)
            p = Path(a.variants) / f'SIDECAR_{bench}_{V}.jsonl'
            side[V] = {s['uid']: s for s in map(json.loads, open(p)) if 'skip' not in s}
            res = []
            for u, e in runs.get(V, {}).items():
                s = side[V][u]; c = runs['C'].get(s['parent_uid'])
                if c is None: continue
                n_eq = s['k'] - 1 if V in ('V2', 'V3', 'V4', 'V5') else 1
                x = prefix_cmp(c, e, n_eq, a.prefix_tol)
                if x: res.append(dict(uid=u, **x))
            p = bool(res) and all(x['pass'] for x in res); ok &= p
            R[f'S2 {bench} {V} equal prefix'] = dict(n_pairs=len(res), pass_=p,
                                                     worst={q: max((x[q] for x in res), default=None) for q in RAWQ},
                                                     channels_worst=max((x.get('channels_max_abs_diff') or 0 for x in res), default=None),
                                                     failures=[x for x in res if not x['pass']][:3])
        st = Path(a.stored_readouts) / f'{bench}__qwen_prm_7b.npz'
        if st.exists() and 'C' in runs:
            D = np.load(st); pos = {u: i for i, u in enumerate(D['uids'].astype(str))}; off = D['offsets']
            mine, ref = [], []
            for u, e in runs['C'].items():
                if u not in pos: continue
                V, sp = tok_values(e); i = pos[u]; r10 = D['readouts'][off[i]:off[i + 1], :, 0]
                t = np.array([np.sort(V[x:y], axis=0)[-min(10, y - x):].mean(0) if y > x else np.full(V.shape[1], np.nan) for x, y in sp])
                if len(t) != len(r10): t = t[~np.isnan(t[:, 0])]
                if len(t) != len(r10): continue
                mine.append(t); ref.append(r10)
            if mine:
                m = np.concatenate(mine); r = np.concatenate(ref)
                rho = [float(spearmanr(m[:, j], r[:, j])[0]) for j in range(m.shape[1])]
                R[f'S3 {bench} C vs stored top10'] = dict(steps=len(m), spearman_min=min(rho), spearman=rho,
                                                          max_abs_diff=float(np.nanmax(np.abs(m - r))), pass_=min(rho) >= .99)
    R['ALL_STOP_RULES_PASS'] = bool(ok)
    json.dump(R, open(a.out, 'w'), indent=1, default=str)
    print(json.dumps(R, indent=1, default=str)[:4000])
    sys.exit(0 if ok else 4)


if __name__ == '__main__':
    main()
