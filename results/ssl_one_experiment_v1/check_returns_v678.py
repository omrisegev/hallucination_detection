"""Check the returned step tables of synthetic benchmark v2 (STEPS_<bench>_V6|V7|V8.npz) against their sidecars and the
parents' C tables. CPU, under a minute. No labels are read. It reads the arrays directly, not through ssl_data, so no
gold file is needed.

  R1 the table's extract gate is PASS; every built sidecar row is in the table, and nothing else is.
  R2 every variant row has its parent's step count (all steps, before ProcessBench drops empty steps).
  R3 equal prefix, structure (stop rule): for k >= 2, steps 1..k-1 have the parent's token and character counts
     (ntok, nchar). Holds for every 2026-11-18 V3/V4 pair: 0 failures, so donor steps did not merge across the
     boundary. --prefix-report-only turns it into a report for the named variants.
  R3v equal prefix, values (reported only): the share of pairs whose prefix token readouts ('top', 'stats') differ
     from the parent's by more than --tol (default 1e-3), and the max difference (NaN padding must match).
     - Teacher forcing identical text normally reproduces the prefix exactly. On CUDA, a different TOTAL length can
       still move bf16 values: the V4 differences below have identical token structure.
     - On the 2026-11-18 tables (NaN-aware recount on the Mac, 2026-11-22):
       - V3: 0/1690 PB and 1/3489 PRMBench pairs, max .0625;
       - V4: 13/421 PB and 341/3418 PRMBench pairs, max .43 / .50.
     - The analysis drops, or marks, the listed rows.
  R4 the corrupted step k differs from the parent's in its token readouts. Reported as a share; expected 1.0.
  R5 coverage: rows and k = 1 rows per table must match the input report (VARIANTS_V678_REPORT.json).

Prints one line per check and 'RETURN CHECK PASS' or 'RETURN CHECK FAIL'; exit code 0 only on PASS.
Lists of rows with a prefix difference go to --out, for the analysis.

    python check_returns_v678.py --W <dir with steps/ and inputs/> [--variants V6 V7 V8] [--tol 1e-3] [--out FILE.json]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np


def load(W, bench, V):
    F = np.load(Path(W) / 'steps' / f'STEPS_{bench}_{V}.npz')
    info = json.loads(str(F['info']))
    g = info.get('gate'); gate_ok = (g.get('status') == 'PASS') if isinstance(g, dict) else ('PASS' in str(g) or 'max_abs_diff' in str(g))
    uids = F['uids'].astype(str)
    return dict(gate_ok=gate_ok, uids=uids, pos={u: i for i, u in enumerate(uids)}, off=F['offsets'].astype(np.int64),
                top=F['top'], stats=F['stats'], step5=F['step5'], ntok=F['ntok'], nchar=F['nchar'])


def maxdiff(x, y):
    """Max abs difference ignoring matched NaN padding; inf if the NaN patterns differ."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    nx, ny = np.isnan(x), np.isnan(y)
    if nx.shape != ny.shape or not np.array_equal(nx, ny): return float('inf')
    d = np.abs(x - y)[~nx]
    return float(d.max()) if d.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', required=True); ap.add_argument('--variants', nargs='+', default=['V6', 'V7', 'V8'])
    ap.add_argument('--tol', type=float, default=1e-3); ap.add_argument('--out', default=None)
    ap.add_argument('--prefix-report-only', nargs='*', default=[], help='variants whose R3 is reported, not a stop rule')
    args = ap.parse_args(); W = Path(args.W); R = {}; ok = True; lists = {}

    def check(name, cond, detail=None, stop=True):
        nonlocal ok
        if stop: ok &= bool(cond)
        R[name] = dict(pass_=bool(cond), stop_rule=stop, detail=detail)
        tag = ('PASS' if cond else 'FAIL') if stop else 'INFO'
        print(f'{tag}  {name}' + (f'  {json.dumps(detail)}' if detail is not None else ''), flush=True)

    rep_path = W / 'inputs' / 'VARIANTS_V678_REPORT.json'
    rep = json.load(open(rep_path)) if rep_path.exists() else {}
    for bench in ('processbench', 'prmbench'):
        C = load(W, bench, 'C')
        for V in args.variants:
            T = load(W, bench, V)
            lines = [json.loads(x) for x in open(W / 'inputs' / f'SIDECAR_{bench}_{V}.jsonl')]
            built = {s['uid']: s for s in lines if 'skip' not in s}
            check(f'R1 {bench} {V} gate and rows', T['gate_ok'] and set(T['uids']) == set(built),
                  dict(gate_ok=T['gate_ok'], table_rows=len(T['uids']), sidecar_built=len(built),
                       missing=len(set(built) - set(T['uids'])), extra=len(set(T['uids']) - set(built))))
            bad_n, bad_struct, bad_val, worst, k_changed, n_pre, k1 = [], [], [], 0.0, [], 0, 0
            for u, s in built.items():
                if u not in T['pos']: continue
                i = T['pos'][u]; j = C['pos'].get(s['parent_uid'])
                if j is None: bad_n.append(u); continue
                a0, a1 = T['off'][i], T['off'][i + 1]; c0, c1 = C['off'][j], C['off'][j + 1]
                if a1 - a0 != c1 - c0: bad_n.append(u); continue
                k = s['k']; k1 += k == 1
                if k >= 2:
                    n_pre += 1
                    same_struct = (np.array_equal(T['ntok'][a0:a0 + k - 1], C['ntok'][c0:c0 + k - 1])
                                   and np.array_equal(T['nchar'][a0:a0 + k - 1], C['nchar'][c0:c0 + k - 1]))
                    if not same_struct: bad_struct.append(u)
                    d = max(maxdiff(T[q][a0:a0 + k - 1], C[q][c0:c0 + k - 1]) for q in ('top', 'stats'))
                    worst = max(worst, d)
                    if d > args.tol: bad_val.append(u)
                k_changed.append(max(maxdiff(T[q][a0 + k - 1], C[q][c0 + k - 1]) for q in ('top', 'stats')) > 1e-6)
            check(f'R2 {bench} {V} step counts', not bad_n, dict(n_bad=len(bad_n), first=bad_n[:3]))
            stop = V not in args.prefix_report_only
            check(f'R3 {bench} {V} equal prefix structure (k >= 2)', n_pre > 0 and (not stop or not bad_struct),
                  dict(pairs=n_pre, n_bad=len(bad_struct), first=bad_struct[:3]), stop=stop or n_pre == 0)
            check(f'R3v {bench} {V} equal prefix values (k >= 2)', True,
                  dict(pairs=n_pre, n_above_tol=len(bad_val), share=round(len(bad_val) / max(1, n_pre), 4),
                       max_abs_diff=worst if np.isfinite(worst) else 'inf (NaN pattern differs)'), stop=False)
            lists[f'{bench}_{V}'] = dict(prefix_structure_differs=bad_struct, prefix_values_above_tol=bad_val)
            check(f'R4 {bench} {V} step k changed', True,
                  dict(share=round(float(np.mean(k_changed)), 4) if k_changed else None), stop=False)
            exp = rep.get(f'{bench}_{V}', {})
            same = (not exp) or (exp.get('rows') == len(built) and exp.get('k1_rows') == k1)
            check(f'R5 {bench} {V} coverage', same, dict(rows=len(built), k1_rows=k1, report_rows=exp.get('rows'),
                                                          report_k1=exp.get('k1_rows')))
    R['RETURN_CHECK_PASS'] = bool(ok); R['row_lists'] = lists
    if args.out:
        out = Path(args.out)
        if out.exists(): sys.exit(f'REFUSING to overwrite {out}')
        json.dump(R, open(out, 'w'), indent=1)
    print('RETURN CHECK', 'PASS' if ok else 'FAIL', flush=True)
    sys.exit(0 if ok else 3)


if __name__ == '__main__':
    main()
