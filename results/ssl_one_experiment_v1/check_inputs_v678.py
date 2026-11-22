"""Independent check of the V6/V7/V8 driver inputs written by make_variants_v678.py (synthetic benchmark v2), BEFORE any
GPU job. CPU, seconds. No labels are read. It reads the outputs and the source answers, and never writes into --inputs.

Checks, per benchmark x variant:
- C1 sidecar: one line per dev parent of SPLIT.json; built uids are '<parent>__v6|v7|v8'; side == 'dev'; 1 <= k <= n.
- C2 rows: the driver rows equal the built sidecar lines (same uids, same order); every row has the parent's step count.
- C3 edits:
  - V6: only step k differs from the parent;
  - V7: only step k differs, and it equals the donor's step text;
  - V8: steps 1..k-1 equal the parent and step k differs;
  - V8 has the same parents and k as V6.
- C4 preflight: the preC rows are the C rows of the first 12 built parents with k >= 2.
- C5 counts and coverage (expected from the local test of 2026-11-22; a shortfall stops the run):
  - PB: V6 >= 1,600, V7 >= 350, V8 == V6;
  - PRMBench: V6 >= 3,400, V7 >= 3,300, V8 == V6;
  - k = 1 rows: at least 100 in every PB V6 / PRMBench V6 / PRMBench V7 run.

Prints one line per check and 'INPUT CHECK PASS' or 'INPUT CHECK FAIL'; exit code 0 only on PASS.

    python check_inputs_v678.py --pb-answers A.json --prmb-meta prmbench_telemetry.pkl|rows.json --inputs $W/inputs \
        [--out $W/inputs/CHECK_INPUTS_V678.json]
"""
import argparse, json, pickle, sys
from pathlib import Path

import make_variants as MV

VARS = ('V6', 'V7', 'V8')
MIN_ROWS = {('processbench', 'V6'): 1600, ('processbench', 'V7'): 350, ('prmbench', 'V6'): 3400, ('prmbench', 'V7'): 3300}
MIN_K1 = {('processbench', 'V6'): 100, ('prmbench', 'V6'): 100, ('prmbench', 'V7'): 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pb-answers', required=True); ap.add_argument('--prmb-meta', required=True)
    ap.add_argument('--inputs', required=True); ap.add_argument('--out', default=None)
    a = ap.parse_args(); inp = Path(a.inputs); R = {}; ok = True
    SP = json.load(open(inp / 'SPLIT.json'))
    src = {'processbench': {r['uid']: r for r in json.load(open(a.pb_answers))},
           'prmbench': {str(r['idx']): r for r in MV.load_prmb_rows(a.prmb_meta)}}

    def check(name, cond, detail=None):
        nonlocal ok
        ok &= bool(cond); R[name] = dict(pass_=bool(cond), detail=detail)
        print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f'  {json.dumps(detail)}' if detail is not None else ''), flush=True)

    for bench in ('processbench', 'prmbench'):
        S = src[bench]; sp = SP[bench]
        dev = [u for u in S if sp[u][1] == 'dev']
        side = {}
        for V in VARS:
            lines = [json.loads(x) for x in open(inp / f'SIDECAR_{bench}_{V}.jsonl')]
            built = [s for s in lines if 'skip' not in s]; side[V] = {s['parent_uid']: s for s in built}
            c1 = (sorted(s['parent_uid'] for s in lines) == sorted(dev)
                  and all(s['uid'] == f"{s['parent_uid']}__{V.lower()}" and s['side'] == 'dev' for s in lines)
                  and all(1 <= s['k'] <= len(S[s['parent_uid']]['steps']) for s in built))
            check(f'C1 {bench} {V} sidecar', c1, dict(lines=len(lines), built=len(built), dev=len(dev)))
            if bench == 'processbench':
                rows = json.load(open(inp / f'pb_{V}_answers.json'))
                pre = json.load(open(inp / f'pb_preC_{V}_answers.json')); pre_ids = [r['uid'] for r in pre]
            else:
                m = pickle.load(open(inp / f'prmbench_{V}_meta.pkl', 'rb')); rows = [m[i] for i in sorted(m)]
                for r in rows: r['uid'] = r['idx']
                p = pickle.load(open(inp / f'prmbench_preC_{V}_meta.pkl', 'rb')); pre_ids = [str(p[i]['idx']) for i in sorted(p)]
            c2 = ([r['uid'] for r in rows] == [s['uid'] for s in built]
                  and all(len(r['steps']) == len(S[s['parent_uid']]['steps']) for r, s in zip(rows, built)))
            check(f'C2 {bench} {V} rows', c2, dict(rows=len(rows)))
            bad = []
            for r, s in zip(rows, built):
                P = S[s['parent_uid']]['steps']; C = r['steps']; k = s['k']
                diff = [j + 1 for j in range(len(P)) if P[j] != C[j]]
                if V in ('V6', 'V7'):
                    good = diff == [k]
                    if V == 'V7' and good:
                        good = C[k - 1] == S[s['donor_uid']]['steps'][s['donor_step'] - 1] and s['donor_step'] != k
                else:
                    good = bool(diff) and diff[0] == k
                if not good: bad.append(s['uid'])
            check(f'C3 {bench} {V} edits', not bad, dict(n_bad=len(bad), first_bad=bad[:3]))
            exp_pre = [s['parent_uid'] for s in built if s['k'] >= 2][:12]
            check(f'C4 {bench} {V} preflight', pre_ids == exp_pre and len(pre_ids) == 12, dict(n=len(pre_ids)))
            k1 = sum(s['k'] == 1 for s in built)
            need = MIN_ROWS.get((bench, V)); need_k1 = MIN_K1.get((bench, V))
            c5 = (need is None or len(built) >= need) and (need_k1 is None or k1 >= need_k1)
            check(f'C5 {bench} {V} counts', c5, dict(rows=len(built), min_rows=need, k1_rows=k1, min_k1=need_k1))
        same = sorted(side['V6']) == sorted(side['V8']) and all(side['V6'][u]['k'] == side['V8'][u]['k'] for u in side['V6'])
        check(f'C3 {bench} V8 has the V6 parents and k', same)
    R['INPUT_CHECK_PASS'] = bool(ok)
    if a.out:
        out = Path(a.out)
        if out.exists(): sys.exit(f'REFUSING to overwrite {out}')
        json.dump(R, open(out, 'w'), indent=1)
    print('INPUT CHECK', 'PASS' if ok else 'FAIL', flush=True)
    sys.exit(0 if ok else 3)


if __name__ == '__main__':
    main()
