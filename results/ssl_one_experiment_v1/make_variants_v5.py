"""V5: all numbers from step k to the end get new values. DEV answers only, realism check only (Omri 2026-11-18 23:11
Israel time in the thread: insert new numbers in place of all the numbers of the answer, not only in one step).

Built on V3 (make_variants_v34.py): the same parents and the same k (read from SIDECAR_<bench>_V3.jsonl), so V5 and V3
differ only in how many numbers change. Steps 1..k-1 are unchanged (k >= 2), so step k is the known first wrong step and
the preflight prefix check has pairs. Every editable number (the same definition and the same +-1 / last-digit rule as V3)
in steps k..n gets a new value, CONSISTENTLY: the same original number text gets the same new text in every step from k on.
No gold label is read. Writes only new files and refuses to overwrite a file.

Outputs (--out, normally $W/inputs): pb_V5_answers.json, prmbench_V5_meta.pkl, pb_preC_V5_answers.json,
  prmbench_preC_V5_meta.pkl, SIDECAR_<bench>_V5.jsonl, VARIANTS_V5_REPORT.json

    python make_variants_v5.py --pb-answers A.json --prmb-meta prmbench_telemetry.pkl|rows.json --inputs $W/inputs --out $W/inputs
"""
import argparse, json, pickle, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

import make_variants as MV
import make_variants_v34 as V34

SEED = MV.SEED


def make_v5(steps, k, rng):
    """All editable numbers of steps k..n (1-based k) -> new values, one new value per distinct original text."""
    found = {j: V34.editable_numbers(steps[j]) for j in range(k - 1, len(steps))}
    olds = sorted({x[2] for v in found.values() for x in v}, key=lambda s: (len(s), s))
    new_of = {o: V34.edit_number(o, rng) for o in olds}
    ns = list(steps); n_num = 0; changed = 0
    for j, nums in found.items():
        t = steps[j]
        for s, e, o in reversed(nums):
            t = t[:s] + new_of[o] + t[e:]; n_num += 1
        if t != steps[j]: changed += 1
        ns[j] = t
    return ns, dict(n_numbers_changed=n_num, n_distinct_numbers=len(olds), n_steps_changed=changed,
                    n_steps_from_k=len(steps) - k + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pb-answers', required=True); ap.add_argument('--prmb-meta', required=True)
    ap.add_argument('--inputs', required=True, help='folder with SPLIT.json and SIDECAR_<bench>_V3.jsonl')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); t0 = time.time(); out = Path(a.out); inp = Path(a.inputs); out.mkdir(parents=True, exist_ok=True)
    names = ['pb_V5_answers.json', 'prmbench_V5_meta.pkl', 'pb_preC_V5_answers.json', 'prmbench_preC_V5_meta.pkl',
             'SIDECAR_processbench_V5.jsonl', 'SIDECAR_prmbench_V5.jsonl', 'VARIANTS_V5_REPORT.json']
    exist = [n for n in names if (out / n).exists()]
    if exist: sys.exit(f'REFUSING: outputs exist in {out}: {exist}')
    SP = json.load(open(inp / 'SPLIT.json'))
    pb = {r['uid']: r for r in json.load(open(a.pb_answers))}
    rows_ = MV.load_prmb_rows(a.prmb_meta); prmb = {str(r['idx']): r for r in rows_}
    if len(prmb) != len(rows_): sys.exit(f'PRMBench idx not unique ({len(prmb)} of {len(rows_)}): stop')
    rep = dict(seed=SEED, decision='Omri 2026-11-18 23:11 Israel (thread): new numbers in place of all numbers, not one step',
               base='same parents and k as V3', side='dev only')
    for bench, src in (('processbench', pb), ('prmbench', prmb)):
        rng = np.random.default_rng([SEED, 5, 0 if bench == 'processbench' else 1])
        v3 = [s for s in map(json.loads, open(inp / f'SIDECAR_{bench}_V3.jsonl')) if 'skip' not in s]
        rows, meta, lines, pre = [], {}, [], []
        for s in v3:
            p = s['parent_uid']; r = src[p]; k = s['k']
            assert SP[bench][p] == [s['group'], s['side']] and s['side'] == 'dev', p
            ns, info = make_v5(list(r['steps']), k, rng)
            assert ns[:k - 1] == list(r['steps'])[:k - 1] and ns[k - 1] != r['steps'][k - 1], p
            line = dict(uid=f'{p}__v5', parent_uid=p, variant='V5', group=s['group'], side=s['side'], k=k,
                        rel_parent=s['rel_parent'], **info)
            lines.append(line)
            if len(pre) < V34.N_PRE: pre.append(p)
            if bench == 'processbench':
                rows.append(dict(uid=line['uid'], question=r['question'], steps=ns, benchmark=r.get('benchmark'), source_id=p))
            else:
                e = dict(r); e['idx'] = line['uid']; e['steps'] = ns; meta[len(meta)] = e        # label fields: placeholders
        if bench == 'processbench':
            V34.write_new(out / 'pb_V5_answers.json', lambda f: json.dump(rows, open(f, 'w')))
            V34.write_new(out / 'pb_preC_V5_answers.json', lambda f: json.dump([src[p] for p in pre], open(f, 'w')))
        else:
            V34.write_new(out / 'prmbench_V5_meta.pkl', lambda f: pickle.dump(meta, open(f, 'wb')))
            V34.write_new(out / 'prmbench_preC_V5_meta.pkl', lambda f: pickle.dump({n: dict(src[p]) for n, p in enumerate(pre)}, open(f, 'wb')))
        V34.write_new(out / f'SIDECAR_{bench}_V5.jsonl', lambda f: open(f, 'w').write(''.join(json.dumps(x) + '\n' for x in lines)))
        q = lambda key: [float(x) for x in np.percentile([l[key] for l in lines], [10, 50, 90])] if lines else None
        info = dict(rows=len(lines), preflight_parents=pre, numbers_changed_p10_50_90=q('n_numbers_changed'),
                    steps_changed_p10_50_90=q('n_steps_changed'), steps_from_k_p10_50_90=q('n_steps_from_k'),
                    share_steps_from_k_changed=float(np.sum([l['n_steps_changed'] for l in lines]) / max(1, np.sum([l['n_steps_from_k'] for l in lines]))))
        rep[f'{bench}_V5'] = info
        print(bench, 'V5', json.dumps({k_: v for k_, v in info.items() if k_ != 'preflight_parents'}), flush=True)
    rep['seconds'] = round(time.time() - t0, 1)
    V34.write_new(out / 'VARIANTS_V5_REPORT.json', lambda f: json.dump(rep, open(f, 'w'), indent=1))
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
