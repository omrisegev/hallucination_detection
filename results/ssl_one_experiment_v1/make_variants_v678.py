"""Synthetic benchmark v2: three known-error variants with the error at ANY step, step 1 (0-based step 0) included.
DEV answers only. Omri 2026-11-22 (AIRCC run of a second synthetic benchmark; plan: results/synthetic_benchmark_v2/PLAN.md
on branch claude/ssl-step-threshold-2026-11-19).

Why: the five-agent ProcessBench F1 search (Step 530, results/pb_f1_search_v1/) found that the matched-prefix
pseudo-benchmark built from V3/V4 (make_variants_v34.py) has structural blind spots:
- its corrupted step is always the LAST step of a prefix;
- V3/V4 never corrupt step 1, while 11.8% of erroneous ProcessBench dev answers fail at their first step;
- prefixes hide the answer-level score.

These variants keep the whole answer (the steps after k are present), so pseudo-benchmarks on full answers become
possible.

No gold label is read. The PB/PRMBench label fields copied into the driver rows are placeholders, as in
make_variants_v34.py. Group and side come from the stored SPLIT.json. The test side is not built. This script writes
only new files and refuses to overwrite a file.

V6 numeric edit anywhere. The V3 rule (the LAST editable number of step k: integer +-1, decimal last digit +-1, the
  make_variants_v34.editable_numbers definition), with k uniform over EVERY non-empty step 1..n that has an editable
  number (V3: steps 2..n only). Steps after k are unchanged. Answers without any editable number are skipped.
V7 on-topic donor anywhere. The V4 rule (step k replaced by a step of another answer to the same question group, donor
  position != k, donor text different from every parent step, levels L0/L1/L2, reuse tiers), with k uniform over
  every non-empty step 1..n (V4: 2..n). Caveat as V4: the donor step can be a correct alternative.
V8 propagated numbers anywhere. The V5 rule (every editable number from step k to the end gets a new value, the same
  original text always the same new text), on the V6 parents and the V6 k. So V6 and V8 differ only in whether the
  change propagates.

All three keep the parent's step count (required by the frozen PRMBench bank builder, aircc/prmb_variant_bank.py). k
is 1-based in the sidecar, as in V3-V5 (ssl_data.load_pool maps it to the local 0-based index).

The preflight rows (first 12) are the first built rows with k >= 2, so smoke_check.py S2 has prefix pairs. A k = 1
row has no equal prefix.

Outputs (--out, normally $W/inputs; nothing existing is overwritten):
  pb_V6_answers.json, pb_V7_answers.json, pb_V8_answers.json                          driver inputs (dev parents)
  prmbench_V6_meta.pkl, prmbench_V7_meta.pkl, prmbench_V8_meta.pkl
  pb_preC_V{6,7,8}_answers.json, prmbench_preC_V{6,7,8}_meta.pkl                       C rows of the preflight parents
  SIDECAR_<bench>_V{6,7,8}.jsonl                                                       one line per DEV parent
  VARIANTS_V678_REPORT.json                                                            counts, skips, k distribution

    python make_variants_v678.py --pb-answers A.json --prmb-meta prmbench_telemetry.pkl|rows.json --inputs $W/inputs --out $W/inputs
"""
import argparse, json, pickle, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import make_variants as MV
import make_variants_v34 as V34
import make_variants_v5 as V5

SEED = MV.SEED
N_PRE = 12
BID = {'processbench': 0, 'prmbench': 1}
VARS = ('V6', 'V7', 'V8')


def rel(k, n):
    return round((k - 1) / (n - 1), 3) if n > 1 else 0.0


def make_v6(answers, bench, dev):
    rng = np.random.default_rng([SEED, 6, BID[bench]]); out = {}
    for i in rng.permutation(len(answers)):
        if i not in dev: continue
        steps = answers[i]['steps']; n = len(steps)
        cand = {k: V34.editable_numbers(steps[k - 1]) for k in range(1, n + 1) if steps[k - 1].strip()}
        ks = [k for k, v in cand.items() if v]
        if not ks:
            out[i] = dict(skip='no editable number in any step'); continue
        k = int(rng.choice(ks)); s, e, old = cand[k][-1]; new = V34.edit_number(old, rng)
        t = steps[k - 1]; nt = t[:s] + new + t[e:]
        assert nt != t and old != new
        ns = list(steps); ns[k - 1] = nt
        out[i] = dict(steps=ns, k=k, step0=k == 1, old=old, new=new, char_start=s, rel_char=round(s / max(1, len(t)), 3),
                      n_editable_in_step=len(cand[k]), n_eligible_steps=len(ks), rel_parent=rel(k, n))
    return out


def make_v7(answers, bench, dev):
    rng = np.random.default_rng([SEED, 7, BID[bench]]); out = {}
    by_group = defaultdict(list)
    for i in sorted(dev): by_group[answers[i]['group']].append(i)
    used = Counter()
    levels = [(.2, .4), (.4, .4), (None, .4)]
    for i in rng.permutation(len(answers)):
        if i not in dev: continue
        r = answers[i]; steps = r['steps']; n = len(steps)
        ks = [k for k in range(1, n + 1) if steps[k - 1].strip()]
        sib = [a for a in by_group[r['group']] if a != i]
        if not ks:
            out[i] = dict(skip='no non-empty step'); continue
        if not sib:
            out[i] = dict(skip='no other answer to the same question'); continue
        k = int(rng.choice(ks)); rk = rel(k, n); L = float(len(steps[k - 1])); pn = {V34.norm(t) for t in steps}
        cand = []
        for a in sib:
            st = answers[a]['steps']; m = len(st)
            for j, t in enumerate(st):
                if t.strip() and j + 1 != k and V34.norm(t) not in pn:
                    cand.append((a, j, j / (m - 1) if m > 1 else 0., len(t)))
        pick = None
        for lev, (dp, dl) in enumerate(levels):
            c = [x for x in cand if abs(x[3] - L) <= dl * L and (dp is None or abs(x[2] - rk) <= dp + 1e-12)]
            if not c: continue
            u = np.array([used[(x[0], x[1])] for x in c])
            for tier in (u == 0, u < 3, np.ones(len(c), bool)):
                if tier.any():
                    idx = np.flatnonzero(tier); pick = c[int(rng.choice(idx))]; break
            if pick: break
        if pick is None:
            out[i] = dict(skip='no donor in the same question group'); continue
        a, j, rd, ld = pick; used[(a, j)] += 1
        ns = list(steps); ns[k - 1] = answers[a]['steps'][j]
        out[i] = dict(steps=ns, k=k, step0=k == 1, donor_uid=answers[a]['uid'], donor_step=j + 1, relax_level=lev,
                      n_candidates=len(cand), n_siblings=len(sib), len_ratio=round(ld / L, 3), rel_parent=rk,
                      rel_donor=round(rd, 3))
    return out, used


def make_v8(answers, bench, v6):
    rng = np.random.default_rng([SEED, 8, BID[bench]]); out = {}
    for i in sorted(v6):
        d = v6[i]
        if 'skip' in d:
            out[i] = dict(skip='V6 skipped: ' + d['skip']); continue
        steps = answers[i]['steps']
        ns, info = V5.make_v5(steps, d['k'], rng)
        assert ns[:d['k'] - 1] == steps[:d['k'] - 1] and ns[d['k'] - 1] != steps[d['k'] - 1]
        out[i] = dict(steps=ns, k=d['k'], step0=d['step0'], rel_parent=d['rel_parent'], **info)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pb-answers', required=True); ap.add_argument('--prmb-meta', required=True)
    ap.add_argument('--inputs', required=True, help='folder with SPLIT.json (the stored 2026-11-18 split)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); t0 = time.time(); out = Path(a.out); inp = Path(a.inputs); out.mkdir(parents=True, exist_ok=True)
    names = [f'pb_{V}_answers.json' for V in VARS] + [f'prmbench_{V}_meta.pkl' for V in VARS] + \
            [f'pb_preC_{V}_answers.json' for V in VARS] + [f'prmbench_preC_{V}_meta.pkl' for V in VARS] + \
            [f'SIDECAR_{b}_{V}.jsonl' for b in ('processbench', 'prmbench') for V in VARS] + ['VARIANTS_V678_REPORT.json']
    exist = [n for n in names if (out / n).exists()]
    if exist: sys.exit(f'REFUSING: outputs exist in {out}: {exist}')
    SP = json.load(open(inp / 'SPLIT.json'))
    pb = json.load(open(a.pb_answers))
    prmb = MV.load_prmb_rows(a.prmb_meta)
    rep = dict(seed=SEED, decision='Omri 2026-11-22: AIRCC run of synthetic benchmark v2 (results/synthetic_benchmark_v2/PLAN.md)',
               side='dev only', variants={'V6': 'numeric edit anywhere', 'V7': 'on-topic donor anywhere',
                                          'V8': 'propagated numbers anywhere (V6 parents and k)'})
    gate = {}
    for bench, src, key in (('processbench', pb, 'uid'), ('prmbench', prmb, 'idx')):
        sp = SP[bench]; uids = [str(r[key]) for r in src]
        missing = [u for u in uids if u not in sp]
        gate[bench] = dict(n_rows=len(uids), n_unique=len(set(uids)), missing_in_split=len(missing),
                           dev=sum(sp[u][1] == 'dev' for u in uids if u in sp),
                           dev_expected=SP['counts'][bench]['answers_dev'])
    ok = all(g['n_rows'] == g['n_unique'] and g['missing_in_split'] == 0 and g['dev'] == g['dev_expected'] for g in gate.values())
    rep['gate_split'] = gate; print('SPLIT GATE', 'PASS' if ok else 'FAIL', json.dumps(gate), flush=True)
    if not ok: sys.exit('answers do not match SPLIT.json: stop')

    for bench, src, key in (('processbench', pb, 'uid'), ('prmbench', prmb, 'idx')):
        sp = SP[bench]
        L = [dict(uid=str(r[key]), steps=list(r['steps']), group=sp[str(r[key])][0], side=sp[str(r[key])][1]) for r in src]
        dev = {i for i, x in enumerate(L) if x['side'] == 'dev'}
        v6 = make_v6(L, bench, dev); v7, used = make_v7(L, bench, dev); v8 = make_v8(L, bench, v6)
        for V, Dd in (('V6', v6), ('V7', v7), ('V8', v8)):
            rows, lines, meta, pre = [], [], {}, []
            for i in sorted(dev):
                d = Dd[i]; p = L[i]
                line = dict(uid=f"{p['uid']}__{V.lower()}", parent_uid=p['uid'], variant=V, group=p['group'], side=p['side'])
                if 'skip' in d:
                    line['skip'] = d['skip']; lines.append(line); continue
                assert len(d['steps']) == len(p['steps']), (bench, V, p['uid'])
                line.update({k: v for k, v in d.items() if k != 'steps'}); lines.append(line)
                if len(pre) < N_PRE and d['k'] >= 2: pre.append(i)
                if bench == 'processbench':
                    rows.append(dict(uid=line['uid'], question=src[i]['question'], steps=d['steps'],
                                     benchmark=src[i].get('benchmark'), source_id=p['uid']))
                else:
                    r = dict(src[i]); r['idx'] = line['uid']; r['steps'] = d['steps']      # label fields: placeholders
                    meta[len(meta)] = r
            if len(pre) < N_PRE: sys.exit(f'{bench} {V}: only {len(pre)} preflight rows with k >= 2: stop')
            if bench == 'processbench':
                V34.write_new(out / f'pb_{V}_answers.json', lambda f: json.dump(rows, open(f, 'w')))
                V34.write_new(out / f'pb_preC_{V}_answers.json', lambda f: json.dump([src[i] for i in pre], open(f, 'w')))
            else:
                V34.write_new(out / f'prmbench_{V}_meta.pkl', lambda f: pickle.dump(meta, open(f, 'wb')))
                V34.write_new(out / f'prmbench_preC_{V}_meta.pkl', lambda f: pickle.dump({n: dict(src[i]) for n, i in enumerate(pre)}, open(f, 'wb')))
            V34.write_new(out / f'SIDECAR_{bench}_{V}.jsonl', lambda f: open(f, 'w').write(''.join(json.dumps(s) + '\n' for s in lines)))
            built = [s for s in lines if 'skip' not in s]
            ks = np.array([s['k'] for s in built]); n_par = np.array([len(L[i]['steps']) for i in sorted(dev) if 'skip' not in Dd[i]])
            info = dict(dev_parents=len(dev), rows=len(built), share_built=round(len(built) / max(1, len(dev)), 4),
                        skipped=dict(Counter(s['skip'] for s in lines if 'skip' in s)),
                        preflight_parents=[L[i]['uid'] for i in pre],
                        k1_rows=int((ks == 1).sum()), k1_share=round(float((ks == 1).mean()), 4) if len(ks) else None,
                        k_rel_mean=round(float(np.mean([s['rel_parent'] for s in built])), 4) if built else None,
                        k_rel_deciles=[round(float(x), 3) for x in np.quantile([s['rel_parent'] for s in built], np.linspace(0, 1, 11))] if built else None,
                        n_steps_mean=round(float(n_par.mean()), 3) if len(n_par) else None)
            if V == 'V6':
                info['old_new_examples'] = [(s['old'], s['new']) for s in built[:8]]
                info['decimal_share'] = round(float(np.mean(['.' in s['old'] for s in built])), 4) if built else None
            elif V == 'V7':
                info['relax_level'] = dict(Counter(s['relax_level'] for s in built))
                info['donor_reuse'] = dict(Counter(int(u) for u in used.values()))
                info['len_ratio_median'] = float(np.median([s['len_ratio'] for s in built])) if built else None
            else:
                info['numbers_changed_median'] = float(np.median([s['n_numbers_changed'] for s in built])) if built else None
                info['steps_changed_mean'] = round(float(np.mean([s['n_steps_changed'] for s in built])), 3) if built else None
            rep[f'{bench}_{V}'] = info
            print(bench, V, json.dumps({k: v for k, v in info.items() if k not in ('preflight_parents', 'k_rel_deciles')}), flush=True)
    rep['seconds'] = round(time.time() - t0, 1)
    V34.write_new(out / 'VARIANTS_V678_REPORT.json', lambda f: json.dump(rep, open(f, 'w'), indent=1))
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
