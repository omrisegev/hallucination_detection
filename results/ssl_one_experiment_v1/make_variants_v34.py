"""Two more known-error variants for the realism check only, DEV answers only (Omri 2026-11-18 22:24 Israel time, decision
card "Numeric edit + on-topic donor"; findings and proposal: analysis_v1/DIAG_REALISM.md section 4, options A and B).

No gold label is read to build anything (the PB gold file is read only for the subset / generator fields, as in
make_variants.py). The split is rebuilt with the functions of make_variants.py and must equal the stored SPLIT.json.
The test side is not built. This script writes only new files and refuses to overwrite a file.

V3 numeric edit (option A). k is uniform over the non-empty steps 2..n that contain an editable number. The LAST editable
  number of step k changes: integer v -> v + 1 or v - 1 (random; 0 -> 1, never negative); decimal -> its last digit
  +1 or -1 without carry (9 -> 8, 0 -> 1). Thousands commas are kept. Not editable: a number that is part of a name
  (letter, '_', '.' or ',' directly before it, e.g. x2, a_1, the 2 of {1,2}), the step label at the start
  ("Step 3:", "3." or "3)") and a step reference ("step 3"). Steps after k are unchanged (they keep the old number).
  Answers without an editable number in steps 2..n are skipped.
V4 on-topic donor (option B). Step k (uniform over the non-empty steps 2..n) is replaced by a step of ANOTHER answer to the
  SAME question group (same benchmark, so the same side). The donor text (whitespace-normalized) must differ from every
  step of the parent, and the donor position must differ from k (on PRMBench a sibling's step at position k is the
  other version of the same step: often the real labeled error itself, which would make the realism check circular).
  Levels: L0 relative position +-.2 and character length +-40%, L1 position +-.4, L2 any position (length always +-40%).
  Inside a level prefer an unused donor step, then one used < 3 times, then any. No donor: skip the answer.
  Caveat (DIAG_REALISM.md 4.B): the donor step can be a correct alternative, so "known wrong" is not certain for V4.

Outputs (--out, normally $W/inputs; nothing existing is overwritten):
  pb_V3_answers.json, pb_V4_answers.json, prmbench_V3_meta.pkl, prmbench_V4_meta.pkl     driver inputs (dev parents)
  pb_preC_V3_answers.json, ..., prmbench_preC_V4_meta.pkl     C rows of the parents of the first 12 variant rows, in the
                                                              same order (preflight C run, so the prefix check has pairs)
  SIDECAR_<bench>_<V3|V4>.jsonl                               one line per DEV parent (built or skip)
  VARIANTS_V34_REPORT.json                                    gates, counts, skips, levels, reuse

    python make_variants_v34.py --pb-answers A.json --pb-gold G.json --prmb-meta prmbench_telemetry.pkl|rows.json \
        --folds FOLDS_V2.json --split SPLIT.json --out DIR [--test-allow-missing]
"""
import argparse, json, pickle, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import make_variants as MV

SEED = MV.SEED
N_PRE = 12
NUM = re.compile(r'(?<![A-Za-z_\d.,\\])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?!\d)')
LABEL = re.compile(r'^\s*[#*>\-\s]*(?:step\s*\d+|\d+\s*[.)](?=\s))', re.I)


def norm(t):
    return ' '.join(t.split())


def editable_numbers(t):
    """[(start, end, text)] of the editable numbers of a step text (see the module doc)."""
    m0 = LABEL.match(t); lab_end = m0.end() if m0 else -1
    out = []
    for m in NUM.finditer(t):
        s = m.start()
        if s < lab_end: continue
        if t[max(0, s - 8):s].lower().rstrip().endswith('step'): continue
        out.append((s, m.end(), m.group(0)))
    return out


def edit_number(txt, rng):
    if '.' in txt:
        head, d = txt[:-1], int(txt[-1])
        nd = d - 1 if d == 9 else d + 1 if d == 0 else d + (1 if rng.random() < .5 else -1)
        return head + str(nd)
    v = int(txt.replace(',', ''))
    nv = 1 if v == 0 else v + (1 if rng.random() < .5 else -1)
    s = f'{nv:,}' if ',' in txt else str(nv)
    if txt.startswith('0') and len(txt) > 1 and ',' not in txt: s = s.zfill(len(txt))
    return s


def make_v3(answers, bench, dev):
    rng = np.random.default_rng([SEED, 3, 0 if bench == 'processbench' else 1]); out = {}
    for i in rng.permutation(len(answers)):
        if i not in dev: continue
        steps = answers[i]['steps']; n = len(steps)
        cand = {k: editable_numbers(steps[k - 1]) for k in range(2, n + 1) if steps[k - 1].strip()}
        ks = [k for k, v in cand.items() if v]
        if n < 2 or not ks:
            out[i] = dict(skip='no editable number in steps 2..n'); continue
        k = int(rng.choice(ks)); s, e, old = cand[k][-1]; new = edit_number(old, rng)
        t = steps[k - 1]; nt = t[:s] + new + t[e:]
        assert nt != t and old != new
        ns = list(steps); ns[k - 1] = nt
        out[i] = dict(steps=ns, k=k, old=old, new=new, char_start=s, rel_char=round(s / max(1, len(t)), 3),
                      n_editable_in_step=len(cand[k]), n_eligible_steps=len(ks), rel_parent=round((k - 1) / (n - 1), 3))
    return out


def make_v4(answers, bench, dev):
    rng = np.random.default_rng([SEED, 4, 0 if bench == 'processbench' else 1]); out = {}
    by_group = defaultdict(list)
    for i in sorted(dev): by_group[answers[i]['group']].append(i)
    used = Counter()
    levels = [(.2, .4), (.4, .4), (None, .4)]
    for i in rng.permutation(len(answers)):
        if i not in dev: continue
        r = answers[i]; steps = r['steps']; n = len(steps)
        ks = [k for k in range(2, n + 1) if steps[k - 1].strip()]
        sib = [a for a in by_group[r['group']] if a != i]
        if n < 2 or not ks:
            out[i] = dict(skip='n<2 or no non-empty step 2..n'); continue
        if not sib:
            out[i] = dict(skip='no other answer to the same question'); continue
        k = int(rng.choice(ks)); rel = (k - 1) / (n - 1); L = float(len(steps[k - 1])); pn = {norm(t) for t in steps}
        cand = []
        for a in sib:
            st = answers[a]['steps']; m = len(st)
            for j, t in enumerate(st):
                if t.strip() and j + 1 != k and norm(t) not in pn:
                    cand.append((a, j, j / (m - 1) if m > 1 else 0., len(t)))
        pick = None
        for lev, (dp, dl) in enumerate(levels):
            c = [x for x in cand if abs(x[3] - L) <= dl * L and (dp is None or abs(x[2] - rel) <= dp + 1e-12)]
            if not c: continue
            u = np.array([used[(x[0], x[1])] for x in c])
            for tier in (u == 0, u < 3, np.ones(len(c), bool)):
                if tier.any():
                    idx = np.flatnonzero(tier); pick = c[int(rng.choice(idx))]; break
            if pick: break
        if pick is None:
            out[i] = dict(skip='no donor in the same question group'); continue
        a, j, rd, ld = pick; used[(a, j)] += 1
        ns = list(steps); ns[k - 1] = answers[a]['steps'][j]; d = answers[a]
        out[i] = dict(steps=ns, k=k, donor_uid=d['uid'], donor_step=j + 1, relax_level=lev, n_candidates=len(cand),
                      n_siblings=len(sib), donor_same_generator=bool(r['generator'] and d['generator'] == r['generator']),
                      len_ratio=round(ld / L, 3), rel_parent=round(rel, 3), rel_donor=round(rd, 3))
    return out, used


def write_new(path, writer):
    if path.exists(): sys.exit(f'REFUSING to overwrite {path}')
    tmp = path.with_name(path.name + '.tmp'); writer(tmp); tmp.rename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pb-answers', required=True); ap.add_argument('--pb-gold', required=True)
    ap.add_argument('--prmb-meta', required=True); ap.add_argument('--folds', required=True)
    ap.add_argument('--split', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--test-allow-missing', action='store_true', help='LOCAL CODE TEST ONLY (as make_variants.py)')
    a = ap.parse_args(); t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = [f'pb_{V}_answers.json' for V in ('V3', 'V4')] + [f'prmbench_{V}_meta.pkl' for V in ('V3', 'V4')] + \
            [f'pb_preC_{V}_answers.json' for V in ('V3', 'V4')] + [f'prmbench_preC_{V}_meta.pkl' for V in ('V3', 'V4')] + \
            [f'SIDECAR_{b}_{V}.jsonl' for b in ('processbench', 'prmbench') for V in ('V3', 'V4')] + ['VARIANTS_V34_REPORT.json']
    exist = [n for n in names if (out / n).exists()]
    if exist: sys.exit(f'REFUSING: outputs exist in {out}: {exist}')
    strata = json.load(open(a.folds))['group_strata']; SP = json.load(open(a.split))
    rep = dict(seed=SEED, decision='Omri 2026-11-18 card "Numeric edit + on-topic donor"; DIAG_REALISM.md 4 A/B', side='dev only')

    prmb = MV.load_prmb_rows(a.prmb_meta); pg, qmap, _ = MV.prmb_groups(prmb)
    pb = json.load(open(a.pb_answers)); gold = json.load(open(a.pb_gold))
    gold = {g['uid']: g for g in (gold if isinstance(gold, list) else gold.values())}
    pbg = [qmap.get(MV.sha(MV.norm_q(r['question'])), 'pb_source_v2:' + MV.sha(MV.norm_q(r['question']))[:24]) for r in pb]
    miss = sorted({g for g in pg + pbg if g not in strata})
    if miss and a.test_allow_missing:
        strata = dict(strata, **{g: 'prmbench' if g.startswith('prmb') else 'pb_test' for g in miss}); miss = []
    if miss: sys.exit(f'{len(miss)} group ids not in FOLDS_V2.json: stop')
    side = MV.split_groups(strata)
    pb_ans = [dict(uid=r['uid'], steps=list(r['steps']), group=g, side=side[g], subset=gold[r['uid']]['subset'],
                   generator=gold[r['uid']].get('generator')) for r, g in zip(pb, pbg)]
    pr_ans = [dict(uid=str(r['idx']), steps=list(r['steps']), group=g, side=side[g], subset=None, generator=None)
              for r, g in zip(prmb, pg)]
    # gate: the rebuilt split equals the stored SPLIT.json (groups and every answer)
    same = side == SP['group_side'] and all(SP[b] == {x['uid']: [x['group'], x['side']] for x in L}
                                            for b, L in (('processbench', pb_ans), ('prmbench', pr_ans)))
    rep['gate_split_equals_stored'] = bool(same); print('SPLIT GATE', 'PASS' if same else 'FAIL', flush=True)
    if not same: sys.exit('rebuilt split differs from SPLIT.json: stop')

    for bench, L, src in (('processbench', pb_ans, pb), ('prmbench', pr_ans, prmb)):
        dev = {i for i, x in enumerate(L) if x['side'] == 'dev'}
        v3 = make_v3(L, bench, dev); v4, used = make_v4(L, bench, dev)
        for V, Dd in (('V3', v3), ('V4', v4)):
            rows, lines, meta, pre = [], [], {}, []
            for i in sorted(dev):
                d = Dd[i]; p = L[i]
                line = dict(uid=f"{p['uid']}__{V.lower()}", parent_uid=p['uid'], variant=V, group=p['group'], side=p['side'])
                if 'skip' in d:
                    line['skip'] = d['skip']; lines.append(line); continue
                line.update({k: v for k, v in d.items() if k != 'steps'}); lines.append(line)
                if len(pre) < N_PRE: pre.append(i)
                if bench == 'processbench':
                    rows.append(dict(uid=line['uid'], question=src[i]['question'], steps=d['steps'],
                                     benchmark=src[i].get('benchmark'), source_id=p['uid']))
                else:
                    r = dict(src[i]); r['idx'] = line['uid']; r['steps'] = d['steps']      # label fields: placeholders
                    meta[len(meta)] = r
            if bench == 'processbench':
                write_new(out / f'pb_{V}_answers.json', lambda f: json.dump(rows, open(f, 'w')))
                write_new(out / f'pb_preC_{V}_answers.json', lambda f: json.dump([src[i] for i in pre], open(f, 'w')))
            else:
                write_new(out / f'prmbench_{V}_meta.pkl', lambda f: pickle.dump(meta, open(f, 'wb')))
                write_new(out / f'prmbench_preC_{V}_meta.pkl', lambda f: pickle.dump({n: dict(src[i]) for n, i in enumerate(pre)}, open(f, 'wb')))
            write_new(out / f'SIDECAR_{bench}_{V}.jsonl', lambda f: open(f, 'w').write(''.join(json.dumps(s) + '\n' for s in lines)))
            built = [s for s in lines if 'skip' not in s]
            info = dict(dev_parents=len(dev), rows=len(built), share_built=round(len(built) / max(1, len(dev)), 4),
                        skipped=dict(Counter(s['skip'] for s in lines if 'skip' in s)), preflight_parents=[L[i]['uid'] for i in pre],
                        k_rel_mean=float(np.mean([s['rel_parent'] for s in built])) if built else None,
                        n_steps_mean=float(np.mean([len(L[i]['steps']) for i in dev if 'skip' not in Dd[i]])) if built else None)
            if V == 'V3':
                info['old_new_examples'] = [(s['old'], s['new']) for s in built[:8]]
                info['decimal_share'] = float(np.mean(['.' in s['old'] for s in built])) if built else None
            else:
                info['relax_level'] = dict(Counter(s['relax_level'] for s in built))
                info['same_generator_share'] = float(np.mean([s['donor_same_generator'] for s in built])) if built else None
                info['donor_reuse'] = dict(Counter(int(u) for u in used.values()))
                info['len_ratio_median'] = float(np.median([s['len_ratio'] for s in built])) if built else None
            rep[f'{bench}_{V}'] = info
            print(bench, V, json.dumps({k: v for k, v in info.items() if k != 'preflight_parents'}), flush=True)
    rep['seconds'] = round(time.time() - t0, 1)
    write_new(out / 'VARIANTS_V34_REPORT.json', lambda f: json.dump(rep, open(f, 'w'), indent=1))
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
