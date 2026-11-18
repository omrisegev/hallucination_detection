"""Inputs of the one SSL experiment (2026-11-18): question-group split and the synthetic known-error variants V1 / V2.

Decisions: research_plan_2026-11-16/mac_handoff/ANSWERS_TO_MAC_2026-11-18.md, answers 1-10 (cloud project files).
No gold label is read to build anything. The PB gold file is read only for the subset and generator fields.

Groups (answers 6-7): the corrected source-question groups of results/localization_source_group_audit_v1
(recipe scripts/audit_localization_source_groups.py):
  PRMBench  source seed = regex (prm_(?:train|test)_p\\d+_\\d+)$ on source_idx; seeds that share an exact
            whitespace-normalized question are joined; id = 'prmb_source_v2:' + sha256('|'.join(sorted seeds))[:24]
  PB        id = the PRMBench group of the same normalized question, else 'pb_source_v2:' + sha256(question)[:24]
  Gate: every id must be a key of FOLDS_V2.json; expected 707 PRMBench + 2,776 PB groups.
Split (answers 8-9): one 50/50 split over all groups of both benchmarks, stratified by FOLDS_V2 group_strata,
  seed 20261116; inside a stratum the first ceil(n/2) groups of a random order go to dev. V1/V2 are built after the
  split; donors come only from the same side.
V2 (answers 1-4, 10): step k (uniform over the non-empty steps 2..n) is replaced by one step of another answer.
  Level 0: same side, different group, same PB subset, relative position within +-.2, character length within +-20%,
  donor step used < 3 times, donor text != parent step text. Inside the candidates prefer: same generator (PB) and
  unused, then same generator, then unused, then any. Relax: L1 length +-40%, L2 position +-.4, L3 any PB subset.
  None: skip and log. Only step k is known wrong.
V1 (answer 5): step 1 kept, steps 2..n permuted; the original last step must not stay last; the step texts must
  change. n = 2: excluded. Flag single_adjacent_swap.

Outputs (--out):
  SPLIT.json                       group -> side, answer -> (group, side), counts
  pb_V1_answers.json, pb_V2_answers.json             driver --answers input (same fields as the PB C answers.json)
  prmbench_V1_meta.pkl, prmbench_V2_meta.pkl         driver --meta-pkl input ({i: row}, parent order); the label
                                                     fields are the parent's PLACEHOLDERS (never read by any fit)
  SIDECAR_<bench>_<V>.jsonl        one line per variant row
  VARIANTS_REPORT.json             counts, relax levels, skips, gates

    python make_variants.py --pb-answers A.json --pb-gold G.json --prmb-meta prmbench_telemetry.pkl|rows.json \
        --folds FOLDS_V2.json --out DIR [--n-parents N]   (N: first N parents only, smoke test)
"""
import argparse, hashlib, json, math, os, pickle, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 20261116
ROW_KEYS = ("idx", "source_idx", "classification", "category", "question", "steps", "error_steps")


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


def norm_q(q):
    return ' '.join(q.split())


def source_seed(s):
    m = re.search(r'(prm_(?:train|test)_p\d+_\d+)$', str(s))
    if not m:
        raise ValueError('UNRECOGNIZED_SOURCE_ID: ' + str(s))
    return m.group(1)


def load_prmb_rows(path):
    """Rows in the driver's --meta-pkl order (sorted keys). A .pkl is a telemetry cache; a .json is a list of rows."""
    if str(path).endswith('.json'):
        rows = json.load(open(path))
    else:
        src = pickle.load(open(path, 'rb'))
        rows = [{k: e[k] for k in ROW_KEYS} for _, e in sorted(src.items())]
        del src
    return rows


def prmb_groups(rows):
    seeds = [source_seed(r['source_idx']) for r in rows]
    parent = {s: s for s in set(seeds)}

    def root(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]; s = parent[s]
        return s
    by_q = defaultdict(set)
    for s, r in zip(seeds, rows):
        by_q[sha(norm_q(r['question']))].add(s)
    for q, ss in by_q.items():
        ss = sorted(ss)
        for s in ss[1:]:
            a, b = root(ss[0]), root(s)
            if a != b: parent[b] = a
    members = defaultdict(list)
    for s in parent: members[root(s)].append(s)
    canon = {s: 'prmb_source_v2:' + sha('|'.join(sorted(members[root(s)])))[:24] for s in parent}
    gid = [canon[s] for s in seeds]
    qmap = {}
    for g, r in zip(gid, rows):
        qmap[sha(norm_q(r['question']))] = g
    return gid, qmap, len(set(seeds))


def split_groups(strata):
    """strata: {group: stratum}. -> {group: 'dev'|'test'}"""
    rng = np.random.default_rng(SEED); side = {}
    by = defaultdict(list)
    for g, s in strata.items(): by[s].append(g)
    for s in sorted(by):
        gs = sorted(by[s]); perm = rng.permutation(len(gs)); ndev = math.ceil(len(gs) / 2)
        for i, j in enumerate(perm):
            side[gs[j]] = 'dev' if i < ndev else 'test'
    return side


class DonorPool:
    def __init__(self, answers, bench):
        """answers: list of dict(uid, steps, group, side, subset, generator)."""
        A, J, M, REL, LEN, TXT = [], [], [], [], [], []
        for a, r in enumerate(answers):
            m = len(r['steps'])
            for j, t in enumerate(r['steps']):
                if not t.strip(): continue
                A.append(a); J.append(j); M.append(m); REL.append(j / (m - 1) if m > 1 else 0.); LEN.append(len(t)); TXT.append(t)
        self.a = np.array(A); self.j = np.array(J); self.rel = np.array(REL); self.len = np.array(LEN, float)
        self.txt = TXT; self.used = np.zeros(len(A), int)
        self.side = np.array([answers[a]['side'] for a in A]); self.group = np.array([answers[a]['group'] for a in A])
        self.subset = np.array([answers[a]['subset'] or '' for a in A]); self.gen = np.array([answers[a]['generator'] or '' for a in A])
        self.bench = bench

    def pick(self, rng, side, group, subset, gen, rel, L, text):
        base = (self.side == side) & (self.group != group) & (self.used < 3)
        levels = [(.2, .2, True), (.4, .2, True), (.4, .4, True), (.4, .4, False)]
        for lev, (dl, dp, same_sub) in enumerate(levels):
            if self.bench == 'prmbench' and lev == 3: break                 # no subset rule on PRMBench
            m = base & (np.abs(self.len - L) <= dl * L) & (np.abs(self.rel - rel) <= dp + 1e-12)
            if same_sub and self.bench == 'processbench': m &= self.subset == subset
            cand = np.flatnonzero(m)
            cand = np.array([c for c in cand if self.txt[c] != text], int)
            if not len(cand): continue
            sg = self.gen[cand] == gen if gen else np.zeros(len(cand), bool); un = self.used[cand] == 0
            for tier in (sg & un, sg, un, np.ones(len(cand), bool)):
                if tier.any():
                    c = int(rng.choice(cand[tier])); self.used[c] += 1
                    return c, lev
        return None, None


def make_v2(answers, bench, n_par):
    rng = np.random.default_rng([SEED, 2, 0 if bench == 'processbench' else 1])
    pool = DonorPool(answers, bench); out = {}; side = {}
    order = rng.permutation(len(answers))
    for i in order:
        if i >= n_par: continue
        r = answers[i]; steps = r['steps']; n = len(steps)
        ks = [k for k in range(2, n + 1) if steps[k - 1].strip()]
        if n < 2 or not ks:
            out[i] = dict(skip='n<2 or no non-empty step 2..n'); continue
        k = int(rng.choice(ks)); rel = (k - 1) / (n - 1); L = float(len(steps[k - 1]))
        c, lev = pool.pick(rng, r['side'], r['group'], r['subset'], r['generator'], rel, L, steps[k - 1])
        if c is None:
            out[i] = dict(skip='no donor'); continue
        new = list(steps); new[k - 1] = pool.txt[c]
        d = answers[pool.a[c]]
        out[i] = dict(steps=new, k=k, donor_uid=d['uid'], donor_step=int(pool.j[c]) + 1, relax_level=lev,
                      donor_same_generator=bool(r['generator'] and d['generator'] == r['generator']),
                      len_ratio=round(len(pool.txt[c]) / L, 3), rel_parent=round(rel, 3), rel_donor=round(float(pool.rel[c]), 3))
    return out, pool


def make_v1(answers, bench, n_par):
    rng = np.random.default_rng([SEED, 1, 0 if bench == 'processbench' else 1]); out = {}
    for i, r in enumerate(answers):
        if i >= n_par: continue
        steps = r['steps']; n = len(steps); m = n - 1
        if n < 3:
            out[i] = dict(skip='n<3'); continue
        ok = False
        for _ in range(200):
            perm = [1, 0] if m == 2 else list(rng.permutation(m))
            if perm[-1] == m - 1: continue
            new = [steps[0]] + [steps[1 + p] for p in perm]
            if new != list(steps): ok = True; break
        if not ok:
            out[i] = dict(skip='no valid permutation'); continue
        moved = [t for t in range(m) if perm[t] != t]
        swap = len(moved) == 2 and moved[1] == moved[0] + 1
        out[i] = dict(steps=new, perm=[0] + [1 + int(p) for p in perm], single_adjacent_swap=bool(swap))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pb-answers', required=True); ap.add_argument('--pb-gold', required=True)
    ap.add_argument('--prmb-meta', required=True); ap.add_argument('--folds', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--n-parents', type=int, default=None)
    ap.add_argument('--test-allow-missing', action='store_true', help='LOCAL CODE TEST ONLY: unknown groups -> own stratum')
    a = ap.parse_args(); t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    folds = json.load(open(a.folds)); strata = folds['group_strata']
    rep = {'seed': SEED, 'decisions': 'ANSWERS_TO_MAC_2026-11-18.md answers 1-10'}

    prmb = load_prmb_rows(a.prmb_meta)
    pg, qmap, n_seeds = prmb_groups(prmb)
    pb = json.load(open(a.pb_answers)); gold = json.load(open(a.pb_gold))
    gold = {g['uid']: g for g in (gold if isinstance(gold, list) else gold.values())}
    pbg = [qmap.get(sha(norm_q(r['question'])), 'pb_source_v2:' + sha(norm_q(r['question']))[:24]) for r in pb]
    miss = sorted({g for g in pg + pbg if g not in strata})
    rep['gate_groups'] = dict(prmb_rows=len(prmb), prmb_seeds=n_seeds, prmb_groups=len(set(pg)),
                              pb_rows=len(pb), pb_groups=len(set(pbg)),
                              pb_groups_pb_prefix=len({g for g in pbg if g.startswith('pb_')}),
                              ids_not_in_folds=len(miss), example_missing=miss[:5],
                              status='PASS' if not miss and len(set(pg)) == 707 else 'FAIL')
    print('GROUP GATE', json.dumps(rep['gate_groups']), flush=True)
    if miss and a.test_allow_missing:
        strata = dict(strata, **{g: 'prmbench' if g.startswith('prmb') else 'pb_test' for g in miss}); miss = []
    if miss:
        json.dump(rep, open(out / 'VARIANTS_REPORT.json', 'w'), indent=1); sys.exit('group ids not in FOLDS_V2.json: stop')
    side = split_groups(strata)
    pb_ans = [dict(uid=r['uid'], steps=list(r['steps']), group=g, side=side[g], subset=gold[r['uid']]['subset'],
                   generator=gold[r['uid']].get('generator')) for r, g in zip(pb, pbg)]
    pr_ans = [dict(uid=str(r['idx']), steps=list(r['steps']), group=g, side=side[g], subset=None, generator=None)
              for r, g in zip(prmb, pg)]
    sp = {'group_side': side, 'processbench': {x['uid']: [x['group'], x['side']] for x in pb_ans},
          'prmbench': {x['uid']: [x['group'], x['side']] for x in pr_ans}}
    cnt = {}
    for b, L in (('processbench', pb_ans), ('prmbench', pr_ans)):
        c = Counter(x['side'] for x in L); g = Counter(side[x] for x in {x['group'] for x in L})
        cnt[b] = dict(answers_dev=c['dev'], answers_test=c['test'], groups_dev=g['dev'], groups_test=g['test'],
                      steps_dev=sum(len(x['steps']) for x in L if x['side'] == 'dev'),
                      steps_test=sum(len(x['steps']) for x in L if x['side'] == 'test'))
    sp['counts'] = cnt; rep['split'] = cnt
    json.dump(sp, open(out / 'SPLIT.json', 'w'))
    # labels of the real PRMBench rows, for evaluation only (never read by a fit); PB gold is in the repository
    json.dump([dict(idx=str(r['idx']), classification=r['classification'], category=r.get('category'),
                    error_steps=r['error_steps'], n_steps=len(r['steps']), source_idx=r['source_idx'])
               for r in prmb], open(out / 'prmbench_C_labels.json', 'w'))
    print('SPLIT', json.dumps(cnt), flush=True)

    for bench, L, src in (('processbench', pb_ans, pb), ('prmbench', pr_ans, prmb)):
        n_par = len(L) if a.n_parents is None else a.n_parents
        v2, pool = make_v2(L, bench, n_par); v1 = make_v1(L, bench, n_par)
        for V, D in (('V1', v1), ('V2', v2)):
            rows, side_lines, meta = [], [], {}
            for i in range(min(n_par, len(L))):
                d = D.get(i, {'skip': 'not built'}); p = L[i]
                line = dict(uid=f"{p['uid']}__{V.lower()}", parent_uid=p['uid'], variant=V, group=p['group'], side=p['side'])
                if 'skip' in d:
                    line['skip'] = d['skip']; side_lines.append(line); continue
                line.update({k: v for k, v in d.items() if k != 'steps'}); side_lines.append(line)
                if bench == 'processbench':
                    rows.append(dict(uid=line['uid'], question=src[i]['question'], steps=d['steps'],
                                     benchmark=src[i].get('benchmark'), source_id=p['uid']))
                else:
                    r = dict(src[i]); r['idx'] = line['uid']; r['steps'] = d['steps']      # label fields: placeholders
                    meta[len(meta)] = r
            if bench == 'processbench':
                json.dump(rows, open(out / f'pb_{V}_answers.json', 'w'))
            else:
                pickle.dump(meta, open(out / f'prmbench_{V}_meta.pkl', 'wb'))
            with open(out / f'SIDECAR_{bench}_{V}.jsonl', 'w') as f:
                for s in side_lines: f.write(json.dumps(s) + '\n')
            built = [s for s in side_lines if 'skip' not in s]
            info = dict(rows=len(built), skipped=dict(Counter(s['skip'] for s in side_lines if 'skip' in s)))
            if V == 'V2':
                info['relax_level'] = dict(Counter(s['relax_level'] for s in built))
                info['same_generator_share'] = float(np.mean([s['donor_same_generator'] for s in built])) if built else None
                info['donor_reuse'] = dict(Counter(int(u) for u in pool.used if u > 0))
                info['k_rel_mean'] = float(np.mean([s['rel_parent'] for s in built])) if built else None
            else:
                info['single_adjacent_swap'] = int(sum(s['single_adjacent_swap'] for s in built))
            rep[f'{bench}_{V}'] = info
            print(bench, V, json.dumps(info), flush=True)
    rep['seconds'] = round(time.time() - t0, 1)
    json.dump(rep, open(out / 'VARIANTS_REPORT.json', 'w'), indent=1)
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
