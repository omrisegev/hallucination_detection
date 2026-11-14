"""Test 2 (ProcessBench): first-error localization on WRONG answers only, the Step-DPO use case.
Step-DPO (arXiv 2406.18629, Sec. 3.2) keeps answers whose final answer is wrong, then locates the first wrong
step "manually or using GPT-4". Here the answer is known to be wrong, so every method must name one step.
Metric = exact first-error accuracy on erroneous answers (= ProcessBench 'error' accuracy = Mind-the-Gap SLA),
macro over the 4 subsets. Inputs: results/pb_first_error_rule_v1/cells/PB_<prm>.npz (standalone base-model cell,
sa_*). Labels used only for scoring.  python -I test2_pb.py"""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
CELLS = ROOT / 'results/pb_first_error_rule_v1/cells'
GOLD = {g['uid']: g for g in json.loads((ROOT / 'results/prm_boost_processbench_v1/local/processbench_gold.json').read_text())}
SUB = ('gsm8k', 'math', 'olympiadbench', 'omnimath')
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
B = 2000


def segs(off):
    return list(zip(off[:-1], off[1:]))


def ans_z(s, off):
    out = np.empty_like(s, dtype=float)
    for a, b in segs(off):
        x = s[a:b]; sd = x.std(); out[a:b] = (x - x.mean()) / max(sd, 1e-8)
    return out


def gap_first(s, off):
    """GAP rule (spectral_utils.prm_anchored_lsml.gap_flags): k = largest gap in the top ceil(m/2) of the sorted
    answer-z scores; flags = top-k; prediction = earliest flagged step. Always >= 1 flag, so never 'no error'."""
    z = ans_z(s, off); pred = []
    for a, b in segs(off):
        x = z[a:b]; m = b - a
        order = np.argsort(-x, kind='stable')
        if m == 1:
            pred.append(0); continue
        srt = x[order]; km = min(int(np.ceil(m / 2)), m - 1)
        k = int(np.argmax(srt[:-1][:km] - srt[1:][:km])) + 1
        pred.append(int(order[:k].min()))
    return np.array(pred)


def argmax_pred(s, off):
    return np.array([int(np.argmax(s[a:b])) for a, b in segs(off)])


def prm_native(p, off):
    """PRM own rule (as pb_first_error_rule_v1/run.py): arm__PRMPROB is the PRM error probability; first step with
    p > .5; if none, argmax p (forced, the error is known to exist)."""
    out = []
    for a, b in segs(off):
        f = np.flatnonzero(p[a:b] > .5)
        out.append(int(f[0]) if len(f) else int(np.argmax(p[a:b])))
    return np.array(out)


def score(ids, pred, rng=None):
    lab = np.array([GOLD[u]['label'] for u in ids]); sub = np.array([GOLD[u]['subset'] for u in ids])
    hit = (pred == lab).astype(float)
    def macro(ix):
        return float(np.mean([100 * hit[ix][(sub[ix] == s) & (lab[ix] >= 0)].mean() for s in SUB]))
    err = np.flatnonzero(lab >= 0)
    per = {s: 100 * hit[(sub == s) & (lab >= 0)].mean() for s in SUB}
    out = dict(macro=macro(np.arange(len(ids))), **per)
    if rng is not None:   # stratified bootstrap over erroneous answers within subset
        by = {s: err[sub[err] == s] for s in SUB}
        bs = [np.mean([100 * hit[rng.choice(by[s], len(by[s]))].mean() for s in SUB]) for _ in range(B)]
        out['lo'], out['hi'] = np.percentile(bs, [2.5, 97.5])
    return out, hit


rows, HITS = [], {}
rng = np.random.default_rng(0)
for prm in PRMS:
    d = np.load(CELLS / f'PB_{prm}.npz'); ids = d['sa_ids']; off = d['sa_off']
    lab = np.array([GOLD[u]['label'] for u in ids]); n_err = int((lab >= 0).sum())
    tele = d['sa_arm__TELE_N16']; mean = d['sa_arm__MEAN_N16']
    arms = {
        'TELE16 standalone + GAP (earliest flag)': gap_first(tele, off),
        'TELE16 standalone, argmax step': argmax_pred(tele, off),
        'PRM native (first p_err>.5, else argmax)': prm_native(d['arm__PRMPROB'], d['off']),
        'Position only + GAP (earliest flag)': gap_first(d['sa_arm__POS_ONLY'], off),
        'Always step 0': np.zeros(len(ids), int),
    }
    assert (d['ids'] == ids).all() and (d['off'] == off).all()
    for name, pred in arms.items():
        m, hit = score(ids, pred, rng); HITS[(prm, name)] = hit
        rows.append(dict(bench='processbench', prm=prm, arm=name, n_err=n_err, first_error_acc=m['macro'],
                         lo=m['lo'], hi=m['hi'], **{f'acc_{s}': m[s] for s in SUB}))
        print(f'{prm:12s} {name:46s} {m["macro"]:6.2f} [{m["lo"]:.2f},{m["hi"]:.2f}]', flush=True)
    # paired: TELE16+GAP minus PRM native, and minus position-only, same erroneous answers
    sub = np.array([GOLD[u]['subset'] for u in ids]); err = np.flatnonzero(lab >= 0); by = {s: err[sub[err] == s] for s in SUB}
    for ref in ('PRM native (first p_err>.5, else argmax)', 'Position only + GAP (earliest flag)'):
        dlt = HITS[(prm, 'TELE16 standalone + GAP (earliest flag)')] - HITS[(prm, ref)]
        pt = np.mean([100 * dlt[by[s]].mean() for s in SUB])
        bs = []
        for _ in range(B):
            bs.append(np.mean([100 * dlt[rng.choice(by[s], len(by[s]))].mean() for s in SUB]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(bench='processbench', prm=prm, arm=f'DIFF TELE16+GAP minus {ref}', n_err=n_err,
                         first_error_acc=pt, lo=lo, hi=hi))
        print(f'{prm:12s}   diff vs {ref[:20]:20s} {pt:+6.2f} [{lo:+.2f},{hi:+.2f}]')
with open(HERE / 'TEST2_PB.csv', 'w', newline='') as f:
    keys = list(dict.fromkeys(k for r in rows for k in r)); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
