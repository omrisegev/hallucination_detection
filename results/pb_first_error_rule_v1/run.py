"""ProcessBench first-error rule study: every arm x 4 PRMs -> RESULTS.csv (+ printed table).
Arms: references (PRM native p>.5, C4U own flags, C4U NC) and the change-point chain posterior (cp_rule.py)
on different evidence channels. 'anchored' = emissions estimated in one EM step from the PRM native partition
(pseudo-labels, no gold); 'EM' = iterated to convergence; 'free' = no PRM anywhere (init and channels)."""
import csv, sys, time
import numpy as np
sys.path.insert(0, '.')
from lab import load, first_pass, em, R, V

PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
rows = []
PREDS, LOCS = {}, {}


def add(prm, arm, pred, note='', P=None):
    m = V.metrics(ids, pred)
    r = dict(prm=prm, arm=arm, note=note, avg_F1=m['avg F1'], macro_SLA=m['macro SLA'], overall_SLA=m['overall SLA'])
    # SLA_loc: Mind the Gap protocol (error assumed present): argmax over steps of the posterior, no 'no error' option
    loc = np.array([int(np.argmax(q[:-1])) for q in P]) if P is not None else np.where(pred < 0, 10 ** 6, pred)
    r['macro_SLA_loc'] = V.metrics(ids, loc)['macro SLA']
    PREDS[(prm, arm)] = np.asarray(pred); LOCS[(prm, arm)] = loc
    for s in V.SUBSETS:
        for k in ('TPR', 'TNR', 'F1', 'early', 'late', 'miss'):
            r[f'{s}_{k}'] = m[s][k]
    rows.append(r); print(f'{prm:12s} {arm:34s} F1 {r["avg_F1"]:6.2f}  SLA {r["macro_SLA"]:6.2f}  SLA_loc {r["macro_SLA_loc"]:6.2f}', flush=True)


def h_from(w1, off):
    tau = [np.flatnonzero(w1[a:b])[0] for a, b in zip(off[:-1], off[1:]) if w1[a:b].any()]
    lens = [b - a for a, b in zip(off[:-1], off[1:]) if w1[a:b].any()]
    best, bh = -np.inf, .15
    for h in R.H_GRID:
        v = sum(np.log(h) + t * np.log1p(-h) - np.log1p(-(1 - h) ** n) for t, n in zip(tau, lens))
        if v > best: best, bh = v, h
    return float(bh)


def free_init(x, off, frac=.5):
    """PRM-free start: first step whose mean channel value passes the global quantile that flags `frac` of answers."""
    m = x.mean(1) if x.ndim == 2 else x
    mx = np.array([m[a:b].max() for a, b in zip(off[:-1], off[1:])]); q = np.quantile(mx, 1 - frac)
    return first_pass(m >= q, off)


for prm in PRMS:
    d = load(prm); ids = d['ids']; off = d['off']; p = d['arm__PRMPROB']; risk = d['arm__PRM']
    w1, w2, nat = first_pass(p > .5, off)
    pi0 = float(np.mean(nat < 0)); h0 = h_from(w1, off)
    add(prm, 'REF PRM native (p>.5)', nat)
    LOCS[(prm, 'REF PRM native (p>.5)')] = np.array([n if n >= 0 else int(np.argmax(risk[a:b])) for n, a, b in zip(nat, off[:-1], off[1:])])
    rows[-1]['macro_SLA_loc'] = V.metrics(ids, LOCS[(prm, 'REF PRM native (p>.5)')])['macro SLA']   # forced: argmax risk if no flag
    add(prm, 'REF C4U own flags', first_pass(d['c4u_flags'], off)[2])
    nc = []
    for a, b in zip(off[:-1], off[1:]):
        k = int((p[a:b] > .5).sum()); nc.append(-1 if k == 0 else int(np.argsort(-d['c4u_rank'][a:b], kind='stable')[:k].min()))
    add(prm, 'REF C4U NC (PRM gate+count)', np.array(nc))
    LOCS[(prm, 'REF C4U NC (PRM gate+count)')] = np.array([n if n >= 0 else int(np.argmax(d['c4u_rank'][a:b])) for n, a, b in zip(nc, off[:-1], off[1:])])
    rows[-1]['macro_SLA_loc'] = V.metrics(ids, LOCS[(prm, 'REF C4U NC (PRM gate+count)')])['macro SLA']
    kw = dict(pi=pi0, h=h0)
    P, _ = em(risk, off, w1, w2, B=20, **kw); add(prm, 'CP PRM (anchored)', R.map_pred(P), f'pi={pi0:.3f} h={h0:.3f}', P)
    F = R.fit(risk, off, model='first', B=20, init=None, iters=40)
    add(prm, 'CP PRM (EM to convergence)', R.map_pred(F['P']), f'pi={F["pi"]:.3f} h={F["h"]:.3f}', F['P'])
    dr = np.concatenate([np.r_[0., np.diff(risk[a:b])] for a, b in zip(off[:-1], off[1:])])
    P, _ = em(np.column_stack([risk, dr]), off, w1, w2, B=10, **kw); add(prm, 'CP PRM + drop (anchored)', R.map_pred(P), '', P)
    P, _ = em(d['c4u_rank'], off, w1, w2, B=20, **kw); add(prm, 'CP C4U rank (PRM anchor)', R.map_pred(P), '', P)
    P, _ = em(np.column_stack([risk, d['c4u_rank']]), off, w1, w2, B=10, **kw); add(prm, 'CP PRM + C4U rank (anchored)', R.map_pred(P), '', P)
    P, _ = em(np.column_stack([risk, d['G16']]), off, w1, w2, B=10, **kw); add(prm, 'CP PRM + TELE16 (anchored)', R.map_pred(P), '', P)
    P, _ = em(d['G16'], off, w1, w2, B=10, **kw); add(prm, 'CP TELE16 (PRM anchor only)', R.map_pred(P), '', P)
    # PRM-free: backbone telemetry (global level) and standalone telemetry (answer-z only)
    for name, X in (('backbone', d['G16']), ('standalone', d['sa_N16'])):
        fw1, fw2, fp = free_init(X, off)
        add(prm, f'FREE {name} first-pass init', fp)
        P, _ = em(X, off, fw1, fw2, B=10, iters=30, pi=None, h=h_from(fw1, off)); add(prm, f'FREE CP TELE16 {name} (EM)', R.map_pred(P), '', P)
        P, _ = em(X, off, fw1, fw2, B=10, iters=30, pi=.5, h=h_from(fw1, off)); add(prm, f'FREE CP TELE16 {name} (EM, pi=1/2)', R.map_pred(P), 'pi fixed 0.5 (no-information prior)', P)
    del d

np.savez_compressed('PREDS.npz', **{f'{k[0]}|{k[1]}': v for k, v in PREDS.items()}, **{f'LOC|{k[0]}|{k[1]}': v for k, v in LOCS.items()})
with open('RESULTS.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
