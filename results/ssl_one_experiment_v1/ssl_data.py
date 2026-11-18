"""Load the step tables of the one SSL experiment into pools (bench x condition x split side). CPU.

Inputs (one folder W, as on AIRCC and in the PC copy):
  W/steps/STEPS_<bench>_<C|V1|V2>.npz      extract_steps.py output
  W/inputs/SPLIT.json, SIDECAR_<bench>_<V1|V2>.jsonl, prmbench_C_labels.json   make_variants.py output
  ProcessBench gold: results/prm_boost_processbench_v1/local/processbench_gold.json (uid, subset, label 0-based / -1)
Empty steps: ProcessBench drops them (as build_bank_external.py); PRMBench keeps them and fills their token readouts
with the answer mean (as build_bank.py). Gold labels are attached to C pools only and are read only for scoring,
the dev choices named in ANSWERS_TO_MAC_2026-11-18.md, and the reports.
"""
import json
from pathlib import Path

import numpy as np

import ssl_lib as L


class Pool:
    """Steps of a set of answers. Arrays over answers: uids, group, parent, k (0-based known-wrong step, V2), nst.
    Arrays over steps: top (S, 11, K), hard (S, 11, 4) [top10, mean, median, p90], step5, act, ntok, pos, nonempty."""

    def subset(self, amask):
        amask = np.asarray(amask, bool); P = Pool(); P.bench, P.V = self.bench, self.V
        rows = np.concatenate([np.arange(a, b) for a, b, m in zip(self.off[:-1], self.off[1:], amask) if m] or [np.zeros(0, int)])
        for k in ('uids', 'group', 'parent', 'k', 'side', 'lab', 'sub', 'y_ans', 'cls'):
            if getattr(self, k, None) is not None: setattr(P, k, getattr(self, k)[amask])
        for k in ('top', 'hard', 'step5', 'act', 'ntok', 'pos', 'nonempty', 'err', 'keep', 'clean_step', 'wrong_step'):
            if getattr(self, k, None) is not None: setattr(P, k, getattr(self, k)[rows])
        P.off = np.r_[0, np.cumsum(np.diff(self.off)[amask])].astype(np.int64)
        return P

    @property
    def n(self):
        return len(self.uids)


def _sidecar(W, bench, V):
    p = Path(W) / 'inputs' / f'SIDECAR_{bench}_{V}.jsonl'
    return {s['uid']: s for s in map(json.loads, open(p)) if 'skip' not in s}


def load_pool(W, bench, V, split, gold=None, labels=None):
    F = np.load(Path(W) / 'steps' / f'STEPS_{bench}_{V}.npz')
    info = json.loads(str(F['info'])); assert info['gate']['status'] == 'PASS', (bench, V, info['gate'])
    uids = F['uids'].astype(str); off = F['offsets'].astype(np.int64); ne = F['nonempty'].astype(bool)
    pos_full = np.concatenate([np.arange(b - a) for a, b in zip(off[:-1], off[1:])])
    keep_rows = ne if bench == 'processbench' else np.ones(len(ne), bool)
    nst = np.add.reduceat(keep_rows.astype(int), off[:-1]) if len(off) > 1 else np.zeros(0, int)
    nst[np.diff(off) == 0] = 0
    P = Pool(); P.bench, P.V = bench, V
    P.uids = uids; P.off = np.r_[0, np.cumsum(nst)].astype(np.int64)
    top = F['top']; st = F['stats']
    P.top = top[keep_rows]; P.hard = st[keep_rows].astype(float); P.step5 = F['step5'][keep_rows].astype(float)
    P.act = F['digit_active'][keep_rows].astype(bool); P.ntok = F['ntok'][keep_rows]; P.nonempty = ne[keep_rows]
    P.pos = np.concatenate([np.arange(n) for n in nst]) if len(nst) else np.zeros(0, int)
    # original step index -> local index (ProcessBench drops empty steps)
    loc = []
    for a, b in zip(off[:-1], off[1:]):
        m = keep_rows[a:b]; l = np.full(b - a, -1); l[m] = np.arange(m.sum()); loc.append(l)
    sp = split[bench]
    if V == 'C':
        P.group = np.array([sp[u][0] for u in uids]); P.side = np.array([sp[u][1] for u in uids])
        P.parent = uids.copy(); P.k = np.full(len(uids), -1)
    else:
        sc = _sidecar(W, bench, V)
        keep_a = np.array([u in sc for u in uids])
        P.group = np.array([sc[u]['group'] if u in sc else '' for u in uids])
        P.side = np.array([sc[u]['side'] if u in sc else '' for u in uids])
        P.parent = np.array([sc[u]['parent_uid'] if u in sc else '' for u in uids])
        if V in ('V2', 'V3', 'V4', 'V5'):      # known (first) wrong step k (V3-V5: make_variants_v34.py / _v5.py, dev only)
            P.k = np.array([loc[j][sc[u]['k'] - 1] if u in sc else -1 for j, u in enumerate(uids)])
            keep_a &= P.k >= 0
        else:
            P.k = np.full(len(uids), -1)
        for j, u in enumerate(uids):        # the condition's group / side must equal the parent's
            if u in sc: assert sp[sc[u]['parent_uid']] == [sc[u]['group'], sc[u]['side']], u
        if not keep_a.all(): P = P.subset(keep_a)
    if V == 'C':
        if bench == 'processbench':
            P.lab = np.array([gold[u]['label'] for u in P.uids]); P.sub = np.array([gold[u]['subset'] for u in P.uids])
            for j, u in enumerate(P.uids):          # gold index -> local index (no empty steps expected in PB)
                if P.lab[j] >= 0:
                    l = loc[j][P.lab[j]]; assert l >= 0, (u, 'gold first error on an empty step'); P.lab[j] = l
            P.y_ans = P.lab != -1
            P.wrong_step = np.zeros(P.off[-1], bool); P.clean_step = np.zeros(P.off[-1], bool)
            for j, (a, b) in enumerate(zip(P.off[:-1], P.off[1:])):
                if P.lab[j] < 0: P.clean_step[a:b] = True
                else: P.wrong_step[a + P.lab[j]] = True; P.clean_step[a:a + P.lab[j]] = True
        else:
            lab = {r['idx']: r for r in labels}
            P.err = np.zeros(P.off[-1], bool); P.cls = np.array([lab[u]['classification'] for u in P.uids])
            for j, u in enumerate(P.uids):
                a, b = P.off[j], P.off[j + 1]
                assert lab[u]['n_steps'] == b - a, (u, lab[u]['n_steps'], b - a)
                P.err[a:b] = np.isin(np.arange(1, b - a + 1), lab[u]['error_steps'])
            P.keep = np.repeat(P.cls != 'correct', np.diff(P.off))     # official PRMScore pool: 'correct' rows out
            P.y_ans = np.add.reduceat(P.err.astype(int), P.off[:-1]) > 0
            P.wrong_step = P.err.copy(); P.clean_step = ~P.err
    return P


def load_all(W, gold_path):
    split = json.load(open(Path(W) / 'inputs' / 'SPLIT.json'))
    gold = {g['uid']: g for g in json.load(open(gold_path))}
    lp = Path(W) / 'inputs' / 'prmbench_C_labels.json'
    labels = json.load(open(lp)) if lp.exists() else None
    out = {}
    for bench in ('processbench', 'prmbench'):
        for V in ('C', 'V1', 'V2'):
            p = Path(W) / 'steps' / f'STEPS_{bench}_{V}.npz'
            if p.exists():
                out[bench, V] = load_pool(W, bench, V, split, gold, labels)
    return out, split


# ------------------------------------------------------------------ readouts
def readout(P, N=None, revert=None):
    """Raw 16 channel step values. Token channels: hard top10 (N None) or count-based soft top-N (N: (11,));
    channels with revert True use hard top10. PRMBench empty steps get the answer mean of each token readout."""
    tok = P.hard[:, :, 0].copy()
    if N is not None:
        rv = np.zeros(11, bool) if revert is None else np.asarray(revert, bool)
        cols = np.flatnonzero(~rv)
        if len(cols):
            tok[:, cols] = L.soft_topn(P.top[:, cols, :].astype(float), np.asarray(N, float)[cols])
    if P.bench == 'prmbench' and not P.nonempty.all():
        tok[~P.nonempty] = np.nan
        tok = L.fill_empty_answer_mean(tok, P.off, P.nonempty)
    return np.column_stack([tok, P.step5])


def candidate(P, c, name):
    """Hard readout candidates of token channel c (answer 30): top1/2/5/10/20, mean, median, p90."""
    if name.startswith('top'):
        n = int(name[3:]); v = P.top[:, c, :n].astype(float)
        out = np.nanmean(v, 1)
    else:
        out = P.hard[:, c, {'mean': 1, 'median': 2, 'p90': 3}[name]].copy()
    if P.bench == 'prmbench' and not P.nonempty.all():
        out = L.fill_empty_answer_mean(np.where(P.nonempty, out, np.nan)[:, None], P.off, P.nonempty)[:, 0]
    return out
