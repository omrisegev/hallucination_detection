"""Benchmark adapters for Hard2Verify and DeltaBench (rule C transfer, 2026-11-21). No change to ssl_data / ssl_lib.

load_c_pool(steps_npz, bench, gold_json=None): a ssl_data.Pool of the C answers of an extract_steps.py table.
  group = question group (H2V: uid without the '-<generator>' suffix -> question id; DeltaBench: whitespace-normalised
  question text, from answers.json), side = 'all' (no dev/test split: every rule-C fit is label-free).
  Labels are attached ONLY when gold_json is given ({uid, correct[], include[]}): err = ~correct, keep = include,
  y_ans = any(err & keep). Variant building and pseudo.py never pass gold_json.
"""
import json
from pathlib import Path
import numpy as np
import ssl_data as D

H2V_GENERATORS = ('-gemini-2.5-pro', '-claude-sonnet-4', '-GPT-5')


def h2v_question(uid):
    q = uid.split(':', 1)[1]
    for g in H2V_GENERATORS:
        if q.endswith(g): return q[: -len(g)]
    return q.rsplit('-', 1)[0]


def load_c_pool(steps_npz, bench, gold_json=None, groups=None):
    F = np.load(steps_npz)
    info = json.loads(str(F['info'])); assert info['gate']['status'] == 'PASS', info['gate']
    uids = F['uids'].astype(str); off = F['offsets'].astype(np.int64); ne = F['nonempty'].astype(bool)
    assert ne.all(), 'empty steps not expected in H2V / DeltaBench banks'
    P = D.Pool(); P.bench, P.V = bench, 'C'
    P.uids = uids; P.off = off
    P.top = F['top']; P.hard = F['stats'].astype(float); P.step5 = F['step5'].astype(float)
    P.act = F['digit_active'].astype(bool); P.ntok = F['ntok']; P.nonempty = ne
    P.pos = np.concatenate([np.arange(b - a) for a, b in zip(off[:-1], off[1:])])
    if groups is None:
        groups = {u: h2v_question(u) for u in uids} if bench == 'hard2verify' else None
    P.group = np.array([groups[u] for u in uids]); P.side = np.array(['all'] * len(uids))
    P.parent = uids.copy(); P.k = np.full(len(uids), -1)
    if gold_json is not None:
        G = {g['uid']: g for g in json.load(open(gold_json, encoding='utf8'))}
        P.err = np.zeros(off[-1], bool); P.keep = np.zeros(off[-1], bool)
        for j, u in enumerate(uids):
            a, b = off[j], off[j + 1]; c = np.asarray(G[u]['correct'], bool); i = np.asarray(G[u]['include'], bool)
            assert len(c) == b - a, (u, len(c), b - a)
            P.err[a:b] = ~c; P.keep[a:b] = i
        P.y_ans = np.add.reduceat((P.err & P.keep).astype(int), off[:-1]) > 0
        P.wrong_step = P.err & P.keep; P.clean_step = ~P.err & P.keep
    return P
