"""Descriptive (labels used): how the step labels of PRMBench and Socratic-PRMBench are laid out, and how much of
Socratic's step-level signal is position, per reasoning-pattern category. Prompted by Omri's question (2026-10-26)
whether Socratic is really a first-error (ProcessBench-like) benchmark.

    python label_structure.py   # -> LABEL_STRUCTURE.json, SOCRATIC_BY_CATEGORY.csv
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval8 as V  # noqa: E402
E = V.E


def structure(err, off, keep=None):
    out = defaultdict(list)
    for a, b in zip(off[:-1], off[1:]):
        m = slice(a, b) if keep is None else np.flatnonzero(keep[a:b]) + a
        e = np.asarray(err[m], bool); n = len(e)
        if n == 0: continue
        out['has_error'].append(e.any())
        if not e.any(): continue
        f = int(np.argmax(e))
        out['n_error_steps'].append(e.sum()); out['single'].append(e.sum() == 1)
        out['suffix'].append(e[f:].all()); out['suffix_multi'].append(e[f:].all() and e.sum() > 1)
        out['contiguous'].append(e[f:f + e.sum()].all()); out['last_is_error'].append(bool(e[-1]))
        out['first_rel_pos'].append((f + .5) / n)
    return {k: float(np.mean(v)) for k, v in out.items()} | {'n_answers': len(out['has_error'])}


def pos_auc(err, off, keep):
    return float(np.nanmean(E.per_answer_auc(err, V.rel_pos(off), off, keep)))


def main():
    res = {}
    c = E.load('prmbench', 'qwen_prm_7b')
    res['prmbench_all_answers'] = structure(c.err, c.off)
    res['prmbench_non_control'] = structure(c.err, c.off, c.keep)
    res['prmbench_position_within_auc'] = pos_auc(c.err, c.off, None)
    del c
    rows = []
    for prm in ('qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm'):
        c = E.load('socratic', prm)
        if prm == 'qwen_prm_7b':
            res['socratic'] = structure(c.err, c.off, c.keep)
            res['socratic_position_within_auc'] = pos_auc(c.err, c.off, c.keep)
        D = np.load(V.SCORES / f'{V.key("socratic", prm)}.npz')
        arms = {'PRM': D['PRM'], 'SUB5_raw_+PRM': D['SUB5__RAW__PPRM'], 'SUB5_pos_+PRM': D['SUB5__POS__PPRM'],
                'SUB5_pos_alone': D['SUB5__POS__alone'], 'position': V.rel_pos(c.off)}
        for cat in sorted(set(c.cat)):
            ans = np.flatnonzero(c.cat == cat)
            sel = np.zeros(c.off[-1], bool)
            for j in ans: sel[c.off[j]:c.off[j + 1]] = True
            keep = c.keep & sel
            row = dict(prm=prm, category=cat, n_answers=len(ans))
            for k, s in arms.items():
                row[f'auc_{k}'] = float(np.nanmean(E.per_answer_auc(c.err, s, c.off, keep)))
            rows.append(row)
        print('done', prm, flush=True); del c
    E.write_csv(HERE / 'SOCRATIC_BY_CATEGORY.csv', rows)
    (HERE / 'LABEL_STRUCTURE.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
