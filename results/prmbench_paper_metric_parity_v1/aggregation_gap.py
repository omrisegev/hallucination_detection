"""Optimal Aggregation of LLM and PRM Signals (arXiv 2510.13918): what can and cannot be computed.

Their metric: final-answer accuracy of Best-of-N / (weighted) majority voting over N sampled responses per
question (N = 32 on MATH, 112 on MATH500), from 5 generator LLMs x 7 PRMs, with weights calibrated on a
LABELED calibration set of questions (1k of the first 2k MATH-train questions; 100 of 200 for MATH500).
It needs, per question: many sampled responses, each with an extracted final answer and its correctness.

PRMBench has none of that: one fixed trace per perturbation, no sampled response pool, no graded final answer.
So their accuracy numbers are a GAP for every one of our arms. Data needed: e.g. MATH500 with N>=32 sampled
responses per question from a generator, final-answer correctness, Qwen2.5-Math-PRM-7B step rewards AND the
PRM-backbone telemetry from the same forward pass for every response (new GPU inference; not authorized here).

As a clearly labeled PROXY (NOT their metric, different task): inside each PRMBench source group (one
original question, 1-3 clean traces + ~8 perturbed variants) pick the trace with the lowest trace-level risk
(max step score) and count a hit if it has no annotated error step. Our fused scores contain an answer-
standardized PRM term, so they are only partly comparable ACROSS answers; this proxy penalizes that design.
"""
import json
import numpy as np
from common import HERE, ARMS, load_population, get_scores


def main():
    P = load_population(); S = get_scores(P)
    off, y, g = P['off'], P['y'], P['group']
    clean = np.array([not y[a:b].any() for a, b in zip(off[:-1], off[1:])])
    out = {'n_groups': int(len(np.unique(g))), 'arms': {}}
    rnd = []
    for u in np.unique(g):
        rnd.append(clean[g == u].mean())
    out['random_pick_hit_rate'] = 100 * float(np.mean(rnd))
    for a in ARMS:
        tr = np.array([S[a][s:e].max() for s, e in zip(off[:-1], off[1:])])
        hits = []
        for u in np.unique(g):
            idx = np.flatnonzero(g == u); hits.append(clean[idx[np.argmin(tr[idx])]])
        out['arms'][a] = 100 * float(np.mean(hits))
    json.dump(out, open(HERE / 'AGGREGATION_PROXY.json', 'w'), indent=1)
    print(out)


if __name__ == '__main__':
    main()
