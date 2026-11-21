"""DeltaBench official section-level evaluation (CPU).

Source: github.com/OpenStellarTeam/DeltaBench @ 6b67c64ad177553ec73f85ff98b18b0e6627341f, evaluation.py
  - target sections of a sample = set(reason_unuseful_section_numbers + reason_error_section_numbers) (process_line)
  - per-sample precision / recall / F1 on section-number sets; predictions above max(target) are dropped (parse_output)
  - Overall = mean over samples (macro) of per-sample P / R / F1 (calculate_accuracies_v2); micro from summed tp/fp/fn
  - a sample with an empty target makes max() raise; process_line catches it and writes the row without metrics, so
    pandas .mean() skips it: such samples are EXCLUDED ('official'). We also report 'empty=1' (empty target and empty
    prediction -> F1 1, else 0) as a second convention.
PRM threshold rule (paper arXiv 2502.19361, not in the script): mu, sigma of the PRM reward over ALL sections of ALL
samples; a section is predicted erroneous if reward < mu - sigma. Appendix variant: reward below the 5% dataset quantile.
For a risk-high score s (ours), the same rule reads s > mu + sigma.
"""
import ast, json
import numpy as np


def official_targets(jsonl, error_only=False):
    out = {}
    for l in open(jsonl, encoding='utf8'):
        r = json.loads(l)
        f = lambda v: v if isinstance(v, list) else ast.literal_eval(v)
        out['deltabench:' + r['id']] = sorted(set(map(int, ([] if error_only else f(r['reason_unuseful_section_numbers'])) + f(r['reason_error_section_numbers']))))
    return out


def per_sample(pred_sets, targets):
    """pred_sets, targets: lists of sets of 1-based section numbers. Returns arrays P, R, F1 (nan = excluded), tp, fp, fn."""
    n = len(targets); P = np.full(n, np.nan); R = P.copy(); F = P.copy(); F_e1 = np.zeros(n); tp = np.zeros(n); fp = tp.copy(); fn = tp.copy()
    for i, (p, t) in enumerate(zip(pred_sets, targets)):
        t = set(t); p = set(p)
        if not t:
            F_e1[i] = 1. if not p else 0.; continue
        p = {x for x in p if x <= max(t)}
        a, b, c = len(p & t), len(p - t), len(t - p); tp[i], fp[i], fn[i] = a, b, c
        P[i] = a / (a + b) if a + b else 0.; R[i] = a / (a + c) if a + c else 0.
        F[i] = 2 * P[i] * R[i] / (P[i] + R[i]) if P[i] + R[i] else 0.; F_e1[i] = F[i]
    return dict(P=P, R=R, F1=F, F1_empty1=F_e1, tp=tp, fp=fp, fn=fn)


def summarize(m, idx=None):
    idx = np.arange(len(m['F1'])) if idx is None else idx
    T = m['tp'][idx].sum(); Fp = m['fp'][idx].sum(); Fn = m['fn'][idx].sum()
    pm = T / (T + Fp) if T + Fp else 0.; rm = T / (T + Fn) if T + Fn else 0.
    return dict(f1_macro=100 * np.nanmean(m['F1'][idx]), precision_macro=100 * np.nanmean(m['P'][idx]),
                recall_macro=100 * np.nanmean(m['R'][idx]), f1_macro_empty1=100 * m['F1_empty1'][idx].mean(),
                f1_micro=100 * (2 * pm * rm / (pm + rm) if pm + rm else 0.), n_scored=int(np.isfinite(m['F1'][idx]).sum()), n=int(len(idx)))


def sets_from_flags(flags, off):
    return [set((np.flatnonzero(flags[a:b]) + 1).tolist()) for a, b in zip(off[:-1], off[1:])]


def musigma_flags(score, risk_high, k=1.0):
    mu, sd = float(np.mean(score)), float(np.std(score))
    return (score > mu + k * sd) if risk_high else (score < mu - k * sd)


def quantile_flags(score, risk_high, q=.05):
    return (score > np.quantile(score, 1 - q)) if risk_high else (score < np.quantile(score, q))


def hit_at_1(score, risk_high, off, targets):
    h = []
    for (a, b), t in zip(zip(off[:-1], off[1:]), targets):
        if not t: continue
        v = score[a:b]; j = int(np.argmax(v) if risk_high else np.argmin(v)) + 1
        h.append(j in set(t))
    return 100 * float(np.mean(h))
