"""Shared pieces for hurdle HMM v2 (calibrated). See docs/experiments/HURDLE_HMM_V2_PREREG_2026-11-02.md."""
import importlib.util, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
_sp = importlib.util.spec_from_file_location('hh', ROOT / 'spectral_utils' / 'hurdle_hmm.py')
H = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(H)


def gap_counts(s, off):  # verbatim copy of banks_robustness.py:gap_counts
    k = []
    for a, b in zip(off[:-1], off[1:]):
        m = b - a
        if m == 1: k.append(1); continue
        z = np.sort(s[a:b])[::-1]; km = min(int(np.ceil(m / 2)), m - 1)
        k.append(int(np.argmax(z[:km] - z[1:km + 1]) + 1))
    return np.array(k)


def topk_flags(s, off, k):
    f = np.zeros(len(s), bool)
    for a, (lo, hi) in enumerate(zip(off[:-1], off[1:])):
        if k[a] > 0:
            f[lo + np.argsort(-s[lo:hi], kind='stable')[:k[a]]] = True
    return f


def macro_f1(y, f, w=None):
    w = np.ones(len(y)) if w is None else w
    def f1(t, p):
        tp = (w * (t & p)).sum(); return 2 * tp / max((w * t).sum() + (w * p).sum(), 1e-12)
    return 0.5 * (f1(y, f) + f1(~y, ~f))


def calibrate(lab_ans, off, y_step, keep, has_err):
    """From the labelled answers only: pi, clean law, step error share, global macro-F1 threshold grid input."""
    m = len(lab_ans)
    pi = float(np.clip(has_err[lab_ans].mean(), 1 / (2 * m), 1 - 1 / (2 * m)))
    st = np.concatenate([np.arange(off[a], off[a + 1]) for a in lab_ans])
    st = st[keep[st]]
    return pi, st


def controls(x_raw, z, off, lab_steps, y_step, pi, s_share):
    n = np.diff(off)
    c = {}
    c['GAP'] = topk_flags(z, off, gap_counts(z, off))
    c['SHARE'] = topk_flags(z, off, np.maximum(1, np.round(s_share * n)).astype(int))
    amax = np.array([x_raw[a:b].max() for a, b in zip(off[:-1], off[1:])])
    gate = amax > np.quantile(amax, 1 - pi)
    c['GATE_GAP'] = c['GAP'] & np.repeat(gate, n)
    # global threshold on raw step score maximizing macro-F1 on the labelled steps (grid: 99 quantiles)
    xs, ys = x_raw[lab_steps], y_step[lab_steps]
    grid = np.quantile(xs, np.linspace(.01, .99, 99))
    best = max(grid, key=lambda t: macro_f1(ys, xs > t))
    c['THR'] = x_raw > best
    return c


def t6(off, y, keep, flags, has_err, ans_idx):
    te, tc = [], []
    for a in ans_idx:
        lo, hi = off[a], off[a + 1]
        fl = np.where(flags[lo:hi])[0]; pred = fl[0] if len(fl) else -1
        if has_err[a]:
            fe = np.where(y[lo:hi] & keep[lo:hi])[0]; te.append(len(fe) > 0 and pred == fe[0])
        else:
            tc.append(pred == -1)
    te, tc = float(np.mean(te)), float(np.mean(tc))
    return 2 * te * tc / (te + tc) if te + tc > 0 else 0.0
