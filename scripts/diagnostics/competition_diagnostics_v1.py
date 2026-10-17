#!/usr/bin/env python
"""Competition diagnostics on the frozen first-error locators (Step 430).

Protocol: docs/experiments/COMPETITION_DIAGNOSTICS_V1.md.  Labels are used for EVALUATION
only; nothing is fitted or selected.

A1  Generation drift.  Do the channels' statistics drift along the answer on a label-free
    reference population (answers the frozen CT7 gate closes)?  Drift curves by relative
    position and by absolute step, slopes with a source-group bootstrap, rank drift of the
    argmax, and the geometry of the fused locators' misses against the drift curve.
A2  Error complementarity.  Hit vectors of every channel x readout locator and of the fused
    locators on the erroneous ProcessBench answers; tied-argmax rates; phi correlation of
    hits; conditional hit rates given a reference miss, real against the shuffled-token null;
    union ceilings (ceilings, never results).
A3  Non-max decision rules on the frozen fused masses with the gate frozen.

Inputs: results/readout_family_v1/{profiles_full,shuffled_full}.npy, OOF_STEP_SCORES.npz, the
CT7 / token L-SML references and the frozen token matrices.  Outputs:
results/competition_diagnostics_v1/.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
sys.path.insert(0, str(ROOT))
from cvf_v2.core import CHANNELS  # noqa: E402
from cvf_v2.data import Dataset, config, digest, dump  # noqa: E402
from cvf_v2.report import csv_write  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#dde3eb', '#fcfcfb'
STRATA = [('steps_2_to_5', 2, 5), ('steps_6_to_10', 6, 10), ('steps_11_plus', 11, 10 ** 9)]
BINS = 10
MAX_STEP_INDEX = 20
FUSED = ['ct7', 'token_lsml', 'top5__all__soft__equal', 'top5__all__pmf__equal']
FAMILIES = {'peak': ['max', 'q90', 'top10', 'top30', 'log_top5', 'boxcar8_max'], 'spread': ['std', 'iqr', 'frac_above_z'],
            'trend': ['slope', 'jump'], 'sequential': ['cusum_top5', 'page_wmax', 'onset80'], 'first_token': ['first_token'],
            'mean': ['mean']}
DEGENERATE_TIE_RATE = 0.9
DELTAS = [0.25, 0.5, 1.0]
PRIMARY_DELTA = 0.5
EPS = np.finfo(float).eps


# ---------------------------------------------------------------- helpers
def answer_z(scores, off):
    z = np.empty(len(scores))
    for a, b in zip(off[:-1], off[1:]):
        v = np.asarray(scores[a:b], float)
        sd = v.std()
        z[a:b] = (v - v.mean()) / sd if sd > 1e-12 else 0.
    return z


def strata_masks(d):
    steps = np.diff(d.off)
    m = {'all': np.ones(d.n, bool)}
    m.update({name: (steps >= lo) & (steps <= hi) for name, lo, hi in STRATA})
    return m


def argmax_earliest(values, off):
    """Vectorised earliest-tie argmax per answer with the cvf_v2.readout.earliest_mode tolerance.
    Returns (local index, tied flag).  Non-finite entries (suffix-masked readouts) never win."""
    v = np.asarray(values, float)
    finite = np.isfinite(v)
    w = np.where(finite, v, -np.inf)
    mx = np.maximum.reduceat(w, off[:-1])
    absmax = np.maximum.reduceat(np.where(finite, np.abs(v), 0.), off[:-1])
    tol = 8 * EPS * np.maximum(1., absmax)
    thr = np.repeat(mx - tol, np.diff(off))
    mask = w >= thr
    idx = np.arange(len(v))
    BIG = len(v) + 1
    first = np.minimum.reduceat(np.where(mask, idx, BIG), off[:-1]) - off[:-1]
    ties = np.add.reduceat(mask.astype(int), off[:-1]) > 1
    return first, ties


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title(title, color=INK, fontsize=10, loc='left')
    ax.set_xlabel(xlabel, color=INK2, fontsize=8)
    ax.set_ylabel(ylabel, color=INK2, fontsize=8)


class Boot:
    """Source-group multinomial bootstrap weights shared by every interval of the run."""
    def __init__(self, d, draws):
        self.ug, self.ginv = np.unique(d.groups, return_inverse=True)
        self.G = len(self.ug)
        rng = np.random.default_rng(d.c['seed'])
        self.W = rng.multinomial(self.G, np.full(self.G, 1 / self.G), size=draws).astype(float)

    def group_sums(self, answer_idx, columns):
        """columns: (n_answers, k) values for the answers answer_idx -> (G, k) sums."""
        out = np.zeros((self.G, columns.shape[1]))
        np.add.at(out, self.ginv[answer_idx], columns)
        return out

    def ratio_ci(self, answer_idx, num, den):
        s = self.W @ self.group_sums(answer_idx, np.column_stack([num, den]))
        r = np.divide(s[:, 0], s[:, 1], out=np.full(len(s), np.nan), where=s[:, 1] > 0)
        return np.nanpercentile(r, [2.5, 97.5]).tolist()

    def diff_ratio_ci(self, answer_idx, num_a, num_b, den):
        s = self.W @ self.group_sums(answer_idx, np.column_stack([num_a, num_b, den]))
        ok = s[:, 2] > 0
        r = (s[ok, 0] - s[ok, 1]) / s[ok, 2]
        return np.percentile(r, [2.5, 97.5]).tolist() if ok.any() else [np.nan, np.nan]


def macro_pb(d, pred, boot=None, ref_pred=None):
    """Gate-free SLA macro8 and macro-F1 under the frozen gate, with optional paired bootstrap
    against a reference prediction of the same answers."""
    m = d.pb_metrics(pred)
    out = {'sla': m['sla'], 'f1': m['macro_f1']}
    for k in ['tolerance_one', 'early', 'late', 'mae']:
        out[k] = float(np.mean([c[k] for c in m['cells'].values()]))
    if boot is None:
        return out
    cells = sorted(set(d.cells[d.pb]))
    err = d.pb & (d.target >= 0); clean = d.pb & (d.target < 0)
    idx = np.flatnonzero(d.pb)
    cols = []
    for cell in cells:
        c = d.cells[idx] == cell
        e = c & (d.target[idx] >= 0); k = c & (d.target[idx] < 0)
        hit = (pred[idx] == d.target[idx]) & e
        rhit = (ref_pred[idx] == d.target[idx]) & e
        cols += [hit, hit & d.gate[idx], rhit, rhit & d.gate[idx], e, k & ~d.gate[idx], k]
    S = boot.W @ boot.group_sums(idx, np.column_stack(cols).astype(float))
    S = S.reshape(len(S), len(cells), 7)
    sla = S[:, :, 0] / S[:, :, 4]; rsla = S[:, :, 2] / S[:, :, 4]
    ea = S[:, :, 1] / S[:, :, 4]; rea = S[:, :, 3] / S[:, :, 4]; ca = S[:, :, 5] / S[:, :, 6]
    f1 = 2 * ca * ea / np.maximum(ca + ea, 1e-12); rf1 = 2 * ca * rea / np.maximum(ca + rea, 1e-12)
    out['delta_sla_vs_argmax'] = out['sla'] - d.pb_metrics(ref_pred)['sla']
    out['delta_sla_ci95'] = np.percentile((sla - rsla).mean(1), [2.5, 97.5]).tolist()
    out['delta_f1_vs_argmax'] = out['f1'] - d.pb_metrics(ref_pred)['macro_f1']
    out['delta_f1_ci95'] = np.percentile((f1 - rf1).mean(1), [2.5, 97.5]).tolist()
    return out


# ---------------------------------------------------------------- inputs
def load_series(d, p_full, oof):
    """Step-level series: fused (answer-standardised) and channel x readout (as stored)."""
    series = {}
    for name in FUSED:
        s = d.references[name] if name in d.references else oof[name]
        series[name] = answer_z(np.asarray(s, float), d.off)
    for j, ch in enumerate(CHANNELS):
        for r, rname in enumerate(d.readouts):
            series[f'{ch}__{rname}'] = np.asarray(p_full[:, j, r], float)
    return series


# ---------------------------------------------------------------- A1 drift
def drift(d, series, boot, out, tokens_path):
    started = time.perf_counter()
    steps = np.diff(d.off); S = int(d.off[-1])
    pos = np.concatenate([np.arange(b - a) for a, b in zip(d.off[:-1], d.off[1:])])
    ans = np.repeat(np.arange(d.n), steps)
    tgt = np.repeat(d.target, steps)
    nst = np.repeat(steps, steps)
    rel = np.where(nst > 1, pos / np.maximum(nst - 1, 1), np.nan)
    rbin = np.where(nst > 1, np.minimum((rel * BINS).astype(int), BINS - 1), -1)
    abin = np.minimum(pos, MAX_STEP_INDEX)
    pops = {'gate_closed': np.repeat(d.pb & ~d.gate, steps),
            'label_clean': np.repeat(d.pb & (d.target < 0), steps),
            'pre_error': np.repeat(d.pb & (d.target >= 0), steps) & (pos < tgt)}
    strata = strata_masks(d)
    centres = (np.arange(BINS) + .5) / BINS
    rows_pos, rows_abs, rows_slope = [], [], []
    curves = {}  # (series, population, stratum) -> bin means, for the miss geometry
    for n, (name, raw) in enumerate(series.items()):
        # Every step series is answer-standardised here (mean / std per answer, non-finite kept)
        # so that drift is measured in within-answer SD units for readouts of any scale.
        fin = np.isfinite(raw)
        v = raw.copy(); v[~fin] = np.nan
        for a, b in zip(d.off[:-1], d.off[1:]):
            x = v[a:b]; ok = np.isfinite(x)
            if ok.sum() > 1 and np.nanstd(x) > 1e-12:
                v[a:b] = (x - np.nanmean(x)) / np.nanstd(x)
            else:
                v[a:b] = np.where(ok, 0., np.nan)
        for pname, pmask in pops.items():
            for sname, smask in strata.items():
                m = pmask & fin & np.repeat(smask, steps)
                mb = m & (rbin >= 0)
                if mb.sum() < 50:
                    continue
                cnt = np.bincount(rbin[mb], minlength=BINS).astype(float)
                sm = np.bincount(rbin[mb], weights=v[mb], minlength=BINS)
                mean = np.divide(sm, cnt, out=np.full(BINS, np.nan), where=cnt > 0)
                curves[(name, pname, sname)] = mean
                for b in range(BINS):
                    rows_pos.append({'series': name, 'population': pname, 'stratum': sname, 'bin': b, 'centre': centres[b],
                                     'mean': mean[b], 'n': int(cnt[b])})
                ok = cnt > 0
                slope = float(np.polyfit(centres[ok], mean[ok], 1)[0]) if ok.sum() >= 3 else np.nan
                row = {'series': name, 'population': pname, 'stratum': sname, 'n_steps': int(mb.sum()),
                       'n_answers': int(len(np.unique(ans[mb]))), 'slope_bin_means': slope,
                       'last_minus_first_bin': mean[BINS - 1] - mean[0] if ok[0] and ok[-1] else np.nan}
                # Source-group bootstrap of the slope of the bin means.
                g = boot.ginv[ans[mb]]
                gs = np.bincount(g * BINS + rbin[mb], weights=v[mb], minlength=boot.G * BINS).reshape(boot.G, BINS)
                gc = np.bincount(g * BINS + rbin[mb], minlength=boot.G * BINS).reshape(boot.G, BINS).astype(float)
                bs = boot.W @ gs; bc = boot.W @ gc
                bm = np.divide(bs, bc, out=np.full(bs.shape, np.nan), where=bc > 0)
                good = np.isfinite(bm).all(1)
                if good.sum() > 10:
                    X = np.column_stack([centres, np.ones(BINS)])
                    beta = np.linalg.lstsq(X, bm[good].T, rcond=None)[0][0]
                    row['slope_ci95'] = np.percentile(beta, [2.5, 97.5]).tolist()
                if sname == 'all':
                    ca = np.bincount(abin[m], minlength=MAX_STEP_INDEX + 1).astype(float)
                    sa = np.bincount(abin[m], weights=v[m], minlength=MAX_STEP_INDEX + 1)
                    for k in range(MAX_STEP_INDEX + 1):
                        if ca[k] > 0:
                            rows_abs.append({'series': name, 'population': pname, 'step_index': k if k < MAX_STEP_INDEX else f'{MAX_STEP_INDEX}+',
                                             'mean': sa[k] / ca[k], 'n': int(ca[k])})
                rows_slope.append(row)
        if n % 40 == 0:
            print(f'A1 drift {n}/{len(series)} series, {time.perf_counter() - started:.1f}s', flush=True)
    csv_write(out / 'A1_DRIFT_BY_POSITION.csv', rows_pos)
    csv_write(out / 'A1_DRIFT_BY_STEP_INDEX.csv', rows_abs)
    # Rank drift: where does the argmax fall on gate-closed answers?
    rows_rank = []
    closed = d.pb & ~d.gate & (steps > 1)
    last_share = np.zeros(d.n); last_share[steps > 1] = np.array([(np.arange(s) / (s - 1) >= .8).mean() for s in steps[steps > 1]])
    for name, v in series.items():
        pred, _ = argmax_earliest(v, d.off)
        relp = np.where(steps > 1, pred / np.maximum(steps - 1, 1), np.nan)
        for sname, smask in strata.items():
            m = closed & smask
            if m.sum() < 20:
                continue
            rows_rank.append({'series': name, 'stratum': sname, 'n_answers': int(m.sum()),
                              'argmax_in_last_fifth': float((relp[m] >= .8).mean()), 'uniform_expectation_last_fifth': float(last_share[m].mean()),
                              'argmax_at_first_step': float((pred[m] == 0).mean()), 'uniform_expectation_first_step': float((1 / steps[m]).mean()),
                              'mean_relative_argmax': float(np.nanmean(relp[m]))})
    csv_write(out / 'A1_RANK_DRIFT.csv', rows_rank)
    for r in rows_slope:
        key = (r['series'], r['stratum'])
        rr = [x for x in rows_rank if (x['series'], x['stratum']) == key]
        if rr and r['population'] == 'gate_closed':
            r['argmax_in_last_fifth_gate_closed'] = rr[0]['argmax_in_last_fifth']
            r['uniform_expectation_last_fifth'] = rr[0]['uniform_expectation_last_fifth']
    csv_write(out / 'A1_DRIFT_SLOPES.csv', rows_slope)
    # Miss geometry of the fused locators against the gate-closed drift curve.
    rows_geo = []
    err = d.pb & (d.target >= 0) & (steps > 1)
    relt = np.where(steps > 1, d.target / np.maximum(steps - 1, 1), np.nan)
    for name in FUSED:
        v = series[name]
        pred, _ = argmax_earliest(v, d.off)
        relp = np.where(steps > 1, pred / np.maximum(steps - 1, 1), np.nan)
        delta = pred - d.target
        for sname, smask in strata.items():
            m = err & smask
            if m.sum() < 20:
                continue
            curve = curves.get((name, 'gate_closed', sname))
            row = {'series': name, 'stratum': sname, 'n_erroneous': int(m.sum()), 'sla': float((delta[m] == 0).mean()),
                   'early': float((delta[m] < 0).mean()), 'late': float((delta[m] > 0).mean()),
                   'mean_relative_target': float(np.nanmean(relt[m]))}
            for kind, km in [('late', m & (delta > 0)), ('early', m & (delta < 0))]:
                if km.sum() < 10:
                    continue
                row[f'{kind}_n'] = int(km.sum())
                row[f'{kind}_mean_relative_pred'] = float(np.nanmean(relp[km]))
                row[f'{kind}_mean_relative_target'] = float(np.nanmean(relt[km]))
                row[f'{kind}_mean_signed_relative_delta'] = float(np.nanmean((relp - relt)[km]))
                if curve is not None and np.isfinite(curve).all():
                    bp = np.minimum((relp[km] * BINS).astype(int), BINS - 1); bt = np.minimum((relt[km] * BINS).astype(int), BINS - 1)
                    diff = curve[bp] - curve[bt]
                    row[f'{kind}_null_at_pred_minus_target'] = float(diff.mean())
                    idx = np.flatnonzero(km)
                    row[f'{kind}_null_at_pred_minus_target_ci95'] = boot.ratio_ci(idx, diff, np.ones(len(idx)))
            rows_geo.append(row)
    csv_write(out / 'A1_MISS_GEOMETRY.csv', rows_geo)
    # Raw token channels at token level (gate-closed answers): relative token position drift.
    tok = np.load(tokens_path); tokens = tok['tokens']; toff = tok['token_offsets']
    rows_tok = []
    T = int(toff[-1]); tlen = np.diff(toff)
    tpos = np.concatenate([np.arange(n) for n in tlen]); tn = np.repeat(tlen, tlen)
    trel = np.minimum((tpos / np.maximum(tn - 1, 1) * BINS).astype(int), BINS - 1)
    for pname, amask in [('gate_closed', d.pb & ~d.gate), ('label_clean', d.pb & (d.target < 0))]:
        tm = np.repeat(amask, tlen)
        for j, ch in enumerate(CHANNELS):
            x = tokens[:, j].astype(float)
            z = np.empty(T)
            for i in np.flatnonzero(amask):
                a, b = toff[i:i + 2]; v = x[a:b]; sd = v.std()
                z[a:b] = (v - v.mean()) / sd if sd > 1e-12 else 0.
            for scale, vals in [('raw', x), ('answer_z', z)]:
                cnt = np.bincount(trel[tm], minlength=BINS).astype(float); sm = np.bincount(trel[tm], weights=vals[tm], minlength=BINS)
                mean = sm / cnt
                slope = float(np.polyfit(centres, mean, 1)[0])
                for b in range(BINS):
                    rows_tok.append({'channel': ch, 'scale': scale, 'population': pname, 'bin': b, 'centre': centres[b], 'mean': mean[b], 'n': int(cnt[b]), 'slope_bin_means': slope})
    csv_write(out / 'A1_TOKEN_DRIFT.csv', rows_tok)
    # Figure: gate-closed drift curves by stratum, fused locators in colour, channels at top5 in grey.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURFACE, sharey=True)
    for ax, (sname, _, _) in zip(axes, STRATA):
        for ch in CHANNELS:
            c = curves.get((f'{ch}__top5', 'gate_closed', sname))
            if c is not None:
                ax.plot(centres, c, color='#b8bcc4', linewidth=1)
                ax.text(centres[-1] + .01, c[-1], ch, fontsize=6, color=INK2, va='center')
        for name, col in zip(FUSED, PALETTE):
            c = curves.get((name, 'gate_closed', sname))
            if c is not None:
                ax.plot(centres, c, color=col, linewidth=2, label=name)
        ax.axhline(0, color=GRID, linewidth=1)
        style(ax, f'{sname}: gate-closed answers', 'relative position in the answer', 'mean answer-standardised step score (SD units)')
        ax.set_xlim(0, 1.15)
    axes[0].legend(fontsize=7.5, frameon=False, loc='upper left')
    fig.suptitle('A1 generation drift: mean step score by relative position on answers the frozen gate closes (channels at top5 in grey)',
                 color=INK, fontsize=10, x=.01, ha='left')
    fig.tight_layout(); fig.savefig(out / 'A1_DRIFT.png', dpi=160, facecolor=SURFACE); plt.close(fig)
    print(f'A1 done: {len(rows_slope)} slope rows, {time.perf_counter() - started:.1f}s', flush=True)
    return rows_slope, rows_geo


# ---------------------------------------------------------------- A2 complementarity
def complementarity(d, series, p_shuf, boot, out):
    started = time.perf_counter()
    steps = np.diff(d.off)
    err_idx = np.flatnonzero(d.pb & (d.target >= 0)); E = len(err_idx); tgt = d.target[err_idx]
    strata = {k: v[err_idx] for k, v in strata_masks(d).items()}
    names, preds, ties, shuf_preds = [], {}, {}, {}
    for name, v in series.items():
        p, t = argmax_earliest(v, d.off); preds[name] = p[err_idx]; ties[name] = t[err_idx]; names.append(name)
    for j, ch in enumerate(CHANNELS):
        for r, rname in enumerate(d.readouts):
            p, _ = argmax_earliest(np.asarray(p_shuf[:, j, r], float), d.off); shuf_preds[f'{ch}__{rname}'] = p[err_idx]
    hit = {n: preds[n] == tgt for n in names}; tol1 = {n: np.abs(preds[n] - tgt) <= 1 for n in names}
    shit = {n: shuf_preds[n] == tgt for n in shuf_preds}
    delta = {n: preds[n] - tgt for n in names}
    # Tied-argmax rates and the degenerate flag.
    rows_tie = []; degenerate = set()
    for n in names:
        rate = float(ties[n].mean())
        if n not in FUSED and rate > DEGENERATE_TIE_RATE:
            degenerate.add(n)
        rows_tie.append({'locator': n, 'tied_argmax_rate': rate, 'degenerate': n in degenerate, 'sla_exact': float(hit[n].mean()),
                         'sla_tolerance_one': float(tol1[n].mean()), 'early': float((delta[n] < 0).mean()), 'late': float((delta[n] > 0).mean()),
                         'shuffled_sla_exact': float(shit[n].mean()) if n in shit else None})
    csv_write(out / 'A2_TIED_RATES.csv', rows_tie)
    np.savez_compressed(out / 'A2_HIT_MATRIX.npz', answer_index=err_idx, locators=np.array(names),
                        hit=np.column_stack([hit[n] for n in names]), tolerance_one=np.column_stack([tol1[n] for n in names]),
                        shuffled_locators=np.array(list(shuf_preds)), shuffled_hit=np.column_stack([shit[n] for n in shuf_preds]))
    # Phi correlation of exact hits.
    def phi(a, b):
        a = a.astype(float); b = b.astype(float)
        sa, sb = a.std(), b.std()
        return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else np.nan
    rows_phi = []
    for ch in CHANNELS:
        for r1 in d.readouts:
            for r2 in d.readouts:
                if r1 >= r2:
                    continue
                a, b = hit[f'{ch}__{r1}'], hit[f'{ch}__{r2}']
                rows_phi.append({'scope': 'within_channel', 'channel': ch, 'a': r1, 'b': r2, 'phi_hits': phi(a, b), 'both': int((a & b).sum()),
                                 'a_only': int((a & ~b).sum()), 'b_only': int((b & ~a).sum()), 'phi_misses_shuffled': phi(shit[f'{ch}__{r1}'], shit[f'{ch}__{r2}'])})
    for rname in ['top5', 'top30']:
        for i, c1 in enumerate(CHANNELS):
            for c2 in CHANNELS[i + 1:]:
                a, b = hit[f'{c1}__{rname}'], hit[f'{c2}__{rname}']
                rows_phi.append({'scope': f'across_channels_{rname}', 'channel': '', 'a': c1, 'b': c2, 'phi_hits': phi(a, b), 'both': int((a & b).sum()),
                                 'a_only': int((a & ~b).sum()), 'b_only': int((b & ~a).sum())})
    for f1 in FUSED:
        for f2 in FUSED:
            if f1 < f2:
                rows_phi.append({'scope': 'fused', 'channel': '', 'a': f1, 'b': f2, 'phi_hits': phi(hit[f1], hit[f2]), 'both': int((hit[f1] & hit[f2]).sum()),
                                 'a_only': int((hit[f1] & ~hit[f2]).sum()), 'b_only': int((hit[f2] & ~hit[f1]).sum())})
    csv_write(out / 'A2_PHI.csv', rows_phi)
    # Conditional hit rates given a reference miss, real vs shuffled locator on the same real misses.
    rows_cond = []
    refs = {'same_channel_top5': None, 'fused_top5_soft_equal': 'top5__all__soft__equal', 'ct7': 'ct7'}
    for ch in CHANNELS:
        for rname in d.readouts:
            n = f'{ch}__{rname}'
            for rkey, rname_ref in refs.items():
                ref = f'{ch}__top5' if rname_ref is None else rname_ref
                for sname, smask in strata.items():
                    for kind in ['any', 'late', 'early']:
                        miss = ~hit[ref] & smask
                        if kind == 'late':
                            miss &= delta[ref] > 0
                        elif kind == 'early':
                            miss &= delta[ref] < 0
                        if miss.sum() < 20:
                            continue
                        real = float(hit[n][miss].mean()); shuf = float(shit[n][miss].mean())
                        row = {'locator': n, 'channel': ch, 'readout': rname, 'reference': rkey, 'stratum': sname, 'miss_type': kind,
                               'n_reference_misses': int(miss.sum()), 'hit_given_miss': real, 'hit_given_miss_tolerance_one': float(tol1[n][miss].mean()),
                               'shuffled_hit_given_miss': shuf, 'real_minus_shuffled': real - shuf, 'degenerate': n in degenerate}
                        if sname in ('all', 'steps_11_plus') and kind in ('any', 'late'):
                            row['real_minus_shuffled_ci95'] = boot.diff_ratio_ci(err_idx, (hit[n] & miss).astype(float), (shit[n] & miss).astype(float), miss.astype(float))
                        rows_cond.append(row)
    csv_write(out / 'A2_CONDITIONAL_HITS.csv', rows_cond)
    # Family-level complementarity: union of a family's readouts (all channels) given a fused miss.
    rows_fam = []
    for fam, members in FAMILIES.items():
        cols = [f'{ch}__{r}' for ch in CHANNELS for r in members if f'{ch}__{r}' not in degenerate]
        if not cols:
            continue
        union = np.any(np.column_stack([hit[c] for c in cols]), 1); sunion = np.any(np.column_stack([shit[c] for c in cols]), 1)
        top5 = np.any(np.column_stack([hit[f'{ch}__top5'] for ch in CHANNELS]), 1); stop5 = np.any(np.column_stack([shit[f'{ch}__top5'] for ch in CHANNELS]), 1)
        for ref in ['top5__all__soft__equal', 'ct7', 'any_channel_top5']:
            rhit = top5 if ref == 'any_channel_top5' else hit[ref]
            rdelta = None if ref == 'any_channel_top5' else delta[ref]
            for sname, smask in strata.items():
                for kind in ['any', 'late']:
                    miss = ~rhit & smask
                    if kind == 'late':
                        if rdelta is None:
                            continue
                        miss &= rdelta > 0
                    if miss.sum() < 20:
                        continue
                    real = float(union[miss].mean()); shuf = float(sunion[miss].mean())
                    rows_fam.append({'family': fam, 'members': ' '.join(members), 'n_locators': len(cols), 'reference': ref, 'stratum': sname, 'miss_type': kind,
                                     'n_reference_misses': int(miss.sum()), 'union_hit_given_miss': real, 'shuffled_union_hit_given_miss': shuf,
                                     'real_minus_shuffled': real - shuf,
                                     'real_minus_shuffled_ci95': boot.diff_ratio_ci(err_idx, (union & miss).astype(float), (sunion & miss).astype(float), miss.astype(float)),
                                     'any_channel_top5_hit_given_miss': float(top5[miss].mean()), 'shuffled_any_channel_top5_hit_given_miss': float(stop5[miss].mean())})
    csv_write(out / 'A2_FAMILY_COMPLEMENTARITY.csv', rows_fam)
    # Union ceilings (ceilings, never results).
    rows_union = []
    def add_union(scope, channel, setname, cols, scols):
        if not cols:
            return
        u = np.any(np.column_stack([hit[c] for c in cols]), 1); ut = np.any(np.column_stack([tol1[c] for c in cols]), 1)
        su = np.any(np.column_stack([shit[c] for c in scols]), 1) if scols else None
        for sname, smask in strata.items():
            rows_union.append({'scope': scope, 'channel': channel, 'set': setname, 'n_locators': len(cols), 'stratum': sname, 'n_erroneous': int(smask.sum()),
                               'union_sla_exact': float(u[smask].mean()), 'union_sla_tolerance_one': float(ut[smask].mean()),
                               'shuffled_union_sla_exact': float(su[smask].mean()) if su is not None else None})
    for ch in CHANNELS:
        base = f'{ch}__top5'
        add_union('within_channel', ch, 'top5', [base], [base])
        for r in d.readouts:
            n = f'{ch}__{r}'
            if r != 'top5' and n not in degenerate:
                add_union('within_channel', ch, f'top5+{r}', [base, n], [base, n])
        for fam, members in FAMILIES.items():
            cols = [base] + [f'{ch}__{r}' for r in members if f'{ch}__{r}' not in degenerate and r != 'top5']
            add_union('within_channel', ch, f'top5+{fam}', cols, cols)
        cols = [f'{ch}__{r}' for r in d.readouts if f'{ch}__{r}' not in degenerate]
        add_union('within_channel', ch, 'all_readouts', cols, cols)
    cols = [f'{ch}__top5' for ch in CHANNELS]; add_union('across_channels', '', 'all_channels_top5', cols, cols)
    cols = [n for n in names if n not in FUSED and n not in degenerate]; add_union('across_channels', '', 'everything', cols, cols)
    for f in FUSED:
        add_union('fused', '', f, [f], [])
        for fam, members in FAMILIES.items():
            cols = [f] + [f'{ch}__{r}' for ch in CHANNELS for r in members if f'{ch}__{r}' not in degenerate]
            add_union('fused', '', f'{f}+{fam}', cols, [])
        add_union('fused', '', f'{f}+everything', [f] + [n for n in names if n not in FUSED and n not in degenerate], [])
    add_union('fused', '', 'all_fused', FUSED, [])
    csv_write(out / 'A2_UNION_CEILINGS.csv', rows_union)
    # Figure: family union hit rate given a fused top5 miss, real vs shuffled, by stratum.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE, sharey=True)
    fams = list(FAMILIES)
    for ax, (sname, _, _) in zip(axes, STRATA):
        y = np.arange(len(fams)); real = []; shuf = []
        for fam in fams:
            r = [x for x in rows_fam if x['family'] == fam and x['reference'] == 'top5__all__soft__equal' and x['stratum'] == sname and x['miss_type'] == 'any']
            real.append(r[0]['union_hit_given_miss'] if r else np.nan); shuf.append(r[0]['shuffled_union_hit_given_miss'] if r else np.nan)
        ax.barh(y + .19, real, height=.34, color=PALETTE[0], label='real tokens')
        ax.barh(y - .19, shuf, height=.34, color=PALETTE[1], label='shuffled tokens (null)')
        for yy, a, b in zip(y, real, shuf):
            if np.isfinite(a):
                ax.text(a + .005, yy + .19, f'{a:.2f}', va='center', fontsize=7, color=INK)
            if np.isfinite(b):
                ax.text(b + .005, yy - .19, f'{b:.2f}', va='center', fontsize=7, color=INK)
        ax.set_yticks(y); ax.set_yticklabels(fams, fontsize=8, color=INK)
        style(ax, f'{sname}', 'P(any locator of the family hits | fused top5 misses)', '')
    axes[0].legend(fontsize=7.5, frameon=False, loc='lower right')
    fig.suptitle('A2 error complementarity: union hit rate of each readout family on the answers the fused top5 locator misses',
                 color=INK, fontsize=10, x=.01, ha='left')
    fig.tight_layout(); fig.savefig(out / 'A2_COMPLEMENTARITY.png', dpi=160, facecolor=SURFACE); plt.close(fig)
    print(f'A2 done: {len(rows_cond)} conditional rows, {len(degenerate)} degenerate locators, {time.perf_counter() - started:.1f}s', flush=True)
    return rows_fam, sorted(degenerate)


# ---------------------------------------------------------------- A3 decision rules
def decision_rules(d, series, boot, out):
    started = time.perf_counter()
    steps = np.diff(d.off); strata = strata_masks(d)
    rng = np.random.default_rng(d.c['seed'])
    err = d.pb & (d.target >= 0)
    rows = []; sla_by = {}
    for name in FUSED:
        z = series[name]  # answer-standardised
        first, _ = argmax_earliest(z, d.off)
        second = np.full(d.n, -1)
        for i, (a, b) in enumerate(zip(d.off[:-1], d.off[1:])):
            if b - a < 2:
                second[i] = first[i]; continue
            v = z[a:b].copy(); v[first[i]] = -np.inf
            second[i] = int(np.argmax(v))
        rules = {'argmax': first, 'earliest_of_top2': np.minimum(first, second), 'latest_of_top2': np.maximum(first, second),
                 'random_of_top2': np.where(rng.random(d.n) < .5, first, second)}
        zmax = np.repeat(np.maximum.reduceat(z, d.off[:-1]), steps)
        idx = np.arange(int(d.off[-1])); BIG = int(d.off[-1]) + 1
        for delta in DELTAS:
            mask = z >= zmax - delta
            rules[f'earliest_within_{delta}sd'] = np.minimum.reduceat(np.where(mask, idx, BIG), d.off[:-1]) - d.off[:-1]
        for rule, pred in rules.items():
            m = macro_pb(d, pred, boot, rules['argmax'])
            row = {'mass': name, 'rule': rule, 'primary_delta': rule == f'earliest_within_{PRIMARY_DELTA}sd' or rule in ('argmax', 'earliest_of_top2'), **m}
            for sname, smask in strata.items():
                e = err & smask
                row[f'sla_{sname}'] = float((pred[e] == d.target[e]).mean())
                row[f'late_{sname}'] = float((pred[e] > d.target[e]).mean())
            rows.append(row); sla_by[(name, rule)] = row
    csv_write(out / 'A3_DECISION_RULES.csv', rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE, sharey=True)
    rule_names = ['argmax', 'earliest_of_top2', 'latest_of_top2', 'random_of_top2', f'earliest_within_{PRIMARY_DELTA}sd']
    for ax, (sname, _, _) in zip(axes, STRATA):
        y = np.arange(len(rule_names)); width = .8 / len(FUSED)
        for k, (name, col) in enumerate(zip(FUSED, PALETTE)):
            vals = [sla_by[(name, r)][f'sla_{sname}'] for r in rule_names]
            ax.barh(y + (k - 1.5) * width, vals, height=width * .92, color=col, label=name)
        ax.set_yticks(y); ax.set_yticklabels(rule_names, fontsize=8, color=INK)
        style(ax, sname, 'first-error SLA (gate-free)', '')
    axes[0].legend(fontsize=7.5, frameon=False, loc='lower right')
    fig.suptitle('A3 decision rules on the frozen fused masses (gate frozen)', color=INK, fontsize=10, x=.01, ha='left')
    fig.tight_layout(); fig.savefig(out / 'A3_DECISION_RULES.png', dpi=160, facecolor=SURFACE); plt.close(fig)
    print(f'A3 done: {len(rows)} rows, {time.perf_counter() - started:.1f}s', flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT / 'configs/readout_family_v1.json'))
    p.add_argument('--out', default=str(ROOT / 'results/competition_diagnostics_v1'))
    p.add_argument('--only', choices=['a1', 'a2', 'a3', 'all'], default='all')
    p.add_argument('--draws', type=int, default=2000)
    args = p.parse_args()
    c = config(args.config); d = Dataset(c)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    p_full = np.load(d.out / d.profile_file); p_shuf = np.load(d.out / d.shuffle_file, mmap_mode='r')
    oof = np.load(d.out / 'OOF_STEP_SCORES.npz')
    series = load_series(d, p_full, oof)
    boot = Boot(d, args.draws)
    previous = json.loads((out / 'MANIFEST.json').read_text(encoding='utf8')) if (out / 'MANIFEST.json').exists() else {}
    manifest = {'protocol': 'docs/experiments/COMPETITION_DIAGNOSTICS_V1.md', 'config': c, 'profiles_full_sha256': digest(d.out / d.profile_file),
                'shuffled_full_sha256': digest(d.out / d.shuffle_file), 'oof_step_scores_sha256': digest(d.out / 'OOF_STEP_SCORES.npz'),
                'ct7_sha256': digest(c['paths']['ct7']), 'tokens_sha256': digest(c['paths']['tokens']), 'joined_sha256': digest(c['paths']['joined']),
                'labels_used_for': 'evaluation only; no fitting, selection or threshold choice', 'bootstrap_draws': args.draws,
                'strata': [s[0] for s in STRATA], 'bins': BINS, 'families': FAMILIES, 'degenerate_tie_rate': DEGENERATE_TIE_RATE,
                'deltas': DELTAS, 'primary_delta': PRIMARY_DELTA, 'fused': FUSED, 'series': len(series)}
    if args.only in ['a1', 'all']:
        drift(d, series, boot, out, c['paths']['tokens'])
    if args.only in ['a2', 'all']:
        _, degenerate = complementarity(d, series, p_shuf, boot, out); manifest['degenerate_locators'] = degenerate
    if args.only in ['a3', 'all']:
        boot10 = Boot(d, 10000)
        decision_rules(d, series, boot10, out)
    if 'degenerate_locators' not in manifest and 'degenerate_locators' in previous:
        manifest['degenerate_locators'] = previous['degenerate_locators']
    manifest['stages_run'] = args.only
    manifest['seconds'] = time.perf_counter() - started
    manifest['outputs'] = {f.name: {'bytes': f.stat().st_size, 'sha256': digest(f)} for f in out.iterdir() if f.is_file() and f.name != 'MANIFEST.json'}
    dump(out / 'MANIFEST.json', manifest)
    print(f'Diagnostics complete in {manifest["seconds"]:.0f}s -> {out}', flush=True)


if __name__ == '__main__':
    main()
