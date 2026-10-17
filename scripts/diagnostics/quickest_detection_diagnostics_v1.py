#!/usr/bin/env python
"""Quickest-detection diagnostics of the frozen first-error locators (Step 428).

Three questions, labels used for EVALUATION only (no fitting, no selection, no threshold
chosen on labels):

B2  Markov persistence on PRMBench per-step flags.  The quickest-detection framing (Itkin,
    arXiv 2606.12476) assumes one latent switch faithful -> erroneous that then persists.
    Does that hold on math chains?  P(err_s | err_{s-1}) vs P(err_s | ok_{s-1}), run-length
    distribution, fraction of answers with more than one error run, hazard by step index.
B4  Empirical delay floor per locator.  A Lorden-type bound says the mean detection delay
    at false-alarm rate alpha is at least ln(1/alpha) / KL(f1 || f0), where f1 is the score
    distribution on first-error steps and f0 on error-free steps.  Estimating that
    divergence per channel x readout and per depth stratum answers whether the long-chain
    collapse is a readout problem (fixable) or a channel problem (not).
B1  Detection delay versus false alarm.  Replace the argmax by a first-crossing rule with
    a swept global threshold on the answer-standardised step score: false alarms on clean
    ProcessBench answers, early stops / delays on erroneous ones, by depth stratum.  Reported
    for the offline (whole-answer standardised) scores and for a causal variant standardised
    on the answer prefix only.

Inputs are the frozen artefacts of the readout-family run (profiles_full.npy) and the
frozen token matrices; nothing is refitted.  Outputs go to
results/quickest_detection_diagnostics_v1/.
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
from spectral_utils.step_readouts_v1 import robust_standardize_tokens, step_page_wmax, step_topk  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette, slots 1-4 in fixed order (dataviz skill); text wears ink.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#dde3eb', '#fcfcfb'
ALPHA = 0.01
STRATA = [('steps_2_to_5', 2, 5), ('steps_6_to_10', 6, 10), ('steps_11_plus', 11, 10 ** 9)]
FA_TARGETS = [0.05, 0.10, 0.20, 0.30]
B1_PAGE_KS = [0.25, 1.0]


# ---------------------------------------------------------------- helpers
def answer_z(scores, off):
    """Whole-answer standardisation (mean / std per answer; a constant answer becomes 0)."""
    z = np.empty(len(scores))
    for a, b in zip(off[:-1], off[1:]):
        v = np.asarray(scores[a:b], float)
        sd = v.std()
        z[a:b] = (v - v.mean()) / sd if sd > 1e-12 else 0.
    return z


def causal_z(x):
    """Prefix-only standardisation of a (T, C) token matrix: token t is standardised with the
    mean and std of tokens 0..t (inclusive); the first token is 0."""
    x = np.asarray(x, float)
    n = np.arange(1, len(x) + 1)[:, None]
    mean = np.cumsum(x, axis=0) / n
    var = np.cumsum(x ** 2, axis=0) / n - mean ** 2
    sd = np.sqrt(np.maximum(var, 0.))
    z = np.where(sd > 1e-12, (x - mean) / np.where(sd > 1e-12, sd, 1.), 0.)
    return z


def strata_masks(d):
    steps = np.diff(d.off)
    return {name: (steps >= lo) & (steps <= hi) for name, lo, hi in STRATA}


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


# ---------------------------------------------------------------- locators
def build_locators(d, p, out):
    """Step-score arrays over all steps, 'higher = more suspect'.  Returns
    (b4_locators, b1_locators): the wide roster for the divergence table and the narrow
    roster for the first-crossing sweep."""
    started = time.perf_counter()
    b4 = {}
    for name in ['ct7', 'token_lsml', 'token_equal', 'mindgap']:
        b4[name] = np.asarray(d.references[name], float)
    for j, ch in enumerate(CHANNELS):
        for r, rname in enumerate(d.readouts):
            b4[f'{ch}__{rname}'] = np.asarray(p[:, j, r], float)
    b4['control__longest_step'] = np.load(d.out / 'step_lengths.npy').astype(float)
    oof = d.out / 'OOF_STEP_SCORES.npz'
    if oof.exists():
        z = np.load(oof)
        for key in ['consensus__all__soft__equal', 'consensus__all__pmf__equal',
                    'consensus__all__soft__continuous_lsml', 'top5__all__soft__equal']:
            if key in z.files:
                b4['oof__' + key] = z[key]
    # Page-k sensitivity and the causal variant need the token matrices.
    tok = np.load(d.c['paths']['tokens'])
    tokens, toff, spans = tok['tokens'], tok['token_offsets'], tok['step_spans']
    extra = {f'{ch}__page_wmax_k{k}': np.empty(int(d.off[-1])) for ch in CHANNELS for k in B1_PAGE_KS}
    extra.update({f'{ch}__{v}': np.empty(int(d.off[-1])) for ch in CHANNELS for v in ['top5_causal', 'page_wmax_causal']})
    for i, (a, b) in enumerate(zip(d.off[:-1], d.off[1:])):
        ta, tb = toff[i:i + 2]
        x = tokens[ta:tb].astype(float)
        sp = spans[a:b]
        z = robust_standardize_tokens(x)
        for k in B1_PAGE_KS:
            w = step_page_wmax(z, sp, k=k)
            for j, ch in enumerate(CHANNELS):
                extra[f'{ch}__page_wmax_k{k}'][a:b] = w[:, j]
        zc = causal_z(x)
        t5 = step_topk(zc, sp, 5)
        wc = step_page_wmax(zc, sp, k=d.c.get('page_k', .5))
        for j, ch in enumerate(CHANNELS):
            extra[f'{ch}__top5_causal'][a:b] = t5[:, j]
            extra[f'{ch}__page_wmax_causal'][a:b] = wc[:, j]
        if i % 2000 == 0:
            print(f'locators {i}/{d.n}, {time.perf_counter() - started:.1f}s', flush=True)
    b4.update(extra)
    b1_names = (['ct7', 'token_lsml', 'token_equal', 'mindgap', 'control__longest_step']
                + [f'{ch}__top5' for ch in CHANNELS] + [f'{ch}__page_wmax' for ch in CHANNELS]
                + list(extra) + [n for n in b4 if n.startswith('oof__')])
    b1 = {n: b4[n] for n in b1_names}
    print(f'locators built: {len(b4)} for B4, {len(b1)} for B1, {time.perf_counter() - started:.1f}s', flush=True)
    return b4, b1


# ---------------------------------------------------------------- B2 persistence
def markov_persistence(d, out, draws=2000):
    prm = np.flatnonzero(d.prm)
    cls = np.array([d.meta_by_id[d.ids[i]]['classification'] for i in prm])
    steps = np.diff(d.off)
    groups = d.groups[prm]
    ug, ginv = np.unique(groups, return_inverse=True)
    # Per-answer transition counts: [err|err numerator, err denominator, err|ok numerator, ok denominator,
    #                                error steps, steps, runs, answers-with-error, answers-with->1-run]
    per = np.zeros((len(prm), 9))
    run_lengths = []
    hazard_num = np.zeros(30)
    hazard_den = np.zeros(30)
    for n, i in enumerate(prm):
        f = d.labels[d.off[i]:d.off[i + 1]].astype(int)
        prev, cur = f[:-1], f[1:]
        per[n, 0] = np.sum((prev == 1) & (cur == 1)); per[n, 1] = np.sum(prev == 1)
        per[n, 2] = np.sum((prev == 0) & (cur == 1)); per[n, 3] = np.sum(prev == 0)
        per[n, 4] = f.sum(); per[n, 5] = len(f)
        edges = np.diff(np.r_[0, f, 0])
        starts = np.flatnonzero(edges == 1); ends = np.flatnonzero(edges == -1)
        per[n, 6] = len(starts); per[n, 7] = f.any(); per[n, 8] = len(starts) > 1
        run_lengths.extend((ends - starts).tolist())
        first = int(np.argmax(f)) if f.any() else len(f)
        for s in range(min(len(f), 30)):
            if s > first:
                break
            hazard_den[s] += 1
            hazard_num[s] += (s == first)
    def summarise(mask):
        c = per[mask].sum(0)
        return {'answers': int(mask.sum()), 'p_err_given_err': c[0] / c[1] if c[1] else None,
                'p_err_given_ok': c[2] / c[3] if c[3] else None, 'error_rate': c[4] / c[5],
                'mean_runs_per_erroneous_answer': c[6] / c[7] if c[7] else None,
                'fraction_erroneous_with_multiple_runs': c[8] / c[7] if c[7] else None}
    rows = [{'scope': 'all', **summarise(np.ones(len(prm), bool))}]
    for c in sorted(set(cls)):
        rows.append({'scope': 'classification=' + c, **summarise(cls == c)})
    for name, lo, hi in [('steps_1', 1, 1)] + STRATA:
        rows.append({'scope': name, **summarise((steps[prm] >= lo) & (steps[prm] <= hi))})
    # Source-group bootstrap of the two conditional probabilities (all PRMBench answers).
    G = len(ug)
    agg = np.zeros((G, 4))
    np.add.at(agg, ginv, per[:, :4])
    rng = np.random.default_rng(d.c['seed'])
    w = rng.multinomial(G, np.full(G, 1 / G), size=draws).astype(float)
    boot = w @ agg
    ci = {'p_err_given_err': np.percentile(boot[:, 0] / boot[:, 1], [2.5, 97.5]).tolist(),
          'p_err_given_ok': np.percentile(boot[:, 2] / boot[:, 3], [2.5, 97.5]).tolist()}
    lengths = np.bincount(np.asarray(run_lengths, int))
    result = {'rows': rows, 'ci95_all_source_bootstrap': ci, 'draws': draws,
              'run_length_histogram': {str(k): int(v) for k, v in enumerate(lengths) if v},
              'hazard_first_error_by_step_index': {'numerator': hazard_num.tolist(), 'denominator': hazard_den.tolist(),
                                                   'hazard': np.divide(hazard_num, hazard_den, out=np.full(30, np.nan), where=hazard_den > 0).tolist()},
              'reading': 'p_err_given_err far above p_err_given_ok = errors persist once they start (the one-switch premise); '
                         'fraction_erroneous_with_multiple_runs = how often the persistence breaks.'}
    dump(out / 'B2_PERSISTENCE.json', result)
    csv_write(out / 'B2_PERSISTENCE.csv', rows)
    # Figure: paired horizontal bars per classification, two series with legend and direct labels.
    cl_rows = [r for r in rows if r['scope'].startswith('classification=') and r['p_err_given_err'] is not None and r['p_err_given_ok'] is not None]
    cl_rows.sort(key=lambda r: r['p_err_given_err'] or 0)
    fig, ax = plt.subplots(figsize=(7.5, .38 * len(cl_rows) + 1.6), facecolor=SURFACE)
    y = np.arange(len(cl_rows))
    a = [r['p_err_given_err'] for r in cl_rows]; b = [r['p_err_given_ok'] for r in cl_rows]
    ax.barh(y + .19, a, height=.34, color=PALETTE[0], label='P(error at s | error at s-1)')
    ax.barh(y - .19, b, height=.34, color=PALETTE[1], label='P(error at s | no error at s-1)')
    for yy, va, vb in zip(y, a, b):
        ax.text(va + .01, yy + .19, f'{va:.2f}', va='center', fontsize=7, color=INK)
        ax.text(vb + .01, yy - .19, f'{vb:.2f}', va='center', fontsize=7, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r['scope'].split('=')[1] for r in cl_rows], fontsize=8, color=INK)
    ax.set_xlim(0, 1.08)
    style(ax, 'PRMBench: does an error persist to the next step?', 'conditional probability', '')
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out / 'B2_PERSISTENCE.png', dpi=160, facecolor=SURFACE); plt.close(fig)
    print('B2 done: P(err|err)=%.3f P(err|ok)=%.3f multi-run=%.3f' % (rows[0]['p_err_given_err'], rows[0]['p_err_given_ok'],
          rows[0]['fraction_erroneous_with_multiple_runs']), flush=True)
    return result


# ---------------------------------------------------------------- B4 delay floor
def delay_floor(d, locators, out, draws=1000, bins=48):
    pb = d.pb; err = pb & (d.target >= 0); clean = pb & (d.target < 0)
    steps = np.diff(d.off)
    # Step-level membership: first-error steps (f1), error-free steps (f0), post-error (f2).
    pos = np.concatenate([np.arange(b - a) for a, b in zip(d.off[:-1], d.off[1:])])
    ans = np.repeat(np.arange(d.n), steps)
    tgt = np.repeat(d.target, steps)
    f1 = np.repeat(err, steps) & (pos == tgt)
    f0 = (np.repeat(err, steps) & (pos < tgt)) | np.repeat(clean, steps)
    f2 = np.repeat(err, steps) & (pos > tgt)
    strata = strata_masks(d)
    scopes = {'macro8_all': np.repeat(pb, steps)}
    for cell in sorted(set(d.cells[pb])):
        scopes['cell=' + cell] = np.repeat(d.cells == cell, steps)
    for name, m in strata.items():
        scopes[name] = np.repeat(pb & m, steps)
    ug, ginv = np.unique(d.groups, return_inverse=True); G = len(ug)
    rng = np.random.default_rng(d.c['seed'])
    W = rng.multinomial(G, np.full(G, 1 / G), size=draws).astype(float)
    rows = []
    started = time.perf_counter()
    def divergence(c0, c1):
        p0 = (c0 + .5) / (c0 + .5).sum(); p1 = (c1 + .5) / (c1 + .5).sum()
        kl = float(np.sum(p1 * np.log(p1 / p0))); bc = float(-np.log(np.sum(np.sqrt(p0 * p1))))
        return kl, bc
    for n, (name, s) in enumerate(locators.items()):
        for scale, values in [('raw', np.asarray(s, float)), ('answer_z', answer_z(s, d.off))]:
            for scope, smask in scopes.items():
                m0 = f0 & smask; m1 = f1 & smask; m2 = f2 & smask
                if m1.sum() < 20 or m0.sum() < 20:
                    continue
                pooled = values[m0 | m1]
                edges = np.unique(np.quantile(pooled, np.linspace(0, 1, bins + 1)))
                if len(edges) < 3:
                    rows.append({'locator': name, 'scale': scale, 'scope': scope, 'kl_first_error_vs_error_free': 0.,
                                 'bhattacharyya': 0., 'floor_steps_alpha_0.01': np.inf, 'n_first_error': int(m1.sum()), 'n_error_free': int(m0.sum())})
                    continue
                edges[0], edges[-1] = -np.inf, np.inf
                b0 = np.clip(np.searchsorted(edges, values[m0], side='right') - 1, 0, len(edges) - 2)
                b1 = np.clip(np.searchsorted(edges, values[m1], side='right') - 1, 0, len(edges) - 2)
                b2 = np.clip(np.searchsorted(edges, values[m2], side='right') - 1, 0, len(edges) - 2) if m2.any() else None
                c0 = np.bincount(b0, minlength=len(edges) - 1).astype(float); c1 = np.bincount(b1, minlength=len(edges) - 1).astype(float)
                kl, bc = divergence(c0, c1)
                row = {'locator': name, 'scale': scale, 'scope': scope, 'kl_first_error_vs_error_free': kl, 'bhattacharyya': bc,
                       'floor_steps_alpha_0.01': np.log(1 / ALPHA) / kl if kl > 0 else np.inf,
                       'n_first_error': int(m1.sum()), 'n_error_free': int(m0.sum())}
                if b2 is not None:
                    c2 = np.bincount(b2, minlength=len(edges) - 1).astype(float)
                    row['kl_post_error_vs_error_free'] = divergence(c0, c2)[0]
                if scope == 'macro8_all' or scope in strata:
                    g0 = np.zeros((G, len(c0))); g1 = np.zeros((G, len(c1)))
                    np.add.at(g0, (ginv[ans[m0]], b0), 1.); np.add.at(g1, (ginv[ans[m1]], b1), 1.)
                    B0 = W @ g0 + .5; B1 = W @ g1 + .5
                    P0 = B0 / B0.sum(1, keepdims=True); P1 = B1 / B1.sum(1, keepdims=True)
                    kls = np.sum(P1 * np.log(P1 / P0), axis=1)
                    row['kl_ci95'] = np.percentile(kls, [2.5, 97.5]).tolist()
                    row['floor_ci95'] = (np.log(1 / ALPHA) / np.percentile(kls, [97.5, 2.5])).tolist()
                rows.append(row)
        if n % 25 == 0:
            print(f'B4 {n}/{len(locators)} locators, {time.perf_counter() - started:.1f}s', flush=True)
    csv_write(out / 'B4_DELAY_FLOOR.csv', rows)
    # Figure: per stratum, floor in steps for each channel at top5 vs its best extended readout,
    # with CT7 and the token fusion as reference lines.  Two series + two references, legend + labels.
    ext = [r for r in d.readouts if r not in ('top5', 'top10', 'max', 'mean', 'log_top5', 'cusum_top5', 'onset80')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=SURFACE, sharey=True)
    table = {(r['locator'], r['scale'], r['scope']): r for r in rows}
    best_rows = []
    for ax, (name, _, _) in zip(axes, STRATA):
        ys = np.arange(len(CHANNELS)); xs_top5 = []; xs_best = []; labels = []
        for ch in CHANNELS:
            base = table.get((f'{ch}__top5', 'answer_z', name), {}).get('floor_steps_alpha_0.01', np.nan)
            cands = [(table.get((f'{ch}__{r}', 'answer_z', name), {}).get('floor_steps_alpha_0.01', np.inf), r) for r in ext]
            best_val, best_r = min(cands)
            xs_top5.append(base); xs_best.append(best_val); labels.append(best_r)
            best_rows.append({'stratum': name, 'channel': ch, 'floor_top5': base, 'best_extended_readout': best_r, 'floor_best_extended': best_val})
        ax.scatter(xs_top5, ys, s=42, color=PALETTE[0], label='top5 readout', zorder=3)
        ax.scatter(xs_best, ys, s=42, color=PALETTE[1], label='best extended readout', zorder=3)
        for yy, xb, lab in zip(ys, xs_best, labels):
            if np.isfinite(xb):
                ax.text(xb, yy + .28, lab, fontsize=6.5, color=INK2, ha='center')
        for ref, col in [('ct7', PALETTE[2]), ('token_lsml', PALETTE[3])]:
            v = table.get((ref, 'answer_z', name), {}).get('floor_steps_alpha_0.01', np.nan)
            if np.isfinite(v):
                ax.axvline(v, color=col, linewidth=2, label=ref)
        ax.set_xscale('log'); ax.set_yticks(ys); ax.set_yticklabels(CHANNELS, fontsize=7.5, color=INK)
        style(ax, f'{name}: delay floor ln(100)/KL', 'minimum mean delay (steps, log scale)', '')
    axes[0].legend(fontsize=7.5, frameon=False, loc='lower right')
    fig.suptitle('Lorden-type delay floor at false-alarm rate 0.01, answer-standardised scores', color=INK, fontsize=11, x=.01, ha='left')
    fig.tight_layout(); fig.savefig(out / 'B4_DELAY_FLOOR.png', dpi=160, facecolor=SURFACE); plt.close(fig)
    csv_write(out / 'B4_BEST_READOUT_BY_CHANNEL.csv', best_rows)
    print(f'B4 done: {len(rows)} rows, {time.perf_counter() - started:.1f}s', flush=True)
    return rows


# ---------------------------------------------------------------- B1 delay vs false alarm
def first_crossing_sweep(d, locators, out, grid=200):
    pb = d.pb; steps = np.diff(d.off); S = int(d.off[-1])
    tok = np.load(d.c['paths']['tokens']); spans = tok['step_spans']
    strata = strata_masks(d)
    scopes = {'macro8_all': pb}
    for name, m in strata.items():
        scopes[name] = pb & m
    for cell in sorted(set(d.cells[pb])):
        scopes['cell=' + cell] = d.cells == cell
    idx = np.arange(S); BIG = S + 1
    q = np.linspace(.005, .995, grid)
    rows = []; summary = []
    started = time.perf_counter()
    for n, (name, s) in enumerate(locators.items()):
        variant = 'causal' if name.endswith('_causal') else 'offline'
        z = answer_z(s, d.off) if variant == 'offline' else np.asarray(s, float)
        thetas = np.unique(np.quantile(z[np.repeat(pb, steps)], q))
        stop = np.full((len(thetas), d.n), -1, int)
        for c in range(0, len(thetas), 50):
            th = thetas[c:c + 50]
            hit = z[None, :] >= th[:, None]
            posm = np.where(hit, idx[None, :], BIG)
            first = np.minimum.reduceat(posm, d.off[:-1], axis=1)
            local = first - d.off[:-1][None, :]
            stop[c:c + 50] = np.where(first >= BIG, -1, local)
        argmax = d.peaks(z)
        for scope, smask in scopes.items():
            clean = smask & (d.target < 0); err = smask & (d.target >= 0)
            if err.sum() < 20:
                continue
            tstart = np.array([spans[d.off[i] + d.target[i], 0] if d.target[i] >= 0 else -1 for i in np.flatnonzero(smask)])
            e_idx = np.flatnonzero(err)
            ref_delta = argmax[e_idx] - d.target[e_idx]
            base = {'locator': name, 'variant': variant, 'scope': scope, 'n_clean': int(clean.sum()), 'n_erroneous': int(err.sum()),
                    'argmax_sla': float((ref_delta == 0).mean()), 'argmax_early': float((ref_delta < 0).mean()), 'argmax_late': float((ref_delta > 0).mean())}
            curve = []
            for t, theta in enumerate(thetas):
                st = stop[t]
                fa = float((st[clean] >= 0).mean()) if clean.any() else np.nan
                se = st[e_idx]; tg = d.target[e_idx]
                fired = se >= 0; early = fired & (se < tg); detect = fired & (se >= tg); exact = se == tg
                dl = (se - tg)[detect]
                tok_delay = np.array([spans[d.off[i] + s_, 0] - spans[d.off[i] + t_, 0] for i, s_, t_ in zip(e_idx[detect], se[detect], tg[detect])])
                curve.append({**base, 'theta_quantile': float(q[min(t, len(q) - 1)]), 'theta': float(theta), 'false_alarm': fa,
                              'early_stop': float(early.mean()), 'detected': float(detect.mean()), 'exact': float(exact.mean()),
                              'no_stop': float((~fired).mean()), 'mean_delay_steps': float(dl.mean()) if detect.any() else np.nan,
                              'median_delay_steps': float(np.median(dl)) if detect.any() else np.nan,
                              'mean_delay_tokens': float(tok_delay.mean()) if detect.any() else np.nan})
            rows.extend(curve)
            fas = np.array([r['false_alarm'] for r in curve])
            for target_fa in FA_TARGETS:
                ok = np.flatnonzero(fas <= target_fa)
                if not len(ok):
                    continue
                r = curve[ok[0]]  # lowest threshold whose false-alarm rate is within budget
                summary.append({**base, 'false_alarm_budget': target_fa, 'false_alarm': r['false_alarm'], 'theta': r['theta'],
                                'detected': r['detected'], 'exact': r['exact'], 'early_stop': r['early_stop'], 'no_stop': r['no_stop'],
                                'mean_delay_steps': r['mean_delay_steps'], 'median_delay_steps': r['median_delay_steps'],
                                'mean_delay_tokens': r['mean_delay_tokens']})
        if n % 10 == 0:
            print(f'B1 {n}/{len(locators)} locators, {time.perf_counter() - started:.1f}s', flush=True)
    csv_write(out / 'B1_DELAY_CURVES.csv', rows)
    csv_write(out / 'B1_SUMMARY.csv', summary)
    # Figure: false alarm vs mean delay per stratum; four fixed series with legend and end labels.
    series = [('ct7', PALETTE[0]), ('token_lsml', PALETTE[1]), ('q15_H1__top5', PALETTE[2]), ('q15_H1__page_wmax', PALETTE[3])]
    for metric, ylabel, fname in [('mean_delay_steps', 'mean delay among detected (steps)', 'B1_DELAY_VS_FALSE_ALARM.png'),
                                  ('detected', 'detected fraction (stop at or after the first error)', 'B1_DETECTION_VS_FALSE_ALARM.png')]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE)
        for ax, (stratum, _, _) in zip(axes, STRATA):
            for name, col in series:
                pts = [(r['false_alarm'], r[metric]) for r in rows if r['locator'] == name and r['scope'] == stratum and np.isfinite(r[metric])]
                if not pts:
                    continue
                pts.sort()
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=col, linewidth=2, label=name)
                ax.text(xs[-1], ys[-1], ' ' + name, fontsize=7, color=INK2, va='center')
            style(ax, stratum, 'false-alarm rate on clean answers', ylabel)
            ax.set_xlim(0, 1)
        axes[0].legend(fontsize=7.5, frameon=False)
        fig.suptitle('First-crossing rule on answer-standardised step scores (offline)', color=INK, fontsize=11, x=.01, ha='left')
        fig.tight_layout(); fig.savefig(out / fname, dpi=160, facecolor=SURFACE); plt.close(fig)
    print(f'B1 done: {len(rows)} curve rows, {time.perf_counter() - started:.1f}s', flush=True)
    return rows, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT / 'configs/readout_family_v1.json'))
    p.add_argument('--out', default=str(ROOT / 'results/quickest_detection_diagnostics_v1'))
    p.add_argument('--only', choices=['b1', 'b2', 'b4', 'all'], default='all')
    args = p.parse_args()
    c = config(args.config); d = Dataset(c)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if not (d.out / 'PROFILES_EXT_COMPLETE.json').exists():
        raise SystemExit('run readout_family_v1.py --stage prepare first')
    p_full = np.load(d.out / d.profile_file, mmap_mode='r')
    started = time.perf_counter()
    manifest = {'config': c, 'profiles_full_sha256': digest(d.out / d.profile_file), 'tokens_sha256': digest(c['paths']['tokens']),
                'ct7_sha256': digest(c['paths']['ct7']), 'joined_sha256': digest(c['paths']['joined']),
                'labels_used_for': 'evaluation only; no fitting, selection or threshold choice', 'alpha': ALPHA,
                'strata': [s[0] for s in STRATA], 'false_alarm_budgets': FA_TARGETS, 'page_k_sensitivity': B1_PAGE_KS}
    if args.only in ['b2', 'all']:
        markov_persistence(d, out)
    if args.only in ['b4', 'b1', 'all']:
        b4, b1 = build_locators(d, p_full, out)
        manifest['b4_locators'] = len(b4); manifest['b1_locators'] = len(b1)
        if args.only in ['b4', 'all']:
            delay_floor(d, b4, out)
        if args.only in ['b1', 'all']:
            first_crossing_sweep(d, b1, out)
    manifest['seconds'] = time.perf_counter() - started
    manifest['outputs'] = {f.name: {'bytes': f.stat().st_size, 'sha256': digest(f)} for f in out.iterdir() if f.is_file() and f.name != 'MANIFEST.json'}
    dump(out / 'MANIFEST.json', manifest)
    print(f'Diagnostics complete in {manifest["seconds"]:.0f}s -> {out}', flush=True)


if __name__ == '__main__':
    main()
