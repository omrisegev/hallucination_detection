"""GAP success vs failure plots, PRMBench, PRM boosting CHAIN_N16 (Qwen2.5-Math-PRM-7B). Development data.

success = GAP's flagged set equals the answer's true error-step set exactly (non-control answers only).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/Users/osegev/Desktop/hallucination_detection')
sys.path.insert(0, str(ROOT / 'results/error_count_label_free_v1'))
import common_ec as C  # noqa: E402

OUT = Path(__file__).resolve().parent
BOOST = Path('/private/tmp/claude-501/-Users-osegev-Desktop-hallucination-detection/98d49308-5ca3-48f7-a452-63da0bb7d2a1/scratchpad/error_count/BOOST_SCORES.npz')
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
GAPGRAY = '#c9c8c1'

plt.rcParams.update({'font.size': 10.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'axes.titleweight': 'bold'})

bank = C.load_tele_bank('N16'); off = bank['offsets'].astype(np.int64); ids = bank['idx']
meta = C.load_meta()
with np.load(BOOST) as z:
    score = z['CHAIN_N16'].astype(float)
assert len(score) == off[-1]
Z = C.answer_z(score, off)
y, cnt, nonc, cls, cat = C.labels(ids, off, meta)
n = np.diff(off)


def gap_count(z):
    m = len(z)
    if m == 1:
        return 1
    s = np.sort(z)[::-1]; kmax = min(int(np.ceil(m / 2)), m - 1)
    return int(np.argmax(s[:kmax] - s[1:kmax + 1]) + 1)


k = np.array([gap_count(Z[a:b]) for a, b in zip(off[:-1], off[1:])])
f_gap = C.topk_flags(Z, off, k)
step_nonc = np.repeat(nonc, n)
ps_q80 = C.prmscore(y, C.q_flags(Z, off, already_z=True), step_nonc)
ps_gap = C.prmscore(y, f_gap, step_nonc)
print(f'parity: Q80 {ps_q80:.2f} (expect 68.25), GAP {ps_gap:.2f} (expect 70.40)')
assert abs(ps_q80 - 68.25) < .01 and abs(ps_gap - 70.40) < .01

ans = np.arange(len(n))
succ = np.array([nonc[j] and np.array_equal(f_gap[a:b], y[a:b]) for j, (a, b) in enumerate(zip(off[:-1], off[1:]))])
fail = nonc & ~succ
print(f'non-control answers {nonc.sum()}, successes {succ.sum()} ({succ.sum() / nonc.sum():.1%}), failures {fail.sum()}')

# ---------- Figure 1: score histograms by true step label, successes vs failures ----------
bins = np.linspace(-3.5, 3.5, 57)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, mask, title in ((axes[0], succ, 'Successful answers'), (axes[1], fail, 'Failed answers')):
    sm = np.repeat(mask, n)
    ok, err = Z[sm & ~y], Z[sm & y]
    ax.hist(ok, bins=bins, density=True, histtype='stepfilled', color=BLUE, alpha=.22, linewidth=0)
    ax.hist(ok, bins=bins, density=True, histtype='step', color=BLUE, linewidth=2, label=f'correct steps (n={len(ok):,})')
    ax.hist(err, bins=bins, density=True, histtype='stepfilled', color=ORANGE, alpha=.22, linewidth=0)
    ax.hist(err, bins=bins, density=True, histtype='step', color=ORANGE, linewidth=2, label=f'error steps (n={len(err):,})')
    ax.set_title(f'{title}  ({mask.sum():,} answers)', loc='left', fontsize=12)
    ax.set_xlabel('step score within its answer (answer-z, higher = riskier)')
    ax.legend(loc='upper left')
axes[0].set_ylabel('share of steps (density)')
fig.suptitle('PRMBench, CHAIN_N16 on Qwen2.5-Math-PRM-7B: where error and correct steps score\n'
             'success = GAP flagged exactly the true error steps', x=.01, ha='left', fontsize=11, color=INK2)
fig.tight_layout(rect=(0, 0, 1, .9))
fig.savefig(OUT / 'fig1_score_histograms.png', dpi=150)

# ---------- Figure 2: one success and one failure answer, sorted scores and gaps ----------
def gaps_of(j):
    a, b = off[j], off[j + 1]
    order = np.argsort(-Z[a:b], kind='stable')
    s = Z[a:b][order]; lab = y[a:b][order]
    return s, lab, s[:-1] - s[1:]


nm = 11  # the median step count of non-control answers is reported below
print('median n (non-control):', int(np.median(n[nonc])))
G1 = np.array([gaps_of(j)[2][:min(int(np.ceil(n[j] / 2)), n[j] - 1)].max() if n[j] > 1 else np.nan for j in ans])


def pick(mask):
    cand = np.flatnonzero(mask & (n == nm) & (cnt == 2))
    med = np.median(G1[mask & (n == nm)])
    return int(cand[np.argmin(np.abs(G1[cand] - med))]), med


js, med_s = pick(succ); jf, med_f = pick(fail)
fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), gridspec_kw={'width_ratios': [1, 1.1]})
for row, (j, lab_txt, med) in enumerate(((js, 'Success', med_s), (jf, 'Failure', med_f))):
    s, lab, g = gaps_of(j)
    K = min(int(np.ceil(n[j] / 2)), n[j] - 1); kstar = int(np.argmax(g[:K]) + 1)
    r = np.arange(1, len(s) + 1)
    ax = axes[row, 0]
    ax.plot(r, s, color=GAPGRAY, linewidth=1.5, zorder=1)
    ax.scatter(r[~lab], s[~lab], s=70, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3, label='correct step')
    ax.scatter(r[lab], s[lab], s=70, color=ORANGE, edgecolor=SURF, linewidth=2, zorder=3, label='error step')
    ax.axvline(kstar + .5, color=INK, linewidth=1.5, linestyle='-', zorder=2)
    ax.text(kstar + .6, s.max(), f' GAP cut: flag top {kstar}', color=INK, va='top', fontsize=9.5)
    ax.annotate('', xy=(kstar + .5, s[kstar]), xytext=(kstar + .5, s[kstar - 1]),
                arrowprops=dict(arrowstyle='<->', color=INK, linewidth=1.2))
    ax.set_xticks(r); ax.set_xlabel('steps sorted by score (rank 1 = highest)')
    ax.set_ylabel('answer-z score')
    ax.set_title(f'{lab_txt}: answer with {n[j]} steps, {cnt[j]} true errors', loc='left', fontsize=11.5)
    if row == 0: ax.legend(loc='upper right')
    ax = axes[row, 1]
    x = np.arange(1, len(g) + 1)
    colors = [INK if i + 1 == kstar else (MUTED if i < K else GAPGRAY) for i in range(len(g))]
    ax.bar(x, g, width=.7, color=colors, zorder=3)
    ax.axvspan(.5, K + .5, color='#f1f0ea', zorder=0)
    ax.text(K + .45, g.max() * 1.1, f'search window (k <= {K})', color=INK2, fontsize=9, va='top', ha='right')
    ax.axvline(cnt[j] + .5, color=ORANGE, linestyle='--', linewidth=1.5, zorder=4)
    ax.text(cnt[j] + .55, g.max() * .72, 'true\ncount', color=ORANGE, fontsize=9)
    others = np.delete(g[:K], kstar - 1)
    ax.set_title(f'Gaps between neighbours: chosen {g[kstar - 1]:.2f}, other window gaps median {np.median(others):.2f}',
                 loc='left', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f'{i}|{i + 1}' for i in x], fontsize=8.5)
    ax.set_xlabel('gap between sorted steps (rank i | i+1)'); ax.set_ylabel('gap size (answer-z)')
    ax.set_ylim(0, g.max() * 1.15)
fig.suptitle(f'Two example answers ({nm} steps, 2 true errors), picked as the {nm}-step answer whose chosen gap is closest to\n'
             f'the median chosen gap of its group (success {med_s:.2f}, failure {med_f:.2f}). Dark bar = gap GAP chose.',
             x=.01, ha='left', fontsize=10.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, .92))
fig.savefig(OUT / 'fig2_example_answers.png', dpi=150)
print('examples:', ids[js], ids[jf])

# ---------- Figure 3: all answers, chosen gap vs the other gaps in the search window ----------
chosen = {True: [], False: []}; other = {True: [], False: []}; ratio = {True: [], False: []}
for j in ans:
    if not nonc[j] or n[j] < 3:
        continue
    _, _, g = gaps_of(j); K = min(int(np.ceil(n[j] / 2)), n[j] - 1)
    w = g[:K]; ks = int(np.argmax(w)); key = bool(succ[j])
    chosen[key].append(w[ks]); rest = np.delete(w, ks); other[key].extend(rest.tolist())
    if len(rest): ratio[key].append(w[ks] / max(np.median(rest), 1e-6))
gb = np.linspace(0, 3.2, 49)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, key, title in ((axes[0], True, 'Successful answers'), (axes[1], False, 'Failed answers')):
    c, o = np.array(chosen[key]), np.array(other[key])
    ax.hist(o, bins=gb, density=True, histtype='stepfilled', color=GAPGRAY, alpha=.7, linewidth=0,
            label=f'other gaps in the window (median {np.median(o):.2f})')
    ax.hist(c, bins=gb, density=True, histtype='step', color=INK, linewidth=2,
            label=f'gap GAP chose (median {np.median(c):.2f})')
    ax.set_title(f'{title}  (median chosen / median other = {np.median(ratio[key]):.1f}x)', loc='left', fontsize=11.5)
    ax.set_xlabel('gap size (answer-z)'); ax.legend(loc='upper right')
axes[0].set_ylabel('share of gaps (density)')
fig.suptitle('All non-control PRMBench answers with >= 3 steps, CHAIN_N16: the chosen gap vs the gaps it was chosen over',
             x=.01, ha='left', fontsize=11, color=INK2)
fig.tight_layout(rect=(0, 0, 1, .92))
fig.savefig(OUT / 'fig3_chosen_vs_other_gaps.png', dpi=150)
for key in (True, False):
    print('success' if key else 'failure', 'median chosen', round(float(np.median(chosen[key])), 3),
          'median other', round(float(np.median(other[key])), 3), 'median ratio', round(float(np.median(ratio[key])), 2))

# ---------- failure types: wrong count vs wrong ranking ----------
f_or = C.topk_flags(Z, off, cnt)
rank_ok = np.array([np.array_equal(f_or[a:b], y[a:b]) for a, b in zip(off[:-1], off[1:])])
fz = fail & (cnt > 0)
print('failures with >=1 error:', fz.sum(), '| ranking perfect (top true-count = true set), only the count is wrong:',
      int((fz & rank_ok).sum()), f'({(fz & rank_ok).sum() / fz.sum():.1%})', '| ranking imperfect:', int((fz & ~rank_ok).sum()))
print('zero-error failures (GAP always flags >=1):', int((fail & (cnt == 0)).sum()))
print('ORACLE exact-set success share among non-control:', f'{(nonc & rank_ok).sum() / nonc.sum():.1%}')
