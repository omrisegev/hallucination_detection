"""Stage-2 analyses and the stage-1 vs stage-2 comparison, with charts (pre-registration
docs/experiments/PRM_VIEW_SELECTION_STAGE2_PREREG_2026-10-26.md).

    python analyze.py summary     # ANALYSIS.json, SUBSETS_NAMED.csv, CHANNEL_FREQ.csv, COMPARISON.json
    python analyze.py exact       # exact re-check of every named subset (both runs) -> NAMED_EXACT.csv
    python analyze.py figures     # figures/*.png
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '4')
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'stage1'))
import stage2 as S2  # noqa: E402
import stage1 as S1  # noqa: E402
E = S1.E
NAMES = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation',
         'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0', 'realized_z', 'realized_drv', 'digit_alternative',
         'digit_spread', 'digit_alternative_innovation']
PB, SO, MS = list(range(5)), list(range(5, 10)), (4, 9)
C8 = [j for j in range(10) if j not in MS]
SHORT = ['PB Qwen-7B', 'PB Qwen-72B', 'PB PURE', 'PB RLHFlow', 'PB Math-Sh.', 'SO Qwen-7B', 'SO Qwen-72B', 'SO PURE',
         'SO RLHFlow', 'SO Math-Sh.']


def names_of(mask): return [NAMES[i] for i in S1.subset_of(int(mask))]
def label(mask): return ' + '.join(names_of(mask))


def load():
    D1, D2 = dict(np.load(HERE.parent / 'stage1' / 'SEARCH.npz')), dict(np.load(HERE / 'SEARCH.npz'))
    assert np.array_equal(D1['masks'], D2['masks'])
    par = json.loads((HERE.parent / 'stage1' / 'PARITY.json').read_text())
    cells = [str(x) for x in D1['cells']]; prm = np.array([par[c]['prm_exact'] for c in cells])
    V1, V2 = D1['valid'].all(1), D2['valid'].all(1)
    mg1 = D1['ps_fused'] - np.maximum(prm[None, :], D1['ps_mean'])          # stage-1 goal
    mg2 = D2['ps_fused'] - D2['ps_mean']                                      # stage-2 primary
    mg2s = D2['ps_fused'] - np.maximum(prm[None, :], D2['ps_mean'])          # stage-2 secondary
    return D1, D2, cells, prm, V1, V2, mg1, mg2, mg2s


def sel(mg, valid, cols):
    m = np.where(valid, mg[:, cols].min(1), -np.inf); i = int(np.argmax(m)); return i, float(m[i])


def top1(mg, valid):
    m = np.where(valid, mg.min(1), -np.inf); q = np.quantile(m[valid], .99); return valid & (m >= q)


def summary():
    D1, D2, cells, prm, V1, V2, mg1, mg2, mg2s = load()
    size = np.array([bin(int(m)).count('1') for m in D2['masks']])
    succ2 = V2 & (mg2 > 0).all(1); succ2s = V2 & (mg2s > 0).all(1)
    out = dict(n_valid_all=int(V2.sum()), n_success_primary=int(succ2.sum()), n_success_secondary=int(succ2s.sum()),
               n_success_primary_binary=int((V2 & ((D2['ps_bin'] - D2['ps_mean']) > 0).all(1)).sum()),
               n_primary_prmbench5=int((V2 & (mg2[:, PB] > 0).all(1)).sum()), n_primary_socratic5=int((V2 & (mg2[:, SO] > 0).all(1)).sum()),
               n_primary_8cells=int((D2['valid'][:, C8].all(1) & (mg2[:, C8] > 0).all(1)).sum()),
               n_beat_prm_any_socratic_strong=int((V2 & (D2['ps_fused'][:, [5, 6, 7]] > prm[[5, 6, 7]][None, :]).any(1)).sum()),
               cells_won_hist={int(k): int((V2 & ((mg2 > 0).sum(1) == k)).sum()) for k in range(11)},
               per_cell={c: dict(prm=float(prm[j]), beat_mean=float((mg2[V2, j] > 0).mean()), beat_prm=float((D2['ps_fused'][V2, j] > prm[j]).mean()),
                                 best_fused=float(D2['ps_fused'][V2, j].max()), best_margin=float(mg2[V2, j].max())) for j, c in enumerate(cells)},
               by_size={int(k): dict(valid=int((V2 & (size == k)).sum()), success=int((succ2 & (size == k)).sum())) for k in np.unique(size)},
               min_margin_quantiles={str(q): float(np.quantile(mg2[V2].min(1), q)) for q in (0, .5, .9, .99, 1)},
               clip_between_rate=float((D2['clip_between'][V2] > 0).mean()))
    named = {'S2 best (primary, all 10)': sel(mg2, V2, list(range(10))), 'S2 best (secondary, all 10)': sel(mg2s, V2, list(range(10))),
             'S2 selected on PRMBench': sel(mg2, V2, PB), 'S2 selected on Socratic': sel(mg2, V2, SO)}
    for p in range(5):
        named[f'S2 LOPO {S1.PRMS[p]}'] = sel(mg2, V2, [j for j in range(10) if j not in (p, p + 5)])
    out['cross_benchmark'] = {'PB->SO': [float(mg2[named['S2 selected on PRMBench'][0], j]) for j in SO],
                              'SO->PB': [float(mg2[named['S2 selected on Socratic'][0], j]) for j in PB]}
    out['lopo_heldout'] = {S1.PRMS[p]: [float(mg2[named[f'S2 LOPO {S1.PRMS[p]}'][0], j]) for j in (p, p + 5)] for p in range(5)}
    # ---- comparison with stage 1
    both = V1 & V2
    m1, m2 = mg1.min(1), mg2.min(1)
    t1, t2 = top1(mg1, V1), top1(mg2, V2)
    joint = np.where(both, np.minimum(m1, m2), -np.inf); ij = int(np.argmax(joint))
    joint8 = np.where(D1['valid'][:, C8].all(1) & D2['valid'][:, C8].all(1), np.minimum(mg1[:, C8].min(1), mg2[:, C8].min(1)), -np.inf)
    ij8 = int(np.argmax(joint8))
    named['JOINT (max of min over 20 margins)'] = (ij, float(joint[ij]))
    named['JOINT 8 cells (no Math-Shepherd)'] = (ij8, float(joint8[ij8]))
    named['S1 best (stage-1 goal, all 10)'] = sel(mg1, V1, list(range(10)))
    named['S1 best 8 cells'] = sel(mg1, D1['valid'][:, C8].all(1), C8)
    comp = dict(n_valid_both=int(both.sum()),
                spearman_minmargin=float(spearmanr(m1[both], m2[both])[0]),
                spearman_mean_margin=float(spearmanr(mg1[both].mean(1), mg2[both].mean(1))[0]),
                top1_sizes=[int(t1.sum()), int(t2.sum())], top1_overlap=int((t1 & t2).sum()),
                top1_jaccard=float((t1 & t2).sum() / max((t1 | t2).sum(), 1)),
                top1_both=[label(D1['masks'][i]) for i in np.flatnonzero(t1 & t2)],
                n_both_goals_10=int((both & (mg1 > 0).all(1) & (mg2 > 0).all(1)).sum()),
                n_both_goals_8=int((D1['valid'][:, C8].all(1) & D2['valid'][:, C8].all(1) & (mg1[:, C8] > 0).all(1) & (mg2[:, C8] > 0).all(1)).sum()),
                joint=dict(subset=label(D1['masks'][ij]), min_of_20=float(joint[ij]), stage1=mg1[ij].tolist(), stage2=mg2[ij].tolist()),
                joint8=dict(subset=label(D1['masks'][ij8]), min_of_16=float(joint8[ij8]), stage1=mg1[ij8].tolist(), stage2=mg2[ij8].tolist()))
    bits = ((D2['masks'][:, None] >> np.arange(16)) & 1).astype(bool)
    ref2 = succ2 if succ2.sum() else t2                     # pre-registered: successes, else the top 1%; stage 1 has no success
    E.write_csv(HERE / 'CHANNEL_FREQ.csv', [dict(channel=NAMES[k], base_valid_s1=float(bits[V1, k].mean()), top1_s1=float(bits[t1, k].mean()),
                                                base_valid_s2=float(bits[V2, k].mean()), top1_s2=float(bits[t2, k].mean()),
                                                success_s2=float(bits[ref2, k].mean())) for k in range(16)])
    rows = []
    for nm, (i, v) in named.items():
        r = dict(selection=nm, subset=label(D2['masks'][i]), size=int(size[i]), select_value=v,
                 s1_cells_won=int((mg1[i] > 0).sum()) if V1[i] else None, s2_cells_won=int((mg2[i] > 0).sum()) if V2[i] else None)
        for j, c in enumerate(cells):
            r[f's1_margin {c}'] = float(mg1[i, j]); r[f's2_margin {c}'] = float(mg2[i, j]); r[f's2_vs_prm {c}'] = float(D2['ps_fused'][i, j] - prm[j])
        rows.append(r)
    E.write_csv(HERE / 'SUBSETS_NAMED.csv', rows)
    out['named'] = {r['selection']: dict(subset=r['subset'], s1_cells_won=r['s1_cells_won'], s2_cells_won=r['s2_cells_won']) for r in rows}
    json.dump(out, open(HERE / 'ANALYSIS.json', 'w'), indent=1, default=float)
    json.dump(comp, open(HERE / 'COMPARISON.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('per_cell', 'by_size')}, indent=1, default=float))
    print(json.dumps({k: v for k, v in comp.items() if k != 'top1_both'}, indent=1, default=float))


def exact():
    """Every named subset, recomputed through the evaluator's loops in both runs, plus TELE_N16 / MEAN_N16 context."""
    rows = E.read_csv(HERE / 'SUBSETS_NAMED.csv')
    subsets = list(dict.fromkeys(r['subset'] for r in rows))
    masks = {s: sum(1 << NAMES.index(n) for n in s.split(' + ')) for s in subsets}
    D1, D2, cells, prm, *_ = load()
    pos = {int(m): i for i, m in enumerate(D2['masks'])}
    out = {s: dict(subset=s) for s in subsets}
    for j, (b, p) in enumerate(S1.CELLS):
        t = S2.TeleCell().build(b, p); c = t.c; s1 = S1.CellData(b, p)
        def ex(cobj, sc, _S):
            z = E.answer_z(sc, cobj.off); f = E.topk_flags(z, cobj.off, E.gap_counts(z, cobj.off))
            if b == 'prmbench':
                return float(E.official_from_counts(E.counts_by_answer(cobj, f).sum(0), 'prmbench'))
            return 100 * float(E.official(cobj, f)[0])
        tele = E.measure(c, 'TELE_N16', 'GAP', fast=True)[0]['official'] * (1 if b == 'prmbench' else 100)
        mean16 = E.measure(c, 'MEAN_N16', 'GAP', fast=True)[0]['official'] * (1 if b == 'prmbench' else 100)
        for s in subsets:
            S = S1.subset_of(masks[s]); i = pos[masks[s]]; r = out[s]
            o2 = t.evaluate(S, want_detail=True)
            if o2['valid']:
                r[f's2_fused {b}/{p}'] = ex(c, o2['fused'], S); r[f's2_mean {b}/{p}'] = ex(c, o2['mean'], S)
                r[f's2_ok {b}/{p}'] = abs(r[f's2_fused {b}/{p}'] - D2['ps_fused'][i, j]) < 1e-9 and abs(r[f's2_mean {b}/{p}'] - D2['ps_mean'][i, j]) < 1e-9
            o1 = s1.evaluate(S, want_detail=True)
            if o1['valid']:
                r[f's1_fused {b}/{p}'] = ex(s1.c, o1['fused'], S); r[f's1_ok {b}/{p}'] = abs(r[f's1_fused {b}/{p}'] - D1['ps_fused'][i, j]) < 1e-9
            r[f'tele_n16 {b}/{p}'] = tele; r[f'mean_n16 {b}/{p}'] = mean16; r[f'prm {b}/{p}'] = float(prm[j])
        print(b, p, 'done', flush=True); del t, s1
    E.write_csv(HERE / 'NAMED_EXACT.csv', list(out.values()))
    oks = [v for r in out.values() for k, v in r.items() if k.startswith(('s1_ok', 's2_ok'))]
    print('exact re-check:', sum(oks), 'of', len(oks), 'match')


# ----------------------------------------------------------------------------------------------- figures
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'      # categorical slots 1-3 (validated all-pairs in the reference palette)


def _ax(ax, title=None):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9); ax.yaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
    if title: ax.set_title(title, loc='left', color=INK, fontsize=11.5, pad=10)


def figures():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'savefig.facecolor': SURF, 'text.color': INK,
                         'axes.labelcolor': INK2})
    F = HERE / 'figures'; F.mkdir(exist_ok=True)
    D1, D2, cells, prm, V1, V2, mg1, mg2, mg2s = load()
    comp = json.loads((HERE / 'COMPARISON.json').read_text())
    # 1. cells-won histogram
    fig, ax = plt.subplots(figsize=(8.5, 4.2)); k = np.arange(11); w = .38
    h1 = np.array([((mg1[V1] > 0).sum(1) == x).sum() for x in k]) / V1.sum() * 100
    h2 = np.array([((mg2[V2] > 0).sum(1) == x).sum() for x in k]) / V2.sum() * 100
    ax.bar(k - w / 2 - .01, h1, w, color=C1, label=f'Stage 1: with PRM (beat PRM and same-set mean), n={V1.sum():,}')
    ax.bar(k + w / 2 + .01, h2, w, color=C2, label=f'Stage 2: telemetry only (beat same-set mean), n={V2.sum():,}')
    _ax(ax, 'How many of the 10 cells does a subset win?'); ax.set_xticks(k); ax.set_xlabel('cells won (of 10)'); ax.set_ylabel('% of valid subsets')
    ax.legend(frameon=False, fontsize=9, loc='upper left'); fig.tight_layout(); fig.savefig(F / '1_cells_won.png', dpi=160); plt.close(fig)
    # 2. per-cell win rate
    fig, ax = plt.subplots(figsize=(10, 4.4)); x = np.arange(10)
    r1 = (mg1[V1] > 0).mean(0) * 100; r2 = (mg2[V2] > 0).mean(0) * 100
    ax.bar(x - w / 2 - .01, r1, w, color=C1, label='Stage 1: with PRM (vs max(PRM, mean))')
    ax.bar(x + w / 2 + .01, r2, w, color=C2, label='Stage 2: telemetry only (vs same-set mean)')
    for xi, (a, b) in enumerate(zip(r1, r2)):
        ax.text(xi - w / 2, a + 1, f'{a:.0f}', ha='center', fontsize=8, color=INK2); ax.text(xi + w / 2, b + 1, f'{b:.0f}', ha='center', fontsize=8, color=INK2)
    ax.axvline(4.5, color=GRID, lw=1.2); ax.text(2, 104, 'PRMBench', ha='center', color=INK2, fontsize=9); ax.text(7, 104, 'Socratic', ha='center', color=INK2, fontsize=9)
    _ax(ax, 'Share of valid subsets that win each cell'); ax.set_xticks(x); ax.set_xticklabels([s[3:] for s in SHORT], fontsize=8.5)
    ax.set_ylim(0, 110); ax.set_ylabel('% of valid subsets'); ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(.5, -.1), ncol=2)
    fig.tight_layout(); fig.savefig(F / '2_per_cell_win_rate.png', dpi=160); plt.close(fig)
    # 3. scatter of min-margins
    both = V1 & V2; m1, m2 = mg1.min(1), mg2.min(1)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.scatter(m1[both], m2[both], s=6, color='#9a998f', alpha=.35, lw=0, label=f'subsets valid in both runs (n={both.sum():,})')
    t1, t2 = top1(mg1, V1) & both, top1(mg2, V2) & both
    ax.scatter(m1[t1], m2[t1], s=16, color=C1, lw=0, label='top 1% in stage 1')
    ax.scatter(m1[t2], m2[t2], s=16, color=C2, lw=0, label='top 1% in stage 2')
    ij = int(np.flatnonzero(D1['masks'] == sum(1 << NAMES.index(n) for n in comp['joint']['subset'].split(' + ')))[0])
    ax.scatter([m1[ij]], [m2[ij]], s=120, facecolor='none', edgecolor=C3, lw=2.2, label='joint best (max of min over 20)')
    ax.axhline(0, color=INK2, lw=.8, ls='--'); ax.axvline(0, color=INK2, lw=.8, ls='--')
    _ax(ax, f'Worst-cell margin, stage 1 vs stage 2 (Spearman {comp["spearman_minmargin"]:+.2f})')
    ax.set_xlabel('stage 1: min over 10 cells of fused − max(PRM, mean)'); ax.set_ylabel('stage 2: min over 10 cells of fused − mean')
    ax.text(.98, .98, 'success in both\nwould be here', transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=8.5, loc='lower left'); fig.tight_layout(); fig.savefig(F / '3_minmargin_scatter.png', dpi=160); plt.close(fig)
    # 4. channel frequency in the top 1%
    fr = E.read_csv(HERE / 'CHANNEL_FREQ.csv'); order = np.argsort([-(float(r['top1_s1']) + float(r['success_s2'])) for r in fr])
    fig, ax = plt.subplots(figsize=(10, 4.6)); x = np.arange(16)
    ax.bar(x - w / 2 - .01, [float(fr[i]['top1_s1']) * 100 for i in order], w, color=C1, label='stage 1 (with PRM): top 1% by worst-cell margin (no success exists)')
    ax.bar(x + w / 2 + .01, [float(fr[i]['success_s2']) * 100 for i in order], w, color=C2, label='stage 2 (telemetry only): the 380 successes')
    ax.scatter(x - w / 2, [float(fr[i]['base_valid_s1']) * 100 for i in order], marker='_', s=110, color=INK, lw=2, label='base rate among valid subsets of that run', zorder=3)
    ax.scatter(x + w / 2, [float(fr[i]['base_valid_s2']) * 100 for i in order], marker='_', s=110, color=INK, lw=2, zorder=3)
    _ax(ax, 'Which channels appear in the best subsets?'); ax.set_xticks(x); ax.set_xticklabels([fr[i]['channel'] for i in order], rotation=40, ha='right', fontsize=8.5)
    ax.set_ylabel('% of subsets containing the channel'); ax.set_ylim(0, 105); ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    fig.tight_layout(); fig.savefig(F / '4_channel_frequency.png', dpi=160); plt.close(fig)
    # 5. per-cell margins of the named subsets, small multiples (one panel per subset)
    rows = {r['selection']: r for r in E.read_csv(HERE / 'SUBSETS_NAMED.csv')}
    show = ['JOINT (max of min over 20 margins)', 'S1 best (stage-1 goal, all 10)', 'S2 best (primary, all 10)', 'JOINT 8 cells (no Math-Shepherd)']
    fig, axs = plt.subplots(len(show), 1, figsize=(10, 2.5 * len(show)), sharex=True); x = np.arange(10)
    for ax, nm in zip(axs, show):
        r = rows[nm]; a = np.array([float(r[f's1_margin {c}']) for c in cells]); b = np.array([float(r[f's2_margin {c}']) for c in cells])
        ax.bar(x - w / 2 - .01, np.clip(a, -6, None), w, color=C1, label='stage 1 margin (vs max(PRM, mean))')
        ax.bar(x + w / 2 + .01, np.clip(b, -6, None), w, color=C2, label='stage 2 margin (vs same-set mean)')
        for xi, v in enumerate(a):
            if v < -6: ax.text(xi - w / 2, -5.6, f'{v:.0f}', ha='center', fontsize=7.5, color='white')
        for xi, v in enumerate(b):
            if v < -6: ax.text(xi + w / 2, -5.6, f'{v:.0f}', ha='center', fontsize=7.5, color='white')
        ax.axhline(0, color=INK, lw=.8); ax.axvline(4.5, color=GRID, lw=1.2)
        _ax(ax); ax.set_title(f'{nm}\n{r["subset"]}', loc='left', fontsize=9.5, color=INK); ax.set_ylabel('PRMScore pts', fontsize=8.5)
    axs[-1].set_xticks(x); axs[-1].set_xticklabels(SHORT, fontsize=8.5)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=9, loc='upper center', ncol=2, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(F / '5_named_subset_margins.png', dpi=160); plt.close(fig)
    print('figures written to', F)



def figure_competitors():
    """6. Qwen-7B under each competitor's protocol: references, each subset +PRM and alone, and the paper's number."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': SURF, 'savefig.facecolor': SURF, 'text.color': INK, 'axes.labelcolor': INK2})
    R = [r for r in E.read_csv(HERE / 'COMPETITORS.csv') if r['prm'] == 'qwen_prm_7b']
    subs = json.loads((HERE / 'COMPETITORS_PARITY.json').read_text())['subsets']
    arms = ['PRM', 'CHAIN_N16', 'TELE_N16'] + [f'{s["id"]} +PRM' for s in subs] + [f'{s["id"]} alone' for s in subs]
    get = lambda proto, arm, col: next((float(r[col]) for r in R if r['protocol'] == proto and
                                        (r['arm'] == arm or (arm == 'PRM' and r['arm'] in ('PRM_native', 'PRM_Q80_PROB', 'PRM_Q80')))), np.nan)
    panels = [('PRISM', 'PRMScore_GAP', 'PRISM: PRMScore (GAP)', 68.0, 'PRISM 68.0', True),
              ('PRISM', 'matchedTPR_FPR', 'PRISM: FPR at TPR 87.3 (lower is better)', 47.13, 'PRISM 47.13', False),
              ('CROP', 'kept_alpha05', 'CROP: % of trace kept at risk 5%', 21.8, 'CROP Trace+PRM 21.8', True),
              ('ARES', 'macro_f1_oof', 'ARES protocol: Macro-F1 (CV threshold)', None, None, True),
              ('SLIDER', 'acc_balanced_threshold', 'SLIDER: redundancy accuracy', 61.9, 'Step-RRI 61.9', True)]
    col = lambda a: '#9a998f' if a in ('PRM', 'CHAIN_N16', 'TELE_N16') else (C1 if a.endswith('+PRM') else C2)
    fig, axs = plt.subplots(1, len(panels), figsize=(17, 6.6), sharey=True); y = np.arange(len(arms))[::-1]
    for ax, (proto, c, title, ref, rlab, hib) in zip(axs, panels):
        v = np.array([get(proto, a, c) for a in arms])
        lo = np.nanmin(np.r_[v, ref if ref else np.nan]) - 2; hi = np.nanmax(np.r_[v, ref if ref else np.nan]) + 1.5
        ax.barh(y, v - lo, left=lo, height=.72, color=[col(a) for a in arms])
        for yi, vi in zip(y, v): ax.text(vi + .15, yi, f'{vi:.1f}', va='center', fontsize=7.5, color=INK2)
        if ref is not None:
            ax.axvline(ref, color=INK, lw=1.2, ls='--')
            ax.text(ref, 1.005, rlab, fontsize=8, color=INK, ha='center', va='bottom', transform=ax.get_xaxis_transform())
        ax.set_xlim(lo, hi + 1.2); _ax(ax); ax.xaxis.grid(True, color=GRID, lw=.8); ax.yaxis.grid(False)
        ax.set_title(title, loc='left', fontsize=9.5, color=INK, pad=16)
    axs[0].set_yticks(y); axs[0].set_yticklabels(arms, fontsize=8.5)
    from matplotlib.patches import Patch
    fig.legend([Patch(color='#9a998f'), Patch(color=C1), Patch(color=C2)], ['reference (PRM, CHAIN_N16, TELE_N16)', 'subset fused WITH the PRM (stage 1)',
               'subset fused WITHOUT the PRM (stage 2)'], frameon=False, fontsize=9, loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.0))
    fig.text(.01, .01, 'Qwen2.5-Math-PRM-7B on PRMBench; subsets were selected with PRMBench labels, so these numbers are in-sample.\n' +
             '\n'.join(f'{s["id"]} = {s["subset"]}' for s in subs), fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, .14, 1, .94)); fig.savefig(HERE / 'figures' / '6_competitors_qwen7b.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    {'summary': summary, 'exact': exact, 'figures': figures, 'fig_competitors': figure_competitors}[sys.argv[1]]()
