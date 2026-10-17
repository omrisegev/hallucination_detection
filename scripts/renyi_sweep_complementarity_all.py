"""Complementarity across ALL sweep arms (post hoc, descriptive): every H_alpha and VE_alpha view.

For each arm: PB exact-hit rate on erroneous answers by relative first-error position (first / middle / last third)
and by absolute step; PRMB per-answer within-AUC by relative position of the first labelled error.
Pairwise: Jaccard of PB hit sets and union hit rate; correlation of per-answer PRMB AUC.
Writes COMPLEMENTARITY_ALL_ALPHAS.{json,md} and a figure COMPLEMENTARITY_ALL_ALPHAS.png.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ['RENYI_V2_ROSTER'] = 'all'
from scripts import run_renyi_view_fusion_v2 as run
from spectral_utils.renyi_alpha_sweep import METHODS, ALPHA_OF, FAMILY_OF

base = run.base
OUT = ROOT / 'results/renyi_alpha_sweep_v1'


def auc(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if not n1 or not n0: return np.nan
    return float((rankdata(s, method='average')[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def label(m):
    a = ALPHA_OF[m]; fam = 'VE' if FAMILY_OF[m] == 'escort_varentropy' else 'H'
    return f"{fam}_{'inf' if np.isinf(a) else ('0lim' if m == 'view__H0lim' else '%g' % a)}"


def main():
    source = Path(sys.argv[1]).resolve(); base.old.configure_source_root(source)
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    offsets, labels, target = joined['offsets'], joined['labels'], joined['target']
    steps = np.diff(offsets); cells = np.array([r['cell'] for r in records]); pb = np.char.startswith(cells, 'pb_')
    arms = list(METHODS); names = [label(m) for m in arms]
    with np.load(OUT / 'SCORES.npz') as npz:   # preload: indexing an NpzFile re-decompresses the array on every access
        Z = {k: npz[k] for k in npz.files if k.split('__', 1)[1] in arms}
    E = np.flatnonzero(pb & (target >= 0)); rel = target[E] / np.maximum(steps[E] - 1, 1)
    rel_s = np.select([rel < 1 / 3, rel < 2 / 3], ['first_third', 'middle_third'], 'last_third')
    abs_s = np.select([target[E] == 0, target[E] == 1, target[E] == 2, target[E] <= 5], ['step0', 'step1', 'step2', 'step3-5'], 'step6+')
    H = np.column_stack([(Z['valid__' + m] & (Z['prediction__' + m] == target))[E] for m in arms])
    # PRMB per-answer AUC
    M = np.flatnonzero(~pb); P = np.full((len(M), len(arms)), np.nan); first_rel = np.full(len(M), np.nan)
    for j, i in enumerate(M):
        y = labels[offsets[i]:offsets[i + 1]]; keep = y >= 0
        if not (y[keep] == 1).any() or not (y[keep] == 0).any(): continue
        first_rel[j] = np.flatnonzero(y == 1)[0] / max(len(y) - 1, 1)
        for k, m in enumerate(arms):
            s = Z['steps__' + m][offsets[i]:offsets[i + 1]]
            if np.isfinite(s).all(): P[j, k] = auc(y[keep] == 1, s[keep])
    ok = np.isfinite(first_rel) & np.isfinite(P).all(1); P = P[ok]; prel = first_rel[ok]
    prel_s = np.select([prel < 1 / 3, prel < 2 / 3], ['first_third', 'middle_third'], 'last_third')
    out = dict(scope='POST_HOC_DESCRIPTIVE', arms=names, n_pb_erroneous=int(len(E)), n_prmb_mixed=int(ok.sum()))
    out['pb_hit_by_relative'] = {k: {n: float(H[rel_s == k, j].mean()) for j, n in enumerate(names)} for k in ('first_third', 'middle_third', 'last_third')}
    out['pb_hit_by_absolute'] = {k: {n: float(H[abs_s == k, j].mean()) for j, n in enumerate(names)} for k in ('step0', 'step1', 'step2', 'step3-5', 'step6+')}
    out['prmb_within_by_relative'] = {k: {n: float(P[prel_s == k, j].mean()) for j, n in enumerate(names)} for k in ('first_third', 'middle_third', 'last_third')}
    inter = H.T.astype(int) @ H.astype(int); cnt = H.sum(0); union = cnt[:, None] + cnt[None, :] - inter
    J = inter / np.maximum(union, 1); U = union / len(E); C = np.corrcoef(P, rowvar=False)
    out['pb_jaccard'] = J.tolist(); out['pb_union_rate'] = U.tolist(); out['prmb_auc_corr'] = C.tolist()
    # best complementary pairs by union
    iu = np.triu_indices(len(arms), 1); order = np.argsort(-U[iu])[:10]
    out['top_union_pairs'] = [dict(a=names[iu[0][o]], b=names[iu[1][o]], union=float(U[iu][o]), jaccard=float(J[iu][o]),
                                   a_rate=float(H[:, iu[0][o]].mean()), b_rate=float(H[:, iu[1][o]].mean())) for o in order]
    base.atomic_json(OUT / 'COMPLEMENTARITY_ALL_ALPHAS.json', run.clean(out))
    L = ['# Complementarity across all sweep arms (post hoc, descriptive)', '', f'PB erroneous answers {len(E)}; PRMB mixed answers {int(ok.sum())}.', '',
         '## PB hit rate by relative first-error position', '', '| arm | overall | first third | middle third | last third | step0 | step6+ |', '|---|---:|---:|---:|---:|---:|---:|']
    for j, n in enumerate(names):
        L.append(f"| {n} | {H[:, j].mean():.3f} | " + ' | '.join(f"{out['pb_hit_by_relative'][k][n]:.3f}" for k in ('first_third', 'middle_third', 'last_third')) +
                 f" | {out['pb_hit_by_absolute']['step0'][n]:.3f} | {out['pb_hit_by_absolute']['step6+'][n]:.3f} |")
    L += ['', '## PRMB within-AUC by relative position of the first labelled error', '', '| arm | overall | first third | middle third | last third |', '|---|---:|---:|---:|---:|']
    for j, n in enumerate(names):
        L.append(f"| {n} | {P[:, j].mean():.4f} | " + ' | '.join(f"{out['prmb_within_by_relative'][k][n]:.4f}" for k in ('first_third', 'middle_third', 'last_third')) + ' |')
    L += ['', '## Ten pairs with the largest PB union (oracle pick per answer)', '', '| A | B | A rate | B rate | union | Jaccard |', '|---|---|---:|---:|---:|---:|']
    L += [f"| {d['a']} | {d['b']} | {d['a_rate']:.3f} | {d['b_rate']:.3f} | {d['union']:.3f} | {d['jaccard']:.2f} |" for d in out['top_union_pairs']]
    (OUT / 'COMPLEMENTARITY_ALL_ALPHAS.md').write_text('\n'.join(L), encoding='utf8'); print('\n'.join(L))
    # figure
    fig, ax = plt.subplots(2, 2, figsize=(18, 14))
    for a_, mat, title in ((ax[0, 0], J, 'PB hit-set Jaccard'), (ax[0, 1], C, 'PRMB per-answer AUC correlation')):
        im = a_.imshow(mat, vmin=0, vmax=1, cmap='viridis'); a_.set_xticks(range(len(names))); a_.set_yticks(range(len(names)))
        a_.set_xticklabels(names, rotation=90, fontsize=7); a_.set_yticklabels(names, fontsize=7); a_.set_title(title); fig.colorbar(im, ax=a_, fraction=.04)
    x = np.arange(len(names))
    for k, c in zip(('first_third', 'middle_third', 'last_third'), ('tab:blue', 'tab:orange', 'tab:green')):
        ax[1, 0].plot(x, [out['pb_hit_by_relative'][k][n] for n in names], '-o', ms=3, color=c, label=k)
        ax[1, 1].plot(x, [out['prmb_within_by_relative'][k][n] for n in names], '-o', ms=3, color=c, label=k)
    for a_, t in ((ax[1, 0], 'PB exact-hit rate by first-error position'), (ax[1, 1], 'PRMB within-AUC by first-error position')):
        a_.set_xticks(x); a_.set_xticklabels(names, rotation=90, fontsize=7); a_.set_title(t); a_.legend(); a_.grid(alpha=.3)
        a_.axvline(names.index('H_1'), color='k', ls=':', lw=.8); a_.axvline(names.index('VE_1'), color='k', ls=':', lw=.8)
    fig.suptitle('Complementarity across all Renyi / escort-varentropy orders (13,769 development answers; post hoc)')
    fig.tight_layout(); fig.savefig(OUT / 'COMPLEMENTARITY_ALL_ALPHAS.png', dpi=110); print('figure written')


if __name__ == '__main__':
    main()
