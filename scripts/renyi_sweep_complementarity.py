"""Complementarity analysis of the sweep's leading single views (POST-HOC, descriptive; no refit, no new scores).

Question (Omri, 2026-09-13): do VE_0 and VE_0.75 (and varentropy15 / H0lim / entropy) find DIFFERENT errors?
E.g. early versus late first errors?

ProcessBench (erroneous answers only; the no-error gate is the same external mean-entropy gate for every arm, so
clean-answer decisions are identical across arms and every PB difference comes from the error-step peak):
  * exact-hit sets per arm, pairwise overlap, union ("oracle pick") and intersection;
  * hit rate stratified by the first-error position: absolute step index (0, 1, 2, 3-5, 6+), relative position
    (target / (steps-1): first / middle / last third), and answer length in steps;
  * miss direction per arm (peak before / after the target) in the same strata;
  * for each pair: hits unique to A vs unique to B, by stratum.
PRMBench (mixed answers): per-answer within-AUC per arm; correlation between arms; fraction of answers where
one arm beats the other by > 0.1; stratified by the relative position of the first labelled error step.
Writes results/renyi_alpha_sweep_v1/COMPLEMENTARITY.{json,md}.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ['RENYI_V2_ROSTER'] = 'all'
from scripts import run_renyi_view_fusion_v2 as run

base = run.base
OUT = ROOT / 'results/renyi_alpha_sweep_v1'
ARMS = {'VE_0': 'view__ve0', 'VE_0.75': 'view__ve0.75', 'varentropy15': 'view__ve1', 'H0lim': 'view__H0lim', 'entropy': 'view__H1'}
PAIRS = [('VE_0', 'VE_0.75'), ('VE_0', 'varentropy15'), ('VE_0.75', 'varentropy15'), ('H0lim', 'VE_0.75'), ('H0lim', 'entropy'), ('VE_0', 'H0lim')]


def auc(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if not n1 or not n0: return np.nan
    return float((rankdata(s, method='average')[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def strata(target, steps):
    rel = target / np.maximum(steps - 1, 1)
    abs_s = np.select([target == 0, target == 1, target == 2, target <= 5], ['step0', 'step1', 'step2', 'step3-5'], 'step6+')
    rel_s = np.select([rel < 1 / 3, rel < 2 / 3], ['first_third', 'middle_third'], 'last_third')
    len_s = np.select([steps <= 4, steps <= 8], ['1-4 steps', '5-8 steps'], '9+ steps')
    return dict(absolute=abs_s, relative=rel_s, length=len_s)


def main():
    source = Path(sys.argv[1]).resolve(); base.old.configure_source_root(source)
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    offsets, labels, target = joined['offsets'], joined['labels'], joined['target']
    steps = np.diff(offsets); cells = np.array([r['cell'] for r in records]); pb = np.char.startswith(cells, 'pb_')
    Z = np.load(OUT / 'SCORES.npz')
    err = pb & (target >= 0); E = np.flatnonzero(err)
    S = strata(target[E], steps[E])
    hit, peak = {}, {}
    for a, m in ARMS.items():
        pred = Z['prediction__' + m]; valid = Z['valid__' + m]; flat = Z['steps__' + m]
        hit[a] = (valid & (pred == target))[E]
        pk = np.array([int(np.argmax(flat[offsets[i]:offsets[i + 1]])) if valid[i] else -1 for i in E]); peak[a] = pk
    out = dict(scope='POST_HOC_DESCRIPTIVE', n_pb_erroneous=int(len(E)), arms=ARMS,
               note='PB: same external gate for every arm, so clean-answer decisions are identical and all PB differences are '
                    'error-peak differences. Hit = decision valid and predicted step == first labelled error step.')
    lines = ['# Complementarity of the leading single views (post hoc, descriptive)', '',
             f'ProcessBench erroneous answers: {len(E)}. The no-error gate is identical across arms, so only error peaks differ.', '']
    # overall hit rates and pairwise overlap
    out['pb_hit_rate'] = {a: float(h.mean()) for a, h in hit.items()}
    lines += ['## PB exact-hit rate on erroneous answers', '', '| arm | hits | rate |', '|---|---:|---:|']
    lines += [f'| {a} | {int(h.sum())} | {h.mean():.3f} |' for a, h in hit.items()]
    out['pb_pairs'] = {}
    lines += ['', '## Pairwise overlap of PB hits', '', '| pair | both | only A | only B | neither | union rate | Jaccard |', '|---|---:|---:|---:|---:|---:|---:|']
    for a, b in PAIRS:
        ha, hb = hit[a], hit[b]
        d = dict(both=int((ha & hb).sum()), only_a=int((ha & ~hb).sum()), only_b=int((~ha & hb).sum()), neither=int((~ha & ~hb).sum()),
                 union_rate=float((ha | hb).mean()), jaccard=float((ha & hb).sum() / max((ha | hb).sum(), 1)))
        out['pb_pairs'][f'{a} vs {b}'] = d
        lines.append(f"| {a} vs {b} | {d['both']} | {d['only_a']} | {d['only_b']} | {d['neither']} | {d['union_rate']:.3f} | {d['jaccard']:.2f} |")
    allhit = np.column_stack([hit[a] for a in ARMS]); out['pb_union_all'] = float(allhit.any(1).mean()); out['pb_intersection_all'] = float(allhit.all(1).mean())
    lines += ['', f"Union of all five arms (oracle pick per answer): {out['pb_union_all']:.3f}; intersection: {out['pb_intersection_all']:.3f}.", '']
    # stratified hit rates + miss direction
    out['pb_strata'] = {}
    for kind, s in S.items():
        keys = [k for k in (['step0', 'step1', 'step2', 'step3-5', 'step6+'] if kind == 'absolute' else
                            ['first_third', 'middle_third', 'last_third'] if kind == 'relative' else ['1-4 steps', '5-8 steps', '9+ steps'])]
        out['pb_strata'][kind] = {}
        lines += [f'## PB hit rate by first-error position ({kind})', '', '| stratum | n | ' + ' | '.join(ARMS) + ' | union | ' +
                  ' | '.join(f'{a} early/late miss' for a in ('VE_0', 'VE_0.75')) + ' |', '|---|---:|' + '---:|' * (len(ARMS) + 3)]
        for k in keys:
            msk = s == k; n = int(msk.sum())
            if not n: continue
            row = {a: float(hit[a][msk].mean()) for a in ARMS}; row['union'] = float(allhit[msk].any(1).mean()); row['n'] = n
            for a in ('VE_0', 'VE_0.75'):
                pk = peak[a][msk]; t = target[E][msk]; miss = ~hit[a][msk] & (pk >= 0)
                row[a + '_miss_early'] = float(np.mean(pk[miss] < t[miss])) if miss.any() else None
                row[a + '_miss_late'] = float(np.mean(pk[miss] > t[miss])) if miss.any() else None
            out['pb_strata'][kind][k] = row
            lines.append(f'| {k} | {n} | ' + ' | '.join(f'{row[a]:.3f}' for a in ARMS) + f" | {row['union']:.3f} | " +
                         ' | '.join((f"{row[a + '_miss_early']:.2f}/{row[a + '_miss_late']:.2f}" if row[a + '_miss_early'] is not None else '-') for a in ('VE_0', 'VE_0.75')) + ' |')
        lines.append('')
    # unique hits by stratum for the main pair
    out['pb_unique_by_relative'] = {}
    lines += ['## Where the unique hits of VE_0 and VE_0.75 fall (relative position of the first error)', '', '| stratum | only VE_0 | only VE_0.75 | both |', '|---|---:|---:|---:|']
    for k in ['first_third', 'middle_third', 'last_third']:
        msk = S['relative'] == k; ha, hb = hit['VE_0'][msk], hit['VE_0.75'][msk]
        d = dict(only_ve0=int((ha & ~hb).sum()), only_ve075=int((~ha & hb).sum()), both=int((ha & hb).sum())); out['pb_unique_by_relative'][k] = d
        lines.append(f"| {k} | {d['only_ve0']} | {d['only_ve075']} | {d['both']} |")
    # PRMB per-answer within-AUC
    M = np.flatnonzero(~pb); per = {a: np.full(len(M), np.nan) for a in ARMS}; first_rel = np.full(len(M), np.nan); nsteps = steps[M]
    for j, i in enumerate(M):
        y = labels[offsets[i]:offsets[i + 1]]; keep = y >= 0
        if not (y[keep] == 1).any() or not (y[keep] == 0).any(): continue
        first_rel[j] = np.flatnonzero(y == 1)[0] / max(len(y) - 1, 1)
        for a, m in ARMS.items():
            s = Z['steps__' + m][offsets[i]:offsets[i + 1]]
            if np.isfinite(s).all(): per[a][j] = auc(y[keep] == 1, s[keep])
    ok = np.isfinite(first_rel) & np.all(np.column_stack([np.isfinite(per[a]) for a in ARMS]), 1)
    out['prmb'] = dict(n_mixed=int(ok.sum()), mean_within={a: float(np.nanmean(per[a][ok])) for a in ARMS}, pairs={}, by_first_error_position={})
    lines += ['', f'## PRMBench per-answer within-AUC ({int(ok.sum())} mixed answers)', '', '| pair | corr of per-answer AUC | A better by >0.1 | B better by >0.1 | mean A−B |', '|---|---:|---:|---:|---:|']
    for a, b in PAIRS:
        d = per[a][ok] - per[b][ok]
        e = dict(corr=float(np.corrcoef(per[a][ok], per[b][ok])[0, 1]), a_better_0_1=float(np.mean(d > .1)), b_better_0_1=float(np.mean(d < -.1)), mean_diff=float(d.mean()))
        out['prmb']['pairs'][f'{a} vs {b}'] = e
        lines.append(f"| {a} vs {b} | {e['corr']:.3f} | {e['a_better_0_1']:.3f} | {e['b_better_0_1']:.3f} | {e['mean_diff']:+.4f} |")
    rel = np.select([first_rel < 1 / 3, first_rel < 2 / 3], ['first_third', 'middle_third'], 'last_third')
    lines += ['', '## PRMBench within-AUC by relative position of the first labelled error', '', '| stratum | n | ' + ' | '.join(ARMS) + ' |', '|---|---:|' + '---:|' * len(ARMS)]
    for k in ['first_third', 'middle_third', 'last_third']:
        msk = ok & (rel == k); row = {a: float(np.mean(per[a][msk])) for a in ARMS}; row['n'] = int(msk.sum())
        out['prmb']['by_first_error_position'][k] = row
        lines.append(f"| {k} | {row['n']} | " + ' | '.join(f'{row[a]:.4f}' for a in ARMS) + ' |')
    base.atomic_json(OUT / 'COMPLEMENTARITY.json', run.clean(out)); (OUT / 'COMPLEMENTARITY.md').write_text('\n'.join(lines), encoding='utf8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
