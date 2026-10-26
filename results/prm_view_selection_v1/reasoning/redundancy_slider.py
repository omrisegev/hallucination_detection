"""Redundancy errors under the SLIDER protocol, pre-declared in PREDECLARATION_REDUNDANCY_SLIDER.md (2026-10-26).

    python redundancy_slider.py   # -> REDUNDANCY_SLIDER.csv, REDUNDANCY_SLIDER_PARITY.json
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval8 as V  # noqa: E402
E, S1, S2 = V.E, V.S1, V.S2
SOC_RED = ['premise_redundancy', 'decomposition_redundancy', 'regather_redundancy', 'integration_redundancy']
ALL16 = list(range(16))


def slider_protocol(err, s, answers):
    """collection_competitors.slider, LR balanced: fit on half the answers, accuracy / balanced accuracy on the rest."""
    acc, bal = [], []
    for sd in range(10):
        perm = np.random.default_rng(sd).permutation(len(answers)); h = len(answers) // 2
        xh = np.concatenate([s[answers[k]] for k in perm[:h]])[:, None]; yh = np.concatenate([err[answers[k]] for k in perm[:h]])
        xt = np.concatenate([s[answers[k]] for k in perm[h:]])[:, None]; yt = np.concatenate([err[answers[k]] for k in perm[h:]])
        p = LogisticRegression(class_weight='balanced').fit(xh, yh).predict(xt).astype(bool)
        acc.append(100 * (p == yt).mean()); bal.append(50 * (p[yt].mean() + (~p[~yt]).mean()))
    return float(np.mean(acc)), float(np.mean(bal))


def label_free(err, s, off, answers):
    f = V.exact_flags(s, off)
    idx = np.concatenate(answers); y = err[idx]; ff = f[idx]
    return 100 * float((ff == y).mean()), 50 * float(ff[y].mean() + (~ff[~y]).mean())


def main():
    sha = hashlib.sha256((HERE / 'PREDECLARATION_REDUNDANCY_SLIDER.md').read_bytes()).hexdigest()
    rows = []
    for b, p in V.CELLS:
        c = E.load(b, p); D = np.load(V.SCORES / f'{V.key(b, p)}.npz')
        sc = {n: D[n.replace('|', '__').replace('+', 'P')] for n in D['names'] if not n.endswith(('|mean', '|bin', '|floor1e-4'))}
        sc['CHAIN_N16'] = c.arms['CHAIN_N16']; sc['TELE_N16'] = c.arms['TELE_N16']
        if b == 'prmbench':
            for a in ('LSML_CONT', 'CHAINRES_N16', 'TELERES_N16'): sc[a] = c.arms[a]
        for rep, d1, d2 in (('RAW', V.VS / 'stage1' / 'cache', V.VS / 'stage2' / 'cache'), ('POS', V.POSC, V.POSC)):
            c1 = S1.CellData.from_light(d1 / (f'{V.key(b, p)}.npz' if rep == 'RAW' else f's1_{V.key(b, p)}.npz'))
            c2 = (S2.TeleCell.load(d2 / f'{V.key(b, p)}.npz') if rep == 'RAW' else S2.TeleCell.load(d2 / f's2_{V.key(b, p)}.npz'))
            for mode, cell in (('+PRM', c1), ('alone', c2)):
                o = cell.evaluate(ALL16, want_detail=True)
                if o['valid']: sc[f'ALL16|{rep}|{mode}'] = o['fused']
        if b == 'prmbench':
            subsets = {'redundency': np.flatnonzero(c.cls == 'redundency')}
        else:
            subsets = {cat: np.flatnonzero(c.cat == cat) for cat in SOC_RED}
            subsets['all_redundancy'] = np.flatnonzero(np.isin(c.cat, SOC_RED))
        for sub, J in subsets.items():
            answers = [np.arange(c.off[j], c.off[j + 1]) for j in J]
            if b != 'prmbench':
                answers = [a[c.keep[a]] for a in answers]
            for arm, s in sc.items():
                acc, bal = slider_protocol(c.err, s, answers)
                lf_acc, lf_bal = label_free(c.err, s, c.off, answers)
                keep = np.zeros(c.off[-1], bool); keep[np.concatenate(answers)] = True
                auc = float(np.nanmean(E.per_answer_auc(c.err, s, c.off, keep)))
                rows.append(dict(cell=V.key(b, p), subset=sub, n_answers=len(J), arm=arm, slider_acc=acc, slider_bal_acc=bal,
                                 gap_acc=lf_acc, gap_bal_acc=lf_bal, within_auc=auc))
        print('done', b, p, flush=True); del c
    E.write_csv(HERE / 'REDUNDANCY_SLIDER.csv', rows)
    # parity with COMP_SLIDER (PRMBench, all steps, LR balanced accuracy)
    ref = {(r['prm'], r['arm']): r for r in E.read_csv(E.TABLES / 'COMP_SLIDER.csv') if r['prm'] != 'paper' and r.get('rule') == 'GAP'}
    diffs = []
    for r in rows:
        if r['subset'] != 'redundency': continue
        k = (r['cell'].split('__')[1], r['arm'])
        if k in ref: diffs.append(abs(r['slider_acc'] - float(ref[k]['all_steps__LR_balanced__acc'])))
    json.dump(dict(predeclaration_sha256=sha, parity_n=len(diffs), parity_max_abs_diff=max(diffs) if diffs else None),
              open(HERE / 'REDUNDANCY_SLIDER_PARITY.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
