"""
temperature_followups.py — Step-158 follow-ups #3 and #7, unblocked by the
phase15_temperature folder pull (2026-07-14).

Data: local_cache/math500_qwen7b_T1.0_run0.pkl + local_cache/phase15_temperature/
math500_qwen7b_T{0.3,0.6,1.5,2.0}_run0.pkl (MATH-500 / Qwen2.5-Math-7B, N=200 each,
per-sample raw schema). Labels are each run's OWN labels (Step-158 Q1 protocol).

#3 — anchor/sign robustness across T. Step-158 Q1 reported an inverted-U
    (0.545/0.644/0.851/0.878/0.629 for T=0.3..2.0, GOOD_5 L-SML, epr anchor) and
    flagged that the label-free epr anchor is weak at low T — so low-T "poor
    detectability" might be a fusion/orientation artifact. Re-fuse per T with:
      (a) anchor = cusum_max instead of epr;
      (b) subset = GOOD_4 (GOOD_5 minus spectral_entropy, whose sign is known to be
          temperature-dependent), under both anchors.
    Also report each feature's oriented single AUROC per T (sign-stability table).

#7 — length-controlled AUROC per T. Hot traces are longer/degenerate; confirm the
    fused signal isn't just trace length:
      - AUROC of (negated) trace length alone per T;
      - AUROC of the fused score residualized on z-scored log-length (OLS);
      - length-stratified AUROC: tercile-stratified average (only strata with both
        classes and >= 15 samples count, sample-weighted).

Output: printed tables + results/phase1/temperature_followups.json
"""
import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from sklearn.metrics import roc_auc_score

from spectral_utils.feature_utils import extract_all_features
from spectral_utils.fusion_utils import zscore, boot_auc, lsml_continuous_pipeline
from spectral_utils.streaming_utils import FEATURE_SIGNS, anchor_orient

GOOD_5 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']
GOOD_4 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max']
Q1_REF = {0.3: 0.545, 0.6: 0.644, 1.0: 0.851, 1.5: 0.878, 2.0: 0.629}

T_PATHS = {
    0.3: 'local_cache/phase15_temperature/math500_qwen7b_T0.3_run0.pkl',
    0.6: 'local_cache/phase15_temperature/math500_qwen7b_T0.6_run0.pkl',
    1.0: 'local_cache/math500_qwen7b_T1.0_run0.pkl',
    1.5: 'local_cache/phase15_temperature/math500_qwen7b_T1.5_run0.pkl',
    2.0: 'local_cache/phase15_temperature/math500_qwen7b_T2.0_run0.pkl',
}


def load_cell(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    rows, labels, lengths = [], [], []
    for idx in sorted(data.keys()):
        s = data[idx]
        if not isinstance(s, dict) or 'token_entropies' not in s:
            continue
        feats = extract_all_features(s['token_entropies'],
                                     spilled_energies=s.get('token_spilled_energies'))
        if feats is None:
            continue
        rows.append(feats)
        labels.append(int(bool(s.get('label'))))
        lengths.append(len(s['token_entropies']))
    keys = sorted({k for k in rows[0] if all(k in r for r in rows)})
    fd = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    return fd, np.asarray(labels, dtype=int), np.asarray(lengths, dtype=float)


def fuse(fd, labels, feats, anchor_name, n_boot=1000):
    """score_subset-style: L-SML continuous + label-free anchor orientation."""
    sub = {f: fd[f] for f in feats}
    score, _ = lsml_continuous_pipeline(sub, feats, FEATURE_SIGNS)
    anchor = zscore(fd[anchor_name] * FEATURE_SIGNS.get(anchor_name, +1))
    score, flipped = anchor_orient(np.asarray(score, dtype=float), anchor)
    auc, lo, hi = boot_auc(labels, score, n=n_boot)
    return np.asarray(score, dtype=float), {
        'auroc': round(float(auc), 4), 'lo': round(float(lo), 4),
        'hi': round(float(hi), 4), 'flipped': bool(flipped)}


def residual_auc(score, lengths, labels):
    """AUROC of the fused score after OLS-residualizing on z-scored log-length."""
    s = zscore(score)
    L = zscore(np.log(lengths))
    beta = float(np.dot(s, L) / np.dot(L, L))
    resid = s - beta * L
    if resid.std() < 1e-12:
        return np.nan, beta
    return float(roc_auc_score(labels, resid)), beta


def stratified_auc(score, lengths, labels, n_bins=3, min_n=15):
    """Sample-weighted mean AUROC within length terciles (both-class strata only)."""
    qs = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1
    aucs, ns = [], []
    for b in range(n_bins):
        m = (lengths >= qs[b]) & (lengths < qs[b + 1])
        if m.sum() < min_n or len(np.unique(labels[m])) < 2:
            continue
        aucs.append(float(roc_auc_score(labels[m], score[m])))
        ns.append(int(m.sum()))
    if not aucs:
        return np.nan, 0
    return float(np.average(aucs, weights=ns)), int(sum(ns))


def main():
    out = {'per_T': {}}
    cells = {}
    for T, path in T_PATHS.items():
        full = os.path.join(REPO, path)
        if not os.path.exists(full):
            print(f'[missing] T={T}: {path}')
            continue
        fd, y, L = load_cell(full)
        cells[T] = (fd, y, L)
        print(f'[loaded] T={T}: n={len(y)} acc={y.mean():.3f} '
              f'len mean={L.mean():.0f} median={np.median(L):.0f}')

    # ── #3: anchor robustness ────────────────────────────────────────────────
    print('\n== #3 anchor/sign robustness: GOOD_5 / GOOD_4 x anchor {epr, cusum_max} ==')
    hdr = (f"{'T':>4} {'acc':>6} | {'Q1 ref':>6} {'G5/epr':>15} {'G5/cusum':>15} "
           f"{'G4/epr':>15} {'G4/cusum':>15}")
    print(hdr)
    for T in sorted(cells):
        fd, y, L = cells[T]
        row = {'acc': round(float(y.mean()), 4), 'n': int(len(y)), 'q1_ref': Q1_REF.get(T)}
        cols = []
        for feats, tag in ((GOOD_5, 'G5'), (GOOD_4, 'G4')):
            for anchor in ('epr', 'cusum_max'):
                _, r = fuse(fd, y, feats, anchor)
                row[f'{tag}_{anchor}'] = r
                flip = '*' if r['flipped'] else ' '
                cols.append(f"{r['auroc']:.3f}[{r['lo']:.2f},{r['hi']:.2f}]{flip}")
        out['per_T'].setdefault(T, {}).update(row)
        print(f"{T:>4} {y.mean():>6.3f} | {Q1_REF.get(T, float('nan')):>6.3f} "
              + ' '.join(f'{c:>15}' for c in cols))
    print('  (* = anchor_orient flipped the fused score)')

    # per-feature oriented singles across T (sign stability)
    print('\n== per-feature oriented single AUROC by T (fixed offline signs) ==')
    feats_all = GOOD_5
    print(f"{'feature':<18}" + ''.join(f"{('T=' + str(T)):>9}" for T in sorted(cells)))
    singles = {}
    for f in feats_all:
        vals = []
        for T in sorted(cells):
            fd, y, _ = cells[T]
            v = zscore(fd[f] * FEATURE_SIGNS.get(f, +1))
            vals.append(round(float(roc_auc_score(y, v)), 4))
        singles[f] = dict(zip([str(t) for t in sorted(cells)], vals))
        print(f'{f:<18}' + ''.join(f'{v:>9.3f}' for v in vals))
    out['singles_by_T'] = singles

    # ── #7: length control ───────────────────────────────────────────────────
    print('\n== #7 length-controlled AUROC per T (GOOD_5, epr anchor) ==')
    print(f"{'T':>4} | {'fused':>7} {'len-only':>9} {'resid':>7} {'beta':>7} "
          f"{'strat':>7} {'n_strat':>7}")
    for T in sorted(cells):
        fd, y, L = cells[T]
        score, r = fuse(fd, y, GOOD_5, 'epr')
        len_auc = float(roc_auc_score(y, -L))       # shorter -> correct convention
        res_auc, beta = residual_auc(score, L, y)
        st_auc, st_n = stratified_auc(score, L, y)
        out['per_T'][T].update({
            'len_only_auroc': round(len_auc, 4),
            'resid_auroc': round(res_auc, 4) if np.isfinite(res_auc) else None,
            'len_beta': round(beta, 4),
            'strat_auroc': round(st_auc, 4) if np.isfinite(st_auc) else None,
            'strat_n': st_n,
        })
        print(f"{T:>4} | {r['auroc']:>7.3f} {len_auc:>9.3f} {res_auc:>7.3f} "
              f"{beta:>7.3f} {st_auc:>7.3f} {st_n:>7}")

    os.makedirs(os.path.join(REPO, 'results', 'phase1'), exist_ok=True)
    out_path = os.path.join(REPO, 'results', 'phase1', 'temperature_followups.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nsaved -> {out_path}')


if __name__ == '__main__':
    main()
