"""
Pivot-alternatives pilot, Track A — unsupervised anomaly scorers as an
aggregation-layer hedge for L-SML (Step 151).

Fits each scorer in spectral_utils.anomaly_utils per eval cell on ALL
unlabeled samples of that cell (features only, zero labels) and scores those
same samples.  This transductive protocol is the matched comparison, not
leakage: L-SML continuous estimates its eigenvector weights from the
unlabeled covariance of the same eval cell it scores, so candidates and
comparator have identical information access.

Pre-registered protocol (decided before the run):
  - Orientation is label-free.  Three AUROC tiers are reported per
    (cell, method, feature set):
      auc_raw      fixed a-priori convention: higher anomaly = hallucination
      auc_anchored PRIMARY (used for Gate A): the confidence score is
                   sign-anchored to oriented epr via anchor_orient — the
                   Step-148 label-free device.  Handles the structural
                   inversion on hallucination-majority cells.
      auc_oracle   max(p, 1-p) — DIAGNOSTIC ONLY, never gated.  This is the
                   only tier strictly comparable to the stored L-SML
                   comparators (upcr_comparison.pkl used label-peeked
                   orientation), so a Gate A pass on the anchored tier is
                   conservative.
  - Gate A, per candidate, per feature set, on common cells (candidate and
    cont_* both finite): macro(auc_anchored) >= macro(cont) - 1pp  => PASS
    (viable hedge); >= macro(cont) + 1pp => STRETCH.  Primary feature set: 16.
  - ae/prae skipped on cells with n < AE_MIN_SAMPLES (80).
  - Diagnostic: if prae <= maha (macro), the verdict is "nonlinearity adds
    nothing" regardless of the gate.

Comparators are read-only, never recomputed: cont_5/cont_16 from
results/upcr_comparison.pkl, avg5/avg16 from
results/method_comparison_table1.csv.

Usage:
    python scripts/pivot_trackA.py [--data-dir ./local_cache]
        [--out results/pivot_trackA.pkl] [--smoke-test] [--force]
        [--methods maha,gmm2,...] [--feature-sets 5 16] [--n-boot 1000]
"""

import os
import sys
import csv
import time
import pickle
import argparse
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils import boot_auc, zscore, anchor_orient, FEAT_NAMES
from spectral_utils.anomaly_utils import (
    TRACKA_METHODS, AE_MIN_SAMPLES, build_feature_matrix, is_saturated,
)
from sklearn.metrics import roc_auc_score

# ── Feature sets / cells (mirrors scripts/logistic_oracle.py) ────────────────

GOOD_5 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']
ALL_H16 = FEAT_NAMES[:16]
FEATURE_SETS = {'5': GOOD_5, '16': ALL_H16}

PKL_NAMES = {
    'math500': 'math500_res.pkl',
    'gsm8k':   'gsm8k_res.pkl',
    'gpqa':    'gpqa_res.pkl',
    'rag':     'rag_feats_all.pkl',
    'qa':      'qa_res.pkl',
}

# Step-134 regime convention
REGIME = {'math500': 'reasoning', 'gsm8k': 'reasoning', 'qa': 'reasoning',
          'gpqa': 'gpqa', 'rag': 'rag'}

CONT_PKL = os.path.join(REPO_DIR, 'results', 'upcr_comparison.pkl')
AVG_CSV  = os.path.join(REPO_DIR, 'results', 'method_comparison_table1.csv')
OUT_PKL  = os.path.join(REPO_DIR, 'results', 'pivot_trackA.pkl')

GATE_MARGIN = 0.01  # 1pp non-inferiority margin


# ── Data loading (local copies — scripts are self-contained by convention) ───

def load_cached_feats(pkl_path):
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'feats' in obj:
        return obj['feats']
    return obj


def iter_cells(data_dir, verbose=True):
    for domain, pkl_name in PKL_NAMES.items():
        path = os.path.join(data_dir, pkl_name)
        feats = load_cached_feats(path)
        if feats is None:
            if verbose:
                print(f'[MISSING] {pkl_name}')
            continue
        for cell_key, payload in feats.items():
            if isinstance(payload, (list, tuple)) and len(payload) == 2:
                fd, lbl = payload
            elif isinstance(payload, dict) and 'feats' in payload:
                fd, lbl = payload['feats'], payload['labels']
            else:
                continue
            lbl_arr = np.asarray(lbl, dtype=int)
            if len(set(lbl_arr.tolist())) < 2:
                continue
            yield domain, f'{domain}/{cell_key}', fd, lbl_arr


def load_comparators():
    """cont_5/cont_16 from upcr_comparison.pkl, avg5/avg16 from table1 CSV."""
    comp = {}
    if os.path.exists(CONT_PKL):
        with open(CONT_PKL, 'rb') as f:
            for r in pickle.load(f):
                if r.get('skipped'):
                    continue
                comp[r['cell']] = {'cont_5': r.get('cont_5'),
                                   'cont_16': r.get('cont_16')}
    else:
        print(f'[WARN] {CONT_PKL} not found — cont comparator empty')
    if os.path.exists(AVG_CSV):
        with open(AVG_CSV, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                cell = f"{row['domain']}/{row['cell_key']}"
                d = comp.setdefault(cell, {})
                for src, dst in (('avg5', 'avg_5'), ('avg16', 'avg_16')):
                    v = (row.get(src) or '').strip().rstrip('%')
                    d[dst] = float(v) / 100.0 if v else None
    else:
        print(f'[WARN] {AVG_CSV} not found — avg comparator empty')
    return comp


# ── Orientation tiers ─────────────────────────────────────────────────────────

def epr_anchor_view(fd, n_samples):
    """Oriented epr view (higher = correct), median-imputed. None if missing."""
    if 'epr' not in fd or fd['epr'] is None or len(fd['epr']) != n_samples:
        return None
    e = np.asarray(fd['epr'], dtype=float)
    bad = ~np.isfinite(e)
    if bad.all():
        return None
    if bad.any():
        e[bad] = np.nanmedian(e)
    if e.std() < 1e-12:
        return None
    return -zscore(e)


def auc_tiers(y, anomaly, anchor, n_boot):
    """Compute the three orientation tiers for one anomaly score vector."""
    conf_raw = -np.asarray(anomaly, dtype=float)
    a_raw, lo_raw, hi_raw = boot_auc(y, conf_raw, n=n_boot)
    if anchor is not None:
        conf_anch, flipped = anchor_orient(conf_raw, anchor)
    else:
        conf_anch, flipped = conf_raw, False
    a_anch, lo_anch, hi_anch = boot_auc(y, conf_anch, n=n_boot)
    oracle = max(a_raw, 1.0 - a_raw) if np.isfinite(a_raw) else float('nan')
    return {
        'auc_raw': (a_raw, lo_raw, hi_raw),
        'auc_anchored': (a_anch, lo_anch, hi_anch),
        'auc_oracle': oracle,
        'anchor_flip': bool(flipped),
        'no_anchor': anchor is None,
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

def smoke_test():
    """Planted-outlier check: every method must separate; ae must track maha."""
    print('=== SMOKE TEST (synthetic planted outliers) ===')
    rng = np.random.default_rng(7)
    n, d, n_out = 120, 16, 24
    X = rng.normal(0, 1, (n, d))
    out_idx = rng.choice(n, n_out, replace=False)
    X[out_idx] += rng.normal(2.5, 1.0, (n_out, d))
    y = np.ones(n, dtype=int)
    y[out_idx] = 0  # outliers = hallucinations (label 0)

    aucs, raw_scores = {}, {}
    for name, fn in TRACKA_METHODS.items():
        s, _ = fn(X)
        aucs[name] = roc_auc_score(y, -s)
        raw_scores[name] = s
        print(f'  {name:8s} AUROC={aucs[name]:.3f}')
    ok = all(a > 0.80 for a in aucs.values())
    # implementation sanity: ae and maha must broadly rank the same points as
    # anomalous (AUROC gaps are expected — the AE partially absorbs strong
    # contamination into its bottleneck, which is PRAE's whole motivation)
    from scipy.stats import spearmanr
    rho = spearmanr(raw_scores['ae'], raw_scores['maha']).statistic
    print(f'  spearman(ae, maha) = {rho:.3f}')
    ok = ok and rho > 0.5
    print(f'SMOKE TEST: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


# ── Main run ──────────────────────────────────────────────────────────────────

def save(res, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(res, f)


def run(args):
    method_names = (args.methods.split(',') if args.methods
                    else list(TRACKA_METHODS))
    res = {}
    if os.path.exists(args.out) and not args.force:
        with open(args.out, 'rb') as f:
            res = pickle.load(f)
        print(f'[resume] loaded {len(res)} cells from {args.out}')

    for domain, cell, fd, y in iter_cells(args.data_dir):
        n = len(y)
        centry = res.setdefault(cell, {
            'domain': domain, 'regime': REGIME[domain], 'n': n,
            'prevalence': float(y.mean()),
            'labels': y.astype(np.int8), 'fs': {},
        })
        anchor = epr_anchor_view(fd, n)
        for fs in args.feature_sets:
            X, avail = build_feature_matrix(fd, FEATURE_SETS[fs], n)
            fentry = centry['fs'].setdefault(fs, {'available': avail,
                                                  'methods': {}})
            if X is None:
                continue
            for m in method_names:
                if m in fentry['methods']:
                    continue  # resume
                t0 = time.time()
                if m in ('ae', 'prae') and n < AE_MIN_SAMPLES:
                    fentry['methods'][m] = None
                    save(res, args.out)
                    continue
                scores, meta = TRACKA_METHODS[m](X)
                entry = auc_tiers(y, scores, anchor, args.n_boot)
                entry['scores'] = np.asarray(scores, dtype=np.float32)
                entry['meta'] = {k: v for k, v in meta.items()
                                 if k != 'gate_probs'}
                if 'gate_probs' in meta:
                    entry['gate_probs'] = np.asarray(meta['gate_probs'],
                                                     dtype=np.float32)
                fentry['methods'][m] = entry
                save(res, args.out)
                a = entry['auc_anchored'][0]
                print(f'  {cell} fs={fs} {m:8s} anch={a:.3f} '
                      f'({time.time() - t0:.1f}s)')
    return res


# ── Gate A verdict ────────────────────────────────────────────────────────────

def macro(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float('nan')


def gate_verdict(res, comp, method_names, feature_sets):
    print('\n' + '=' * 74)
    print('GATE A — anomaly scorer vs L-SML continuous (anchored tier, '
          f'non-inferiority margin {GATE_MARGIN * 100:.0f}pp)')
    print('=' * 74)
    verdicts = {}
    for fs in feature_sets:
        cont_key = f'cont_{fs}'
        print(f'\n--- feature set {fs} '
              f'({"PRIMARY" if fs == "16" else "secondary"}) ---')
        header = f'{"method":9s} {"n_cells":>7s} {"macro_anch":>10s} ' \
                 f'{"macro_cont":>10s} {"delta":>7s}  verdict   regimes(anch)'
        print(header)
        for m in method_names:
            pairs, regime_vals = [], {}
            for cell, ce in res.items():
                me = ce.get('fs', {}).get(fs, {}).get('methods', {}).get(m)
                cv = comp.get(cell, {}).get(cont_key)
                if me is None or cv is None or not np.isfinite(cv):
                    continue
                a = me['auc_anchored'][0]
                if not np.isfinite(a):
                    continue
                pairs.append((a, cv))
                regime_vals.setdefault(ce['regime'], []).append(a)
            if not pairs:
                print(f'{m:9s} {"0":>7s}  — no common cells')
                verdicts[(m, fs)] = 'NO DATA'
                continue
            ma = macro([p[0] for p in pairs])
            mc = macro([p[1] for p in pairs])
            delta = ma - mc
            if delta >= GATE_MARGIN:
                v = 'STRETCH'
            elif delta >= -GATE_MARGIN:
                v = 'PASS'
            else:
                v = 'FAIL'
            verdicts[(m, fs)] = v
            regs = ' '.join(f'{r}={macro(vs):.3f}'
                            for r, vs in sorted(regime_vals.items()))
            print(f'{m:9s} {len(pairs):>7d} {ma:>10.3f} {mc:>10.3f} '
                  f'{delta * 100:>+6.1f}pp  {v:8s} {regs}')

    # nonlinearity diagnostic
    print('\n--- diagnostic: does the AE nonlinearity add anything? ---')
    for fs in feature_sets:
        vals = {}
        for m in ('maha', 'ae', 'prae'):
            cells = [ce['fs'][fs]['methods'][m]['auc_anchored'][0]
                     for ce in res.values()
                     if ce.get('fs', {}).get(fs, {}).get('methods', {}).get(m)
                     and np.isfinite(
                         ce['fs'][fs]['methods'][m]['auc_anchored'][0])]
            vals[m] = macro(cells)
        print(f'  fs={fs}: maha={vals["maha"]:.3f} ae={vals["ae"]:.3f} '
              f'prae={vals["prae"]:.3f} '
              f'(prae-maha={100 * (vals["prae"] - vals["maha"]):+.1f}pp, '
              f'prae-ae={100 * (vals["prae"] - vals["ae"]):+.1f}pp)')
        if vals['prae'] <= vals['maha']:
            print(f'  fs={fs}: PRAE <= Mahalanobis — nonlinearity adds '
                  'nothing on this battery')
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=os.path.join(REPO_DIR, 'local_cache'))
    ap.add_argument('--out', default=OUT_PKL)
    ap.add_argument('--smoke-test', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--methods', default=None,
                    help='comma-separated subset of methods')
    ap.add_argument('--feature-sets', nargs='+', default=['5', '16'],
                    choices=list(FEATURE_SETS))
    ap.add_argument('--n-boot', type=int, default=1000)
    args = ap.parse_args()

    if args.smoke_test:
        sys.exit(smoke_test())

    t0 = time.time()
    res = run(args)
    comp = load_comparators()
    method_names = (args.methods.split(',') if args.methods
                    else list(TRACKA_METHODS))
    gate_verdict(res, comp, method_names, args.feature_sets)
    print(f'\ntotal {time.time() - t0:.0f}s — results in {args.out}')


if __name__ == '__main__':
    main()
