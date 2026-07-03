"""
Pivot-alternatives pilot, Track B — temporal models on raw entropy traces
(Step 151).  TRIAGE GRADE: one clean cell only.

Candidates (spectral_utils.temporal_models, all unsupervised, fit per cell on
unlabeled traces):
    hmm_occ / hmm_tail / hmm_switch — 2-state Gaussian HMM on pooled-
        standardized (H(t), dE(t)) (1-D H(t) where spilled is missing);
        scores = high-entropy-regime posterior statistics.  Honest version of
        the pivot doc's IMM Option 5.
    bocpd_ecp / bocpd_meanp0 / bocpd_map — BOCPD (Adams-MacKay) on raw H(t);
        hazard lambda=100 primary, {50, 200} sensitivity.
    ar2_mse / ar2_ratio — AR(2) innovation scores per trace.
    kalman_mse / kalman_nis — constant-velocity Kalman innovations.
        The AR/Kalman rows are the KalmanNet go/no-go: zero label signal in
        innovations => a learned Kalman gain has nothing to amplify.

Baselines recomputed in-script on the identical trace set (required for
per-sample pairing): DeepConf lowest-group-confidence w in {32,64,128}, tail
confidence, oriented epr, max entropy, full-trace L-SML continuous (GOOD_5 and
H16, epr-anchored — the Step-148 recipe).  Recomputed full-trace AUROCs are
cross-checked against the stored streaming-pilot values (results/sp_gsm8k.pkl).

Pre-registered Gate B (primary cell gsm8k/Llama-3.1-8B only):
    candidate anchored AUROC >= best DeepConf window (by point AUROC — this
    choice favours the baseline) AND >= full-trace lsml5, AND the paired
    bootstrap 95% CI of (candidate - best DeepConf) excludes 0
        => PROMOTE to Colab replication (raw-trace re-inference is already
           the roadmap's next Colab item).  Otherwise: tried-and-rejected.
    With n=200 the CI clause is deliberately strict (~±9pp half-width).

The MATH-500/Qwen-1.5B cell is SECONDARY / NON-CANONICAL (K=8 correlated
traces per question, no spilled energy, its epr AUROC is far below the
canonical cell): cluster bootstrap by question, never enters the gate.

Usage:
    python scripts/pivot_trackB.py [--data-dir local_cache/raw_traces]
        [--out results/pivot_trackB.pkl] [--smoke-test] [--force]
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils import (  # noqa: E402
    FEAT_NAMES, FEATURE_SIGNS, anchor_orient, boot_auc, paired_boot_delta_auc,
    deepconf_lowest_group_conf, deepconf_tail_conf, iter_trace_records,
    lsml_continuous_pipeline, prefix_feature_matrix,
    fit_gaussian_hmm, hmm_trace_scores, bocpd_gaussian,
    ar_innovation_scores, kalman_innovation_scores,
)

H16 = FEAT_NAMES[:16]
GOOD_5 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']
DEEPCONF_WINDOWS = [32, 64, 128]
BOCPD_LAMBDAS = [100.0, 50.0, 200.0]  # first = primary

CELLS = {
    'p1_gsm8k_llama8b': {'name': 'gsm8k/Llama-3.1-8B', 'primary': True},
    'math500_T1.0': {'name': 'math500/Qwen2.5-Math-1.5B_T1.0', 'primary': False},
}
PRIMARY_CANDIDATES = ['hmm_occ', 'bocpd_ecp', 'ar2_mse']

OUT_PKL = os.path.join(REPO_DIR, 'results', 'pivot_trackB.pkl')
SP_PKL = os.path.join(REPO_DIR, 'results', 'sp_gsm8k.pkl')


# ── Loading ───────────────────────────────────────────────────────────────────

def load_cell(path):
    """Load one raw-trace cache -> dict of aligned lists (len >= 8 filter)."""
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    recs = [r for r in iter_trace_records(obj) if len(r['ents']) >= 8]
    if len(recs) < 30:
        return None
    return {
        'traces': [r['ents'] for r in recs],
        'spilled': [r['spilled'] for r in recs],
        'labels': np.array([r['label'] for r in recs], dtype=int),
        'groups': np.array([r['group'] for r in recs], dtype=int),
    }


def hmm_observations(cell):
    """Pooled-standardized (H, dE) observations; 1-D H when spilled missing.

    Standardization uses POOLED token statistics across the cell (not
    per-trace), so trace-level entropy differences — the main signal — are
    preserved; only the scale is normalized for EM stability.
    """
    has_spilled = all(s is not None for s in cell['spilled'])
    cols = 2 if has_spilled else 1
    pooled_h = np.concatenate(cell['traces'])
    mu_h, sd_h = pooled_h.mean(), max(pooled_h.std(), 1e-8)
    if has_spilled:
        pooled_s = np.concatenate(cell['spilled'])
        mu_s, sd_s = pooled_s.mean(), max(pooled_s.std(), 1e-8)
    obs = []
    for i, tr in enumerate(cell['traces']):
        h = (tr - mu_h) / sd_h
        if has_spilled:
            s = (cell['spilled'][i] - mu_s) / sd_s
            obs.append(np.column_stack([h, s]))
        else:
            obs.append(h[:, None])
    return obs, cols


# ── Candidate scores (all ANOMALY-oriented: higher = suspect) ────────────────

def candidate_scores(cell):
    out, meta = {}, {}

    obs, obs_dim = hmm_observations(cell)
    hmm = fit_gaussian_hmm(obs)
    sc = hmm_trace_scores(hmm, obs)
    out['hmm_occ'] = sc['occupancy']
    out['hmm_tail'] = sc['tail_occupancy']
    out['hmm_switch'] = sc['switch_rate']
    meta['hmm'] = {'obs_dim': obs_dim, 'degenerate': hmm['degenerate'],
                   'occupancy': hmm['occupancy'].tolist(),
                   'means_H': hmm['means'][:, 0].tolist(),
                   'trans_diag': np.diag(hmm['trans']).tolist()}

    for lam in BOCPD_LAMBDAS:
        tag = '' if lam == BOCPD_LAMBDAS[0] else f'_l{int(lam)}'
        ecp, mp0, mapc = [], [], []
        for tr in cell['traces']:
            b = bocpd_gaussian(tr, hazard_lambda=lam)
            ecp.append(b['ecp'])
            mp0.append(b['mean_p0'])
            mapc.append(b['map_cp_count'])
        out[f'bocpd_ecp{tag}'] = np.array(ecp)
        out[f'bocpd_meanp0{tag}'] = np.array(mp0)
        out[f'bocpd_map{tag}'] = np.array(mapc)

    ar_m, ar_r, ka_m, ka_n = [], [], [], []
    for tr in cell['traces']:
        a = ar_innovation_scores(tr)
        k = kalman_innovation_scores(tr)
        ar_m.append(a['mse_innov'])
        ar_r.append(a['innov_ratio'])
        ka_m.append(k['mse_innov'])
        ka_n.append(k['nis'])
    out['ar2_mse'] = np.array(ar_m)
    out['ar2_ratio'] = np.array(ar_r)
    out['kalman_mse'] = np.array(ka_m)
    out['kalman_nis'] = np.array(ka_n)
    return out, meta


# ── Baselines (CONFIDENCE-oriented: higher = correct; Step-148 recipe) ───────

def baseline_scores(traces):
    fd, valid = prefix_feature_matrix(traces, 10 ** 9, H16)
    tr_valid = [t for t, v in zip(traces, valid) if v]
    scores = {}
    epr_view = -fd['epr']
    for tag, feats in (('lsml16', H16), ('lsml5', GOOD_5)):
        fused, m = lsml_continuous_pipeline(fd, feats, FEATURE_SIGNS)
        fused, flipped = anchor_orient(fused, epr_view)
        scores[tag] = fused
    scores['epr'] = epr_view
    scores['max_ent'] = np.array([-t.max() for t in tr_valid])
    scores['tail_conf'] = np.array([deepconf_tail_conf(t) for t in tr_valid])
    for w in DEEPCONF_WINDOWS:
        scores[f'deepconf_w{w}'] = np.array(
            [deepconf_lowest_group_conf(t, w) for t in tr_valid])
    return scores, valid, epr_view


# ── Smoke test ────────────────────────────────────────────────────────────────

def smoke_test():
    print('=== SMOKE TEST (synthetic 2-regime traces) ===')
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    traces, labels = [], []
    for i in range(80):
        T = int(rng.integers(60, 180))
        x = rng.normal(1.0, 0.3, T)
        if i % 2 == 0:
            k = T // 2
            x[k:] = rng.normal(2.2, 0.5, T - k)
            labels.append(0)
        else:
            labels.append(1)
        traces.append(x)
    labels = np.array(labels)
    cell = {'traces': traces, 'spilled': [None] * len(traces),
            'labels': labels, 'groups': np.arange(len(traces))}
    cand, meta = candidate_scores(cell)
    wrong = 1 - labels  # anomaly scores should rank wrong traces higher
    ok = True
    for tag in ('hmm_occ', 'bocpd_ecp', 'ar2_mse'):
        a = roc_auc_score(wrong, cand[tag])
        print(f'  {tag:10s} AUROC(wrong)={a:.3f}')
        if tag == 'hmm_occ':
            ok = ok and a > 0.9
        else:
            ok = ok and a > 0.6
    ok = ok and not meta['hmm']['degenerate']
    hi, lo = meta['hmm']['means_H']
    ok = ok and hi < lo  # states ordered ascending by H mean
    print(f'  hmm means_H={np.round(meta["hmm"]["means_H"], 2)} '
          f'degenerate={meta["hmm"]["degenerate"]}')
    print(f'SMOKE TEST: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


# ── Main ──────────────────────────────────────────────────────────────────────

def save(res, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(res, f)


def cross_check(cell_res):
    """Compare recomputed full-trace baseline AUROCs to the stored Step-148
    streaming-pilot values (same traces, same recipe — should match)."""
    if not os.path.exists(SP_PKL):
        print('[cross-check] sp_gsm8k.pkl not found — skipped')
        return
    with open(SP_PKL, 'rb') as f:
        sp = pickle.load(f)
    stored = sp['gsm8k/Llama-3.1-8B']['e1_e2']['abs']['full']['auc']
    print('[cross-check] recomputed vs stored (Step 148):')
    worst = 0.0
    for tag in ('lsml5', 'lsml16', 'epr', 'deepconf_w32', 'deepconf_w64'):
        a_new = cell_res['baselines'][tag]['auc'][0]
        a_old = stored[tag][0]
        worst = max(worst, abs(a_new - a_old))
        print(f'  {tag:12s} new={a_new:.4f} stored={a_old:.4f} '
              f'diff={abs(a_new - a_old):.4f}')
    print(f'[cross-check] max |diff| = {worst:.4f} '
          f'({"OK" if worst < 0.02 else "MISMATCH — investigate"})')


def run(args):
    res = {}
    if os.path.exists(args.out) and not args.force:
        with open(args.out, 'rb') as f:
            res = pickle.load(f)
        print(f'[resume] loaded {len(res)} cells from {args.out}')

    for stem, info in CELLS.items():
        name = info['name']
        if name in res:
            print(f'[skip] {name} already computed')
            continue
        path = os.path.join(args.data_dir, stem + '.pkl')
        if not os.path.exists(path):
            print(f'[MISSING] {path}')
            continue
        cell = load_cell(path)
        if cell is None:
            print(f'[skip] {stem}: too few usable traces')
            continue
        y_all = cell['labels']
        print(f'\n=== {name} ({"PRIMARY" if info["primary"] else "SECONDARY / NON-CANONICAL"}) '
              f'n={len(y_all)} frac_correct={y_all.mean():.3f} ===')

        t0 = time.time()
        bscores, valid, _ = baseline_scores(cell['traces'])
        y = y_all[valid]
        groups = cell['groups'][valid]
        # candidates on the SAME valid subset for exact pairing
        sub = {
            'traces': [t for t, v in zip(cell['traces'], valid) if v],
            'spilled': [s for s, v in zip(cell['spilled'], valid) if v],
            'labels': y, 'groups': groups,
        }
        cand, cmeta = candidate_scores(sub)
        epr_anchor = bscores['epr']

        entry = {
            'primary': info['primary'], 'n': int(valid.sum()),
            'frac_correct': float(y.mean()),
            'labels': y.astype(np.int8), 'groups': groups,
            'has_spilled': cmeta['hmm']['obs_dim'] == 2,
            'meta': cmeta, 'baselines': {}, 'candidates': {},
        }
        for tag, sc in bscores.items():
            a, lo, hi = boot_auc(y, sc, n=args.n_boot)
            entry['baselines'][tag] = {
                'auc': (float(a), float(lo), float(hi)),
                'scores': np.asarray(sc, dtype=np.float32)}
        for tag, sc in cand.items():
            conf_raw = -np.asarray(sc, dtype=float)
            a_raw = boot_auc(y, conf_raw, n=args.n_boot)
            conf_anch, flipped = anchor_orient(conf_raw, epr_anchor)
            a_anch = boot_auc(y, conf_anch, n=args.n_boot)
            entry['candidates'][tag] = {
                'auc_raw': tuple(float(v) for v in a_raw),
                'auc_anchored': tuple(float(v) for v in a_anch),
                'auc_oracle': float(max(a_raw[0], 1 - a_raw[0]))
                              if np.isfinite(a_raw[0]) else float('nan'),
                'anchor_flip': bool(flipped),
                'scores': np.asarray(sc, dtype=np.float32),
                'conf_anchored': np.asarray(conf_anch, dtype=np.float32),
            }
        res[name] = entry
        save(res, args.out)
        print(f'  computed in {time.time() - t0:.0f}s')
        if info['primary']:
            cross_check(entry)
    return res


def gate_verdict(res, n_boot):
    print('\n' + '=' * 74)
    print('GATE B — temporal candidates vs DeepConf + full-trace L-SML '
          '(triage; primary cell only)')
    print('=' * 74)
    for name, entry in res.items():
        primary = entry['primary']
        clusters = entry['groups'] if not primary else None
        tag_note = 'PRIMARY' if primary else \
            'SECONDARY / NON-CANONICAL — K=8 correlated traces, cluster bootstrap'
        print(f'\n--- {name} ({tag_note}) ---')
        y = entry['labels']
        dc_tags = [f'deepconf_w{w}' for w in DEEPCONF_WINDOWS]
        best_dc = max(dc_tags, key=lambda t: entry['baselines'][t]['auc'][0])
        dc_auc = entry['baselines'][best_dc]['auc'][0]
        dc_scores = entry['baselines'][best_dc]['scores']
        lsml5_auc = entry['baselines']['lsml5']['auc'][0]
        print(f'  targets: best DeepConf = {best_dc} {dc_auc:.3f}, '
              f'lsml5 full-trace = {lsml5_auc:.3f}')
        print(f'  {"candidate":14s} {"raw":>6s} {"anch":>6s} {"oracle":>6s} '
              f'{"d_vs_dc":>8s} {"95% CI":>18s}  verdict')
        for tag, ce in entry['candidates'].items():
            a_anch = ce['auc_anchored'][0]
            d, lo, hi = paired_boot_delta_auc(
                y, ce['conf_anchored'], dc_scores, n=n_boot,
                clusters=clusters)
            beats = (np.isfinite(a_anch) and a_anch >= dc_auc
                     and a_anch >= lsml5_auc and lo > 0)
            is_primary_cand = tag in PRIMARY_CANDIDATES
            if not primary:
                v = 'secondary-only'
            elif beats:
                v = 'PROMOTE' if is_primary_cand else 'promote(2nd)'
            else:
                v = 'rejected' if is_primary_cand else '-'
            print(f'  {tag:14s} {ce["auc_raw"][0]:>6.3f} {a_anch:>6.3f} '
                  f'{ce["auc_oracle"]:>6.3f} {d * 100:>+7.1f}p '
                  f'[{lo * 100:+6.1f},{hi * 100:+6.1f}]  {v}')
        if primary:
            promoted = [t for t in PRIMARY_CANDIDATES
                        if entry['candidates'][t]['auc_anchored'][0] >= dc_auc
                        and entry['candidates'][t]['auc_anchored'][0] >= lsml5_auc]
            # CI clause checked in the loop print; recompute for the verdict
            final = []
            for t in promoted:
                _, lo, _ = paired_boot_delta_auc(
                    y, entry['candidates'][t]['conf_anchored'], dc_scores,
                    n=n_boot)
                if lo > 0:
                    final.append(t)
            if final:
                print(f'\n  GATE B: PROMOTE {final} to Colab replication')
            else:
                print('\n  GATE B: no primary candidate clears the gate — '
                      'temporal models tried-and-rejected at pilot scale')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir',
                    default=os.path.join(REPO_DIR, 'local_cache', 'raw_traces'))
    ap.add_argument('--out', default=OUT_PKL)
    ap.add_argument('--smoke-test', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()

    if args.smoke_test:
        sys.exit(smoke_test())

    t0 = time.time()
    res = run(args)
    gate_verdict(res, args.n_boot)
    print(f'\ntotal {time.time() - t0:.0f}s — results in {args.out}')


if __name__ == '__main__':
    main()
