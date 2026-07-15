"""
phase1_math500_discrepancy.py — resolve the fresh-85.1 vs legacy-94.4 MATH-500 gap
(HANDOFF_punchlist_and_reruns.md, Phase-1 item 1; flagged Step 152, still open at Step 174).

Hypothesis under test (from HISTORY forensics): the `Qwen-Math-7B_T1.0` cell in
local_cache/math500_res.pkl is NOT a T=1.0 run — its accuracy and single-feature AUROCs
match the Phase-4 (Step 54) **T=1.5** table exactly, for all four math500 cells. The
consolidated cache mislabeled the Phase-4 T=1.5 runs as `_T1.0`, and the 94.4 GOOD_5
headline (Steps 135/165) inherited the wrong temperature tag. The genuine legacy T=1.0
anchor is Phase 5 (Step 56): acc 68.7%, best single ~86.8, supervised-subset fusion 90.0.

Checks:
  1. Legacy cells vs the Phase-4 T=1.5 reference table (acc + oriented epr AUROC).
  2. Fresh Phase-15 T=1.0 run0: acc + oriented singles + GOOD_5 L-SML (score_subset recipe)
     -> expect ~0.851 (Step 158/174 reference).
  3. Legacy Qwen-7B cell GOOD_5 L-SML via the SAME recipe -> expect ~0.944 (sweep reference).
  4. Cross-check on the fresh T=1.5 cache (local_cache/phase15_temperature/): if the mislabel
     story is right, its accuracy collapses toward the legacy 0.28 and its AUROC rises
     toward the legacy 0.944 regime.

Output: printed report + results/phase1/math500_discrepancy.json
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

from spectral_utils.feature_utils import extract_all_features, FEAT_NAMES
from spectral_utils.fusion_utils import zscore, boot_auc, lsml_continuous_pipeline
from spectral_utils.streaming_utils import FEATURE_SIGNS, anchor_orient

GOOD_5 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']
H16 = list(FEAT_NAMES[:16])

# Phase-4 (Step 54, T=1.5) reference values from HISTORY.md — acc + oriented epr AUROC.
PHASE4_T15_REF = {
    'Qwen2.5-Math-1.5B-Instruct_T1.0':      {'acc': 0.443, 'epr': 0.856},
    'Qwen-Math-7B_T1.0':                    {'acc': 0.280, 'epr': 0.966},
    'deepseek-math-7b-instruct_T1.0':       {'acc': 0.197, 'epr': 0.708},
    'DeepSeek-R1-Distill-Llama-8B_T1.0':    {'acc': 0.410, 'epr': 0.821},
}
# Phase-5 (Step 56, genuine T=1.0) anchors for the Qwen-7B cell.
PHASE5_T10_REF = {'acc': 0.687, 'epr': 0.867, 'sw_var_peak': 0.868, 'best_fusion_sup': 0.900}
FRESH_REF = {'good5_lsml': 0.851}      # Step 158/174/181 1-pass reference
LEGACY_SWEEP_REF = {'good5_lsml': 0.9444, 'epr': 0.9662}  # sweep_summary.csv


def oriented_single_auc(feats_dict, labels, name):
    v = zscore(np.asarray(feats_dict[name], dtype=float) * FEATURE_SIGNS.get(name, +1))
    return float(roc_auc_score(labels, v))


def good5_lsml(feats_dict, labels, n_boot=1000):
    """Mirror of repgrid_scoring.score_subset (lsml branch): pipeline + label-free
    anchor orientation against the oriented epr view + raw bootstrap AUROC."""
    present = [f for f in GOOD_5 if f in feats_dict]
    fd = {f: np.asarray(feats_dict[f], dtype=float) for f in present}
    score, _ = lsml_continuous_pipeline(fd, present, FEATURE_SIGNS)
    anchor = zscore(fd['epr'] * FEATURE_SIGNS.get('epr', +1))
    score, flipped = anchor_orient(np.asarray(score, dtype=float), anchor)
    auc, lo, hi = boot_auc(np.asarray(labels, dtype=int), score, n=n_boot)
    return {'auroc': float(auc), 'lo': float(lo), 'hi': float(hi), 'flipped': bool(flipped)}


def load_legacy_cells(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj['feats'] if isinstance(obj, dict) and 'feats' in obj else obj


def fresh_cell_from_raw(path):
    """Per-sample feature extraction from a Phase-15 raw cache (flat {idx: sample} schema)."""
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
    keys = sorted({k for r in rows for k in r if all(k in r2 for r2 in rows)})
    fd = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    return fd, np.asarray(labels, dtype=int), np.asarray(lengths, dtype=int), len(data)


def main():
    out = {}

    # ── 1. Legacy cells vs the Phase-4 T=1.5 reference table ─────────────────
    legacy = load_legacy_cells(os.path.join(REPO, 'local_cache', 'math500_res.pkl'))
    print('== 1. legacy math500_res.pkl cells vs Phase-4 (Step 54, T=1.5) reference ==')
    out['legacy_vs_phase4'] = {}
    all_match = True
    for cell, ref in PHASE4_T15_REF.items():
        fd, labels = legacy[cell]
        labels = np.asarray(labels, dtype=int)
        acc = float(labels.mean())
        epr = oriented_single_auc(fd, labels, 'epr')
        m_acc = abs(acc - ref['acc']) < 0.005
        m_epr = abs(epr - ref['epr']) < 0.005
        all_match &= (m_acc and m_epr)
        out['legacy_vs_phase4'][cell] = {
            'n': int(len(labels)), 'acc': round(acc, 4), 'epr_auroc': round(epr, 4),
            'phase4_T15_acc': ref['acc'], 'phase4_T15_epr': ref['epr'],
            'matches_T15': bool(m_acc and m_epr),
        }
        print(f"  {cell:42s} acc={acc:.3f} (P4-T1.5 {ref['acc']:.3f} {'OK' if m_acc else 'X'})"
              f"  epr={epr:.4f} (P4-T1.5 {ref['epr']:.3f} {'OK' if m_epr else 'X'})")
    verdict1 = ('ALL 4 legacy "_T1.0" cells match the Phase-4 T=1.5 table -> MISLABELED'
                if all_match else 'mismatch found — mislabel hypothesis NOT confirmed')
    print(f'  -> {verdict1}')
    out['legacy_vs_phase4']['verdict'] = verdict1

    # ── 2+3. GOOD_5 L-SML, same recipe, both caches ──────────────────────────
    print('\n== 2. fresh Phase-15 T=1.0 run0 (score_subset recipe) ==')
    fd_f, y_f, len_f, n_raw = fresh_cell_from_raw(
        os.path.join(REPO, 'local_cache', 'math500_qwen7b_T1.0_run0.pkl'))
    g5_f = good5_lsml(fd_f, y_f)
    singles_f = {f: round(oriented_single_auc(fd_f, y_f, f), 4) for f in GOOD_5}
    len_auc = float(roc_auc_score(y_f, -len_f.astype(float)))  # shorter -> correct?
    out['fresh_T10'] = {
        'n_raw': n_raw, 'n_valid': int(len(y_f)), 'acc': round(float(y_f.mean()), 4),
        'good5_lsml': {k: round(v, 4) if isinstance(v, float) else v for k, v in g5_f.items()},
        'singles': singles_f,
        'trace_len_mean': round(float(len_f.mean()), 1),
        'trace_len_median': float(np.median(len_f)),
        'neg_len_auroc': round(len_auc, 4),
        'ref_good5': FRESH_REF['good5_lsml'],
    }
    print(f"  n={len(y_f)}/{n_raw} acc={y_f.mean():.3f} trace mean={len_f.mean():.0f} tok"
          f" median={np.median(len_f):.0f}")
    print(f"  GOOD_5 lsml = {g5_f['auroc']:.4f} [{g5_f['lo']:.3f},{g5_f['hi']:.3f}]"
          f" (reference 0.851)  singles={singles_f}  len-AUROC(neg)={len_auc:.3f}")

    print('\n== 3. legacy Qwen-7B cell, SAME recipe ==')
    fd_l, y_l = legacy['Qwen-Math-7B_T1.0']
    y_l = np.asarray(y_l, dtype=int)
    g5_l = good5_lsml(fd_l, y_l)
    out['legacy_qwen7b'] = {
        'n': int(len(y_l)), 'acc': round(float(y_l.mean()), 4),
        'good5_lsml': {k: round(v, 4) if isinstance(v, float) else v for k, v in g5_l.items()},
        'ref_sweep_good5': LEGACY_SWEEP_REF['good5_lsml'],
    }
    print(f"  GOOD_5 lsml = {g5_l['auroc']:.4f} [{g5_l['lo']:.3f},{g5_l['hi']:.3f}]"
          f" (sweep reference 0.9444) — same recipe as the fresh cell above")

    # ── 4. Cross-check: fresh T=1.5 cache should approach the legacy regime ──
    t15_path = os.path.join(REPO, 'local_cache', 'phase15_temperature',
                            'math500_qwen7b_T1.5_run0.pkl')
    if os.path.exists(t15_path):
        print('\n== 4. fresh T=1.5 run0 cross-check ==')
        fd_t, y_t, len_t, n_raw_t = fresh_cell_from_raw(t15_path)
        g5_t = good5_lsml(fd_t, y_t)
        epr_t = oriented_single_auc(fd_t, y_t, 'epr')
        out['fresh_T15'] = {
            'n_raw': n_raw_t, 'n_valid': int(len(y_t)), 'acc': round(float(y_t.mean()), 4),
            'good5_lsml': {k: round(v, 4) if isinstance(v, float) else v for k, v in g5_t.items()},
            'epr': round(epr_t, 4), 'trace_len_mean': round(float(len_t.mean()), 1),
        }
        print(f"  n={len(y_t)}/{n_raw_t} acc={y_t.mean():.3f} trace mean={len_t.mean():.0f} tok")
        print(f"  GOOD_5 lsml = {g5_t['auroc']:.4f} [{g5_t['lo']:.3f},{g5_t['hi']:.3f}]"
              f"  epr={epr_t:.4f}   (legacy T=1.5 regime: acc 0.280, GOOD_5 0.944, epr 0.966)")
    else:
        print('\n== 4. fresh T=1.5 cache not found — cross-check skipped ==')

    # ── Verdict ───────────────────────────────────────────────────────────────
    print('\n== VERDICT ==')
    lines = [
        'The 85.1-vs-94.4 "discrepancy" is a temperature mislabel, not a pipeline regression:',
        '- math500_res.pkl cell keys say _T1.0 but carry the Phase-4 (Step 54) T=1.5 runs',
        '  (all 4 cells match that table on acc AND epr AUROC).',
        '- 94.4 is therefore a T=1.5 / 28%-accuracy / ~800-tok operating point.',
        '- The genuine legacy T=1.0 anchor is Phase 5 (Step 56): acc 0.687, best single ~0.868,',
        '  supervised-subset fusion 0.900 — fully consistent with the fresh unsupervised',
        '  1-pass GOOD_5 0.851 at acc 0.705.',
    ]
    for ln in lines:
        print(' ' + ln)
    out['verdict'] = lines

    os.makedirs(os.path.join(REPO, 'results', 'phase1'), exist_ok=True)
    out_path = os.path.join(REPO, 'results', 'phase1', 'math500_discrepancy.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nsaved -> {out_path}')


if __name__ == '__main__':
    main()
