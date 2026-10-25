"""Stage 0 parity (plan, 2026-10-24): reproduce the shipped PRM-boosting numbers before any new arm is computed.

(A) Metric parity. For the 15 in-focus cells (PRMBench, Socratic, Hard2Verify x 5 PRMs), recompute within-answer
    AUC and GAP PRMScore of the exported arms (PRM, TELE_N16, MEAN_N16, EQ_N16, HALF_N16, CHAIN_N16) with the
    collection evaluator and compare with results/prm_boost_collection_v1/tables/STATUS_MAIN.csv.
    Pass: |diff| < 1e-9.
(B) Refit parity. Refit TELE_N16 and CHAIN_N16 from the exported N16 channels and the PRM risk, exactly as
    results/prm_boost_ablation5_v1/ablation5.py does (lsml_continuous_guarded, anchor flips, answer-z of the group
    scores). Compare with the exported scores. Pass: max |diff| < 1e-6. A cell whose grouping is flagged degenerate
    is reported, not excused.

Labels are read only by the evaluator, after every score is fixed.

    COLLECTION_DATA=<data root> python stage0_parity.py <out_dir>
"""
import csv, hashlib, json, os, sys
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests')); sys.path.insert(0, str(ROOT / 'results/prm_boost_collection_v1'))
from lever_imports import ensure_spectral_package  # noqa: E402

ensure_spectral_package()
import evaluate_collection as E  # noqa: E402
from collection_parity import h2v_common_uids  # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous_guarded  # noqa: E402

BENCHES = ('prmbench', 'socratic', 'hard2verify')
ARMS = ('PRM', 'TELE_N16', 'MEAN_N16', 'EQ_N16', 'HALF_N16', 'CHAIN_N16')


def prm_arm(c):
    return 'PRM_Q80' if c.bench == 'prmbench' else 'PRM'


def lsml(V, groups=None, anchor=None):
    s, d = lsml_continuous_guarded(*[V[:, j] for j in range(V.shape[1])], groups=groups)
    s = np.asarray(s, float)
    ref = V[:, 0] if anchor is None else anchor
    if np.corrcoef(s, ref)[0, 1] < 0:
        s = -s
    return s, d


def refit_chain(c, V):
    """ablation5.py's TELE and CHAIN construction on the cell's own steps."""
    zp = E.answer_z(c.arms[prm_arm(c)], c.off)
    tele, dt = lsml(V, anchor=V.mean(1))
    vg = np.asarray(dt['virtual_classifiers'], float)
    VC = np.column_stack([zp] + [E.answer_z(vg[:, g], c.off) for g in range(vg.shape[1])])
    chain, dc = lsml(VC, groups=np.arange(VC.shape[1]))
    return tele, chain, VC, dt, dc


def status_rows():
    with open(ROOT / 'results/prm_boost_collection_v1/tables/STATUS_MAIN.csv', newline='') as f:
        return {(r['benchmark'], r['prm'], r['arm']): r for r in csv.DictReader(f) if r['row'] == r['arm']}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    S = status_rows(); common = h2v_common_uids()
    rows, refit, manifest = [], [], {}
    for bench in BENCHES:
        for prm in E.PRMS:
            c = E.load(bench, prm, uids_keep=common) if bench == 'hard2verify' else E.load(bench, prm)
            for arm in ARMS:
                a = prm_arm(c) if arm == 'PRM' else arm
                m, _, _ = E.measure(c, a, 'GAP', fast=True)
                ref = S.get((bench, prm, a))
                rows.append({'benchmark': bench, 'prm': prm, 'arm': a, 'within_auc': m['within_auc'],
                             # the evaluator returns the external official metrics on [0, 1]; STATUS_MAIN shows x100
                             'gap_prmscore': m['official'] * (1. if bench == 'prmbench' else 100.),
                             'status_within_auc': float(ref['within_auc']) if ref else None,
                             'status_gap': float(ref['GAP']) if ref and ref['GAP'] else None})
            # refit on the cell's full answer set (the export was fitted on all answers the PRM could read)
            cf = E.load(bench, prm)
            V = E.channels(cf, 'N16')
            tele, chain, VC, dt, dc = refit_chain(cf, V)
            refit.append({'benchmark': bench, 'prm': prm, 'n_steps': int(cf.off[-1]), 'n_answers': len(cf.ids),
                          'K_tele': int(dt['K']), 'degenerate': bool(dt.get('degenerate', False)),
                          'max_abs_diff_TELE': float(np.max(np.abs(tele - cf.arms['TELE_N16']))),
                          'max_abs_diff_CHAIN': float(np.max(np.abs(chain - cf.arms['CHAIN_N16'])))})
            d = 'prmbench' if bench == 'prmbench' else {'socratic': 'socratic', 'hard2verify': 'h2v'}[bench]
            for p in sorted((E.DATA / d).glob(f'*{E.EXT_NAME.get(prm, prm) if bench != "prmbench" else prm}*.npz')):
                manifest[str(p.relative_to(E.DATA))] = sha256(p)
            print(bench, prm, 'refit CHAIN max|diff| %.2e' % refit[-1]['max_abs_diff_CHAIN'], flush=True)
    for name, rr in (('PARITY_METRICS.csv', rows), ('PARITY_REFIT.csv', refit)):
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, list(rr[0])); w.writeheader(); w.writerows(rr)
    dm = [abs(r['within_auc'] - r['status_within_auc']) for r in rows if r['status_within_auc'] is not None]
    dg = [abs(r['gap_prmscore'] - r['status_gap']) for r in rows if r['status_gap'] is not None]
    summary = {'metric_rows_checked': len(dm), 'max_within_diff': max(dm), 'gap_rows_checked': len(dg),
               'max_gap_diff': max(dg), 'metric_pass': max(dm) < 1e-9 and max(dg) < 1e-9,
               'refit_cells': len(refit), 'refit_max_chain_diff': max(r['max_abs_diff_CHAIN'] for r in refit),
               'refit_pass_cells': sum(r['max_abs_diff_CHAIN'] < 1e-6 for r in refit),
               'refit_fail': [(r['benchmark'], r['prm'], r['max_abs_diff_CHAIN'], r['degenerate']) for r in refit
                              if r['max_abs_diff_CHAIN'] >= 1e-6],
               'data_root': str(E.DATA), 'inputs_sha256': manifest}
    (out / 'PARITY_SUMMARY.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'inputs_sha256'}, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
