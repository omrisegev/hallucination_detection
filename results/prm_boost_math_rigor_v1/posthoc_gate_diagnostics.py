"""POST HOC diagnostics of the failed pre-registered gate (GATE_PREREG.md). Written after stage2_gate.py ran.

Nothing here changes the pre-registered verdict, and nothing here is a new method.
  1. Third-order sign check on the 10 primary cells. Under the model (FRAMING_THEORY.md P5 c), every telemetry view
     should have C_0k > 0, k(0,0,k) > 0 and k(0,k,k) > 0 when p < 1/2. Counted on the gate's own estimation matrix
     (answer-z views, residualized on the covariate basis) and on a centered-only version (no per-answer SD
     scaling), to see whether the scaling (review W6) is what breaks it.
  2. Numbers for NEGATIVE_RESULT.md from stage1/ and stage2/: mean within-AUC of the arm each rule picks, the
     win/loss record of the gate against "always fuse", and a paired bootstrap over cells (10,000 draws, seed 0).

    COLLECTION_DATA=<data root> python posthoc_gate_diagnostics.py <out_dir>
"""
import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
from spectral_utils.fusion_theory import covariate_basis, within_answer_center, within_answer_residualize  # noqa: E402

HERE = Path(__file__).resolve().parent


def sign_counts(M, Z, off):
    X = within_answer_residualize(M, Z, off); X = X - X.mean(0)
    c = (X[:, [0]] * X[:, 1:]).mean(0)
    k00 = (X[:, [0]] ** 2 * X[:, 1:]).mean(0); k0k = (X[:, [0]] * X[:, 1:] ** 2).mean(0)
    return int(((c > 0) & (k00 > 0) & (k0k > 0)).sum()), len(c)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bench in ('prmbench', 'socratic'):
        for prm in K.E.PRMS:
            c, VC, w, fused, info = K.chain_inputs(bench, prm)
            pos, loglen = K.covariates(c); Z = covariate_basis(pos, loglen)
            V = K.E.channels(c, 'N16'); _, dt, _ = K.lsml(V, anchor=V.mean(1))
            vg = np.asarray(dt['virtual_classifiers'], float)
            raw = np.column_stack([c.arms[K.prm_arm(c)]] + [vg[:, g] for g in range(vg.shape[1])])
            a_ok, n = sign_counts(VC, Z, c.off)
            b_ok, _ = sign_counts(within_answer_center(raw, c.off), Z, c.off)
            rows.append({'benchmark': bench, 'prm': prm, 'n_views': n, 'model_sign_ok_answer_z': a_ok,
                         'model_sign_ok_center_only': b_ok})
            print(rows[-1], flush=True)
    with open(out / 'POSTHOC_CUMULANT_SIGNS.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)

    S1 = {(r['benchmark'], r['prm']): r for r in csv.DictReader(open(HERE / 'stage1/STAGE1_DIAGNOSTICS.csv'))}
    G = [r for r in csv.DictReader(open(HERE / 'stage2/GATE_RESULTS.csv')) if r['permuted_telemetry'] == 'False']
    prim = [r for r in G if r['benchmark'] in ('prmbench', 'socratic')]
    auc = lambda r, arm: float(S1[(r['benchmark'], r['prm'])][arm])
    pick = lambda r, rule: auc(r, 'auc_chain') if (rule == 'fuse' or (rule == 'gate' and r['gate_action'] == 'fuse')
                                                   or (rule == 'reviewer' and r['reviewer_decision'] == 'fuse')) else auc(r, 'auc_prm')
    table = {}
    for rule in ('fuse', 'prm', 'gate', 'reviewer'):
        for bench in ('prmbench', 'socratic'):
            table[f'{rule}/{bench}_mean_within_auc'] = float(np.mean([pick(r, rule) for r in prim if r['benchmark'] == bench]))
        table[f'{rule}/primary_correct'] = int(sum(((pick(r, rule) == auc(r, 'auc_chain')) == (float(r['truth_chain_minus_prm']) > 0)) for r in prim))
    d_auc = np.array([pick(r, 'gate') - pick(r, 'fuse') for r in prim])
    d_cor = np.array([int(r['gate_correct'] == 'True') - int(float(r['truth_chain_minus_prm']) > 0) for r in prim])
    rng = np.random.default_rng(0); idx = rng.integers(0, len(prim), (10_000, len(prim)))
    ci = lambda x: [float(np.quantile(x[idx].mean(1), .025)), float(np.quantile(x[idx].mean(1), .975))]
    table.update({'gate_vs_fuse_wins': int((d_auc > 0).sum()), 'gate_vs_fuse_losses': int((d_auc < 0).sum()),
                  'gate_minus_fuse_mean_within_auc': float(d_auc.mean()), 'gate_minus_fuse_mean_within_auc_ci95': ci(d_auc),
                  'gate_minus_fuse_correct_share': float(d_cor.mean()), 'gate_minus_fuse_correct_share_ci95': ci(d_cor),
                  'cumulant_sign_ok_answer_z': f"{sum(r['model_sign_ok_answer_z'] for r in rows)}/{sum(r['n_views'] for r in rows)}",
                  'cumulant_sign_ok_center_only': f"{sum(r['model_sign_ok_center_only'] for r in rows)}/{sum(r['n_views'] for r in rows)}",
                  'note': 'post hoc; cells are the 10 primary cells; bootstrap resamples cells'})
    (out / 'POSTHOC_NUMBERS.json').write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
