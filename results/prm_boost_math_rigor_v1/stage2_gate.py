"""Stage 2 (plan, 2026-10-24): the pre-registered label-free gate on the 15 in-focus cells (GATE_PREREG.md).

For each cell, the gate sees only CHAIN_N16's cross-stage views VC and CHAIN's effective weights w (common.py),
residualized within answers on the covariate basis (cubic relative position, plus log step tokens on PRMBench). It
outputs fuse / prm / abstain via spectral_utils.fusion_theory.gate_decision with the estimator fixed in the
pre-registration. Labels enter only afterwards, to score the choice against the observed sign of
within-AUC(CHAIN_N16) - within-AUC(PRM).

Also reported, with the same scoring:
  - the review's gate: rank-one 'complete'-scale loadings of the whole off-diagonal covariance, G on the point
    estimate (fuse if G > 0);
  - "always fuse" and "always PRM";
  - the PERM negative control on PRMBench: telemetry moved to another answer with the same step count
    (ablation5.py's construction, parity-checked). The gate should not choose fuse.

    COLLECTION_DATA=<data root> python stage2_gate.py <out_dir>
"""
import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
from spectral_utils.fusion_theory import covariate_basis, gate_decision, gate_statistic, within_answer_residualize  # noqa: E402
from spectral_utils.fusion_utils import _rank1_masked  # noqa: E402

PREREG = json.loads((Path(__file__).resolve().parent / 'GATE_PREREG.json').read_text())
METHOD, B, LEVEL, SEED = PREREG['method'], PREREG['B'], PREREG['level'], PREREG['seed']


def reviewer_gate(Xe, w):
    C = np.cov(Xe.T)
    u = np.abs(_rank1_masked(C, np.eye(len(C), dtype=bool), 'complete'))
    if np.any(u ** 2 >= np.diag(C)):
        return 'abstain', None
    G = gate_statistic(C, u, w)['G']
    return ('fuse' if G > 0 else 'prm'), float(G)


def run_cell(bench, prm, permute=False):
    c, VC, w, fused, info = K.chain_inputs(bench, prm, permute=permute)
    pos, loglen = K.covariates(c)
    Xe = within_answer_residualize(VC, covariate_basis(pos, loglen), c.off)
    dec = gate_decision(Xe, c.off, w, method=METHOD, B=B, level=LEVEL, seed=SEED)
    rev, Grev = reviewer_gate(Xe, w)
    _, ans_mask = K.eval_rows(c, bench)
    truth = K.within_auc(c, fused, ans_mask) - K.within_auc(c, VC[:, 0], ans_mask)
    action = 'fuse' if dec['decision'] == 'fuse' else 'prm'
    return {'benchmark': bench, 'prm': prm, 'permuted_telemetry': permute, 'K': info['K'],
            'truth_chain_minus_prm': truth, 'gate_decision': dec['decision'], 'gate_action': action,
            'gate_correct': (action == 'fuse') == (truth > 0), 'G': dec.get('G'), 'G_lo': dec.get('G_lo'),
            'G_hi': dec.get('G_hi'), 'G_fisher': dec.get('G_fisher'), 'n_boot_identified': dec.get('n_boot_identified'),
            'n_negative_anchor_cov': dec.get('n_negative_anchor_cov'), 'reason': dec.get('reason'),
            'u': None if dec.get('u') is None else np.round(dec['u'], 5).tolist(),
            'w_chain': np.round(w, 5).tolist(),
            'w_fisher': None if dec.get('w_fisher') is None else np.round(dec['w_fisher'], 5).tolist(),
            'reviewer_decision': rev, 'reviewer_G': Grev, 'reviewer_correct': (rev == 'fuse') == (truth > 0)}


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bench in K.BENCHES:
        for prm in K.E.PRMS:
            rows.append(run_cell(bench, prm)); r = rows[-1]
            print(bench, prm, 'truth %+.3f | gate %s (G %s) %s | reviewer %s' % (
                r['truth_chain_minus_prm'], r['gate_decision'], None if r['G'] is None else round(r['G'], 4),
                'OK' if r['gate_correct'] else 'WRONG', r['reviewer_decision']), flush=True)
    perm = []
    for prm in K.E.PRMS:
        perm.append(run_cell('prmbench', prm, permute=True)); r = perm[-1]
        print('PERM prmbench', prm, 'truth %+.3f | gate %s' % (r['truth_chain_minus_prm'], r['gate_decision']), flush=True)
    with open(out / 'GATE_RESULTS.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows + perm)
    prim = [r for r in rows if r['benchmark'] in ('prmbench', 'socratic')]
    h2v = [r for r in rows if r['benchmark'] == 'hard2verify']
    cnt = lambda rr, k: int(sum(r[k] for r in rr))
    summ = {'method': METHOD, 'B': B, 'level': LEVEL,
            'primary_10_cells_gate_correct': cnt(prim, 'gate_correct'),
            'primary_always_fuse_correct': int(sum(r['truth_chain_minus_prm'] > 0 for r in prim)),
            'primary_reviewer_correct': cnt(prim, 'reviewer_correct'),
            'h2v_gate_correct': cnt(h2v, 'gate_correct'), 'h2v_abstain': int(sum(r['gate_decision'] == 'abstain' for r in h2v)),
            'all15_gate_correct': cnt(rows, 'gate_correct'), 'all15_reviewer_correct': cnt(rows, 'reviewer_correct'),
            'all15_always_fuse_correct': int(sum(r['truth_chain_minus_prm'] > 0 for r in rows)),
            'perm_control_not_fuse': int(sum(r['gate_decision'] != 'fuse' for r in perm)),
            'decisions': {f"{r['benchmark']}/{r['prm']}": r['gate_decision'] for r in rows}}
    summ['primary_verdict'] = ('pass' if summ['primary_10_cells_gate_correct'] >= PREREG['primary_threshold'] else 'partial' if summ['primary_10_cells_gate_correct'] >= PREREG['partial_threshold'] else 'fail')
    summ['perm_pass'] = summ['perm_control_not_fuse'] >= PREREG['perm_threshold']
    (out / 'GATE_SUMMARY.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
