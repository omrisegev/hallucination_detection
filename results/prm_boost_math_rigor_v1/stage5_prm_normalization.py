"""Normalization of the PRM view inside CHAIN_N16 (review W6; FRAMING_THEORY.md P1). Development comparison.

CHAIN's cross stage gets [PRM view, answer_z(group scores)]; only the PRM view changes:
  answer_z  (shipped)  per-answer mean 0, sd 1
  center               per-answer mean removed, then one global sd (no per-answer scaling)
  global_z             global mean/sd only (answer level kept)
Telemetry stays as exported (answer-z channels, answer-z group scores). Same lsml cross stage and anchor flip as
ablation5. Reported: within-answer AUC and GAP PRMScore, with the collection's paired bootstrap against the shipped
arm. Not a pre-registered candidate.

    COLLECTION_DATA=<data root> python stage5_prm_normalization.py <out_dir>
"""
import csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
from spectral_utils.fusion_theory import within_answer_center  # noqa: E402

E = K.E


def prm_views(c):
    risk = c.arms[K.prm_arm(c)].astype(float)
    cen = within_answer_center(risk[:, None], c.off)[:, 0]
    return {'answer_z': E.answer_z(risk, c.off), 'center': cen / cen.std(),
            'global_z': (risk - risk.mean()) / risk.std()}


def main(out):
    out.mkdir(parents=True, exist_ok=True); rows = []
    for bench in ('prmbench', 'socratic'):
        for prm in E.PRMS:
            c, VC, w, fused, info = K.chain_inputs(bench, prm)
            assert np.allclose(VC[:, 0], E.answer_z(c.arms[K.prm_arm(c)], c.off))
            for name, v in prm_views(c).items():
                X = VC.copy(); X[:, 0] = v
                s, d, _ = K.lsml(X, groups=np.arange(X.shape[1]))
                c.arms[f'NORM_{name}'] = s
                m, _, _ = E.measure(c, f'NORM_{name}', 'GAP', fast=True)
                cw = np.abs(np.asarray(d['cross_weights'], float))
                row = {'benchmark': bench, 'prm': prm, 'prm_view': name, 'within_auc': m['within_auc'],
                       'gap_prmscore': m['official'] * (1. if bench == 'prmbench' else 100.),
                       'prm_cross_share': float(cw[0] / cw.sum())}
                if name != 'answer_z':
                    for ct in E.contrast(c, (f'NORM_{name}', 'GAP'), ('NORM_answer_z', 'GAP')):
                        sc = 1. if (bench == 'prmbench' or ct['metric'] == 'within_auc') else 100.
                        row[f"d_{ct['metric']}"] = ct['delta'] * sc
                        row[f"d_{ct['metric']}_ci"] = [round(ct['ci_low'] * sc, 4), round(ct['ci_high'] * sc, 4)]
                rows.append(row)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    with open(out / 'PRM_NORMALIZATION.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r))); wr.writeheader(); wr.writerows(rows)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
