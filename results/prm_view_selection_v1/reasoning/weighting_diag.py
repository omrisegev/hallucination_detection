"""POST-HOC diagnostic (not in the pre-declaration; labels used; written after EVAL8_ARMS.csv was read).

Question: in the position-neutral representation (POS), is the loss of [PRM + S_R] to the PRM on strong-PRM Socratic
cells a problem of the channels or of the label-free weights?

Following Jaffe et al. (2016, Sec. 5), the label-free DS-EM weights are compared with 'Oracle-CI': the same linear
log-odds rule (their Eq. 5) given the TRUE sensitivities and specificities of each vote. Also reported: the label-free
DS-EM estimates against the truth, and PRMScore along the PRM's share a (fused = a*Z_PRM + (1-a)*Z_T), where Z_T is the
answer-z of the telemetry-only DS-EM score of the same set.

    python weighting_diag.py   # -> WEIGHTING_DIAG.csv, WEIGHTING_CURVE.csv
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval8 as V  # noqa: E402
E, A, S1, S2 = V.E, V.A, V.S1, V.S2


def true_rates(v, err, ok):
    psi = np.clip((v[err] > 0).mean(0), 1e-6, 1 - 1e-6); eta = np.clip((v[ok] < 0).mean(0), 1e-6, 1 - 1e-6)
    return psi, eta


def main():
    rule = json.loads((HERE / 'RULE_POS.json').read_text()); sets = {'S_R': rule['S_R'], 'C3': V.C3}
    rows, curve = [], []
    for b, p in V.CELLS:
        c = E.load(b, p)
        c1 = S1.CellData.from_light(V.POSC / f's1_{V.key(b, p)}.npz')
        c2 = S2.TeleCell.load(V.POSC / f's2_{V.key(b, p)}.npz')
        err = c1._nyv_m & c1.pool; ok = c1._yv_m & c1.pool
        prm_ps = float(c1.ps_prm)
        for nm, Sn in sets.items():
            S = [V.NAMES.index(n) for n in Sn]; idx = np.r_[0, np.asarray(S) + 1]
            vp = c1.V[c1.pool][:, idx]
            ds = A.ds_em_votes(vp, anchor=0)
            psi_t, eta_t = true_rates(c1.V[:, idx], err, ok)
            w_or = np.clip(np.log(psi_t * eta_t / ((1 - psi_t) * (1 - eta_t))), 0, None)
            fused_ds = c1.Z[:, idx] @ ds['w']; fused_or = c1.Z[:, idx] @ w_or
            m_ds = V.metrics(c, fused_ds, b); m_or = V.metrics(c, fused_or, b)
            for j, view in enumerate(['PRM'] + Sn):
                rows.append(dict(cell=V.key(b, p), set=nm, view=view, est_psi=float(ds['psi'][j]), est_eta=float(ds['eta'][j]),
                                 true_psi=float(psi_t[j]), true_eta=float(eta_t[j]),
                                 est_bal=float((ds['psi'][j] + ds['eta'][j]) / 2), true_bal=float((psi_t[j] + eta_t[j]) / 2),
                                 share_ds=float(ds['w'][j] / ds['w'].sum()), share_oracle=float(w_or[j] / w_or.sum()),
                                 est_prevalence=ds['prevalence'], true_prevalence=float(err.sum() / (err.sum() + ok.sum())),
                                 prm_alone=prm_ps, ps_ds=m_ds['PRMScore'], ps_oracle_ci=m_or['PRMScore'],
                                 auc_ds=m_ds['within_auc'], auc_oracle_ci=m_or['within_auc']))
            # PRMScore along the PRM's share, telemetry composite = the set's telemetry-only DS-EM score in POS
            o = V.eval_ci(c2, S, with_prm=False)
            if o['valid']:
                zt = E.answer_z(o['fused'], c.off); zp = c1.Z[:, 0]
                for a in np.round(np.linspace(0, 1, 21), 2):
                    m = V.metrics(c, a * zp + (1 - a) * zt, b)
                    curve.append(dict(cell=V.key(b, p), set=nm, prm_share=float(a), PRMScore=m['PRMScore'], within_auc=m['within_auc'],
                                      prm_alone=prm_ps))
        print('done', b, p, flush=True); del c
    E.write_csv(HERE / 'WEIGHTING_DIAG.csv', rows); E.write_csv(HERE / 'WEIGHTING_CURVE.csv', curve)


if __name__ == '__main__':
    main()
