"""Add GSP/C4U without the PRM, standalone telemetry, to the Bracha-LR comparison charts (2026-11-11).

Bars (PRMBench PRMScore, Socratic PRMScore, Hard2Verify balanced F1; own label-free flags) and the PRMBench TPR/FPR
scatter (positive = correct step, non-control rows, as PRISM). Existing arms are the values of Omri's chart:
PRM (GAP rule), the 22 Oct fusion (CHAIN), the LR trained on PRM800K labels (R1 rerun on the Mac, numbers not on
this machine) and GSP full. Here we compute the new arm, C4U-T on the standalone model's telemetry, and recompute
the arms we can (PRM GAP, CHAIN, GSP full on PRMBench) as a parity check against the chart values.

    COLLECTION_DATA=C:/Users/omris/TAU/hd_collection_data python bracha_panel.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as Rn  # noqa: E402
E, N, T, RA = Rn.E, Rn.N, Rn.T, Rn.RA
P = N.PRMS
from bracha_panel_values import CHART  # noqa: E402
OUT, CHK = {}, []


def metric(c, flags):
    v = float(E.official_from_counts(E.counts_by_answer(c, flags).sum(0), c.bench))
    return v if c.bench == 'prmbench' else 100 * v


def sa_X(c, prm, bench):
    """The 16 R1 telemetry columns from the standalone bank (same construction as c4u_competitors.build_X)."""
    p = RA.ALL / f'full_sa_{T.PFILE.get(prm, prm)}_{bench}' / 'bank_v3' / 'FEATURES.npz'
    F = np.load(p); names = list(F['names'].astype(str)); V = F['values']; act = F['digit_active'].astype(bool)
    # align by full step position: both the bank and the PRM scores file carry a per-position nonempty mask, and the
    # cell keeps the PRM's nonempty steps (a step can tokenize to empty for one model only)
    S = np.load(Path(E.DATA) / {'socratic': 'socratic', 'hard2verify': 'h2v'}[bench] / f'SCORES_{T.PFILE.get(prm, prm)}.npz')
    rows = []
    for lay in (F, S):
        ne = lay['nonempty'].astype(bool); L = lay['nonempty_lengths'].astype(np.int64)
        st = np.concatenate([[0], np.cumsum(L)]); rid = np.cumsum(ne) - 1          # full position -> row of nonempty steps
        pos = {u: j for j, u in enumerate(lay['uids'].astype(str))}
        rows.append((ne, st, rid, pos))
    (bne, bst, brid, bpos), (pne, pst, _, ppos) = rows
    rows, miss = [], 0
    for u in c.ids:
        a, b = bpos[u], ppos[u]; assert bst[a + 1] - bst[a] == pst[b + 1] - pst[b], u
        for k in range(pst[b + 1] - pst[b]):
            if not pne[pst[b] + k]: continue
            q = bst[a] + k
            if bne[q]: rows.append(brid[q])
            else: rows.append(-1); miss += 1
    rows = np.array(rows); assert len(rows) == c.off[-1] and miss == 0, (len(rows), c.off[-1], miss)
    rv = np.column_stack([V[rows, names.index(n)] for n in T.N16]).astype(float); act = act[rows]
    G = np.column_stack([T.rank_gauss(rv[:, j], act[:, T.DIG.index(n)] if n in T.DIG else None) for j, n in enumerate(T.N16)])
    return np.column_stack([E.answer_z(G[:, j], c.off) for j in range(16)])


def rates(c, f):
    m = c.keep; yv = ~c.err[m]; pv = ~f[m]                  # positive = correct step
    return dict(TPR=100 * float((pv & yv).sum() / yv.sum()), FPR=100 * float((pv & ~yv).sum() / (~yv).sum()))


for i, prm in enumerate(P):
    # PRMBench: standalone cell (same answers/labels as the backbone cell), C4U-T on standalone telemetry
    RA.MODE = 'sa'
    c = E.load('prmbench', f'{prm}__standalone')
    c.arms['PRMPROB'] = 1. - c.extra['reward']; c.arms['PRM'] = c.arms[N.K.prm_arm(c)]
    X, ab, rep, g = T.build_X(c, prm, 'prmbench')
    tel = Rn.c4u_t(c, X[:, 2:])
    OUT[('prmbench', prm)] = dict(value=metric(c, tel['flags']), **rates(c, tel['flags']))
    # backbone cell: parity of PRM GAP / CHAIN / GSP full against the chart, and their TPR/FPR
    cb, ans_eval, Xb, abb, _ = N.load('prmbench', prm, 'R1')
    scale = 'PRM_Q80_PROB' if Rn.C.A.META[f'prmbench/{prm}']['prm_reference_scale'] == '1-reward' else 'PRM_Q80'
    full = N.c4u(cb, Xb, abb)
    for arm, f in (('PRM', cb.flags(scale, 'GAP')), ('OCT22', cb.flags('CHAIN_N16', 'GAP')), ('GSP', full['flags'])):
        v = metric(cb, f); OUT[('prmbench', prm, arm)] = dict(value=v, **rates(cb, f))
        CHK.append(dict(check=f'prmbench {prm} {arm}', got=round(v, 1), chart=CHART['prmbench'][arm][i], ok=abs(round(v, 1) - CHART['prmbench'][arm][i]) < .051))
    for bench in ('socratic', 'hard2verify'):
        c = E.load(bench, prm)
        tel = Rn.c4u_t(c, sa_X(c, prm, bench))
        OUT[(bench, prm)] = dict(value=metric(c, tel['flags']))
        # the chart's PRM row uses the PRM's reference scale (logit PRM or probability PRMPROB, per cell): accept either
        v = {a: round(metric(c, c.flags(a, 'GAP')), 1) for a in ('PRM', 'PRMPROB')}
        CHK.append(dict(check=f'{bench} {prm} PRM GAP (PRM or PRMPROB scale)', got=v, chart=CHART[bench]['PRM'][i],
                        ok=any(abs(x - CHART[bench]['PRM'][i]) < .051 for x in v.values())))
    print(prm, 'done', flush=True)

rows = [dict(benchmark=k[0], prm=k[1], arm=k[2] if len(k) > 2 else 'C4U-T standalone (no PRM)', **v) for k, v in OUT.items()]
E.write_csv(HERE / 'BRACHA_PANEL.csv', rows)
json.dump(CHK, open(HERE / 'BRACHA_PANEL_PARITY.json', 'w'), indent=1)
for r in CHK: print(r)
