"""GSP-T weight variants (2026-11-12, Omri: 'in L-SML the weights are without absolute value; maybe we do that too').
Frozen GSP-T (c4u_telemetry_only_v1.run.c4u_t) has two weight stages that differ from L-SML:
  within group: GSP = |rank-one off-diagonal eigenvector|;     L-SML = the same eigenvector, signed (sml_fuse_signed)
  across groups: GSP = 0.5(logit psi + logit eta) from the binary spectral fit on GAP votes (can saturate at eta=1);
                 L-SML = signed rank-one eigenvector of the continuous group scores
Variants (everything else frozen; R1 columns; count = GSP posterior count or GAP on the new rank):
  W  within signed (sign of the eigenvector resolved so most components are positive, as sml_fuse_signed)
  X  across = signed SML on the continuous group scores (no psi/eta log-odds)
  WX both = L-SML-style weights on GSP's groups
Reference: L-SML fused score s itself (same groups). Output RESULTS_signed.csv."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation as AB  # noqa: E402
import r0 as R0  # noqa: E402
T, E, Rn, N = AB.T, AB.E, AB.Rn, AB.N
from spectral_utils.binary_view_estimation import rank_one_offdiag, spectral_binary  # noqa: E402
from spectral_utils.fusion_utils import sml_fuse_signed  # noqa: E402


def signed(u):
    u = np.asarray(u, float)
    return -u if (u > 0).sum() < len(u) / 2 else u


def gsp_v(c, X, within_signed, cross_sml):
    off = c.off
    s, g, _ = N.lsml(X, X.mean(1))
    sign = np.array([1. if np.corrcoef(X[:, j], s)[0, 1] >= 0 else -1. for j in range(X.shape[1])])
    V = np.column_stack([np.where(AB.S8.gap_flags(c, X[:, j] * sign[j]), 1., -1.) for j in range(X.shape[1])])
    Xo = X * sign; Xo = (Xo - Xo.mean(0)) / Xo.std(0)
    VG, XG, ws = [], [], []
    for k in np.unique(g[g >= 0]):
        idx = np.flatnonzero(g == k); wt = np.ones(len(idx))
        if len(idx) >= 3:
            u = rank_one_offdiag(np.cov(Xo[:, idx].T))
            wt = signed(u) if within_signed else np.abs(u)
        wt = wt / np.abs(wt).sum()
        VG.append(np.where(V[:, idx] @ wt > 0, 1., -1.))
        xs = Xo[:, idx] @ wt; XG.append((xs - xs.mean()) / xs.std()); ws.append(wt)
    VG, XG = np.column_stack(VG), np.column_stack(XG)
    ag = int(np.argmax([np.corrcoef(XG[:, k], s)[0, 1] for k in range(XG.shape[1])]))
    sp = spectral_binary(VG, anchor=ag)
    P = N.posterior(VG, sp['psi'], sp['eta'], float(np.clip(sp['prevalence'], .01, .99)))
    if cross_sml:
        rank, cw = sml_fuse_signed(*[XG[:, k] for k in range(XG.shape[1])])
        if np.corrcoef(rank, s)[0, 1] < 0: rank, cw = -rank, -cw
    else:
        cw = N.lw(sp['psi'], sp['eta']); rank = XG @ cw
    k = np.rint(np.add.reduceat(P, off[:-1])).astype(int)
    return dict(rank=rank, flags=E.topk_flags(rank, off, k), cross=cw, within=ws, s=s)


VARIANTS = {'GSP-T (frozen)': (False, False), 'GSP-T W within signed': (True, False),
            'GSP-T X cross SML': (False, True), 'GSP-T WX both signed': (True, True)}


def run_cell(c, X, bench, prm, rows, pos=None):
    for name, (w, x) in VARIANTS.items():
        tel = gsp_v(c, X, w, x)
        if bench == 'processbench':
            for k, mk, v in R0.pb_extra(c, tel, name, pos):
                if 'CP' not in k: rows.append(dict(bench=bench, prm=prm, arm=k, metric=mk, value=v))
        else:
            for k, f in ((f'{name} own count', tel['flags']), (f'{name} + GAP', AB.gap(c, tel['rank']))):
                for mk, mv in AB.score(c, bench, tel['rank'], f).items():
                    rows.append(dict(bench=bench, prm=prm, arm=k, metric=mk, value=mv))
        rows.append(dict(bench=bench, prm=prm, arm=name, metric='cross weights', value=' '.join(f'{v:.2f}' for v in tel['cross'])))
        rows.append(dict(bench=bench, prm=prm, arm=name, metric='negative within weights',
                         value=int(sum((np.asarray(wt) < 0).sum() for wt in tel['within']))))
    # reference: L-SML score s with GAP
    s = tel['s']
    if bench == 'processbench':
        for k, mk, v in R0.pb_extra(c, dict(rank=s, flags=AB.gap(c, s)), 'L-SML', pos):
            if 'CP' not in k and 'own count' not in k: rows.append(dict(bench=bench, prm=prm, arm=k, metric=mk, value=v))
    else:
        for mk, mv in AB.score(c, bench, s, AB.gap(c, s)).items():
            rows.append(dict(bench=bench, prm=prm, arm='L-SML + GAP', metric=mk, value=mv))


def main():
    rows = []
    for bench in AB.BENCHES:
        for prm in AB.PRMS:
            c, X, ab = AB.load_cell(bench, prm)
            run_cell(c, X[:, 2:], bench, prm, rows)
            print(bench, prm, 'done', flush=True); del c
    import runpy
    g = runpy.run_path(str(HERE / 'qwen3_8b_pb.py'))
    run_cell(g['c'], g['X'], 'processbench', 'qwen3_8b_mtg', rows, g['pos'])
    E.write_csv(HERE / 'RESULTS_signed.csv', rows)


if __name__ == '__main__':
    main()
