"""LABEL-FREE rule C fit on Hard2Verify / DeltaBench (2026-11-22). Port of agentS2/pseudo.py (frozen 2026-11-19), rule C only.

Unchanged from pseudo.py (imported, not copied): prefix_pool, rez11, pipeline (R0 -> method-of-record L-SML linear map ->
GAP flags; sg = RankGauss(pool) @ Wg), the beta grid 0, .01, ..., 1, the tie rule (largest beta), the quantile
definition tau = (1 - beta) quantile of sg over GAP-flagged pseudo-clean steps, the sg orientation (sign corr(sg, s0)),
pseudo pairs (parent steps 0..k vs child steps 0..k, local error step k >= 1).
Changes (benchmark adapters only):
  - pool: the FULL C pool of the benchmark (no dev/test split; every fit is label-free), loaded by bench_adapters
    WITHOUT a gold file. load_dev's fit steps are repeated without its label-using metric block.
  - children: STEPS_<bench>_<V>.npz + SIDECAR_<bench>_<V>.jsonl (k is 1-based in the sidecar -> local k - 1).
  - pseudo metric: hard2verify = PRMScore with keep = ones (the pseudo.py non-ProcessBench branch, unchanged function);
    deltabench = DeltaBench official section macro-F1 (deltabench_official.per_sample) with the pseudo error section
    k + 1 as target; pseudo-clean prefixes (empty target) are scored 1 if no section is flagged, else 0 ('empty=1'):
    the official script drops empty-target samples, which would remove every pseudo-clean prefix (pre-declared deviation).
No gold label, no answer correctness and no labelled metric is read in this file.
    HD_ROOT=$WS/hd python pseudo_ext.py BENCH STEPS_DIR SIDECAR_DIR GROUPS_JSON OUT_DIR V3[,V4] [C|CR]
"""
import json, sys
from pathlib import Path
import numpy as np

WS = 'C:/Users/omris/TAU/rule_c_ws/run'
sys.path.insert(0, f'{WS}/agentX1'); sys.path.insert(0, f'{WS}/agentS2'); sys.path.insert(0, str(Path(__file__).parent))
import pseudo as PS                       # frozen agentS2 code (path-replaced working copy)
from common import RA, L, S, D            # noqa: E402
import bench_adapters as A               # noqa: E402
import deltabench_official as O          # noqa: E402

ONE = np.ones(16)
bench, sdir, vdir, gpath, outd, VS = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5]), sys.argv[6].split(',')
CNAME = sys.argv[7] if len(sys.argv) > 7 else 'C'          # 'CR' = C re-run in the children's container (amendment 2026-11-22)
outd.mkdir(parents=True, exist_ok=True)
G = json.load(open(gpath))

# ---- full C pool fit (load_dev without labels)
Cd = A.load_c_pool(sdir / f'STEPS_{bench}_{CNAME}.npz', bench, groups=G)
rawC = D.readout(Cd)
X0 = L.r0_cols(rawC, Cd.act, Cd.off); s0, W, meta = L.lsml_linear(X0); f0 = L.gap_flags(s0, Cd.off)
AM = S.AnswerModel(Cd, rawC, s0)
FC, _, _, sgC = RA.feats(AM, Cd, rawC, ONE); AM.fit_fusion(FC)
B = dict(ev0=dict(W=W.tolist()), AM=AM)
assert np.allclose(X0 @ W, s0, atol=1e-9)
rho_sg = float(np.corrcoef(sgC, s0)[0, 1]); sg_sign = 1. if rho_sg >= 0 else -1.


def load_v(V):
    F = np.load(sdir / f'STEPS_{bench}_{V}.npz')
    sc = {s['uid']: s for s in map(json.loads, open(vdir / f'SIDECAR_{bench}_{V}.jsonl')) if 'skip' not in s}
    P = A.load_c_pool(sdir / f'STEPS_{bench}_{V}.npz', bench, groups={u: sc[u]['group'] for u in F['uids'].astype(str)})
    P.V = V; P.parent = np.array([sc[u]['parent_uid'] for u in P.uids]); P.k = np.array([sc[u]['k'] - 1 for u in P.uids])
    return P


items_c, items_w, kk, fam = [], [], [], []
n_children = {}
for V in VS:
    P = load_v(V); n_children[V] = int(P.n); P, drop = RA.aligned(P, Cd)
    assert drop == 0, (V, drop)
    pj = RA.parent_index(P, Cd)
    for j in range(P.n):
        k = int(P.k[j])
        if k < 1: continue
        items_c.append((Cd, pj[j], k)); items_w.append((P, j, k)); kk.append(k); fam.append(V)
kk = np.array(kk); fam = np.array(fam)
Pc = PS.prefix_pool(items_c, bench); Pw = PS.prefix_pool(items_w, bench)
rc, rw = PS.pipeline(Pc, B), PS.pipeline(Pw, B)
npair = len(kk)
off = np.r_[0, np.cumsum(np.r_[kk + 1, kk + 1])].astype(np.int64)
f = np.r_[rc['f'], rw['f']]; sg = sg_sign * np.r_[rc['sg'], rw['sg']]
kerr = np.r_[np.full(npair, -1), kk]


def pseudo_metric(flags):
    if bench == 'deltabench':
        tg = [set() if k < 0 else {int(k) + 1} for k in kerr]
        m = O.per_sample(O.sets_from_flags(flags, off), tg)
        return 100 * float(m['F1_empty1'].mean())
    return PS.pseudo_metric(bench, flags, off, kerr, None)        # PRMScore, keep = ones (pseudo.py, unchanged)


refv = sg[:off[npair]][f[:off[npair]]]
curve = []
for be in PS.ALPHAS:
    tau = -np.inf if be >= 1 else (np.max(refv) if be <= 0 else np.quantile(refv, 1 - be))
    curve.append(pseudo_metric(f & (sg > tau)))
curve = np.array(curve); best = PS.ALPHAS[np.flatnonzero(curve >= curve.max() - 1e-12)].max()
tau = -np.inf if best >= 1 else (np.max(refv) if best <= 0 else np.quantile(refv, 1 - best))
fl_real = f0 & (sg_sign * sgC > tau)
out = dict(bench=bench, c_pool=CNAME, variants=VS, n_answers=int(Cd.n), n_steps=int(Cd.off[-1]), n_groups=int(len(set(Cd.group))),
           n_children=n_children, n_pairs=int(npair), n_pairs_by_family={V: int((fam == V).sum()) for V in VS},
           prefix_len_mean=float(np.mean(kk + 1)), sg_vs_s0_corr=rho_sg, lsml=meta,
           pseudo_wrong_hit_rate_gap=float(np.mean(L.pb_predict(rw['f'], Pw.off) == kk)),
           rule_C=dict(beta=float(best), tau=float(tau), sg_sign=sg_sign, pseudo_metric=('official_macroF1_empty1' if bench == 'deltabench' else 'PRMScore_keep_ones'),
                       pseudo_metric_max=float(curve.max()), pseudo_metric_open_all=float(curve[-1]),
                       n_ref_steps=int(len(refv)), real_zero_flag_share=float(np.mean(np.add.reduceat(fl_real.astype(int), Cd.off[:-1]) == 0)),
                       real_flag_keep_share=float(fl_real.sum() / f0.sum()), pseudo_curve=curve.tolist()))
np.savez_compressed(outd / f'RULEC_{bench}.npz', uids=Cd.uids, offsets=Cd.off, flagsC=fl_real, f0=f0, s0=s0, sgC=sg_sign * sgC,
                    pseudo_kerr=kerr, pseudo_len=np.r_[kk + 1, kk + 1], pseudo_flags=f, pseudo_sg=sg, pseudo_family=np.r_[fam, fam],
                    pseudo_parent=np.r_[Pc.parent, Pw.parent], alphas=PS.ALPHAS)
json.dump(RA.rnd(out, 5), open(outd / f'PSEUDO_LABELFREE_{bench}.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in RA.rnd(out, 4).items() if k != 'rule_C'}))
print('rule C', json.dumps({k: v for k, v in RA.rnd(out['rule_C'], 4).items() if k != 'pseudo_curve'}))
