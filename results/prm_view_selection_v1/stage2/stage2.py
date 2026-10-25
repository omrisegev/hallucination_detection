"""Stage 2 of view selection: telemetry-only L-SML, exhaustive subset search, no PRM anywhere (pre-registration
docs/experiments/PRM_VIEW_SELECTION_STAGE2_PREREG_2026-10-26.md). Same machinery as stage 1 (stage1.py).

    python stage2.py cache            # light per-cell caches (stored channel orientation), with a parity check
    python stage2.py check            # vectorized GAP / metric vs the evaluator on random fused scores
    python stage2.py time [n]
    python stage2.py run [workers]    # all 64,839 subsets, checkpointed in chunks/
    python stage2.py assemble         # chunks -> SEARCH.npz
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'results/prm_boost_collection_v1'))
sys.path.insert(0, str(HERE.parent / 'stage1'))
import evaluate_collection as E  # noqa: E402
import stage1 as S1  # noqa: E402
from spectral_utils import prm_anchored_lsml as A  # noqa: E402

CELLS, PRMS, N_CH, CHUNK = S1.CELLS, S1.PRMS, S1.N_CH, S1.CHUNK
PRM_ALONE = json.loads((HERE.parent / 'stage1' / 'PARITY.json').read_text())   # prm_exact per cell (stage 1, exact loop)


class TeleCell:
    """One cell with the 16 channels in their stored (frozen-bank) orientation; no PRM."""

    def build(self, bench, prm):
        c = E.load(bench, prm); self.bench, self.prm = bench, prm
        X = E.channels(c, 'N16')
        lens = np.repeat(np.diff(c.off), np.diff(c.off)); self.pool = c.keep & (lens > 1)
        self.Z = np.column_stack([E.answer_z(X[:, k], c.off) for k in range(N_CH)])
        m16 = self.Z.mean(1)
        self.anticorr = int(sum(np.corrcoef(self.Z[self.pool, k], m16[self.pool])[0, 1] < 0 for k in range(N_CH)))
        sd = np.array([[X[a:b, k].std() for k in range(N_CH)] for a, b in zip(c.off[:-1], c.off[1:])])
        const = np.repeat(sd < 1e-8, np.diff(c.off), axis=0)
        V = np.empty_like(self.Z)
        for k in range(N_CH):
            f = E.topk_flags(self.Z[:, k], c.off, E.gap_counts(self.Z[:, k], c.off)); f[const[:, k]] = False
            V[:, k] = np.where(f, 1., -1.)
        self.V = V; self.Vp = V[self.pool]; self.R = np.cov(np.ascontiguousarray(self.Vp).T)
        self.off = c.off; self.ans = A.Answers(c.off)
        yv = ~c.err; m = c.keep; self._yv_m, self._nyv_m = yv & m, ~yv & m
        self.empty = np.zeros(4)
        if bench != 'prmbench' and len(c.empty['correct']):
            e = c.empty; self.empty = np.array([0, 0, np.sum(~e['correct'] & e['include']), np.sum(e['correct'] & e['include'])], float)
        self.ps_prm = float(PRM_ALONE[f'{bench}/{prm}']['prm_exact'])
        self.c = c
        return self

    def save(self, path):
        np.savez(path, Z=self.Z, V=self.V.astype(np.int8), pool=self.pool, R=self.R, off=self.off, yv_m=self._yv_m,
                 nyv_m=self._nyv_m, empty=self.empty, ps_prm=self.ps_prm, anticorr=self.anticorr, bench=self.bench, prm=self.prm)

    @classmethod
    def load(cls, path):
        D = np.load(path); o = cls(); o.c = None
        o.bench, o.prm = str(D['bench']), str(D['prm'])
        o.Z, o.V, o.pool, o.R, o.off = D['Z'], D['V'].astype(float), D['pool'], D['R'], D['off']
        o._yv_m, o._nyv_m, o.empty, o.ps_prm, o.anticorr = D['yv_m'], D['nyv_m'], D['empty'], float(D['ps_prm']), int(D['anticorr'])
        o.Vp = o.V[o.pool]; o.ans = A.Answers(o.off)
        return o

    metric = S1.CellData.metric
    flags = S1.CellData.flags

    def evaluate(self, S, want_detail=False):
        cols = np.asarray(S)
        K, cg = A.groups_from_cov(self.R[np.ix_(cols, cols)])
        out = dict(K=K, valid=K >= 3)
        if not out['valid']:
            return out
        mean = self.Z[:, cols].mean(1); fm = self.flags(mean)
        vmean = np.where(fm, 1., -1.); vmean_p = vmean[self.pool]
        xi_p, xi, comps, nclip = [], [], [], 0
        for g in range(K):
            idx = cols[cg == g]
            w, nneg = A.sml_weights_nonneg(self.R[np.ix_(idx, idx)]); nclip += nneg
            if len(idx) == 1:
                a_p, a = self.Vp[:, idx[0]], self.V[:, idx[0]]
            else:
                sp, s = self.Vp[:, idx] @ w, self.V[:, idx] @ w
                a_p = np.where(sp > 0, 1., np.where(sp < 0, -1., vmean_p)); a = np.where(s > 0, 1., np.where(s < 0, -1., vmean))
            xi_p.append(a_p); xi.append(a); comps.append(self.ans.z(self.Z[:, idx] @ w))
        ds = A.ds_em_votes(np.column_stack(xi_p), anchor=None)
        wb = ds['w']
        if wb.sum() <= 0:
            out.update(valid=False, reason='all between weights clipped'); return out
        fused = sum(wb[g] * comps[g] for g in range(K))
        fused_bin = np.column_stack(xi) @ wb + 1e-6 * mean
        out.update(ps_fused=self.metric(self.flags(fused)), ps_mean=self.metric(fm), ps_bin=self.metric(self.flags(fused_bin)),
                   max_group_share=float(wb.max() / wb.sum()), n_clip_within=nclip, n_clip_between=int((ds['w_raw'] < 0).sum()))
        if want_detail:
            out.update(fused=fused, mean=mean, fused_bin=fused_bin, groups=cg, between=ds)
        return out


_CELLS = None


def _init_light():
    global _CELLS
    _CELLS = [TeleCell.load(HERE / 'cache' / f'{b}__{p}.npz') for b, p in CELLS]
    print(f'{time.strftime("%H:%M:%S")} worker {os.getpid()} ready', flush=True)


def _run_chunk(ci):
    path = HERE / 'chunks' / f'chunk_{ci:04d}.npz'
    if path.exists():
        return ci, 'done'
    masks = S1.all_subsets()[ci * CHUNK:(ci + 1) * CHUNK]; nC = len(_CELLS)
    R = {k: np.full((len(masks), nC), np.nan) for k in ('ps_fused', 'ps_mean', 'ps_bin', 'max_group_share')}
    Kk = np.zeros((len(masks), nC), np.int8); valid = np.zeros((len(masks), nC), bool)
    clip_w = np.zeros((len(masks), nC), np.int16); clip_b = np.zeros((len(masks), nC), np.int16)
    for i, m in enumerate(masks):
        S = S1.subset_of(int(m))
        for j, cell in enumerate(_CELLS):
            o = cell.evaluate(S); Kk[i, j] = o['K']; valid[i, j] = o['valid']
            if not o['valid']:
                break
            for k in R: R[k][i, j] = o[k]
            clip_w[i, j], clip_b[i, j] = o['n_clip_within'], o['n_clip_between']
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, masks=masks, K=Kk, valid=valid, clip_within=clip_w, clip_between=clip_b, **R)
    os.replace(tmp, path)
    return ci, 'ok'


def build_cache():
    (HERE / 'cache').mkdir(exist_ok=True); meta = {}
    for b, p in CELLS:
        cell = TeleCell().build(b, p); path = HERE / 'cache' / f'{b}__{p}.npz'; cell.save(path)
        light = TeleCell.load(path)
        for S in ([0, 1, 2, 3], [0, 4, 7, 10, 12], list(range(16))):
            a, z = cell.evaluate(S), light.evaluate(S)
            assert a.keys() == z.keys() and all((a[k] == z[k]) or (a[k] != a[k] and z[k] != z[k]) for k in a), (b, p, S)
        meta[f'{b}/{p}'] = dict(anticorr_with_mean16=cell.anticorr, ps_prm=cell.ps_prm)
        print(b, p, 'cached, light parity OK, channels anti-correlated with the 16-mean:', cell.anticorr, flush=True)
    json.dump(meta, open(HERE / 'CACHE_META.json', 'w'), indent=1)


def check():
    rng = np.random.default_rng(0); rep = {}
    for b, p in CELLS:
        cell = TeleCell().build(b, p); c = cell.c; n_eq, n_t, maxd = 0, 0, 0.
        for _ in range(40):
            S = sorted(rng.choice(N_CH, size=int(rng.integers(4, 12)), replace=False).tolist())
            o = cell.evaluate(S, want_detail=True)
            if not o['valid']: continue
            n_t += 1
            for s, key in ((o['fused'], 'ps_fused'), (o['mean'], 'ps_mean')):
                z = E.answer_z(s, c.off); fe = E.topk_flags(z, c.off, E.gap_counts(z, c.off))
                n_eq += np.array_equal(fe, cell.flags(s))
                ex = float(E.official_from_counts(E.counts_by_answer(c, fe).sum(0), 'prmbench')) if b == 'prmbench' else 100 * float(E.official(c, fe)[0])
                maxd = max(maxd, abs(ex - o[key]))
        rep[f'{b}/{p}'] = dict(tested=n_t, flags_equal_of_2x=n_eq, max_abs_metric_diff=maxd)
        print(b, p, rep[f'{b}/{p}'], flush=True); del cell
    json.dump(rep, open(HERE / 'PARITY.json', 'w'), indent=1, default=float)


def timing(n):
    global _CELLS
    _init_light(); rng = np.random.default_rng(1); masks = rng.choice(S1.all_subsets(), n, replace=False)
    t = time.time(); nv = 0
    for m in masks:
        S = S1.subset_of(int(m))
        for cell in _CELLS:
            if not cell.evaluate(S)['valid']: break
        else:
            nv += 1
    dt = time.time() - t
    print(f'{n} subsets: {dt:.1f}s ({dt / n:.3f}s per subset, all-valid {nv}); full ~{dt / n * len(S1.all_subsets()) / 3600:.1f} core-h')


def run(workers):
    import multiprocessing as mp
    (HERE / 'chunks').mkdir(exist_ok=True)
    n_chunks = int(np.ceil(len(S1.all_subsets()) / CHUNK)); t0 = time.time()
    with mp.get_context('spawn').Pool(workers, initializer=_init_light) as pool:
        for k, (ci, st) in enumerate(pool.imap_unordered(_run_chunk, range(n_chunks))):
            print(f'{time.strftime("%H:%M:%S")} chunk {ci} {st} ({k + 1}/{n_chunks}, {time.time() - t0:.0f}s)', flush=True)


def assemble():
    parts = [dict(np.load(f)) for f in sorted((HERE / 'chunks').glob('chunk_*.npz')) if not f.name.endswith('.tmp.npz')]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert len(out['masks']) == len(S1.all_subsets())
    np.savez_compressed(HERE / 'SEARCH.npz', cells=np.array([f'{b}/{p}' for b, p in CELLS]), **out)
    print('assembled', len(out['masks']))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'cache': build_cache()
    elif cmd == 'check': check()
    elif cmd == 'time': timing(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif cmd == 'run': run(int(sys.argv[2]) if len(sys.argv) > 2 else 14)
    elif cmd == 'assemble': assemble()
