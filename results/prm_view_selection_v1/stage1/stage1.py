"""Stage 1 of PRM-anchored view selection: exhaustive subset search (pre-registration
docs/experiments/PRM_VIEW_SELECTION_STAGE1_PREREG_2026-10-25.md).

    python stage1.py check            # parity of the vectorized GAP / metric / groups against the evaluator
    python stage1.py time [n]         # time n random subsets on all 10 cells
    python stage1.py run [workers]    # all 64,839 subsets, checkpointed per chunk in chunks/
    python stage1.py assemble         # chunks -> SEARCH.npz
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
import evaluate_collection as E  # noqa: E402
from spectral_utils import prm_anchored_lsml as A  # noqa: E402
from spectral_utils.fusion_utils import detect_dependent_groups  # noqa: E402

PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd']
CELLS = [(b, p) for b in ('prmbench', 'socratic') for p in PRMS]
META = json.loads((ROOT / 'results/prm_boost_collection_v1/tables/STATUS_META.json').read_text())
STATUS = {(r['benchmark'], r['prm'], r['row']): r for r in E.read_csv(ROOT / 'results/prm_boost_collection_v1/tables/STATUS_MAIN.csv')}
CHUNK = 500
N_CH = 16


def prm_arm(bench, prm):
    prob = META[f'{bench}/{prm}']['prm_reference_scale'] == '1-reward'
    return ('PRM_Q80_PROB' if prob else 'PRM_Q80') if bench == 'prmbench' else ('PRMPROB' if prob else 'PRM')


class CellData:
    def __init__(self, bench, prm):
        c = E.load(bench, prm); self.c = c; self.bench, self.prm = bench, prm
        arm = prm_arm(bench, prm); self.arm = arm
        raw = np.column_stack([c.arms[arm], E.channels(c, 'N16')])
        lens = np.repeat(np.diff(c.off), np.diff(c.off)); self.pool = c.keep & (lens > 1)
        Zr = np.column_stack([E.answer_z(raw[:, k], c.off) for k in range(raw.shape[1])])
        sign = np.sign([1.] + [np.corrcoef(Zr[self.pool, 0], Zr[self.pool, k])[0, 1] for k in range(1, raw.shape[1])])
        sign[sign == 0] = 1; self.sign = sign
        X = raw * sign
        self.Z = np.column_stack([E.answer_z(X[:, k], c.off) for k in range(X.shape[1])])
        sd = np.array([[X[a:b, k].std() for k in range(X.shape[1])] for a, b in zip(c.off[:-1], c.off[1:])])
        const = np.repeat(sd < 1e-8, np.diff(c.off), axis=0)
        V = np.empty_like(self.Z)
        for k in range(X.shape[1]):
            f = E.topk_flags(self.Z[:, k], c.off, E.gap_counts(self.Z[:, k], c.off)); f[const[:, k]] = False
            V[:, k] = np.where(f, 1., -1.)
        self.V = V; self.Vp = V[self.pool]
        self.R = np.cov(np.ascontiguousarray(self.Vp).T)
        self.ans = A.Answers(c.off)
        yv = ~c.err; m = c.keep
        self._yv_m, self._nyv_m = yv & m, ~yv & m
        self.empty = np.zeros(4)
        if bench != 'prmbench' and len(c.empty['correct']):
            e = c.empty; yv_e, inc = e['correct'], e['include']
            self.empty = np.array([0, 0, np.sum(~yv_e & inc), np.sum(yv_e & inc)], float)   # predicted incorrect
        fP = E.topk_flags(self.Z[:, 0], c.off, E.gap_counts(self.Z[:, 0], c.off))
        self.ps_prm = self.metric_exact(fP)
        ref = STATUS.get((bench, prm, 'PRM same rule (better scale under GAP)'))
        self.ps_prm_status = float(ref['GAP']) if ref else None

    def metric(self, f):
        pv = ~f
        C = np.array([np.sum(self._yv_m & pv), np.sum(self._nyv_m & pv), np.sum(self._nyv_m & f), np.sum(self._yv_m & f)], float)
        C += self.empty
        if self.bench == 'prmbench':
            return float(E.official_from_counts(C, 'prmbench'))
        tp, fp, tn, fn = C
        pc, rc = tp / (tp + fp), tp / (tp + fn); pe, re = tn / (tn + fn), tn / (tn + fp)
        return 100 * .5 * (2 * pc * rc / (pc + rc) + 2 * pe * re / (pe + re))

    def metric_exact(self, f):
        if self.bench == 'prmbench':
            return float(E.official_from_counts(E.counts_by_answer(self.c, f).sum(0), 'prmbench'))
        return 100 * float(E.official(self.c, f)[0])

    def flags(self, s):
        return self.ans.gap_flags(self.ans.z(s))

    def evaluate(self, S, want_detail=False):
        """S: 0-based channel indices (column = index + 1). Returns a dict; valid=False if K < 3."""
        cols = np.asarray(S) + 1
        K, cg = A.groups_from_cov(self.R[np.ix_(cols, cols)])
        out = dict(K=K, valid=K >= 3)
        if not out['valid']:
            return out
        vprm_p, vprm = self.Vp[:, 0], self.V[:, 0]
        xi_p, xi, comps, wg_all, nclip = [vprm_p], [vprm], [], [], 0
        for g in range(K):
            idx = cols[cg == g]
            w, nneg = A.sml_weights_nonneg(self.R[np.ix_(idx, idx)]); nclip += nneg
            if len(idx) == 1:
                a_p, a = self.Vp[:, idx[0]], self.V[:, idx[0]]
            else:
                sp, s = self.Vp[:, idx] @ w, self.V[:, idx] @ w
                a_p = np.where(sp > 0, 1., np.where(sp < 0, -1., vprm_p)); a = np.where(s > 0, 1., np.where(s < 0, -1., vprm))
            xi_p.append(a_p); xi.append(a)
            comps.append(self.ans.z(self.Z[:, idx] @ w)); wg_all.append((idx, w))
        ds = A.ds_em_votes(np.column_stack(xi_p), anchor=0)
        wb = ds['w']
        if wb.sum() <= 0:
            out.update(valid=False, reason='all between weights clipped'); return out
        fused = wb[0] * self.Z[:, 0] + sum(wb[g + 1] * comps[g] for g in range(K))
        fused_bin = np.column_stack(xi) @ wb + 1e-6 * self.Z[:, 0]
        mean = self.Z[:, np.r_[0, cols]].mean(1)
        out.update(ps_fused=self.metric(self.flags(fused)), ps_mean=self.metric(self.flags(mean)),
                   ps_bin=self.metric(self.flags(fused_bin)), w_prm_share=float(wb[0] / wb.sum()),
                   n_clip_within=nclip, n_clip_between=int((ds['w_raw'] < 0).sum()))
        if want_detail:
            out.update(fused=fused, mean=mean, fused_bin=fused_bin, groups=cg, between=ds, within=wg_all)
        return out


_CELLS = None


def _init():
    global _CELLS
    _CELLS = [CellData(b, p) for b, p in CELLS]


def all_subsets():
    masks = np.array([m for m in range(1, 1 << N_CH) if bin(m).count('1') >= 4], np.int64)
    return masks


def subset_of(mask):
    return [i for i in range(N_CH) if (mask >> i) & 1]


def _run_chunk(ci):
    path = HERE / 'chunks' / f'chunk_{ci:04d}.npz'
    if path.exists():
        return ci, 'done'
    masks = all_subsets()[ci * CHUNK:(ci + 1) * CHUNK]
    nC = len(_CELLS)
    R = {k: np.full((len(masks), nC), np.nan) for k in ('ps_fused', 'ps_mean', 'ps_bin', 'w_prm_share')}
    Kk = np.zeros((len(masks), nC), np.int8); valid = np.zeros((len(masks), nC), bool)
    clip_w = np.zeros((len(masks), nC), np.int16); clip_b = np.zeros((len(masks), nC), np.int16)
    for i, m in enumerate(masks):
        S = subset_of(int(m))
        for j, cell in enumerate(_CELLS):
            o = cell.evaluate(S); Kk[i, j] = o['K']; valid[i, j] = o['valid']
            if not o['valid']:
                break                                      # invalid in one cell -> skip the rest (pre-registered)
            for k in R: R[k][i, j] = o[k]
            clip_w[i, j], clip_b[i, j] = o['n_clip_within'], o['n_clip_between']
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, masks=masks, K=Kk, valid=valid, clip_within=clip_w, clip_between=clip_b, **R)
    os.replace(tmp, path)
    return ci, 'ok'


def check():
    _init(); rng = np.random.default_rng(0); rep = {}
    for cell in _CELLS:
        c = cell.c
        # 1. PRM view: fast metric on the exact flags, exact metric, STATUS reference
        fP = E.topk_flags(cell.Z[:, 0], c.off, E.gap_counts(cell.Z[:, 0], c.off))
        r = dict(prm_exact=cell.ps_prm, prm_status=cell.ps_prm_status, prm_fast_metric=cell.metric(fP),
                 prm_fast_flags_equal=bool(np.array_equal(cell.flags(cell.Z[:, 0]), fP)))
        # 2. 50 random fused scores: fast flags/metric vs evaluator loop
        n_eq, maxd, n_tested, gp_eq = 0, 0., 0, 0
        for _ in range(50):
            S = sorted(rng.choice(N_CH, size=int(rng.integers(4, 12)), replace=False).tolist())
            o = cell.evaluate(S, want_detail=True)
            cols = np.asarray(S) + 1
            Kd, cd, *_ = detect_dependent_groups([cell.Vp[:, k] for k in cols])
            K2, c2 = A.groups_from_cov(cell.R[np.ix_(cols, cols)])
            same = Kd == K2 and len({(a, b) for a, b in zip(cd, c2)}) == Kd
            gp_eq += same
            if not o['valid']: continue
            n_tested += 1
            fe = E.topk_flags(E.answer_z(o['fused'], c.off), c.off, E.gap_counts(E.answer_z(o['fused'], c.off), c.off))
            ff = cell.flags(o['fused'])
            n_eq += np.array_equal(fe, ff)
            maxd = max(maxd, abs(cell.metric_exact(fe) - o['ps_fused']))
        r.update(fused_tested=n_tested, fused_flags_equal=n_eq, fused_max_abs_metric_diff=maxd, groups_equal_of_50=gp_eq)
        rep[f'{cell.bench}/{cell.prm}'] = r; print(cell.bench, cell.prm, r, flush=True)
    json.dump(rep, open(HERE / 'PARITY.json', 'w'), indent=1, default=float)


def timing(n):
    t0 = time.time(); _init(); print(f'init {time.time() - t0:.1f}s', flush=True)
    rng = np.random.default_rng(1); masks = rng.choice(all_subsets(), n, replace=False)
    t = time.time(); nv = 0
    for m in masks:
        S = subset_of(int(m))
        for cell in _CELLS:
            o = cell.evaluate(S)
            if not o['valid']: break
        else:
            nv += 1
    dt = time.time() - t
    print(f'{n} subsets: {dt:.1f}s ({dt / n:.3f}s per subset, all-valid {nv}); full search ~{dt / n * len(all_subsets()) / 3600:.1f} core-h')


def run(workers):
    import multiprocessing as mp
    (HERE / 'chunks').mkdir(exist_ok=True)
    n_chunks = int(np.ceil(len(all_subsets()) / CHUNK)); t0 = time.time()
    with mp.get_context('spawn').Pool(workers, initializer=_init) as pool:
        for k, (ci, st) in enumerate(pool.imap_unordered(_run_chunk, range(n_chunks))):
            print(f'{time.strftime("%H:%M:%S")} chunk {ci} {st} ({k + 1}/{n_chunks}, {time.time() - t0:.0f}s)', flush=True)


def assemble():
    files = sorted((HERE / 'chunks').glob('chunk_*.npz'))
    parts = [dict(np.load(f)) for f in files if not f.name.endswith('.tmp.npz')]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert len(out['masks']) == len(all_subsets()), (len(out['masks']), len(all_subsets()))
    np.savez_compressed(HERE / 'SEARCH.npz', cells=np.array([f'{b}/{p}' for b, p in CELLS]), **out)
    print('assembled', len(out['masks']))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'check': check()
    elif cmd == 'time': timing(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif cmd == 'run': run(int(sys.argv[2]) if len(sys.argv) > 2 else 16)
    elif cmd == 'assemble': assemble()
