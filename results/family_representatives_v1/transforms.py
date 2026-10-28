"""Candidate transformations T: N56 -> Z (spec section 10). All fitted without labels.

Every function takes (X, names, off) with X answer-z-scored and returns (Z, new_names, log).
New columns are answer-z-scored again so every view enters L-SML on the same scale.
"""
from __future__ import annotations
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import norm, rankdata


def answer_z(x, off):
    out = np.zeros_like(x, dtype=float)
    for a, b in zip(off[:-1], off[1:]):
        s = x[a:b]; sd = s.std(0)
        out[a:b] = np.divide(s - s.mean(0), sd, out=np.zeros_like(s, dtype=float), where=sd > 1e-12)
    return out


def drop_copies(X, names, off, thr=.999):
    """T1: keep the first channel of every set with |r| >= thr (bank order)."""
    R = np.abs(np.corrcoef(X, rowvar=False)); keep, log = [], []
    for j in range(X.shape[1]):
        hit = [k for k in keep if R[j, k] >= thr]
        if hit:
            log.append(dict(dropped=names[j], copy_of=names[hit[0]], r=round(float(R[j, hit[0]]), 5)))
        else:
            keep.append(j)
    return X[:, keep], [names[k] for k in keep], log


def redundancy_blocks(X, thr=.8):
    R = np.abs(np.corrcoef(X, rowvar=False)); D = 1 - R; np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), 'average'), t=1 - thr, criterion='distance')
    return [np.flatnonzero(lab == k) for k in np.unique(lab)]


def compress_blocks(X, names, off, mode='pc1', thr=.8, n_pc=2):
    """T2: replace each block with |r| >= thr (average linkage) by a summary.

    mode 'mean'  : sign-oriented mean (signs from the block's leading eigenvector)
    mode 'pc1'   : projection on the leading eigenvector of the block correlation
    mode 'pcK'   : leading n_pc projections
    mode 'pc1res': PC1 plus each member's residual on PC1 for members whose
                   residual keeps >= 10% of its variance (the LR-style contrast)
    """
    cols, new, log = [], [], []
    for idx in redundancy_blocks(X, thr):
        if len(idx) == 1:
            cols.append(X[:, idx[0]]); new.append(names[idx[0]]); continue
        B = X[:, idx]; R = np.corrcoef(B, rowvar=False); w, U = np.linalg.eigh(R)
        U = U[:, ::-1]; w = w[::-1]
        U = U * np.where((U > 0).sum(0) >= len(idx) / 2, 1, -1)   # majority-positive gauge
        tag = '+'.join(names[i] for i in idx[:2]) + ('+...' if len(idx) > 2 else '')
        rec = dict(block=[names[i] for i in idx], eig=[round(float(x), 3) for x in w[:4]])
        if mode == 'mean':
            cols.append((B * np.sign(U[:, 0])).mean(1)); new.append(f'MEAN[{tag}]')
        else:
            k = 1 if mode in ('pc1', 'pc1res') else min(n_pc, len(idx))
            for q in range(k):
                cols.append(B @ U[:, q]); new.append(f'PC{q+1}[{tag}]')
            if mode == 'pc1res':
                p1 = B @ U[:, 0]; p1 = p1 / p1.std()
                for i in idx:
                    r = X[:, i] - np.dot(X[:, i], p1) / len(p1) * p1
                    if r.var() >= .10:
                        cols.append(r); new.append(f'RES[{names[i]}|PC1]')
        rec['out'] = new[-1] if mode == 'mean' else [n for n in new if tag in n]
        log.append(rec)
    return answer_z(np.column_stack(cols), off), new, log


def rank_gauss(X, names, off):
    """T4: monotone reshaping, pooled rank -> standard normal quantile per column."""
    n = X.shape[0]
    Z = np.column_stack([norm.ppf((rankdata(X[:, j]) - .5) / n) for j in range(X.shape[1])])
    return answer_z(Z, off), list(names), []


def residualize_nuisance(X, names, off, covariates, n_knots=8):
    """T3: remove the part of every column explained by observable step covariates.

    covariates: (n_steps, q) label-free step descriptors (relative position, log token
    length). Additive cubic B-spline basis per covariate, pooled least squares, then the
    residual is answer-z-scored. Length/position are observed, not the hidden truth; their
    shared effect on every channel is dependence that the L-SML model does not allow.
    Risk: if the truth itself moves with length, part of the signal is removed with it.
    """
    from sklearn.preprocessing import SplineTransformer
    C = answer_z(np.asarray(covariates, float), off)
    Bs = SplineTransformer(n_knots=n_knots, degree=3, knots='quantile').fit_transform(C)
    Bs = np.column_stack([Bs, np.ones(len(Bs))])
    beta, *_ = np.linalg.lstsq(Bs, X, rcond=None)
    R = X - Bs @ beta
    r2 = 1 - R.var(0) / np.maximum(X.var(0), 1e-12)
    log = [dict(channel=n, r2_removed=round(float(q), 4)) for n, q in zip(names, r2)]
    return answer_z(R, off), [f'{n}|res' for n in names], log


def step_covariates(off, token_lengths):
    L = np.diff(off)
    pos = np.concatenate([np.arange(l) / max(l - 1, 1) for l in L])
    return np.column_stack([pos, np.log1p(np.asarray(token_lengths, float))])


class FrozenT:
    """The two frozen candidates as fit/apply objects (fit on training steps only).

    Recipe: T1 drop copies |r| >= .999 (bank order) -> T2 blocks |r| >= .8 (average linkage)
    replaced by their PC1 (majority-positive gauge), singletons kept -> answer-z ->
    [candidate B only] T3 additive cubic-spline residualization on answer-z relative step
    position and log step token length (8 quantile knots), pooled least squares -> answer-z.
    """
    COPY_THR, BLOCK_THR, KNOTS = .999, .8, 8

    def __init__(self, residualize):
        self.residualize = bool(residualize)

    def fit(self, X, names, off, cov=None):
        _, kept, self.copy_log = drop_copies(X, names, off, self.COPY_THR)
        self.keep = [names.index(n) for n in kept]
        Xk = X[:, self.keep]
        self.blocks, self.names_out, self.block_log = [], [], []
        for idx in redundancy_blocks(Xk, self.BLOCK_THR):
            if len(idx) == 1:
                self.blocks.append((idx, np.ones(1))); self.names_out.append(kept[idx[0]]); continue
            w, U = np.linalg.eigh(np.corrcoef(Xk[:, idx], rowvar=False))
            u = U[:, -1] * (1 if (U[:, -1] > 0).sum() >= len(idx) / 2 else -1)
            self.blocks.append((idx, u))
            self.names_out.append('PC1[' + ', '.join(kept[i] for i in idx) + ']')
            self.block_log.append(dict(members=[kept[i] for i in idx], loadings=[round(float(a), 4) for a in u],
                                       eig=[round(float(a), 3) for a in w[::-1][:3]], pc1_share=round(float(w[-1] / w.sum()), 3)))
        if self.residualize:
            from sklearn.preprocessing import SplineTransformer
            self.spline = SplineTransformer(n_knots=self.KNOTS, degree=3, knots='quantile').fit(answer_z(cov, off))
            Z = self._compress(X, off)
            self.beta = np.linalg.lstsq(self._basis(cov, off), Z, rcond=None)[0]
            R = Z - self._basis(cov, off) @ self.beta
            self.r2 = {n: round(float(1 - R[:, j].var() / max(Z[:, j].var(), 1e-12)), 4) for j, n in enumerate(self.names_out)}
        return self

    def _compress(self, X, off):
        Xk = X[:, self.keep]
        return answer_z(np.column_stack([Xk[:, idx] @ u for idx, u in self.blocks]), off)

    def _basis(self, cov, off):
        B = self.spline.transform(answer_z(cov, off)); return np.column_stack([B, np.ones(len(B))])

    def transform(self, X, off, cov=None):
        Z = self._compress(X, off)
        if self.residualize:
            Z = answer_z(Z - self._basis(cov, off) @ self.beta, off)
        return Z

    def describe(self):
        d = dict(residualize=self.residualize, n_out=len(self.names_out), names_out=self.names_out,
                 copies_dropped=self.copy_log, blocks=self.block_log)
        if self.residualize:
            d['r2_removed_by_position_length'] = self.r2
        return d
