"""Shared pieces of the calibration-corrected stages (handoff 2026-09-24,
docs/reviews/CLAUDE_TOKEN_TAIL_LSML_CORRECTION_HANDOFF_20260924_HE.md).

- Population: the 13,769-answer development roster with folds, groups, labels, PRMBench metadata.
- Roles: outer fold k evaluates fold k, calibrates on fold (k+1)%5 and fits on the other three.
- ScoreBundle: write-once score arrays keyed by (method, outer_fold, role, answer) with one
  model_id per (method, outer_fold) shared by the calibration and evaluation roles.
- extgen: the frozen external-transfer fusion module, loaded by path from the main checkout.
- L-SML fit with the extgen semantics (anchor index and orientation matrix as parameters).
- Tail marks: historical (position tie-break) and tie-aware.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection')
STEP_EVIDENCE = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
EXTGEN_DIR = MAIN / 'spectral_utils/external_generalization'
EPS = 1e-12
ROLES = ('eval', 'cal')


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, v) -> None:
    Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')


def load_extgen():
    """Load main-checkout spectral_utils/external_generalization as package 'extgen' (no clash
    with another worktree's spectral_utils on sys.path)."""
    if 'extgen' not in sys.modules:
        spec = importlib.util.spec_from_file_location('extgen', EXTGEN_DIR / '__init__.py', submodule_search_locations=[str(EXTGEN_DIR)])
        mod = importlib.util.module_from_spec(spec); sys.modules['extgen'] = mod; spec.loader.exec_module(mod)
    return importlib.import_module('extgen.fusion')


def extgen_hashes() -> dict:
    return {str(p.relative_to(MAIN)): sha(p) for p in (EXTGEN_DIR / 'fusion.py', EXTGEN_DIR / '_bank11/fusion_utils.py')}


def roles_of(k: int) -> tuple[list[int], int, int]:
    """(fit folds, calibration fold, evaluation fold) for outer fold k."""
    cal = (k + 1) % 5
    return [f for f in range(5) if f not in (k, cal)], cal, k


class Population:
    def __init__(self):
        ans = pd.read_csv(STEP_EVIDENCE / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); z = np.load(STEP_EVIDENCE / 'OOF_STEP_SCORES.npz')
        self.off = z['offsets'].astype(np.int64); self.labels = z['labels'].astype(bool)   # True = error step (PRMB); PB rows are -2 -> True, never used
        self.n = len(ans); self.ns = np.diff(self.off); self.total = int(self.off[-1])
        self.pb = ans.cell.str.startswith('pb_').to_numpy(); self.prm = ~self.pb
        self.fold = ans.fold.to_numpy().astype(int); self.cells = ans.cell.to_numpy(); self.groups = ans.source_group.to_numpy()
        self.ids = ans.id.astype(str).to_numpy(); self.uids = ans.uid.astype(str).to_numpy(); self.target = ans.target.to_numpy().astype(int)
        self.step_fold = np.repeat(self.fold, self.ns); self.step_answer = np.repeat(np.arange(self.n), self.ns)
        freeze = json.loads((STEP_EVIDENCE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
        self.meta_path = Path(freeze['prm_metadata']['path'])
        self.meta = {m['idx']: m for m in pickle.load(open(self.meta_path, 'rb')).values()}
        self.noncontrol = np.array([self.prm[i] and self.meta[self.ids[i]]['classification'] != 'correct' for i in range(self.n)])
        self.eligible = np.array([self.prm[i] and self.labels[self.off[i]:self.off[i + 1]].any() and (~self.labels[self.off[i]:self.off[i + 1]]).any() for i in range(self.n)])
        self.ct7 = z['ct7'].astype(float)
        self.inputs = {'oof_answers': STEP_EVIDENCE / 'OOF_ANSWERS.csv', 'oof_step_scores': STEP_EVIDENCE / 'OOF_STEP_SCORES.npz', 'prm_metadata': self.meta_path}
        # official error steps (one-based in metadata) must equal the derived labels on every PRMB answer
        for i in np.flatnonzero(self.prm):
            a, b = self.off[i:i + 2]; es = set(self.meta[self.ids[i]]['error_steps'])
            assert np.array_equal(self.labels[a:b], [j + 1 in es for j in range(b - a)]), self.ids[i]
        # source groups never cross folds
        g2f = {}
        for g, f in zip(self.groups, self.fold):
            assert g2f.setdefault(g, f) == f, g

    def rows(self, answer_mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.repeat(answer_mask, self.ns))

    def answer_z(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, float); out = np.zeros_like(s)
        for a, b in zip(self.off[:-1], self.off[1:]):
            v = s[a:b]; sd = v.std()
            if sd > 1e-8:
                out[a:b] = (v - v.mean()) / sd
        return out

    def answer_standardize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float); out = np.zeros_like(X)
        for a, b in zip(self.off[:-1], self.off[1:]):
            blk = X[a:b]; sd = blk.std(0)
            out[a:b] = np.divide(blk - blk.mean(0), sd, out=np.zeros_like(blk), where=sd > EPS)
        return out


def model_id(method: str, k: int, record: dict) -> str:
    return hashlib.sha256(json.dumps({'method': method, 'outer_fold': k, **record}, sort_keys=True, default=lambda x: np.asarray(x).tolist()).encode()).hexdigest()[:16]


class ScoreBundle:
    """Write-once step scores per (method, role); model ids per (method, outer fold, role)."""

    def __init__(self, pop: Population):
        self.pop = pop; self.scores = {}; self.written = {}; self.ids = {}; self.models = []

    def _init(self, m):
        if (m, 'eval') not in self.scores:
            for r in ROLES:
                self.scores[m, r] = np.full(self.pop.total, np.nan); self.written[m, r] = np.zeros(self.pop.n, bool)

    def put_full(self, method: str, k: int, full_scores: np.ndarray, record: dict) -> str:
        """Score vector over all steps from ONE fitted model k; stores its eval-fold and cal-fold answers."""
        _fit, cal, ev = roles_of(k); mid = model_id(method, k, record)
        for role, f in (('eval', ev), ('cal', cal)):
            self.put_answers(method, k, role, np.flatnonzero(self.pop.fold == f), full_scores, mid)
        self.models.append({'method': method, 'outer_fold': k, 'fit_folds': _fit, 'cal_fold': cal, 'eval_fold': ev, 'model_id': mid, **record})
        return mid

    def put_answers(self, method, k, role, answers, full_scores, mid):
        self._init(method); fit, cal, ev = roles_of(k)
        assert role in ROLES and (self.pop.fold[answers] == (ev if role == 'eval' else cal)).all(), (method, k, role)
        assert not self.written[method, role][answers].any(), f'overwrite attempt {method} {role} fold {k}'
        rows = self.pop.rows(np.isin(np.arange(self.pop.n), answers))
        self.scores[method, role][rows] = np.asarray(full_scores, float)[rows]
        self.written[method, role][answers] = True
        assert self.ids.setdefault((method, k, role), mid) == mid

    def finalize(self, out_dir: Path) -> dict:
        methods = sorted({m for m, _ in self.scores}); checks = {'methods': len(methods), 'problems': []}
        for m in methods:
            for r in ROLES:
                if not self.written[m, r].all():
                    checks['problems'].append(f'{m} {r}: {int((~self.written[m, r]).sum())} answers never written')
                if not np.isfinite(self.scores[m, r]).all():
                    checks['problems'].append(f'{m} {r}: non-finite scores')
            for k in range(5):
                a, b = self.ids.get((m, k, 'eval')), self.ids.get((m, k, 'cal'))
                if a is None or a != b:
                    checks['problems'].append(f'{m} fold {k}: eval model {a} != cal model {b}')
        # a calibration answer is always scored by a different model than the one that evaluates it
        # (fold j is calibration for outer fold j-1 and evaluation for outer fold j)
        checks['role_model_map'] = {'eval': 'outer fold = answer fold', 'cal': 'outer fold = (answer fold - 1) mod 5'}
        pop = self.pop
        for k in range(5):
            fit, cal, ev = roles_of(k); gs = {r: set(pop.groups[np.isin(pop.fold, f)]) for r, f in (('fit', fit), ('cal', [cal]), ('eval', [ev]))}
            for a, b in (('fit', 'cal'), ('fit', 'eval'), ('cal', 'eval')):
                if gs[a] & gs[b]:
                    checks['problems'].append(f'fold {k}: {len(gs[a] & gs[b])} source groups shared by {a} and {b}')
        checks['status'] = 'PASS' if not checks['problems'] else 'FAIL'
        np.savez_compressed(out_dir / 'SCORES.npz', offsets=pop.off, answer_fold=pop.fold, answer_uid=pop.uids,
                            **{f'{r}__{m}': self.scores[m, r] for m in methods for r in ROLES})
        with open(out_dir / 'MODELS.jsonl', 'w', encoding='utf8') as f:
            for rec in self.models:
                f.write(json.dumps(rec, default=lambda x: np.asarray(x).tolist()) + '\n')
        dump(out_dir / 'BUNDLE_CHECKS.json', checks)
        assert checks['status'] == 'PASS', checks['problems'][:10]
        return checks


# ------------------------------------------------------------------ L-SML with the frozen external semantics
def lsml_fit(X: np.ndarray, anchor: int = 0, orient_X: np.ndarray | None = None) -> dict:
    """extgen.fusion.fit_weights generalized: anchor column index and a separate matrix for the
    orientation step (tail fits learn on marks, orient on the continuous features)."""
    fz = load_extgen(); nb = fz.numerical_backend
    x = np.asarray(X, np.float64); ox = x if orient_X is None else np.asarray(orient_X, np.float64)
    if x.ndim != 2 or x.shape[1] < 3 or len(x) < 3 * x.shape[1] or not np.isfinite(x).all():
        raise ValueError('insufficient or invalid fitting observations')
    if ox[:, anchor].std() <= EPS:
        raise ValueError('orientation anchor is inactive')
    nb.NUMERICAL_FAILURES.clear()
    _, meta = fz.lsml_continuous(*x.T, compute_score_matrix=False, small_m_guard=True)
    if nb.NUMERICAL_FAILURES or not np.isfinite(meta['residual']):
        raise ValueError('numerical estimator failure: ' + repr(nb.NUMERICAL_FAILURES))
    w = np.zeros(x.shape[1])
    for cross, (idx, within) in zip(meta['cross_weights'], meta['group_weights']):
        w[np.asarray(idx, int)] = np.asarray(within) * cross
    rho = float(spearmanr(ox @ w, ox[:, anchor]).statistic)
    if not np.isfinite(w).all() or abs(w).sum() <= EPS or not np.isfinite(rho):
        raise ValueError('invalid weights or undefined anchor orientation')
    if rho < 0:
        w *= -1
    w /= abs(w).sum()
    return {'weights': w, 'groups': np.asarray(meta['c'], int), 'K': int(len(np.unique(meta['c']))), 'anchor_spearman': abs(rho),
            'anchor_flipped': rho < 0, 'residual': float(meta['residual']), 'small_m_guarded': [list(v) for v in meta['small_m_guarded']]}


def partition_equal(groups) -> np.ndarray:
    return load_extgen().partition_equal(groups)


# ------------------------------------------------------------------ tail marks
def tail_marks(V: np.ndarray, off: np.ndarray, frac: float | None, tie_aware: bool, centred: bool = True) -> tuple[np.ndarray, dict]:
    """Per answer and column: mark the top ceil(frac*n) steps (frac None = top 1).
    Historical: stable argsort, ties broken by position. Tie-aware: ties at the boundary share the
    remaining mass, so a constant column carries no mark. Centred within answer if requested."""
    T = np.zeros_like(V, dtype=float); const = tie = cells = 0
    for a, b in zip(off[:-1], off[1:]):
        blk = V[a:b]; n = b - a; k = 1 if frac is None else max(1, int(np.ceil(frac * n)))
        if tie_aware:
            t = np.zeros_like(blk)
            for j in range(blk.shape[1]):
                v = blk[:, j]; thr = np.sort(v)[::-1][k - 1]; gt = v > thr; eq = v == thr
                t[gt, j] = 1.0; t[eq, j] = (k - gt.sum()) / eq.sum()
        else:
            order = np.argsort(-blk, axis=0, kind='stable'); t = np.zeros_like(blk)
            np.put_along_axis(t, order[:k], 1.0, axis=0)
        sd = blk.std(0); cells += blk.shape[1]; const += int((sd <= EPS).sum())
        srt = -np.sort(-blk, axis=0); thr = srt[k - 1]
        tie += int((((blk == thr).sum(0) > 1) & ((blk > thr).sum(0) + (blk == thr).sum(0) > k) & (sd > EPS)).sum())
        T[a:b] = t - t.mean(0) if centred else t
    return T, {'answer_columns': cells, 'constant_rate': const / cells, 'boundary_tie_rate': tie / cells}
