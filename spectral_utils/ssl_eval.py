"""Shared evaluation for the SSL plan stages (S2+): frozen population frame, PB/PRMB metrics,
C-calibrated PRMScore, paired source-group bootstrap.  Labels are used ONLY here."""
from pathlib import Path
import hashlib, json, pickle, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from .ssl_s1 import first_argmax, within_auc, depth_bin

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); W = MAIN / '.worktrees/readout-quickest-detection-v1'; R = W / 'results/step_evidence_v1'
QGRID = np.round(np.linspace(.5, .99, 50), 2)


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p, v): Path(p).write_text(json.dumps(v, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


class Frame:
    """The frozen Step 432 population: answers, folds, cells, labels, CT7 gate, PRMB metadata."""
    def __init__(self):
        self.ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Zs = np.load(R / 'OOF_STEP_SCORES.npz'); self.Zs = Zs
        self.off = Zs['offsets']; self.labels = Zs['labels']; self.n = len(self.ans); self.nsteps = np.diff(self.off)
        a = self.ans; self.pb = a.cell.str.startswith('pb_').to_numpy(); self.prm = ~self.pb; self.fold = a.fold.to_numpy(); self.cells = a.cell.to_numpy(); self.groups = a.source_group.to_numpy()
        self.uid = a.uid.to_numpy(); self.ids = a.id.to_numpy(); self.target = a.target.to_numpy(); self.gate = a.ct7_gate.to_numpy().astype(bool)
        freeze = read(R / 'INPUT_FREEZE.json'); self.prm_meta_path = Path(freeze['prm_metadata']['path'])
        self.meta = {m['idx']: m for m in pickle.load(open(self.prm_meta_path, 'rb')).values()}
        self.noncontrol = np.array([self.prm[i] and self.meta[self.ids[i]]['classification'] != 'correct' for i in range(self.n)])
        self.task_of = np.where(self.prm, 'prm', np.where(np.char.endswith(self.cells.astype(str), 'q4'), 'pb_q4', 'pb_q8'))
        self.pb_cells = sorted(set(self.cells[self.pb])); self.prm_idx = np.flatnonzero(self.prm)
        self.step_len = np.load(R / 'step_lengths.npy')
        self.kind = np.array(['' if not self.prm[i] else 'clean' if self.labels[self.off[i]:self.off[i+1]].sum() == 0 else 'single' if self.labels[self.off[i]:self.off[i+1]].sum() == 1 else 'multi' for i in range(self.n)])
        sys.path.insert(0, str(MAIN)); from spectral_utils.prmbench import prmbench_evaluate; self._official = prmbench_evaluate
    def roles(self, task, k):
        m = self.task_of == task
        return {'H': m & (self.fold == k), 'C': m & (self.fold == (k + 1) % 5), 'B': m & (self.fold == (k + 2) % 5), 'A': m & np.isin(self.fold, [(k + 3) % 5, (k + 4) % 5])}
    def splits_json(self, tasks):
        out = {}
        for task in tasks:
            for k in range(5):
                rm = self.roles(task, k); gs = {r: set(self.groups[m]) for r, m in rm.items()}
                out[f'{task}/fold{k}'] = {r: {'answers': int(m.sum()), 'groups': len(gs[r])} for r, m in rm.items()} | {'pairwise_group_overlap': int(sum(len(gs[x] & gs[y]) for x in gs for y in gs if x < y)), 'covers_task': bool(sum(m.sum() for m in rm.values()) == (self.task_of == task).sum())}
        assert all(v['pairwise_group_overlap'] == 0 and v['covers_task'] for v in out.values())
        return out
    def official_prmscore(self, valid, idx):
        r = self._official([{'idx': self.ids[i], 'labels': valid[self.off[i]:self.off[i+1]].astype(int).tolist()} for i in idx], [self.meta[self.ids[i]] for i in idx])['total']
        return .5 * (r['f1'] + r['negative_f1'])


def prmscore_grid(valid, y, e):
    v = np.asarray(valid, bool)[..., e]; good = y[e] == 0; tp = (v & good).sum(-1); fp = (v & ~good).sum(-1); tn = (~v & ~good).sum(-1); fn = (~v & good).sum(-1)
    def ratio(a, b): return np.divide(a, b, out=np.full(np.shape(a), -1., float), where=b != 0)
    p = ratio(tp, tp + fp); r = ratio(tp, tp + fn); f = ratio(2 * p * r, p + r); p = ratio(tn, tn + fn); r = ratio(tn, tn + fp); nf = ratio(2 * p * r, p + r); return (f + nf) / 2


def zt(s): return (s - s.mean()) / max(s.std(), 1e-8)


def evaluate(F, scores, new_methods, comparators, cal_scores, out):
    """scores: method -> step array; cal_scores: (method, k) -> {answer: C-scores} for the new methods.
    Writes OOF_ANSWERS.csv, EVAL_JOINED.csv, METRICS.csv. Returns (pred, auc, metrics, covered)."""
    methods = new_methods + comparators; off, n = F.off, F.n
    covered = {m: np.array([np.isfinite(scores[m][off[i]:off[i+1]]).all() for i in range(n)]) for m in methods}
    pred = {m: np.array([first_argmax(scores[m][off[i]:off[i+1]]) if covered[m][i] and F.pb[i] else -999 for i in range(n)]) for m in methods}
    auc = {m: np.array([within_auc(F.labels[off[i]:off[i+1]], scores[m][off[i]:off[i+1]]) if F.prm[i] and covered[m][i] else np.nan for i in range(n)]) for m in methods}
    oa = F.ans[['uid', 'id', 'source_group', 'fold', 'cell', 'target', 'ct7_gate']].copy(); oa['n_steps'] = F.nsteps; oa['task'] = F.task_of
    for m in methods: oa[m + '__pred'] = pred[m]; oa[m + '__within_auc'] = auc[m]; oa[m + '__covered'] = covered[m]
    oa.to_csv(out / 'OOF_ANSWERS.csv', index=False)
    pd.DataFrame({'uid': np.repeat(F.uid, F.nsteps), 'step': np.concatenate([np.arange(s) for s in F.nsteps]), 'label': F.labels}).to_csv(out / 'EVAL_JOINED.csv', index=False)
    elig = np.repeat(F.noncontrol, F.nsteps); prm_steps = np.repeat(F.prm, F.nsteps); metrics = []
    prmscore = {}
    for m in new_methods:
        for variant, T in [('raw', lambda s: s), ('answer_z', zt)]:
            v_sel = np.zeros(int(off[-1]), bool); v_q80 = v_sel.copy(); ok = True
            for k in range(5):
                if (m, k) not in cal_scores: ok = False; break
                cs = np.concatenate([T(v) for v in cal_scores[(m, k)].values()]); cy = np.concatenate([F.labels[off[i]:off[i+1]] for i in cal_scores[(m, k)]]); ce = np.concatenate([np.full(F.nsteps[i], F.noncontrol[i]) for i in cal_scores[(m, k)]])
                thr = np.quantile(cs, QGRID); best = int(np.argmax(prmscore_grid(cs[None, :] < thr[:, None], cy, ce))); tau80 = float(np.quantile(cs, .8))
                for i in np.flatnonzero(F.prm & (F.fold == k)):
                    a, b = off[i:i+2]; s = T(scores[m][a:b]); v_sel[a:b] = s < thr[best]; v_q80[a:b] = s < tau80
            if ok: prmscore[(m, variant)] = {'inner': F.official_prmscore(v_sel, F.prm_idx), 'q80': F.official_prmscore(v_q80, F.prm_idx)}
    for m in comparators:
        idx = F.prm_idx[covered[m][F.prm_idx]]; s = scores[m]; v = np.zeros(int(off[-1]), bool)
        for k in range(5):
            trs = np.repeat(F.prm & covered[m] & (F.fold != k), F.nsteps); tau = np.quantile(s[trs], .8); tst = np.repeat(F.prm & covered[m] & (F.fold == k), F.nsteps); v[tst] = s[tst] < tau
        prmscore[(m, 'raw')] = {'q80': F.official_prmscore(v, idx)}
    for m in methods:
        blk = {}
        for c in F.pb_cells:
            take = (F.cells == c) & covered[m]; e = take & (F.target >= 0); cl = take & (F.target < 0); d = pred[m][e] - F.target[e]
            ca = float((~F.gate[cl]).mean()); ea = float(((d == 0) & F.gate[e]).mean())
            blk[c] = {'n': int(take.sum()), 'errors': int(e.sum()), 'sla': float((d == 0).mean()), 'tol1': float((abs(d) <= 1).mean()), 'early': float((d < 0).mean()), 'late': float((d > 0).mean()), 'f1': 2 * ca * ea / (ca + ea) if ca + ea else 0., 'clean_acc': ca, 'gated_err_acc': ea}
            for key in ['sla', 'tol1', 'early', 'late', 'f1', 'clean_acc', 'gated_err_acc']: metrics.append({'method': m, 'benchmark': 'pb', 'metric': key, 'stratum': c, 'N': blk[c]['errors'] if key != 'clean_acc' else blk[c]['n'] - blk[c]['errors'], 'estimate': blk[c][key], 'coverage': blk[c]['n']})
        for scope, suf in [('macro8', ''), ('q4', 'q4'), ('q8', 'q8')]:
            vals = [r for c, r in blk.items() if c.endswith(suf)]
            for key in ['sla', 'tol1', 'early', 'late', 'f1']: metrics.append({'method': m, 'benchmark': 'pb', 'metric': key, 'stratum': scope, 'N': sum(r['errors'] for r in vals), 'estimate': float(np.mean([r[key] for r in vals])), 'coverage': sum(r['n'] for r in vals)})
        el = np.isfinite(auc[m]); metrics.append({'method': m, 'benchmark': 'prm', 'metric': 'within_auc', 'stratum': 'all', 'N': int(el.sum()), 'estimate': float(np.nanmean(auc[m])), 'coverage': int((F.prm & covered[m]).sum())})
        fa, fp_ = [], []
        for k in range(5):
            st = np.repeat(F.prm & covered[m] & (F.fold == k), F.nsteps); fa.append(roc_auc_score(F.labels[st], scores[m][st])); fp_.append(average_precision_score(F.labels[st], scores[m][st]))
        metrics += [{'method': m, 'benchmark': 'prm', 'metric': 'step_auroc_mean_folds', 'stratum': 'all', 'N': int(prm_steps.sum()), 'estimate': float(np.mean(fa)), 'coverage': int((F.prm & covered[m]).sum())}, {'method': m, 'benchmark': 'prm', 'metric': 'step_auprc_mean_folds', 'stratum': 'all', 'N': int(prm_steps.sum()), 'estimate': float(np.mean(fp_)), 'coverage': int((F.prm & covered[m]).sum())}]
        for (mm, variant), d in prmscore.items():
            if mm == m:
                for kind, val in d.items(): metrics.append({'method': m, 'benchmark': 'prm', 'metric': f'prmscore_{kind}_{variant}', 'stratum': 'all', 'N': int(F.noncontrol.sum()), 'estimate': val, 'coverage': int((F.prm & covered[m]).sum())})
        for kd in ['single', 'multi']:
            sel = (F.kind == kd) & np.isfinite(auc[m]); metrics.append({'method': m, 'benchmark': 'prm', 'metric': 'within_auc', 'stratum': f'kind={kd}', 'N': int(sel.sum()), 'estimate': float(np.nanmean(auc[m][sel])), 'coverage': int(sel.sum())})
        dbin = np.array([depth_bin(s) for s in F.nsteps])
        for db in ['2-5', '6-10', '11+']:
            e = F.pb & (F.target >= 0) & covered[m] & (dbin == db); metrics.append({'method': m, 'benchmark': 'pb', 'metric': 'sla', 'stratum': f'depth={db}', 'N': int(e.sum()), 'estimate': float((pred[m][e] == F.target[e]).mean()), 'coverage': int(e.sum())})
            sel = F.prm & np.isfinite(auc[m]) & (dbin == db); metrics.append({'method': m, 'benchmark': 'prm', 'metric': 'within_auc', 'stratum': f'depth={db}', 'N': int(sel.sum()), 'estimate': float(np.nanmean(auc[m][sel])), 'coverage': int(sel.sum())})
    pd.DataFrame(metrics).to_csv(out / 'METRICS.csv', index=False)
    return pred, auc, metrics, covered


def bootstrap(F, pred, auc, covered, methods, contrasts, K, out, draws=100_000, seed=20260923):
    """contrasts: list of (a, b, primary). Writes CONTRASTS.csv and BOOTSTRAP_DELTAS.npz. Returns rows."""
    Gpb, gpb_inv = np.unique(F.groups[F.pb], return_inverse=True); Gpr, gpr_inv = np.unique(F.groups[F.prm], return_inverse=True); cell_idx = {c: j for j, c in enumerate(F.pb_cells)}
    def pbs(m):
        hits = np.zeros((len(Gpb), 8)); cnt = np.zeros((len(Gpb), 8))
        for gi, i in zip(gpb_inv, np.flatnonzero(F.pb)):
            if F.target[i] < 0 or not covered[m][i]: continue
            j = cell_idx[F.cells[i]]; cnt[gi, j] += 1; hits[gi, j] += pred[m][i] == F.target[i]
        return hits, cnt
    def prs(m):
        s = np.zeros(len(Gpr)); c = np.zeros(len(Gpr))
        for gi, i in zip(gpr_inv, np.flatnonzero(F.prm)):
            if np.isfinite(auc[m][i]): s[gi] += auc[m][i]; c[gi] += 1
        return s, c
    spb = {m: pbs(m) for m in methods}; spr = {m: prs(m) for m in methods}
    rng = np.random.default_rng(seed); est_pb = {m: np.empty(draws) for m in methods}; est_pr = {m: np.empty(draws) for m in methods}; invalid = np.zeros(draws, bool); pos = 0
    while pos < draws:
        nb = min(5000, draws - pos)
        wpb = rng.multinomial(len(Gpb), np.full(len(Gpb), 1 / len(Gpb)), size=nb).astype(float); wpr = rng.multinomial(len(Gpr), np.full(len(Gpr), 1 / len(Gpr)), size=nb).astype(float)
        for m in methods:
            h, c = spb[m]; H_ = wpb @ h; C_ = wpb @ c; bad = (C_ == 0).any(1); invalid[pos:pos + nb] |= bad
            with np.errstate(invalid='ignore', divide='ignore'): est_pb[m][pos:pos + nb] = np.where(bad, np.nan, (H_ / C_).mean(1))
            s, cc = spr[m]; est_pr[m][pos:pos + nb] = (wpr @ s) / (wpr @ cc)
        pos += nb
    point_pb = {m: float(np.mean([(pred[m][e] == F.target[e]).mean() for c in F.pb_cells for e in [(F.cells == c) & (F.target >= 0) & covered[m]]])) for m in methods}
    point_pr = {m: float(np.nanmean(auc[m])) for m in methods}
    rows = []; deltas = {}; seen = set()
    for a, b, primary in contrasts:
        if (a, b) in seen: continue
        seen.add((a, b))
        for ep, est, pt in [('pb_sla_macro8', est_pb, point_pb), ('prm_within_auc', est_pr, point_pr)]:
            d = est[a] - est[b]; dv = d[np.isfinite(d)]; deltas[f'{a}__minus__{b}__{ep}'] = d
            rows.append({'contrast_id': f'{a} - {b}', 'primary': primary, 'endpoint': ep, 'paired_N': int(len(dv)), 'delta': pt[a] - pt[b], 'ci95_lo': float(np.quantile(dv, .025)), 'ci95_hi': float(np.quantile(dv, .975)),
                         'ci_adj_lo': float(np.quantile(dv, .05 / K / 2)) if primary else None, 'ci_adj_hi': float(np.quantile(dv, 1 - .05 / K / 2)) if primary else None, 'family_K': K if primary else None, 'B': draws, 'invalid_draw_rate': float(np.isnan(d).mean()), 'units': 'AUC' if ep.startswith('prm') else 'fraction (x100 = pp)'})
    pd.DataFrame(rows).to_csv(out / 'CONTRASTS.csv', index=False)
    np.savez_compressed(out / 'BOOTSTRAP_DELTAS.npz', **deltas, invalid=invalid, seed=seed, draws=draws, pb_groups=len(Gpb), prm_groups=len(Gpr))
    return rows, float(invalid.mean())


def rescue_rows(F, pred, auc, scores, arms, base='BASE'):
    """Teacher/student-style 2x2 on PB and paired per-answer AUC / pair counts on PRMB, vs `base`."""
    err = F.pb & (F.target >= 0); out = []
    for m in arms:
        hs = pred[m][err] == F.target[err]; ht = pred[base][err] == F.target[err]
        out.append({'benchmark': 'pb', 'arm': m, 'agreement_with_base': float((pred[m][err] == pred[base][err]).mean()), 'both_correct': int((hs & ht).sum()), 'arm_only': int((hs & ~ht).sum()), 'base_only': int((~hs & ht).sum()), 'both_wrong': int((~hs & ~ht).sum()), 'net': int((hs & ~ht).sum() - (~hs & ht).sum()),
                    'moved_earlier': int((pred[m][err] < pred[base][err]).sum()), 'moved_later': int((pred[m][err] > pred[base][err]).sum())})
        d = auc[m] - auc[base]; okm = np.isfinite(d); fixed = broken = 0
        for i in np.flatnonzero(okm):
            a, b = F.off[i:i+2]; y = F.labels[a:b].astype(bool); sc, sr = scores[m][a:b], scores[base][a:b]
            wc = sc[y][:, None] > sc[~y][None, :]; wr = sr[y][:, None] > sr[~y][None, :]; fixed += int((wc & ~wr).sum()); broken += int((~wc & wr).sum())
        out.append({'benchmark': 'prm', 'arm': m, 'mean_delta_auc_vs_base': float(d[okm].mean()), 'answers_improved': int((d[okm] > 1e-12).sum()), 'answers_worsened': int((d[okm] < -1e-12).sum()), 'pairs_corrected': fixed, 'pairs_destroyed': broken})
    return out
