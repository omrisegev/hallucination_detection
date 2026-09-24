"""tail_lsml_banks_v1: tail-fitted L-SML on bank11 and on the CT7 step channels. Derived from tail_weighted_fusion_v1: between-group weights from top-k tail co-exceedance (SML / L-SML) versus covariance SML and equal,
on the named groups of named_group_fusion_v1 (within = mean).  Derived from named groups (AUC>=0.60 features, grouped by shared errors, level split
into 3 or 4), within-group mean/SML, between-group equal/SML/L-SML.  Protocol:
results/named_group_fusion_v1/PROTOCOL.json.  LABEL-USING DEVELOPMENT (orientation, filter, grouping).
"""
from pathlib import Path
import hashlib, json, pickle, sys, time
from datetime import datetime
import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
DEPTH = MAIN / '.worktrees/depth-feature-fusion-v1'; sys.path.insert(0, str(DEPTH))
from spectral_utils.lsml_gate_locator_research import FusionRecipe, answer_standardize, fit_fusion_weights, _orient  # noqa: E402
from spectral_utils.fusion_utils import sml_fuse_signed  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

STAGE = ROOT / 'results/tail_lsml_banks_v1'; OUT = STAGE / 'run_20260924'; OUT.mkdir(parents=True, exist_ok=True)
P = json.loads((STAGE / 'PROTOCOL.json').read_text(encoding='utf8'))
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
SCR = Path('C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/2d14a8c9-8b3f-489b-b26f-812b4b84a8b3/scratchpad')
INPUTS = {'pool_z': SCR / 'pool_z.npy', 'pool_names': SCR / 'pool_names.json', 'oof_answers': R / 'OOF_ANSWERS.csv',
          'oof_step_scores': R / 'OOF_STEP_SCORES.npz', 'pool_structure': ROOT / 'results/indbank_lsml_prmbench_v1/POOL_STRUCTURE.csv'}
FOLDS = list(range(5)); DRAWS = 20_000; SEED = 20260924; FIT_SEED = 20260919
DROP = ['ct7_ve1', 'hist_entropy_series', 'hist_spilled_series', 'hist_trace_length_series']
T0 = time.perf_counter()
def dump(p, v): Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds')}; dump(OUT / 'RUN_STATUS.json', status)

# ------------------------------------------------------------------ population frame (as declared_joint_run.py)
ans = pd.read_csv(INPUTS['oof_answers'], encoding='utf-8-sig'); Zs = np.load(INPUTS['oof_step_scores'])
off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans); ns = np.diff(off)
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); cells = ans.cell.to_numpy(); groups = ans.source_group.to_numpy(); ids = ans.id.to_numpy(); target = ans.target.to_numpy()
freeze = json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig')); meta = {m['idx']: m for m in pickle.load(open(Path(freeze['prm_metadata']['path']), 'rb')).values()}
noncontrol = np.array([prm[i] and meta[ids[i]]['classification'] != 'correct' for i in range(n)])
eligible = np.array([prm[i] and labels[off[i]:off[i+1]].any() and (~labels[off[i]:off[i+1]]).any() for i in range(n)])

# ------------------------------------------------------------------ channels and virtual cores (label-free)
POOL = np.load(INPUTS['pool_z']); PN = json.loads(INPUTS['pool_names'].read_text(encoding='utf8')); assert POOL.shape == (int(off[-1]), len(PN))
names = [c for c in PN if c not in DROP]
Z = answer_standardize(POOL[:, [PN.index(c) for c in names]].astype(np.float64), off); assert np.isfinite(Z).all()
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
# --- label-using design block (PRMBench eligible answers only): orientation + error correlation
el_rows = np.concatenate([np.arange(off[i], off[i+1]) for i in np.flatnonzero(eligible)])
yy = labels[el_rows].astype(float); Ze = Z[el_rows]
el_off = np.r_[0, np.cumsum(ns[eligible])]; starts = el_off[:-1]; lens = np.diff(el_off).astype(float)
S1 = np.add.reduceat(Ze * yy[:, None], starts, axis=0); n1 = np.add.reduceat(yy, starts); tot = np.add.reduceat(Ze, starts, axis=0)
m1 = S1 / n1[:, None]; m0 = (tot - S1) / (lens - n1)[:, None]
# orientation by the sign of the mean within-answer AUC (rank based); v1 used the mean difference,
# which disagreed with the AUC sign on 5 channels (3 wrong flips, 2 missed)
def _wauc(y, s):
    y = y.astype(bool); k1 = y.sum(); return (rankdata(s)[y].sum() - k1 * (k1 + 1) / 2) / (k1 * (len(y) - k1))
chan_auc = np.array([np.mean([_wauc(labels[off[i]:off[i+1]], Z[off[i]:off[i+1], j]) for i in np.flatnonzero(eligible)]) for j in range(len(names))])
sign = np.where(chan_auc >= 0.5, 1.0, -1.0)                                  # high = error after orientation
seg_id = np.repeat(np.arange(len(lens)), lens.astype(int))
resid = Ze - np.where(yy[:, None] > 0, m1[seg_id], m0[seg_id])             # remove the label within answer
Cw = np.corrcoef((resid * sign).T)
Zo = Z * sign

chan_auc_map = {c: float(a) for c, a in zip(names, chan_auc)}
SPLITS = P['splits']; kept = [c for c in names if max(chan_auc_map[c], 1 - chan_auc_map[c]) >= 0.60]
for sp, G in SPLITS.items():
    flat = sum(G.values(), []); assert sorted(flat) == sorted(kept), (sp, sorted(set(flat) ^ set(kept)))
col = {c: Zo[:, names.index(c)] for c in names}
dump(OUT / 'DESIGN.json', {'orientation': {c: int(s) for c, s in zip(names, sign)}, 'prm_auc_oriented': {c: max(a, 1 - a) for c, a in chan_auc_map.items()}, 'kept': kept, 'splits': SPLITS})
dump(OUT / 'INPUT_MANIFEST.json', {k: {'path': str(v), 'sha256': sha(v)} for k, v in INPUTS.items()} | {'script_sha256': sha(Path(__file__)), 'protocol_sha256': sha(STAGE / 'PROTOCOL.json')})

def rows_of(mask): return np.concatenate([np.arange(off[i], off[i+1]) for i in np.flatnonzero(mask)])
REF = {'all48': (Z, names, names.index('q15_H1')), 'all48_oriented': (Zo, names, names.index('q15_H1')),
       'kept28_oriented': (Zo[:, [names.index(c) for c in kept]], kept, kept.index('q15_H1')),
       'B11': (Z[:, [names.index(c) for c in PN[:11]]], PN[:11], 0)}
def tail_indicators(V, frac):
    """Per answer: 1 if the step is in the column's top-k steps, centred within answer."""
    T = np.zeros_like(V)
    for i in range(len(off) - 1):
        a, b = int(off[i]), int(off[i + 1]); blk = V[a:b]; n_ = b - a
        kk = 1 if frac is None else max(1, int(np.ceil(frac * n_)))
        order = np.argsort(-blk, axis=0, kind='stable'); t = np.zeros_like(blk)
        np.put_along_axis(t, order[:kk], 1.0, axis=0); T[a:b] = t - t.mean(0)
    return T

CT7C = ['ct7_H0lim', 'ct7_ve0', 'ct7_ve0.75', 'ct7_ve1', 'ct7_H0lim_prefix_innovation', 'ct7_bocpd_residual', 'ct7_chosen_std_excess']
B11N = PN[:11]
def cols_from_pool(cs): return answer_standardize(POOL[:, [PN.index(c) for c in cs]].astype(np.float64), off)
XB = cols_from_pool(B11N); XC = cols_from_pool(CT7C)
sgnB = np.array([sign[names.index(c)] for c in B11N])
BANKS = {'B11': (XB, B11N, 0), 'B11_oriented': (XB * sgnB, B11N, 0), 'CT7s': (XC, CT7C, 0)}
RULES = ['equal', 'cov_lsml', 'tail20_lsml', 'tail1_lsml']
ARMS = [f'{b}_{r}' for b in BANKS for r in RULES] + ['CT7s_block421']
scores = {a: np.full(int(off[-1]), np.nan) for a in ARMS}; scores['ct7'] = Zs['ct7'].astype(float)
TB = {b: {'tail20': tail_indicators(X, 0.2), 'tail1': tail_indicators(X, None)} for b, (X, _m, _a) in BANKS.items()}
print('tail indicators built', flush=True)
blk = np.array([0, 0, 0, 0, 1, 1, 2]); w421 = 1.0 / (3 * np.bincount(blk)[blk])
fit_log = []; failures = []
for k in FOLDS:
    t = time.perf_counter(); cal = (k + 1) % 5; fitf = [f for f in range(5) if f not in (k, cal)]
    fit_rows = rows_of(np.isin(fold, fitf)); out_rows = np.concatenate([rows_of(fold == k), rows_of(fold == cal)])
    for b, (X, mem, anc) in BANKS.items():
        W = {'equal': np.ones(X.shape[1]) / X.shape[1]}
        W['cov_lsml'], m0 = fit_fusion_weights(X[fit_rows], FusionRecipe(name=b, members=tuple(mem), mode='continuous', anchor=anc), seed=FIT_SEED)
        groups_log = {'cov_lsml': np.asarray(m0['groups']).tolist()}
        for tk, T in TB[b].items():
            try:
                wl, ml = fit_fusion_weights(T[fit_rows], FusionRecipe(name=b + tk, members=tuple(mem), mode='continuous', anchor=anc), seed=FIT_SEED)
                W[f'{tk}_lsml'], _o = _orient(X[fit_rows], wl, anc); groups_log[f'{tk}_lsml'] = np.asarray(ml['groups']).tolist()
            except Exception as e:
                failures.append({'fold': k, 'arm': f'{b}_{tk}_lsml', 'reason': repr(e)})
        if b == 'CT7s': W['block421'] = w421
        for r, w in W.items(): scores[f'{b}_{r}'][out_rows] = X[out_rows] @ w
        fit_log.append({'fold': k, 'bank': b, 'groups': groups_log, 'weights': {r: {c: round(float(x), 4) for c, x in zip(mem, w / np.abs(w).sum())} for r, w in W.items()}})
    print(f'fold {k} done ({time.perf_counter()-t:.0f}s)', flush=True)
with open(OUT / 'FIT_MANIFEST.jsonl', 'w', encoding='utf8') as f:
    for r in fit_log: f.write(json.dumps(r) + '\n')
ev_all = rows_of(np.isin(fold, FOLDS)); METHODS = [m for m in scores if np.isfinite(scores[m][ev_all]).all()]
np.savez_compressed(OUT / 'STEP_SCORES.npz', offsets=off, **{m: scores[m] for m in METHODS})

# ------------------------------------------------------------------ evaluation (labels enter here only)
def within_auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def zt(s): return (s - s.mean()) / max(s.std(), 1e-8)
def prmscore_from_counts(tp, fp, tn, fn):
    def ratio(a, b): return np.divide(a, b, out=np.full(np.shape(a), -1., float), where=b != 0)
    p = ratio(tp, tp + fp); r = ratio(tp, tp + fn); f = ratio(2 * p * r, p + r); p2 = ratio(tn, tn + fn); r2 = ratio(tn, tn + fp); nf = ratio(2 * p2 * r2, p2 + r2); return (f + nf) / 2
auc, conf, official, pbhit, metrics = {}, {}, {}, {}, []
Gpr, ginv = np.unique(groups[prm], return_inverse=True); prm_pos = np.flatnonzero(prm)
pb_err = np.flatnonzero(pb & (target >= 0)); Gpb, gpbinv = np.unique(groups[pb_err], return_inverse=True)
for m in METHODS:
    s = scores[m]
    auc[m] = np.array([within_auc(labels[off[i]:off[i+1]], s[off[i]:off[i+1]]) if eligible[i] else np.nan for i in range(n)])
    v = np.zeros(int(off[-1]), bool)
    for k in FOLDS:
        cal = (k + 1) % 5; cs = np.concatenate([zt(s[off[i]:off[i+1]]) for i in np.flatnonzero(prm & (fold == cal))]); tau = float(np.quantile(cs, .8))
        for i in np.flatnonzero(prm & (fold == k)):
            a, b = off[i:i+2]; v[a:b] = zt(s[a:b]) < tau
    official[m] = prmbench_evaluate([{'idx': ids[i], 'labels': v[off[i]:off[i+1]].astype(int).tolist()} for i in prm_pos], [meta[ids[i]] for i in prm_pos])['total']
    good = ~labels; c = np.zeros((len(Gpr), 4))
    for gi, i in zip(ginv, prm_pos):
        if not noncontrol[i]: continue
        a, b = off[i:i+2]; vv = v[a:b]; gg = good[a:b]
        c[gi] += [(vv & gg).sum(), (vv & ~gg).sum(), (~vv & ~gg).sum(), (~vv & gg).sum()]
    conf[m] = c
    pbhit[m] = np.array([int(np.flatnonzero(s[off[i]:off[i+1]] >= s[off[i]:off[i+1]].max() - 8 * np.finfo(float).eps)[0]) == target[i] for i in pb_err], float)
    sla = [float(pbhit[m][cells[pb_err] == cell].mean()) for cell in sorted(set(cells[pb_err]))]
    metrics += [{'method': m, 'metric': 'prm_within_auc', 'N': int(np.isfinite(auc[m]).sum()), 'estimate': float(np.nanmean(auc[m]))},
                {'method': m, 'metric': 'prmscore', 'N': int(noncontrol.sum()), 'estimate': float(.5 * (official[m]['f1'] + official[m]['negative_f1'])), 'from_counts': float(prmscore_from_counts(*c.sum(0)))},
                {'method': m, 'metric': 'pb_sla_macro8', 'N': len(pb_err), 'estimate': float(np.mean(sla))}]
MT = pd.DataFrame(metrics); MT.to_csv(OUT / 'METRICS.csv', index=False)

# ------------------------------------------------------------------ paired source-group bootstrap
sums = {m: np.zeros(len(Gpr)) for m in METHODS}; cnts = np.zeros(len(Gpr))
for gi, i in zip(ginv, prm_pos):
    if eligible[i]:
        cnts[gi] += 1
        for m in METHODS: sums[m][gi] += auc[m][i]
pbs = {m: np.bincount(gpbinv, weights=pbhit[m], minlength=len(Gpb)) for m in METHODS}; pbc = np.bincount(gpbinv, minlength=len(Gpb)).astype(float)
CON = [(f'{b}_{r}', f'{b}_cov_lsml') for b in BANKS for r in ('tail20_lsml', 'tail1_lsml', 'equal')] \
    + [('CT7s_tail20_lsml', 'CT7s_block421'), ('CT7s_block421', 'ct7'), ('CT7s_tail20_lsml', 'ct7'), ('B11_tail20_lsml', 'ct7')]
rng = np.random.default_rng(SEED); est = {(m, e): np.empty(DRAWS) for m in METHODS for e in ('auc', 'ps', 'pb')}; pos = 0
while pos < DRAWS:
    nb = min(2000, DRAWS - pos)
    W = rng.multinomial(len(Gpr), np.full(len(Gpr), 1 / len(Gpr)), size=nb).astype(float); V = rng.multinomial(len(Gpb), np.full(len(Gpb), 1 / len(Gpb)), size=nb).astype(float)
    for m in METHODS:
        est[m, 'auc'][pos:pos+nb] = (W @ sums[m]) / (W @ cnts)
        cc = W @ conf[m]; est[m, 'ps'][pos:pos+nb] = prmscore_from_counts(cc[:, 0], cc[:, 1], cc[:, 2], cc[:, 3])
        est[m, 'pb'][pos:pos+nb] = (V @ pbs[m]) / (V @ pbc)
    pos += nb
point = {m: {'auc': float(np.nanmean(auc[m])), 'ps': float(prmscore_from_counts(*conf[m].sum(0))), 'pb': float(pbhit[m].mean())} for m in METHODS}
rows = []
for a, b in CON:
    if a not in METHODS or b not in METHODS: continue
    for e, lab in [('auc', 'prm_within_auc'), ('ps', 'prmscore'), ('pb', 'pb_sla_pooled')]:
        d = est[a, e] - est[b, e]
        rows.append({'contrast': f'{a} - {b}', 'endpoint': lab, 'delta': point[a][e] - point[b][e], 'ci95_lo': float(np.quantile(d, .025)), 'ci95_hi': float(np.quantile(d, .975))})
pd.DataFrame(rows).to_csv(OUT / 'CONTRASTS.csv', index=False)
status.update({'status': 'COMPLETE' if not failures else 'INCOMPLETE', 'finished': datetime.now().isoformat(timespec='seconds'), 'failures': failures, 'methods': METHODS, 'seconds': time.perf_counter() - T0}); dump(OUT / 'RUN_STATUS.json', status)
pd.set_option('display.width', 250)
print(MT.pivot(index='method', columns='metric', values='estimate').round(4).to_string())
print(pd.DataFrame(rows).round(4).to_string()); print(json.dumps(status, default=str))
