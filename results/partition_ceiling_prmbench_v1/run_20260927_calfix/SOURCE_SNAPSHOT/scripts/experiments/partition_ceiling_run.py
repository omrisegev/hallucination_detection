"""partition_ceiling_prmbench_v1: exhaustive ceiling over every block-equal partition of the
11-channel step bank, with the project's 3 fit / 1 calibration / 1 evaluation split.

Frozen protocol: results/partition_ceiling_prmbench_v1/PROTOCOL.json.
Labels enter (a) the declared LABEL-SELECTED arm's selection inside the three fit folds and
(b) evaluation. Every other arm is label-free.
"""
from pathlib import Path
import hashlib, itertools, json, os, pickle, shutil, subprocess, sys, time
from collections import Counter
from datetime import datetime
import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
DEPTH = MAIN / '.worktrees/depth-feature-fusion-v1'; sys.path.insert(0, str(DEPTH))
from spectral_utils.lsml_gate_locator_research import FusionRecipe, answer_standardize, fit_fusion_weights  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260927'
STAGE = ROOT / 'results/partition_ceiling_prmbench_v1'; OUT = STAGE / RUN_ID; OUT.mkdir(parents=True, exist_ok=True)
if (OUT / 'RUN_STATUS.json').exists(): raise SystemExit(f'{OUT} already holds a run; pass a new run id (old runs are evidence)')
P = json.loads((STAGE / 'PROTOCOL.json').read_text(encoding='utf8'))
DRAWS = int(os.environ.get('PC_DRAWS', 100_000)); SEED = 20260927; FIT_SEED = 20260919
NULL_SEEDS = [int(x) for x in os.environ.get('PC_NULL_SEEDS', '0,1,2').split(',')]
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
TPF = MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1'
INPUTS = {'level_bank': TPF / 'DERIVATIVE_CHANNELS.npz', 'oof_answers': R / 'OOF_ANSWERS.csv',
          'oof_step_scores': R / 'OOF_STEP_SCORES.npz', 'folds_v2': MAIN / 'results/localization_source_group_audit_v1/FOLDS_V2.json'}
T0 = time.perf_counter(); timing = {}
def dump(p, v): Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'run_id': RUN_ID}; dump(OUT / 'RUN_STATUS.json', status)

# ------------------------------------------------------------------ population
ans = pd.read_csv(INPUTS['oof_answers'], encoding='utf-8-sig'); Zs = np.load(INPUTS['oof_step_scores'])
off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans); ns = np.diff(off)
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy()
cells = ans.cell.to_numpy(); groups = ans.source_group.to_numpy(); ids = ans.id.to_numpy(); target = ans.target.to_numpy()
freeze = json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig')); meta_path = Path(freeze['prm_metadata']['path'])
meta = {m['idx']: m for m in pickle.load(open(meta_path, 'rb')).values()}
noncontrol = np.array([prm[i] and meta[ids[i]]['classification'] != 'correct' for i in range(n)])
elig = np.array([prm[i] and labels[off[i]:off[i + 1]].any() and (~labels[off[i]:off[i + 1]]).any() for i in range(n)])
eidx = np.flatnonzero(elig); efold = fold[eidx]

lv = np.load(INPUTS['level_bank']); NAMES = list(lv['channels'].astype(str)); m = 11
values = answer_standardize(lv['level'].astype(float), off)
print(f'{n} answers, {elig.sum()} eligible PRMBench, {time.perf_counter()-T0:.0f}s', flush=True)

# pair index restricted to eligible answers
keep = np.concatenate([np.arange(off[i], off[i + 1]) for i in eidx])
remap = -np.ones(int(off[-1]), np.int64); remap[keep] = np.arange(len(keep))
Xe = np.ascontiguousarray(values[keep], dtype=np.float32)
ip, ineg, starts, ylab = [], [], [], []
pos = 0
for i in eidx:
    s, e = off[i:i + 2]; yy = labels[s:e]; ylab.append(yy)
    p_ = remap[np.flatnonzero(yy) + s]; q_ = remap[np.flatnonzero(~yy) + s]
    A, B = np.meshgrid(p_, q_, indexing='ij')
    ip.append(A.ravel()); ineg.append(B.ravel()); starts.append(pos); pos += A.size
ip = np.concatenate(ip); ineg = np.concatenate(ineg)
starts = np.asarray(starts, np.int64); npair = np.diff(np.append(starts, pos)).astype(np.float32)
print(f'{len(ip):,} (error, clean) step pairs', flush=True)

# ------------------------------------------------------------------ enumerate every distinct block-equal weight vector
def int_partitions(t, mx=None):
    mx = t if mx is None else mx
    if t == 0: yield ()
    for k in range(min(t, mx), 0, -1):
        for rest in int_partitions(t - k, k): yield (k,) + rest
maps, Ks = [], []
for lam in int_partitions(m):
    K = len(lam); c = Counter(lam); sizes = sorted(c); nper = [s * c[s] for s in sizes]
    def rec(rem, si, cur):
        if si == len(sizes): maps.append(cur.copy()); Ks.append(K); return
        for pick in itertools.combinations(rem, nper[si]):
            ps = set(pick)
            for j in pick: cur[j] = sizes[si]
            rec(tuple(r for r in rem if r not in ps), si + 1, cur)
    rec(tuple(range(m)), 0, [0] * m)
SZ = np.asarray(maps, np.int16); KK = np.asarray(Ks, np.int16)
Wall = (1.0 / (KK[:, None].astype(np.float64) * SZ)).astype(np.float32)
Wall, uidx = np.unique(np.round(Wall, 10), axis=0, return_index=True); SZ, KK = SZ[uidx], KK[uidx]
V = len(Wall); Wt = np.ascontiguousarray(Wall.T)
assert abs(Wall.sum(1) - 1).max() < 1e-6, 'block-equal weights must sum to 1'
print(f'{V:,} distinct block-equal weight vectors (from 678,570 set partitions)  {time.perf_counter()-T0:.0f}s', flush=True)

# ------------------------------------------------------------------ per-answer AUC for every profile, accumulated per fold
def sweep(pair_win_labels):
    """Returns (V, 5) fold-mean AUC and (V,) overall, using the given per-pair orientation."""
    S = np.zeros((V, 5), np.float64); C5 = np.array([(efold == f).sum() for f in range(5)], float)
    fm = [efold == f for f in range(5)]
    CH = 512
    for s0 in range(0, V, CH):
        sc = Xe @ Wt[:, s0:s0 + CH]
        a = sc[ip]; b = sc[ineg]
        wins = (a > b).astype(np.float32); wins += 0.5 * (a == b)
        if pair_win_labels is not None: wins = np.where(pair_win_labels[:, None], wins, 1.0 - wins)
        per = np.add.reduceat(wins, starts, axis=0) / npair[:, None]
        for f in range(5): S[s0:s0 + CH, f] = per[fm[f]].mean(0)
        if (s0 // CH) % 100 == 0: print(f'   {s0:,}/{V:,} {time.perf_counter()-T0:.0f}s', flush=True)
    return S, (S * C5).sum(1) / C5.sum()

AUCF, AUC = sweep(None)
timing['sweep_s'] = time.perf_counter() - T0
np.savez_compressed(OUT / 'EXHAUSTIVE_AUC.npz', auc=AUC, auc_fold=AUCF, sizes=SZ, K=KK, names=np.array(NAMES))
order = np.argsort(-AUC)
nans = np.array([(efold == f).sum() for f in range(5)], float)
print(f'sweep done {timing["sweep_s"]:.0f}s; ceiling {AUC.max():.4f}', flush=True)

def find_w(wv):
    d = np.abs(Wall - np.asarray(wv, np.float32)).max(1); j = int(np.argmin(d)); return j if d[j] < 1e-6 else None
def prof(j): return {'auc_all': float(AUC[j]), 'K': int(KK[j]), 'sizes': sorted(Counter(SZ[j].tolist()).elements(), reverse=True),
                     'weights': {nm: round(float(w), 4) for nm, w in zip(NAMES, Wall[j])},
                     'rank': int(np.flatnonzero(order == j)[0]) + 1, 'per_fold': [round(float(v), 4) for v in AUCF[j]]}

# ------------------------------------------------------------------ arms
scores = {a: np.full(int(off[-1]), np.nan) for a in ['equal', 'lsml', 'declared_equal', 'energy_level_alone', 'profile_selected']}
for s in NULL_SEEDS: scores[f'null_seed{s}'] = np.full(int(off[-1]), np.nan)
scores['ct7'] = Zs['ct7'].astype(float)
# 2026-09-27 fix (LESSONS 2026-09-27): the old loop wrote each model's scores into ONE array for both roles, so fold 0
# ended up scored by the fold-4 model and the thresholds of folds 0-3 were read from the next fold's model. Now every
# fold-k model writes ONLY its evaluation rows (write-once), and its calibration-fold scores are kept apart for its own threshold.
cal_src = {}
def put(m, k, ev_rows, cal_rows, ev_vals, cal_vals):
    if not np.isnan(scores[m][ev_rows]).all(): raise RuntimeError(f'write-once violated: {m} evaluation rows, fold {k}')
    if (m, k) in cal_src: raise RuntimeError(f'write-once violated: {m} calibration scores, fold {k}')
    scores[m][ev_rows] = ev_vals; c = np.full(int(off[-1]), np.nan); c[cal_rows] = cal_vals; cal_src[(m, k)] = c
DECL = dict(q15_H1=5, q15_VE1=5, logprob_margin=5, true_tail50=5, energy_level=5,
            energy_innovation=3, top15_turnover=3, top50_js=3, chosen_surprisal=3, bocpd_p0=3, dominant_freq16=3)
w_eq = np.full(m, 1 / m); w_decl = np.array([1 / (3 * DECL[nm]) for nm in NAMES]); w_en = (np.array(NAMES) == 'energy_level').astype(float)
rows_of = lambda mask: np.concatenate([np.arange(off[i], off[i + 1]) for i in np.flatnonzero(mask)])
sel_log = []
rng_master = np.random.default_rng(SEED)
# shuffled-label selection null: shuffle labels WITHIN each fit-fold answer, re-select
null_fold_auc = {}
for s in NULL_SEEDS:
    rg = np.random.default_rng([SEED, s])
    flip = np.zeros(len(ip), bool); pos2 = 0
    for a_, i in enumerate(eidx):
        yy = ylab[a_]; k = npair[a_].astype(int)
        perm = rg.permutation(len(yy)); ysh = yy[perm]
        p_ = np.flatnonzero(yy); q_ = np.flatnonzero(~yy)
        A, B = np.meshgrid(p_, q_, indexing='ij')
        flip[pos2:pos2 + k] = ysh[A.ravel()] > ysh[B.ravel()]          # pair still counts only if shuffled labels disagree the same way
        pos2 += k
    null_fold_auc[s] = sweep(flip)[0]
    print(f'null seed {s} sweep done {time.perf_counter()-T0:.0f}s', flush=True)

for k in range(5):
    cal = (k + 1) % 5; fitf = [f for f in range(5) if f not in (k, cal)]
    ev = rows_of(fold == k); cl = rows_of(fold == cal); fit_rows = rows_of(np.isin(fold, fitf))
    wsel = nans[fitf] / nans[fitf].sum()
    j_sel = int(np.argmax(AUCF[:, fitf] @ wsel))
    for arm, w in [('equal', w_eq), ('declared_equal', w_decl), ('energy_level_alone', w_en), ('profile_selected', Wall[j_sel])]:
        put(arm, k, ev, cl, values[ev] @ w, values[cl] @ w)
    for s in NULL_SEEDS:
        jn = int(np.argmax(null_fold_auc[s][:, fitf] @ wsel))
        put(f'null_seed{s}', k, ev, cl, values[ev] @ Wall[jn], values[cl] @ Wall[jn])
        sel_log.append({'fold': k, 'arm': f'null_seed{s}', 'sizes': sorted(Counter(SZ[jn].tolist()).elements(), reverse=True), 'weights': {nm: round(float(v), 4) for nm, v in zip(NAMES, Wall[jn])}})
    wl, ml = fit_fusion_weights(values[fit_rows], FusionRecipe(name='lsml', members=tuple(NAMES), mode='continuous', anchor=0), seed=FIT_SEED)
    put('lsml', k, ev, cl, values[ev] @ wl, values[cl] @ wl)
    sel_log.append({'fold': k, 'arm': 'profile_selected', 'fit_folds': fitf, 'cal_fold': cal, 'sizes': sorted(Counter(SZ[j_sel].tolist()).elements(), reverse=True),
                    'weights': {nm: round(float(v), 4) for nm, v in zip(NAMES, Wall[j_sel])}, 'rank_overall': int(np.flatnonzero(order == j_sel)[0]) + 1,
                    'auc_on_fit_folds': float(AUCF[j_sel, fitf] @ wsel), 'auc_on_eval_fold': float(AUCF[j_sel, k])})
    sel_log.append({'fold': k, 'arm': 'lsml', 'K': int(ml['K']), 'groups': list(map(int, ml['groups'])), 'weights': {nm: round(float(v), 4) for nm, v in zip(NAMES, wl)}})
    print(f'fold {k}: fit {fitf} cal {cal} -> selected sizes {sel_log[-2]["sizes"]} (rank {sel_log[-2]["rank_overall"]:,})', flush=True)
timing['fits_s'] = time.perf_counter() - T0
ARMS = ['equal', 'lsml', 'declared_equal', 'energy_level_alone', 'profile_selected'] + [f'null_seed{s}' for s in NULL_SEEDS]
METHODS = ARMS + ['ct7']   # saved for audit (added with the 2026-09-27 fix)
np.savez_compressed(OUT / 'STEP_SCORES.npz', offsets=off, **{m: scores[m] for m in METHODS})
np.savez_compressed(OUT / 'CAL_SCORES.npz', offsets=off, **{f'{m}__fold{k}': v for (m, k), v in cal_src.items() if m in METHODS})   # threshold inputs, same model
with open(OUT / 'SELECTION_LOG.jsonl', 'w', encoding='utf8') as f:
    for r in sel_log: f.write(json.dumps(r, default=float) + '\n')

# ------------------------------------------------------------------ evaluation
def within_auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def zt(s): return (s - s.mean()) / max(s.std(), 1e-8)
def ps_from_counts(tp, fp, tn, fn):
    def r(a, b): return np.divide(a, b, out=np.full(np.shape(a), -1., float), where=b != 0)
    p = r(tp, tp + fp); rc = r(tp, tp + fn); f1 = r(2 * p * rc, p + rc)
    p2 = r(tn, tn + fn); r2 = r(tn, tn + fp); nf = r(2 * p2 * r2, p2 + r2); return (f1 + nf) / 2
METHODS = ARMS + ['ct7']
auc = {}; conf = {}; metrics = []
Gpr, ginv = np.unique(groups[prm], return_inverse=True); prm_pos = np.flatnonzero(prm)
for mth in METHODS:
    s = scores[mth]
    auc[mth] = np.array([within_auc(labels[off[i]:off[i + 1]], s[off[i]:off[i + 1]]) if elig[i] else np.nan for i in range(n)])
    v = np.zeros(int(off[-1]), bool)
    for k in range(5):
        cal = (k + 1) % 5; src = s if mth == 'ct7' else cal_src[(mth, k)]   # the SAME fold-k model's calibration-fold scores; ct7 is fixed
        cs = np.concatenate([zt(src[off[i]:off[i + 1]]) for i in np.flatnonzero(prm & (fold == cal))]); tau = float(np.quantile(cs, .8))
        for i in np.flatnonzero(prm & (fold == k)):
            a, b = off[i:i + 2]; v[a:b] = zt(s[a:b]) < tau
    off_ = prmbench_evaluate([{'idx': ids[i], 'labels': v[off[i]:off[i + 1]].astype(int).tolist()} for i in prm_pos], [meta[ids[i]] for i in prm_pos])['total']
    good = ~labels; c = np.zeros((len(Gpr), 4))
    for gi, i in zip(ginv, prm_pos):
        if not noncontrol[i]: continue
        a, b = off[i:i + 2]; vv = v[a:b]; gg = good[a:b]
        c[gi] += [(vv & gg).sum(), (vv & ~gg).sum(), (~vv & ~gg).sum(), (~vv & gg).sum()]
    conf[mth] = c
    el = np.isfinite(auc[mth])
    metrics.append({'method': mth, 'metric': 'within_auc', 'stratum': 'all', 'N': int(el.sum()), 'estimate': float(np.nanmean(auc[mth]))})
    metrics.append({'method': mth, 'metric': 'prmscore_answer_z_q80', 'stratum': 'all', 'N': int(noncontrol.sum()), 'estimate': float(.5 * (off_['f1'] + off_['negative_f1']))})
    sla = []
    for cell in sorted(set(cells[pb])):
        e = (cells == cell) & (target >= 0)
        if e.any(): sla.append(float(np.mean([int(np.flatnonzero(s[off[i]:off[i+1]] >= s[off[i]:off[i+1]].max() - 8 * np.finfo(float).eps)[0]) == target[i] for i in np.flatnonzero(e)])))
    metrics.append({'method': mth, 'metric': 'sla', 'stratum': 'macro8_context', 'N': int((pb & (target >= 0)).sum()), 'estimate': float(np.mean(sla))})
pd.DataFrame(metrics).to_csv(OUT / 'METRICS.csv', index=False)

# ------------------------------------------------------------------ paired bootstrap
sums = {mth: np.zeros(len(Gpr)) for mth in METHODS}; cnts = np.zeros(len(Gpr))
for gi, i in zip(ginv, prm_pos):
    if elig[i]:
        cnts[gi] += 1
        for mth in METHODS: sums[mth][gi] += auc[mth][i]
rng = np.random.default_rng(SEED); eA = {mth: np.empty(DRAWS) for mth in METHODS}; eP = {mth: np.empty(DRAWS) for mth in METHODS}; pos = 0
while pos < DRAWS:
    nb = min(5000, DRAWS - pos); Wd = rng.multinomial(len(Gpr), np.full(len(Gpr), 1 / len(Gpr)), size=nb).astype(float); den = Wd @ cnts
    for mth in METHODS:
        eA[mth][pos:pos + nb] = (Wd @ sums[mth]) / den
        cc = Wd @ conf[mth]; eP[mth][pos:pos + nb] = ps_from_counts(cc[:, 0], cc[:, 1], cc[:, 2], cc[:, 3])
    pos += nb
pt = {mth: {'auc': float(np.nanmean(auc[mth])), 'ps': float(ps_from_counts(*conf[mth].sum(0)))} for mth in METHODS}
prim = [('profile_selected', 'equal'), ('profile_selected', 'lsml'), ('profile_selected', 'ct7'), ('profile_selected', f'null_seed{NULL_SEEDS[0]}')]
sec = [('profile_selected', 'energy_level_alone'), ('energy_level_alone', 'equal'), ('energy_level_alone', 'ct7'), ('lsml', 'equal'), ('declared_equal', 'equal'), ('lsml', 'ct7'), ('declared_equal', 'ct7')]
sec += [('profile_selected', f'null_seed{s}') for s in NULL_SEEDS[1:]] + [(f'null_seed{s}', 'equal') for s in NULL_SEEDS]
K = 8; rows = []; deltas = {}
for a, b in prim + sec:
    if a not in METHODS or b not in METHODS: continue
    pr = (a, b) in prim
    for ep, est, key in [('prm_within_auc', eA, 'auc'), ('prmscore_answer_z_q80', eP, 'ps')]:
        d = est[a] - est[b]; deltas[f'{a}__minus__{b}__{ep}'] = d.astype(np.float32)
        rows.append({'contrast_id': f'{a} - {b}', 'primary': pr, 'endpoint': ep, 'delta': pt[a][key] - pt[b][key],
                     'ci95_lo': float(np.quantile(d, .025)), 'ci95_hi': float(np.quantile(d, .975)),
                     'ci_adj_lo': float(np.quantile(d, .05 / K / 2)) if pr else None, 'ci_adj_hi': float(np.quantile(d, 1 - .05 / K / 2)) if pr else None,
                     'family_K': K if pr else None, 'B': DRAWS, 'paired_groups': len(Gpr)})
pd.DataFrame(rows).to_csv(OUT / 'CONTRASTS.csv', index=False); np.savez_compressed(OUT / 'BOOTSTRAP_DELTAS.npz', **deltas, seed=SEED, draws=DRAWS)

# ------------------------------------------------------------------ manifests + ranking of the label-free arms
diag = {'n_distinct_weight_vectors': V, 'ceiling': prof(int(order[0])), 'worst': prof(int(order[-1])),
        'quantiles': {k: float(np.quantile(AUC, v)) for k, v in [('min', 0), ('p01', .01), ('p25', .25), ('median', .5), ('p75', .75), ('p99', .99), ('max', 1)]},
        'rank_of_label_free_arms': {}, 'selected_identical_in_all_folds': None}
for nm, wv in [('equal', w_eq), ('declared_5_3_3', w_decl)]:
    j = find_w(wv); diag['rank_of_label_free_arms'][nm] = prof(j) if j is not None else None
diag['n_profiles_above_equal'] = int((AUC > AUC[find_w(w_eq)]).sum())
selsz = [tuple(r['sizes']) for r in sel_log if r['arm'] == 'profile_selected']
diag['selected_identical_in_all_folds'] = bool(len(set(selsz)) == 1); diag['selected_sizes_per_fold'] = [list(s) for s in selsz]
dump(OUT / 'DIAGNOSTICS.json', diag)
dump(OUT / 'INPUT_MANIFEST.json', {k: {'path': str(v), 'bytes': v.stat().st_size, 'sha256': sha(v)} for k, v in INPUTS.items()} |
     {'channels': NAMES, 'population': {'answers': n, 'eligible': int(elig.sum()), 'noncontrol': int(noncontrol.sum()), 'steps': int(off[-1])}})
snap = OUT / 'SOURCE_SNAPSHOT'; code = {}
for rel, base in [('scripts/experiments/partition_ceiling_run.py', ROOT), ('spectral_utils/lsml_gate_locator_research.py', DEPTH), ('spectral_utils/fusion_utils.py', DEPTH)]:
    dst = snap / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(base / rel, dst); code[f'{base.name}/{rel}'] = sha(base / rel)
git = lambda cwd, *a: subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head': git(ROOT, 'rev-parse', 'HEAD'), 'protocol_sha256': sha(STAGE / 'PROTOCOL.json')})
timing['total_s'] = time.perf_counter() - T0; dump(OUT / 'TIMING.json', timing)
status.update({'status': 'COMPLETE', 'finished': datetime.now().isoformat(timespec='seconds'), 'methods': METHODS,
               'selected_identical_in_all_folds': diag['selected_identical_in_all_folds']}); dump(OUT / 'RUN_STATUS.json', status)
M = pd.DataFrame(metrics); print(M[M.stratum.isin(['all', 'macro8_context'])].pivot(index='method', columns='metric', values='estimate').round(4).to_string())
print(pd.DataFrame(rows)[['contrast_id', 'primary', 'endpoint', 'delta', 'ci95_lo', 'ci95_hi', 'ci_adj_lo', 'ci_adj_hi']].round(4).to_string(index=False))
print(json.dumps({k: diag[k] for k in ['n_distinct_weight_vectors', 'n_profiles_above_equal', 'selected_identical_in_all_folds', 'quantiles']}, indent=1, default=str))
print('ceiling', diag['ceiling']['auc_all'], diag['ceiling']['sizes'], '| equal rank', diag['rank_of_label_free_arms']['equal']['rank'], '| declared rank', diag['rank_of_label_free_arms']['declared_5_3_3']['rank'])
print(json.dumps(timing, indent=1))
