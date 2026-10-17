"""S1 orchestration: pseudo-label targets on one linear student (plan v1.1, section 7).

Frozen protocol: results/ssl_pseudolabel_residual_v1/S1/PROTOCOL.json.  Outputs to S1/<run_id>/.
Labels enter ONLY the evaluator / diagnostics section; fit_head never sees them (tests/test_ssl_s1.py).
"""
from pathlib import Path
import hashlib, json, pickle, shutil, subprocess, sys, time
from collections import Counter
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); W = MAIN / '.worktrees/readout-quickest-detection-v1'; R = W / 'results/step_evidence_v1'
from spectral_utils import ssl_s1 as S
sys.path.insert(0, str(MAIN)); from spectral_utils.prmbench import prmbench_evaluate

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260923'
STAGE = ROOT / 'results/ssl_pseudolabel_residual_v1/S1'; OUT = STAGE / RUN_ID; OUT.mkdir(parents=True, exist_ok=True)
P = json.loads((STAGE / 'PROTOCOL.json').read_text(encoding='utf8'))
ARMS = ['P_HARD', 'P_SOFT', 'P_AGREE', 'P_RANDOM', 'P_POSITION_LENGTH', 'P_SOFT_COVERAGE_MATCH']
TASKS = ['pb_q4', 'pb_q8', 'prm']
COMPARATORS = ['ct7', 'token_lsml', 'token_equal', 'evidence30__all__plain__equal', 'evidence__all__position__equal', 'evidence__all__position2__equal']
DRAWS = 100_000; SEED = 20260923
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p, v): Path(p).write_text(json.dumps(v, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
T0 = time.perf_counter(); timing = {}
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'run_id': RUN_ID}; dump(OUT / 'RUN_STATUS.json', status)

# ------------------------------------------------------------------ inputs + manifests
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Zs = np.load(R / 'OOF_STEP_SCORES.npz')
off = Zs['offsets']; labels = Zs['labels']; n = len(ans); nsteps = np.diff(off)
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); cells = ans.cell.to_numpy(); groups = ans.source_group.to_numpy()
uid = ans.uid.to_numpy(); ids = ans.id.to_numpy(); target = ans.target.to_numpy(); gate = ans.ct7_gate.to_numpy().astype(bool)
prof = np.load(R / 'profiles_full.npy', mmap_mode='r'); Pmat = np.ascontiguousarray(prof[:, :, 0])           # frozen top5
step_len = np.load(R / 'step_lengths.npy')
freeze = read(R / 'INPUT_FREEZE.json'); meta = {m['idx']: m for m in pickle.load(open(freeze['prm_metadata']['path'], 'rb')).values()}
noncontrol = np.array([prm[i] and meta[ids[i]]['classification'] != 'correct' for i in range(n)])
inputs = {'profiles_full.npy': R / 'profiles_full.npy', 'OOF_ANSWERS.csv': R / 'OOF_ANSWERS.csv', 'OOF_STEP_SCORES.npz': R / 'OOF_STEP_SCORES.npz', 'step_lengths.npy': R / 'step_lengths.npy',
          'prmbench_prm.pkl': Path(freeze['prm_metadata']['path']), 'FOLDS_V2.json': MAIN / 'results/localization_source_group_audit_v1/FOLDS_V2.json'}
dump(OUT / 'INPUT_MANIFEST.json', {k: {'path': str(v), 'bytes': v.stat().st_size, 'sha256': sha(v)} for k, v in inputs.items()} | {'population': {'answers': n, 'pb': int(pb.sum()), 'prm': int(prm.sum()), 'steps': int(off[-1])}, 'labels': 'v3 (joined in Step 432, verified in S0)', 'folds': 'source_groups_v2', 'representation': 'profiles_full[:, :, 0] = frozen top5'})
snap = OUT / 'SOURCE_SNAPSHOT'; snap.mkdir(exist_ok=True); code = {}
for rel in ['spectral_utils/ssl_s1.py', 'scripts/experiments/ssl_s1_run.py', 'tests/test_ssl_s1.py']:
    dst = snap / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(ROOT / rel, dst); code[rel] = sha(ROOT / rel)
git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head': git('rev-parse', 'HEAD'), 'dirty': git('status', '--porcelain', '--', 'spectral_utils/ssl_s1.py', 'scripts/experiments/ssl_s1_run.py', 'tests/test_ssl_s1.py'), 'protocol_sha256': sha(STAGE / 'PROTOCOL.json')})

# ------------------------------------------------------------------ representation, teachers
t = time.perf_counter()
Zall = np.empty_like(Pmat); q_teacher = np.empty(int(off[-1]))
for i in range(n):
    a, b = off[i:i+2]; Zall[a:b] = S.answer_z(Pmat[a:b])
    q_teacher[a:b] = S.teacher_pb(Zall[a:b]) if pb[i] else S.teacher_prm(Zall[a:b])
timing['representation_s'] = time.perf_counter() - t
task_of = np.where(prm, 'prm', np.where(np.char.endswith(cells.astype(str), 'q4'), 'pb_q4', 'pb_q8'))
def role_masks(task, k):
    m = task_of == task
    return {'H': m & (fold == k), 'C': m & (fold == (k + 1) % 5), 'B': m & (fold == (k + 2) % 5), 'A': m & np.isin(fold, [(k + 3) % 5, (k + 4) % 5])}
splits = {}
for task in TASKS:
    for k in range(5):
        rm = role_masks(task, k); gs = {r: set(groups[m]) for r, m in rm.items()}
        splits[f'{task}/fold{k}'] = {r: {'answers': int(m.sum()), 'groups': len(gs[r])} for r, m in rm.items()} | {'pairwise_group_overlap': int(sum(len(gs[x] & gs[y]) for x in gs for y in gs if x < y)), 'covers_task': bool(sum(m.sum() for m in rm.values()) == (task_of == task).sum())}
assert all(v['pairwise_group_overlap'] == 0 and v['covers_task'] for v in splits.values())
dump(OUT / 'SPLITS.json', splits)

# ------------------------------------------------------------------ fits
scores = {'BASE': q_teacher.copy()}; fit_log = []; failures = []; pseudo = {}; coverage_rows = []
cal_scores = {}   # (arm, task, k) -> scores on C from the model fitted on B
for arm in ARMS:
    scores[arm] = np.full(int(off[-1]), np.nan)
for task in TASKS:
    is_pb = task != 'prm'
    for k in range(5):
        rm = role_masks(task, k); B = np.flatnonzero(rm['B']); H = np.flatnonzero(rm['H']); C = np.flatnonzero(rm['C'])
        w_base = S.answer_weights(cells[B], groups[B], task)
        agree_kept_bins = Counter(); agree_pb_keep = {}; agree_prm_sel = {}
        for arm in ARMS:
            t = time.perf_counter()
            X_list, T_list, M_list, w_mult, keep, infos = [], [], [], [], [], []
            for j, i in enumerate(B):
                a, b = off[i:i+2]; Z = Zall[a:b]; rng = np.random.default_rng(np.random.SeedSequence([SEED, int(i)]))
                if is_pb:
                    tg, mult, kp, info = S.targets_pb(Z, arm, rng=rng); M_list.append(None)
                    if arm == 'P_AGREE':
                        agree_pb_keep[i] = kp
                        if kp: agree_kept_bins[(cells[i], S.depth_bin(b - a), S.rel_bin(S.first_argmax(q_teacher[a:b]), b - a))] += 1
                else:
                    tg, m, sel, info = S.targets_prm(Z, arm, rng=rng); mult = 1.; kp = bool(sel.any()); M_list.append(m * sel)
                    if arm == 'P_AGREE':
                        agree_prm_sel[i] = sel
                        for s in np.flatnonzero(sel): agree_kept_bins[('prm', S.depth_bin(b - a), S.rel_bin(s, b - a))] += 1
                X_list.append(Z if arm != 'P_POSITION_LENGTH' else S.position_length_features(b - a, step_len[a:b]))
                T_list.append(tg); w_mult.append(mult); keep.append(kp); infos.append(info)
            keep = np.array(keep); w_mult = np.array(w_mult)
            if arm == 'P_SOFT_COVERAGE_MATCH':
                if is_pb:
                    cands = [((cells[i], S.depth_bin(nsteps[i]), S.rel_bin(S.first_argmax(q_teacher[off[i]:off[i+1]]), nsteps[i])), S.coverage_key(uid[i], 'answer'), int(i)) for i in B]
                    chosen = S.coverage_match(agree_kept_bins, cands); keep = np.array([int(i) in chosen for i in B]); w_mult = np.ones(len(B))
                else:
                    cands = [(('prm', S.depth_bin(nsteps[i]), S.rel_bin(s, nsteps[i])), S.coverage_key(uid[i], int(s)), (int(i), int(s))) for i in B for s in range(nsteps[i])]
                    chosen = S.coverage_match(agree_kept_bins, cands)
                    for j, i in enumerate(B):
                        sel = np.array([(int(i), s) in chosen for s in range(nsteps[i])]); M_list[j] = sel.astype(float); keep[j] = sel.any()
            w = w_base * w_mult * keep
            if is_pb: w = w * np.array([len(tg) > 1 for tg in T_list])
            if arm == 'P_POSITION_LENGTH':
                rows = np.concatenate(X_list); wrow = np.concatenate([np.full(len(x), w[j] / len(x)) for j, x in enumerate(X_list)])
                mu, sd = S.weighted_scaler(rows, wrow if wrow.sum() > 0 else np.ones(len(rows))); X_list = [(x - mu) / sd for x in X_list]
            n_train = int((w > 0).sum())
            coverage_rows.append({'task': task, 'fold': k, 'arm': arm, 'B_answers': len(B), 'trainable_answers': n_train, 'trainable_fraction': n_train / len(B),
                                  'selected_steps': int(sum(m.sum() > 0 and (m > 0).sum() for m in M_list if m is not None)) if not is_pb else None, 'total_steps': int(nsteps[B].sum())})
            try:
                if n_train == 0: raise ValueError('UNTRAINABLE: no source with positive weight')
                model = S.fit_head(X_list, T_list, w, 'pb' if is_pb else 'prm', M_list=None if is_pb else M_list)
            except Exception as e:
                failures.append({'stage': 'S1', 'task': task, 'fold': k, 'arm': arm, 'reason': str(e), 'retry': 0, 'fallback': 'none (arm marked UNTRAINABLE for this fold)'}); continue
            if arm == 'P_POSITION_LENGTH': model['scaler'] = {'mu': mu, 'sd': sd}
            fit_log.append({'model_id': f'{task}/fold{k}/{arm}', 'role': 'B', 'train_groups_sha256': hashlib.sha256(','.join(sorted(set(groups[B]))).encode()).hexdigest(), 'train_answers': len(B), 'contributing': model['contributing_answers'],
                            'converged': model['converged'], 'status': model['status'], 'message': model['message'], 'iterations': model['iterations'], 'loss': model['loss'], 'grad_inf_norm': model['grad_inf_norm'],
                            'w': model['w'].tolist(), 'b': model['b'], 'ridge': S.RIDGE, 'seed': SEED, 'elapsed_s': time.perf_counter() - t, 'device': 'cpu'})
            feats = lambda i: Zall[off[i]:off[i+1]] if arm != 'P_POSITION_LENGTH' else (S.position_length_features(nsteps[i], step_len[off[i]:off[i+1]]) - mu) / sd
            for i in H: scores[arm][off[i]:off[i+1]] = S.predict_head(feats(i), model)
            if not is_pb: cal_scores[(arm, k)] = {int(i): S.predict_head(feats(i), model) for i in C}
            pseudo[f'{task}/fold{k}/{arm}'] = {'B': B, 'targets': np.concatenate(T_list), 'offsets': np.concatenate([[0], np.cumsum([len(x) for x in T_list])]), 'weights': w, 'keep': keep,
                                             'step_weights': (np.concatenate(M_list) if not is_pb else np.array([])), 'info': infos}
        # PRMB C-scores for BASE
        if not is_pb: cal_scores[('BASE', k)] = {int(i): q_teacher[off[i]:off[i+1]] for i in C}
    print(f'{task} fitted ({time.perf_counter()-T0:.0f}s)', flush=True)
timing['fit_s'] = time.perf_counter() - T0 - timing['representation_s']
with open(OUT / 'FIT_MANIFEST.jsonl', 'w', encoding='utf8') as f:
    for r in fit_log: f.write(json.dumps(r) + '\n')
pd.DataFrame(coverage_rows).to_csv(OUT / 'TRAINING_COVERAGE.csv', index=False)
pd.DataFrame(failures or [{'stage': 'S1', 'task': '', 'fold': '', 'arm': '', 'reason': 'none', 'retry': 0, 'fallback': ''}]).to_csv(OUT / 'FAILURES.csv', index=False)
np.savez_compressed(OUT / 'PSEUDO_TARGETS.npz', **{k.replace('/', '__') + '__' + f: v for k, d in pseudo.items() for f, v in d.items() if f != 'info'}, teacher_version='plan5.2_stepz_softmax_sigmoid_T1')
for c in COMPARATORS: scores[c] = Zs[c]
METHODS = ['BASE'] + ARMS + COMPARATORS
covered = {m: np.array([np.isfinite(scores[m][off[i]:off[i+1]]).all() for i in range(n)]) for m in METHODS}
np.savez_compressed(OUT / 'OOF_STEP_SCORES.npz', offsets=off, uid=uid, **{m: scores[m] for m in METHODS})

# ------------------------------------------------------------------ evaluation (labels enter here only)
t = time.perf_counter()
pred = {m: np.array([S.first_argmax(scores[m][off[i]:off[i+1]]) if covered[m][i] and pb[i] else -999 for i in range(n)]) for m in METHODS}
auc = {m: np.array([S.within_auc(labels[off[i]:off[i+1]], scores[m][off[i]:off[i+1]]) if prm[i] and covered[m][i] else np.nan for i in range(n)]) for m in METHODS}
oa = ans[['uid', 'id', 'source_group', 'fold', 'cell', 'target', 'ct7_gate']].copy(); oa['n_steps'] = nsteps; oa['task'] = task_of
for m in METHODS: oa[m + '__pred'] = pred[m]; oa[m + '__within_auc'] = auc[m]; oa[m + '__covered'] = covered[m]
oa.to_csv(OUT / 'OOF_ANSWERS.csv', index=False)
pd.DataFrame({'uid': np.repeat(uid, nsteps), 'step': np.concatenate([np.arange(s) for s in nsteps]), 'label': labels}).to_csv(OUT / 'EVAL_JOINED.csv', index=False)
pb_cells = sorted(set(cells[pb])); metrics = []
def pb_block(m):
    rows = {}
    for c in pb_cells:
        take = (cells == c) & covered[m]; e = take & (target >= 0); cl = take & (target < 0); d = pred[m][e] - target[e]
        ca = float((~gate[cl]).mean()); ea = float(((d == 0) & gate[e]).mean())
        rows[c] = {'n': int(take.sum()), 'errors': int(e.sum()), 'sla': float((d == 0).mean()), 'tol1': float((abs(d) <= 1).mean()), 'early': float((d < 0).mean()), 'late': float((d > 0).mean()), 'f1': 2 * ca * ea / (ca + ea) if ca + ea else 0., 'clean_acc': ca, 'gated_err_acc': ea}
    return rows
qgrid = np.round(np.linspace(.5, .99, 50), 2); elig_steps = np.repeat(noncontrol, nsteps); prm_steps = np.repeat(prm, nsteps); prm_idx = np.flatnonzero(prm)
def prmscore_counts(valid, y, e):
    v = np.asarray(valid, bool)[..., e]; good = y[e] == 0; tp = (v & good).sum(-1); fp = (v & ~good).sum(-1); tn = (~v & ~good).sum(-1); fn = (~v & good).sum(-1)
    def ratio(a, b): return np.divide(a, b, out=np.full(np.shape(a), -1., float), where=b != 0)
    p = ratio(tp, tp + fp); r = ratio(tp, tp + fn); f = ratio(2 * p * r, p + r); p = ratio(tn, tn + fn); r = ratio(tn, tn + fp); nf = ratio(2 * p * r, p + r); return (f + nf) / 2
def official(valid, idx):
    r = prmbench_evaluate([{'idx': ids[i], 'labels': valid[off[i]:off[i+1]].astype(int).tolist()} for i in idx], [meta[ids[i]] for i in idx])['total']; return .5 * (r['f1'] + r['negative_f1'])
def zt(s): return (s - s.mean()) / max(s.std(), 1e-8)
prmscore = {}
for m in ['BASE'] + ARMS:
    for variant, T in [('raw', lambda s: s), ('answer_z', zt)]:
        v_sel = np.zeros(int(off[-1]), bool); v_q80 = v_sel.copy(); ok = True
        for k in range(5):
            if (m, k) not in cal_scores: ok = False; break
            cs = np.concatenate([T(v) for v in cal_scores[(m, k)].values()]); cy = np.concatenate([labels[off[i]:off[i+1]] for i in cal_scores[(m, k)]]); ce = np.concatenate([np.full(nsteps[i], noncontrol[i]) for i in cal_scores[(m, k)]])
            thr = np.quantile(cs, qgrid); grid = prmscore_counts(cs[None, :] < thr[:, None], cy, ce); best = int(np.argmax(grid)); tau80 = float(np.quantile(cs, .8))
            for i in np.flatnonzero(prm & (fold == k)):
                a, b = off[i:i+2]; s = T(scores[m][a:b]); v_sel[a:b] = s < thr[best]; v_q80[a:b] = s < tau80
        if ok: prmscore[(m, variant)] = {'inner': official(v_sel, prm_idx), 'q80': official(v_q80, prm_idx)}
for m in COMPARATORS:   # comparators: fixed run values are the reference (S0 replayed them exactly); recompute official on covered answers with the run's decisions unavailable here -> report raw-score q80 only
    idx = prm_idx[covered[m][prm_idx]]; s = scores[m]
    v = np.zeros(int(off[-1]), bool)
    for k in range(5):
        trs = np.repeat(prm & covered[m] & (fold != k), nsteps); tau = np.quantile(s[trs], .8); tst = np.repeat(prm & covered[m] & (fold == k), nsteps); v[tst] = s[tst] < tau
    prmscore[(m, 'raw')] = {'q80': official(v, idx)}
for m in METHODS:
    blk = pb_block(m)
    for c, r in blk.items():
        for key in ['sla', 'tol1', 'early', 'late', 'f1', 'clean_acc', 'gated_err_acc']: metrics.append({'method': m, 'benchmark': 'pb', 'metric': key, 'stratum': c, 'N': r['errors'] if key != 'clean_acc' else r['n'] - r['errors'], 'estimate': r[key], 'coverage': r['n']})
    for scope, suf in [('macro8', ''), ('q4', 'q4'), ('q8', 'q8')]:
        vals = [r for c, r in blk.items() if c.endswith(suf)]
        for key in ['sla', 'tol1', 'early', 'late', 'f1']: metrics.append({'method': m, 'benchmark': 'pb', 'metric': key, 'stratum': scope, 'N': sum(r['errors'] for r in vals), 'estimate': float(np.mean([r[key] for r in vals])), 'coverage': sum(r['n'] for r in vals)})
    el = np.isfinite(auc[m]); metrics.append({'method': m, 'benchmark': 'prm', 'metric': 'within_auc', 'stratum': 'all', 'N': int(el.sum()), 'estimate': float(np.nanmean(auc[m])), 'coverage': int((prm & covered[m]).sum())})
    fa, fp_ = [], []
    for k in range(5):
        st = np.repeat(prm & covered[m] & (fold == k), nsteps); fa.append(roc_auc_score(labels[st], scores[m][st])); fp_.append(average_precision_score(labels[st], scores[m][st]))
    metrics += [{'method': m, 'benchmark': 'prm', 'metric': 'step_auroc_mean_folds', 'stratum': 'all', 'N': int(prm_steps.sum()), 'estimate': float(np.mean(fa)), 'coverage': int((prm & covered[m]).sum())}, {'method': m, 'benchmark': 'prm', 'metric': 'step_auprc_mean_folds', 'stratum': 'all', 'N': int(prm_steps.sum()), 'estimate': float(np.mean(fp_)), 'coverage': int((prm & covered[m]).sum())}]
    for (mm, variant), d in prmscore.items():
        if mm == m:
            for kind, val in d.items(): metrics.append({'method': m, 'benchmark': 'prm', 'metric': f'prmscore_{kind}_{variant}', 'stratum': 'all', 'N': int(noncontrol.sum()), 'estimate': val, 'coverage': int((prm & covered[m]).sum())})
    # by kind / depth for PRM
    kind_of = np.array(['' if not prm[i] else 'clean' if labels[off[i]:off[i+1]].sum() == 0 else 'single' if labels[off[i]:off[i+1]].sum() == 1 else 'multi' for i in range(n)])
    for kd in ['single', 'multi']:
        sel = (kind_of == kd) & np.isfinite(auc[m]); metrics.append({'method': m, 'benchmark': 'prm', 'metric': 'within_auc', 'stratum': f'kind={kd}', 'N': int(sel.sum()), 'estimate': float(np.nanmean(auc[m][sel])), 'coverage': int(sel.sum())})
    for db in ['2-5', '6-10', '11+']:
        e = pb & (target >= 0) & covered[m] & np.array([S.depth_bin(s) == db for s in nsteps]); metrics.append({'method': m, 'benchmark': 'pb', 'metric': 'sla', 'stratum': f'depth={db}', 'N': int(e.sum()), 'estimate': float((pred[m][e] == target[e]).mean()), 'coverage': int(e.sum())})
        sel = prm & np.isfinite(auc[m]) & np.array([S.depth_bin(s) == db for s in nsteps]); metrics.append({'method': m, 'benchmark': 'prm', 'metric': 'within_auc', 'stratum': f'depth={db}', 'N': int(sel.sum()), 'estimate': float(np.nanmean(auc[m][sel])), 'coverage': int(sel.sum())})
pd.DataFrame(metrics).to_csv(OUT / 'METRICS.csv', index=False)
timing['eval_s'] = time.perf_counter() - t

# ------------------------------------------------------------------ paired source-group bootstrap
t = time.perf_counter()
Gpb, gpb_inv = np.unique(groups[pb], return_inverse=True); Gpr, gpr_inv = np.unique(groups[prm], return_inverse=True)
cell_idx = {c: j for j, c in enumerate(pb_cells)}
def pb_group_stats(m):
    hits = np.zeros((len(Gpb), 8)); cnt = np.zeros((len(Gpb), 8))
    for gi, i in zip(gpb_inv, np.flatnonzero(pb)):
        if target[i] < 0 or not covered[m][i]: continue
        j = cell_idx[cells[i]]; cnt[gi, j] += 1; hits[gi, j] += pred[m][i] == target[i]
    return hits, cnt
def prm_group_stats(m):
    s = np.zeros(len(Gpr)); c = np.zeros(len(Gpr))
    for gi, i in zip(gpr_inv, np.flatnonzero(prm)):
        if np.isfinite(auc[m][i]): s[gi] += auc[m][i]; c[gi] += 1
    return s, c
stats_pb = {m: pb_group_stats(m) for m in METHODS}; stats_pr = {m: prm_group_stats(m) for m in METHODS}
rng = np.random.default_rng(SEED); est_pb = {m: np.empty(DRAWS) for m in METHODS}; est_pr = {m: np.empty(DRAWS) for m in METHODS}; invalid = np.zeros(DRAWS, bool); pos = 0
while pos < DRAWS:
    nb = min(5000, DRAWS - pos)
    wpb = rng.multinomial(len(Gpb), np.full(len(Gpb), 1 / len(Gpb)), size=nb).astype(float); wpr = rng.multinomial(len(Gpr), np.full(len(Gpr), 1 / len(Gpr)), size=nb).astype(float)
    for m in METHODS:
        h, c = stats_pb[m]; H_ = wpb @ h; C_ = wpb @ c; bad = (C_ == 0).any(1); invalid[pos:pos + nb] |= bad
        with np.errstate(invalid='ignore', divide='ignore'): est_pb[m][pos:pos + nb] = np.where(bad, np.nan, (H_ / C_).mean(1))
        s, cc = stats_pr[m]; est_pr[m][pos:pos + nb] = (wpr @ s) / (wpr @ cc)
    pos += nb
contrasts = [('P_SOFT', 'P_HARD', True), ('P_AGREE', 'P_SOFT', True)] + [(a, 'BASE', False) for a in ARMS] + [('P_HARD', 'BASE', False), ('P_SOFT', 'P_RANDOM', False), ('P_SOFT', 'P_POSITION_LENGTH', False), ('P_AGREE', 'P_SOFT_COVERAGE_MATCH', False), ('P_AGREE', 'P_HARD', False)] + [(a, c, False) for a in ['BASE', 'P_SOFT', 'P_AGREE'] for c in COMPARATORS]
K = 4; crow = []; deltas = {}
point = {m: {'pb': float(np.mean([r['estimate'] for r in metrics if r['method'] == m and r['benchmark'] == 'pb' and r['metric'] == 'sla' and r['stratum'] == 'macro8'])), 'prm': float(np.nanmean(auc[m]))} for m in METHODS}
for a, b, primary in contrasts:
    for ep, est in [('pb_sla_macro8', est_pb), ('prm_within_auc', est_pr)]:
        d = est[a] - est[b]; dv = d[np.isfinite(d)]; deltas[f'{a}__minus__{b}__{ep}'] = d
        crow.append({'contrast_id': f'{a} - {b}', 'primary': primary, 'endpoint': ep, 'paired_N': int(len(dv)), 'delta': point[a][ep.split('_')[0]] - point[b][ep.split('_')[0]], 'ci95_lo': float(np.quantile(dv, .025)), 'ci95_hi': float(np.quantile(dv, .975)),
                     'ci_adj_lo': float(np.quantile(dv, .05 / K / 2)) if primary else None, 'ci_adj_hi': float(np.quantile(dv, 1 - .05 / K / 2)) if primary else None, 'family_K': K if primary else None, 'B': DRAWS, 'invalid_draw_rate': float(np.isnan(d).mean()), 'units': 'AUC' if ep.startswith('prm') else 'fraction (x100 = pp)'})
pd.DataFrame(crow).to_csv(OUT / 'CONTRASTS.csv', index=False)
np.savez_compressed(OUT / 'BOOTSTRAP_DELTAS.npz', **deltas, invalid=invalid, seed=SEED, draws=DRAWS, pb_groups=len(Gpb), prm_groups=len(Gpr))
timing['bootstrap_s'] = time.perf_counter() - t

# ------------------------------------------------------------------ teacher / student diagnostics
t = time.perf_counter(); diag = []
err = pb & (target >= 0)
q_rank = np.array([int((np.argsort(-q_teacher[off[i]:off[i+1]], kind='stable') == target[i]).nonzero()[0][0]) + 1 if err[i] else -1 for i in range(n)])
q_mass = np.array([q_teacher[off[i] + target[i]] if err[i] else np.nan for i in range(n)])
q_ent = np.array([S.entropy_conf(q_teacher[off[i]:off[i+1]]) if pb[i] else np.nan for i in range(n)])
agree_info = {}
for i in np.flatnonzero(pb):
    tg, mult, kp, info = S.targets_pb(Zall[off[i]:off[i+1]], 'P_AGREE'); agree_info[i] = (info['v'], info['c'], kp)
v_arr = np.array([agree_info[i][0] if pb[i] else np.nan for i in range(n)]); c_arr = np.array([agree_info[i][1] if pb[i] else np.nan for i in range(n)]); keep_arr = np.array([agree_info[i][2] if pb[i] else False for i in range(n)])
for c in pb_cells + ['all_pb']:
    sel = err if c == 'all_pb' else err & (cells == c); cl = (pb & (target < 0)) if c == 'all_pb' else pb & (target < 0) & (cells == c)
    diag.append({'benchmark': 'pb', 'stratum': c, 'n_err': int(sel.sum()), 'teacher_top1': float((pred['BASE'][sel] == target[sel]).mean()), 'true_step_mass': float(np.nanmean(q_mass[sel])), 'true_step_rank': float(q_rank[sel].mean()), 'conf_c_err': float(np.nanmean(c_arr[sel])), 'conf_c_clean': float(np.nanmean(c_arr[cl])),
                 'agree_v_err': float(np.nanmean(v_arr[sel])), 'agree_v_clean': float(np.nanmean(v_arr[cl])), 'agree_keep_err': float(keep_arr[sel].mean()), 'agree_keep_clean': float(keep_arr[cl].mean()), 'teacher_top1_on_kept': float((pred['BASE'][sel & keep_arr] == target[sel & keep_arr]).mean()) if (sel & keep_arr).any() else np.nan})
kind_of = np.array(['' if not prm[i] else 'clean' if labels[off[i]:off[i+1]].sum() == 0 else 'single' if labels[off[i]:off[i+1]].sum() == 1 else 'multi' for i in range(n)])
for kd in ['clean', 'single', 'multi', 'all_prm']:
    sel = prm if kd == 'all_prm' else kind_of == kd; st = np.repeat(sel, nsteps); y = labels[st]; q = q_teacher[st]; hard = q >= .5
    tp = int((hard & (y == 1)).sum()); fpp = int((hard & (y == 0)).sum()); fn = int((~hard & (y == 1)).sum())
    diag.append({'benchmark': 'prm', 'stratum': kd, 'n_answers': int(sel.sum()), 'steps': int(st.sum()), 'hard_pseudo_precision': tp / (tp + fpp) if tp + fpp else np.nan, 'hard_pseudo_recall': tp / (tp + fn) if tp + fn else np.nan, 'pseudo_positives_per_answer': float(hard.sum() / sel.sum()), 'true_errors_per_answer': float((y == 1).sum() / sel.sum()),
                 'soft_auprc': float(average_precision_score(y, q)) if (y == 1).any() and (y == 0).any() else np.nan, 'agree_selected_step_fraction': float(np.mean([S.targets_prm(Zall[off[i]:off[i+1]], 'P_AGREE')[2].mean() for i in np.flatnonzero(sel)]))})
for m in ARMS:
    hs = pred[m][err] == target[err]; ht = pred['BASE'][err] == target[err]
    row = {'benchmark': 'pb', 'stratum': f'student={m}', 'agreement_with_teacher': float((pred[m][err] == pred['BASE'][err]).mean()), 'both_correct': int((hs & ht).sum()), 'student_only': int((hs & ~ht).sum()), 'teacher_only': int((~hs & ht).sum()), 'both_wrong': int((~hs & ~ht).sum()), 'net': int((hs & ~ht).sum() - (~hs & ht).sum())}
    for lab_, msk in [('kept', keep_arr), ('abstained', ~keep_arr)]:
        e2 = err & msk; row[f'student_top1_{lab_}'] = float((pred[m][e2] == target[e2]).mean()); row[f'teacher_top1_{lab_}'] = float((pred['BASE'][e2] == target[e2]).mean()); row[f'n_{lab_}'] = int(e2.sum())
    diag.append(row)
    d = auc[m] - auc['BASE']; okm = np.isfinite(d); fixed = broken = 0
    for i in np.flatnonzero(okm):
        a, b = off[i:i+2]; y = labels[a:b].astype(bool); sc, sr = scores[m][a:b], scores['BASE'][a:b]
        wc = sc[y][:, None] > sc[~y][None, :]; wr = sr[y][:, None] > sr[~y][None, :]; fixed += int((wc & ~wr).sum()); broken += int((~wc & wr).sum())
    diag.append({'benchmark': 'prm', 'stratum': f'student={m}', 'mean_delta_auc_vs_teacher': float(d[okm].mean()), 'answers_improved': int((d[okm] > 1e-12).sum()), 'answers_worsened': int((d[okm] < -1e-12).sum()), 'pairs_corrected': fixed, 'pairs_destroyed': broken})
# confidence quintiles: edges fixed on B of each fold, accuracy on H
qrows = []
for k in range(5):
    for task in ['pb_q4', 'pb_q8']:
        rm = role_masks(task, k); Bc = c_arr[rm['B']]; edges = np.quantile(Bc[np.isfinite(Bc)], [.2, .4, .6, .8]); He = np.flatnonzero(rm['H'] & err)
        binned = np.searchsorted(edges, c_arr[He])
        for q5 in range(5):
            sel = He[binned == q5]
            if len(sel): qrows.append({'task': task, 'fold': k, 'conf_quintile': q5 + 1, 'n': len(sel), 'teacher_top1': float((pred['BASE'][sel] == target[sel]).mean()), **{f'{m}_top1': float((pred[m][sel] == target[sel]).mean()) for m in ARMS}})
qq = pd.DataFrame(qrows); qq.to_csv(OUT / 'CONFIDENCE_QUINTILES_PER_FOLD.csv', index=False)
qq.groupby('conf_quintile').apply(lambda g: pd.Series({'n': g.n.sum(), **{c: np.average(g[c], weights=g.n) for c in g.columns if c.endswith('top1')}}), include_groups=False).to_csv(OUT / 'CONFIDENCE_QUINTILES.csv')
pd.DataFrame(diag).to_csv(OUT / 'TEACHER_DIAGNOSTICS.csv', index=False)
# deterministic examples (13.6): ids only, no text in these arrays
ex = []
for cat, msk in [('rescued', (pred['P_AGREE'] == target) & (pred['BASE'] != target) & err), ('damaged', (pred['P_AGREE'] != target) & (pred['BASE'] == target) & err), ('both_wrong', (pred['P_AGREE'] != target) & (pred['BASE'] != target) & err)]:
    cand = sorted(np.flatnonzero(msk), key=lambda i: hashlib.sha256(uid[i].encode()).hexdigest())[:5]
    for i in cand: ex.append({'category': cat, 'uid': uid[i], 'cell': cells[i], 'n_steps': nsteps[i], 'true_first_error': target[i], 'teacher_pred': pred['BASE'][i], 'P_AGREE_pred': pred['P_AGREE'][i], 'P_SOFT_pred': pred['P_SOFT'][i], 'ct7_pred': pred['ct7'][i], 'conf_c': c_arr[i], 'agree_v': v_arr[i]})
pd.DataFrame(ex).to_csv(OUT / 'EXAMPLES.csv', index=False)
timing['diagnostics_s'] = time.perf_counter() - t; timing['total_s'] = time.perf_counter() - T0; timing['output_bytes'] = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
dump(OUT / 'TIMING.json', timing)
status.update({'status': 'COMPLETE' if not failures else 'INCOMPLETE', 'finished': datetime.now().isoformat(timespec='seconds'), 'fits': len(fit_log), 'failures': len(failures), 'non_converged_fits': int(sum(not r['converged'] for r in fit_log)), 'methods': METHODS, 'invalid_bootstrap_draw_rate': float(invalid.mean())})
dump(OUT / 'RUN_STATUS.json', status)
print(pd.DataFrame(crow)[['contrast_id', 'primary', 'endpoint', 'delta', 'ci95_lo', 'ci95_hi']].to_string()); print(json.dumps(status, indent=1, default=str)); print(json.dumps(timing, indent=1))
