"""bank20_lsml_prmbench_v1: extend the 11-channel step bank to 20 channels under the bank11
CONT L-SML recipe, PRMBench first.  Frozen protocol: results/bank20_lsml_prmbench_v1/PROTOCOL.json.

Fusion code is imported from the depth-feature-fusion-v1 worktree (the one carrying
small_m_guard); nothing there is edited.  Labels enter only the evaluation block.
"""
from pathlib import Path
import hashlib, json, os, pickle, shutil, subprocess, sys, time
from datetime import datetime
import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
DEPTH = MAIN / '.worktrees/depth-feature-fusion-v1'; sys.path.insert(0, str(DEPTH))
from spectral_utils.lsml_gate_locator_research import FusionRecipe, answer_standardize, effective_rank, fit_fusion_weights  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260924'
STAGE = ROOT / 'results/bank20_lsml_prmbench_v1'; OUT = STAGE / RUN_ID; OUT.mkdir(parents=True, exist_ok=True)
P = json.loads((STAGE / 'PROTOCOL.json').read_text(encoding='utf8'))
SMOKE = {k: os.environ[k] for k in ['B20_FOLDS', 'B20_DRAWS'] if k in os.environ}
FOLDS = [int(x) for x in SMOKE['B20_FOLDS'].split(',')] if 'B20_FOLDS' in SMOKE else list(range(5))
DRAWS = int(SMOKE.get('B20_DRAWS', 100_000)); SEED = 20260924; FIT_SEED = 20260919
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
TPF = MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1'
INPUTS = {'level_bank': TPF / 'DERIVATIVE_CHANNELS.npz', 'ct7_tokens': MAIN / 'results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz',
          'profiles_full': R / 'profiles_full.npy', 'evidence_drop': TPF / 'EVIDENCE_DROP.npz', 'token_matrices': TPF / 'TOKEN_MATRICES.npz',
          'oof_answers': R / 'OOF_ANSWERS.csv', 'oof_step_scores': R / 'OOF_STEP_SCORES.npz',
          'baseline_scores': DEPTH / 'results/step_level_bank_baseline_v1/STEP_SCORES.npz', 'folds_v2': MAIN / 'results/localization_source_group_audit_v1/FOLDS_V2.json'}
T0 = time.perf_counter(); timing = {}
def dump(p, v): Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'run_id': RUN_ID, 'smoke_overrides': SMOKE}; dump(OUT / 'RUN_STATUS.json', status)

# ------------------------------------------------------------------ population frame (mirrors ssl_eval.Frame without importing it)
ans = pd.read_csv(INPUTS['oof_answers'], encoding='utf-8-sig'); Zs = np.load(INPUTS['oof_step_scores'])
off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans); ns = np.diff(off)
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); cells = ans.cell.to_numpy(); groups = ans.source_group.to_numpy(); ids = ans.id.to_numpy(); target = ans.target.to_numpy()
freeze = json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig')); meta_path = Path(freeze['prm_metadata']['path'])
meta = {m['idx']: m for m in pickle.load(open(meta_path, 'rb')).values()}
noncontrol = np.array([prm[i] and meta[ids[i]]['classification'] != 'correct' for i in range(n)])
classification = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
prm_idx = np.flatnonzero(prm); prm_steps = np.repeat(prm, ns); nc_steps = np.repeat(noncontrol, ns)
eligible = np.array([prm[i] and labels[off[i]:off[i+1]].any() and (~labels[off[i]:off[i+1]]).any() for i in range(n)])

# ------------------------------------------------------------------ channels
t = time.perf_counter()
lv = np.load(INPUTS['level_bank']); level = lv['level'].astype(float); names11 = list(lv['channels'].astype(str)); assert level.shape == (int(off[-1]), 11)
tk = np.load(INPUTS['token_matrices']); spans = tk['step_spans']; toff = tk['token_offsets']; assert spans.shape[0] == int(off[-1])
B = np.load(INPUTS['ct7_tokens']); assert np.array_equal(B['token_offsets'], toff) and np.array_equal(B['step_spans'], spans)
cb = list(B['channels']); pick = ['chosen_std_excess', 'bocpd_residual', 'H0lim_prefix_innovation', 've0']; ci = [cb.index(c) for c in pick]
XB = B['tokens'][:, ci]; VB = B['valid'][:, ci]
ED = np.load(INPUTS['evidence_drop']); assert np.array_equal(ED['token_offsets'], toff); risk_tok = ED['risk_token'].astype(float)
def top10(v):
    v = v[np.isfinite(v)]
    if len(v) == 0: return np.nan
    k = min(10, len(v)); return float(np.partition(v, len(v) - k)[-k:].mean())
add = np.full((int(off[-1]), 5), np.nan)
for s in range(int(off[-1])):
    a, b = spans[s]
    for j in range(4):
        x = XB[a:b, j].copy(); x[~VB[a:b, j]] = np.nan; add[s, j] = top10(x)
    add[s, 4] = top10(risk_tok[a:b])
prof = np.load(INPUTS['profiles_full'], mmap_mode='r'); shape = np.ascontiguousarray(prof[:, 0, [14, 11, 12, 10]]).astype(float)  # first_token, slope, jump, frac_above_z of q15_H1
names_add = ['ct7_' + c for c in pick] + ['H1_first_token', 'H1_slope', 'H1_jump', 'H1_frac_above_z', 'evidence_drop_risk']
raw = np.column_stack([level, add[:, :4], shape, add[:, 4:5]]); names = names11 + names_add; assert raw.shape[1] == 20
missing = {nm: int(np.isnan(raw[:, j]).sum()) for j, nm in enumerate(names)}
for i in range(n):                                                     # missing step value -> answer column mean (-> 0 after z)
    a, b = off[i:i+2]; blk = raw[a:b]
    if np.isnan(blk).any():
        mu = np.nanmean(blk, 0); mu = np.where(np.isfinite(mu), mu, 0.); idx = np.where(np.isnan(blk)); blk[idx] = mu[idx[1]]
assert np.isfinite(raw).all()
values = answer_standardize(raw, off)
BANKS = {'B11': list(range(11)), 'B15': list(range(15)), 'B19': list(range(19)), 'B20': list(range(20))}
timing['channels_s'] = time.perf_counter() - t
dump(OUT / 'INPUT_MANIFEST.json', {k: {'path': str(v), 'bytes': v.stat().st_size, 'sha256': sha(v)} for k, v in INPUTS.items()} | {'channels': names, 'missing_step_values_filled': missing, 'population': {'answers': n, 'prm': int(prm.sum()), 'eligible': int(eligible.sum()), 'noncontrol': int(noncontrol.sum()), 'steps': int(off[-1])}})
snap = OUT / 'SOURCE_SNAPSHOT'; code = {}
for rel, base in [('scripts/experiments/bank20_lsml_run.py', ROOT), ('spectral_utils/lsml_gate_locator_research.py', DEPTH), ('spectral_utils/fusion_utils.py', DEPTH)]:
    dst = snap / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(base / rel, dst); code[f'{base.name}/{rel}'] = sha(base / rel)
git = lambda cwd, *a: subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head_ssl_worktree': git(ROOT, 'rev-parse', 'HEAD'), 'git_head_depth_worktree': git(DEPTH, 'rev-parse', 'HEAD'), 'protocol_sha256': sha(STAGE / 'PROTOCOL.json')})
print(f'channels ready in {timing["channels_s"]:.0f}s; missing filled: {missing}', flush=True)

# ------------------------------------------------------------------ fits: eval fold k, cal (k+1)%5, fit = the other three
def rows_of(mask): return np.concatenate([np.arange(off[i], off[i+1]) for i in np.flatnonzero(mask)])
scores = {}; fit_log = []; failures = []
for bank, cols in BANKS.items():
    for arm in ['lsml', 'equal', 'group_equal']: scores[f'{bank}_{arm}'] = np.full(int(off[-1]), np.nan)
scores['B11_lsml_replay4'] = np.full(int(off[-1]), np.nan); scores['ct7'] = Zs['ct7'].astype(float)
cal_thr = {}
for k in FOLDS:
    t = time.perf_counter(); cal = (k + 1) % 5; fitf = [f for f in range(5) if f not in (k, cal)]
    fit_rows = rows_of(np.isin(fold, fitf)); ev_rows = rows_of(fold == k); cal_rows = rows_of(fold == cal)
    for bank, cols in BANKS.items():
        X = values[:, cols]; members = tuple(names[c] for c in cols)
        try:
            w, m = fit_fusion_weights(X[fit_rows], FusionRecipe(name=f'{bank}_lsml', members=members, mode='continuous', anchor=0), seed=FIT_SEED)
        except Exception as e:
            failures.append({'fold': k, 'bank': bank, 'arm': 'lsml', 'reason': str(e)}); continue
        g = np.asarray(m['groups'], int); K = int(m['K']); sizes = np.bincount(g, minlength=K)
        wg = 1.0 / (K * sizes[g]); we = np.ones(len(cols)) / len(cols)
        for arm, ww in [('lsml', w), ('equal', we), ('group_equal', wg)]:
            for rr in (ev_rows, cal_rows): scores[f'{bank}_{arm}'][rr] = X[rr] @ ww
        fit_log.append({'fold': k, 'cal_fold': cal, 'fit_folds': fitf, 'bank': bank, 'K': K, 'groups': g.tolist(), 'group_sizes': sizes.tolist(), 'weights': np.round(w, 5).tolist(), 'weight_ipr': float(1 / np.sum((np.abs(w) / np.abs(w).sum()) ** 2)),
                        'anchor_spearman': m['anchor_spearman'], 'anchor_flipped': m['anchor_flipped'], 'residual': m['residual'], 'small_m_guarded': m['small_m_guarded'], 'fit_steps': int(len(fit_rows)), 'members': list(members)})
    # replay reference: B11, fit on the four non-evaluation folds (the original allocation)
    X = values[:, :11]; fr4 = rows_of(fold != k)
    w4, m4 = fit_fusion_weights(X[fr4], FusionRecipe(name='replay', members=tuple(names11), mode='continuous', anchor=0), seed=FIT_SEED)
    scores['B11_lsml_replay4'][ev_rows] = X[ev_rows] @ w4
    fit_log.append({'fold': k, 'cal_fold': None, 'fit_folds': [f for f in range(5) if f != k], 'bank': 'B11_replay4', 'K': int(m4['K']), 'groups': list(map(int, m4['groups'])), 'weights': np.round(w4, 5).tolist(), 'members': names11})
    print(f'fold {k}: fit folds {fitf}, cal {cal}; ' + ', '.join(f'{r["bank"]} K={r["K"]}' for r in fit_log if r['fold'] == k) + f'  ({time.perf_counter()-t:.0f}s)', flush=True)
timing['fit_s'] = time.perf_counter() - T0
# replay assert against the saved baseline scores (tolerance from the protocol)
base = np.load(INPUTS['baseline_scores']); ev_all = rows_of(np.isin(fold, FOLDS))
rep = {'max_abs_diff_continuous': float(np.nanmax(np.abs(scores['B11_lsml_replay4'][ev_all] - base['continuous'][ev_all]))), 'max_abs_diff_equal_vs_B11_equal_fit3': float(np.nanmax(np.abs(scores['B11_equal'][ev_all] - base['equal'][ev_all])))}
rep['replay_exact'] = bool(rep['max_abs_diff_continuous'] <= 1e-9)
print('replay:', rep, flush=True)
with open(OUT / 'FIT_MANIFEST.jsonl', 'w', encoding='utf8') as f:
    for r in fit_log: f.write(json.dumps(r, default=float) + '\n')
METHODS = [m for m in scores if np.isfinite(scores[m][ev_all]).all()]
np.savez_compressed(OUT / 'STEP_SCORES.npz', offsets=off, **{m: scores[m] for m in METHODS})

# ------------------------------------------------------------------ evaluation (labels enter here only)
t = time.perf_counter()
def within_auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def zt(s): return (s - s.mean()) / max(s.std(), 1e-8)
def prmscore_from_counts(tp, fp, tn, fn):
    def ratio(a, b): return np.divide(a, b, out=np.full(np.shape(a), -1., float), where=b != 0)
    p = ratio(tp, tp + fp); r = ratio(tp, tp + fn); f = ratio(2 * p * r, p + r); p2 = ratio(tn, tn + fn); r2 = ratio(tn, tn + fp); nf = ratio(2 * p2 * r2, p2 + r2); return (f + nf) / 2
auc = {}; valid = {}; conf = {}; metrics = []
Gpr, ginv = np.unique(groups[prm], return_inverse=True); prm_pos = np.flatnonzero(prm)
official = {}
for m in METHODS:
    s = scores[m]
    auc[m] = np.array([within_auc(labels[off[i]:off[i+1]], s[off[i]:off[i+1]]) if eligible[i] and np.isfinite(s[off[i]:off[i+1]]).all() else np.nan for i in range(n)])
    # PRMScore: answer-z, threshold = 0.8-quantile of the calibration fold's answer-z PRMB step scores (label-free); ct7 uses the same rule with cal = (k+1)%5
    v = np.zeros(int(off[-1]), bool); thr = {}
    for k in FOLDS:
        cal = (k + 1) % 5; cs = np.concatenate([zt(s[off[i]:off[i+1]]) for i in np.flatnonzero(prm & (fold == cal))]); tau = float(np.quantile(cs, .8)); thr[k] = tau
        for i in np.flatnonzero(prm & (fold == k)):
            a, b = off[i:i+2]; v[a:b] = zt(s[a:b]) < tau
    valid[m] = v; cal_thr[m] = thr
    ev_prm = [i for i in prm_pos if fold[i] in FOLDS]
    official[m] = prmbench_evaluate([{'idx': ids[i], 'labels': v[off[i]:off[i+1]].astype(int).tolist()} for i in ev_prm], [meta[ids[i]] for i in ev_prm])['total']
    good = ~labels; c = np.zeros((len(Gpr), 4))
    for gi, i in zip(ginv, prm_pos):
        if not noncontrol[i] or fold[i] not in FOLDS: continue
        a, b = off[i:i+2]; vv = v[a:b]; gg = good[a:b]
        c[gi] += [(vv & gg).sum(), (vv & ~gg).sum(), (~vv & ~gg).sum(), (~vv & gg).sum()]
    conf[m] = c
    el = np.isfinite(auc[m]); ps = float(.5 * (official[m]['f1'] + official[m]['negative_f1']))
    metrics.append({'method': m, 'benchmark': 'prm', 'metric': 'within_auc', 'stratum': 'all', 'N': int(el.sum()), 'estimate': float(np.nanmean(auc[m]))})
    metrics.append({'method': m, 'benchmark': 'prm', 'metric': 'prmscore_answer_z_q80', 'stratum': 'all', 'N': int(sum(noncontrol[i] for i in ev_prm)), 'estimate': ps, 'from_counts': float(prmscore_from_counts(*c.sum(0)))})
    for cl in sorted(set(classification[prm]) - {''}):
        sel = (classification == cl) & el; metrics.append({'method': m, 'benchmark': 'prm', 'metric': 'within_auc', 'stratum': f'class={cl}', 'N': int(sel.sum()), 'estimate': float(np.nanmean(auc[m][sel])) if sel.any() else np.nan})
    # ProcessBench context: macro8 SLA of the earliest argmax on erroneous answers
    sla = []
    for cell in sorted(set(cells[pb])):
        e = (cells == cell) & (target >= 0) & np.isin(fold, FOLDS)
        if not e.any(): continue
        hit = [int(np.flatnonzero(s[off[i]:off[i+1]] >= s[off[i]:off[i+1]].max() - 8 * np.finfo(float).eps)[0]) == target[i] for i in np.flatnonzero(e)]
        sla.append(float(np.mean(hit)))
    metrics.append({'method': m, 'benchmark': 'pb', 'metric': 'sla', 'stratum': 'macro8_context', 'N': int((pb & (target >= 0) & np.isin(fold, FOLDS)).sum()), 'estimate': float(np.mean(sla)) if sla else np.nan})
pd.DataFrame(metrics).to_csv(OUT / 'METRICS.csv', index=False)
# single-stream diagnostic for the nine additions + effective rank per bank (labels for evaluation only)
diag = {'single_stream_within_auc': {}, 'effective_rank_prm_steps': {}}
for j, nm in enumerate(names):
    diag['single_stream_within_auc'][nm] = float(np.nanmean([within_auc(labels[off[i]:off[i+1]], values[off[i]:off[i+1], j]) for i in np.flatnonzero(eligible)]))
for bank, cols in BANKS.items(): diag['effective_rank_prm_steps'][bank] = float(effective_rank(values[prm_steps][:, cols]))
diag['replay'] = rep; diag['calibration_thresholds'] = cal_thr
dump(OUT / 'DIAGNOSTICS.json', diag)
timing['eval_s'] = time.perf_counter() - t

# ------------------------------------------------------------------ paired source-group bootstrap: within-AUC and PRMScore (group confusion sums)
t = time.perf_counter()
sums = {m: np.zeros(len(Gpr)) for m in METHODS}; cnts = np.zeros(len(Gpr))
for gi, i in zip(ginv, prm_pos):
    if np.isfinite(auc[METHODS[0]][i]):
        cnts[gi] += 1
        for m in METHODS: sums[m][gi] += auc[m][i]
rng = np.random.default_rng(SEED); est_auc = {m: np.empty(DRAWS) for m in METHODS}; est_ps = {m: np.empty(DRAWS) for m in METHODS}; pos = 0
while pos < DRAWS:
    nb = min(5000, DRAWS - pos); W = rng.multinomial(len(Gpr), np.full(len(Gpr), 1 / len(Gpr)), size=nb).astype(float)
    den = W @ cnts
    for m in METHODS:
        est_auc[m][pos:pos+nb] = (W @ sums[m]) / den
        cc = W @ conf[m]; est_ps[m][pos:pos+nb] = prmscore_from_counts(cc[:, 0], cc[:, 1], cc[:, 2], cc[:, 3])
    pos += nb
point = {m: {'auc': float(np.nanmean(auc[m])), 'ps': float(prmscore_from_counts(*conf[m].sum(0)))} for m in METHODS}
prim = [('B20_lsml', 'B20_equal'), ('B20_lsml', 'B11_lsml'), ('B20_lsml', 'ct7')]
sec = [('B15_lsml', 'B11_lsml'), ('B19_lsml', 'B15_lsml'), ('B20_lsml', 'B19_lsml'), ('B20_lsml', 'B20_group_equal'), ('B11_lsml_replay4', 'ct7'), ('B11_lsml', 'ct7'), ('B15_lsml', 'ct7'), ('B20_lsml', 'B11_lsml_replay4')]
sec += [(f'{b}_lsml', f'{b}_equal') for b in ['B11', 'B15', 'B19']] + [(f'{b}_lsml', f'{b}_group_equal') for b in ['B11', 'B15', 'B19']] + [(f'{b}_equal', 'B11_equal') for b in ['B15', 'B19', 'B20']]
K = 6; rows = []; deltas = {}
for a, b in prim + sec:
    if a not in METHODS or b not in METHODS: continue
    primary = (a, b) in prim
    for ep, est, key in [('prm_within_auc', est_auc, 'auc'), ('prmscore_answer_z_q80', est_ps, 'ps')]:
        d = est[a] - est[b]; deltas[f'{a}__minus__{b}__{ep}'] = d.astype(np.float32)
        rows.append({'contrast_id': f'{a} - {b}', 'primary': primary, 'endpoint': ep, 'delta': point[a][key] - point[b][key], 'ci95_lo': float(np.quantile(d, .025)), 'ci95_hi': float(np.quantile(d, .975)),
                     'ci_adj_lo': float(np.quantile(d, .05 / K / 2)) if primary else None, 'ci_adj_hi': float(np.quantile(d, 1 - .05 / K / 2)) if primary else None, 'family_K': K if primary else None, 'B': DRAWS, 'paired_groups': len(Gpr)})
pd.DataFrame(rows).to_csv(OUT / 'CONTRASTS.csv', index=False); np.savez_compressed(OUT / 'BOOTSTRAP_DELTAS.npz', **deltas, seed=SEED, draws=DRAWS)
timing['bootstrap_s'] = time.perf_counter() - t; timing['total_s'] = time.perf_counter() - T0; dump(OUT / 'TIMING.json', timing)
pd.DataFrame(failures or [{'fold': '', 'bank': '', 'arm': '', 'reason': 'none'}]).to_csv(OUT / 'FAILURES.csv', index=False)
status.update({'status': 'COMPLETE' if not failures and rep['replay_exact'] else 'INCOMPLETE', 'finished': datetime.now().isoformat(timespec='seconds'), 'fits': len(fit_log), 'failures': len(failures), 'replay': rep, 'methods': METHODS}); dump(OUT / 'RUN_STATUS.json', status)
M = pd.DataFrame(metrics); print(M[(M.stratum == 'all')].pivot(index='method', columns='metric', values='estimate').round(4).to_string())
print(pd.DataFrame(rows)[['contrast_id', 'primary', 'endpoint', 'delta', 'ci95_lo', 'ci95_hi']].round(4).to_string()); print(json.dumps(status, indent=1, default=str)); print(json.dumps(timing, indent=1))
