"""position_channel_v1: step position as an additional label-free channel (POS = step index; CUM = accumulated evidence, the
running mean of the answer's own plain-average score), fed to the existing label-free machinery: the Dawid-Skene filter decides
whether it is kept (plain average), or it is its own group whose weight comes from the group-level Dawid-Skene estimate.
Banks B16/B23/B35/B54.  Frozen protocol: results/position_channel_v1/PROTOCOL.json.  Bank construction as partition_switch_run.py
(replays algorithm_decisions_v1 P0__BASE); learning as algorithm_decisions_run.py learn() (P0, EQ within, DSM between).
Smoke: ER_FOLDS=0 ER_DRAWS=2000 ER_NULL_PERMS=5."""
from pathlib import Path
import hashlib, json, os, pickle, subprocess, sys, time
from datetime import datetime
import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN / '.worktrees/depth-feature-fusion-v1'))
from spectral_utils.lsml_gate_locator_research import answer_standardize  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
from scipy.stats import rankdata, trim_mean, spearmanr  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
import tail_calib_common as TC, er_stage_a as SA, er_stage_b as SB, er_stage_b2 as C2, lsml_merge_step as MS, ds_group_weights as GW  # noqa: E402
from calfix_common import tail_marks  # noqa: E402

GEN = ROOT / 'results/position_channel_v1'; OUT = GEN / (sys.argv[1] if len(sys.argv) > 1 else 'run_20260929'); OUT.mkdir(parents=True, exist_ok=True)
AD = ROOT / 'results/algorithm_decisions_v1/run_20260928'
P = json.loads((GEN / 'PROTOCOL.json').read_text(encoding='utf8'))
SMOKE = {k: os.environ[k] for k in ['ER_FOLDS', 'ER_DRAWS', 'ER_NULL_PERMS'] if k in os.environ}
FOLDS = [int(x) for x in SMOKE['ER_FOLDS'].split(',')] if 'ER_FOLDS' in SMOKE else list(range(5))
DRAWS = int(SMOKE.get('ER_DRAWS', 50_000)); NPERM = int(SMOKE.get('ER_NULL_PERMS', 200)); SEED = 20260927
T0 = time.perf_counter(); timing = {}
def jd(v): return v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v)
def dump(p, v): Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=jd), encoding='utf8')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
if P.get('status') != 'FROZEN': raise SystemExit('protocol not frozen')
if (OUT / 'RUN_STATUS.json').exists() and not SMOKE: raise SystemExit(f'{OUT} already holds a run')
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'smoke_overrides': SMOKE}; checks = {}; dump(OUT / 'RUN_STATUS.json', status)
def hard_stop(reason):
    status.update({'status': 'STOPPED', 'reason': reason, 'checks': checks}); dump(OUT / 'RUN_STATUS.json', status); raise SystemExit('HARD STOP: ' + reason)
def _crash(et, ev, tb):
    import traceback; traceback.print_exception(et, ev, tb)
    if status.get('status') == 'RUNNING': status.update({'status': 'CRASHED', 'reason': repr(ev), 'checks': checks}); dump(OUT / 'RUN_STATUS.json', status)
sys.excepthook = _crash
dump(OUT / 'CODE_MANIFEST.json', {'script_sha256': sha(Path(__file__)), 'protocol_sha256': sha(GEN / 'PROTOCOL.json'),
     'helpers': {h: sha(ROOT / 'scripts/experiments' / h) for h in ('er_stage_a.py', 'er_stage_b.py', 'er_stage_b2.py', 'ds_group_weights.py', 'lsml_merge_step.py', 'tail_calib_common.py', 'calfix_common.py')},
     'depth_git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=MAIN / '.worktrees/depth-feature-fusion-v1', capture_output=True, text=True).stdout.strip(),
     'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()})

# ------------------------------------------------------------------ population (as algorithm_decisions_run.py)
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Zs = np.load(R / 'OOF_STEP_SCORES.npz')
off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans); ns = np.diff(off); S = int(off[-1])
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); cells = ans.cell.to_numpy(); groups = ans.source_group.to_numpy(); ids = ans.id.to_numpy(); target = ans.target.to_numpy()
freeze = json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig')); meta = {m['idx']: m for m in pickle.load(open(Path(freeze['prm_metadata']['path']), 'rb')).values()}
noncontrol = np.array([prm[i] and meta[ids[i]]['classification'] != 'correct' for i in range(n)])
prm_steps = np.repeat(prm, ns); step_answer = np.repeat(np.arange(n), ns); step_pos = (np.arange(S) - off[step_answer]).astype(float)
eligible = np.array([prm[i] and labels[off[i]:off[i+1]].any() and (~labels[off[i]:off[i+1]]).any() for i in range(n)])
has_error = np.array([prm[i] and labels[off[i]:off[i+1]].any() for i in range(n)])
def rows_of(mask): return np.concatenate([np.arange(off[i], off[i+1]) for i in np.flatnonzero(mask)])
def within_auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def zt(s): return (s - s.mean()) / max(s.std(), 1e-8)
def answer_z(s):
    out = np.empty(len(s))
    for a, b in zip(off[:-1], off[1:]): out[a:b] = zt(s[a:b])
    return out
kneed = np.maximum(1, np.ceil(.2 * ns)).astype(int)
def marks_ok(v): return bool(np.all(np.add.reduceat((v > 0).astype(np.int64), off[:-1], axis=0) == kneed[:, None]))

# ------------------------------------------------------------------ banks (as partition_switch_run.py bank(); replays P0__BASE)
MI = json.loads((AD / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))
paths = {k: Path(v['path']) for k, v in MI.items() if isinstance(v, dict) and 'path' in v}
for k in ('level_bank', 'ct7_profiles', 'ct7_profile_validation', 'pool_z', 'pool_names', 'digit_features', 'oof_answers', 'oof_step_scores', 'pool_structure'):
    if sha(paths[k]) != MI[k]['sha256']: hard_stop(f'input {k} differs from the algorithm_decisions_v1 manifest')
lv = np.load(paths['level_bank']); names11 = list(map(str, lv['channels']))
prof = np.load(paths['ct7_profiles']).astype(float); pn = json.loads(paths['ct7_profile_validation'].read_text(encoding='utf8'))['channels']
values = answer_standardize(np.column_stack([lv['level'].astype(float), prof[:, pn.index('chosen_token_z_despiked')], lv['derivative'][:, names11.index('chosen_surprisal')].astype(float)]), off)
DF = np.load(paths['digit_features']); act = DF['active'].astype(bool); DFV = DF['values']; D3 = np.zeros((S, 3))
for a, b in zip(off[:-1], off[1:]):
    for j in range(3):
        v = act[a:b, j]; y = DFV[a:b, j][v]
        if len(y) and y.std() > 1e-12: D3[a:b, j][v] = (y - y.mean()) / y.std()
POOL = np.load(paths['pool_z']); PN = json.loads(paths['pool_names'].read_text(encoding='utf8')); PS = pd.read_csv(paths['pool_structure']).set_index('channel')
BN = MI['banks']; DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
def bank(bk):
    cols = []
    for c in BN[bk]:
        if c in names11 + ['realized_z', 'realized_drv'] and bk in ('B13', 'B16'): cols.append(values[:, (names11 + ['realized_z', 'realized_drv']).index(c)])
        elif c in DIG: cols.append(D3[:, DIG.index(c)])
        else: cols.append(None)
    need = [i for i, c in enumerate(cols) if c is None]
    if need:
        Vp = answer_standardize(np.column_stack([POOL[:, PN.index(BN[bk][i][4:] if BN[bk][i].startswith('lf__') else BN[bk][i])] for i in need]), off)
        for q, i in enumerate(need):
            if BN[bk][i].startswith('lf__'): Vp[:, q] *= float(np.sign(PS.loc[BN[bk][i][4:], 'r_level_marginal'])) or 1.0
            cols[i] = Vp[:, q]
    return np.column_stack(cols)
BANKS = ['B16', 'B23', 'B35', 'B54']
_SC = np.load(AD / 'STEP_SCORES.npz'); REF = {f'{bk}__{a}': _SC[f'{bk}__P0__{a}'] for bk in BANKS for a in ('BASE', 'EQ_DSM')}; REF['ct7'] = _SC['ct7']; REF['fam421'] = _SC['fam421']; del _SC
VB = {bk: bank(bk) for bk in BANKS}; MARKS = {}; TTA = {}
for bk, V in VB.items():
    MARKS[bk] = SB.random_tie_marks(V, off, .2, np.random.default_rng(20260928).random((S, V.shape[1])))
    if not marks_ok(MARKS[bk]): hard_stop(f'mark counts {bk}')
    TTA[bk] = tail_marks(V, off, .2, tie_aware=True, centred=True)[0]
POSZ = answer_z(step_pos); KEYP = np.random.default_rng(20260931).random((S, 1)); POSM = SB.random_tie_marks(POSZ[:, None], off, .2, KEYP)
KEYC = np.random.default_rng(20260933).random((S, 1)); KEYG = np.random.default_rng(20260929).random((S, 13))
if not marks_ok(POSM): hard_stop('POS mark counts')
timing['setup_s'] = time.perf_counter() - T0; print(f'setup {timing["setup_s"]:.0f}s', flush=True)

# ------------------------------------------------------------------ fits
ARMS = ['BASE', 'BASE_POS', 'BASE_CUM', 'GRP', 'GRP_POS', 'GRP_CUM', 'POS_ALONE']
ALL = ['ct7', 'fam421'] + [f'{bk}__{a}' for bk in BANKS for a in ARMS]
scores = {m: np.full(S, np.nan) for m in ALL}; written = {m: np.zeros(S, np.int8) for m in ALL}; tau = {m: {} for m in ALL}; fitted = {m: [] for m in ALL}
diag = []; replay = {}
def cal_tau(s_full, cal): return float(np.quantile(np.concatenate([zt(s_full[off[i]:off[i+1]]) for i in np.flatnonzero(prm & (fold == cal))]), .8))
def put(m, k, ev, s_full, cal):
    if written[m][ev].any(): raise AssertionError(f'{m} fold {k} written twice')
    if not np.isfinite(s_full).all(): hard_stop(f'non-finite scores {m} fold {k}')
    scores[m][ev] = s_full[ev]; written[m][ev] += 1; tau[m][k] = cal_tau(s_full, cal); fitted[m].append(k)
for k in FOLDS:
    t = time.perf_counter(); cal = (k + 1) % 5; fit_rows = rows_of(np.isin(fold, [f for f in range(5) if f not in (k, cal)])); ev = rows_of(fold == k); pf = fit_rows[prm_steps[fit_rows]]
    put('ct7', k, ev, REF['ct7'].astype(float), cal); put('fam421', k, ev, REF['fam421'], cal)
    for bk in BANKS:
        V = VB[bk]; nm = BN[bk]; m = V.shape[1]; d = {'bank': bk, 'fold': k}; out = {}
        est = SA.em_estimate(MARKS[bk][pf], 'ds'); surv = np.flatnonzero(est['pi'] > 0.5)
        if 0 not in surv: hard_stop(f'{bk} fold {k}: anchor filtered out')
        X = V[:, surv]; out['BASE'] = X @ np.full(len(surv), 1 / len(surv))
        # BASE_POS: one DS fit on all channels + POS
        e2 = SA.em_estimate(np.column_stack([MARKS[bk], POSM])[pf], 'ds'); s2 = np.flatnonzero(e2['pi'] > 0.5)
        Xp = np.column_stack([V, POSZ]); out['BASE_POS'] = Xp[:, s2] @ np.full(len(s2), 1 / len(s2))
        d['label_free_orientation_spearman_pos_vs_base_fit_rows'] = float(spearmanr(POSZ[fit_rows], out['BASE'][fit_rows]).statistic)
        d['pos_pi_hat_with_bank'] = float(e2['pi'][-1]); d['pos_kept'] = bool(m in s2); d['survivors_changed_by_pos'] = sorted(set(s2) - {m}) != sorted(surv.tolist())
        # CUM: running mean of the answer-z BASE score
        bz = answer_z(out['BASE']); cum = np.empty(S)
        for a, b in zip(off[:-1], off[1:]): cum[a:b] = np.cumsum(bz[a:b]) / np.arange(1, b - a + 1)
        CUMZ = answer_z(cum); CUMM = SB.random_tie_marks(CUMZ[:, None], off, .2, KEYC)
        if not marks_ok(CUMM): hard_stop(f'CUM mark counts {bk} fold {k}')
        e3 = SA.em_estimate(np.column_stack([MARKS[bk][:, surv], CUMM])[pf], 'ds'); s3 = np.flatnonzero(e3['pi'] > 0.5)
        out['BASE_CUM'] = np.column_stack([X, CUMZ])[:, s3] @ np.full(len(s3), 1 / len(s3)); d['survivors_changed_by_cum'] = sorted(set(s3) - {len(surv)}) != list(range(len(surv))); d['cum_pi_hat'] = float(e3['pi'][-1]); d['cum_kept'] = bool(len(surv) in s3)
        # GRP (= P0__EQ_DSM) and GRP + a POS / CUM group
        T = TTA[bk][:, surv]; anchor = int(np.flatnonzero(surv == 0)[0])
        gb = MS.canon(TC.lsml_fit_scaled(T[fit_rows], anchor, X[fit_rows], standardize=True, loading_scale='unit')['groups'])
        gbm, _ = MS.absorb_merge(np.corrcoef(T[fit_rows], rowvar=False), gb); G = int(gbm.max()) + 1
        if G > KEYG.shape[1]: hard_stop(f'{bk} fold {k}: {G} groups exceed the key')
        Z = GW.group_matrix(X, off, gbm, None, answer_standardize); gv = SB.random_tie_marks(Z, off, .2, KEYG[:, :G])
        if not marks_ok(gv): hard_stop(f'group mark counts {bk} fold {k}')
        eg = SA.em_estimate(gv[pf], 'ds'); w = SB.mle_weights(eg['psi'], eg['eta'])
        if w.sum() <= 0: hard_stop(f'{bk} fold {k}: GRP weights all 0')
        out['GRP'] = GW.weighted_group_score(Z, w)
        for tag, col, mk in (('POS', POSZ, POSM), ('CUM', CUMZ, CUMM)):
            ex = SA.em_estimate(np.column_stack([gv, mk])[pf], 'ds'); wx = SB.mle_weights(ex['psi'], ex['eta'])
            if wx.sum() <= 0: hard_stop(f'{bk} fold {k}: GRP_{tag} weights all 0')
            out[f'GRP_{tag}'] = GW.weighted_group_score(np.column_stack([Z, col]), wx)
            d[f'grp_{tag.lower()}_weight_share'] = float(wx[-1] / wx.sum()); d[f'grp_{tag.lower()}_pi_hat'] = float(ex['pi'][-1])
        out['POS_ALONE'] = POSZ
        for a in ('BASE', 'EQ_DSM'):
            mine = out['BASE'] if a == 'BASE' else out['GRP']
            replay[f'{bk}__{a}_fold{k}'] = float(np.max(np.abs(mine[ev] - REF[f'{bk}__{a}'][ev])))
            if not replay[f'{bk}__{a}_fold{k}'] <= 1e-9: hard_stop(f'{bk} fold {k}: {a} replay {replay[f"{bk}__{a}_fold{k}"]}')
        for a in ARMS: put(f'{bk}__{a}', k, ev, out[a], cal)
        d |= {'K': G, 'n_survivors': int(len(surv))}; diag.append(d)
        print(f"  fold {k} {bk}: POS pi {d['pos_pi_hat_with_bank']:.3f} kept {d['pos_kept']} | CUM pi {d['cum_pi_hat']:.3f} kept {d['cum_kept']} | POS group share {d['grp_pos_weight_share']:.3f} CUM group share {d['grp_cum_weight_share']:.3f}", flush=True)
    print(f'fold {k} done ({time.perf_counter()-t:.0f}s)', flush=True)
checks['replay_max'] = max(replay.values()); checks['written_once'] = all(written[m].max() <= 1 for m in ALL)
pd.DataFrame(diag).to_csv(OUT / 'CHANNEL_DECISIONS.csv', index=False); np.savez_compressed(OUT / 'STEP_SCORES.npz', offsets=off, **scores); dump(OUT / 'THRESHOLDS.json', tau)
timing['fit_s'] = time.perf_counter() - T0

# ------------------------------------------------------------------ evaluation (as algorithm_decisions_run.py)
def prmscore_from_counts(tp, fp, tn, fn):
    def ratio(a, b): return np.divide(a, b, out=np.full(np.shape(a), -1., float), where=b != 0)
    p = ratio(tp, tp + fp); r = ratio(tp, tp + fn); f = ratio(2 * p * r, p + r); p2 = ratio(tn, tn + fn); r2 = ratio(tn, tn + fp); nf = ratio(2 * p2 * r2, p2 + r2); return (f + nf) / 2
def earliest_argmax(v): return int(np.flatnonzero(v >= v.max() - 8 * np.finfo(float).eps)[0])
PBC = sorted(set(cells[pb])); cell_idx = np.array([PBC.index(c) if c in PBC else -1 for c in cells]); cov = np.isin(fold, FOLDS)
aucA, confA, hitA, metrics = {}, {}, {}, []
for m in ALL:
    s = scores[m]
    aucA[m] = np.array([within_auc(labels[off[i]:off[i+1]], s[off[i]:off[i+1]]) if eligible[i] and cov[i] else np.nan for i in range(n)])
    confA[m] = np.zeros((n, 4)); flags = {}
    for i in np.flatnonzero(prm & cov):
        a, b = off[i:i+2]; vv = zt(s[a:b]) < tau[m][fold[i]]; gg = ~labels[a:b]; flags[i] = vv
        if noncontrol[i]: confA[m][i] = [(vv & gg).sum(), (vv & ~gg).sum(), (~vv & ~gg).sum(), (~vv & gg).sum()]
    hitA[m] = np.array([float(earliest_argmax(s[off[i]:off[i+1]]) == target[i]) if pb[i] and target[i] >= 0 and cov[i] else np.nan for i in range(n)])
    ev_prm = [i for i in np.flatnonzero(prm & cov)]
    tot = prmbench_evaluate([{'idx': ids[i], 'labels': flags[i].astype(int).tolist()} for i in ev_prm], [meta[ids[i]] for i in ev_prm])['total']
    percell = [np.nanmean(hitA[m][cell_idx == ci]) for ci in range(len(PBC))]
    metrics.append({'method': m, 'within_auc': float(np.nanmean(aucA[m])), 'prmscore': float(.5 * (tot['f1'] + tot['negative_f1'])), 'pb_sla_macro8': float(np.mean(percell)),
                    **{f'pb_{c}': v for c, v in zip(PBC, percell)}})
M = pd.DataFrame(metrics); M.to_csv(OUT / 'METRICS.csv', index=False); timing['eval_s'] = time.perf_counter() - T0 - timing['fit_s']

# ------------------------------------------------------------------ paired source-group bootstrap (per-arm draws on the common population)
Gpr, gi = np.unique(groups[prm], return_inverse=True); gpr = np.full(n, -1); gpr[prm] = gi
Gpb, gj = np.unique(groups[pb], return_inverse=True); gpbx = np.full(n, -1); gpbx[pb] = gj
e_all = cov & eligible; nc_all = cov & noncontrol; pe_all = cov & pb & (target >= 0)
cntG = np.bincount(gpr[e_all], minlength=len(Gpr)).astype(float); hc = np.zeros((len(Gpb), len(PBC))); np.add.at(hc, (gpbx[pe_all], cell_idx[pe_all]), 1)
st = {}
for m in ALL:
    hs = np.zeros((len(Gpb), len(PBC))); np.add.at(hs, (gpbx[pe_all], cell_idx[pe_all]), hitA[m][pe_all])
    st[m] = {'auc': np.bincount(gpr[e_all], weights=aucA[m][e_all], minlength=len(Gpr)), 'conf': np.stack([np.bincount(gpr[nc_all], weights=confA[m][nc_all, q], minlength=len(Gpr)) for q in range(4)], 1), 'hit': hs}
dr = {m: {e: np.empty(DRAWS, np.float32) for e in ('auc', 'ps', 'sla')} for m in ALL}
rng = np.random.default_rng(SEED); rng2 = np.random.default_rng(SEED + 1); pos = 0
while pos < DRAWS:
    nb = min(2500, DRAWS - pos); W = rng.multinomial(len(Gpr), np.full(len(Gpr), 1 / len(Gpr)), size=nb).astype(float); W2 = rng2.multinomial(len(Gpb), np.full(len(Gpb), 1 / len(Gpb)), size=nb).astype(float)
    den = W @ cntG; hden = W2 @ hc
    for m in ALL:
        dr[m]['auc'][pos:pos+nb] = (W @ st[m]['auc']) / den; dr[m]['ps'][pos:pos+nb] = prmscore_from_counts(*(W @ st[m]['conf']).T)
        dr[m]['sla'][pos:pos+nb] = np.nanmean(np.divide(W2 @ st[m]['hit'], hden, out=np.full(hden.shape, np.nan), where=hden > 0), 1)
    pos += nb
Mx = M.set_index('method')
PRIM = [(f'{bk}__{a}', f'{bk}__{b}') for bk in BANKS for a, b in (('BASE_POS', 'BASE'), ('GRP_POS', 'GRP'))]; K = len(PRIM) * 1
SEC = [(f'{bk}__{a}', f'{bk}__{b}') for bk in BANKS for a, b in (('BASE_CUM', 'BASE'), ('GRP_CUM', 'GRP'), ('GRP_POS', 'BASE'), ('POS_ALONE', 'BASE'), ('GRP', 'BASE'), ('BASE_POS', 'POS_ALONE'), ('GRP_POS', 'POS_ALONE'))]
rows = []
for a, b in PRIM + SEC:
    prim = (a, b) in PRIM; r = {'contrast_id': f'{a} - {b}', 'primary': prim}
    for e, col in (('auc', 'within_auc'), ('ps', 'prmscore'), ('sla', 'pb_sla_macro8')):
        x = dr[a][e].astype(float) - dr[b][e].astype(float); r[f'{col}_delta'] = float(Mx.loc[a, col] - Mx.loc[b, col])
        r[f'{col}_lo95'], r[f'{col}_hi95'] = np.nanquantile(x, [.025, .975]).tolist()
        if prim and e == 'auc': r['within_auc_lo_bonf'], r['within_auc_hi_bonf'] = np.nanquantile(x, [.025 / K, 1 - .025 / K]).tolist()
    rows.append(r)
CON = pd.DataFrame(rows); CON.to_csv(OUT / 'CONTRASTS.csv', index=False); timing['bootstrap_s'] = time.perf_counter() - T0 - timing['fit_s'] - timing['eval_s']

# ------------------------------------------------------------------ nulls and concentration for the primary contrasts
nulls = {}; conc = {}; E = np.flatnonzero(e_all)
for a, b in PRIM + [(f'{bk}__BASE_CUM', f'{bk}__BASE') for bk in BANKS]:
    cid = f'{a} - {b}'; dx = aucA[a][E] - aucA[b][E]; sg = float(np.sign(dx.mean())) or 1.0; top = np.argsort(-sg * dx, kind='stable')[:int(np.ceil(.01 * len(dx)))]
    conc[cid] = {'mean_delta': float(dx.mean()), 'share_from_top1pct': float(dx[top].sum() / dx.sum()) if dx.sum() != 0 else None, 'mean_without_top1pct': float(np.delete(dx, top).mean()), 'trimmed5_mean': float(trim_mean(dx, .05))}
    Rk, loc = C2.within_ranks(np.column_stack([scores[a], scores[b]]), off, E); yE = np.concatenate([labels[off[i]:off[i+1]] for i in E]).astype(float)
    def stat(y): A = np.nanmean(C2.auc_from_ranks(Rk, loc, y), 0); return float(A[0] - A[1])
    obs = stat(yE); nulls[cid] = {'observed': obs}
    for nm_, fn, sd in (('within_answer_shuffle', C2.shuffle_within, 11), ('whole_answer_same_length_swap', C2.swap_same_length, 12)):
        rg = np.random.default_rng(sd); x = np.array([stat(fn(yE, loc, rg)) for _ in range(NPERM)])
        nulls[cid][nm_] = {'mean': float(x.mean()), 'sd': float(x.std()), 'share_ge_observed': float(np.mean(x >= obs))}
    cdiff = obs - nulls[cid]['whole_answer_same_length_swap']['mean']; nulls[cid]['content_diff_descriptive'] = cdiff; nulls[cid]['content_share'] = cdiff / obs if obs > 0 else None
dump(OUT / 'NULLS.json', nulls); dump(OUT / 'CONCENTRATION.json', conc); timing['nulls_s'] = time.perf_counter() - T0 - timing['fit_s'] - timing['eval_s'] - timing['bootstrap_s']
timing['total_s'] = time.perf_counter() - T0; dump(OUT / 'TIMING.json', timing)
status.update({'status': 'COMPLETE', 'finished': datetime.now().isoformat(timespec='seconds'), 'checks': checks}); dump(OUT / 'RUN_STATUS.json', status)
print(M[['method', 'within_auc', 'prmscore', 'pb_sla_macro8']].round(4).to_string(index=False))
print(CON[['contrast_id', 'within_auc_delta', 'within_auc_lo95', 'within_auc_hi95', 'prmscore_delta', 'pb_sla_macro8_delta']].round(4).to_string(index=False))
print(json.dumps({k: {'obs': round(v['observed'], 4), 'swap': round(v['whole_answer_same_length_swap']['mean'], 4), 'content_diff': round(v['content_diff_descriptive'], 4)} for k, v in nulls.items()}, indent=0))
print(json.dumps(timing))
