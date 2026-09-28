"""partition_switch_v1: a label-free stopping rule that switches, per bank and fold, between the grouped DS-estimate fusion
(algorithm_decisions_v1 P0__EQ_DSM / P0__HEM_DSM) and the DS-filtered plain average (P0__BASE), plus a parameter-free blend.
Frozen protocol: results/partition_switch_v1/PROTOCOL.json.  No method is refitted: label-free statistics are computed on each
fold's fit rows from the recorded partitions, and the stored scores of algorithm_decisions_v1 run_20260928 are recombined.
The rule for (bank b, fold k) is chosen on cells with bank != b AND fold != k (the banks share their answers)."""
from pathlib import Path
import hashlib, json, pickle, sys, time
from datetime import datetime
import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN / '.worktrees/depth-feature-fusion-v1'))
from spectral_utils.lsml_gate_locator_research import answer_standardize  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
from scipy.stats import rankdata, trim_mean  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
import er_stage_a as SA, er_stage_b as SB, er_stage_b2 as C2, ds_group_weights as GW  # noqa: E402

GEN = ROOT / 'results/partition_switch_v1'; OUT = GEN / (sys.argv[1] if len(sys.argv) > 1 else 'run_20260928'); OUT.mkdir(parents=True, exist_ok=True)
AD = ROOT / 'results/algorithm_decisions_v1/run_20260928'
P = json.loads((GEN / 'PROTOCOL.json').read_text(encoding='utf8'))
T0 = time.perf_counter()
def jd(v): return v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v)
def dump(p, v): Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=jd), encoding='utf8')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
if (OUT / 'RUN_STATUS.json').exists(): raise SystemExit(f'{OUT} already holds a run')
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds')}; checks = {}; dump(OUT / 'RUN_STATUS.json', status)
def hard_stop(reason):
    status.update({'status': 'STOPPED', 'reason': reason, 'checks': checks}); dump(OUT / 'RUN_STATUS.json', status); raise SystemExit('HARD STOP: ' + reason)
def _crash(et, ev, tb):
    import traceback; traceback.print_exception(et, ev, tb)
    if status.get('status') == 'RUNNING': status.update({'status': 'CRASHED', 'reason': repr(ev), 'checks': checks}); dump(OUT / 'RUN_STATUS.json', status)
sys.excepthook = _crash
import subprocess
if P.get('status') != 'FROZEN': hard_stop('protocol not frozen')
dump(OUT / 'CODE_MANIFEST.json', {'script_sha256': sha(Path(__file__)), 'protocol_sha256': sha(GEN / 'PROTOCOL.json'), 'helpers': {h: sha(ROOT / 'scripts/experiments' / h) for h in ('er_stage_a.py', 'er_stage_b.py', 'er_stage_b2.py', 'ds_group_weights.py')},
      'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()})

# ------------------------------------------------------------------ population (as algorithm_decisions_run.py)
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Zs = np.load(R / 'OOF_STEP_SCORES.npz')
off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans); ns = np.diff(off); S = int(off[-1])
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); cells = ans.cell.to_numpy(); groups = ans.source_group.to_numpy(); ids = ans.id.to_numpy(); target = ans.target.to_numpy()
freeze = json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig')); meta = {m['idx']: m for m in pickle.load(open(Path(freeze['prm_metadata']['path']), 'rb')).values()}
noncontrol = np.array([prm[i] and meta[ids[i]]['classification'] != 'correct' for i in range(n)])
prm_steps = np.repeat(prm, ns)
eligible = np.array([prm[i] and labels[off[i]:off[i+1]].any() and (~labels[off[i]:off[i+1]]).any() for i in range(n)]); E = np.flatnonzero(eligible)
def rows_of(mask): return np.concatenate([np.arange(off[i], off[i+1]) for i in np.flatnonzero(mask)])
def zt(s): return (s - s.mean()) / max(s.std(), 1e-8)

# ------------------------------------------------------------------ banks (as algorithm_decisions_run.py)
MI = json.loads((AD / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))
paths = {k: Path(v['path']) for k, v in MI.items() if isinstance(v, dict) and 'path' in v}
for k in ('level_bank', 'ct7_profiles', 'ct7_profile_validation', 'pool_z', 'pool_names', 'digit_features', 'oof_answers', 'oof_step_scores', 'pool_structure'):
    if sha(paths[k]) != MI[k]['sha256']: hard_stop(f'input {k} differs from the algorithm_decisions_v1 manifest')
checks['step_scores_sha256'] = sha(AD / 'STEP_SCORES.npz'); checks['activity_sha256'] = sha(AD / 'ACTIVITY.jsonl')
lv = np.load(paths['level_bank']); names11 = list(map(str, lv['channels']))
prof = np.load(paths['ct7_profiles']).astype(float); pn = json.loads(paths['ct7_profile_validation'].read_text(encoding='utf8'))['channels']
values = answer_standardize(np.column_stack([lv['level'].astype(float), prof[:, pn.index('chosen_token_z_despiked')], lv['derivative'][:, names11.index('chosen_surprisal')].astype(float)]), off)
DF = np.load(paths['digit_features']); act = DF['active'].astype(bool); DFV = DF['values']; D3 = np.zeros((S, 3))          # load once (LESSONS: never index an npz in a loop)
for a, b in zip(off[:-1], off[1:]):
    for j in range(3):
        v = act[a:b, j]; y = DFV[a:b, j][v]
        if len(y) and y.std() > 1e-12: D3[a:b, j][v] = (y - y.mean()) / y.std()
POOL = np.load(paths['pool_z']); PN = json.loads(paths['pool_names'].read_text(encoding='utf8'))
BN = MI['banks']
BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']
def bank(bk):
    cols = []
    for c in BN[bk]:
        if c in names11 + ['realized_z', 'realized_drv'] and bk in ('B13', 'B16'): cols.append(values[:, (names11 + ['realized_z', 'realized_drv']).index(c)])
        elif c in ('digit_alternative', 'digit_spread', 'digit_alternative_innovation'): cols.append(D3[:, ['digit_alternative', 'digit_spread', 'digit_alternative_innovation'].index(c)])
        else: cols.append(None)
    need = [i for i, c in enumerate(cols) if c is None]
    if need:
        raw = []
        for i in need:
            c = BN[bk][i]; base = c[4:] if c.startswith('lf__') else c; col = POOL[:, PN.index(base)]
            raw.append(col)
        Vp = answer_standardize(np.column_stack(raw), off)
        if bk in ('B32', 'B35'):                                                      # label-free orientation of Step 439 for the IND channels
            PS = pd.read_csv(ROOT / 'results/indbank_lsml_prmbench_v1/POOL_STRUCTURE.csv').set_index('channel')
            for q, i in enumerate(need):
                c = BN[bk][i]
                if c.startswith('lf__'): Vp[:, q] *= float(np.sign(PS.loc[c[4:], 'r_level_marginal'])) or 1.0
        for q, i in enumerate(need): cols[i] = Vp[:, q]
    return np.column_stack(cols)
_SCZ = np.load(AD / 'STEP_SCORES.npz'); SC = {f'{bk}__P0__{a}': _SCZ[f'{bk}__P0__{a}'] for bk in BANKS for a in ('BASE', 'EQ_DSM', 'HEM_DSM')}; del _SCZ   # load once
TH = json.loads((AD / 'THRESHOLDS.json').read_text(encoding='utf8'))
ACT = {(d['bank'], d['fold']): d for d in map(json.loads, open(AD / 'ACTIVITY.jsonl', encoding='utf8')) if d['learn'] == 'raw'}
KEYG = np.random.default_rng(20260929).random((S, 13))

# ------------------------------------------------------------------ label-free statistics per bank and fold (+ BASE replay)
t = time.perf_counter(); stats = []; replay = {}
for bk in BANKS:
    V = bank(bk); nm = BN[bk]
    for k in range(5):
        cal = (k + 1) % 5; fit_rows = rows_of(np.isin(fold, [f for f in range(5) if f not in (k, cal)])); pf = fit_rows[prm_steps[fit_rows]]; ev_rows = rows_of(fold == k)
        d = ACT[(bk, k)]; surv = [nm.index(c) for c in d['survivors']]; X = V[:, surv]
        replay[f'{bk}_fold{k}'] = float(np.max(np.abs(X[ev_rows] @ np.full(len(surv), 1 / len(surv)) - SC[f'{bk}__P0__BASE'][ev_rows])))
        g = np.zeros(len(surv), int)
        for h, gr in enumerate(d['part_binM']):
            for c in gr: g[d['survivors'].index(c)] = h
        G = int(g.max()) + 1
        Rm = np.corrcoef(X[fit_rows], rowvar=False); offd = ~np.eye(len(g), dtype=bool); diff = g[:, None] != g[None, :]
        s1 = float(np.abs(Rm[offd & diff]).mean()); s2 = float(np.bincount(g).max() / len(g))
        Z = GW.group_matrix(X, off, g, None, answer_standardize); gv = SB.random_tie_marks(Z, off, .2, KEYG[:, :G])
        Q = np.cov(gv[pf].astype(float), rowvar=False, bias=True); lam, v = SA.rank_one_completion(Q); o2 = ~np.eye(G, dtype=bool)
        s3 = float(np.linalg.norm((Q - lam * np.outer(v, v))[o2]) / np.linalg.norm(Q[o2]))
        stats.append({'bank': bk, 'fold': k, 'K': G, 'S1_between_dependence': s1, 'S2_largest_group_share': s2, 'S3_rank_one_misfit': s3})
checks['base_replay_max'] = max(replay.values())
if not checks['base_replay_max'] <= 1e-9: hard_stop(f'BASE replay failed {checks["base_replay_max"]}')
ST = pd.DataFrame(stats); print(f'statistics ready ({time.perf_counter()-t:.0f}s); base replay {checks["base_replay_max"]:.1e}', flush=True)

# ------------------------------------------------------------------ per-answer AUC, gains per (bank, fold)
def answer_auc(s):
    out = np.full(n, np.nan)
    Rk, loc = C2.within_ranks(s[:, None], off, E); yE = np.concatenate([labels[off[i]:off[i+1]] for i in E]).astype(float)
    out[E] = C2.auc_from_ranks(Rk, loc, yE)[:, 0]; return out
AUC = {}
for bk in BANKS:
    for a in ('P0__BASE', 'P0__EQ_DSM', 'P0__HEM_DSM'): AUC[f'{bk}__{a}'] = answer_auc(SC[f'{bk}__{a}'])
STATN = ['S1_between_dependence', 'S2_largest_group_share', 'S3_rank_one_misfit']
for on in ('EQ_DSM', 'HEM_DSM'):
    ST[f'gain_{on}'] = [float(np.nanmean((AUC[f'{r.bank}__P0__{on}'] - AUC[f'{r.bank}__P0__BASE'])[eligible & (fold == r.fold)])) for r in ST.itertuples()]

# ------------------------------------------------------------------ the leak-free rule and the switched / blended scores
def choose(train, on):
    best = None
    for j, sn in enumerate(STATN):
        cand = [-np.inf] + sorted(train[sn].unique()) + [np.inf]
        for tau in cand:
            obj = float(np.mean(train[f'gain_{on}'] * (train[sn] <= tau)))
            key = (round(obj, 12), -tau if np.isfinite(tau) else (np.inf if tau < 0 else -np.inf), -j)
            if best is None or key > best[0]: best = (key, sn, tau, obj)
    return best[1], best[2], best[3]
decisions = []; OUTS = {}
for on in ('EQ_DSM', 'HEM_DSM'):
    for bk in BANKS:
        sw = np.full(S, np.nan); bl = np.full(S, np.nan); tau_sw = {}
        for k in range(5):
            train = ST[(ST.bank != bk) & (ST.fold != k)]
            if len(train) != 28: hard_stop(f'{len(train)} training cells for {bk} fold {k}')
            if ((train.bank == bk) | (train.fold == k)).any(): hard_stop('leak in the rule selection')
            sn, tau, obj = choose(train, on); srow = ST[(ST.bank == bk) & (ST.fold == k)].iloc[0]; use = bool(srow[sn] <= tau)
            ev_rows = rows_of(fold == k); arm = f'{bk}__P0__{on}' if use else f'{bk}__P0__BASE'
            sw[ev_rows] = SC[arm][ev_rows]; tau_sw[str(k)] = TH[arm][str(k)]
            decisions.append({'on': on, 'bank': bk, 'fold': k, 'statistic': sn, 'tau': tau, 'training_objective': obj, 'value': float(srow[sn]), 'use_grouped': use,
                              'gain_if_used_label_diag': float(srow[f'gain_{on}'])})
        for i in range(n):                                                                    # parameter-free blend of the two stored scores
            a, b = off[i], off[i + 1]; bl[a:b] = zt(SC[f'{bk}__P0__BASE'][a:b]) + zt(SC[f'{bk}__P0__{on}'][a:b])
        if not (np.isfinite(sw).all() and np.isfinite(bl).all()): hard_stop(f'non-finite switched or blended scores {bk} {on}')
        sfx = '' if on == 'EQ_DSM' else '_HEM'
        OUTS[f'{bk}__SW{sfx}'] = (sw, tau_sw); OUTS[f'{bk}__BLEND{sfx}'] = (bl, None)
        full = float(np.nanmean(AUC[f'{bk}__P0__{on}'] - AUC[f'{bk}__P0__BASE']))             # label-using ceiling (in-sample)
        orc = f'{bk}__P0__{on}' if full > 0 else f'{bk}__P0__BASE'
        OUTS[f'{bk}__ORACLE_SWITCH{sfx}'] = (SC[orc].copy(), TH[orc])
DEC = pd.DataFrame(decisions); DEC.to_csv(OUT / 'DECISIONS.csv', index=False); ST.to_csv(OUT / 'STATS.csv', index=False)
for m, (s, _) in OUTS.items(): AUC[m] = answer_auc(s)

# ------------------------------------------------------------------ evaluation: within-AUC, PRMScore (SW / ORACLE only), ProcessBench SLA
def earliest_argmax(v): return int(np.flatnonzero(v >= v.max() - 8 * np.finfo(float).eps)[0])
PBC = sorted(set(cells[pb])); cell_idx = np.array([PBC.index(c) if c in PBC else -1 for c in cells])
def pb_hits(s): return np.array([float(earliest_argmax(s[off[i]:off[i+1]]) == target[i]) if pb[i] and target[i] >= 0 else np.nan for i in range(n)])
def prmscore(s, taus):
    ev = [i for i in np.flatnonzero(prm)]
    res = prmbench_evaluate([{'idx': ids[i], 'labels': (zt(s[off[i]:off[i+1]]) < taus[str(fold[i])]).astype(int).tolist()} for i in ev], [meta[ids[i]] for i in ev])['total']
    return float(.5 * (res['f1'] + res['negative_f1']))
ALLM = [f'{bk}__P0__{a}' for bk in BANKS for a in ('BASE', 'EQ_DSM', 'HEM_DSM')] + list(OUTS)
HIT = {}; rows = []
for m in ALLM:
    s = SC[m] if m in SC else OUTS[m][0]; HIT[m] = pb_hits(s)
    percell = [np.nanmean(HIT[m][cell_idx == ci]) for ci in range(len(PBC))]
    taus = TH.get(m) if m in SC else OUTS[m][1]
    rows.append({'method': m, 'within_auc': float(np.nanmean(AUC[m])), 'prmscore': prmscore(s, taus) if taus else None, 'pb_sla_macro8': float(np.mean(percell))})
MET = pd.DataFrame(rows); MET.to_csv(OUT / 'METRICS.csv', index=False)
ADM = pd.read_csv(AD / 'METRICS.csv'); ADM = ADM[(ADM.metric == 'within_auc') & (ADM.stratum == 'all')].set_index('method').estimate
checks['stored_arm_within_auc_replay_max'] = max(abs(MET.set_index('method').loc[m, 'within_auc'] - ADM[m]) for m in SC)
if not checks['stored_arm_within_auc_replay_max'] <= 1e-12: hard_stop('stored-arm within-AUC does not replay the runner METRICS')

# ------------------------------------------------------------------ paired source-group bootstrap per bank and for the 8-bank mean
Gs, gi = np.unique(groups[E], return_inverse=True); cnt = np.bincount(gi, minlength=len(Gs)).astype(float)
rng = np.random.default_rng(20260930); W = rng.multinomial(len(Gs), np.full(len(Gs), 1 / len(Gs)), size=20000).astype(float); den = W @ cnt
def draws(a, b):
    dx = AUC[a][E] - AUC[b][E]; return (W @ np.bincount(gi, weights=dx, minlength=len(Gs))) / den, float(dx.mean())
CON = []; succ = {}
for arm in ('SW', 'BLEND', 'SW_HEM', 'BLEND_HEM', 'ORACLE_SWITCH', 'ORACLE_SWITCH_HEM', 'P0__EQ_DSM', 'P0__HEM_DSM'):
    per = []; allx = []
    for bk in BANKS:
        a = f'{bk}__{arm}'; x, pt = draws(a, f'{bk}__P0__BASE'); allx.append(x)
        per.append({'arm': arm, 'bank': bk, 'delta': pt, 'lo': float(np.quantile(x, .025)), 'hi': float(np.quantile(x, .975)),
                    'pb_sla_delta': float(MET.set_index('method').loc[a, 'pb_sla_macro8'] - MET.set_index('method').loc[f'{bk}__P0__BASE', 'pb_sla_macro8'])})
    m8 = np.mean(allx, 0); pt8 = float(np.mean([r['delta'] for r in per]))
    per.append({'arm': arm, 'bank': 'MEAN8', 'delta': pt8, 'lo': float(np.quantile(m8, .025)), 'hi': float(np.quantile(m8, .975))})
    CON += per
    succ[arm] = {'losses': [r['bank'] for r in per[:-1] if r['hi'] < 0], 'wins': [r['bank'] for r in per[:-1] if r['lo'] > 0], 'mean8_delta': pt8, 'mean8_lo': per[-1]['lo'], 'mean8_hi': per[-1]['hi']}
    succ[arm]['success'] = (not succ[arm]['losses']) and per[-1]['lo'] > 0
pd.DataFrame(CON).to_csv(OUT / 'CONTRASTS.csv', index=False)

# ------------------------------------------------------------------ nulls and concentration where the arm differs from BASE
nulls = {}; conc = {}
for arm in ('SW', 'BLEND', 'SW_HEM', 'BLEND_HEM'):
    for bk in BANKS:
        a, b = f'{bk}__{arm}', f'{bk}__P0__BASE'; sa = SC[b]; s1 = OUTS[a][0]
        dx = AUC[a][E] - AUC[b][E]
        if np.all(np.abs(dx) < 1e-12): nulls[f'{a} - {b}'] = conc[f'{a} - {b}'] = {'note': 'identical to BASE'}; continue
        sg = float(np.sign(dx.mean())) or 1.0; top = np.argsort(-sg * dx, kind='stable')[:int(np.ceil(.01 * len(dx)))]
        conc[f'{a} - {b}'] = {'mean_delta': float(dx.mean()), 'share_from_top1pct': float(dx[top].sum() / dx.sum()) if dx.sum() != 0 else None,
                              'mean_without_top1pct': float(np.delete(dx, top).mean()), 'trimmed5_mean': float(trim_mean(dx, .05))}
        Rk, loc = C2.within_ranks(np.column_stack([s1, sa]), off, E); yE = np.concatenate([labels[off[i]:off[i+1]] for i in E]).astype(float)
        def stat(y): A = np.nanmean(C2.auc_from_ranks(Rk, loc, y), 0); return float(A[0] - A[1])
        nulls[f'{a} - {b}'] = {'observed': stat(yE)}
        for nm_, fn, sd in [('within_answer_shuffle', C2.shuffle_within, 11), ('whole_answer_same_length_swap', C2.swap_same_length, 12)]:
            rg = np.random.default_rng(sd); x = np.array([stat(fn(yE, loc, rg)) for _ in range(200)])
            nulls[f'{a} - {b}'][nm_] = {'mean': float(x.mean()), 'sd': float(x.std())}
dump(OUT / 'NULLS.json', nulls); dump(OUT / 'CONCENTRATION.json', conc); dump(OUT / 'SUCCESS.json', succ)
status.update({'status': 'COMPLETE', 'finished': datetime.now().isoformat(timespec='seconds'), 'checks': checks, 'success': succ, 'total_s': time.perf_counter() - T0})
dump(OUT / 'RUN_STATUS.json', status)
print(DEC.pivot_table(index=['on', 'bank'], columns='fold', values='use_grouped', aggfunc='first').to_string())
print(pd.DataFrame(CON).round(4).to_string()); print(json.dumps(succ, indent=1)); print(f'total {time.perf_counter()-T0:.0f}s')
