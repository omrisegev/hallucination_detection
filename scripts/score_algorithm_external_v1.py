"""algorithm_external_v1 external scoring (label-free; never opens external labels or quality files).
Per external cell: F arms = the frozen source bundle applied unchanged; R arms = the same label-free learning on the cell's own
unlabelled steps (thresholds = 0.8 quantile of the cell's own answer-z scores); SW = diagnostic stopping rule; references ct7,
frozen_lsml, frozen_equal copied from the earlier sealed external records by uid (telemetry identity checked).
Writes results/algorithm_external_v1/<run>/<cell>/PREDICTIONS_UNSEALED.json.  Protocol: results/algorithm_external_v1/PROTOCOL.json."""
from pathlib import Path
import hashlib, json, subprocess, sys, time
import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN / '.worktrees/depth-feature-fusion-v1'))
from spectral_utils.lsml_gate_locator_research import answer_standardize  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts/experiments')); sys.path.insert(0, str(ROOT / 'scripts'))
import tail_calib_common as TC, er_stage_a as SA, er_stage_b as SB, lsml_merge_step as MS, ds_group_weights as GW  # noqa: E402
from calfix_common import tail_marks  # noqa: E402

GEN = ROOT / 'results/algorithm_external_v1'; RUN = GEN / (sys.argv[1] if len(sys.argv) > 1 else 'run_20260929')
FE = ROOT / 'results/external_banks_v4'; OLD = MAIN / 'results/lsml_external_generalization_v1/evaluation'
INPUTS = MAIN / 'scratch/external_generalization_private/inputs'
CELLS = {'hard2verify_qwen3_8b': 'hard2verify', 'socratic_qwen3_8b': 'socratic', 'socratic_qwq32b': 'socratic'}
BANKS = ['B16', 'B23', 'B35', 'B54']; DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']; REFS = ['ct7', 'frozen_lsml', 'frozen_equal']
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def stop(msg): raise SystemExit('HARD STOP: ' + msg)
P = json.loads((GEN / 'PROTOCOL.json').read_text(encoding='utf8')); B = json.loads((GEN / 'BUNDLE.json').read_text(encoding='utf8'))
if P.get('status') != 'FROZEN' or B['protocol_sha256'] != sha(GEN / 'PROTOCOL.json'): stop('protocol not frozen or changed since the bundle')
if (RUN / 'SCORING_STATUS.json').exists(): stop(f'{RUN} already scored')
RUN.mkdir(parents=True, exist_ok=True)
PS = pd.read_csv(ROOT / 'results/indbank_lsml_prmbench_v1/POOL_STRUCTURE.csv').set_index('channel')

def masked_std(x, offsets, active):
    out = np.zeros(x.shape)
    for a, b in zip(offsets[:-1], offsets[1:]):
        for j in range(x.shape[1]):
            v = active[a:b, j]; y = x[a:b, j][v]
            if len(y) and y.std() > 1e-12: out[a:b, j][v] = (y - y.mean()) / y.std()
    return out
def build_bank(cols, Xr, nm, dact, offsets):
    plain = [c for c in cols if c not in DIG]; src = [c[4:] if c.startswith('lf__') else c for c in plain]
    Vp = answer_standardize(Xr[:, [nm.index(c) for c in src]], offsets)
    for q, c in enumerate(plain):
        if c.startswith('lf__'): Vp[:, q] *= float(np.sign(PS.loc[c[4:], 'r_level_marginal'])) or 1.0
    Vd = masked_std(Xr[:, [nm.index(c) for c in DIG]], offsets, dact)
    return np.column_stack([Vd[:, DIG.index(c)] if c in DIG else Vp[:, plain.index(c)] for c in cols])
def answer_z(s, offsets):
    out = np.empty(len(s))
    for a, b in zip(offsets[:-1], offsets[1:]): v = s[a:b]; out[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    return out
def between_dependence(Xs, g):
    Rm = np.corrcoef(Xs, rowvar=False); offd = ~np.eye(len(g), dtype=bool); diff = g[:, None] != g[None, :]
    return float(np.abs(Rm[offd & diff]).mean())

t0 = time.perf_counter(); status = {'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'bundle_sha256': sha(GEN / 'BUNDLE.json'), 'cells': {}}
for cell, bench in CELLS.items():
    tc = time.perf_counter()
    F = np.load(FE / cell / 'FEATURES.npz'); nm = [str(x) for x in F['names']]; Xr = F['values']; DA = F['digit_active']; off = F['offsets']
    uids = [str(u) for u in F['uids']]; tele = [str(t) for t in F['telemetry_sha256']]; lens = F['nonempty_lengths']; ne = F['nonempty'].astype(bool)
    starts = np.concatenate([[0], np.cumsum(lens)])
    answers = {r['uid']: r for r in json.loads((INPUTS / bench / 'answers.json').read_text(encoding='utf8'))}   # public text/steps, no labels
    if set(answers) != set(uids): stop(f'{cell}: answer population differs from the features')
    for i, u in enumerate(uids):
        if len(answers[u]['steps']) != lens[i] or int(ne[starts[i]:starts[i+1]].sum()) != off[i+1] - off[i]: stop(f'{cell}: step alignment {u}')
    Sc = int(off[-1]); risk = {}; thr = {}; diag = {}
    for bk in BANKS:
        bb = B['banks'][bk]; cols = bb['channels']; V = build_bank(cols, Xr, nm, DA, off)
        # ---- F: frozen from development
        surv = [cols.index(c) for c in bb['survivors']]; Xs = V[:, surv]; g = np.asarray(bb['group_labels'])
        Z = GW.group_matrix(Xs, off, g, None, answer_standardize); w = np.asarray(bb['group_weights'])
        risk[f'F_{bk}_BASE'] = answer_z(Xs.mean(1), off); thr[f'F_{bk}_BASE'] = bb['thresholds']['BASE']
        risk[f'F_{bk}_GRP'] = answer_z(Z @ w, off); thr[f'F_{bk}_GRP'] = bb['thresholds']['GRP']
        rule = B['stopping_rule']; stat = {'S1_between_dependence': between_dependence(Xs, g), 'S2_largest_group_share': float(np.bincount(g).max() / len(g))}
        if rule['statistic'] not in stat:
            G = int(g.max()) + 1; gv = SB.random_tie_marks(Z, off, .2, np.random.default_rng(20260929).random((Sc, 13))[:, :G])
            Q = np.cov(gv.astype(float), rowvar=False, bias=True); lam, v = SA.rank_one_completion(Q); o2 = ~np.eye(G, dtype=bool)
            stat['S3_rank_one_misfit'] = float(np.linalg.norm((Q - lam * np.outer(v, v))[o2]) / np.linalg.norm(Q[o2]))
        use = stat[rule['statistic']] <= rule['tau']
        risk[f'F_{bk}_SW'] = risk[f'F_{bk}_GRP'] if use else risk[f'F_{bk}_BASE']; thr[f'F_{bk}_SW'] = thr[f'F_{bk}_GRP'] if use else thr[f'F_{bk}_BASE']
        # ---- R: label-free refit on the cell itself (all its steps)
        m = V.shape[1]; allrows = np.arange(Sc)
        marks = SB.random_tie_marks(V, off, .2, np.random.default_rng(20260928).random((Sc, m)))
        est = SA.em_estimate(marks, 'ds'); rs = np.flatnonzero(est['pi'] > 0.5)
        if 0 not in rs: stop(f'{cell} {bk}: anchor filtered out in the refit')
        Xr_s = V[:, rs]; T = tail_marks(Xr_s, off, .2, tie_aware=True, centred=True)[0]
        gb = MS.canon(TC.lsml_fit_scaled(T, int(np.flatnonzero(rs == 0)[0]), Xr_s, standardize=True, loading_scale='unit')['groups'])
        gbm, seq = MS.absorb_merge(np.corrcoef(T, rowvar=False), gb); G2 = int(gbm.max()) + 1
        Z2 = GW.group_matrix(Xr_s, off, gbm, None, answer_standardize); gv2 = SB.random_tie_marks(Z2, off, .2, np.random.default_rng(20260929).random((Sc, 13))[:, :G2])
        estG = SA.em_estimate(gv2, 'ds'); w2 = SB.mle_weights(estG['psi'], estG['eta'])
        if w2.sum() <= 0: stop(f'{cell} {bk}: refit group weights all 0')
        risk[f'R_{bk}_BASE'] = answer_z(Xr_s.mean(1), off); risk[f'R_{bk}_GRP'] = answer_z(Z2 @ (w2 / w2.sum()), off)
        for a in ('BASE', 'GRP'): thr[f'R_{bk}_{a}'] = float(np.quantile(risk[f'R_{bk}_{a}'], .8, method='linear'))
        diag[bk] = {'switch_statistics': stat, 'switch_uses_grouping': bool(use),
                    'refit': {'survivors': [cols[j] for j in rs], 'dropped': [cols[j] for j in range(m) if j not in set(rs)], 'prevalence_hat': float(est['prevalence']),
                              'partition': [[cols[rs[j]] for j in np.flatnonzero(gbm == h)] for h in range(G2)], 'merged': not np.array_equal(gb, gbm), 'merge_log': seq,
                              'group_weights': (w2 / w2.sum()).tolist()}}
        print(f'  {cell} {bk}: frozen survivors {len(surv)}, refit survivors {len(rs)}/{m}, refit groups {G2}, switch S1 {stat.get("S1_between_dependence"):.3f} -> {"GRP" if use else "BASE"}', flush=True)
    # ---- assemble full-length rows (empty steps: score null, prediction 0) + references
    rows = {}
    oldmap = {}
    for fpath in sorted((OLD / cell / 'shard_000').glob('*.record.json')):
        oldmap[json.loads(fpath.read_text(encoding='utf8'))['uid']] = fpath
    if set(oldmap) != set(uids): stop(f'{cell}: reference records do not cover the population')
    for i, u in enumerate(uids):
        a, b = off[i], off[i+1]; mask = ne[starts[i]:starts[i+1]]; n = int(lens[i])
        old = json.loads(oldmap[u].read_text(encoding='utf8'))
        if old['uid'] != u or old['payload']['telemetry_sha256'] != tele[i]: stop(f'{cell}: reference telemetry identity {u}')
        if old['payload']['nonempty'] != mask.tolist(): stop(f'{cell}: reference nonempty mask {u}')
        sc, pr = {}, {}
        for arm, s in risk.items():
            full = np.full(n, np.nan); full[mask] = s[a:b]; p = np.zeros(n, int); p[mask] = (s[a:b] < thr[arm]).astype(int)
            sc[arm] = [float(v) if np.isfinite(v) else None for v in full]; pr[arm] = p.tolist()
        for arm in REFS: sc[arm] = old['payload']['scores'][arm]; pr[arm] = old['payload']['predictions'][arm]
        rows[u] = {'scores': sc, 'predictions': pr, 'nonempty': mask.tolist(), 'telemetry_sha256': tele[i]}
    (RUN / cell).mkdir(exist_ok=True)
    (RUN / cell / 'PREDICTIONS_UNSEALED.json').write_text(json.dumps(rows), encoding='utf8')
    (RUN / cell / 'SCORING_DIAGNOSTICS.json').write_text(json.dumps({'thresholds': thr, 'banks': diag}, indent=1, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf8')
    status['cells'][cell] = {'answers': len(uids), 'steps': int(lens.sum()), 'nonempty_steps': Sc, 'arms': list(risk) + REFS, 'seconds': time.perf_counter() - tc}
    print(cell, 'scored', len(uids), 'answers', f'({time.perf_counter()-tc:.0f}s)', flush=True)
status['code'] = {'score_script_sha256': sha(Path(__file__)), 'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()}
status['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S'); status['seconds'] = time.perf_counter() - t0; status['external_labels_opened'] = False
(RUN / 'SCORING_STATUS.json').write_text(json.dumps(status, indent=1), encoding='utf8'); print('SCORING COMPLETE')
