"""Track T: raw telemetry, ProcessBench x qwen_prm_7b. Pre-registration:
docs/experiments/ANSWER_MIXTURE_GATE_PREREG_2026-10-29.md.

Step 0 (parity, required before any result): from the locally rebuilt bank, recompute TELE_N16 and CHAIN_N16 with the
exact logic of score_chain_pb_integration.py (origin/claude/integration-2026-10-17, the scorer that wrote the stored
SCORES_CHAIN_qwen_prm.npz: guarded L-SML, answer-z bank) and compare with the stored scores (tol 1e-6). Stop on fail.

Step scores (higher = more likely wrong; no per-answer standardization anywhere below):
  globally standardized raw channel = each N16 raw bank column z-scored over ALL steps of the cell; the three digit
    channels over their ACTIVE steps (inactive steps 0), the global analogue of the bank's masked_std.
  T-MEAN : plain mean of the 16 globally standardized raw channels.
  T-LSML : TELE_N16's fitted weights (group within-weights x cross weights, with its anchor flip) on those channels.
  T-CHAIN: CHAIN_N16's fitted cross weights (with its flip) on [global-z PRM raw risk, global-z of each T-LSML group
    score (group within-weights on the globally standardized channels)].
Answer score, covariates (log n steps, log answer tokens from the raw spans), mixtures: as Track P (mixture.py).
Labels are read only after all fits.
    python -I -B results/answer_mixture_gate_v1/trackT/track_t.py <rebuilt bank dir>
"""
import csv, importlib.util, json, os, sys, types
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE.parent))
import mixture as MX  # noqa: E402

STORED = Path(os.environ.get('COLLECTION_DATA', ROOT.parent / 'hd_collection_data')) / 'processbench_scores'
PB_GOLD = ROOT / 'results/prm_boost_processbench_v1/local/processbench_gold.json'
TOK = HERE / 'PB_QWEN_ANSWER_TOKENS.npz'
DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
TOL = 1e-6

# --- the frozen modules, loaded exactly as score_chain_pb_integration.py does (M = repo root)
M = ROOT
_src = M / 'results/prm_boosting_generalization_v1/run_v1/evaluate_amended.py'
EV = types.ModuleType('ev'); EV.__file__ = str(M / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
sys.modules['ev'] = EV
exec(compile(_src.read_text(), str(_src), 'exec'), EV.__dict__)
from spectral_utils.fusion_utils import lsml_continuous_guarded as lsml_continuous  # noqa: E402
fs = importlib.util.spec_from_file_location('prm_fuse', M / 'results/prm_boosting_v1/fuse.py')
FU = importlib.util.module_from_spec(fs); fs.loader.exec_module(FU); ext = FU.ext


def lsml(V, groups=None, anchor=None):
    """score_chain_pb.lsml, also returning whether the anchor flip was applied."""
    s, d = lsml_continuous(*[V[:, j] for j in range(V.shape[1])], groups=groups)
    s = np.asarray(s, float)
    ref = V[:, 0] if anchor is None else anchor
    flip = bool(np.corrcoef(s, ref)[0, 1] < 0)
    return (-s if flip else s), d, flip


def gz(x):
    x = np.asarray(x, float); return (x - x.mean()) / x.std()


def global_channels(raw, names, active):
    out = np.zeros_like(raw)
    for j, n in enumerate(names):
        if n in DIG:
            a = active[:, DIG.index(n)]; v = raw[a, j]
            out[a, j] = (v - v.mean()) / v.std()
        else:
            out[:, j] = gz(raw[:, j])
    return out


def linear_tele(d, m):
    """Full per-channel weight vector of a fitted L-SML (group within-weights times cross weights)."""
    w = np.zeros(m); cw = np.asarray(d['cross_weights'], float)
    assert len(cw) == len(d['group_weights'])
    for g, (idx, wg) in enumerate(d['group_weights']):
        w[np.asarray(idx, int)] = cw[g] * np.asarray(wg, float)
    return w


def main(bank):
    F = dict(np.load(bank / 'FEATURES.npz')); R = dict(np.load(bank / 'REWARDS.npz'))
    assert np.array_equal(F['uids'], R['uids']) and np.array_equal(F['nonempty_lengths'], R['nonempty_lengths'])
    ne = F['nonempty'].astype(bool); o = F['offsets'].astype(np.int64)
    F['cells'] = np.array([EV.CELL] * len(F['uids'])); F['row_ids'] = F['uids']
    ans = np.arange(len(F['uids']))
    pop = ext.Population(F, ['N16'])
    prm = EV.risk(np.clip(R['rewards'][ne], 1e-6, 1 - 1e-6)); zp = EV.az(prm, o)
    b = pop.cell_bank('N16', ans); assert np.array_equal(b['off'], o)
    V = np.asarray(b['V'], float)
    tele, dt, flip_t = lsml(V, anchor=V.mean(1))
    vg = np.asarray(dt['virtual_classifiers'], float)
    VC = np.column_stack([zp] + [EV.az(vg[:, g], o) for g in range(vg.shape[1])])
    chain, dc, flip_c = lsml(VC, groups=np.arange(VC.shape[1]))

    # ---------------- parity (before anything else is computed)
    S = np.load(STORED / 'SCORES_CHAIN_qwen_prm.npz')
    assert np.array_equal(S['uids'], F['uids']) and np.array_equal(S['nonempty'], ne)
    par = {'TELE_N16': float(np.max(np.abs(tele - S['score__TELE_N16']))),
           'CHAIN_N16': float(np.max(np.abs(chain - S['score__CHAIN_N16']))),
           'PRM': float(np.max(np.abs(prm - S['score__PRM'])))}
    par['pass'] = bool(par['TELE_N16'] <= TOL and par['CHAIN_N16'] <= TOL)
    par.update(tol=TOL, bank=str(bank), tele_groups=np.asarray(dt['c']).tolist(), tele_K=int(dt['K']),
               tele_flip=flip_t, chain_flip=flip_c, chain_cross_weights=np.asarray(dc['cross_weights'], float).tolist())
    (HERE / 'PARITY.json').write_text(json.dumps(par, indent=1))
    print('PARITY', json.dumps({k: par[k] for k in ('TELE_N16', 'CHAIN_N16', 'PRM', 'pass')}), flush=True)
    if not par['pass']:
        raise SystemExit('PARITY FAIL: Track T not run')

    # sanity: the bank V is the plain answer-z of the raw columns (orientation as stored, no sign flips)
    names = list(b['names']); raw = np.asarray(b['raw'], float); m = len(names)
    assert not b['diagnostics']['sign_flips']
    w_t = linear_tele(dt, m)
    lin_gap = float(np.max(np.abs((-1 if flip_t else 1) * (V @ w_t) - tele)))
    assert lin_gap < 1e-8, lin_gap

    # ---------------- global (not per-answer) step scores
    act = F['digit_active'].astype(bool)[b['rows']]
    Xg = global_channels(raw, names, act)
    t_mean = Xg.mean(1)
    t_lsml = (-1 if flip_t else 1) * (Xg @ w_t)
    groups_g = [Xg[:, np.asarray(idx, int)] @ np.asarray(wg, float) for idx, wg in dt['group_weights']]
    VCg = np.column_stack([gz(prm)] + [gz(g) for g in groups_g])
    cw_c = np.asarray(dc['cross_weights'], float)
    assert all(len(idx) == 1 for idx, _ in dc['group_weights'])
    chain_lin_gap = float(np.max(np.abs((-1 if flip_c else 1) * (VC @ cw_c) - chain)))
    assert chain_lin_gap < 1e-8, chain_lin_gap
    t_chain = (-1 if flip_c else 1) * (VCg @ cw_c)
    steps = {'T-MEAN': t_mean, 'T-LSML': t_lsml, 'T-CHAIN': t_chain}

    # covariates
    T = np.load(TOK); pos = {u: j for j, u in enumerate(T['uids'].astype(str))}
    uids = F['uids'].astype(str); j = np.array([pos[u] for u in uids]); nst = np.diff(o)
    assert np.array_equal(T['n_nonempty'][j], nst)
    cov = [np.log(nst), np.log(T['answer_tokens'][j].astype(float))]

    fits = {}
    for arm, st in steps.items():
        sd = float((np.add.reduceat(st, o[:-1]) / nst).std()); assert sd > 1e-3, (arm, sd)
        for sname, sfun in (('max', MX.answer_max), ('top2', MX.answer_top2)):
            s_a = sfun(st, o); res, beta = MX.residualize(s_a, cov)
            for model in ('raw', 'yeo-johnson'):
                fits[(arm, sname, model)] = (s_a, res, beta, sd, MX.fit_mixture(res, None if model == 'raw' else 'yeo-johnson'))

    # ---------------- labels: only now, only to score
    G = {g['uid']: g for g in json.loads(PB_GOLD.read_text())}
    y = np.array([G[u]['label'] != -1 for u in uids])
    rows, fj = [], {}
    for (arm, sname, model), (s_a, res, beta, sd, f) in fits.items():
        s = MX.score(f, y, s_a)
        rows.append(dict(track='T', bench='processbench', prm='qwen_prm_7b', score=arm, answer_score=sname, model=model,
                         primary=(sname == 'max' and model == 'raw'), covariates='log_n_steps+log_answer_tokens',
                         per_answer_mean_sd=sd,
                         **{k: s[k] for k in ('n', 'pi_true', 'pi_hat', 'abs_err', 'bic1', 'bic2', 'bic2_minus_bic1',
                                              'bic_prefers_2', 'converged', 'clean_closed', 'err_kept',
                                              'share_called_clean', 'gate_useful', 'auroc_s_a')},
                         auroc_residual=MX.auc(y, res)))
        fj[f'{arm}|{sname}|{model}'] = {'beta': beta.tolist(), 'means': s['means'], 'sds': s['sds'],
                                        'weights': s['weights'], 'yj_lambda': s['yj_lambda']}
        if sname == 'max' and model == 'raw':
            print(f"{arm:8s} pi={s['pi_true']:.3f} pi_hat={s['pi_hat']:.3f} |d|={s['abs_err']:.3f} "
                  f"dBIC={s['bic2_minus_bic1']:.1f} clean_closed={s['clean_closed']:.3f} err_kept={s['err_kept']:.3f} "
                  f"AUROC={s['auroc_s_a']:.3f}", flush=True)
    with open(HERE / 'TRACK_T.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (HERE / 'TRACK_T_FITS.json').write_text(json.dumps(fj, indent=1))
    # verdict vs Track P's same cell (primary)
    P = [r for r in csv.DictReader(open(HERE.parent / 'TRACK_P.csv'))
         if r['bench'] == 'processbench' and r['prm'] == 'qwen_prm_7b' and r['primary'] == 'True'][0]
    p_err = float(P['abs_err']); prim = [r for r in rows if r['primary']]
    ok = [r['score'] for r in prim if r['abs_err'] <= .05 and r['abs_err'] < p_err]
    verdict = {'rule': '|pi_hat - pi| <= .05 for at least one of T-MEAN/T-LSML/T-CHAIN and closer to pi than Track P (primary)',
               'track_p_abs_err_same_cell': p_err, 'abs_err': {r['score']: r['abs_err'] for r in prim},
               'passing_scores': ok, 'pass': bool(ok),
               'gate_useful_primary': [r['score'] for r in prim if r['gate_useful']]}
    (HERE / 'TRACK_T_VERDICT.json').write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
