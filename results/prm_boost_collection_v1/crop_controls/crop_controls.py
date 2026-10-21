"""CROP controls C1-C7 (plan docs/experiments/PRM_BOOSTING_COLLECTION_PLAN_2026-10-21.md, section 2.2, added 2026-10-21).
AIRCC, CPU, one PRM per call. PRMBench labels are read here (allowed on AIRCC); no new inference.

    python3 crop_controls.py <tag> <FEATURES.npz> <telemetry.pkl> <export dir (ABL5_SCORES/CHANNELS)> <jsonl> <out dir>

Protocol = collection_competitors.crop (= results/prmbench_paper_metric_parity_v1/crop_metrics.py): 10 seeds 2806-2815,
trace-level 60/20/20 split stratified by has-error, CROP's lambda rule (largest lambda with (1+#contaminated)/(n+1) <= alpha,
infeasible -> smallest lambda), CROP's label rule (step t is an error iff t+1 in error_steps, values < 1 clamped to 0).

Subsets
  approx : first 5,000 non-control rows of our loader in Hub order (the earlier package; parity with COMP_CROP_T6.csv)
  C1     : first 5,000 RAW rows of hitsmy/PRMBench_Preview (jsonl order) INCLUDING CROP's 5 duplicate classification_idx
           rows. Those 5 have a different process from their twin, so no existing telemetry/PRM score covers them:
           they cannot be scored without new inference and are reported, then left out (4,995 scored traces).
Grids (C2), one label-free monotone map fixed in advance for every proxy:
  fixed101_minmax : u = clip((s - min_train) / (max_train - min_train), 0, 1) over the training-split steps, grid {0,.01,..,1}
  quantile201     : 201 quantiles of the training-split step scores (CROP's target-domain rule; map-invariant)
  fixed101_raw    : CROP's own fixed grid on the raw score, only for proxies already in [0,1] (Direct PRM, LRs, Random)
Proxies: Random; Direct PRM (1-reward); Position/length LR (as COMP_CROP_T6); C3 Format/position LR, Format/position+PRM LR,
  Telemetry13+PRM LR, Telemetry16+PRM LR (supervised, CROP's pipeline); TELE_N16, CHAIN_N16, CHAIN_N13, LSML_CONT
  (label-free, transductive: fitted once on all 6,969 answers); C4: the same four fitted INDUCTIVELY on the 60% training
  traces only (unlabeled), frozen (groups, within-group weights, cross weights, sign) and applied to calibration/test.
C5 label shuffle (supervised LRs: training-trace label vectors permuted among traces with equal step count; label-free and
  PRM: calibration labels permuted the same way), order shuffle (steps permuted within each trace jointly with their labels
  before prefix construction; scores are the ones computed in the original order, since re-scoring permuted text needs
  inference; LR features are rebuilt on the permuted order, so position carries no information).
C6 alpha in {.01,.025,.05,.075,.10,.15,.20}: kept, risk, empty-prefix rate, infeasible flag, 1/(n_cal+1) > alpha mark.
C7 boundary deviation at every alpha: mean |M-O|/T, over-withholding mean max(O-M,0)/T, unsafe overshoot mean max(M-O,0)/T,
  with O = first-error index (T if the trace has no error).
"""
import csv, importlib.util, json, re, sys, time
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

NESTED = {'N11': ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level',
                  'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0']}
NESTED['N13'] = NESTED['N11'] + ['realized_z', 'realized_drv']
NESTED['N16'] = NESTED['N13'] + ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
S3 = ['energy_level', 'hist_entropy_cusum_abs_series', 'realized_drv']
SEEDS = list(range(2806, 2816)); N_TRACES = 5000
ALPHAS = [.01, .025, .05, .075, .10, .15, .20]
import os
if os.environ.get('CROP_SMOKE') == '1':                 # preflight smoke: 1 seed, 2 alphas, 300 traces
    SEEDS, ALPHAS, N_TRACES = [2806], [.05, .10], 300
M = Path('/shared/cycle2_tau_averbuch_prj/omrisegev1/eval_mirror_prm_gen_v1')


# ------------------------------------------------------------------------------------------------ setup (as ablation5.py)
def setup(tag, feat, tele):
    spec = importlib.util.spec_from_file_location('ev', M / 'results/prm_boosting_generalization_v1/evaluate_amended.py')
    EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)
    sys.path.insert(0, str(M))
    from spectral_utils.fusion_utils import lsml_continuous_guarded as lsml_continuous
    F = dict(np.load(feat)); off_all = F['offsets'].astype(np.int64)
    ans = np.flatnonzero(F['cells'].astype(str) == EV.CELL)
    ids = F['row_ids'][ans].astype(str); o = np.r_[0, np.cumsum(np.diff(off_all)[ans])]
    T = EV.metadata_from(tele); meta = [dict(T[i], idx=i) for i in ids]
    prm = np.concatenate([EV.risk(r['rewards']) for r in meta]); zp = EV.az(prm, o)
    fs = importlib.util.spec_from_file_location('prm_fuse', M / 'results/prm_boosting_v1/fuse.py')
    FU = importlib.util.module_from_spec(fs); fs.loader.exec_module(FU); ext = FU.ext
    banks = {'N13': NESTED['N13'], 'N16': NESTED['N16']}
    ext.BN = dict(ext.BN, TRANSFER=list(S3), **banks)
    pop = ext.Population(F, ['TRANSFER'] + list(banks))
    V = {bk: np.asarray(pop.cell_bank(bk, ans)['V'], float) for bk in banks}
    bt = pop.cell_bank('TRANSFER', ans)
    V4 = np.column_stack([zp] + [np.asarray(bt['V'], float)[:, bt['names'].index(c)] for c in S3])
    return dict(EV=EV, lsml_continuous=lsml_continuous, ids=ids, o=o, meta=meta, prm=prm, zp=zp, V=V, V4=V4)


# ------------------------------------------------------------------------------------------------ frozen L-SML
def fit_lsml(L, X, rows, groups=None, anchor=None):
    """lsml() of ablation5 fitted on X[rows]; returns frozen params (group weights, cross weights, sign)."""
    s, d = L(*[X[rows, j] for j in range(X.shape[1])], groups=groups)
    s = np.asarray(s, float); ref = (X[rows, 0] if anchor is None else anchor[rows])
    sign = -1. if np.corrcoef(s, ref)[0, 1] < 0 else 1.
    return dict(gw=[(np.asarray(i, int), np.asarray(w, float)) for i, w in d['group_weights']],
                cw=np.asarray(d['cross_weights'], float), sign=sign, K=int(d['K']), fit_score=sign * s)


def virtual(X, p):
    return np.column_stack([X[:, i[0]].astype(float) if len(i) == 1 else X[:, i] @ w for i, w in p['gw']])


def apply_lsml(X, p):
    vg = virtual(X, p)
    if vg.shape[1] == 1:
        i, w = p['gw'][0]
        s = X[:, i] @ w if len(i) > 1 else vg[:, 0]
    else:
        s = vg @ p['cw']
    return p['sign'] * s


def fused_arms(S, rows):
    """TELE_N16, CHAIN_N16, CHAIN_N13, LSML_CONT fitted on `rows` (unlabeled), applied to every step."""
    L, EV, o, zp = S['lsml_continuous'], S['EV'], S['o'], S['zp']; out = {}
    for bk in ('N16', 'N13'):
        V = S['V'][bk]
        pt = fit_lsml(L, V, rows, anchor=V.mean(1))
        if bk == 'N16': out['TELE_N16'] = apply_lsml(V, pt)
        vg = virtual(V, pt)
        VC = np.column_stack([zp] + [EV.az(vg[:, g], o) for g in range(vg.shape[1])])
        pc = fit_lsml(L, VC, rows, groups=np.arange(VC.shape[1]))
        out[f'CHAIN_{bk}'] = apply_lsml(VC, pc)
    p4 = fit_lsml(L, S['V4'], rows)
    out['LSML_CONT'] = apply_lsml(S['V4'], p4)
    return out


# ------------------------------------------------------------------------------------------------ CROP protocol
def split_traces(has_err, seed):
    rng = np.random.default_rng(seed); tr, ca, te = [], [], []
    for key in (False, True):
        idx = np.flatnonzero(has_err == key).tolist(); rng.shuffle(idx)
        n = len(idx); n_tr = int(round(.6 * n)); n_ca = int(round(.2 * n))
        tr += idx[:n_tr]; ca += idx[n_tr:n_tr + n_ca]; te += idx[n_tr + n_ca:]
    for part in (tr, ca, te): rng.shuffle(part)
    return np.array(tr), np.array(ca), np.array(te)


def first_err(y):
    f = np.flatnonzero(y); return int(f[0]) if len(f) else None


class Prep:
    """Per-trace running max of the scores (padded with +inf), first-error index and length, for a set of traces.
    Exactly CROP's rules, vectorized: prefix length M = first index with score > lambda (= #running-max values <= lambda);
    a calibration trace is contaminated at lambda iff lambda >= running max up to its first error; a test prefix is
    contaminated iff M > first-error index."""
    def __init__(self, sc, labels, idx):
        idx = np.asarray(idx); n = np.array([len(sc[i]) for i in idx]); W = int(n.max()) if len(n) else 1
        R = np.full((len(idx), W), np.inf)
        for r, i in enumerate(idx): R[r, :n[r]] = np.maximum.accumulate(np.asarray(sc[i], float))
        fe = np.array([first_err(labels[i]) if first_err(labels[i]) is not None else -1 for i in idx])
        self.R, self.n, self.fe = R, n, fe


def crop_lambda(P, grid, alpha):
    has = P.fe >= 0
    need = P.R[np.flatnonzero(has), P.fe[has]]
    risk = (1 + (grid[:, None] >= need[None, :]).sum(1)) / (len(P.fe) + 1)
    ok = np.flatnonzero(risk <= alpha)
    return (grid[ok[np.argmax(grid[ok])]] if len(ok) else grid.min()), bool(len(ok))


def crop_eval(P, lam):
    M_ = (P.R <= lam).sum(1).astype(float); T_ = P.n.astype(float)
    O_ = np.where(P.fe >= 0, P.fe, P.n).astype(float); c = (P.fe >= 0) & (M_ > P.fe)
    return dict(kept=100 * float(np.mean(M_ / np.maximum(T_, 1))), risk=100 * float(c.mean()), empty=100 * float(np.mean(M_ == 0)),
                bdev=100 * float(np.mean(np.abs(M_ - O_) / T_)), over_withhold=100 * float(np.mean(np.maximum(O_ - M_, 0) / T_)),
                overshoot=100 * float(np.mean(np.maximum(M_ - O_, 0) / T_)))


def grids(v, rows_tr, raw_ok):
    """Returns {grid name: (per-step transformed score, grid)} for one proxy."""
    lo, hi = float(v[rows_tr].min()), float(v[rows_tr].max())
    u = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1)
    g = {'fixed101_minmax': (u, np.linspace(0, 1, 101)), 'quantile201': (v, np.quantile(v[rows_tr], np.linspace(0, 1, 201)))}
    if raw_ok: g['fixed101_raw'] = (v, np.linspace(0, 1, 101))
    return g


def permute_equal_length(n_steps, idx, rng):
    """Permutation of trace indices `idx` within groups of equal step count."""
    out = np.array(idx).copy()
    for k in np.unique(n_steps[idx]):
        j = np.flatnonzero(n_steps[idx] == k)
        out[j] = np.array(idx)[j][rng.permutation(len(j))]
    return out


# ------------------------------------------------------------------------------------------------ text features (C3)
OPS = re.compile(r'[+\-*/^×÷]')


def format_feats(steps):
    rows = []
    for t, s in enumerate(steps):
        rows.append([len(s), len(s.split()), sum(ch.isdigit() for ch in s), sum(ch.isalpha() for ch in s), s.count('='),
                     len(OPS.findall(s)), s.count('\n'), s.count(':'), s.count('(') + s.count(')'), s.count('<') + s.count('>'),
                     (t + .5) / len(steps), t + 1, len(steps)])
    return np.array(rows, float)


def main():
    tag, feat, tele, exp_dir, jsonl, out = sys.argv[1:7]
    out = Path(out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    S = setup(tag, feat, tele); o = S['o']; ids = S['ids']; n_ans = len(ids)
    E = dict(np.load(Path(exp_dir) / f'ABL5_SCORES_{tag}.npz')); CH = np.load(Path(exp_dir) / f'ABL5_CHANNELS_{tag}.npz')
    assert np.array_equal(E['row_ids'].astype(str), ids) and np.array_equal(E['offsets'], o)
    par = {'tag': tag}
    # C4 parity: the inductive machinery (fit, freeze, apply) fitted on 100% of the steps must equal the exported
    # transductive arms (ablation5.py's lsml() on all 6,969 answers).
    full = np.arange(o[-1]); tr_all = fused_arms(S, full)
    par['C4_inductive100_vs_transductive_export_maxabs'] = {k: float(np.max(np.abs(v - E[f'score__{k}']))) for k, v in tr_all.items()}
    print('parity', json.dumps(par), flush=True)

    raw = [json.loads(l) for l in open(jsonl) if l.strip()]
    pos = {i: j for j, i in enumerate(ids)}
    seen, raw_keys, dups = set(), [], []
    for r in raw:
        k = f"{r['classification']}_{r['idx']}"
        raw_keys.append(k if k not in seen else None)
        if k in seen: dups.append(k)
        seen.add(k)
    text = {}
    for r, k in zip(raw, raw_keys):
        if k is not None: text[k] = r
    subsets = {}
    # earlier package: first 5,000 non-control loader rows in HUB order (= deduplicated jsonl order = prmbench_meta.json
    # order). NOT the FEATURES row order (crop_controls_v1 used that by mistake and failed parity).
    nonc = [k for k in raw_keys if k is not None]
    subsets['approx'] = nonc[:N_TRACES]
    c1_raw = raw[:N_TRACES]; c1_keys = raw_keys[:N_TRACES]
    subsets['C1'] = [k for k in c1_keys if k is not None]
    c1_info = dict(n_raw_rows=len(c1_raw), n_steps_raw=int(sum(len(r['modified_process']) for r in c1_raw)),
                   n_unscored_duplicates=sum(k is None for k in c1_keys),
                   unscored=[dict(pos=j, key=f"{r['classification']}_{r['idx']}", n_steps=len(r['modified_process']), n_error_steps=len(r['error_steps']))
                             for j, (r, k) in enumerate(zip(c1_raw, c1_keys)) if k is None])
    e_raw = sum(sum(t in {max(int(e) - 1, 0) for e in r['error_steps']} for t in range(len(r['modified_process']))) for r in c1_raw)
    c1_info['error_rate_raw'] = e_raw / c1_info['n_steps_raw']
    par['C1'] = c1_info
    tok = np.expm1(E['log_step_tokens']); nst = np.diff(o)
    step_idx = np.concatenate([np.arange(1, k + 1) for k in nst]).astype(float); tot = np.repeat(nst, nst).astype(float)
    pref = np.concatenate([np.r_[0, np.cumsum(tok[a:b])[:-1]] for a, b in zip(o[:-1], o[1:])])
    atok = np.repeat(np.add.reduceat(tok, o[:-1]), nst)
    F_pos = np.column_stack([step_idx, tot, step_idx / tot, tok, pref, atok])
    ch13 = CH['channels__N13'].astype(float); ch16 = CH['channels__N16'].astype(float); prmrisk = E['score__PRM_native']
    F_fmt = np.zeros((o[-1], 13))
    for j, i in enumerate(ids):
        if i in text:
            f = format_feats(text[i]['modified_process'])
            if len(f) == o[j + 1] - o[j]: F_fmt[o[j]:o[j + 1]] = f
    F_list = {'Position/length LR (supervised)': F_pos, 'C3 Format/position LR (supervised)': F_fmt,
              'C3 Format/position+PRM LR (supervised)': np.column_stack([F_fmt, prmrisk]),
              'Telemetry13+PRM LR (supervised)': np.column_stack([prmrisk, ch13]),
              'Telemetry16+PRM LR (supervised)': np.column_stack([prmrisk, ch16])}
    rows_out, c7_out = [], []
    for sname, sub in subsets.items():
        J = np.array([pos[i] for i in sub]); sl = [slice(o[j], o[j + 1]) for j in J]
        labels = []
        for i, s in zip(sub, sl):
            es = text[i]['error_steps'] if (sname == 'C1' and i in text) else S['meta'][pos[i]]['error_steps']
            e0 = {max(int(e) - 1, 0) for e in es}; labels.append(np.array([t in e0 for t in range(s.stop - s.start)]))
        has_err = np.array([l.any() for l in labels]); nst_s = np.array([s.stop - s.start for s in sl])
        par[f'{sname}_subset'] = dict(n_traces=len(sub), n_steps=int(nst_s.sum()), error_rate=float(sum(l.sum() for l in labels) / nst_s.sum()))
        allrows = np.concatenate([np.arange(s.start, s.stop) for s in sl])
        risk1 = 1 - E['reward']
        for seed in SEEDS:
            tr, ca, te = split_traces(has_err, seed)
            rows_tr = np.concatenate([np.arange(sl[k].start, sl[k].stop) for k in tr]); ytr = np.concatenate([labels[k] for k in tr])
            rr = np.random.default_rng(seed)
            cand = {'Random': (np.r_[np.zeros(0)], True), 'Direct PRM (1-reward)': (risk1, True)}
            rnd = np.zeros(o[-1]); rnd[allrows] = np.concatenate([rr.random(s.stop - s.start) for s in sl]); cand['Random'] = (rnd, True)
            for nm, F in F_list.items():
                mdl = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                    LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, class_weight='balanced', random_state=seed))
                mdl.fit(F[rows_tr], ytr); p = np.zeros(o[-1]); p[allrows] = mdl.predict_proba(F[allrows])[:, 1]
                cand[nm] = (p, True)
            for a in ('TELE_N16', 'CHAIN_N16', 'CHAIN_N13', 'LSML_CONT'):
                cand[a] = (E[f'score__{a}'], False)
            if sname == 'C1':                                   # C4: inductive fit on this seed's training traces (unlabeled)
                ind = fused_arms(S, rows_tr)
                for a, v in ind.items(): cand[f'{a} (inductive, C4)'] = (v, False)
            yte = np.concatenate([labels[k] for k in te])
            for nm, (v, raw_ok) in cand.items():
                au = roc_auc_score(yte, np.concatenate([v[sl[k]] for k in te]))
                for gname, (vv, grid) in grids(v, rows_tr, raw_ok).items():
                    sc = [vv[s] for s in sl]; Pc, Pt = Prep(sc, labels, ca), Prep(sc, labels, te)
                    for al in ALPHAS:
                        lam, feas = crop_lambda(Pc, grid, al)
                        r = crop_eval(Pt, lam)
                        rows_out.append(dict(prm=tag, subset=sname, control='none', proxy=nm, grid=gname, alpha=al, seed=seed, auroc=au,
                                             lam=float(lam), feasible=feas, impossible=bool(1 / (len(ca) + 1) > al), **r))
                    if sname == 'C1' and gname != 'fixed101_raw':
                        # C5 label shuffle: supervised -> refit on permuted training labels; others -> permuted calibration labels
                        rng = np.random.default_rng(seed + 1000)
                        if 'supervised' in nm:
                            pt = permute_equal_length(nst_s, tr, rng)
                            ysh = np.concatenate([labels[k] for k in pt])
                            mdl = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                                LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, class_weight='balanced', random_state=seed))
                            mdl.fit(F_list[nm][rows_tr], ysh); p = np.zeros(o[-1]); p[allrows] = mdl.predict_proba(F_list[nm][allrows])[:, 1]
                            vv2, grid2 = grids(p, rows_tr, True)[gname]; sc2 = [vv2[s] for s in sl]; lab_cal = labels
                            au2 = roc_auc_score(yte, np.concatenate([p[sl[k]] for k in te]))
                        else:
                            pc = permute_equal_length(nst_s, ca, rng); lab_cal = list(labels)
                            for a_, b_ in zip(ca, pc): lab_cal[a_] = labels[b_]
                            sc2, grid2, au2 = sc, grid, au
                        lam, feas = crop_lambda(Prep(sc2, lab_cal, ca), grid2, .05); r = crop_eval(Prep(sc2, labels, te), lam)
                        rows_out.append(dict(prm=tag, subset=sname, control='C5 label shuffle', proxy=nm, grid=gname, alpha=.05, seed=seed,
                                             auroc=au2, lam=float(lam), feasible=feas, impossible=False, **r))
            if sname == 'C1':                                   # C5 order shuffle: permute steps within each trace
                rng = np.random.default_rng(seed + 2000)
                perm = [rng.permutation(s.stop - s.start) for s in sl]
                lab_p = [labels[k][perm[k]] for k in range(len(sl))]
                rows_p = np.concatenate([np.arange(sl[k].start, sl[k].stop)[perm[k]] for k in range(len(sl))])
                remap = np.zeros(o[-1], int); remap[allrows] = rows_p        # new position -> original step row
                ytr_p = np.concatenate([lab_p[k] for k in tr]); yte_p = np.concatenate([lab_p[k] for k in te])
                rows_tr_p = np.concatenate([np.arange(sl[k].start, sl[k].stop) for k in tr])
                for nm, (v, raw_ok) in cand.items():
                    if 'supervised' in nm:
                        F = F_list[nm][remap].copy()
                        pos_cols = {'Position/length LR (supervised)': [0, 2, 4], 'C3 Format/position LR (supervised)': [10, 11],
                                    'C3 Format/position+PRM LR (supervised)': [10, 11]}.get(nm, [])
                        for k in range(len(sl)):                       # position features follow the NEW order
                            a, b = sl[k].start, sl[k].stop; n = b - a
                            if 0 in pos_cols: F[a:b, 0] = np.arange(1, n + 1)
                            if 2 in pos_cols: F[a:b, 2] = np.arange(1, n + 1) / n
                            if 4 in pos_cols: F[a:b, 4] = np.r_[0, np.cumsum(F[a:b, 3])[:-1]]
                            if 10 in pos_cols: F[a:b, 10] = (np.arange(n) + .5) / n
                            if 11 in pos_cols: F[a:b, 11] = np.arange(1, n + 1)
                        mdl = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                            LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, class_weight='balanced', random_state=seed))
                        mdl.fit(F[rows_tr_p], ytr_p); vp = np.zeros(o[-1]); vp[allrows] = mdl.predict_proba(F[allrows])[:, 1]
                    else:
                        vp = np.zeros(o[-1]); vp[allrows] = v[rows_p]
                    au = roc_auc_score(yte_p, np.concatenate([vp[sl[k]] for k in te]))
                    for gname, (vv, grid) in grids(vp, rows_tr_p, raw_ok).items():
                        if gname == 'fixed101_raw': continue
                        sc = [vv[s] for s in sl]
                        lam, feas = crop_lambda(Prep(sc, lab_p, ca), grid, .05); r = crop_eval(Prep(sc, lab_p, te), lam)
                        rows_out.append(dict(prm=tag, subset=sname, control='C5 order shuffle', proxy=nm, grid=gname, alpha=.05, seed=seed,
                                             auroc=au, lam=float(lam), feasible=feas, impossible=False, **r))
            print(tag, sname, 'seed', seed, f'{time.time() - t0:.0f}s', flush=True)
    keys = list(dict.fromkeys(k for r in rows_out for k in r))
    with open(out / f'CROP_CONTROLS_{tag}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows_out)
    par['seconds'] = time.time() - t0; par['labels_read'] = 'PRMBench error_steps (metric only; no label enters a label-free score)'
    (out / f'CROP_CONTROLS_PARITY_{tag}.json').write_text(json.dumps(par, indent=1))
    print('DONE', tag, json.dumps({k: v for k, v in par.items() if k != 'C1'}), flush=True)


if __name__ == '__main__':
    main()
