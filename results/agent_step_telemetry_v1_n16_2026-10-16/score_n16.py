"""Score Continuous L-SML and IU-PCR on the N16 bank of agent_step_telemetry_v1 (Qwen3-8B).

Fitting mirrors results/monday_cont_baselines_v1/run.py (fit_lsml / fit_iu, native sign primary,
_LEVELSIGN check) as re-run in thinkfix_rescore_v1/thinkfix_mask1/cont_baselines (commit 080877eb).
Bank construction mirrors monday_external_panel_v1/score.py build_bank (answer_standardize of the 13
plain channels, masked_std of the 3 digit channels with the digit-active mask).
Fitting uses ONLY the unlabeled feature matrices. Labels are opened afterwards, for evaluation only.

    python -I score_n16.py <out_dir> <labels_dir> <apb_ref_dir> <apb_judge_dir>
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

W = Path('/Users/osegev/.codex/worktrees/agent-step-telemetry-v1/hallucination_detection')
sys.path.insert(0, str(W))
from spectral_utils.fusion_utils import lsml_continuous  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402

OUT, LAB, APB_REF, APB_JUDGE = map(Path, sys.argv[1:5])
N16 = json.loads((W / 'results/monday_nested_banks_v2/BANKS.json').read_text())['N16']
DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
LEVEL = ('q15_H1', 'q15_VE1', 'logprob_margin', 'true_tail50', 'energy_level')   # monday_fusion_core.LEVEL
EPS = 1e-12
rng = np.random.default_rng(0)
B = 1000


# ---------------- verbatim helpers ----------------
def answer_standardize(values, offsets):   # lsml_gate_locator_research.answer_standardize
    x = np.asarray(values, dtype=np.float64); out = np.zeros_like(x, dtype=np.float64)
    for i in range(len(offsets) - 1):
        sl = slice(int(offsets[i]), int(offsets[i + 1])); block = x[sl]
        mean = block.mean(axis=0); sd = block.std(axis=0)
        out[sl] = np.divide(block - mean, sd, out=np.zeros_like(block), where=sd > EPS)
    return out


def masked_std(x, offsets, active):   # monday_external_panel_v1/score.py
    out = np.zeros(x.shape)
    for a, b in zip(offsets[:-1], offsets[1:]):
        for j in range(x.shape[1]):
            v = active[a:b, j]; y = x[a:b, j][v]
            if len(y) and y.std() > 1e-12: out[a:b, j][v] = (y - y.mean()) / y.std()
    return out


def fit_lsml(V):   # monday_cont_baselines_v1/run.py
    s, meta = lsml_continuous(*[V[:, j] for j in range(V.shape[1])])
    K = int(meta['K'])
    if K < 3: raise RuntimeError(f'L-SML selected K={K} groups: fewer than 3 groups')
    return np.asarray(s, float), {'K': K, 'degenerate': bool(meta['degenerate']), 'group_sizes': np.bincount(np.unique(meta['c'], return_inverse=True)[1]).tolist()}


def fit_iu(V):   # monday_cont_baselines_v1/run.py
    sd = V.std(0); keep = sd > 0
    if keep.sum() < 3: raise RuntimeError('fewer than 3 non-constant channels')
    F = ((V[:, keep] - V[:, keep].mean(0)) / sd[keep]).T
    res = upcr_fit(F, **IU_FIT_DEFAULTS)
    if not np.all(np.isfinite(res.w)): raise RuntimeError('non-finite IU-PCR weights')
    return np.asarray(res.w @ F, float), {'n_neg_weights': int((res.w < 0).sum()), 'm': int(keep.sum()), 'abstained': bool(getattr(res, 'abstained', False)),
                                         'weights': dict(zip(np.asarray(N16)[keep].tolist(), np.round(res.w, 4).tolist()))}


# ---------------- bank + fitting (label-free) ----------------
def build_V(X, nm, DA, off, norm):
    """norm='ansz': canonical per-answer z (build_bank). norm='cellz': same transforms over the whole fitted cell."""
    o = off if norm == 'ansz' else np.array([0, len(X)])
    plain = [c for c in N16 if c not in DIG]
    Vp = answer_standardize(X[:, [nm.index(c) for c in plain]], o)
    Vd = masked_std(X[:, [nm.index(c) for c in DIG]], o, DA[:, [nm.index(c) - nm.index(DIG[0]) for c in DIG]])
    V = np.zeros((len(X), len(N16)))
    for j, c in enumerate(N16): V[:, j] = Vd[:, DIG.index(c)] if c in DIG else Vp[:, plain.index(c)]
    level = answer_standardize(X[:, [nm.index(c) for c in LEVEL]], o).mean(1)
    return V, level


def fit_all(Fz, cells_of_answer):
    nm = [str(x) for x in Fz['names']]; X = Fz['values'].astype(float); DA = Fz['digit_active'].astype(bool)
    off = Fz['offsets'].astype(np.int64); S = int(off[-1])
    scores, diag = {}, {}
    for norm in ('ansz', 'cellz'):
        for scope, cell_ids in cells_of_answer.items():
            out = {k: np.full(S, np.nan) for k in ('LSML_CONT', 'IU_PCR', 'LSML_CONT_LEVELSIGN', 'IU_PCR_LEVELSIGN', 'ALL_EQ')}
            for ce in sorted(set(cell_ids)):
                ans = np.flatnonzero(cell_ids == ce)
                rows = np.concatenate([np.arange(off[i], off[i + 1]) for i in ans]); o = np.r_[0, np.cumsum(np.diff(off)[ans])]
                V, lev = build_V(X[rows], nm, DA[rows], o, norm)
                out['ALL_EQ'][rows] = V.mean(1)
                d0 = {'n_answers': int(len(ans)), 'n_steps': int(len(rows)), 'constant_channels': [N16[j] for j in range(16) if V[:, j].std() <= 0]}
                for name, fn in (('LSML_CONT', fit_lsml), ('IU_PCR', fit_iu)):
                    key = f'{norm}/{scope}/{ce}/{name}'
                    try:
                        s, d = fn(V)
                        if not np.all(np.isfinite(s)): raise RuntimeError('non-finite score')
                        r = np.corrcoef(s, lev)[0, 1] if lev.std() > 0 else np.nan
                        d['corr_with_level'] = float(r); out[name][rows] = s
                        out[name + '_LEVELSIGN'][rows] = s * ((np.sign(r) if np.isfinite(r) else 1.) or 1.)
                        diag[key] = {**d0, **d}
                    except Exception as exc:
                        diag[key] = {**d0, 'FAILED': f'{type(exc).__name__}: {exc}'[:300]}
            for k, v in out.items(): scores[f'{norm}/{scope}/{k}'] = v
    return scores, diag


# ---------------- metrics (definitions = agent_baselines.py) ----------------
def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0 or not np.isfinite(s).all(): return np.nan
    r = rankdata(s); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot(fn, n):
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [float(x) for x in np.nanpercentile(vals, [2.5, 97.5])]


def labels(b): return {d['uid']: d for d in map(json.loads, open(LAB / f'{b}.jsonl'))}


def per_answer(Fz, s):
    off = Fz['offsets']; keys = Fz['step_keys']
    return [(str(u), str(sub), keys[a:b], s[a:b]) for u, sub, a, b in zip(Fz['uids'], Fz['subsets'], off[:-1], off[1:])]


def tprm_metrics(Fz, s):
    L = labels('tprm'); pairs = {}
    for u, sub, k, ss in per_answer(Fz, s):
        lab = L[u]; pairs.setdefault(lab['pair_id'], {})[lab['action_chosen']] = (ss[-1], sub)
    pairs = [v for v in pairs.values() if True in v and False in v]
    if any(not np.isfinite(v[True][0]) or not np.isfinite(v[False][0]) for v in pairs): return None
    w = np.array([1.0 if v[True][0] < v[False][0] else 0.5 if v[True][0] == v[False][0] else 0.0 for v in pairs])
    sub = np.array([v[True][1] for v in pairs])
    return {'n_pairs': len(w), 'all': float(w.mean()), 'ci': boot(lambda i: w[i].mean(), len(w)),
            **{x: float(w[sub == x].mean()) for x in sorted(set(sub))}, 'tie_rate': float((w == 0.5).mean())}


def apb_metrics(Fz, s):
    L = labels('apb'); S, Y, SUB = [], [], []; hit, rnd, ty, tmax, tmean = [], [], [], [], []
    for u, sub, keys, ss in per_answer(Fz, s):
        lab = L[u]['step_labels']; trs = [(int(k), float(v), lab[str(int(k))]) for k, v in zip(keys, ss) if str(int(k)) in lab]
        for k, v, y in trs:
            if y != 0: S.append(v); Y.append(y == -1); SUB.append(sub)
        if not trs: continue
        errs = [k for k, v, y in trs if y == -1]
        if errs:
            hit.append(max(trs, key=lambda x: x[1])[0] == errs[0]); rnd.append(1 / len(trs))
        fl = L[u]['final_label']
        if fl in (1, -1):
            ty.append(fl == -1); tmax.append(max(x[1] for x in trs)); tmean.append(np.mean([x[1] for x in trs]))
    S = np.array(S); Y = np.array(Y); SUB = np.array(SUB); ty = np.array(ty)
    if not np.isfinite(S).all(): return None
    return {'n_err': int(Y.sum()), 'n_ok': int((~Y).sum()), 'step_auroc': auroc(Y, S), 'ci': boot(lambda i: auroc(Y[i], S[i]), len(Y)),
            **{f'step_auroc_{x}': auroc(Y[SUB == x], S[SUB == x]) for x in sorted(set(SUB))},
            'loc_n': len(hit), 'loc_acc': float(np.mean(hit)), 'loc_random': float(np.mean(rnd)),
            'traj_n_fail': int(ty.sum()), 'traj_n_ok': int((~ty).sum()), 'traj_auroc_max': auroc(ty, tmax), 'traj_auroc_mean': auroc(ty, tmean)}


def aft_metrics(Fz, s):
    L = labels('aft'); y, dom, src, amean, amax = [], [], [], [], []; hit, rnd = [], []
    for u, sub, keys, ss in per_answer(Fz, s):
        lab = L[u]; unsafe = lab['label'] != 'safe'
        y.append(unsafe); dom.append(lab['domain']); src.append(lab['unsafe_source'] if unsafe else 'safe')
        amean.append(float(np.mean(ss))); amax.append(float(np.max(ss)))
        kl = [int(k) for k in keys]
        if unsafe and lab['mistake_step'] in kl:
            rnd.append(1 / len(kl)); hit.append(kl[int(np.argmax(ss))] == lab['mistake_step'])
    y = np.array(y); dom = np.array(dom); src = np.array(src)
    out = {'n_unsafe': int(y.sum()), 'n_safe': int((~y).sum()), 'loc_n': len(hit), 'loc_acc': float(np.mean(hit)), 'loc_random': float(np.mean(rnd))}
    for agg, v in (('mean', np.array(amean)), ('max', np.array(amax))):
        if not np.isfinite(v).all(): return None
        out[f'{agg}_all'] = auroc(y, v); out[f'{agg}_ci'] = boot(lambda i: auroc(y[i], v[i]), len(y))
        for d in sorted(set(dom)): out[f'{agg}_{d}'] = auroc(y[dom == d], v[dom == d])
        for s_ in ('injected', 'diagnosed'):
            m = (src == s_) | (src == 'safe')
            # pooled over domains, each unsafe-source split vs ALL safe runs, and vs the safe runs of the same domains
            out[f'{agg}_{s_}_vs_allsafe'] = auroc(src[m] == s_, v[m])
            md = m & np.isin(dom, np.unique(dom[src == s_]))
            out[f'{agg}_{s_}'] = auroc(src[md] == s_, v[md]); out[f'{agg}_{s_}_ci'] = boot(lambda i, md=md: auroc((src[md] == s_)[i], v[md][i]), int(md.sum()))
            for d in sorted(set(dom)):
                mm = m & (dom == d)
                if ((src == s_) & (dom == d)).any(): out[f'{agg}_{d}_{s_}'] = auroc(src[mm] == s_, v[mm])
    return out


# ---------------- APB same-backbone LLM-judge reference ----------------
OFFS = {'hotpotqa': 0, 'gaia_dev': 250, 'bfcl': 500, 'tau2': 750}


def first_neg1(sl):
    idx = [int(k) for k, v in sl.items() if v == -1]; return min(idx) if idx else -1


def judge_reference(our_uids):
    ref, pred = {}, {}
    for d, o in OFFS.items():
        for r in map(json.loads, open(APB_REF / f'{d}.jsonl')):
            ref[f'apb:{d}:{r["total_index"]}'] = {str(k): int(v) for k, v in r['step_labels'].items()}
        for r in map(json.loads, open(APB_JUDGE / f'{d}__blind_Qwen3-8B-Non-Thinking.jsonl')):   # one line per record (250 each)
            pred[f'apb:{d}:{o + r["query_index"] * 5 + r["sample_index"]}'] = {str(k): int(v) for k, v in (r.get('step_labels') or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}   # compare.py _normalize_step_labels drops None
    L = labels('apb')
    agree = sum(ref[u] == {str(k): int(v) for k, v in L[u]['step_labels'].items()} for u in L)
    def acc(keys): return float(np.mean([first_neg1(pred.get(k, {})) == first_neg1(ref[k]) for k in keys]))
    ours = [u for u in our_uids if u in ref]
    err = [u for u in ours if first_neg1(ref[u]) >= 0]
    # same-backbone step AUROC: judge label as a 3-level score (-1 -> 1 = error, 0 -> 0.5, +1 -> 0)
    yy, ss = [], []
    for u in ours:
        for k, v in ref[u].items():
            if v != 0: yy.append(v == -1); ss.append({-1: 1., 0: .5, 1: 0.}.get(pred.get(u, {}).get(k), .5))
    return {'n_ref': len(ref), 'evaluator_labels_equal_reference': f'{agree}/{len(L)}',
            'first_neg1_idx_acc_official_all1000': acc(list(ref)), 'first_neg1_idx_acc_our_items': acc(ours), 'n_our_items': len(ours),
            'first_error_hit_on_error_trajs': acc(err), 'n_error_trajs': len(err),
            'per_subset_official': {d: acc([k for k in ref if k.split(':')[1] == d]) for d in OFFS},
            'judge_step_auroc_3level': auroc(np.array(yy), np.array(ss)), 'missing_judge_records': sum(k not in pred for k in ref)}


def main():
    t0 = time.time(); res = {}; alld = {}
    for b in ('tprm', 'apb', 'aft'):
        Fz = dict(np.load(OUT / f'FEATURES_{b}.npz'))
        subsets = Fz['subsets'].astype(str)
        scopes = {'bench': np.zeros(len(subsets), int), 'subset': np.unique(subsets, return_inverse=True)[1]}
        scores, diag = fit_all(Fz, scopes)           # <- label-free; labels not yet opened
        np.savez_compressed(OUT / f'SCORES_{b}.npz', offsets=Fz['offsets'], uids=Fz['uids'], step_keys=Fz['step_keys'],
                            **{k.replace('/', '__'): v for k, v in scores.items()})
        alld[b] = diag
        fn = {'tprm': tprm_metrics, 'apb': apb_metrics, 'aft': aft_metrics}[b]
        res[b] = {k: fn(Fz, v) for k, v in scores.items()}
        print(b, 'done', round(time.time() - t0), flush=True)
    res['apb_judge_qwen3_8b'] = judge_reference([str(u) for u in np.load(OUT / 'FEATURES_apb.npz')['uids']])
    (OUT / 'FIT_DIAGNOSTICS.json').write_text(json.dumps(alld, indent=1, default=float))
    (OUT / 'METRICS.json').write_text(json.dumps(res, indent=1, default=float))
    print('DONE', round(time.time() - t0))


if __name__ == '__main__':
    main()
