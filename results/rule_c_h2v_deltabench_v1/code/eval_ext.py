"""The ONE labelled evaluation of rule C on Hard2Verify / DeltaBench (2026-11-22). Reads the frozen label-free outputs of
pseudo_ext.py (RULEC_<bench>.npz, hash printed), then the gold labels. Run once per benchmark.

Hard2Verify: official step Balanced F1 (evaluate_collection.official path, x100) and the PRMScore form
  (50 * (F1 valid + F1 error), pooled, keep = include); population 'own' (the 181 answers with telemetry) and 'common'
  (the 177 answers scored by all 5 PRMs). PRM arms: stored PRM-mode scores (hd_collection_data/h2v/SCORES_<prm>.npz),
  native rule (PRM's own valid label) and GAP.
DeltaBench: official section macro-F1 (deltabench_official; target = unuseful + error sections; also error-only) and the
  old Macro-F1 for continuity is not recomputed here (evaluate_cab.py numbers stand). Arms: rule C flags, method of record
  GAP flags, method of record s0 under mu-sigma (s0 > mu + sigma), PRMs under mu-sigma and GAP.
CI: paired group bootstrap over question groups (GROUPS_<bench>.json), 2,000 draws, seed 20261122.
    HD_ROOT=$WS/hd COLLECTION_DATA=<hd_collection_data> python eval_ext.py BENCH RULEC_DIR GROUPS_JSON OUT_JSON [DELTABENCH_JSONL SCORES_DIR]
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
HD = Path(os.environ['HD_ROOT'])
sys.path.insert(0, str(HD / 'results/ssl_one_experiment_v1')); sys.path.insert(0, str(HD / 'results/prm_boost_collection_v1'))
sys.path.insert(0, str(Path(__file__).parent))
import ssl_lib as L  # noqa: E402
import deltabench_official as O  # noqa: E402

bench, rdir, gpath, outp = sys.argv[1:5]
Rp = Path(rdir) / f'RULEC_{bench}.npz'
print('RULEC sha256', hashlib.sha256(open(Rp, 'rb').read()).hexdigest())
Z = np.load(Rp); uids = Z['uids'].astype(str); off = Z['offsets']; G = json.load(open(gpath))
rng = np.random.default_rng(20261122); DRAWS = 2000
res = dict(bench=bench, rulec_sha256=hashlib.sha256(open(Rp, 'rb').read()).hexdigest())


def boot_idx(groups):
    u, inv = np.unique(groups, return_inverse=True); members = [np.flatnonzero(inv == g) for g in range(len(u))]
    return [np.concatenate([members[g] for g in rng.integers(0, len(u), len(u))]) for _ in range(DRAWS)]


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


if bench == 'hard2verify':
    import evaluate_collection as E  # noqa: E402
    from collection_parity import h2v_common_uids  # noqa: E402

    def cell(prm, keep_uids):
        c = E.load_external('hard2verify', prm, uids_keep=keep_uids)
        return c

    def counts(c, f, native=False):
        return E.counts_by_answer(c, f, native)

    def metrics(C):
        bal = 100 * E.official_from_counts(C.sum(0), 'hard2verify')
        prms = float(L.official_from_counts(C.sum(0)))
        return float(bal), prms

    pos = {u: j for j, u in enumerate(uids)}
    for pop, keep in (('own', set(uids)), ('common', h2v_common_uids())):
        c = cell('qwen_prm_7b', keep & set(uids))
        rows = np.concatenate([np.arange(off[pos[u]], off[pos[u] + 1]) for u in c.ids]); assert len(rows) == c.off[-1]
        arms = {'rule_C': Z['flagsC'][rows], 'MoR_GAP': Z['f0'][rows]}
        Cn = {k: counts(c, v) for k, v in arms.items()}
        for prm in E.PRMS:
            cp = cell(prm, set(c.ids))
            if list(cp.ids) != list(c.ids): continue          # PRM did not score the same answers
            for rule in ('native', 'GAP'):
                fl = cp.native_flag if rule == 'native' else E.topk_flags(E.answer_z(cp.arms['PRM'], cp.off), cp.off, E.gap_counts(E.answer_z(cp.arms['PRM'], cp.off), cp.off))
                Cn[f'PRM_{prm}_{rule}'] = counts(cp, fl, native=(rule == 'native'))
        grp = np.array([G[u] for u in c.ids]); B = boot_idx(grp)
        R = {}
        for k, C in Cn.items():
            bal, prms = metrics(C)
            bb = np.array([metrics(C[b]) for b in B])
            R[k] = dict(balanced_f1=bal, balanced_f1_ci=ci(bb[:, 0]), prmscore=prms, prmscore_ci=ci(bb[:, 1]),
                        zero_flag_share=None)
        for k in ('rule_C', 'MoR_GAP'):
            R[k]['zero_flag_share'] = float(np.mean(np.add.reduceat(arms[k].astype(int), c.off[:-1]) == 0))
        diff = {}
        for ref in [k for k in Cn if k != 'rule_C']:
            d = np.array([np.subtract(metrics(Cn['rule_C'][b]), metrics(Cn[ref][b])) for b in B])
            p = np.subtract(metrics(Cn['rule_C']), metrics(Cn[ref]))
            diff[f'C_minus_{ref}'] = dict(balanced_f1=float(p[0]), balanced_f1_ci=ci(d[:, 0]), prmscore=float(p[1]), prmscore_ci=ci(d[:, 1]))
        res[pop] = dict(n_answers=int(len(c.ids)), n_groups=int(len(set(grp))), arms=R, diffs=diff)
else:
    jsonl, scdir = sys.argv[5], Path(sys.argv[6])
    T = O.official_targets(jsonl); TE = O.official_targets(jsonl, error_only=True)
    s0 = Z['s0']
    arms = {'rule_C': Z['flagsC'], 'MoR_GAP': Z['f0'], 'MoR_s0_musigma': O.musigma_flags(s0, True)}
    for prm, cl in (('qwen_prm_7b', 'qwen_prm_deltabench_ext'), ('qwen_prm_72b', 'qwen_prm_72b_deltabench_ext'),
                    ('pure_prm', 'pure_prm_deltabench_ext'), ('rlhflow_prm', 'rlhflow_prm_deltabench')):
        P = np.load(scdir / f'SCORES_prm_{cl}.npz'); assert (P['uids'].astype(str) == uids).all()
        r = 1. - P['score__PRMPROB'].astype(float)
        arms[f'PRM_{prm}_musigma'] = O.musigma_flags(r, False); arms[f'PRM_{prm}_GAP'] = L.gap_flags(-r, off)
    grp = np.array([G[u] for u in uids]); B = boot_idx(grp)
    for tname, TT in (('union', T), ('error_only', TE)):
        tg = [TT[u] for u in uids]; M = {k: O.per_sample(O.sets_from_flags(v, off), tg) for k, v in arms.items()}
        R = {}
        for k, m in M.items():
            f1 = m['F1']; R[k] = dict(f1_macro=100 * float(np.nanmean(f1)), f1_macro_ci=ci([100 * np.nanmean(f1[b]) for b in B]),
                                      f1_macro_empty1=100 * float(m['F1_empty1'].mean()), **{q: v for q, v in O.summarize(m).items() if q != 'f1_macro'})
            R[k]['zero_flag_share'] = float(np.mean(np.add.reduceat(arms[k].astype(int), off[:-1]) == 0))
        diff = {}
        for ref in [k for k in M if k != 'rule_C']:
            a, b_ = M['rule_C']['F1'], M[ref]['F1']
            diff[f'C_minus_{ref}'] = dict(f1_macro=100 * float(np.nanmean(a) - np.nanmean(b_)),
                                          ci=ci([100 * (np.nanmean(a[b]) - np.nanmean(b_[b])) for b in B]))
        res[tname] = dict(n_answers=int(len(uids)), n_groups=int(len(set(grp))), arms=R, diffs=diff)
json.dump(res, open(outp, 'w'), indent=1, default=float)
print(json.dumps(res, default=float)[:3000])
