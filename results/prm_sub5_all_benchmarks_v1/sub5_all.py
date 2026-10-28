"""SUB5 (the 'JOINT 8 cells (no Math-Shepherd)' subset of results/prm_view_selection_v1) on every benchmark with data.

SUB5 = q15_H1 + chosen_surprisal + top15_turnover + dominant_freq16 + bocpd_p0 + digit_spread (6 of the 16 N16 channels).
  'SUB5 alone' = stage2.TeleCell.evaluate (telemetry only: stored frozen-bank orientation, per-answer GAP votes,
                 L-SML groups K>=3, DS-EM between groups, weights applied to the answer-z continuous values)
  'SUB5 +PRM'  = stage1.CellData.evaluate (the PRM as one more channel and the DS-EM anchor); backbone cells only
Both classes are used unchanged; only E.load / E.channels / stage1.prm_arm are redirected to the new cells (as
h2v_holdout/h2v_sub4_sub5.py redirects E.load to the Hard2Verify common set). Metric code is re-used unchanged from
evaluate_collection.py, competitors.py (PRISM same-TPR, CROP on PRMBench), and the vendored copies of evaluate_cab.py /
crop_cab.py (branch claude/prm-boost-missing-benchmarks-2026-10-26 @ 109bf203c) and eval_processbench.py / pb_metrics.py
(claude/integration-2026-10-17 @ 46b3a6318). No labels enter any fit, except the 'include' pool mask that the existing
stage1/stage2 code reads (PRMBench: non-control answers; Socratic/H2V/CRV/PRM800K/DeltaBench: every step). ProcessBench's
'include' drops the steps after the first error, so for ProcessBench the fit pool is set to every step instead.

PRM scale inside 'SUB5 +PRM' and for the 'PRM' (GAP) row: PRMBench / Socratic / Hard2Verify keep STATUS_META's scale (as
the existing tables); every other benchmark takes the PRMBench scale of the same PRM (fixed before looking at them).

    python sub5_all.py            # -> SUB5_ALL.csv, PARITY.json  (stops with exit 3 if a parity gate fails)
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '4')
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT.parent / 'hd_collection_data'
os.environ.setdefault('CAB_DATA', str(DATA / 'crop_ares_v1'))
VS = ROOT / 'results/prm_view_selection_v1'
sys.path.insert(0, str(ROOT / 'results/prm_boost_collection_v1'))
sys.path.insert(0, str(VS / 'stage1')); sys.path.insert(0, str(VS / 'stage2'))
import stage1 as S1  # noqa: E402
import stage2 as S2  # noqa: E402
import competitors as CMP  # noqa: E402
E = S1.E
sys.path.insert(0, str(HERE / 'vendor'))
import evaluate_cab as CAB  # noqa: E402
import crop_cab as C6  # noqa: E402
import eval_processbench as EP  # noqa: E402
from collection_parity import h2v_common_uids  # noqa: E402

NAMES = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation',
         'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0', 'realized_z', 'realized_drv', 'digit_alternative',
         'digit_spread', 'digit_alternative_innovation']
SUB5_NAMES = 'q15_H1 + chosen_surprisal + top15_turnover + dominant_freq16 + bocpd_p0 + digit_spread'.split(' + ')
SUB5 = [NAMES.index(n) for n in SUB5_NAMES]
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']          # Math-Shepherd PRM excluded
PFILE = {'qwen_prm_7b': 'qwen_prm'}
CAB_DIRS = [DATA / 'crop_ares_v1', DATA / 'sub5_all_v1']                    # sub5_all_v1: ProcessBench prm mode
CHAN_DIRS = [DATA / 'crop_ares_v1' / 'channels', DATA / 'sub5_all_v1' / 'scores']
BACKBONE = ['prmbench', 'socratic', 'hard2verify', 'processbench', 'crv_arith', 'crv_gsm8k', 'prm800k', 'deltabench']
IN_SAMPLE = {'prmbench', 'socratic'}          # SUB5 was selected on the backbone cells of these two
COLLECTION = {'prmbench', 'socratic', 'hard2verify'}   # backbone cells live in the evaluate_collection data
STANDALONE = ['prmbench', 'socratic', 'hard2verify', 'processbench', 'crv_arith', 'crv_gsm8k', 'prm800k', 'deltabench']
CROP_BENCHES = {'crv_arith', 'crv_gsm8k', 'prm800k'}
TOL = 0.01
PB_GOLD = {g['uid']: g for g in json.loads((ROOT / 'results/prm_boost_processbench_v1/local/processbench_gold.json').read_text(encoding='utf8'))}
META = S1.META


# ------------------------------------------------------------------ redirect the loaders used by stage1 / stage2
REG = {}                       # bench key 'cab|<mode>|<bench>' -> dict(mode, bench, common)
_load, _channels, _prm_arm = E.load, E.channels, S1.prm_arm
_H2V = h2v_common_uids()


def cab_dir(mode, prm, bench):
    for d in CAB_DIRS:
        if (d / 'scores' / f'SCORES_{mode}_{PFILE.get(prm, prm)}_{bench}.npz').exists():
            return d
    return None


def load(bench, prm, **kw):
    if bench in REG:
        r = REG[bench]; CAB.D = cab_dir(r['mode'], prm, r['bench'])
        c = CAB.load(r['mode'], PFILE.get(prm, prm), r['bench'], r['common'])
        c.src = (r['mode'], PFILE.get(prm, prm), r['bench'])
        if r['bench'] == 'processbench':
            c.keep_gold = c.keep; c.keep = np.ones(len(c.keep), bool)      # label-free fit pool (see docstring)
        return c
    if bench == 'prmbench' and prm.endswith('__standalone'):
        return E.load_prmbench(prm, 'prmbench_standalone')
    if bench == 'hard2verify':
        kw = dict(kw, uids_keep=_H2V)
    return _load(bench, prm, **kw)


def channels(c, bank, variant=None):
    if getattr(c, 'src', None):
        mode, pf, bench = c.src
        p = next(d / f'SCORES_{mode}_{pf}_{bench}_CHANNELS.npz' for d in CHAN_DIRS
                 if (d / f'SCORES_{mode}_{pf}_{bench}_CHANNELS.npz').exists())
        Z = np.load(p); assert list(Z[f'channel_names__{bank}']) == NAMES, p
        uids = Z['uids'].astype(str); off = Z['offsets'].astype(np.int64); pos = {u: j for j, u in enumerate(uids)}
        M = Z[f'channels__{bank}']                       # read once (NpzFile re-reads the member on every access)
        X = np.concatenate([M[off[pos[u]]:off[pos[u] + 1]] for u in c.ids]).astype(float)
        assert len(X) == c.off[-1], (p, len(X), c.off[-1])
        return X
    if c.bench == 'prmbench' and c.prm.endswith('__standalone'):
        return _channels(c, bank, variant='prmbench_standalone')
    return _channels(c, bank, variant)


def prm_scale_arm(bench, prm):
    """'PRMPROB'/'PRM' (external layout) on the predeclared scale."""
    b = bench if bench in ('prmbench', 'socratic', 'hard2verify') else 'prmbench'
    prob = META[f'{b}/{prm}']['prm_reference_scale'] == '1-reward'
    return 'PRMPROB' if prob else 'PRM'


def prm_arm(bench, prm):
    if bench in REG:
        return prm_scale_arm(REG[bench]['bench'], prm)
    return _prm_arm(bench, prm.replace('__standalone', ''))


E.load, E.channels, S1.prm_arm = load, channels, prm_arm


def common_uids(bench):
    """evaluate_cab's 'common' population (all cells of the benchmark, Math-Shepherd included), as in CAB_EVAL_ALL."""
    CAB.D = DATA / 'crop_ares_v1'          # ProcessBench: CAB's cells are the 5 standalone ones (all 3,400 answers)
    files = [CAB.D / 'scores' / f'SCORES_{m}_{p}_{bench}.npz' for m, p in CAB.cells(bench)]
    return set.intersection(*[set(np.load(f)['uids'].astype(str)) for f in files])


# ------------------------------------------------------------------ SUB5 scores for one cell
def sub5_scores(key, prm, with_prm):
    S2.PRM_ALONE.setdefault(f'{key}/{prm}', {'prm_exact': float('nan')})
    t = S2.TeleCell().build(key, prm); c = t.c
    o = t.evaluate(SUB5, want_detail=True); out = {'SUB5 alone': o['fused'] if o['valid'] else None}
    out['SUB5 mean'] = t.Z[:, SUB5].mean(1)        # the 'average' control: same 6 channels, equal weights, no L-SML
    info = {'alone_K': int(o['K']), 'alone_valid': bool(o['valid'])}
    if with_prm:
        s1 = S1.CellData(key, prm); assert np.array_equal(s1.c.off, c.off)
        o1 = s1.evaluate(SUB5, want_detail=True); out['SUB5 +PRM'] = o1['fused'] if o1['valid'] else None
        out['SUB5 +PRM mean'] = s1.Z[:, np.r_[0, np.asarray(SUB5) + 1]].mean(1)   # PRM + the 6 channels, equal weights
        info.update(prm_K=int(o1['K']), prm_valid=bool(o1['valid']), prm_share=o1.get('w_prm_share'), prm_arm=s1.arm)
    return c, out, info


def gapf(c, s):
    return E.topk_flags(E.answer_z(s, c.off), c.off, E.gap_counts(E.answer_z(s, c.off), c.off))


# ------------------------------------------------------------------ metrics per benchmark family
def rows_cab(c, mode, bench, prm, arms):
    """Socratic-form benchmarks and Hard2Verify: evaluate_cab.row (common population)."""
    out = []
    for label, arm, rule in arms:
        r = CAB.row(c, mode, bench, 'common', arm, rule)
        mname = 'balanced_f1_GAP' if bench == 'hard2verify' else 'macro_f1_GAP'
        if rule == 'native': mname = mname.replace('GAP', 'native')
        out += [dict(metric=mname, value=r['macro_f1']), dict(metric='within_auc', value=r['within_auc']),
                dict(metric='pooled_auroc', value=r['pooled_auroc'])]
        for x in out[-3:]: x.update(arm=label)
    return out


def rows_crop(c, arms):
    """crop_cab.py protocol (q201 grid, 10 seeds, alpha 0.05) on the cell's answers."""
    labels = [c.err[a:b] for a, b in zip(c.off[:-1], c.off[1:])]; has_err = np.array([y.any() for y in labels]); out = []
    from sklearn.metrics import roc_auc_score
    for label, arm, _ in arms:
        if arm == 'PRM' and label == 'PRM native': continue
        s = c.arms[arm]; per = [s[a:b] for a, b in zip(c.off[:-1], c.off[1:])]; res = []
        for seed in C6.SEEDS:
            tr, ca, te = C6.split_traces(has_err, seed)
            grid = np.unique(np.quantile(np.concatenate([per[i] for i in tr]), np.linspace(0, 1, 201)))
            r = C6.crop(per, labels, ca, te, grid)
            r['auroc'] = float(roc_auc_score(np.concatenate([labels[i] for i in te]), np.concatenate([per[i] for i in te])))
            res.append(r)
        out += [dict(arm=label, metric='crop_kept_a05', value=float(np.mean([r['kept'] for r in res]))),
                dict(arm=label, metric='crop_auroc', value=float(np.mean([r['auroc'] for r in res])))]
    return out


def rows_pb(c, arms):
    """eval_processbench.evaluate on the cell (NC rule avg F1 and the prefix-domain within-AUC)."""
    mode, pf, bench = c.src
    S = np.load(cab_dir(mode, pf, bench) / 'scores' / f'SCORES_{mode}_{pf}_{bench}.npz')
    lay = {k: S[k] for k in ('uids', 'nonempty', 'nonempty_lengths', 'native_valid')}
    assert list(lay['uids'].astype(str)) == list(c.ids) and c.off[-1] == lay['nonempty'].sum()
    A = {arm: c.arms[arm] for _, arm, _ in arms}; A['PRM'] = c.arms['PRM']
    R, _, _ = EP.evaluate(lay, A, PB_GOLD, 'scored', PB_GOLD, False)
    R = {r['arm']: r for r in R}; out = []
    for label, arm, rule in arms:
        name = 'PRM_native' if label == 'PRM native' else f'{arm}_NC'
        out += [dict(arm=label, metric='avg_f1_NC' if label != 'PRM native' else 'avg_f1_native', value=R[name]['avg_f1']),
                dict(arm=label, metric='within_auc', value=R[name]['within_auc'])]
    return out


def rows_prmbench(c, arms):
    """PRMBench: PRMScore GAP and same-TPR (87.3) via competitors.prism_rows, within-AUC, CROP via competitors.crop_rows."""
    out = []
    P = {r['arm']: r for r in CMP.prism_rows(c, [a for _, a, _ in arms if a != 'PRM_native'])}
    Cr = {r['arm']: r for r in CMP.crop_rows(c, [a for _, a, _ in arms if a not in ('PRM_Q80', 'PRM_Q80_PROB')])}
    for label, arm, rule in arms:
        if arm == 'PRM_native':
            out.append(dict(arm=label, metric='PRMScore_native', value=E.official(c, c.native_flag)[0]))
            out += [dict(arm='PRM (1-reward)', metric='crop_kept_a05', value=Cr[arm]['kept_alpha05']),
                    dict(arm='PRM (1-reward)', metric='crop_auroc', value=Cr[arm]['auroc'])]
            continue
        out += [dict(arm=label, metric='PRMScore_GAP', value=P[arm]['PRMScore_GAP']),
                dict(arm=label, metric='PRMScore_sameTPR87.3', value=P[arm]['matchedTPR_PRMScore']),
                dict(arm=label, metric='within_auc', value=float(np.nanmean(E.within(c, arm))))]
        if arm in Cr:
            out += [dict(arm=label, metric='crop_kept_a05', value=Cr[arm]['kept_alpha05']),
                    dict(arm=label, metric='crop_auroc', value=Cr[arm]['auroc'])]
    return out


# ------------------------------------------------------------------ parity references
def refs():
    cab = E.read_csv(HERE / 'vendor/CAB_EVAL_ALL.csv')
    cab = {(r['benchmark'], r['mode'], r['prm'], r['arm'], r['rule']): r for r in cab if r['population'] == 'common'}
    pb = {(r['model'], r['arm']): r for r in E.read_csv(HERE / 'vendor/RESULTS_processbench.csv') if r['population'] == 'scored'}
    sa = {(r['prm'], r['arm']): r for r in E.read_csv(ROOT / 'results/prm_standalone_telemetry_v1/tables/STANDALONE_MAIN.csv')}
    return cab, pb, sa


def main(jobs=None, do_gates=True, part=None):
    cab_ref, pb_ref, sa_ref = refs(); rows, parity, info_all, fails = [], [], {}, []

    def gate(name, got, ref, tol=TOL):
        d = abs(float(got) - float(ref)); ok = d <= tol
        parity.append(dict(check=name, got=float(got), ref=float(ref), abs_diff=d, tol=tol, ok=ok))
        print(('PASS ' if ok else 'FAIL ') + name, round(float(got), 4), round(float(ref), 4), flush=True)
        if not ok: fails.append(name)

    # Gate 2: reproduce SUB5 on PRMBench Qwen-7B (alone 68.78, +PRM 69.90) and Hard2Verify Qwen-7B (alone 49.35)
    if do_gates:
        gate2(gate)
    if fails:
        json.dump(dict(parity=parity, failed=fails), open(HERE / 'PARITY.json', 'w'), indent=1); raise SystemExit(3)
    return run_jobs(jobs, gate, rows, parity, info_all, fails, part, cab_ref, pb_ref, sa_ref)


def gate2(gate):
    c, sc, _ = sub5_scores('prmbench', 'qwen_prm_7b', True)
    for k, v in sc.items(): c.arms[k] = v
    P = {r['arm']: r for r in CMP.prism_rows(c, ['SUB5 alone', 'SUB5 +PRM'])}
    ref2 = {r['arm']: r for r in E.read_csv(VS / 'stage2/COMPETITORS.csv') if r['protocol'] == 'PRISM' and r['prm'] == 'qwen_prm_7b'}
    gate('gate2 PRMBench qwen_prm_7b SUB5 alone PRMScore GAP', P['SUB5 alone']['PRMScore_GAP'], ref2['SUB5 alone']['PRMScore_GAP'])
    gate('gate2 PRMBench qwen_prm_7b SUB5 +PRM PRMScore GAP', P['SUB5 +PRM']['PRMScore_GAP'], ref2['SUB5 +PRM']['PRMScore_GAP'])
    del c
    c, sc, _ = sub5_scores('hard2verify', 'qwen_prm_7b', False)
    ref3 = {(r['set'], r['version']): r for r in E.read_csv(VS / 'h2v_holdout/H2V_SUB4_SUB5.csv') if r['prm'] == 'qwen_prm_7b'}
    gate('gate2 Hard2Verify qwen_prm_7b SUB5 alone balanced F1 GAP', 100 * float(E.official(c, gapf(c, sc['SUB5 alone']))[0]),
         ref3[('SUB5', 'alone')]['fused_ps'])
    del c


def gate_collection(gate, tag, bench, prm, R):
    """Backbone PRMBench / Socratic / Hard2Verify: TELE_N16 vs STATUS_MAIN, SUB5 vs the view-selection outputs."""
    mn = {'prmbench': 'PRMScore_GAP', 'socratic': 'macro_f1_GAP', 'hard2verify': 'balanced_f1_GAP'}[bench]
    get = lambda arm, m=mn: next(r['value'] for r in R if r['arm'] == arm and r['metric'] == m)
    st = S1.STATUS[(bench, prm, 'TELE_N16')]
    gate(f'gate1 {tag} TELE_N16 {mn}', get('TELE_N16'), st['GAP'])
    gate(f'gate1 {tag} TELE_N16 within-AUC', get('TELE_N16', 'within_auc'), st['within_auc'], 1e-3)
    if bench in IN_SAMPLE:      # stage2/NAMED_EXACT.csv: s2_fused / s2_mean / s1_fused of SUB5 on this cell
        ne = next(r for r in E.read_csv(VS / 'stage2/NAMED_EXACT.csv') if r['subset'] == ' + '.join(SUB5_NAMES))
        k = f'{bench}/{prm}'
        gate(f'gate2 {tag} SUB5 alone', get('SUB5 alone'), ne[f's2_fused {k}'])
        gate(f'gate2 {tag} SUB5 mean', get('SUB5 mean'), ne[f's2_mean {k}'])
        gate(f'gate2 {tag} SUB5 +PRM', get('SUB5 +PRM'), ne[f's1_fused {k}'])
    else:                       # h2v_holdout/H2V_SUB4_SUB5.csv
        h = {r['version']: r for r in E.read_csv(VS / 'h2v_holdout/H2V_SUB4_SUB5.csv') if r['prm'] == prm and r['set'] == 'SUB5'}
        gate(f'gate2 {tag} SUB5 alone', get('SUB5 alone'), h['alone']['fused_ps'])
        gate(f'gate2 {tag} SUB5 mean', get('SUB5 mean'), h['alone']['mean_ps'])
        gate(f'gate2 {tag} SUB5 +PRM', get('SUB5 +PRM'), h['+PRM']['fused_ps'])
        gate(f'gate2 {tag} SUB5 +PRM mean', get('SUB5 +PRM mean'), h['+PRM']['mean_ps'])


def all_jobs():
    return [(m, b, p) for m, bs in (('prm', BACKBONE), ('sa', STANDALONE)) for b in bs for p in PRMS]


def run_jobs(jobs, gate, rows, parity, info_all, fails, part, cab_ref, pb_ref, sa_ref):
    skipped = []
    for mode, bench, prm in (all_jobs() if jobs is None else jobs):
        if True:
            tag = f'{bench}/{mode}/{prm}'
            if mode == 'prm' and bench in COLLECTION:
                key, pname = bench, prm
            elif bench == 'prmbench':
                key, pname = 'prmbench', f'{prm}__standalone'
                if not (DATA / 'prmbench_standalone' / f'ABL5_CHANNELS_{pname}.npz').exists():
                    skipped.append(dict(cell=tag, reason='no standalone PRMBench channel file')); continue
            else:
                if cab_dir(mode, prm, bench) is None or not any((d / f'SCORES_{mode}_{PFILE.get(prm, prm)}_{bench}_CHANNELS.npz').exists() for d in CHAN_DIRS):
                    skipped.append(dict(cell=tag, reason='no SCORES or CHANNELS file')); continue
                key, pname = f'cab|{mode}|{bench}', prm
                REG[key] = dict(mode=mode, bench=bench, common=common_uids(bench))
            c, sc, info = sub5_scores(key, pname, mode == 'prm')
            info_all[tag] = dict(info, n_answers=int(len(c.ids)))
            if not hasattr(c, 'n_gold'): c.n_gold = len(c.ids)
            # channel alignment: MEAN_N16 must be the plain mean of the loaded channels (float32 storage)
            X = E.channels(c, 'N16'); dmean = float(np.max(np.abs(X.mean(1) - c.arms['MEAN_N16'])))
            info_all[tag]['mean_n16_alignment_maxdiff'] = dmean
            if dmean > 1e-4: fails.append(f'{tag} channel alignment {dmean}')
            for k, v in sc.items():
                if v is not None: c.arms[k] = v
                else: skipped.append(dict(cell=f'{tag} {k}', reason='K < 3 or all between-group weights clipped'))
            # arm list: (label, arm in c.arms, rule)
            if bench == 'prmbench':
                scale = 'PRM_Q80_PROB' if META[f'prmbench/{prm}']['prm_reference_scale'] == '1-reward' else 'PRM_Q80'
                arms = [('PRM native', 'PRM_native', 'native'), ('PRM', scale, 'GAP')]
            else:
                arms = [('PRM native', 'PRM', 'native'), ('PRM', prm_scale_arm(bench, prm), 'GAP')]
            arms += [(a, a, 'GAP') for a in ('TELE_N16', 'MEAN_N16', 'CHAIN_N16')]
            arms += [(k, k, 'GAP') for k in ('SUB5 alone', 'SUB5 mean', 'SUB5 +PRM', 'SUB5 +PRM mean') if k in c.arms]
            if mode == 'prm' and bench in COLLECTION:
                R = rows_prmbench(c, arms) if bench == 'prmbench' else rows_cab(c, mode, bench, prm, arms)
                gate_collection(gate, tag, bench, prm, R)
            elif bench == 'prmbench':
                R = rows_prmbench(c, arms)
                gate(f'gate1 {tag} TELE_N16 PRMScore GAP', next(r['value'] for r in R if r['arm'] == 'TELE_N16' and r['metric'] == 'PRMScore_GAP'),
                     sa_ref[(prm, 'SA_TELE_N16')]['GAP'])
                gate(f'gate1 {tag} TELE_N16 within-AUC', next(r['value'] for r in R if r['arm'] == 'TELE_N16' and r['metric'] == 'within_auc'),
                     sa_ref[(prm, 'SA_TELE_N16')]['within_auc'], 1e-3)
            elif bench == 'processbench':
                R = rows_pb(c, arms)
                if mode == 'prm':
                    ref = pb_ref[(PFILE.get(prm, prm), 'TELE_N16_NC')]
                    gate(f'gate1 {tag} TELE_N16 avg F1 NC', next(r['value'] for r in R if r['arm'] == 'TELE_N16' and r['metric'] == 'avg_f1_NC'), ref['avg_f1'])
                    gate(f'gate1 {tag} TELE_N16 within-AUC', next(r['value'] for r in R if r['arm'] == 'TELE_N16' and r['metric'] == 'within_auc'), ref['within_auc'], 1e-3)
                else:   # no NC table exists for standalone ProcessBench: check the loader with CAB_EVAL_ALL's within-AUC
                    c.keep = c.keep_gold
                    got = CAB.row(c, 'sa', bench, 'common', 'TELE_N16', 'GAP')['within_auc']
                    gate(f'gate1 {tag} TELE_N16 within-AUC (evaluate_cab definition)', got,
                         cab_ref[(bench, mode, PFILE.get(prm, prm), 'TELE_N16', 'GAP')]['within_auc'], 1e-3)
            else:
                R = rows_cab(c, mode, bench, prm, arms)
                ref = cab_ref[(bench, mode, PFILE.get(prm, prm), 'TELE_N16', 'GAP')]
                mn = 'balanced_f1_GAP' if bench == 'hard2verify' else 'macro_f1_GAP'
                gate(f'gate1 {tag} TELE_N16 {mn}', next(r['value'] for r in R if r['arm'] == 'TELE_N16' and r['metric'] == mn), ref['macro_f1'])
                gate(f'gate1 {tag} TELE_N16 within-AUC', next(r['value'] for r in R if r['arm'] == 'TELE_N16' and r['metric'] == 'within_auc'), ref['within_auc'], 1e-3)
                if bench in CROP_BENCHES:
                    assert len(c.ids) == len(np.load(cab_dir(mode, prm, bench) / 'scores' / f'SCORES_{mode}_{PFILE.get(prm, prm)}_{bench}.npz')['uids'])
                    R += rows_crop(c, arms)
            for r in R:
                rows.append(dict(benchmark=bench, mode={'prm': 'backbone', 'sa': 'standalone'}[mode], prm=prm, **r))
            print(tag, 'done', info_all[tag], flush=True)
            del c

    if part is not None:
        json.dump(dict(rows=rows, parity=parity, failed=fails, skipped=skipped, cells=info_all), open(part, 'w'), indent=1, default=float)
        return
    # CROP parity for TELE_N16 (crop_cab's own table)
    crop_ref = {(r['benchmark'], r['mode'], r['prm'], r['arm']): r for r in E.read_csv(HERE / 'vendor/CROP_TABLE6_CAB.csv') if r['grid'] == 'q201'}
    for r in rows:
        if r['arm'] == 'TELE_N16' and r['metric'] == 'crop_kept_a05' and r['benchmark'] in CROP_BENCHES:
            m = 'prm' if r['mode'] == 'backbone' else 'sa'
            gate(f"gate1 {r['benchmark']}/{m}/{r['prm']} TELE_N16 CROP kept", r['value'], crop_ref[(r['benchmark'], m, PFILE.get(r['prm'], r['prm']), 'TELE_N16')]['kept'])
    E.write_csv(HERE / 'SUB5_ALL.csv', rows)
    json.dump(dict(parity=parity, failed=fails, skipped=skipped, cells=info_all, sub5=SUB5_NAMES,
                   n_checks=len(parity), max_abs_diff=max(p['abs_diff'] for p in parity)),
              open(HERE / 'PARITY.json', 'w'), indent=1, default=float)
    print('rows', len(rows), 'parity checks', len(parity), 'failed', fails, 'skipped', skipped)
    if fails: raise SystemExit(3)


EXTRA_ARMS = ['PRMG_N16', 'EQ_N16', 'HALF_N16']   # TELE_N16 + PRM (L-SML, PRM as a channel) and two equal-weight controls


def run_extra(mode, bench, prm, part):
    """Evaluate the stored PRMG_N16 / EQ_N16 / HALF_N16 arms of one cell with the same metric functions (no refit).
    TELE_N16 is re-evaluated as the cell's parity anchor; the new arms are checked wherever a table has them."""
    cab_ref, pb_ref, sa_ref = refs(); rows, parity, fails = [], [], []

    def gate(name, got, ref, tol=TOL):
        d = abs(float(got) - float(ref)); ok = d <= tol
        parity.append(dict(check=name, got=float(got), ref=float(ref), abs_diff=d, tol=tol, ok=ok))
        print(('PASS ' if ok else 'FAIL ') + name, round(float(got), 4), round(float(ref), 4), flush=True)
        if not ok: fails.append(name)

    tag = f'{bench}/{mode}/{prm}'
    if mode == 'prm' and bench in COLLECTION:
        key, pname = bench, prm
    elif bench == 'prmbench':
        key, pname = 'prmbench', f'{prm}__standalone'
    else:
        key, pname = f'cab|{mode}|{bench}', prm
        REG[key] = dict(mode=mode, bench=bench, common=common_uids(bench))
    c = E.load(key, pname)
    if not hasattr(c, 'n_gold'): c.n_gold = len(c.ids)
    arms = [(a, a, 'GAP') for a in ['TELE_N16'] + EXTRA_ARMS if a in c.arms]
    if bench == 'prmbench':
        R = rows_prmbench(c, arms)
        get = lambda a, m='PRMScore_GAP': next(r['value'] for r in R if r['arm'] == a and r['metric'] == m)
        if mode == 'prm':
            for a in ['TELE_N16', 'EQ_N16', 'HALF_N16']:
                gate(f'extra {tag} {a} PRMScore GAP', get(a), S1.STATUS[(bench, prm, a)]['GAP'])
        else:
            gate(f'extra {tag} TELE_N16 PRMScore GAP', get('TELE_N16'), sa_ref[(prm, 'SA_TELE_N16')]['GAP'])
    elif bench == 'processbench':
        R = rows_pb(c, arms)
        if mode == 'prm':
            for a in ['TELE_N16'] + EXTRA_ARMS:
                gate(f'extra {tag} {a} avg F1 NC', next(r['value'] for r in R if r['arm'] == a and r['metric'] == 'avg_f1_NC'),
                     pb_ref[(PFILE.get(prm, prm), f'{a}_NC')]['avg_f1'])
        else:
            c.keep = c.keep_gold
            for a in ['TELE_N16', 'HALF_N16']:
                gate(f'extra {tag} {a} within-AUC (evaluate_cab definition)', CAB.row(c, 'sa', bench, 'common', a, 'GAP')['within_auc'],
                     cab_ref[(bench, mode, PFILE.get(prm, prm), a, 'GAP')]['within_auc'], 1e-3)
    else:
        R = rows_cab(c, mode, bench, prm, arms)
        mn = 'balanced_f1_GAP' if bench == 'hard2verify' else 'macro_f1_GAP'
        get = lambda a: next(r['value'] for r in R if r['arm'] == a and r['metric'] == mn)
        for a in ['TELE_N16'] + EXTRA_ARMS:
            if mode == 'prm' and bench in COLLECTION:
                if (bench, prm, a) in S1.STATUS: gate(f'extra {tag} {a} {mn}', get(a), S1.STATUS[(bench, prm, a)]['GAP'])
            elif (bench, mode, PFILE.get(prm, prm), a, 'GAP') in cab_ref:
                gate(f'extra {tag} {a} {mn}', get(a), cab_ref[(bench, mode, PFILE.get(prm, prm), a, 'GAP')]['macro_f1'])
        if bench in CROP_BENCHES:
            R += rows_crop(c, arms)
    R = [r for r in R if r['arm'] != 'TELE_N16']
    rows = [dict(benchmark=bench, mode={'prm': 'backbone', 'sa': 'standalone'}[mode], prm=prm, **r) for r in R]
    json.dump(dict(rows=rows, parity=parity, failed=fails, skipped=[], cells={}), open(part, 'w'), indent=1, default=float)
    print(tag, 'extra done', len(rows), flush=True)


def assemble():
    """Merge parts/*.json (one process per cell, to bound memory) and add the CROP parity rows."""
    rows, parity, fails, skipped, cells = [], [], [], [], {}
    for f in sorted((HERE / 'parts').glob('*.json')) + sorted((HERE / 'parts_extra').glob('*.json')):
        d = json.load(open(f)); rows += d['rows']; parity += d['parity']; fails += d['failed']; skipped += d['skipped']; cells.update(d['cells'])
    crop_ref = {(r['benchmark'], r['mode'], r['prm'], r['arm']): r for r in E.read_csv(HERE / 'vendor/CROP_TABLE6_CAB.csv') if r['grid'] == 'q201'}
    for r in rows:
        if r['arm'] == 'TELE_N16' and r['metric'] == 'crop_kept_a05' and r['benchmark'] in CROP_BENCHES:
            m = 'prm' if r['mode'] == 'backbone' else 'sa'
            ref = float(crop_ref[(r['benchmark'], m, PFILE.get(r['prm'], r['prm']), 'TELE_N16')]['kept']); dd = abs(r['value'] - ref)
            parity.append(dict(check=f"gate1 {r['benchmark']}/{m}/{r['prm']} TELE_N16 CROP kept", got=r['value'], ref=ref, abs_diff=dd, tol=TOL, ok=dd <= TOL))
            if dd > TOL: fails.append(parity[-1]['check'])
    E.write_csv(HERE / 'SUB5_ALL.csv', rows)
    json.dump(dict(parity=parity, failed=fails, skipped=skipped, cells=cells, sub5=SUB5_NAMES, n_checks=len(parity),
                   max_abs_diff=max(p['abs_diff'] for p in parity)), open(HERE / 'PARITY.json', 'w'), indent=1, default=float)
    print('rows', len(rows), 'parity checks', len(parity), 'failed', fails, 'skipped', skipped)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd == 'gates':      # python sub5_all.py gates  -> parts/_gates.json
        (HERE / 'parts').mkdir(exist_ok=True); main([], True, HERE / 'parts' / '_gates.json')
    elif cmd == 'job':      # python sub5_all.py job <prm|sa> <bench> <prm>  -> parts/<mode>_<bench>_<prm>.json
        (HERE / 'parts').mkdir(exist_ok=True); m, b, p = sys.argv[2:5]
        main([(m, b, p)], False, HERE / 'parts' / f'{m}_{b}_{p}.json')
    elif cmd == 'extra':    # python sub5_all.py extra <prm|sa> <bench> <prm>  -> parts_extra/ (PRMG_N16, EQ_N16, HALF_N16)
        (HERE / 'parts_extra').mkdir(exist_ok=True); m, b, p = sys.argv[2:5]
        run_extra(m, b, p, HERE / 'parts_extra' / f'{m}_{b}_{p}.json')
    elif cmd == 'list':
        print('\n'.join(' '.join(j) for j in all_jobs()))
    elif cmd == 'assemble':
        assemble()
    else:
        main()
