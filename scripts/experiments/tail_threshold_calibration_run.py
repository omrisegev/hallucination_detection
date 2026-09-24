"""tail_threshold_calibration_v1: why step-level tail-mark L-SML finds K=2, and calibration of the
tail threshold (common and per family).  Protocol: results/tail_threshold_calibration_v1/PROTOCOL.json.

Stage 0 (label-free diagnosis) -> fits per outer fold (V1 replay, standardized fix, threshold curve,
nested common threshold, label-free and label-selected per-family thresholds, references) ->
write-once bundle -> label-using descriptive diagnostics -> calfix_evaluate.

    python -B scripts/experiments/tail_threshold_calibration_run.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from calfix_common import Population, ScoreBundle, dump, extgen_hashes, lsml_fit, partition_equal, roles_of, sha, tail_marks  # noqa: E402
import calfix_evaluate as EV  # noqa: E402
import tail_calib_common as TC  # noqa: E402

STAGE = ROOT / 'results/tail_threshold_calibration_v1'; P = json.loads((STAGE / 'PROTOCOL.json').read_text(encoding='utf8'))
OUT = STAGE / f'run_{datetime.now():%Y%m%d_%H%M}'; OUT.mkdir(parents=True, exist_ok=False)
V1 = ROOT / 'results/family_tail_calfix_v1'; V1RUN = V1 / 'run_20260924_1542'
SCR = Path('C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/2d14a8c9-8b3f-489b-b26f-812b4b84a8b3/scratchpad')
POOL_SHA = 'd9abccfb561f459d2f3a6c5b4346a1f6766df7fdbaddb230338aa40349077a16'
LOCK_SHA = '65b35336fcc2f66b7843ec040d3bdafbbaa03bb44ae5f61f1d00335abfaea5cf'
DROP = ['ct7_ve1', 'hist_entropy_series', 'hist_spilled_series', 'hist_trace_length_series']
GRID = TC.GRID; SCALES = ('unit', 'eigen', 'complete')
T0 = time.perf_counter()
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds')}; dump(OUT / 'RUN_STATUS.json', status)


def log(*a):
    print(f'[{time.perf_counter() - T0:6.0f}s]', *a, flush=True)


def frozen_hashes():
    files = sorted(p for p in V1.rglob('*') if p.is_file()) + [HERE / 'calfix_common.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


FROZEN0 = frozen_hashes(); assert FROZEN0['results/family_tail_calfix_v1/TRANSFER_LOCK_V1.json'] == LOCK_SHA

# ------------------------------------------------------------------ inputs (identical construction to family_tail_calfix_run / the lock)
pop = Population(); off = pop.off; ns = pop.ns
assert sha(SCR / 'pool_z.npy') == POOL_SHA
POOL = np.load(SCR / 'pool_z.npy'); PN = json.loads((SCR / 'pool_names.json').read_text(encoding='utf8')); assert POOL.shape == (pop.total, len(PN))
names = [c for c in PN if c not in DROP]; ix = {c: j for j, c in enumerate(names)}
Z = pop.answer_standardize(POOL[:, [PN.index(c) for c in names]]); del POOL
B11N = PN[:11]
D = json.loads((V1RUN / 'DESIGN.json').read_text(encoding='utf8'))['retrospective']
sign = np.array([float(D['orientation'][c]) for c in names]); kept = D['kept']
S3 = json.loads((ROOT / 'results/named_group_fusion_v1/PROTOCOL.json').read_text(encoding='utf8'))['splits']['S3_M15']; FAMS = list(S3)
F15 = pop.answer_standardize(np.column_stack([np.column_stack([Z[:, ix[c]] * sign[ix[c]] for c in mem]).mean(1) for mem in S3.values()]))
Zo = Z * sign
X = {'F15': F15, 'K28': Zo[:, [ix[c] for c in kept]], 'A48o': Zo, 'B11': Z[:, [ix[c] for c in B11N]]}
COLS = {'F15': FAMS, 'K28': kept, 'A48o': names, 'B11': B11N}
ANCHOR = {'F15': FAMS.index('level_entropy'), 'K28': kept.index('q15_H1'), 'A48o': ix['q15_H1'], 'B11': 0}
A = ANCHOR['F15']
dump(OUT / 'INPUT_MANIFEST.json', {'pool_z': {'path': str(SCR / 'pool_z.npy'), 'sha256': POOL_SHA}, 'pool_names': sha(SCR / 'pool_names.json'),
      **{k: {'path': str(v), 'sha256': sha(v)} for k, v in pop.inputs.items()}, 'design': sha(V1RUN / 'DESIGN.json'), 'extgen': extgen_hashes(),
      'script_sha256': sha(Path(__file__)), 'tail_calib_common_sha256': sha(HERE / 'tail_calib_common.py'), 'calfix_common_sha256': sha(HERE / 'calfix_common.py'),
      'evaluator_sha256': sha(HERE / 'calfix_evaluate.py'), 'protocol_sha256': sha(STAGE / 'PROTOCOL.json')})

# ------------------------------------------------------------------ marks
RANK, DEGEN = {}, {}
for q in GRID:
    RANK[q], DEGEN[f'F15_rank_{q}'] = tail_marks(F15, off, q, tie_aware=True)
BANK20 = {'F15': RANK[.2]}
for b in ('K28', 'A48o', 'B11'):
    BANK20[b], DEGEN[f'{b}_rank_0.2'] = tail_marks(X[b], off, .2, tie_aware=True)
KQ = {q: np.repeat(np.maximum(1, np.ceil(q * ns)) / ns, ns) for q in GRID}          # uncentred mark mean per answer = k/n
log('inputs and marks built')

# ------------------------------------------------------------------ stage 0: label-free diagnosis of K
def kinfo(R, scale, cols):
    K, c, r, curve = TC.groups_from_R(R, scale); rs = sorted(x[1] for x in curve)
    return {'K': int(K), 'groups': {int(u): [cols[i] for i in np.flatnonzero(c == u)] for u in np.unique(c)}, 'labels': c.tolist(),
            'curve': [[int(k), float(rr)] for k, rr, _ in curve], 'gap_rel_to_runner_up': float((rs[1] - rs[0]) / abs(rs[0])) if rs[0] else None}


def corr_of(M):
    Xs, _, _ = TC.col_standardize(M); return np.cov(Xs.T)


v1_models = [json.loads(l) for l in (V1RUN / 'MODELS.jsonl').read_text(encoding='utf8').splitlines()]
v1K = {(r['method'], r['outer_fold']): r.get('K') for r in v1_models}
diag = {'banks': {}, 'scale_control_F15_continuous': {}, 'ari_F15': {}, 'v1_K_match': {}}
parts = {'raw': [], 'std': [], 'cont': []}
for b in X:
    diag['banks'][b] = []
    for k in range(5):
        fit_f, _cal, _ev = roles_of(k); fr = pop.rows(np.isin(pop.fold, fit_f))
        T = BANK20[b][fr]; Rraw = np.cov(T.T); Rstd = corr_of(T); Rc = np.cov(X[b][fr].T)
        rec = {'outer_fold': k, 'mark_variance': dict(zip(COLS[b], np.round(np.diag(Rraw), 4).tolist())),
               **{f'{nm}_{sc}': kinfo(R, sc, COLS[b]) for nm, R in (('raw_marks', Rraw), ('std_marks', Rstd), ('continuous', Rc)) for sc in SCALES}}
        diag['banks'][b].append(rec)
        vk = v1K.get((f'{b}_tailtie_lsml', k)); diag['v1_K_match'][f'{b}_fold{k}'] = {'v1_K': vk, 'raw_unit_K': rec['raw_marks_unit']['K']}
        if b == 'F15':
            parts['raw'].append(rec['raw_marks_unit']['labels']); parts['std'].append(rec['std_marks_unit']['labels']); parts['cont'].append(rec['continuous_unit']['labels'])
            diag['scale_control_F15_continuous'][f'fold{k}'] = {f'x{f}_{sc}': kinfo(Rc * f * f, sc, FAMS)['K'] for f in (1.0, .4) for sc in SCALES}
        log(f'diagnosis {b} fold {k}: K raw/std/cont (unit) = {rec["raw_marks_unit"]["K"]}/{rec["std_marks_unit"]["K"]}/{rec["continuous_unit"]["K"]}')
diag['ari_F15'] = {nm: float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(v, 2)])) for nm, v in parts.items()}
diag['v1_K_all_match'] = all(v['v1_K'] == v['raw_unit_K'] for v in diag['v1_K_match'].values() if v['v1_K'] is not None)
diag['mark_degeneracy'] = DEGEN
# mark position (label-free), answers with >= 5 steps
rel = np.concatenate([np.arange(n) / max(n - 1, 1) for n in ns]); long = np.repeat(ns >= 5, ns)
last = np.zeros(pop.total, bool)
for a, b_ in zip(off[:-1], off[1:]):
    last[b_ - int(np.ceil(.2 * (b_ - a))):b_] = True
diag['position'] = {}
for q in GRID:
    U = RANK[q] + KQ[q][:, None]; W = U[long]
    diag['position'][str(q)] = {f: {'mean_rel_pos': float((W[:, j] * rel[long]).sum() / W[:, j].sum()), 'share_last20pct': float((W[:, j] * last[long]).sum() / W[:, j].sum())} for j, f in enumerate(FAMS)}
dump(OUT / 'DIAGNOSIS.json', diag)
log('stage 0 done; V1 K match', diag['v1_K_all_match'], 'ARI', diag['ari_F15'])

# ------------------------------------------------------------------ fits
B = ScoreBundle(pop); failures = []; selection = {}
yv = (~pop.labels).astype(float); nc_steps = np.repeat(pop.noncontrol, ns)
U20 = RANK[.2] + KQ[.2][:, None]


def rec_fit(f, cols=FAMS):
    return {'weights': dict(zip(cols, np.round(f['weights'], 10))), 'groups': dict(zip(cols, f['groups'].tolist())), 'K': f['K'],
            'anchor_spearman': f['anchor_spearman'], 'anchor_flipped': f['anchor_flipped'], 'residual': f['residual'], 'small_m_guarded': f['small_m_guarded'],
            **({'fit_sd': dict(zip(cols, np.round(f['fit_sd'], 8))), 'grouping_degenerate': f['grouping_degenerate'], 'residual_gap_rel': f['residual_gap_rel']} if 'fit_sd' in f else {})}


def emit(m, k, raw, record):
    B.put_full(m, k, pop.answer_z(raw), record)


for k in range(5):
    t = time.perf_counter(); fit_f, cal, ev = roles_of(k)
    fr = pop.rows(np.isin(pop.fold, fit_f)); cal_rows = pop.rows(pop.fold == cal)
    sel_rows = np.flatnonzero(nc_steps & np.isin(pop.step_fold, fit_f))
    ym = yv.copy(); ym[np.isin(pop.step_fold, [cal, ev])] = np.nan              # withheld from every selection
    Ff = F15[fr]; RF = {q: RANK[q][fr] for q in GRID}

    def objective_prm(raw):
        return TC.fit_prmscore(TC.fast_answer_z(raw, off), ym, sel_rows, cal_rows)

    def std_fit(Tfit, scale='unit'):
        return TC.lsml_fit_scaled(Tfit, A, Ff, standardize=True, loading_scale=scale)

    # references
    emit('F15_equal', k, F15.mean(1), {'rule': 'equal'})
    f = lsml_fit(Ff, A); emit('F15_cov_lsml', k, F15 @ f['weights'], {'rule': 'continuous L-SML', **rec_fit(f)})
    f = lsml_fit(X['B11'][fr], 0); emit('B11_lsml', k, X['B11'] @ f['weights'], {'rule': 'frozen bank11 L-SML', **rec_fit(f, B11N)})
    emit('ct7', k, pop.ct7, {'rule': 'frozen CT7 step scores, answer-z'})
    # V1 replay and the minimal fix
    f = lsml_fit(RF[.2], A, Ff); emit('T20raw', k, F15 @ f['weights'], {'rule': 'rank .20 raw marks, unit (V1)', **rec_fit(f)})
    f20 = std_fit(RF[.2]); emit('T20s', k, F15 @ f20['weights'], {'rule': 'rank .20 standardized, unit', **rec_fit(f20)})
    f = std_fit(RF[.2], 'complete'); emit('T20s_complete', k, F15 @ f['weights'], {'rule': 'rank .20 standardized, complete', **rec_fit(f)})
    emit('T20s_votes', k, U20 @ (f20['weights'] / f20['fit_sd']), {'rule': 'T20s weights on uncentred marks (w/sd)', 'weights': dict(zip(FAMS, f20['weights'] / f20['fit_sd']))})
    w = partition_equal(f20['groups']); emit('T20s_partition_equal', k, F15 @ w, {'rule': 'equal on T20s groups', 'weights': dict(zip(FAMS, w))})
    # threshold curve and nested common threshold
    cand, curve_scores = {}, {}
    for q in GRID:
        tag = f'Q{int(round(q * 100)):02d}'
        if q == .2:
            cand['T20s'] = (F15 @ f20['weights'], {'rule': 'rank .20 standardized, unit', **rec_fit(f20)})
        else:
            f = std_fit(RF[q]); cand[f'{tag}r'] = (F15 @ f['weights'], {'rule': f'rank {q} standardized, unit', **rec_fit(f)})
            emit(f'{tag}r', k, *cand[f'{tag}r'])
    for q in GRID:
        tag = f'Q{int(round(q * 100)):02d}'; Tv, tau = TC.value_marks(F15, off, q, fr)
        f = std_fit(Tv[fr]); cand[f'{tag}v'] = (F15 @ f['weights'], {'rule': f'value {q} standardized, unit', 'tau': dict(zip(FAMS, tau)), **rec_fit(f)})
        emit(f'{tag}v', k, *cand[f'{tag}v'])
    order = [f'Q{int(round(q * 100)):02d}r' if q != .2 else 'T20s' for q in GRID] + [f'Q{int(round(q * 100)):02d}v' for q in GRID]
    curve_scores = {m: objective_prm(cand[m][0]) for m in order}
    best = max(order, key=lambda m: (curve_scores[m], -order.index(m)))
    emit('Qnest', k, cand[best][0], {'rule': 'nested choice among 14 curve fits by fit-fold PRMScore', 'chosen': best, 'fit_fold_prmscore': curve_scores, **cand[best][1]})
    log(f'fold {k}: curve + Qnest done (chosen {best})')
    # per-family, label-free: relative latent-group misfit from one Gram matrix of all (family, q) columns
    XS = np.column_stack([TC.col_standardize(RF[q][:, j:j + 1])[0][:, 0] for j in range(15) for q in GRID])
    G = np.cov(XS.T); dg = np.sqrt(np.diag(G)); G = G / np.outer(dg, dg); del XS
    def obj_free(qs):
        idx = [j * len(GRID) + GRID.index(q) for j, q in enumerate(qs)]
        return TC.rel_residual(G[np.ix_(idx, idx)], 'complete')[0]
    qs_free, v_free, tr_free = TC.coordinate_descent(15, obj_free, maximize=False)
    f = std_fit(TC.mixed_marks(RF, qs_free))
    emit('PFfree', k, F15 @ f['weights'], {'rule': 'per-family rank fractions, label-free (min relative latent-group misfit)', 'fractions': dict(zip(FAMS, qs_free)),
                                          'objective': v_free, 'objective_at_start': tr_free[0]['value'], **rec_fit(f)})
    log(f'fold {k}: PFfree done {dict(zip([x[:6] for x in FAMS], qs_free))} misfit {tr_free[0]["value"]:.4f}->{v_free:.4f}')
    # per-family, label-selected (nested)
    def obj_lab(qs):
        try:
            ff = std_fit(TC.mixed_marks(RF, qs))
        except ValueError as e:
            failures.append({'where': 'PFlab search', 'outer_fold': k, 'fractions': qs, 'reason': str(e)}); return -np.inf
        return objective_prm(F15 @ ff['weights'])
    qs_lab, v_lab, tr_lab = TC.coordinate_descent(15, obj_lab, maximize=True)
    f = std_fit(TC.mixed_marks(RF, qs_lab))
    emit('PFlab', k, F15 @ f['weights'], {'rule': 'per-family rank fractions, nested fit-fold PRMScore', 'fractions': dict(zip(FAMS, qs_lab)),
                                         'objective': v_lab, 'objective_at_start': tr_lab[0]['value'], **rec_fit(f)})
    selection[k] = {'Qnest': {'chosen': best, 'fit_fold_prmscore': curve_scores}, 'PFfree': {'fractions': dict(zip(FAMS, qs_free)), 'trace': tr_free},
                    'PFlab': {'fractions': dict(zip(FAMS, qs_lab)), 'trace': tr_lab}}
    log(f'fold {k}: PFlab done {dict(zip([x[:6] for x in FAMS], qs_lab))} fit PRMScore {tr_lab[0]["value"]:.4f}->{v_lab:.4f}; fold {time.perf_counter() - t:.0f}s')
    dump(OUT / 'SELECTION.json', selection)

# ------------------------------------------------------------------ V1 replay (references and the V1 candidate must equal run_20260924_1542 exactly)
old = np.load(V1RUN / 'SCORES.npz'); replay = {}
for new, oldm in (('T20raw', 'F15_tailtie_lsml'), ('F15_equal', 'F15_equal'), ('F15_cov_lsml', 'F15_cov_lsml'), ('B11_lsml', 'B11_cov_lsml'), ('ct7', 'ct7')):
    replay[new] = max(float(np.abs(B.scores[new, r] - old[f'{r}__{oldm}']).max()) for r in ('eval', 'cal'))
replay['status'] = 'PASS' if max(replay.values()) < 1e-12 else 'FAIL'
dump(OUT / 'REPLAY_V1.json', replay); log('V1 replay', replay)
assert replay['status'] == 'PASS', replay
checks = B.finalize(OUT); dump(OUT / 'FAILURES.json', failures)
log('bundle', checks['status'], 'methods', checks['methods'], 'search failures', len(failures))

# ------------------------------------------------------------------ label-using descriptive diagnostics (after all fits; never used for selection)
err = pop.labels & nc_steps; val = ~pop.labels & nc_steps; dl = {'d_rank': {}, 'd_value': {}}
for q in GRID:
    U = RANK[q] + KQ[q][:, None]; dl['d_rank'][str(q)] = dict(zip(FAMS, (U[err].mean(0) - U[val].mean(0)).round(5).tolist()))
    Vm = (F15 > np.quantile(F15, 1 - q, axis=0)).astype(float); dl['d_value'][str(q)] = dict(zip(FAMS, (Vm[err].mean(0) - Vm[val].mean(0)).round(5).tolist()))
Cv, Ce, Cm = (np.corrcoef(RANK[.2][m].T) for m in (val, err, nc_steps))
m0 = [json.loads(l) for l in (OUT / 'MODELS.jsonl').read_text(encoding='utf8').splitlines() if json.loads(l)['method'] == 'T20s' and json.loads(l)['outer_fold'] == 0][0]
g0 = np.array([m0['groups'][f] for f in FAMS])
same = (g0[:, None] == g0[None, :]) & ~np.eye(15, dtype=bool); cross = g0[:, None] != g0[None, :]
dl['conditional_corr_20'] = {'groups_T20s_fold0': dict(zip(FAMS, g0.tolist())),
                             'mean_abs': {nm: {'inside_groups': float(np.abs(C[same]).mean()), 'across_groups': float(np.abs(C[cross]).mean())} for nm, C in (('given_valid', Cv), ('given_error', Ce), ('marginal', Cm))},
                             'given_valid': np.round(Cv, 4).tolist(), 'given_error': np.round(Ce, 4).tolist(), 'marginal': np.round(Cm, 4).tolist(), 'families': FAMS}
dump(OUT / 'DIAG_LABEL_DESCRIPTIVE.json', dl)

# ------------------------------------------------------------------ evaluation
PRIMARY = [tuple(x) for x in P['primary_contrasts_P1_prmscore_holm7']]; DESC = [tuple(x) for x in P['descriptive_contrasts']]
res = EV.evaluate(OUT, OUT, PRIMARY, DESC, pop=pop)
FROZEN1 = frozen_hashes(); dump(OUT / 'FROZEN_HASHES.json', {'before': FROZEN0, 'after_equal': FROZEN1 == FROZEN0})
assert FROZEN1 == FROZEN0, 'frozen files changed'
status.update({'status': 'COMPLETE', 'finished': datetime.now().isoformat(timespec='seconds'), 'search_failures': len(failures), 'seconds': time.perf_counter() - T0})
dump(OUT / 'RUN_STATUS.json', status)
import pandas as pd  # noqa: E402
pd.set_option('display.width', 250)
MT = pd.read_csv(OUT / 'METRICS.csv'); print(MT[['method', 'prmscore_P1', 'prmscore_P1b', 'prmscore_P2', 'within_auc', 'pb_sla_macro8']].sort_values('prmscore_P1', ascending=False).round(4).to_string(index=False))
CT = res['contrasts']; print(CT[CT.endpoint == 'prmscore_P1'][['family', 'a', 'b', 'delta', 'ci95_lo', 'ci95_hi', 'p_boot', 'p_holm_primary', 'folds_a_gt_b']].round(4).to_string(index=False))
log('done')
