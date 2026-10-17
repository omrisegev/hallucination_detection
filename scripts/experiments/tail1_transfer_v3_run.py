"""tail1_transfer_v3: maximal-step (top-1) tail-mark L-SML with standardized marks on family15 and
bank11 (unoriented and oriented).  Source five-fold evaluation, then TRANSFER_LOCK_V3 (= V2 unchanged
+ three rows; deployment fit folds 0-3, q80 fold 4).  All three rows go external regardless of the
source result (PROTOCOL.json).

    python -B scripts/experiments/tail1_transfer_v3_run.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from calfix_common import Population, ScoreBundle, dump, extgen_hashes, lsml_fit, roles_of, sha, tail_marks  # noqa: E402
import calfix_evaluate as EV  # noqa: E402
import tail_calib_common as TC  # noqa: E402

STAGE = ROOT / 'results/tail1_transfer_v3'; P = json.loads((STAGE / 'PROTOCOL.json').read_text(encoding='utf8'))
OUT = STAGE / f'run_{datetime.now():%Y%m%d_%H%M}'; OUT.mkdir(parents=True, exist_ok=False)
PREV = ROOT / 'results/tail_threshold_calibration_v1/run_20260924_2058'
V2LOCK = ROOT / 'results/tail_threshold_calibration_v1/TRANSFER_LOCK_V2.json'; V2SHA = '0c5c55996503394edf2faeb1bc066e3fdec3c6eb315c79c0e7d59ca7d61c986c'
SCR = Path('C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/2d14a8c9-8b3f-489b-b26f-812b4b84a8b3/scratchpad')
POOL_SHA = 'd9abccfb561f459d2f3a6c5b4346a1f6766df7fdbaddb230338aa40349077a16'
T0 = time.perf_counter()


def log(*a):
    print(f'[{time.perf_counter() - T0:6.0f}s]', *a, flush=True)


assert sha(V2LOCK) == V2SHA
v2 = json.loads(V2LOCK.read_text(encoding='utf8')); R = v2['recipe']
pop = Population(); off = pop.off
assert sha(SCR / 'pool_z.npy') == POOL_SHA
POOL = np.load(SCR / 'pool_z.npy'); PN = json.loads((SCR / 'pool_names.json').read_text(encoding='utf8'))
names = R['channels_48']; ix = {c: j for j, c in enumerate(names)}
Z = pop.answer_standardize(POOL[:, [PN.index(c) for c in names]]); del POOL
S3 = R['families_15']; FAMS = list(S3); sign = R['source_signs']; B11N = R['bank11']
F15 = pop.answer_standardize(np.column_stack([np.column_stack([Z[:, ix[c]] * sign[c] for c in mem]).mean(1) for mem in S3.values()]))
X = {'F15': F15, 'B11': Z[:, [ix[c] for c in B11N]], 'B11o': np.column_stack([Z[:, ix[c]] * sign[c] for c in B11N])}
COLS = {'F15': FAMS, 'B11': B11N, 'B11o': B11N}; ANCHOR = {'F15': FAMS.index('level_entropy'), 'B11': 0, 'B11o': 0}
MARKS, DEGEN = {}, {}
for b in X:
    MARKS[b], DEGEN[b] = tail_marks(X[b], off, None, tie_aware=True)
T20, _ = tail_marks(F15, off, .2, tie_aware=True)
dump(OUT / 'INPUT_MANIFEST.json', {'pool_z_sha256': POOL_SHA, 'lock_v2_sha256': V2SHA, 'script_sha256': sha(Path(__file__)), 'tail_calib_common_sha256': sha(HERE / 'tail_calib_common.py'),
                                   'calfix_common_sha256': sha(HERE / 'calfix_common.py'), 'evaluator_sha256': sha(HERE / 'calfix_evaluate.py'), 'protocol_sha256': sha(STAGE / 'PROTOCOL.json'), 'extgen': extgen_hashes()})
log('inputs and top-1 marks built', {b: {k: round(v, 4) for k, v in d.items()} for b, d in DEGEN.items()})


def rec_fit(f, cols):
    return {'weights': dict(zip(cols, np.round(f['weights'], 10))), 'groups': dict(zip(cols, f['groups'].tolist())), 'K': f['K'], 'anchor_spearman': f['anchor_spearman'],
            'small_m_guarded': f['small_m_guarded'], **({'fit_sd': dict(zip(cols, np.round(f['fit_sd'], 8))), 'grouping_degenerate': f['grouping_degenerate']} if 'fit_sd' in f else {})}


B = ScoreBundle(pop); diag = {'K_raw_unit': {}, 'K_std_unit': {}, 'mark_degeneracy': DEGEN}
for k in range(5):
    fit_f, _cal, _ev = roles_of(k); fr = pop.rows(np.isin(pop.fold, fit_f))
    def emit(m, raw, record):
        B.put_full(m, k, pop.answer_z(raw), record)
    emit('F15_equal', F15.mean(1), {'rule': 'equal'})
    f = lsml_fit(F15[fr], ANCHOR['F15']); emit('F15_cov_lsml', F15 @ f['weights'], {'rule': 'continuous L-SML', **rec_fit(f, FAMS)})
    f = lsml_fit(X['B11'][fr], 0); emit('B11_lsml', X['B11'] @ f['weights'], {'rule': 'frozen bank11 L-SML', **rec_fit(f, B11N)})
    emit('B11_equal', X['B11'].mean(1), {'rule': 'equal, unoriented bank11'}); emit('B11o_equal', X['B11o'].mean(1), {'rule': 'equal, oriented bank11'})
    emit('ct7', pop.ct7, {'rule': 'frozen CT7 step scores, answer-z'})
    f = TC.lsml_fit_scaled(T20[fr], ANCHOR['F15'], F15[fr], standardize=True); emit('F15_tailstd_lsml', F15 @ f['weights'], {'rule': 'top-20% standardized (V2 row)', **rec_fit(f, FAMS)})
    for b in X:
        f = TC.lsml_fit_scaled(MARKS[b][fr], ANCHOR[b], X[b][fr], standardize=True, loading_scale='unit')
        emit(f'{b}_tail1s_lsml', X[b] @ f['weights'], {'rule': 'top-1 (maximal step) standardized, unit', **rec_fit(f, COLS[b])})
        diag['K_std_unit'].setdefault(b, []).append(f['K'])
        diag['K_raw_unit'].setdefault(b, []).append(TC.groups_from_R(np.cov(MARKS[b][fr].T), 'unit')[0])
    log(f'fold {k} done; K std', {b: diag['K_std_unit'][b][-1] for b in X}, 'raw', {b: diag['K_raw_unit'][b][-1] for b in X})

old = np.load(PREV / 'SCORES.npz'); replay = {}
for new, oldm in (('F15_equal', 'F15_equal'), ('F15_cov_lsml', 'F15_cov_lsml'), ('B11_lsml', 'B11_lsml'), ('ct7', 'ct7'), ('F15_tailstd_lsml', 'T20s')):
    replay[new] = max(float(np.abs(B.scores[new, r] - old[f'{r}__{oldm}']).max()) for r in ('eval', 'cal'))
replay['status'] = 'PASS' if max(replay.values()) < 1e-12 else 'FAIL'
dump(OUT / 'REPLAY.json', replay); log('replay', replay); assert replay['status'] == 'PASS'
checks = B.finalize(OUT); dump(OUT / 'DIAGNOSIS.json', diag); log('bundle', checks['status'])
PRIMARY = [tuple(x) for x in P['primary_contrasts_P1_prmscore_holm5']]; DESC = [tuple(x) for x in P['descriptive_contrasts']]
res = EV.evaluate(OUT, OUT, PRIMARY, DESC, pop=pop)

# ------------------------------------------------------------------ TRANSFER_LOCK_V3 (rows pre-declared to go external regardless of the source result)
fit_rows = pop.rows(pop.fold < 4); cal_rows = pop.rows(pop.fold == 4)
lock = json.loads(json.dumps(v2)); new = {}
for b in X:
    f = TC.lsml_fit_scaled(MARKS[b][fit_rows], ANCHOR[b], X[b][fit_rows], standardize=True, loading_scale='unit')
    s = pop.answer_z(X[b] @ f['weights'])
    new[f'{b}_tail1s_lsml'] = {'weights': dict(zip(COLS[b], np.asarray(f['weights'], float))), 'q80_threshold_fold4': float(np.quantile(s[cal_rows], .8)), 'K': f['K'],
                               'groups': dict(zip(COLS[b], f['groups'].tolist())), 'small_m_guarded': f['small_m_guarded'], 'anchor_spearman': f['anchor_spearman'],
                               'fit_sd_of_marks': dict(zip(COLS[b], np.asarray(f['fit_sd'], float))), 'grouping_degenerate': f['grouping_degenerate']}
MT = {r['method']: r for r in __import__('csv').DictReader(open(OUT / 'METRICS.csv', encoding='utf8'))}
lock.update({
 'lock': 'family_tail_transfer_lock_v3', 'date': datetime.now().isoformat(timespec='seconds'),
 'status': 'LOCKED (source side). Extends TRANSFER_LOCK_V2 (sha256 %s) by THREE rows (maximal-step tail marks, standardized); every V1/V2 row, recipe item, weight and threshold is copied unchanged.' % V2SHA,
 'reason_v3': 'Omri, 2026-09-24: run the same (fixed) procedure with a tail that is only the maximal step, on bank11 and family 15. Both bank11 orientations are declared (results/tail1_transfer_v3/PROTOCOL.json).',
 'exposure_v3': 'V1 and V2 external results had been read before this lock. The three rows are defined by the request and the declared fix, not chosen with external data, and go external regardless of their source result. Exploratory follow-up on exposed benchmarks.',
 'source_evidence_v3': {'run': str(OUT.relative_to(ROOT)), 'metrics_sha256': sha(OUT / 'METRICS.csv'), 'contrasts_sha256': sha(OUT / 'CONTRASTS.csv'),
                        'prmscore_P1': {m: float(MT[m]['prmscore_P1']) for m in MT}},
})
lock['rows'].update({'F15_tail1s_lsml': P['rows_new']['F15_tail1s_lsml'], 'B11_tail1s_lsml': P['rows_new']['B11_tail1s_lsml'], 'B11o_tail1s_lsml': P['rows_new']['B11o_tail1s_lsml']})
lock['recipe']['tail1_marks_v3'] = 'tie-aware top-1 step per answer and column (calfix_common.tail_marks frac=None, tie_aware=True), centred within answer, then z-scored over the pooled fit rows before lsml_continuous (unit, small_m_guard)'
lock['recipe']['bank11_oriented_v3'] = 'B11o columns = bank11 channels after answer-z multiplied by source_signs (same signs as the families); B11 columns = unoriented bank11 as the frozen bank11 row'
lock['deployment'].update(new)
lock['external_primary_contrasts_v3'] = [list(x[:2]) for x in P['primary_contrasts_P1_prmscore_holm5']]
lock['external_multiplicity_v3'] = 'per cell (3 cells); paired source-question bootstrap, 100,000 draws, seed 20260924; Bonferroni over 5 contrasts x 3 cells = 15. V1 (18) and V2 (12) contrasts replayed as bridges.'
lock['code_hashes_v3'] = {'script': sha(Path(__file__)), 'tail_calib_common': sha(HERE / 'tail_calib_common.py'), 'calfix_common': sha(HERE / 'calfix_common.py'), **extgen_hashes()}
dump(STAGE / 'TRANSFER_LOCK_V3.json', lock)
import pandas as pd  # noqa: E402
pd.set_option('display.width', 250)
print(pd.read_csv(OUT / 'METRICS.csv')[['method', 'prmscore_P1', 'within_auc', 'pb_sla_macro8']].sort_values('prmscore_P1', ascending=False).round(4).to_string(index=False))
CT = res['contrasts']; print(CT[CT.endpoint == 'prmscore_P1'][['family', 'a', 'b', 'delta', 'ci95_lo', 'ci95_hi', 'p_holm_primary', 'folds_a_gt_b']].round(4).to_string(index=False))
print('deployment', {m: (v['K'], round(v['q80_threshold_fold4'], 4)) for m, v in new.items()})
log('lock v3 sha256', sha(STAGE / 'TRANSFER_LOCK_V3.json'))
