"""TRANSFER_LOCK_V2: V1 unchanged plus ONE corrected row, F15_tailstd_lsml (the tail marks
standardized before lsml_continuous, its documented input contract; tail_threshold_calibration_v1).
Deployment fit on source folds 0-3, q80 threshold on fold 4, exactly as V1.  Touches no external data.
Written before any external score of the corrected row and before reading V1 external results.

    python -B scripts/experiments/family_tail_transfer_lock_v2.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from calfix_common import Population, dump, extgen_hashes, lsml_fit, sha, tail_marks  # noqa: E402
import tail_calib_common as TC  # noqa: E402

V1LOCK = ROOT / 'results/family_tail_calfix_v1/TRANSFER_LOCK_V1.json'; V1SHA = '65b35336fcc2f66b7843ec040d3bdafbbaa03bb44ae5f61f1d00335abfaea5cf'
STAGE = ROOT / 'results/tail_threshold_calibration_v1'; RUN = STAGE / 'run_20260924_2058'
SCR = Path('C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/2d14a8c9-8b3f-489b-b26f-812b4b84a8b3/scratchpad')
assert sha(V1LOCK) == V1SHA
v1 = json.loads(V1LOCK.read_text(encoding='utf8')); R = v1['recipe']
pop = Population(); off = pop.off
POOL = np.load(SCR / 'pool_z.npy'); PN = json.loads((SCR / 'pool_names.json').read_text(encoding='utf8'))
names = R['channels_48']; ix = {c: j for j, c in enumerate(names)}
Z = pop.answer_standardize(POOL[:, [PN.index(c) for c in names]]); del POOL
S3 = R['families_15']; FAMS = list(S3); sign = R['source_signs']
F15 = pop.answer_standardize(np.column_stack([np.column_stack([Z[:, ix[c]] * sign[c] for c in mem]).mean(1) for mem in S3.values()]))
A = FAMS.index('level_entropy')
T, degen = tail_marks(F15, off, .2, tie_aware=True)
fit_rows = pop.rows(pop.fold < 4); cal_rows = pop.rows(pop.fold == 4)

# replay of the V1 candidate row: same F15, marks, rows, weights and threshold as the V1 lock
f1 = lsml_fit(T[fit_rows], A, F15[fit_rows]); d1 = v1['deployment']['F15_tailtie_lsml']
q1 = float(np.quantile(pop.answer_z(F15 @ f1['weights'])[cal_rows], .8))
replay = {'F15_tailtie_lsml_weights_max_abs_diff': float(max(abs(d1['weights'][c] - w) for c, w in zip(FAMS, f1['weights']))),
          'F15_tailtie_lsml_threshold_abs_diff': abs(d1['q80_threshold_fold4'] - q1), 'F15_tailtie_lsml_K': f1['K']}
replay['status'] = 'PASS' if replay['F15_tailtie_lsml_weights_max_abs_diff'] < 1e-12 and replay['F15_tailtie_lsml_threshold_abs_diff'] < 1e-12 else 'FAIL'
assert replay['status'] == 'PASS', replay

# the corrected row
f2 = TC.lsml_fit_scaled(T[fit_rows], A, F15[fit_rows], standardize=True, loading_scale='unit')
s2 = pop.answer_z(F15 @ f2['weights']); q2 = float(np.quantile(s2[cal_rows], .8))
new = {'weights': dict(zip(FAMS, np.asarray(f2['weights'], float))), 'q80_threshold_fold4': q2, 'K': f2['K'],
       'groups': dict(zip(FAMS, f2['groups'].tolist())), 'small_m_guarded': f2['small_m_guarded'], 'anchor_spearman': f2['anchor_spearman'],
       'fit_sd_of_marks': dict(zip(FAMS, np.asarray(f2['fit_sd'], float))), 'grouping_degenerate': f2['grouping_degenerate'], 'residual_gap_rel': f2['residual_gap_rel']}
M = {r['method']: r for r in __import__('csv').DictReader(open(RUN / 'METRICS.csv', encoding='utf8'))}

lock = json.loads(json.dumps(v1))                                   # V1 content kept verbatim
lock.update({
 'lock': 'family_tail_transfer_lock_v2', 'date': datetime.now().isoformat(timespec='seconds'),
 'status': 'LOCKED (source side). Extends TRANSFER_LOCK_V1 (sha256 %s) by ONE row, F15_tailstd_lsml; every V1 row, recipe, weight and threshold is copied unchanged. V1 stays the registered study; V2 adds the defect-corrected candidate for an external comparison requested by Omri on 2026-09-24.' % V1SHA,
 'reason': 'Defect in the V1 candidate recipe (tail_threshold_calibration_v1): the centred tail marks (variance ~0.18) were passed to lsml_continuous unstandardized, although its documented input is z-scored; its default unit loading-scale K criterion is scale sensitive, so K collapsed to 2 on every fold (the same continuous F15 scaled by 0.4 also gives K=2). After pooled standardization K=4 with a stable partition (ARI 1 across folds).',
 'exposure': 'V1 external results (results/family_tail_external_v1, commit 1d3c23681, 2026-09-24 21:16) existed before this lock; they were NOT read before it was written. The corrected row is fully determined by the defect fix declared in results/tail_threshold_calibration_v1/PROTOCOL.json before any external result; no variant was chosen with external data. The external benchmarks remain previously exposed: exploratory follow-up, not confirmation.',
 'source_evidence_for_the_new_row': {'run': str(RUN.relative_to(ROOT)), 'metrics_sha256': sha(RUN / 'METRICS.csv'), 'contrasts_sha256': sha(RUN / 'CONTRASTS.csv'),
   'prmscore_P1': {m: float(M[m]['prmscore_P1']) for m in ('T20s', 'T20raw', 'F15_equal', 'F15_cov_lsml', 'B11_lsml', 'ct7')},
   'note': 'T20s is this row on source (five-fold); it is LOWER than the V1 candidate (T20raw) by 0.35 [-0.64, -0.05] (Holm .096) and below F15_equal by 0.16. It is added because it is the correct recipe, not because it scored better.'},
})
lock['rows']['F15_tailstd_lsml'] = 'defect-corrected candidate: the V1 candidate with the tail marks z-scored (pooled over the fit rows) before lsml_continuous (loading_scale unit, small_m_guard); weights applied to the continuous family features, then within-answer z. Source K=4 (level block / margin-change block / changepoint+tail_ratio / CUSUM pair).'
lock['recipe']['tail_marks_standardization_F15_tailstd_lsml'] = 'marks as V1 (tie-aware top ceil(0.2 n), centred within answer), then each column z-scored over the pooled fit rows (ddof 0) before lsml_continuous; the V1 row keeps its unstandardized marks'
lock['deployment']['F15_tailstd_lsml'] = new
lock['external_primary_contrasts_v2'] = [['F15_tailstd_lsml', 'F15_tailtie_lsml'], ['F15_tailstd_lsml', 'F15_equal'], ['F15_tailstd_lsml', 'F15_cov_lsml'], ['F15_tailstd_lsml', 'B11_lsml']]
lock['external_multiplicity_v2'] = 'per cell (Hard2Verify/Qwen3-8B, Socratic/Qwen3-8B, Socratic/QwQ-32B); paired source-question bootstrap, 100,000 draws, seed 20260924; Bonferroni over 4 contrasts x 3 cells = 12. The six V1 contrasts are replayed unchanged as a bridge, not re-tested. All 11 arms are reported; nothing is selected, dropped or re-tuned after external labels are seen.'
lock['v1_replay'] = replay
lock['code_hashes_v2'] = {'lock_v2_script': sha(Path(__file__)), 'tail_calib_common': sha(HERE / 'tail_calib_common.py'), 'calfix_common': sha(HERE / 'calfix_common.py'), **extgen_hashes()}
out = STAGE / 'TRANSFER_LOCK_V2.json'; dump(out, lock)
print(json.dumps(replay), {k: v for k, v in new.items() if k in ('K', 'q80_threshold_fold4', 'groups')})
print('weights', {k[:10]: round(v, 4) for k, v in new['weights'].items()})
print('lock v2 sha256', sha(out))
