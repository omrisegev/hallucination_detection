"""Add smoke-derived supplements (supervised mechanics, Joint outliers, memory projection) to FEASIBILITY.json."""
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_varentropy_expansion_fusion_v1 import OUT, atomic_json_retry, clean

TOKENS = {'pb_gsm8k': 114502, 'pb_math': 523899, 'pb_olympiadbench': 781284, 'pb_omnimath': 773607, 'prmbench_qwen3_8b': 2582195}


def main():
    feasibility = json.loads((OUT / 'FEASIBILITY.json').read_text(encoding='utf8'))
    con = sqlite3.connect((OUT / 'SMOKE.sqlite').as_uri() + '?mode=ro', uri=True)
    infos = [json.loads(r[0]) for r in con.execute('SELECT info FROM answers ORDER BY idx')]
    outliers = []
    for d in infos:
        for m, x in d['diagnostics'].items():
            if m.endswith('__joint') and (not x['converged'] or x['selected_start_sweeps'] >= 1000):
                outliers.append(dict(uid=d['uid'], method=m, n_tokens=d['n_tokens'], seconds=d['seconds'][m], sweeps=x['selected_start_sweeps'],
                                     converged=x['converged'], multistart_status=x['multistart_status'], model_covariance_condition=x['model_covariance_condition']))
    flips = {m: int(sum(d['diagnostics'][m]['orientation_flipped'] for d in infos if m in d['diagnostics'])) for m in infos[0]['diagnostics']}
    alpha_one = {m: int(sum(d['diagnostics'][m]['alpha'] >= 1.0 for d in infos if m in d['diagnostics'])) for m in infos[0]['diagnostics'] if m.endswith('__shrink')}
    sup = json.loads((OUT / 'supervised/SMOKE.json').read_text(encoding='utf8'))
    fits = [f for f in sup['fits'] if f['status'] == 'FIT']
    memory = {}
    for cell, n in TOKENS.items():
        train = n * 0.8
        memory[cell] = dict(train_tokens_per_fold=int(train), B2_sel_float32_GB=train * 138 * 4 / 1e9, B2d_sel_float32_GB=train * 33 * 4 / 1e9,
                            B2_sel_float64_GB=train * 138 * 8 / 1e9)
    workers = feasibility['workers']; hours = feasibility['projected_full_fit_hours']
    feasibility['supplement'] = dict(
        joint_outliers=outliers,
        joint_note='Joint fits are ~99% of runtime. One smoke answer (71 tokens) hit the 5000-sweep cap on B2d/B2d_sel (43 s / 36 s, converged=False, multistart BLOCKED); it still scored because the model-inverse map was finite (declared in the protocol: reported, not gated).',
        orientation_flips_on_smoke=flips,
        orientation_note='equal_identity on the pair banks (B2, B2_sel) anti-correlates with the raw varentropy anchor on every smoke answer (mean corr -0.40 before orientation) and is flipped by _orient on 27/27; on the diagonal banks (B2d, B2d_sel) it is not flipped.',
        shrink_alpha_at_one=alpha_one,
        shrink_note='The memory-bounded LW alpha clips to 1.0 on 27/27 answers for B2/B2_sel and 25/27 for B2d/B2d_sel (min 0.87): the shrinkage arm collapses onto the joint target matrix on almost every answer, so `shrink` is effectively IU on the rank-1-completed cross-group covariance.',
        projected_runtime_hours={'workers_1': hours, 'workers_2': hours / 2, 'workers_4': hours / 4, 'workers_1_excluding_joint': feasibility['projected_full_fit_hours_excluding_joint']},
        supervised=dict(status=sup['status'], n_answers=sup['n_answers'], fold_fits=len(fits), declared_fold_failures=sup['n_declared_fold_failures'],
                        scored_answers=sup['scored_answers'], mean_fit_seconds_on_smoke=float(np.mean([f['seconds'] for f in fits])) if fits else None,
                        max_iterations_on_smoke=int(max(f['iterations'] for f in fits)) if fits else None,
                        note='Smoke folds contain 1-3 answers: mechanics only. Full-run per-fold memory projection below (float32 cache; the driver stores banks as float32).',
                        memory_projection=memory),
        smoke_note='27 answers (shortest / median / 95th-percentile trace per cell); no benchmark ranking; not performance evidence.')
    atomic_json_retry(OUT / 'FEASIBILITY.json', clean(feasibility))
    print('supplement written;', len(outliers), 'joint outliers;', len(fits), 'supervised fold fits')


if __name__ == '__main__':
    main()
