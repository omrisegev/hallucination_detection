"""Fill base/PRM_UPDATE.json (slides 12-13) from the saved same-checkpoint L-SML results.

Source: AIRCC job 284086 (sources/prm_slide12/...eval_slide12/SLIDE12_*), script lsml_slide12.py =
lsml_four.py (job 283967) plus single-channel arms and L-SML of the 3 channels only. Its 4-view LSML_CONT
rows reproduce job 283967's LSML4_RESULTS exactly. Every number below is read from those CSVs.
"""
import csv, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
X = HERE / 'sources/eval_slide12'   # copied from AIRCC results/prm_boost_gen_v1/eval_slide12 (+ script, run file, log)
SRC = 'AIRCC job 284086 · results/prm_boost_gen_v1/eval_slide12'


def rows(tag):
    return {r['arm']: r for r in csv.DictReader((X / f'SLIDE12_RESULTS_{tag}.csv').open())}


def ci(tag, contrast, metric):
    for r in csv.DictReader((X / f'SLIDE12_BOOTSTRAP_{tag}.csv').open()):
        if r['contrast'] == contrast and r['metric'] == metric:
            return [float(r['delta']), float(r['ci_low']), float(r['ci_high'])]
    raise KeyError((tag, contrast, metric))


diag7 = json.loads((X / 'SLIDE12_DIAG_qwen_prm_7b.json').read_text())
r7 = rows('qwen_prm_7b')
ARMS = [('PRM_native', 'PRM reward head · native 0.5', 'native reward < 0.5'),
        ('PRM_Q80', 'PRM reward head · Q80', 'ANSWER_Z_Q80'),
        ('CH_energy_level', 'energy_level alone', 'ANSWER_Z_Q80'),
        ('CH_hist_entropy_cusum_abs_series', 'hist_entropy_cusum_abs_series alone', 'ANSWER_Z_Q80'),
        ('CH_realized_drv', 'realized_drv alone', 'ANSWER_Z_Q80'),
        ('LSML3', 'L-SML · 3 backbone signals', 'ANSWER_Z_Q80'),
        ('LSML_CONT', 'L-SML · PRM risk + 3 backbone signals', 'ANSWER_Z_Q80')]
boost = []
for arm, label, rule in ARMS:
    r = r7[arm]
    boost.append({'label': label, 'arm': arm, 'checkpoint': 'Qwen2.5-Math-PRM-7B', 'rule': rule,
                  'source': f'{SRC}/SLIDE12_RESULTS_qwen_prm_7b.csv', 'prmscore': float(r['prmscore']),
                  'auc': float(r['within_auc']), 'n_answers': int(r['n_answers']), 'n_auc_answers': int(r['n_auc'])})
boost_ci = {c: {m: ci('qwen_prm_7b', c, m) for m in ('prmscore', 'within_auc')}
            for c in ('LSML_CONT - PRM_native', 'LSML_CONT - PRM_Q80', 'LSML_CONT - LSML3', 'LSML3 - PRM_native')}

MODELS = [('qwen_prm_7b', 'Qwen2.5-Math-PRM-7B (feature-selection model)'), ('qwen_prm_72b', 'Qwen2.5-Math-PRM-72B'),
          ('pure_prm', 'PURE-PRM-7B'), ('rlhflow_prm', 'RLHFlow Llama-3.1-8B PRM'), ('math_shepherd', 'MathShepherd Mistral-7B PRM')]
gen = []
for tag, name in MODELS:
    r = rows(tag); d = json.loads((X / f'SLIDE12_DIAG_{tag}.json').read_text())['LSML_CONT']
    gen.append({'checkpoint': name, 'tag': tag, 'rule': 'native PRM rule vs L-SML at ANSWER_Z_Q80',
                'source': f'{SRC}/SLIDE12_RESULTS_{tag}.csv', 'same_checkpoint': True, 'fixed_features': True,
                'prm_prmscore': float(r['PRM_native']['prmscore']), 'lsml_prmscore': float(r['LSML_CONT']['prmscore']),
                'prm_auc': float(r['PRM_native']['within_auc']), 'lsml_auc': float(r['LSML_CONT']['within_auc']),
                'delta_prmscore_ci': ci(tag, 'LSML_CONT - PRM_native', 'prmscore'),
                'delta_auc_ci': ci(tag, 'LSML_CONT - PRM_native', 'within_auc'),
                'lsml_K': d['K'], 'lsml_groups': d['groups'], 'prm_weight': d['weights_normalized']['prm_risk']})

U = {'updated': '2026-10-14',
     'boosting': {'status': 'complete', 'fusion': 'LSML_CONT', 'same_checkpoint': True,
                  'features': ['energy_level', 'hist_entropy_cusum_abs_series', 'realized_drv'],
                  'source': ('Qwen2.5-Math-PRM-7B on PRMBench, 6,969 answers (AUC on 6,030). One forward: reward head + '
                             'backbone token distributions. Views are per-answer z-scored, oriented so higher = riskier. '
                             'L-SML = spectral_utils.fusion_utils.lsml_continuous, automatic grouping, fitted label-free on '
                             'the pooled PRMBench steps. Flags: ANSWER_Z_Q80 except the native row. ' + SRC),
                  'lsml_diagnostics': diag7, 'paired_ci_7b': boost_ci, 'rows': boost},
     'generalization': {'status': 'complete', 'fusion': 'LSML_CONT', 'same_checkpoint': True,
                        'source': ('Each PRM: its own reward head + its own 3 backbone signals, same frozen feature list and '
                                   'L-SML procedure; no re-selection. Paired source-group bootstrap (707 groups, 2,000 draws). '
                                   + SRC),
                        'rows': gen}}
(HERE / 'base/PRM_UPDATE.json').write_text(json.dumps(U, indent=2) + '\n')
for r in boost: print(f"{r['label']:45s} {r['prmscore']:6.2f} {r['auc']:.4f}")
for r in gen: print(f"{r['checkpoint']:45s} {r['prm_prmscore']:6.2f} -> {r['lsml_prmscore']:6.2f}  CI {r['delta_prmscore_ci']}")
