"""Immutable-input and historical replay gates. No new arms before PASS."""
import argparse
import ast
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from spectral_utils.monday_fusion_protocol import ROOT, WORK, CODE, SSL, RUNS, BASE, sha, dump

def audit():
    old_path = SSL / 'results/algorithm_decisions_v1/run_20260928/INPUT_MANIFEST.json'
    old = json.loads(old_path.read_text(encoding='utf8'))
    freeze_path = ROOT / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/INPUT_FREEZE.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8-sig'))
    manifest = {'version': 2, 'base': BASE, 'frozen_manifest': {'path': str(old_path), 'sha256': sha(old_path)},
                'banks': old['banks'], 'population': old['population'], 'inputs': {}}
    expected = {k: v for k,v in old.items() if isinstance(v,dict) and 'path' in v}
    for k in ('prm_metadata','folds','joined'):
        expected[k] = freeze[k]
    for k in ('pool_z','pool_names'):
        expected[k] = dict(expected[k], path=str(SSL / 'results/algorithm_decisions_v1/inputs_backup' / Path(expected[k]['path']).name))
    extras = {'release_v3': ROOT / 'results/localization_prm_label_audit_v1/RELEASE_V3.json',
              'step457': SSL / 'results/algorithm_decisions_v1/run_20260928/STEP_SCORES.npz',
              'step462': SSL / 'results/per_dataset_fit_v1/run_20260929/STEP_SCORES.npz',
              'answer_features': ROOT / '.worktrees/decision-rule-v1/results/answer_gate_v1/ANSWER_FEATURES.npz',
              'input_freeze': freeze_path}
    problems = []
    for k, spec in list(expected.items()) + [(k, {'path': str(v)}) for k,v in extras.items()]:
        p = Path(spec['path'])
        if not p.is_file():
            problems.append(f'Missing input {k}: {p}'); continue
        h = sha(p)
        manifest['inputs'][k] = {'path':str(p),'bytes':p.stat().st_size,'sha256':h,'expected_sha256':spec.get('sha256')}
        if spec.get('sha256') and h != spec['sha256']:
            problems.append(f'Input hash mismatch {k}: expected {spec["sha256"]}, got {h}')
    dump(WORK / 'INPUT_AUDIT.json', {'status':'FAIL' if problems else 'HASH_PASS', 'problems':problems,'n_checked':len(manifest['inputs'])})
    for out in RUNS.values():
        if (out / 'INPUT_MANIFEST.json').exists():
            prior = json.loads((out / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))
            if prior != manifest: raise RuntimeError('Refuse to change existing manifest')
        else:
            dump(out / 'INPUT_MANIFEST.json', manifest)
    if problems: raise RuntimeError('\n'.join(problems))
    sys.path.insert(0,str(CODE / 'scripts/experiments'))
    import calfix_common as CC
    pop = CC.Population()  # all PRMB annotations and group/fold isolation assertions
    folds = json.loads(Path(manifest['inputs']['folds']['path']).read_text(encoding='utf8'))['outer']
    assert all(folds[g] == int(f) for g,f in zip(pop.groups,pop.fold)), 'FOLDS_V2 alignment'
    release = json.loads(Path(manifest['inputs']['release_v3']['path']).read_text(encoding='utf8'))
    bycell = {c:{r['row_id']:r for r in d['rows']} for c,d in release['cells'].items()}
    for i in range(pop.n):
        row = bycell[pop.cells[i]][pop.ids[i]]
        assert row['steps'] == int(pop.ns[i]) and row['group_id'] == pop.groups[i], ('release alignment',i)
        assert -1 <= pop.target[i] < pop.ns[i] if pop.pb[i] else True
    actual = {'answers':pop.n,'prm':int(pop.prm.sum()),'eligible':int(pop.eligible.sum()),
              'noncontrol':int(pop.noncontrol.sum()),'steps':pop.total,'pb_erroneous':int((pop.pb & (pop.target>=0)).sum())}
    assert actual == old['population'], ('population',actual,old['population'])
    gate = {'status':'PASS','inputs_checked':len(manifest['inputs']),'population':actual,
            'folds_checked':pop.n,'raw_prm_annotations_checked':int(pop.prm.sum()),'release_rows_checked':pop.n,
            'source_groups':len(set(pop.groups)),'cells':sorted(set(pop.cells))}
    dump(WORK / 'INPUT_AUDIT.json',gate)
    print(json.dumps(gate),flush=True)
    return manifest, pop

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=['audit','raw-audit','replay','zero','diagnose-zero','feasibility','level','estimator-a','estimator-b','layers','lr','contrasts','report','continue','combined','freeze-rule','competitors'])
    p.add_argument('--experiment',choices=list(RUNS),default='level_reduction_v1')
    p.add_argument('--wait-for-level',action='store_true')
    p.add_argument('--rule',choices=['ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80'])
    p.add_argument('--authorization')
    args = p.parse_args()
    started = time.time()
    try:
        if args.wait_for_level:
            from spectral_utils.monday_fusion_workflow import wait_level
            wait_level()
        manifest, pop = audit()
        if args.stage == 'replay':
            from spectral_utils.monday_fusion_replay import replay
            replay(manifest,pop)
        elif args.stage=='raw-audit':
            from spectral_utils.monday_fusion_replay import HistoricalData
            from spectral_utils.monday_fusion_decision import raw_pool,entropy_feature
            data=HistoricalData(manifest,pop,banks=[])
            x=raw_pool(data); entropy=entropy_feature(data)
            print(json.dumps({'raw_shape':list(x.shape),'entropy_answers':len(entropy),'status':'PASS'}),flush=True)
        elif args.stage=='contrasts':
            from spectral_utils.monday_fusion_statistics import core_contrasts
            core_contrasts(pop,args.experiment)
        elif args.stage=='report':
            from spectral_utils.monday_fusion_figures import render
            from spectral_utils.monday_fusion_report import report
            render(args.experiment)
            report(args.experiment)
        elif args.stage=='diagnose-zero':
            from spectral_utils.monday_fusion_replay import HistoricalData
            from spectral_utils.monday_fusion_core import variant_bank
            data=HistoricalData(manifest,pop,banks=['B32']); bank=variant_bank(data,'B32','pb_gsm8k_q4','original')
            expected=data.env['cell_bank']('B32',bank['rows'],'pb_gsm8k_q4')
            delta=bank['TT']-expected[2][bank['rows']]
            problem={'max_abs_difference':float(np.abs(delta).max()),'n_differences':int((delta!=0).sum()),
                     'columns':[bank['names'][j] for j in np.flatnonzero((delta!=0).any(0))],
                     'historical_flips':expected[3], 'V_exact':bool(np.array_equal(bank['V'],expected[0][bank['rows']])),
                     'marks_exact':bool(np.array_equal(bank['marks'],expected[1][bank['rows']]))}
            dump(WORK/'ZERO_DIAGNOSIS.json',problem); print(json.dumps(problem),flush=True)
        elif args.stage in ('zero','feasibility','level','estimator-a','estimator-b','layers','lr','continue','combined','freeze-rule','competitors'):
            from spectral_utils.monday_fusion_replay import HistoricalData
            from spectral_utils.monday_fusion_execution import zero_removal,run_level
            data=HistoricalData(manifest,pop,banks=['B54'] if args.stage=='feasibility' else None)
            if args.stage=='zero': zero_removal(data)
            elif args.stage in ('feasibility','level'): run_level(data,feasibility=args.stage=='feasibility')
            elif args.stage in ('estimator-a','estimator-b'):
                from spectral_utils.monday_fusion_estimators import run_estimators
                run_estimators(data,'answer' if args.stage=='estimator-a' else 'cell')
            elif args.stage in ('layers','lr'):
                from spectral_utils.monday_fusion_layers import run_position_decision,run_lr
                (run_position_decision if args.stage=='layers' else run_lr)(data)
            elif args.stage=='continue':
                from spectral_utils.monday_fusion_workflow import continue_workflow
                continue_workflow(data)
            elif args.stage in ('combined','freeze-rule'):
                from spectral_utils.monday_fusion_combined import run_combined,freeze_decision
                if args.stage=='combined': run_combined(data)
                else: freeze_decision(data,args.rule,args.authorization)
            else:
                from spectral_utils.monday_fusion_competitors import run_competitors
                run_competitors(data,ROOT/'results/monday_combined_recipe_v1/run_20261003/LOCKED_RECIPE.json')
    except Exception as e:
        dump(WORK / 'GATE_STATUS.json', {'status':'STOPPED','stage':args.stage,'reason':str(e),'runtime_s':time.time()-started})
        print('HARD STOP: '+str(e),flush=True)
        raise
    dump(WORK / 'GATE_STATUS.json', {'status':'PASS','stage':args.stage,'runtime_s':time.time()-started})
